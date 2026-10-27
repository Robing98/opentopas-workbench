#!/usr/bin/env python3
"""OpenTOPAS Workbench: a small local web UI for OpenTOPAS and TOPAS.

Manage parameter files, run them in a queue, look at scorer output, and generate parameter sets from plugins.
Pure Python standard library. Runs on Linux, macOS and WSL.

    opentopas-workbench [--workspace DIR] [--port 8765] [--no-browser]
    python3 -m opentopas_workbench [...]

Security: listens on 127.0.0.1 only, checks the Host header and requires a per-session token for every API call,
and only touches files inside the workspace.
"""
import argparse, base64, csv, fnmatch, io, zipfile, importlib.util, json, os, re, secrets, shutil, signal, subprocess, sys, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import __version__ as VERSION
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_EXT = {".txt", ".py", ".sh", ".bat", ".json", ".csv", ".md", ".log", ".in", ".mac", ".xml", ".header", ".binheader", ".mcc", ".asc", ".dat", ".tsv"}
TRASH = "_trash"
TOKEN = secrets.token_urlsafe(24)
STATE = {"root": None, "cfg": {}, "lock": threading.Lock(), "jobs": [], "proc": None, "stop": False, "next_id": 1}

# ------------------------------------------------------------------ config
def wb_dir(): return os.path.join(STATE["root"], ".workbench")


def detect_binaries():
    home = os.path.expanduser("~")
    found = []
    if os.path.exists(os.path.join(home, ".local/bin/topas4")):
        found.append({"name": "OpenTOPAS (topas4)", "cmd": os.path.join(home, ".local/bin/topas4"), "env": {}, "subdir": ""})
    ot = os.path.join(home, "Applications/TOPAS/OpenTOPAS-install/bin/topas")
    if os.path.exists(ot):
        found.append({"name": "OpenTOPAS (Applications)", "cmd": ot, "env": {"TOPAS_G4_DATA_DIR": os.path.join(home, "Applications/GEANT4/G4DATA")}, "subdir": ""})
    t39 = os.path.join(home, "topas/topas/bin/topas")
    if os.path.exists(t39):
        found.append({"name": "TOPAS 3.9", "cmd": t39, "env": {"TOPAS_G4_DATA_DIR": os.path.join(home, "topas/G4Data")}, "subdir": ""})
    p = shutil.which("topas")
    if p and p not in [b["cmd"] for b in found]:
        found.append({"name": "topas (PATH)", "cmd": p, "env": {}, "subdir": ""})
    return found


def load_cfg():
    path = os.path.join(wb_dir(), "config.json")
    cfg = {}
    try:
        with open(path) as fh: cfg = json.load(fh)
    except (OSError, ValueError): pass
    if not cfg.get("binaries"): cfg["binaries"] = detect_binaries()
    cfg.setdefault("lang", "en")
    STATE["cfg"] = cfg


def save_cfg():
    os.makedirs(wb_dir(), exist_ok=True)
    with open(os.path.join(wb_dir(), "config.json"), "w") as fh: json.dump(STATE["cfg"], fh, indent=1)


def load_jobs():
    try:
        with open(os.path.join(wb_dir(), "jobs.json")) as fh: STATE["jobs"] = json.load(fh)
    except (OSError, ValueError): STATE["jobs"] = []
    for j in STATE["jobs"]:
        if j["status"] in ("running", "queued"): j["status"] = "aborted"
    STATE["next_id"] = max([j["id"] for j in STATE["jobs"]] + [0]) + 1


def save_jobs():
    os.makedirs(wb_dir(), exist_ok=True)
    with open(os.path.join(wb_dir(), "jobs.json"), "w") as fh: json.dump(STATE["jobs"][-500:], fh, indent=0)

# ------------------------------------------------------------------ paths
def resolve(rel):
    root = root_real()
    p = os.path.realpath(os.path.join(root, rel or ""))
    if p != root and not p.startswith(root + os.sep): raise ValueError("path outside workspace")
    return p


_ROOT_REAL = {}


def root_real():
    r = STATE["root"]
    if r not in _ROOT_REAL: _ROOT_REAL[r] = os.path.realpath(r)   # realpath stats every path component; slow on /mnt/c
    return _ROOT_REAL[r]


def relp(p):
    r = os.path.relpath(p, root_real()).replace(os.sep, "/")
    return "" if r == "." else r


def natkey(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

# ------------------------------------------------------------------ parameter files
PARAM_RE = re.compile(r"^\s*(?:(d|u|i|b|s|dv|uv|iv|bv|sv):)?([A-Za-z0-9_/\.\-]+)\s*=\s*(.*?)\s*$")


def is_param_file(path):
    if not path.endswith(".txt"): return False
    try:
        with open(path, errors="replace") as fh: head = fh.read(6000)
    except OSError: return False
    return bool(re.search(r"^\s*(d|u|i|b|s|dv|sv|iv|uv|bv):(Ge|So|Sc|Ph|Ts|Gr|Ma|El|Tf|Vr)/", head, re.M)) or "includeFile" in head


def includes(text, base_dir=None):
    """includeFile targets: list of (line, name, path relative to the workspace or None if missing)."""
    out = []
    for n, raw in enumerate(text.splitlines(), 1):
        m = re.match(r"^\s*includeFile\s*=\s*(.+?)\s*$", raw.split("#", 1)[0])
        if not m: continue
        for name in m.group(1).replace('"', " ").split():
            rel = None
            if base_dir is not None:
                try:
                    cand = resolve(relp(os.path.join(base_dir, name)))
                    if os.path.isfile(cand): rel = relp(cand)
                except ValueError:
                    pass
            out.append([n, name, rel])
    return out


def lint(text, base_dir=None):
    """Light syntax check of a TOPAS parameter file. Returns list of (line, severity, message-key, detail)."""
    issues = []
    if base_dir is not None:
        for n, name, rel in includes(text, base_dir):
            if rel is None: issues.append([n, "error", "lint_include", name])
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line: continue
        if line.startswith("includeFile"):
            continue
        m = PARAM_RE.match(line)
        if not m:
            issues.append([n, "error", "lint_syntax", raw.strip()[:60]]); continue
        typ, name, val = m.groups()
        if typ is None:
            issues.append([n, "error", "lint_notype", name]); continue
        if typ == "b" and val.strip('"').lower() not in ("true", "false", "t", "f", "1", "0") and "/" not in val:
            issues.append([n, "warn", "lint_bool", val])
        if typ == "s" and not (val.startswith('"') and val.endswith('"')) and "/" not in val:
            issues.append([n, "warn", "lint_string", val])
        if typ == "d" and val and not re.search(r"[A-Za-z]", val):
            issues.append([n, "warn", "lint_unit", val])
        if typ in ("i",) and not re.fullmatch(r"-?\d+", val) and "/" not in val:
            issues.append([n, "warn", "lint_int", val])
        if typ.endswith("v"):
            toks = val.replace('"', ' " ').split() if typ == "sv" else val.split()
            if toks and toks[0].isdigit():
                cnt = int(toks[0])
                rest = toks[1:]
                if typ in ("dv",) and rest and re.fullmatch(r"[A-Za-z/%]+\w*", rest[-1]): rest = rest[:-1]
                if typ == "sv":
                    rest = re.findall(r'"[^"]*"', val)
                if len(rest) != cnt:
                    issues.append([n, "warn", "lint_count", f"{cnt} / {len(rest)}"])
    return issues


def total_histories(text):
    """Sum of i:So/*/NumberOfHistoriesInRun, None if not set in this file."""
    vals = [int(v) for v in re.findall(r"^\s*i:So/[^/]+/NumberOfHistoriesInRun\s*=\s*(\d+)", text, re.M)]
    return sum(vals) if vals else None


ENERGY_TOTAL_RE = re.compile(r"^\s*d:So/[^/]+/BeamEnergy\s*=\s*([\d.eE+\-]+)\s*(\w+)", re.M)
_SIZE_CACHE = {}


def job_size(path):
    """(histories, total beam energy in MeV) of a parameter file, cached by modification time."""
    try: key = (path, os.path.getmtime(path))
    except OSError: return None, None
    if key not in _SIZE_CACHE:
        with open(path, errors="replace") as fh: text = fh.read()
        m = ENERGY_TOTAL_RE.search(text)
        e = float(m.group(1)) * {"eV": 1e-6, "keV": 1e-3, "MeV": 1, "GeV": 1e3}.get(m.group(2), 1) if m else None
        _SIZE_CACHE[key] = (total_histories(text), e)
    return _SIZE_CACHE[key]


LIVE = {}   # job id -> {"samples": [(time, histories)], "cpu": (time, ticks)}
TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def proc_usage(pgid):
    """CPU ticks and resident memory of all processes in a process group (Linux /proc), or None."""
    if not os.path.isdir("/proc"): return None
    ticks = rss = 0; found = False
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try:
            with open(f"/proc/{d}/stat") as fh: st = fh.read()
            f = st[st.rindex(")") + 2:].split()
            if int(f[2]) != pgid: continue        # field 5 (pgrp)
            ticks += int(f[11]) + int(f[12]); rss += int(f[21]) * PAGE_SIZE; found = True   # utime, stime, rss
        except (OSError, ValueError, IndexError):
            continue
    return (ticks, rss) if found else None


def log_state(path):
    """Phase of a TOPAS run and the last meaningful log line."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 65536)); tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return "init", ""
    lines = [l.strip() for l in tail.splitlines() if l.strip() and not l.startswith("# ")]
    last = lines[-1][:160] if lines else ""
    if re.search(r"run sequence complete|Results have been written|Execution done", tail): phase = "output"
    elif HIST_RE.search(tail.encode()): phase = "run"
    else: phase = "init"
    return phase, last


def live_info(job):
    """Live status of the running job: phase, histories per second, CPU cores in use, memory, last log line."""
    now = time.time(); L = LIVE.setdefault(job["id"], {"samples": [], "cpu": None})
    phase, last = log_state(resolve(job["log"])) if job.get("log") else ("init", "")
    info = {"phase": phase, "last": last, "elapsed": round(now - job.get("t0", now))}
    done = job.get("done")
    if done is not None and (not L["samples"] or L["samples"][-1][1] != done): L["samples"].append((now, done))
    S = [x for x in L["samples"] if x[1] > 0] or L["samples"]
    if len(S) >= 2 and S[-1][1] > S[0][1]:
        info["rate"] = (S[-1][1] - S[0][1]) / max(S[-1][0] - S[0][0], 1e-6)
        if job.get("total"):  # smooth estimate between the history counts TOPAS prints
            info["est_done"] = min(job["total"], S[-1][1] + info["rate"] * (now - S[-1][0]))
    proc = STATE.get("proc")
    if proc is not None and proc.poll() is None:
        u = proc_usage(proc.pid)
        if u:
            if L["cpu"] and now > L["cpu"][0]:
                info["cores"] = round((u[0] - L["cpu"][1]) / TICK / (now - L["cpu"][0]), 1)
            elif L.get("cores") is not None:
                info["cores"] = L["cores"]
            if L["cpu"] is None or now - L["cpu"][0] > 1.5:
                L["cpu"] = (now, u[0]); L["cores"] = info.get("cores")
            info["rss_mb"] = round(u[1] / 1e6)
    return info


def queue_estimate(jobs):
    """Remaining time of the queue from the finished jobs: seconds per history and MeV (track length grows with energy),
    or seconds per history when energies are unknown. None until at least one job of the same program has finished."""
    est = {}
    for b in {j["binary"] for j in jobs if j["status"] in ("queued", "running")}:
        done = [j for j in jobs if j["binary"] == b and j["status"] == "done" and j.get("seconds") and j.get("total")][-20:]
        if not done: continue
        for j in done:  # jobs from before this version have no energy recorded
            if "energy" not in j:
                try: j["energy"] = job_size(resolve(j["file"]))[1]
                except ValueError: j["energy"] = None
        pe = sorted(j["seconds"] / (j["total"] * j["energy"]) for j in done if j.get("energy"))
        ph = sorted(j["seconds"] / j["total"] for j in done)
        est[b] = (pe[len(pe) // 2] if pe else None, ph[len(ph) // 2])
    total, unknown = 0.0, 0
    for j in jobs:
        if j["status"] not in ("queued", "running") or j["binary"] not in est: 
            if j["status"] in ("queued", "running"): unknown += 1
            continue
        if j["status"] == "running" and j.get("eta") is not None:
            total += j["eta"]; continue
        try: n, e = job_size(resolve(j["file"]))
        except ValueError: n, e = None, None
        if not n: unknown += 1; continue
        rate_e, rate_h = est[j["binary"]]
        t = n * e * rate_e if (rate_e and e) else n * rate_h
        if j["status"] == "running": t = max(0.0, t - (time.time() - j.get("t0", time.time())))
        total += t
    if not est: return None
    return {"seconds": round(total), "unknown": unknown, "finish": time.strftime("%H:%M", time.localtime(time.time() + total))}


HIST_RE = re.compile(rb"History:\s*(\d+)")
ERR_RE = re.compile(r"(?i)(\berror\b|exception|quitting|exiting|segmentation|abort|unknown parameter)")
ERR_IGNORE = re.compile(r"(?i)not necessarily an error")


def log_progress(path):
    """Highest history number printed by TOPAS in the log (TOPAS prints 'History: N' at ShowHistoryCountAtInterval)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 65536)); tail = fh.read()
    except OSError:
        return None
    nums = [int(x) for x in HIST_RE.findall(tail)]
    return max(nums) if nums else None


def log_hints(path, limit=4):
    """Last log lines that look like errors."""
    try:
        with open(path, errors="replace") as fh: lines = fh.readlines()[-3000:]
    except OSError:
        return []
    hits = [l.strip() for l in lines if ERR_RE.search(l) and not ERR_IGNORE.search(l)]
    return hits[-limit:]


OUT_EXT = (".csv", ".bin", ".binheader", ".root", ".xml", ".phsp", ".header", ".dcm")


def has_output(names, base):
    """True if a TOPAS output file for scorer base name 'base' is among names."""
    return any(n.startswith(base + ".") and n.lower().endswith(OUT_EXT) for n in names)


def scorer_outputs(text):
    """Base names of scorer output files declared in a parameter file."""
    return re.findall(r'^\s*s:Sc/[^/]+/OutputFile\s*=\s*"([^"]+)"', text, re.M)

# ------------------------------------------------------------------ job queue
class KeepAwake:
    """Blocks system sleep while jobs are queued or running (not the display).

    WSL: a hidden PowerShell calls SetThreadExecutionState and exits once a heartbeat file in .workbench gets stale.
    macOS: caffeinate -i bound to this process. Linux: systemd-inhibit bound to this process."""

    def __init__(self):
        self.proc, self.flag, self.method, self.error, self.last = None, None, None, None, 0

    @staticmethod
    def platform():
        if sys.platform == "darwin": return "macos"
        try:
            with open("/proc/version") as fh:
                if "microsoft" in fh.read().lower(): return "wsl"
        except OSError:
            pass
        return "linux" if sys.platform.startswith("linux") else None

    def active(self):
        return self.proc is not None and self.proc.poll() is None

    def touch(self):
        if self.flag and time.time() - self.last > 20:
            try:
                with open(self.flag, "w") as fh: fh.write(str(time.time()))
                self.last = time.time()
            except OSError:
                pass

    def start(self):
        if self.active(): self.touch(); return
        plat = self.platform(); pid = str(os.getpid()); self.error = None
        try:
            if plat == "wsl":
                os.makedirs(wb_dir(), exist_ok=True)
                self.flag = os.path.join(wb_dir(), "keepawake.flag"); self.last = 0; self.touch()
                win = subprocess.run(["wslpath", "-w", self.flag], capture_output=True, text=True, timeout=10).stdout.strip()
                ps = ("Add-Type -Name P -Namespace W -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);';"
                      "[void][W.P]::SetThreadExecutionState(0x80000001);"
                      f"$f='{win}';while((Test-Path $f) -and (((Get-Date)-(Get-Item $f).LastWriteTime).TotalSeconds -lt 120)){{Start-Sleep 15}}")
                enc = base64.b64encode(ps.encode("utf-16-le")).decode()  # avoids quoting problems across the WSL boundary
                self.proc = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-EncodedCommand", enc],
                                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="/mnt/c")
                self.method = "Windows (SetThreadExecutionState)"
            elif plat == "macos":
                self.proc = subprocess.Popen(["caffeinate", "-i", "-w", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.method = "caffeinate"
            elif plat == "linux" and shutil.which("systemd-inhibit"):
                self.proc = subprocess.Popen(["systemd-inhibit", "--what=sleep:idle", "--who=OpenTOPAS Workbench", "--why=TOPAS run",
                                              "tail", f"--pid={pid}", "-f", "/dev/null"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.method = "systemd-inhibit"
            else:
                self.error = "not supported on this system"
        except (OSError, subprocess.SubprocessError) as e:
            self.proc = None; self.error = str(e)

    def stop(self):
        if self.flag:
            try: os.remove(self.flag)
            except OSError: pass
        if self.proc is not None and self.proc.poll() is None:
            try: self.proc.terminate()
            except OSError: pass
        self.proc = None

    def status(self):
        return {"active": self.active(), "method": self.method if self.active() else None, "error": self.error}


AWAKE = KeepAwake()


def windows_toast(title, text):
    """Windows notification from WSL, used when no browser page is open."""
    if KeepAwake.platform() != "wsl" or not STATE["cfg"].get("toast", True): return
    q = lambda v: str(v).replace("'", "''")[:200]
    ps = ("[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime]>$null;"
          "$x=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
          f"$t=$x.GetElementsByTagName('text');[void]$t.Item(0).AppendChild($x.CreateTextNode('{q(title)}'));[void]$t.Item(1).AppendChild($x.CreateTextNode('{q(text)}'));"
          "$n=[Windows.UI.Notifications.ToastNotification]::new($x);"
          "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe').Show($n)")
    try:
        subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-EncodedCommand",
                          base64.b64encode(ps.encode("utf-16-le")).decode()], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, cwd="/mnt/c")
    except OSError:
        pass


def notify_job(job):
    """Server-side notification when no page has asked for the job list recently (the page notifies itself otherwise)."""
    if time.time() - STATE.get("last_poll", 0) < 60: return
    name = os.path.basename(job["file"])
    if job["status"] in ("failed",):
        windows_toast("OpenTOPAS Workbench", f"Run failed: {name}")
    elif not any(j["status"] in ("queued", "running") for j in STATE["jobs"] if j is not job):
        windows_toast("OpenTOPAS Workbench", f"Queue finished, last run: {name} ({job['status']})")


def job_worker():
    while True:
        time.sleep(0.5)
        busy = STATE["proc"] is not None or any(j["status"] == "queued" for j in STATE["jobs"])
        if busy and STATE["cfg"].get("keep_awake", True): AWAKE.start()
        elif AWAKE.active() or AWAKE.flag: AWAKE.stop()
        with STATE["lock"]:
            if STATE["proc"] is not None: continue
            job = next((j for j in STATE["jobs"] if j["status"] == "queued"), None)
            if job is None: continue
            job["status"] = "running"; job["start"] = time.strftime("%Y-%m-%d %H:%M:%S"); save_jobs()
        run_job(job)


def run_job(job):
    binary = next((b for b in STATE["cfg"]["binaries"] if b["name"] == job["binary"]), None)
    logdir = os.path.join(wb_dir(), "logs"); os.makedirs(logdir, exist_ok=True)
    job["log"] = relp(os.path.join(logdir, f"job{job['id']:05d}.log"))
    try:
        param = resolve(job["file"])
        out = resolve(os.path.join(os.path.dirname(job["file"]), job.get("subdir") or ""))
        os.makedirs(out, exist_ok=True)
        if binary is None: raise RuntimeError(f"binary '{job['binary']}' not configured")
        env = dict(os.environ, **{k: os.path.expanduser(v) for k, v in binary.get("env", {}).items()})
        # copy companion geometry files (STL/PLY) so relative InputFile paths work in a subfolder
        if out != os.path.dirname(param):
            for f in os.listdir(os.path.dirname(param)):
                if f.lower().endswith((".stl", ".ply")) and not os.path.exists(os.path.join(out, f)):
                    shutil.copy2(os.path.join(os.path.dirname(param), f), out)
        def snapshot():
            return {f: os.path.getmtime(os.path.join(out, f)) for f in os.listdir(out) if os.path.isfile(os.path.join(out, f))}
        before = snapshot()
        job["total"], job["energy"] = job_size(param)
        t0 = time.time(); job["t0"] = t0
        with open(resolve(job["log"]), "w") as log:
            log.write(f"# {job['start']}  {binary['cmd']} {param}\n# cwd: {out}\n"); log.flush()
            proc = subprocess.Popen([os.path.expanduser(binary["cmd"]), param], cwd=out, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            with STATE["lock"]: STATE["proc"] = proc
            rc = proc.wait()
        job["seconds"] = round(time.time() - t0)
        job["outputs"] = sorted(relp(os.path.join(out, f)) for f, mt in snapshot().items()
                                if before.get(f) != mt and not f.lower().endswith((".stl", ".ply")))
        job["status"] = "stopped" if STATE["stop"] else ("done" if rc == 0 else "failed")
        job["rc"] = rc
        job["hints"] = log_hints(resolve(job["log"]))
        if job["status"] == "failed" and not job["hints"]:
            with open(resolve(job["log"]), errors="replace") as fh:
                job["hints"] = [l.rstrip() for l in fh.readlines()[-3:] if l.strip()]
    except Exception as e:  # report everything in the job record
        job["status"] = "failed"; job["error"] = str(e)
    finally:
        with STATE["lock"]:
            STATE["proc"] = None; STATE["stop"] = False; job["end"] = time.strftime("%Y-%m-%d %H:%M:%S"); save_jobs()
        invalidate_cache()
        notify_job(job)


def enqueue(files, binary, subdir, skip_existing):
    added, skipped = [], []
    for f in files:
        p = resolve(f)
        if not os.path.isfile(p): raise ValueError(f"not a file: {f}")
        if skip_existing:
            with open(p, errors="replace") as fh: outs = scorer_outputs(fh.read())
            od = resolve(os.path.join(os.path.dirname(f), subdir or ""))
            if outs and os.path.isdir(od) and all(has_output(os.listdir(od), o) for o in outs):
                skipped.append(f); continue
        with STATE["lock"]:
            STATE["jobs"].append({"id": STATE["next_id"], "file": f, "binary": binary, "subdir": subdir, "status": "queued",
                                  "queued": time.strftime("%Y-%m-%d %H:%M:%S")})
            STATE["next_id"] += 1
        added.append(f)
    with STATE["lock"]: save_jobs()
    return added, skipped


def stop_current():
    with STATE["lock"]:
        p = STATE["proc"]
        if p is not None and p.poll() is None:
            STATE["stop"] = True
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)

# ------------------------------------------------------------------ TOPAS CSV reader
def read_topas_csv(path, stat=0, collapse=None, slice_index=None):
    dims, quantity, stats = [], "", []
    with open(path, errors="replace") as fh:
        lines = fh.readlines()
    for l in lines:
        if not l.startswith("#"): break
        m = re.match(r"#\s*(\w+)\s+in\s+(\d+)\s+bins?\s+of\s+([\d.eE+\-]+)\s*(\w+)", l)
        if m: dims.append({"name": m.group(1), "n": int(m.group(2)), "width": float(m.group(3)), "unit": m.group(4)})
        m = re.match(r"#\s*(.+?)\s*:\s*(.+)$", l)
        if m and "bins" not in l and ("(" in m.group(1) or "Sum" in m.group(2) or "Mean" in m.group(2)):
            quantity, stats = m.group(1).strip(), m.group(2).split()
    if len(dims) != 3: raise ValueError("no TOPAS 3D binning header found")
    shape = [d["n"] for d in dims]
    import array
    nstat = max(len(stats), 1)
    data = [array.array("d", bytes(8 * shape[0] * shape[1] * shape[2])) for _ in range(nstat)]
    for l in lines:
        if l.startswith("#") or not l.strip(): continue
        parts = l.replace(",", " ").split()
        i, j, k = int(parts[0]), int(parts[1]), int(parts[2])
        idx = (i * shape[1] + j) * shape[2] + k
        for s in range(min(nstat, len(parts) - 3)): data[s][idx] = float(parts[3 + s])
    vals = data[min(stat, nstat - 1)]
    varying = [a for a in range(3) if shape[a] > 1]
    def at(i, j, k): return vals[(i * shape[1] + j) * shape[2] + k]
    res = {"dims": dims, "quantity": quantity, "stats": stats or ["value"], "stat": stat}
    if len(varying) <= 1:
        a = varying[0] if varying else 2
        res["mode"] = "1d"; res["axis"] = a
        res["x"] = [(n + 0.5) * dims[a]["width"] for n in range(shape[a])]
        res["y"] = [at(*[n if q == a else 0 for q in range(3)]) for n in range(shape[a])]
        return res
    if len(varying) == 3:
        c = collapse if collapse in (0, 1, 2) else 2
        varying = [a for a in range(3) if a != c]
    else:
        c = [a for a in range(3) if a not in varying][0]
    a, b = varying
    grid = []
    for u in range(shape[a]):
        row = []
        for v in range(shape[b]):
            if shape[c] == 1: row.append(at(*[{a: u, b: v}.get(q, 0) for q in range(3)]))
            elif slice_index is not None:
                row.append(at(*[{a: u, b: v, c: min(slice_index, shape[c] - 1)}[q] for q in range(3)]))
            else:
                row.append(sum(at(*[{a: u, b: v, c: w}[q] for q in range(3)]) for w in range(shape[c])))
        grid.append(row)
    res.update(mode="2d", axes=[a, b], collapsed=c, grid=grid)
    return res

def read_measured(path, scan=0):
    """Measured or tabulated curves: PTW .mcc, IBA .asc, or two or more numeric columns (x [mm], y, ...)."""
    with open(path, errors="replace") as fh: text = fh.read()
    scans, cur, kind = [], None, ""
    if "BEGIN_SCAN" in text:  # PTW mephysto
        for l in text.splitlines():
            l = l.strip()
            if l.startswith("SCAN_CURVETYPE="): kind = l.split("=", 1)[1]
            elif l == "BEGIN_DATA": cur = []
            elif l == "END_DATA": scans.append((kind, cur)); cur = None
            elif cur is not None:
                v = l.split()
                if len(v) >= 2: cur.append((float(v[0]), float(v[1])))
    elif re.search(r"^\s*%PTS|^\s*=\s*-?[\d.]", text, re.M):  # IBA OmniPro ASCII: "= x y z value"
        for l in text.splitlines():
            l = l.strip()
            if l.startswith("%PTS") or l.startswith("$STOM"): cur = []
            m = re.match(r"=\s*(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)", l)
            if m:
                if cur is None: cur = []
                cur.append(tuple(float(g) for g in m.groups()))
            if (l.startswith("$ENOM") or l.startswith(":EOM")) and cur:
                xs = list(zip(*cur))
                axis = max(range(3), key=lambda a: max(xs[a]) - min(xs[a]))  # the coordinate that varies
                scans.append(("XYZ"[axis], [(c[axis], c[3]) for c in cur])); cur = None
        if cur:
            xs = list(zip(*cur)); axis = max(range(3), key=lambda a: max(xs[a]) - min(xs[a]))
            scans.append(("XYZ"[axis], [(c[axis], c[3]) for c in cur]))
    else:  # generic columns
        unit = "mm"
        rows = []
        for l in text.splitlines():
            m = re.search(r"unit\s*[:=]\s*(mm|cm|m)\b", l, re.I)
            if m and l.lstrip().startswith(("#", "%", "//")): unit = m.group(1).lower()
            l = l.strip()
            if ";" in l or "\t" in l:  # semicolon or tab separated: a comma is a decimal comma (German exports)
                v = [x.strip().replace(",", ".") for x in re.split(r"[;\t]+", l)]
            else:
                v = re.split(r"[, ]+", l)
            try: rows.append((float(v[0]), float(v[1])))
            except (ValueError, IndexError): continue
        f = {"mm": 1, "cm": 10, "m": 1000}[unit]
        scans.append(("", [(x * f, y) for x, y in rows]))
    scans = [sc for sc in scans if len(sc[1]) >= 2]
    if not scans: raise ValueError("no numeric data found")
    kind, pts = scans[min(scan, len(scans) - 1)]
    pts.sort()
    return {"mode": "1d", "axis": 0, "measured": True, "scans": len(scans), "kind": kind,
            "dims": [{"name": "Depth" if kind in ("", "PDD", "Z") else "Position", "n": len(pts), "width": 0, "unit": "mm"}],
            "quantity": os.path.basename(path), "stats": ["value"], "stat": 0,
            "x": [x for x, _ in pts], "y": [y for _, y in pts]}


def read_any(path, stat=0, collapse=None, slice_index=None, scan=0):
    try:
        return read_topas_csv(path, stat, collapse, slice_index)
    except ValueError:
        return read_measured(path, scan)

# ------------------------------------------------------------------ parameter variations
def slug(v):
    return re.sub(r"[^A-Za-z0-9]+", "", str(v).replace("-", "m").replace(".", "p")) or "v"


def make_variants(src_rel, param, values, pattern="{name}_{value}"):
    """Copies of a parameter file with one parameter set to each value. Scorer output names get the same suffix."""
    src = resolve(src_rel)
    with open(src, errors="replace") as fh: text = fh.read()
    name_re = re.compile(r"^(\s*)((?:[a-z]{1,2}):)?(" + re.escape(param.split(":", 1)[-1]) + r")(\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
    m = name_re.search(text)
    if not m: raise ValueError(f"parameter not found: {param}")
    typ = (m.group(2) or "")[:-1]
    old = m.group(5)
    unit = old.split()[-1] if typ in ("d", "dv") and old.split() and re.fullmatch(r"[A-Za-z/%]+\w*", old.split()[-1]) else ""
    out = []
    stem = os.path.splitext(os.path.basename(src))[0]
    for v in values:
        v = str(v).strip()
        if not v: continue
        new = v
        if typ == "s" and not (v.startswith('"') and v.endswith('"')): new = f'"{v}"'
        if typ == "d" and unit and not re.search(r"[A-Za-z]", v): new = f"{v} {unit}"
        if typ == "b" and v.lower() in ("true", "false"): new = f'"{v.capitalize()}"'
        suffix = slug(v.strip('"'))
        body = name_re.sub(lambda k: f"{k.group(1)}{k.group(2) or ''}{k.group(3)}{k.group(4)}{new}{k.group(6)}", text, count=1)
        body = re.sub(r'^(\s*s:Sc/[^/]+/OutputFile\s*=\s*")([^"]+)(")', lambda k: f"{k.group(1)}{k.group(2)}_{suffix}{k.group(3)}", body, flags=re.M)
        body = f"# Variation of {os.path.basename(src)}: {param} = {new}\n" + body
        fname = pattern.replace("{name}", stem).replace("{value}", suffix)
        if not re.fullmatch(r"[\w.\-]+", fname): raise ValueError("invalid file name pattern")
        out.append((relp(os.path.join(os.path.dirname(src), fname + ".txt")), body))
    if not out: raise ValueError("no values")
    return out

# ------------------------------------------------------------------ results index
DATA_EXT = (".csv", ".mcc", ".asc", ".dat", ".tsv")
IMG_EXT = (".png", ".jpg", ".jpeg", ".svg")
PARTICLE_RE = re.compile(r'^\s*s:So/[^/]+/BeamParticle\s*=\s*"([^"]+)"', re.M)
ENERGY_RE = re.compile(r"^\s*d:So/[^/]+/BeamEnergy\s*=\s*([\d.eE+\-]+)\s*(\w+)", re.M)
PARTICLE_NAMES = {"proton": ("p", 1), "alpha": ("He-4", 4), "deuteron": ("d", 2), "triton": ("t", 3), "He3": ("He-3", 3),
                  "e-": ("e-", 0), "e+": ("e+", 0), "gamma": ("gamma", 0), "neutron": ("n", 1)}
ELEMENTS = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne", 18: "Ar", 54: "Xe"}


def beam_info(text):
    """(particle label, energy per nucleon in MeV/u or total MeV for leptons/photons) from a parameter file."""
    m = PARTICLE_RE.search(text)
    if not m: return None, None
    part = m.group(1)
    g = re.match(r"GenericIon\((\d+),\s*(\d+)", part)
    if g:
        z, a = int(g.group(1)), int(g.group(2)); label = f"{ELEMENTS.get(z, 'Z' + str(z))}-{a}"
    else:
        label, a = PARTICLE_NAMES.get(part, (part, 0))
    e = ENERGY_RE.search(text)
    if not e: return label, None
    val = float(e.group(1)) * {"eV": 1e-6, "keV": 1e-3, "MeV": 1, "GeV": 1e3}.get(e.group(2), 1)
    return label, round(val / a, 3) if a else round(val, 3)


def file_tags(rel):
    base = os.path.splitext(os.path.basename(rel))[0]
    return [t for t in re.split(r"[_\-\s.]+", base) if t and not t.isdigit()]


def metrics_1d(x, y):
    """Peak, R80, BPW80 (same units as x) of a depth curve; None where undefined."""
    if len(y) < 5 or max(y) <= 0: return {}
    i = max(range(len(y)), key=y.__getitem__); m = y[i]
    def cross(start, step):
        k = start
        while 0 < k < len(y) - 1 if step < 0 else k < len(y) - 1:
            k2 = k + step
            if y[k] >= 0.8 * m > y[k2]:
                return x[k] + (x[k2] - x[k]) * (y[k] - 0.8 * m) / (y[k] - y[k2])
            k = k2
            if k <= 0: break
        return None
    r80, p80 = cross(i, 1), cross(i, -1)
    if r80 is None: return {"peak": round(x[i], 3)}
    out = {"peak": round(x[i], 3), "r80": round(r80, 3), "bpw80": round(r80 - p80, 3) if p80 is not None else None}
    k = i
    while k < len(y) - 1 and not (y[k] >= 0.2 * m > y[k + 1]): k += 1
    if k < len(y) - 1:
        out["dfo"] = round(x[k] + (x[k + 1] - x[k]) * (y[k] - 0.2 * m) / (y[k] - y[k + 1]) - r80, 3)
    # fragmentation tail: dose beyond the distal fall-off, at R80 + max(10 mm, 4 x (D80-D20)), relative to the maximum
    xt = r80 + max(10.0, 4 * out.get("dfo", 2.5))
    for k in range(len(x) - 1):
        if x[k] <= xt <= x[k + 1]:
            out["tail"] = round((y[k] + (y[k + 1] - y[k]) * (xt - x[k]) / ((x[k + 1] - x[k]) or 1)) / m, 5); break
    return out


def producer_of(rel):
    """Parameter file that wrote a scorer output (same folder or the folder above), with its beam."""
    base = os.path.splitext(os.path.basename(rel))[0]
    folder = os.path.dirname(rel)
    for d in (folder, os.path.dirname(folder)) if folder else ("",):
        try: names = sorted(os.listdir(resolve(d)))
        except (OSError, ValueError): continue
        for n in names:
            if not n.endswith(".txt"): continue
            fp = resolve(os.path.join(d, n))
            if not is_param_file(fp): continue
            with open(fp, errors="replace") as fh: text = fh.read()
            if base in scorer_outputs(text):
                part, energy = beam_info(text)
                e = ENERGY_RE.search(text)
                return {"param": relp(fp), "particle": part, "energy": energy,
                        "total": f"{e.group(1)} {e.group(2)}" if e else None}
    return None


def build_index(scope=""):
    """All result files in the workspace with beam info (from the parameter file that wrote them) and depth-curve metrics.
    Cached in .workbench/index.json by path, size and modification time."""
    root = resolve(scope) if scope else root_real()
    cache_path = os.path.join(wb_dir(), "index.json")
    try:
        with open(cache_path) as fh: cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    # scorer output name -> parameter file, keyed by the parameter file's folder
    producers, files = {}, []
    new_cache = dict(cache) if scope else {}   # keep entries of other folders when only a sub folder is indexed
    for dp, dns, fns in os.walk(root):  # one walk; parameter files are parsed only when they changed
        rel_dp = relp(dp)
        dns[:] = sorted(d for d in dns if scan_allowed(rel_dp, d))
        if not rel_dp and not scan_root_files_allowed(): fns = []
        if scope and rel_dp != scope and not rel_dp.startswith(scope + "/"): fns = []
        for fn in fns:
            fp = os.path.join(dp, fn); rel = relp(fp)
            low = fn.lower()
            if low.endswith(".txt"):
                try: st = os.stat(fp)
                except OSError: continue
                pkey = f"P|{rel}|{st.st_size}|{int(st.st_mtime)}"
                info = cache.get(pkey)
                if info is None:
                    info = {}
                    if is_param_file(fp):
                        try:
                            with open(fp, errors="replace") as fh: text = fh.read()
                            part, energy = beam_info(text)
                            info = {"part": part, "energy": energy, "outs": scorer_outputs(text)}
                        except OSError:
                            pass
                new_cache[pkey] = info
                for o in info.get("outs", []):
                    producers[(os.path.dirname(rel), o)] = (rel, info["part"], info["energy"])
            elif low.endswith(DATA_EXT + IMG_EXT):
                files.append((fp, rel, fn, low))
            if len(files) > 20000: break
    items = []
    for fp, rel, fn, low in files:
        try: st = os.stat(fp)
        except OSError: continue
        key = f"v3|{rel}|{st.st_size}|{int(st.st_mtime)}"
        folder = os.path.dirname(rel)
        item = {"path": rel, "name": fn, "folder": folder, "mtime": st.st_mtime, "size": st.st_size,
                "kind": "image" if low.endswith(IMG_EXT) else "data", "tags": file_tags(rel)}
        base = os.path.splitext(fn)[0]
        prod = producers.get((folder, base)) or producers.get((os.path.dirname(folder), base))
        if prod:
            item.update(param=prod[0], particle=prod[1], energy=prod[2])
        if item["kind"] == "data":
            if key in cache:
                item.update(cache[key])
            else:
                extra = {}
                try:
                    d = read_any(fp)
                    extra["kind"] = "measured" if d.get("measured") else "data"
                    extra["mode"] = d["mode"]
                    extra["quantity"] = d.get("quantity", "")
                    if d["mode"] == "1d":
                        f = {"cm": 10, "m": 1000}.get(d["dims"][d["axis"]]["unit"], 1)
                        extra.update(metrics_1d([v * f for v in d["x"]], d["y"]))
                except (ValueError, OSError, IndexError, KeyError):
                    extra["kind"] = "other"
                item.update(extra)
                cache[key] = extra
            new_cache[key] = {k: item[k] for k in ("kind", "mode", "quantity", "peak", "r80", "bpw80", "dfo", "tail") if k in item}
        items.append(item)
    try:
        os.makedirs(wb_dir(), exist_ok=True)
        with open(cache_path, "w") as fh: json.dump(new_cache, fh)
    except OSError:
        pass
    items.sort(key=lambda i: natkey(i["path"]))
    return items

# ------------------------------------------------------------------ generator plugins
def load_generators():
    gens = {}
    for d in (os.path.join(TOOL_DIR, "generators"), os.path.join(wb_dir(), "generators")):
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py") or f.startswith("_"): continue
            try:
                spec = importlib.util.spec_from_file_location("wbgen_" + f[:-3], os.path.join(d, f))
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                gens[f[:-3]] = mod
            except Exception as e:
                print(f"generator {f} not loaded: {e}", file=sys.stderr)
    return gens


def gen_schema():
    return [{"id": k, "name": getattr(m, "NAME", {"en": k}), "description": getattr(m, "DESCRIPTION", {}),
             "fields": getattr(m, "FIELDS", [])} for k, m in load_generators().items()]

# ------------------------------------------------------------------ file manager
def fm_ls(rel):
    p = resolve(rel)
    items = []
    for f in os.listdir(p):
        if f == ".workbench": continue
        fp = os.path.join(p, f)
        try: st = os.stat(fp)
        except OSError: continue
        items.append({"name": f, "dir": os.path.isdir(fp), "size": st.st_size, "mtime": st.st_mtime})
    items.sort(key=lambda i: (not i["dir"], natkey(i["name"])))
    return {"path": relp(p), "items": items}


def fm_target(src, dst):
    s = resolve(src)
    d = resolve(dst if "/" in dst else os.path.join(os.path.dirname(relp(s)), dst))
    if os.path.isdir(d): d = os.path.join(d, os.path.basename(s))
    if os.path.exists(d): raise ValueError("target exists")
    return s, d


_FILE_INFO = {}    # path -> (mtime_ns, size, {"param": bool, "outs": [...], "names": [...]})
_DIR_NAMES = {}    # folder -> (mtime_ns, [names])
_WALKS = {}         # scope -> {"t", "root", "txt", "res"}
WALK_TTL = 15.0    # seconds; writes through the UI and finished jobs invalidate earlier
NAME_RE = re.compile(r"^\s*((?:d|u|i|b|s|dv|uv|iv|bv|sv):[A-Za-z0-9_/\.\-]+)\s*=", re.M)
PARAM_HINT_RE = re.compile(r"^\s*(d|u|i|b|s|dv|sv|iv|uv|bv):(Ge|So|Sc|Ph|Ts|Gr|Ma|El|Tf|Vr)/", re.M)


DEFAULT_EXCLUDE = ["__pycache__", "node_modules", "venv", ".venv", "build", "dist", "*.egg-info"]


def scan_allowed(rel_dir, name):
    """Whether a sub folder is searched for parameter files and results (Settings: include / exclude)."""
    if name.startswith(".") or name == TRASH: return False
    rel = f"{rel_dir}/{name}" if rel_dir else name
    pats = STATE["cfg"].get("exclude", DEFAULT_EXCLUDE)
    if any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel, p) for p in pats): return False
    inc = [i.strip("/") for i in STATE["cfg"].get("include", []) if i.strip("/")]
    if inc and not any(rel == i or rel.startswith(i + "/") or i.startswith(rel + "/") for i in inc): return False
    return True


def scan_root_files_allowed():
    """Files directly in the workspace root are skipped when only certain folders are included."""
    return not [i for i in STATE["cfg"].get("include", []) if i.strip("/")]


def invalidate_cache():
    _WALKS.clear()


def scope_dir(scope):
    """Workspace sub folder a tab searches ('' = whole workspace)."""
    scope = (scope or "").strip("/")
    if scope and not os.path.isdir(resolve(scope)): raise ValueError(f"folder not found: {scope}")
    return scope


def file_info(fp, st=None):
    """Parsed facts about a .txt file, re-read only when it changed."""
    try: st = st or os.stat(fp)
    except OSError: return None
    c = _FILE_INFO.get(fp)
    if c and c[0] == st.st_mtime_ns and c[1] == st.st_size: return c[2]
    try:
        with open(fp, errors="replace") as fh: text = fh.read()
    except OSError:
        return None
    is_param = bool(PARAM_HINT_RE.search(text[:6000])) or "includeFile" in text[:6000]
    info = {"param": is_param, "outs": scorer_outputs(text) if is_param else [],
            "names": sorted(set(NAME_RE.findall(text))) if is_param else []}
    _FILE_INFO[fp] = (st.st_mtime_ns, st.st_size, info)
    return info


def dir_names(folder):
    """Directory listing, cached by the folder's modification time."""
    try: m = os.stat(folder).st_mtime_ns
    except OSError: return []
    c = _DIR_NAMES.get(folder)
    if c and c[0] == m: return c[1]
    try: names = os.listdir(folder)
    except OSError: names = []
    _DIR_NAMES[folder] = (m, names)
    return names


def _walk(scope=""):
    root = root_real()
    w = _WALKS.get(scope)
    if w and w["root"] == root and time.time() - w["t"] < WALK_TTL: return w
    out = []
    def scan(d, rel):
        try: it = sorted(os.scandir(d), key=lambda e: natkey(e.name))
        except OSError: return
        for e in it:
            if e.is_dir(follow_symlinks=False):
                if scan_allowed(rel, e.name): scan(e.path, f"{rel}/{e.name}" if rel else e.name)
            elif e.name.endswith(".txt") and (rel or scan_root_files_allowed()):
                try: out.append((e.path, e.stat()))
                except OSError: pass
            if len(out) > 20000: return
    if scope: scan(resolve(scope), scope)
    else: scan(root, "")
    w = _WALKS[scope] = {"t": time.time(), "root": root, "txt": out, "res": {}}
    return w


def _walk_txt(scope=""):
    return _walk(scope)["txt"]


def list_param_files(scope=""):
    out = []
    for fp, st in _walk_txt(scope):
        info = file_info(fp, st)
        if info and info["param"]: out.append(relp(fp))
        if len(out) > 5000: break
    return out


def existing_results(rel, outs):
    """Subfolders (relative to the parameter file, "" = same folder) that hold all scorer outputs of the file."""
    if not outs: return []
    d = os.path.dirname(resolve(rel))
    here = dir_names(d)
    cands = [""] + sorted(x for x in here if not x.startswith((".", TRASH)) and "." not in x and os.path.isdir(os.path.join(d, x)))
    found = []
    for sub in cands:
        have = dir_names(os.path.join(d, sub)) if sub else here
        if all(has_output(have, o) for o in outs): found.append(sub)
    return found


def workspace_names(limit=4000):
    """Parameter names (with type prefix) used anywhere in the workspace, for autocompletion."""
    names = set()
    for fp, st in _walk_txt():
        info = file_info(fp, st)
        if info: names.update(info["names"])
        if len(names) > limit: break
    return sorted(names)


def list_params(scope=""):
    out = list_param_files(scope)
    results = {}
    memo = _walk(scope)["res"]   # valid for one walk cycle (TTL or until a write / finished job)
    for f in out:
        if f not in memo:
            info = file_info(resolve(f))
            memo[f] = existing_results(f, info["outs"]) if info else []
        r = memo[f]
        if r: results[f] = r
    last = {}
    for j in STATE["jobs"]:
        last[(j["file"], j["binary"])] = {"status": j["status"], "end": j.get("end", "")}
    return {"files": out, "results": results, "last": {f"{k[0]}|{k[1]}": v for k, v in last.items()}}

# ------------------------------------------------------------------ HTTP
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
        self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        try:
            self.end_headers(); self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser closed the request (reload, tab switch); nothing to report

    def guard(self, need_token=True):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("localhost", "127.0.0.1"): self.send(403, {"error": "bad host"}); return False
        if need_token and self.headers.get("X-Token") != TOKEN and parse_qs(urlparse(self.path).query).get("token", [""])[0] != TOKEN:
            self.send(403, {"error": "bad token"}); return False
        return True

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            if not self.guard(False): return
            page = PAGE.replace("%TOKEN%", TOKEN).replace("%VERSION%", VERSION).replace("%LANG%", STATE["cfg"].get("lang", "en"))
            return self.send(200, page, "text/html")
        if not self.guard(): return
        try:
            if u.path == "/api/info":
                return self.send(200, {"root": STATE["root"], "cfg": STATE["cfg"], "version": VERSION, "detected": detect_binaries(),
                                       "default_exclude": DEFAULT_EXCLUDE,
                                       "folders": sorted(e.name for e in os.scandir(root_real()) if e.is_dir() and not e.name.startswith("."))})
            if u.path == "/api/ls": return self.send(200, fm_ls(q.get("p", "")))
            if u.path == "/api/read":
                p = resolve(q.get("p", ""))
                if os.path.splitext(p)[1].lower() not in TEXT_EXT: raise ValueError("not a text file")
                if os.path.getsize(p) > 3_000_000: raise ValueError("file too large for the editor (> 3 MB)")
                with open(p, errors="replace") as fh: txt = fh.read()
                base = os.path.dirname(p)
                return self.send(200, {"text": txt, "lint": lint(txt, base) if p.endswith(".txt") else [],
                                       "includes": includes(txt, base) if p.endswith(".txt") else []})
            if u.path == "/api/params": return self.send(200, list_params(scope_dir(q.get("scope", ""))))
            if u.path == "/api/names": return self.send(200, {"names": workspace_names()})
            if u.path == "/api/index": return self.send(200, {"items": build_index(scope_dir(q.get("scope", "")))})
            if u.path == "/api/views":
                try:
                    with open(os.path.join(wb_dir(), "views.json")) as fh: return self.send(200, {"views": json.load(fh)})
                except (OSError, ValueError):
                    return self.send(200, {"views": []})
            if u.path == "/api/jobs":
                STATE["last_poll"] = time.time()
                with STATE["lock"]: jobs = [dict(j) for j in STATE["jobs"][-200:]]; running = STATE["proc"] is not None
                for j in jobs:
                    if j["status"] == "running" and j.get("log"):
                        done = log_progress(resolve(j["log"]))
                        j["done"] = done
                        j["live"] = live_info(j)
                        lv = j["live"]
                        if lv.get("rate") and j.get("total"):  # rate from the history counts, excludes the initialisation time
                            j["progress"] = lv["est_done"] / j["total"]
                            j["eta"] = round((j["total"] - lv["est_done"]) / lv["rate"])
                        elif done and j.get("total"):
                            el = time.time() - j.get("t0", time.time())
                            j["progress"] = min(1.0, done / j["total"])
                            j["eta"] = round(el * (j["total"] - done) / done) if done < j["total"] else 0
                return self.send(200, {"jobs": jobs, "running": running, "awake": AWAKE.status(), "queue_eta": queue_estimate(jobs)})
            if u.path == "/api/log":
                p = resolve(q.get("p", ""))
                with open(p, errors="replace") as fh: fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 12000)); return self.send(200, {"text": fh.read()})
            if u.path == "/api/csv":
                col = q.get("collapse"); sl = q.get("slice")
                res = read_any(resolve(q.get("p", "")), int(q.get("stat", 0) or 0),
                               int(col) if col not in (None, "") else None, int(sl) if sl not in (None, "") else None,
                               int(q.get("scan", 0) or 0))
                if not res.get("measured"): res["beam"] = producer_of(relp(resolve(q.get("p", ""))))
                return self.send(200, res)
            if u.path == "/api/generators": return self.send(200, gen_schema())
            if u.path == "/file":
                p = resolve(q.get("p", ""))
                ext = os.path.splitext(p)[1].lower()
                types = {".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".csv": "text/plain", ".txt": "text/plain"}
                if ext not in types: raise ValueError("type not served")
                with open(p, "rb") as fh: data = fh.read()
                self.send_response(200); self.send_header("Content-Type", types[ext]); self.send_header("Content-Length", str(len(data)))
                if q.get("dl"): self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(p)}"')
                self.send_header("Cache-Control", "no-store")
                try:
                    self.end_headers(); self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send(404, {"error": "not found"})
        except (OSError, ValueError, RuntimeError) as e:
            self.send(400, {"error": str(e)})

    def do_POST(self):
        if not self.guard(): return
        u = urlparse(self.path)
        if u.path in ("/api/write", "/api/mkdir", "/api/newfile", "/api/move", "/api/copy", "/api/trash", "/api/generate", "/api/variants"):
            invalidate_cache()
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            if u.path == "/api/write":
                p = resolve(b["p"])
                if os.path.splitext(p)[1].lower() not in TEXT_EXT: raise ValueError("text files only")
                with open(p, "w", newline="") as fh: fh.write(b["text"])
                base = os.path.dirname(p)
                return self.send(200, {"ok": True, "lint": lint(b["text"], base) if p.endswith(".txt") else [],
                                       "includes": includes(b["text"], base) if p.endswith(".txt") else []})
            if u.path == "/api/lint":
                base = os.path.dirname(resolve(b["p"])) if b.get("p") else None
                return self.send(200, {"lint": lint(b.get("text", ""), base), "includes": includes(b.get("text", ""), base)})
            if u.path == "/api/mkdir":
                d = resolve(b["p"])
                if os.path.exists(d): raise ValueError("exists")
                os.makedirs(d); return self.send(200, {"ok": True})
            if u.path == "/api/newfile":
                p = resolve(b["p"])
                if os.path.exists(p): raise ValueError("exists")
                if os.path.splitext(p)[1].lower() not in TEXT_EXT: raise ValueError("text files only")
                open(p, "w").close(); return self.send(200, {"ok": True})
            if u.path == "/api/move":
                s, d = fm_target(b["src"], b["dst"]); os.rename(s, d); return self.send(200, {"ok": True})
            if u.path == "/api/copy":
                s, d = fm_target(b["src"], b["dst"]); (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, d)
                return self.send(200, {"ok": True})
            if u.path == "/api/trash":
                p = resolve(b["p"])
                if p == os.path.realpath(STATE["root"]) or os.path.basename(p) == TRASH: raise ValueError("not allowed")
                t = resolve(TRASH); os.makedirs(t, exist_ok=True)
                os.rename(p, os.path.join(t, time.strftime("%Y%m%d-%H%M%S_") + os.path.basename(p)))
                return self.send(200, {"ok": True})
            if u.path == "/api/enqueue":
                a, s = enqueue(b["files"], b["binary"], b.get("subdir", ""), b.get("skip", False))
                return self.send(200, {"added": a, "skipped": s})
            if u.path == "/api/stop": stop_current(); return self.send(200, {"ok": True})
            if u.path == "/api/views":
                os.makedirs(wb_dir(), exist_ok=True)
                with open(os.path.join(wb_dir(), "views.json"), "w") as fh: json.dump(b.get("views", [])[:200], fh, indent=1)
                return self.send(200, {"ok": True})
            if u.path == "/api/variants":
                files = make_variants(b["src"], b["param"], b.get("values", []), b.get("pattern") or "{name}_{value}")
                if b.get("preview"): return self.send(200, {"names": [n for n, _ in files], "example": files[0][1]})
                written, skipped = [], []
                for rel, body in files:
                    path = resolve(rel)
                    if os.path.exists(path) and not b.get("overwrite"): skipped.append(rel); continue
                    with open(path, "w") as fh: fh.write(body)
                    written.append(rel)
                return self.send(200, {"written": written, "skipped": skipped})
            if u.path == "/api/zip":
                buf, total = io.BytesIO(), 0
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for rel in b.get("paths", [])[:2000]:
                        path = resolve(rel)
                        if not os.path.isfile(path): continue
                        total += os.path.getsize(path)
                        if total > 500_000_000: raise ValueError("selection larger than 500 MB")
                        z.write(path, rel)
                return self.send(200, buf.getvalue(), "application/zip")
            if u.path == "/api/dequeue":
                with STATE["lock"]:
                    for j in STATE["jobs"]:
                        if j["id"] in b.get("ids", []) and j["status"] == "queued": j["status"] = "removed"
                    save_jobs()
                return self.send(200, {"ok": True})
            if u.path == "/api/clearhistory":
                with STATE["lock"]:
                    STATE["jobs"] = [j for j in STATE["jobs"] if j["status"] in ("queued", "running")]; save_jobs()
                return self.send(200, {"ok": True})
            if u.path == "/api/config":
                cfg = STATE["cfg"]
                if "lang" in b: cfg["lang"] = b["lang"] if b["lang"] in ("de", "en") else "en"
                if "keep_awake" in b: cfg["keep_awake"] = bool(b["keep_awake"])
                if "toast" in b: cfg["toast"] = bool(b["toast"])
                for k in ("exclude", "include"):
                    if k in b: cfg[k] = [str(x).strip() for x in b[k] if str(x).strip()][:200]; invalidate_cache()
                if "binaries" in b:
                    bins = []
                    for x in b["binaries"]:
                        if x.get("name") and x.get("cmd"):
                            bins.append({"name": str(x["name"])[:60], "cmd": str(x["cmd"]), "env": {str(k): str(v) for k, v in (x.get("env") or {}).items()},
                                         "subdir": str(x.get("subdir", ""))})
                    cfg["binaries"] = bins
                save_cfg(); return self.send(200, {"ok": True, "cfg": cfg})
            if u.path == "/api/generate":
                gens = load_generators()
                mod = gens.get(b["id"])
                if mod is None: raise ValueError("unknown generator")
                files = mod.generate(b.get("values", {}))
                folder = b.get("folder", "")
                if b.get("preview"):
                    first = next((t for _, t in files if isinstance(t, str)), "")
                    return self.send(200, {"names": [n for n, _ in files], "example": first})
                written, skipped = [], []
                for name, content in files:
                    p = resolve(os.path.join(folder, name))
                    if os.path.exists(p) and not b.get("overwrite"): skipped.append(name); continue
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "wb" if isinstance(content, bytes) else "w") as fh: fh.write(content)
                    written.append(name)
                return self.send(200, {"written": written, "skipped": skipped})
            self.send(404, {"error": "not found"})
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as e:
            self.send(400, {"error": str(e)})


PAGE = open(os.path.join(TOOL_DIR, "ui.html"), encoding="utf-8").read() if os.path.exists(os.path.join(TOOL_DIR, "ui.html")) else "<p>ui.html missing</p>"


def open_browser(url):
    """Open the page; under WSL the Windows browser is used."""
    try:
        with open("/proc/version") as fh: wsl = "microsoft" in fh.read().lower()
    except OSError:
        wsl = False
    try:
        if wsl:
            subprocess.Popen(["cmd.exe", "/c", "start", "", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="/mnt/c")
        else:
            webbrowser.open(url)
    except Exception:
        print("open the address above in your browser")


def main(argv=None):
    ap = argparse.ArgumentParser(description="OpenTOPAS Workbench")
    ap.add_argument("--workspace", default=os.getcwd(), help="folder with your parameter files (default: current folder)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    STATE["root"] = os.path.realpath(a.workspace)
    if not os.path.isdir(STATE["root"]): sys.exit(f"workspace not found: {STATE['root']}")
    load_cfg(); load_jobs(); save_cfg()
    threading.Thread(target=job_worker, daemon=True).start()
    srv = None
    for port in range(a.port, a.port + 20):  # the default port may be taken by an earlier instance
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), H); break
        except OSError:
            print(f"port {port} is in use, trying {port + 1}")
    if srv is None: sys.exit(f"no free port between {a.port} and {a.port + 19}")
    url = f"http://localhost:{port}/"
    print(f"OpenTOPAS Workbench {VERSION}\n  workspace: {STATE['root']}\n  open:      {url}\n  stop:      Ctrl+C")
    if not a.no_browser: open_browser(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_current()
    finally:
        AWAKE.stop()


if __name__ == "__main__":
    main()
