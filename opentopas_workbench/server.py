#!/usr/bin/env python3
"""OpenTOPAS Workbench: a small local web UI for OpenTOPAS and TOPAS.

Manage parameter files, run them in a queue, look at scorer output, and generate parameter sets from plugins.
Pure Python standard library. Runs on Linux, macOS and WSL.

    opentopas-workbench [--workspace DIR] [--port 8765] [--no-browser]
    python3 -m opentopas_workbench [...]

Security: listens on 127.0.0.1 only, checks the Host header and requires a per-session token for every API call,
and only touches files inside the workspace. The one exception is the Machines view: it reads matRad machine files
(.mat) that the page names, by default from the matRad folder in the user's home, and writes nothing there.
"""
import argparse, base64, bisect, csv, fnmatch, io, zipfile, importlib.util, json, math, os, re, secrets, shlex, shutil, signal, socketserver, subprocess, sys, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import __version__ as VERSION
from . import beamprofile, letd, machines
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_EXT = {".txt", ".py", ".sh", ".bat", ".json", ".csv", ".md", ".log", ".in", ".mac", ".xml", ".header", ".binheader", ".mcc", ".asc", ".dat", ".tsv"}
TRASH = "_trash"
TOKEN = secrets.token_urlsafe(24)
STATE = {"root": None, "cfg": {}, "lock": threading.Lock(), "jobs": [], "proc": None, "stop": False, "next_id": 1, "rev": 0}

NOTE_REQUEUED = "re-queued after server restart"
NOTE_ABORTED = "aborted: the server ended twice during this run"
AWAKE_EVERY = 5   # seconds between two keep-awake steps while a job runs
CTRL_C_WINDOW = 3   # seconds in which a second Ctrl+C ends the server while a job runs

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


_DETECTED = {"t": 0.0, "found": []}


def detected_binaries():
    """detect_binaries() at most once a minute: shutil.which walks PATH, which under WSL includes many slow /mnt/c folders."""
    if time.time() - _DETECTED["t"] > 60: _DETECTED.update(t=time.time(), found=detect_binaries())
    return _DETECTED["found"]


def load_cfg():
    path = os.path.join(wb_dir(), "config.json")
    cfg = {}
    try:
        with open(path) as fh: cfg = json.load(fh)
    except (OSError, ValueError): pass
    if not cfg.get("binaries"): cfg["binaries"] = detect_binaries()
    cfg.setdefault("lang", "en")
    STATE["cfg"] = cfg


CFG_LOCK = threading.Lock()


def save_cfg():
    """Atomic and one at a time: the LETd run saves its record from its own thread while requests save too, and a
    half-written config.json would load as {} and lose every setting."""
    os.makedirs(wb_dir(), exist_ok=True)
    path = os.path.join(wb_dir(), "config.json")
    with CFG_LOCK:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh: json.dump(STATE["cfg"], fh, indent=1)
        os.replace(tmp, path)


def load_jobs():
    try:
        with open(os.path.join(wb_dir(), "jobs.json")) as fh: STATE["jobs"] = json.load(fh)
    except (OSError, ValueError): STATE["jobs"] = []
    # Waiting jobs keep waiting. The job that ran when the server ended runs once more; if the server ends again during
    # that second attempt, the job is given up, so a run that takes the server down cannot loop.
    changed = False
    for j in STATE["jobs"]:
        if j["status"] != "running": continue
        changed = True
        if j.get("restarts", 0) >= 1:
            j["status"] = "aborted"; j["note"] = NOTE_ABORTED
        else:
            # a FLUKA run goes back on hold: only its user starts FLUKA, also after a restart
            j["status"] = "held" if j.get("binary") == "FLUKA" else "queued"
            j["restarts"] = j.get("restarts", 0) + 1; j["note"] = NOTE_REQUEUED
            for k in ("start", "t0", "threads", "subruns"): j.pop(k, None)
    STATE["next_id"] = max([j["id"] for j in STATE["jobs"]] + [0]) + 1
    if changed: save_jobs()   # the count must be on disk before the job starts again


def save_jobs():
    STATE["rev"] += 1   # lets the page skip re-rendering the queue when nothing changed
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


def ws_path(rel):
    """Absolute path for a workspace path the workbench produced itself (walk results, job records checked at enqueue).
    Only a lexical check: realpath stats every path component, about a millisecond each on WSL /mnt/c, which adds up
    to seconds for a few hundred cases. Paths from requests always go through resolve()."""
    root = root_real()
    p = os.path.normpath(os.path.join(root, rel or ""))
    if p != root and not p.startswith(root + os.sep): raise ValueError("path outside workspace")
    return p


def inside_ws(path):
    """True if a symlink found during a walk points into the workspace."""
    try: resolve(relp(path)); return True
    except ValueError: return False


def relp(p):
    root = root_real()
    if p.startswith(root + os.sep): return p[len(root) + 1:].replace(os.sep, "/")   # the common case, no syscalls
    r = os.path.relpath(p, root).replace(os.sep, "/")
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


def _seed(text):
    """i:Ts/Seed of a parameter file, None if this file does not set it (TOPAS then uses 1)."""
    m = re.search(r"^\s*i:Ts/Seed\s*=\s*(\d+)", text, re.M)
    return int(m.group(1)) if m else None


ENERGY_TOTAL_RE = re.compile(r"^\s*d:So/[^/]+/BeamEnergy\s*=\s*([\d.eE+\-]+)\s*(\w+)", re.M)
UNIT_MEV = {"eV": 1e-6, "keV": 1e-3, "MeV": 1, "GeV": 1e3}


def job_size(path, max_age=10.0):
    """(histories, total beam energy in MeV) of a parameter file. The queue estimate asks for every waiting job on each
    poll, so a file checked within max_age seconds is not stat'ed again."""
    info = file_info(path, max_age=max_age)
    return (info["hist"], info["etot"]) if info else (None, None)


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
    phase, last = log_state(ws_path(job["log"])) if job.get("log") else ("init", "")
    info = {"phase": phase, "last": last, "elapsed": round(now - job.get("t0", now))}
    done = job.get("done")
    if job.get("subruns"):   # the job's log has no histories: they come from the outputs of the sub-runs
        if L["samples"]: done = job["done"] = max(done or 0, L["samples"][-1][1])   # files move when a cycle ends
        info["phase"] = "init" if not done else "output" if job.get("total") and done >= job["total"] else "run"
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
                try: j["energy"] = job_size(ws_path(j["file"]))[1]
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
        try: n, e = (j.get("total"), None) if j["binary"] == FLUKA_BINARY else job_size(ws_path(j["file"]))
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
    """Blocks system sleep while jobs are queued or running, optionally also display power-off.

    WSL: a hidden PowerShell holds power requests (system and execution required, display required on request) plus
    SetThreadExecutionState, and exits once a heartbeat file in .workbench gets stale. On laptops with Modern Standby,
    Windows may still enter standby once the display is off or the session is locked; keeping the display on avoids that.
    macOS: caffeinate bound to this process. Linux: systemd-inhibit bound to this process."""

    def __init__(self):
        self.proc, self.flag, self.method, self.error, self.last, self.display = None, None, None, None, 0, False

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

    def start(self, display=False):
        if self.active() and bool(display) != self.display: self.stop()   # option changed: restart with the new requests
        if self.active(): self.touch(); return
        if self.proc is not None:   # it was running and has ended: keep the reason visible, retry at most once a minute
            rc = self.proc.poll(); self.proc = None
            self.error = f"helper exited with code {rc}, see .workbench/keepawake.log"
            self.retry_at = time.time() + 60
        if time.time() < getattr(self, "retry_at", 0): self.touch(); return
        plat = self.platform(); pid = str(os.getpid())
        try:
            if plat == "wsl":
                os.makedirs(wb_dir(), exist_ok=True)
                self.flag = os.path.join(wb_dir(), "keepawake.flag"); self.last = 0; self.touch()
                win = subprocess.run(["wslpath", "-w", self.flag], capture_output=True, text=True, timeout=10).stdout.strip()
                # power request types: 0 display, 1 system, 3 execution required; ES flags 0x80000001 or 0x80000003 with display
                types, es = ("0,1,3", "0x80000003") if display else ("1,3", "0x80000001")
                ps = ("$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';"   # no CLIXML progress noise in the log
                      "Add-Type -Name P -Namespace W -MemberDefinition '"
                      "[StructLayout(LayoutKind.Sequential,CharSet=CharSet.Unicode)] public struct RC{public uint V;public uint F;public string S;}"
                      "[DllImport(\"kernel32.dll\",SetLastError=true)] public static extern IntPtr PowerCreateRequest(ref RC c);"
                      "[DllImport(\"kernel32.dll\",SetLastError=true)] public static extern bool PowerSetRequest(IntPtr h,int t);"
                      "[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(int f);';"
                      "$c=New-Object W.P+RC;$c.V=0;$c.F=1;$c.S='OpenTOPAS Workbench: TOPAS runs queued';"
                      "$h=[W.P]::PowerCreateRequest([ref]$c);"
                      f"$ok=foreach($t in {types}){{[W.P]::PowerSetRequest($h,$t)}};"
                      f"$r=[W.P]::SetThreadExecutionState({es});"  # PowerShell reads 0x8000000x as Int32, hence the int parameter
                      f"$f='{win}';\"$(Get-Date -f s) started pid $PID, PowerSetRequest({types})=$($ok -join '/'), SetThreadExecutionState=$r, flag=$f\";"
                      "while((Test-Path $f) -and (((Get-Date)-(Get-Item $f).LastWriteTime).TotalSeconds -lt 120)){Start-Sleep 15};"
                      "\"$(Get-Date -f s) stopped: flag missing or older than 120 s\"")
                enc = base64.b64encode(ps.encode("utf-16-le")).decode()  # avoids quoting problems across the WSL boundary
                log = open(os.path.join(wb_dir(), "keepawake.log"), "ab")
                self.proc = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-EncodedCommand", enc],
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd="/mnt/c")
                log.close()
                self.method = "Windows power request" + (", display on" if display else ""); self.error = None
            elif plat == "macos":
                self.proc = subprocess.Popen(["caffeinate", "-di" if display else "-i", "-w", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.method = "caffeinate" + (", display on" if display else "")
            elif plat == "linux" and shutil.which("systemd-inhibit"):
                self.proc = subprocess.Popen(["systemd-inhibit", "--what=sleep:idle", "--who=OpenTOPAS Workbench", "--why=TOPAS run",
                                              "tail", f"--pid={pid}", "-f", "/dev/null"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.method = "systemd-inhibit"
            else:
                self.error = "not supported on this system"
            self.display = bool(display)
        except (OSError, subprocess.SubprocessError) as e:
            self.proc = None; self.error = str(e)

    def stop(self):
        self.retry_at = 0
        if self.flag:
            try: os.remove(self.flag)
            except OSError: pass
        if self.proc is not None and self.proc.poll() is None:
            try: self.proc.terminate()
            except OSError: pass
            if self.flag:   # the Windows helper is ended before it can write its own last line, so the log gets it from here
                try:
                    with open(os.path.join(os.path.dirname(self.flag), "keepawake.log"), "a") as fh:
                        fh.write(time.strftime("%Y-%m-%dT%H:%M:%S") + " stopped by the workbench\n")
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


def tend_awake():
    """One keep-awake step: hold the helper while a job waits or runs, release it once the queue is empty.

    The Windows helper ends by itself when the heartbeat file gets older than 120 s, so this has to be called
    every few seconds for as long as the queue is busy, also from inside a run."""
    busy = STATE["proc"] is not None or any(j["status"] in ("queued", "running") for j in STATE["jobs"])
    if busy and STATE["cfg"].get("keep_awake", True): AWAKE.start(STATE["cfg"].get("keep_display", False))
    elif AWAKE.active() or AWAKE.flag: AWAKE.stop()


def worker_step():
    """Starts the next waiting job, if none runs, and returns when it has ended."""
    tend_awake()
    with STATE["lock"]:
        if STATE["proc"] is not None: return
        job = next((j for j in STATE["jobs"] if j["status"] == "queued"), None)
        if job is None: return
        job["status"] = "running"; job["start"] = time.strftime("%Y-%m-%d %H:%M:%S"); save_jobs()
    run_job(job)


def job_worker():
    while True:
        time.sleep(0.5)
        worker_step()


# ------------------------------------------------------------------ FLUKA jobs
# FLUKA is licensed to its user personally: the workbench never starts it on its own. A FLUKA input enters the queue
# held, and only the Start button of the queue (/api/release) lets it run.
FLUKA_BINARY = "FLUKA"
FLUKA_DIR_DEFAULT = "/home/robin/fluka/fluka4-5.2/bin"
FLUKA_UNIT = 21     # the USRBIN unit of generators/_fluka.py: rfluka names the cycle files NAME001_fort.21 and so on
# FLUKA runs on one core, so a case is split into sub-runs (sub_runs of generators/_fluka.py), one process each.
FLUKA_PARTS_MAX = 16
FLUKA_SPARE_CORES = 2
FLUKA_RUN = "run"   # every sub-run's input is run.inp in a folder of its own: short names for FLUKA and usbsuw
FLUKA_MARK = ".workbench-subruns"   # in a sub-run folder that the workbench made, the only kind that it empties
MARK_TEXT = "Sub-runs of one case, emptied when it runs again.\n"
RUN_ENV = "WORKBENCH_RUN"   # inherited by each process of a run, so a stop also finds one that left the process group
STOP_GRACE = 5      # seconds between SIGTERM and SIGKILL


def fluka_dir(): return os.path.expanduser(STATE["cfg"].get("fluka_dir") or FLUKA_DIR_DEFAULT)


def usable_cores():
    try: return len(os.sched_getaffinity(0))
    except AttributeError: return os.cpu_count() or 1


def free_cores(seconds=1.0):
    """Cores that idle now. Measured over a second and not taken from the load average, which still counts the case
    that ended just before this one and would leave the next case of a queue two or three cores."""
    def cpu():
        with open(os.path.join(PROC_DIR, "stat")) as fh: v = [int(x) for x in fh.readline().split()[1:9]]
        return v[3] + v[4], sum(v)   # idle and iowait, all
    n = usable_cores()
    try:
        i0, t0 = cpu(); time.sleep(seconds); i1, t1 = cpu()
        return min(n, round((i1 - i0) / (t1 - t0) * (os.cpu_count() or n)))
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        pass
    # No /proc (macOS): the CPU shares of the processes alive now, as ps gives them. Not the load average, which
    # still counts the case that just ended; ps lists only processes that still exist.
    try:
        r = subprocess.run(["ps", "-A", "-o", "%cpu="], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            busy = sum(float(x.replace(",", ".")) for x in r.stdout.split()) / 100
            return max(0, min(n, round(n - busy)))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return n


def fluka_parts():
    """Sub-runs of the case that starts now: the setting, else the free cores minus two, at most 16."""
    try: n = int(STATE["cfg"].get("fluka_parts") or 0)
    except (TypeError, ValueError): n = 0
    return max(1, min(FLUKA_PARTS_MAX, n if n > 0 else free_cores() - FLUKA_SPARE_CORES))


_FLUKA_MOD = []


def fluka_module():
    """generators/_fluka.py, loaded once: the walk asks it for every .inp file of the workspace."""
    if not _FLUKA_MOD:
        spec = importlib.util.spec_from_file_location("wb_fluka", os.path.join(TOOL_DIR, "generators", "_fluka.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        _FLUKA_MOD.append(mod)
    return _FLUKA_MOD[0]


FLUKA_RUNS_SUFFIX = "_fluka_runs"
def fluka_runs_dir(name): return name + FLUKA_RUNS_SUFFIX


def fluka_check():
    """The installation that a job needs, checked before anything of an earlier run is deleted. The folder must be
    absolute: the sub-runs start from folders of their own, where a relative one would point elsewhere."""
    d = fluka_dir()
    if not os.path.isabs(d): raise RuntimeError(f"the FLUKA folder '{d}' is not an absolute path (Settings: FLUKA)")
    for p in ("rfluka", "usbsuw", "usbrea"):
        f = os.path.join(d, p)
        if not (os.path.isfile(f) and os.access(f, os.X_OK)): raise RuntimeError(f"FLUKA program {p} not found in {d} (Settings: FLUKA)")


def fluka_prepare(param, out, name):
    """A fresh folder per sub-run below NAME_fluka_runs, each with its input. (folder, primaries of a cycle of each,
    cycles of each), or None when the job was stopped meanwhile. Fresh, because rfluka goes on from a seed file that
    an earlier run left beside the input. Nothing of the earlier run is deleted before the installation was found and
    the job was not stopped: those sub-runs are often the only result of a long run."""
    fluka_check()
    with open(param) as fh: text = fh.read()
    inputs, starts, cycles = fluka_module().sub_runs(text, fluka_parts())
    if STATE["stop"] or STATE.get("closing"): return None   # the measurement of the free cores takes a second
    top = os.path.join(out, fluka_runs_dir(name))
    mark = os.path.join(top, FLUKA_MARK)
    if os.path.lexists(top):
        ours = not os.path.islink(top) and os.path.isdir(top) and (not os.listdir(top) or os.path.isfile(mark))
        if not ours: raise RuntimeError(f"{fluka_runs_dir(name)} exists and was not made by the workbench: move it away")
        try: shutil.rmtree(top)
        except OSError:   # half emptied: the mark goes back, or every later run would refuse the folder as foreign
            if os.path.isdir(top) and not os.path.exists(mark):
                with open(mark, "w") as fh: fh.write(MARK_TEXT)
            raise
    os.makedirs(top)
    with open(mark, "w") as fh: fh.write(MARK_TEXT)
    for k, body in enumerate(inputs, 1):
        os.mkdir(os.path.join(top, f"{k:02d}"))
        with open(os.path.join(top, f"{k:02d}", FLUKA_RUN + ".inp"), "w") as fh: fh.write(body)
    return top, starts, cycles


def fluka_script(name, parts, cycles):
    """The shell steps of one FLUKA job in its output folder: the sub-runs side by side, then usbsuw merges the binary
    USRBIN files of all their cycles (file names one a line, a blank line, the output name) and usbrea prints the
    merged file as text (input name, output name). Both read their answers from stdin, as the manual's description of
    the programs gives them. The first sub-run that fails, or that ends without its results, ends the job unmerged.

    Every program is called by its path in the fluka_dir setting, never through PATH: a wrong setting must fail the
    job, not start another FLUKA found on PATH. The loop polls, since 'wait -n' needs a newer bash than macOS has."""
    q, d = shlex.quote, fluka_dir()
    dirs = [f"{k:02d}" for k in range(1, parts + 1)]
    results = [f"{FLUKA_RUN}{c:03d}_fort.{FLUKA_UNIT}" for c in range(1, cycles + 1)]
    files = " ".join(f"{p}/{r}" for p in dirs for r in results)
    return "\n".join([
        "set -e",
        f"for p in rfluka usbsuw usbrea; do [ -x {q(d)}/$p ] || {{ echo \"FLUKA program $p not found in {d} (Settings: FLUKA)\"; exit 127; }}; done",
        f"cd {q('./' + fluka_runs_dir(name))}",   # with ./ bash does not search CDPATH
        f"set -- {' '.join(dirs)}",
        "pids=()",
        'for s in "$@"; do',
        f'  ( cd "./$s" && exec {q(d + "/rfluka")} -N0 -M{cycles} {FLUKA_RUN} ) > "$s/console.log" 2>&1 &',
        "  pids+=($!)",
        "done",
        f'echo "Started $# sub-runs, cycles each: {cycles}"',
        "left=$#",
        'while [ "$left" -gt 0 ]; do',
        "  i=0",
        '  for s in "$@"; do',
        '    pid="${pids[$i]}"',
        '    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then',
        '      rc=0; wait "$pid" || rc=$?',
        f'      for r in {" ".join(results)}; do [ "$rc" -ne 0 ] || [ -s "$s/$r" ] || rc=1; done',
        '      if [ "$rc" -ne 0 ]; then',
        '        tail -n 20 "$s/console.log"',
        '        echo "error: sub-run $s of $# failed (rc $rc), nothing merged"',
        '        exit "$rc"',
        "      fi",
        '      pids[$i]=""; left=$((left - 1)); echo "Sub-run $s of $# finished"',
        "    fi",
        "    i=$((i + 1))",
        "  done",
        '  [ "$left" -eq 0 ] || sleep 0.5',
        "done",
        f"{{ printf '%s\\n' {files}; printf '\\nmerged.bnn\\n'; }} > usbsuw.in",
        f"{q(d + '/usbsuw')} < usbsuw.in",
        f"printf 'merged.bnn\\nmerged.txt\\n' | {q(d + '/usbrea')}",
        f"mv -f merged.bnn ../{q(f'{name}_{FLUKA_UNIT}.bnn')}",
        f"mv -f merged.txt ../{q(f'{name}_{FLUKA_UNIT}.txt')}",
        ""])


def fluka_done(job):
    """Primaries handled over all sub-runs of a running job. A cycle whose result file is there counts in full; the
    current one by the last progress line of its standard output, which lies in rfluka's fluka_NNNN folder while it
    runs. Links are skipped: that folder also links into the installation."""
    sr, fk = job["subruns"], fluka_module()
    def handled(path, start):
        with open(path, "rb") as fh:
            fh.seek(0, 2); fh.seek(max(0, fh.tell() - 65536))
            return fk.read_handled(fh.read().decode("utf-8", "replace"), start)
    done = 0
    for k, start in enumerate(sr["starts"], 1):
        here = os.path.join(ws_path(sr["dir"]), f"{k:02d}")
        for c in range(1, sr["cycles"] + 1):
            if os.path.isfile(os.path.join(here, f"{FLUKA_RUN}{c:03d}_fort.{FLUKA_UNIT}")): done += start; continue
            best = 0
            try:
                cand = [os.path.join(here, f"{FLUKA_RUN}{c:03d}.out")]
                with os.scandir(here) as it:
                    tmp = [e.path for e in it if e.name.startswith("fluka_") and e.is_dir(follow_symlinks=False)]
                for t in tmp:
                    with os.scandir(t) as it:
                        cand += [e.path for e in it if (e.name.endswith(".out") or e.name == "fort.11")   # unit 11: standard output
                                 and e.is_file(follow_symlinks=False)]
            except OSError:
                break   # rfluka removes its folder between two cycles
            for p in cand:
                try: best = max(best, handled(p, start))
                except OSError: pass
            done += best
            break
    return done


def run_pids(pgid, mark):
    """The processes of a run that are alive: its process group, and whoever carries its mark in the environment.
    Without /proc the whole group as one negative number, as os.kill takes it."""
    if not os.path.isdir(PROC_DIR):
        try: os.kill(-pgid, 0); return [-pgid]
        except OSError: return []
    found, tag = [], f"{RUN_ENV}={mark}".encode()
    for d in os.listdir(PROC_DIR):
        if not d.isdigit() or int(d) == os.getpid(): continue
        try:
            st = _proc_read(d, "stat").rsplit(")", 1)[1].split()
            if st[0] in ("Z", "X"): continue
            if int(st[2]) == pgid or tag in _proc_read(d, "environ", True).split(b"\0"): found.append(int(d))
        except (OSError, ValueError, IndexError):
            continue   # gone meanwhile, or another user's process
    return found


def end_run(pgid, mark):
    """Ends every process of a run: SIGTERM, and SIGKILL for what is left after STOP_GRACE seconds."""
    for sig, wait in ((signal.SIGTERM, STOP_GRACE), (signal.SIGKILL, 2)):
        pids = run_pids(pgid, mark)
        for p in pids:
            try: os.kill(p, sig)
            except OSError: pass
        end = time.time() + wait
        while pids and time.time() < end:
            time.sleep(0.1); pids = run_pids(pgid, mark)
        if not pids: return


def fluka_version():
    """The FLUKA version as the name of the installation folder gives it (fluka4-5.2/bin gives 4-5.2), or None. Only
    the setting is read, never the installation itself (zone rule: FLUKA files are Robin's licence)."""
    for part in reversed(fluka_dir().replace("\\", "/").split("/")):   # a folder of its own, not any path with "fluka" in it
        m = re.fullmatch(r"fluka[-_]?(\d[\w.\-]*)", part, re.I)
        if m: return m.group(1).rstrip(".-")
    return None


def fluka_convert(param, out, parts=None):
    """usbrea's text output to the workbench CSV, named like the TOPAS scorer file of the same case plus _fluka, so
    both results of a case can sit in one folder (IDD_NAME.csv and IDD_NAME_fluka.csv). Its head names the code,
    version, histories and seed, so the result shows its code without the job record or its folder name."""
    fk = fluka_module()
    with open(param) as fh: text = fh.read()
    m = fk.read_case(text)
    seed = re.search(r"^(?=RANDOMIZ).{20}(.{10})", text, re.M)   # WHAT(2), columns 21 to 30
    meta = [("FLUKA Version", fluka_version()), ("Histories", fk.primaries(m)),
            ("Seed", _num_text(seed.group(1)) if seed else None), ("Sub-runs", parts if parts and parts > 1 else None)]
    name = os.path.splitext(os.path.basename(param))[0]
    with open(os.path.join(out, f"{name}_{FLUKA_UNIT}.txt")) as fh:
        vals, errs = fk.read_usbrea(fh.read(), m["phantom"]["zbins"])
    path = os.path.join(out, m["score"]["name"] + "_fluka.csv")
    with open(path, "w") as fh: fh.write(fk.to_csv(vals, errs, m, meta))
    return path


def job_code(job, param):
    """The code of a finished job as the head of its first result names it, with the histories and seed of its case;
    None when it wrote no result that names a code."""
    for o in job.get("outputs") or []:
        if o.lower().endswith(".csv"):
            c = result_code(resolve(o))
            if c: return with_case(c, file_info(param))
    return None


def _num_text(x):
    """A FLUKA card field as the number it means: "54217137." gives "54217137"."""
    try: v = float(x)
    except ValueError: return None
    return str(int(v)) if v == int(v) else str(v)


GEN_HEAD_RE = re.compile(r"Generated by OpenTOPAS Workbench \((\w+)\)")


def fluka_input_from_topas(f):
    """The FLUKA input (workspace path) of a TOPAS file that a generator wrote, written beside it as NAME.inp, the name
    that the generator's own FLUKA backend gives it. ValueError says why a file has none. An existing NAME.inp with
    other content is left alone: it may be edited by hand, and the queue must not run something else under its name."""
    p = resolve(f)
    with open(p, errors="replace") as fh: text = fh.read()
    m = GEN_HEAD_RE.search(text.split("\n", 1)[0])
    if m is None: raise ValueError("not written by a generator of the workbench, so there is no case model for FLUKA; "
                                   "a FLUKA input (.inp) runs directly")
    mod = load_generators().get(m.group(1))
    if mod is None or not hasattr(mod, "fluka_from_topas"):
        raise ValueError(f"the generator {m.group(1)} writes no FLUKA input yet")
    body = mod.fluka_from_topas(text)
    inp = os.path.splitext(f)[0] + ".inp"
    q = resolve(inp)
    try:    # "x" and not an exists() check first: a NAME.inp that appears in between is never overwritten
        with open(q, "x") as fh: fh.write(body)
        invalidate_cache(soft=True)
    except FileExistsError:
        with open(q, errors="replace") as fh:
            if fh.read() != body: raise ValueError(f"{os.path.basename(inp)} exists with other content: rename or delete it")
    return inp


def release_jobs(ids=None):
    """Held FLUKA jobs to queued: the start by the user's hand. None releases every held job."""
    n = 0
    with STATE["lock"]:
        for j in STATE["jobs"]:
            if j["status"] == "held" and (ids is None or j["id"] in ids): j["status"] = "queued"; n += 1
        save_jobs()
    return n


def run_job(job):
    fluka = job["binary"] == FLUKA_BINARY and job["file"].lower().endswith(".inp")
    binary = None if fluka else next((b for b in STATE["cfg"]["binaries"] if b["name"] == job["binary"]), None)
    logdir = os.path.join(wb_dir(), "logs"); os.makedirs(logdir, exist_ok=True)
    job["log"] = relp(os.path.join(logdir, f"job{job['id']:05d}.log"))
    try:
        param = resolve(job["file"])
        out = resolve(os.path.join(os.path.dirname(job["file"]), job.get("subdir") or ""))
        os.makedirs(out, exist_ok=True)
        if binary is None and not fluka: raise RuntimeError(f"binary '{job['binary']}' not configured")
        env = dict(os.environ, **{k: os.path.expanduser(v) for k, v in (binary or {}).get("env", {}).items()})
        mark = None
        if fluka:   # the case's input stays beside its results, each sub-run has its own below them
            name = os.path.splitext(os.path.basename(param))[0]
            prepared = fluka_prepare(param, out, name)
            if prepared is None:
                if STATE.get("closing"): return   # the record stays "running": the next start runs it again
                with open(resolve(job["log"]), "w") as log: log.write(f"# {job['start']}  stopped before the sub-runs started\n")
                job["status"] = "stopped"; return
            top, starts, cycles = prepared
            if out != os.path.dirname(param): shutil.copy2(param, os.path.join(out, name + ".inp"))
            job["subruns"] = {"dir": relp(top), "starts": starts, "cycles": cycles}
            job["threads"] = len(starts)   # the cores that the job holds, for whoever watches the computer's load
            with STATE["lock"]: save_jobs()   # so jobs.json shows the held cores as the API does
            LIVE.pop(job["id"], None)
            mark = secrets.token_hex(8); env[RUN_ENV] = mark
            cmd = ["bash", "-c", fluka_script(name, len(starts), cycles)]
        else:
            cmd = [os.path.expanduser(binary["cmd"]), param]
        # copy companion geometry files (STL/PLY) so relative InputFile paths work in a subfolder
        if out != os.path.dirname(param):
            for f in os.listdir(os.path.dirname(param)):
                if f.lower().endswith((".stl", ".ply")) and not os.path.exists(os.path.join(out, f)):
                    shutil.copy2(os.path.join(os.path.dirname(param), f), out)
        def snapshot():
            return {f: os.path.getmtime(os.path.join(out, f)) for f in os.listdir(out) if os.path.isfile(os.path.join(out, f))}
        before = snapshot()
        if fluka: job["total"], job["energy"] = sum(starts) * cycles, None
        else: job["total"], job["energy"] = job_size(param, max_age=0)   # the file may have been edited right before the start
        t0 = time.time(); job["t0"] = t0
        with open(resolve(job["log"]), "w") as log:
            # not the script itself: its error messages would show in every log as errors
            what = f"{fluka_dir()}/rfluka -N0 -M{cycles} {FLUKA_RUN}, {len(starts)} sub-runs in {top}" if fluka else binary["cmd"] + " " + param
            log.write(f"# {job['start']}  {what}\n# cwd: {out}\n"); log.flush()
            proc, rc = None, None
            with STATE["lock"]:   # one step with stop_current, so a stop during the preparation starts nothing
                # nor does the end of the server: shut_down ignores signals, and a process started now would inherit that
                if not STATE["stop"] and not STATE.get("closing"):
                    proc = subprocess.Popen(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                    STATE["proc"], STATE["mark"] = proc, mark
            while proc is not None:   # this thread is the only one that feeds the keep-awake heartbeat, so it must not block for the whole run
                try: rc = proc.wait(timeout=AWAKE_EVERY); break
                except subprocess.TimeoutExpired:
                    tend_awake()
                    if mark and STATE["stop"]: end_run(proc.pid, mark)
            if mark and proc is not None and (rc != 0 or STATE["stop"]): end_run(proc.pid, mark)   # the other sub-runs still run
        if STATE.get("closing"): return   # the server is ending: the record stays "running", so the next start runs the job again
        if fluka and rc == 0 and not STATE["stop"]:
            try: fluka_convert(param, out, len(starts))
            except (OSError, ValueError, KeyError) as e:
                rc = 1
                with open(resolve(job["log"]), "a") as log: log.write(f"\nFLUKA result not converted: {e}\n")
        job["seconds"] = round(time.time() - t0)
        job["outputs"] = sorted(relp(os.path.join(out, f)) for f, mt in snapshot().items()
                                if before.get(f) != mt and not f.lower().endswith((".stl", ".ply")))
        job["code"] = job_code(job, param)
        job["status"] = "stopped" if STATE["stop"] else ("done" if rc == 0 else "failed")
        job["rc"] = rc
        job["hints"] = log_hints(resolve(job["log"]))
        if job["status"] == "failed" and not job["hints"]:
            with open(resolve(job["log"]), errors="replace") as fh:
                job["hints"] = [l.rstrip() for l in fh.readlines()[-3:] if l.strip()]
    except Exception as e:  # report everything in the job record
        job["status"] = "failed"; job["error"] = str(e)
    finally:
        if not STATE.get("closing"):
            with STATE["lock"]:
                STATE["proc"] = STATE["mark"] = None; STATE["stop"] = False; job["end"] = time.strftime("%Y-%m-%d %H:%M:%S"); save_jobs()
            invalidate_cache(soft=True)
            notify_job(job)


def enqueue(files, binary, subdir, skip_existing):
    """(added, skipped, refused). A TOPAS file sent to FLUKA goes in as the FLUKA input of its case; one without a case
    model is refused with the reason, and the other files of the selection still go in."""
    added, skipped, refused = [], [], []
    for f in files:
        p = resolve(f)
        if not os.path.isfile(p): raise ValueError(f"not a file: {f}")
        source = None
        if binary == FLUKA_BINARY and not f.lower().endswith(".inp"):
            # OSError too: one unwritable folder must not fail the request after earlier files of it were queued
            try: source, f = f, fluka_input_from_topas(f)
            except (ValueError, OSError) as e: refused.append({"file": f, "reason": str(e)}); continue
            p = resolve(f)
        if f.lower().endswith(".inp") and any(j.get("file") == f and j.get("status") in ("held", "queued") for j in STATE["jobs"]):
            skipped.append(f); continue     # a second copy would run the same case into the same IDD_NAME_fluka.csv
        if skip_existing:
            info = file_info(p)
            outs = info["outs"] if info else []
            od = resolve(os.path.join(os.path.dirname(f), subdir or ""))
            if outs and os.path.isdir(od) and all(has_output(dir_names(od), o) for o in outs):
                skipped.append(f); continue
        fluka = f.lower().endswith(".inp")
        size = {}
        if fluka:   # the case record gives the queue's finish time its histories
            try:
                with open(p) as fh: size["total"] = fluka_module().read_case(fh.read())["beam"]["histories"]
            except (ValueError, KeyError, IndexError, TypeError): pass   # an input from elsewhere
        with STATE["lock"]:
            STATE["jobs"].append({"id": STATE["next_id"], "file": f, "binary": FLUKA_BINARY if fluka else binary, "subdir": subdir,
                                  "status": "held" if fluka else "queued",
                                  "queued": time.strftime("%Y-%m-%d %H:%M:%S"), **size, **({"source": source} if source else {})})
            STATE["next_id"] += 1
        added.append(f)
    with STATE["lock"]: save_jobs()
    return added, skipped, refused


PROC_DIR = "/proc"
SHELLS = {"sh", "bash", "dash", "zsh"}
_EXTERNAL = {"t": 0.0, "root": None, "runs": []}


def _proc_read(pid, name, binary=False):
    with open(os.path.join(PROC_DIR, str(pid), name), "rb" if binary else "r") as fh: return fh.read()


def external_runs(max_age=3.0):
    """TOPAS processes that the queue did not start but that work in the workspace (their working folder or parameter file
    lies in it), so the Runs tab shows every run, also one started in a terminal or by an agent. Linux and WSL read /proc;
    elsewhere the list is empty. Matched by program name: topas, topas4, and the configured programs."""
    root = root_real()
    if time.time() - _EXTERNAL["t"] < max_age and _EXTERNAL["root"] == root: return _EXTERNAL["runs"]
    runs = []
    if os.path.isdir(PROC_DIR):
        names = {"topas", "topas4"} | {os.path.basename(os.path.expanduser(b.get("cmd", ""))) for b in STATE["cfg"].get("binaries", [])}
        prog_of = {os.path.basename(os.path.expanduser(b.get("cmd", ""))): b["name"] for b in STATE["cfg"].get("binaries", []) if b.get("name")}
        own = STATE["proc"]; own_pgid = own.pid if own is not None else None   # queue jobs run in their own session
        try:
            with open(os.path.join(PROC_DIR, "stat")) as fh: btime = next(int(l.split()[1]) for l in fh if l.startswith("btime"))
        except (OSError, StopIteration, ValueError, IndexError): btime = None
        tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        found = {}
        for d in os.listdir(PROC_DIR):
            if not d.isdigit(): continue
            try:
                argv = [a.decode("utf-8", "replace") for a in _proc_read(d, "cmdline", True).split(b"\0") if a]
                if not argv: continue
                if os.path.basename(argv[0]) in SHELLS and len(argv) > 1 and not argv[1].startswith("-"): argv = argv[1:]
                prog = os.path.basename(argv[0])
                if prog not in names: continue
                stat = _proc_read(d, "stat").rsplit(")", 1)[1].split()   # the command name in brackets may contain spaces
                ppid, pgid, start = int(stat[1]), int(stat[2]), int(stat[19])
                if own_pgid is not None and pgid == own_pgid: continue
                cwd = os.readlink(os.path.join(PROC_DIR, d, "cwd"))
            except (OSError, ValueError, IndexError):
                continue   # gone meanwhile, or another user's process
            param = next((a for a in argv[1:] if not a.startswith("-")), "")
            ppath = os.path.realpath(os.path.join(cwd, param)) if param else ""
            inside = lambda p: p == root or p.startswith(root + os.sep)
            if not (inside(os.path.realpath(cwd)) or (ppath and inside(ppath))): continue
            run = {"pid": int(d), "program": prog_of.get(prog, prog), "file": relp(ppath) if ppath and inside(ppath) else param,
                   "cwd": relp(os.path.realpath(cwd)) if inside(os.path.realpath(cwd)) else cwd}
            if btime is not None:
                t0 = btime + start / tick
                run["start"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)); run["elapsed"] = round(time.time() - t0)
            found[int(d)] = (ppid, run)
        # a wrapper script and the TOPAS binary it starts are one run: keep the outermost process
        runs = sorted((r for p, (pp, r) in found.items() if pp not in found), key=lambda r: r["pid"])
    _EXTERNAL.update(t=time.time(), root=root, runs=runs)
    return runs


def stop_current():
    with STATE["lock"]:
        p = STATE["proc"]
        if p is None:   # a job that prepares its start has no process yet
            if any(j["status"] == "running" for j in STATE["jobs"]): STATE["stop"] = True
        elif p.poll() is None:
            STATE["stop"] = True
            try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError: pass   # it ended just now

# ------------------------------------------------------------------ TOPAS CSV reader
BIN_RE = re.compile(r"#\s*(\w+)\s+in\s+(\d+)\s+bins?\s+of\s+([\d.eE+\-]+)\s*(\w+)")


def topas_header(lines):
    """(dims, quantity, statistics) from the '#' header lines of a TOPAS CSV; stops at the first data line."""
    dims, quantity, stats = [], "", []
    for l in lines:
        if not l.startswith("#"): break
        m = BIN_RE.match(l)
        if m: dims.append({"name": m.group(1), "n": int(m.group(2)), "width": float(m.group(3)), "unit": m.group(4)})
        m = re.match(r"#\s*(.+?)\s*:\s*(.+)$", l)
        if m and "bins" not in l and ("(" in m.group(1) or "Sum" in m.group(2) or "Mean" in m.group(2)):
            quantity, stats = m.group(1).strip(), m.group(2).split()
    return dims, quantity, stats


def read_topas_header(path, max_lines=300):
    """Header only, without reading the data rows (a 3D dose file can have millions of them)."""
    with open(path, errors="replace") as fh:
        return topas_header(fh.readline() for _ in range(max_lines))


HEAD_KV_RE = re.compile(r"^#\s*([A-Za-z][\w -]*?)\s*:\s*(.+?)\s*$")


def result_code(path):
    """{"code", "version", "histories", "seed", "parts"} of a result file, from the lines of its head that name them
    ("# TOPAS Version: 4.3", or what fluka_convert writes), or None for a file that names no code (a measurement).
    TOPAS 4 is OpenTOPAS. Histories and seed of a TOPAS result are not in its head; with_case() adds them from the
    parameter file. The folder name is never used: a folder often holds the results of several codes."""
    try:
        with open(path, errors="replace") as fh: head = [fh.readline() for _ in range(12)]
    except OSError: return None
    kv = {}
    for l in head:
        m = HEAD_KV_RE.match(l)
        if m: kv.setdefault(m.group(1).lower(), m.group(2))
    def num(k):
        try: return int(float(kv[k]))
        except (KeyError, ValueError): return None
    if "topas version" in kv:
        v = kv["topas version"]
        major = re.match(r"\d+", v)
        return {"code": "OpenTOPAS" if major and int(major.group()) >= 4 else "TOPAS", "version": v,
                "histories": None, "seed": None, "parts": None}
    if head and head[0].startswith("# FLUKA"):
        return {"code": "FLUKA", "version": kv.get("fluka version"), "histories": num("histories"), "seed": num("seed"),
                "parts": num("sub-runs")}
    return None


def with_case(code, info):
    """The code of a result with histories and seed of the parameter file that wrote it, where its head has none."""
    if code is None or not info: return code
    code = dict(code)
    if code.get("histories") is None: code["histories"] = info.get("hist")
    if code.get("seed") is None: code["seed"] = info.get("seed")
    return code


def read_topas_csv(path, stat=0, collapse=None, slice_index=None):
    with open(path, errors="replace") as fh:
        lines = fh.readlines()
    dims, quantity, stats = topas_header(lines)
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
        try: dp = ws_path(d)   # rel comes from resolve(); its parent folders are inside the workspace too
        except ValueError: continue
        # parsed facts are cached per file, so plotting ten curves from a folder with 100 cases reads each case once
        for n in sorted(dir_names(dp)):
            if not n.endswith(".txt"): continue
            info = file_info(os.path.join(dp, n), max_age=5.0)
            if info and info["param"] and base in info["outs"]:
                return {"param": f"{d}/{n}" if d else n, "particle": info["part"], "energy": info["energy"],
                        "total": info["total"], "hist": info.get("hist"), "seed": info.get("seed")}
    return None


# ------------------------------------------------------------------ code comparison
PLATEAU_MM = (2.0, 6.0)   # the plateau of the code comparison of task 3358 (workspace/offen/fluka_vergleich)
SEED_TOKEN = re.compile(r"(?:s|seed)\d+", re.I)
CMP_KEYS = ("r80", "bpw80", "dfo", "peak_plateau")
CMP_MAX_FILES = 400


def code_name(code): return code["code"] + (" " + code["version"] if code.get("version") else "")


def case_key(name, code):
    """The case a result belongs to: its file name without the parts that name its code or its seed, so that
    IDD_c_s1.csv of TOPAS 3.9, of OpenTOPAS 4.3 (often in folders t39 and t43) and IDD_c_s2_fluka.csv of FLUKA all
    give IDD_c. Only tokens of the result's own code are dropped: t4 is a filter, t43 the version of OpenTOPAS 4.3.
    The code name glued to its version (topas39, fluka4) counts too; the major alone only after a full name, never
    after a bare t, because t3 and t4 name filters.
    A bare number is never dropped: it may be an energy or a thickness that the other codes keep."""
    groups = re.findall(r"\d+", code.get("version") or "")
    names = {code["code"].lower(), "topas", "opentopas"} if code["code"] != "FLUKA" else {"fluka"}
    own = set(names)
    # 4.3.0 is also written 43, FLUKA 4-5.2 also 45 or 452
    for v in {"".join(groups), "".join(groups[:2])} - {""}:
        own |= {n + v for n in names}
        if len(v) > 1: own |= {"t" + v, "ot" + v, "v" + v, "f" + v}   # a version "4" alone must not take the filter t4
    if groups: own |= {n + groups[0] for n in names}
    toks = [t for t in re.split(r"[_\-\s.]+", os.path.splitext(name)[0]) if t]
    kept = [t for t in toks if t.lower() not in own and not SEED_TOKEN.fullmatch(t)]
    return "_".join(kept or toks)


def _stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return None, None
    m = sum(xs) / len(xs)
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None)


def _plateau(x, y):
    pl = [v for xv, v in zip(x, y) if PLATEAU_MM[0] <= xv <= PLATEAU_MM[1]] or y[:5]
    return sum(pl) / len(pl) if pl else 0.0


def _interp(xs, ys, x):
    if x < xs[0] or x > xs[-1]: return None
    k = max(1, bisect.bisect_left(xs, x)); k = min(k, len(xs) - 1)
    x0, x1 = xs[k - 1], xs[k]
    return ys[k - 1] + (ys[k] - ys[k - 1]) * ((x - x0) / (x1 - x0) if x1 != x0 else 0.0)


def _cmp_run(rel, stat):
    fp = resolve(rel)
    code = result_code(fp)
    if code is None: return None, "no code in its head"
    d = read_any(fp, stat)
    if d.get("mode") != "1d" or d.get("measured"): return None, "not a depth curve"
    code = with_case(code, producer_of(relp(fp)))
    if code.get("seed") is None:
        s = [t for t in re.split(r"[_\-\s.]+", os.path.splitext(os.path.basename(fp))[0]) if SEED_TOKEN.fullmatch(t)]
        if s: code["seed"] = int(re.sub(r"\D", "", s[-1]))
    f = {"cm": 10, "m": 1000}.get(d["dims"][d["axis"]]["unit"], 1)
    x, y = [v * f for v in d["x"]], d["y"]
    m = metrics_1d(x, y)
    if "r80" not in m: return None, "no R80"
    pl = _plateau(x, y)
    return {"path": relp(fp), "code": code, "name": code_name(code), "case": case_key(os.path.basename(fp), code),
            "x": x, "y": y, "plateau": pl, "r80": m.get("r80"), "bpw80": m.get("bpw80"), "dfo": m.get("dfo"),
            "peak_plateau": max(y) / pl if pl > 0 else None}, None


def code_compare(paths, ref=None, base_tag="rifi0", stat=0):
    """Results of the same case from different codes side by side: per case and code the seed mean and spread of R80,
    BPW80, D80-D20 and peak/plateau, their difference to the reference code, the filter shift R80(base) - R80(case)
    and broadening BPW80(case) - BPW80(base) against the case with base_tag in its name, and the difference curve to
    the reference code. Curves are scaled to their plateau, the difference is in % of the reference's maximum."""
    runs, skipped = [], []
    for rel in list(dict.fromkeys(paths))[:CMP_MAX_FILES]:
        try: r, why = _cmp_run(rel, stat)
        except (OSError, ValueError, IndexError, KeyError) as e: r, why = None, str(e)
        if r: runs.append(r)
        else: skipped.append({"path": rel, "why": why})
    groups = {}
    for r in runs: groups.setdefault(r["case"], {}).setdefault(r["name"], []).append(r)
    in_cases = {}
    for codes in groups.values():
        for n in codes: in_cases[n] = in_cases.get(n, 0) + 1
    names = sorted(in_cases, key=natkey)
    if ref not in in_cases:
        # the code found in most cases; on a tie OpenTOPAS, the code of the thesis
        ref = max(names, key=lambda n: (in_cases[n], n.startswith("OpenTOPAS")), default=None)
    rows = {}
    for case, codes in groups.items():
        for n, rs in codes.items():
            row = {"code": n, "files": [r["path"] for r in rs], "seeds": sorted(r["code"]["seed"] for r in rs if r["code"].get("seed") is not None),
                   "runs": len(rs), "histories": sorted({r["code"]["histories"] for r in rs if r["code"].get("histories")})}
            for k in CMP_KEYS: row[k], row[k + "_sd"] = _stat([r[k] for r in rs])
            # the seed mean of the curves, on the grid of the first run
            x0 = rs[0]["x"]
            # a point outside a seed's depth range is left out of the mean, not counted as dose 0
            ys = [[(lambda v: v / r["plateau"] if v is not None and r["plateau"] > 0 else None)(_interp(r["x"], r["y"], xv))
                   for xv in x0] for r in rs]
            mean = [[v for v in c if v is not None] for c in zip(*ys)]
            row["_curve"] = ([xv for xv, c in zip(x0, mean) if c], [sum(c) / len(c) for c in mean if c])
            rows[(case, n)] = row
    def base_of(case):
        toks = case.split("_")
        if base_tag in toks: return None
        best = None
        for b in groups:
            bt = b.split("_")
            if base_tag not in bt: continue
            rest, it = [t for t in bt if t != base_tag], iter(toks)
            if all(t in it for t in rest) and (best is None or len(rest) > len(best[1])): best = (b, rest)
        return best[0] if best else None
    cases = []
    for case in sorted(groups, key=natkey):
        base = base_of(case) if base_tag else None
        ref_row = rows.get((case, ref))
        out, diff = [], []
        for n in sorted(groups[case], key=lambda n: (n != ref, natkey(n))):
            row = dict(rows[(case, n)]); row.pop("_curve")
            if ref_row and n != ref:
                for k in CMP_KEYS:
                    a, b = row[k], ref_row[k]
                    row["d_" + k] = a - b if a is not None and b is not None else None
                    row["d_" + k + "_sd"] = math.hypot(row[k + "_sd"] or 0, ref_row[k + "_sd"] or 0) if row["d_" + k] is not None else None
                rx, ry = ref_row["_curve"]; cx, cy = rows[(case, n)]["_curve"]; top = max(ry) or 1.0
                pts = [(xv, _interp(cx, cy, xv)) for xv in rx]
                diff.append({"code": n, "x": [round(xv, 4) for xv, v in pts if v is not None],
                             "y": [round(100 * (v - yv) / top, 4) for (xv, v), yv in zip(pts, ry) if v is not None]})
            b = rows.get((base, n)) if base else None
            if b:
                row["shift"] = b["r80"] - row["r80"]; row["broadening"] = row["bpw80"] - b["bpw80"] if row["bpw80"] is not None and b["bpw80"] is not None else None
                row["shift_sd"] = math.hypot(b["r80_sd"] or 0, row["r80_sd"] or 0)
                row["broadening_sd"] = math.hypot(b["bpw80_sd"] or 0, row["bpw80_sd"] or 0) if row["broadening"] is not None else None
            out.append(row)
        cases.append({"case": case, "base": base, "rows": out, "diff": diff})
    return {"codes": names, "ref": ref, "base_tag": base_tag, "cases": cases, "skipped": skipped}


INDEX_READ_MAX = 64_000_000   # larger data files are listed without metrics instead of being parsed


def index_entry(fp, size):
    """Kind, mode, quantity, code and depth-curve metrics of one data file for the Overview tab."""
    extra = {}
    code = result_code(fp)
    if code: extra["code"] = code
    try:
        dims, quantity, _ = read_topas_header(fp)
        if len(dims) == 3 and sum(d["n"] > 1 for d in dims) >= 2:
            # maps get no metrics, so the data rows (millions for a 3D dose grid) are not needed for the index
            return {"kind": "data", "mode": "2d", "quantity": quantity}
        if size > INDEX_READ_MAX:
            return {"kind": "data" if len(dims) == 3 else "other"}
        d = read_any(fp)
        extra["kind"] = "measured" if d.get("measured") else "data"
        extra["mode"] = d["mode"]
        extra["quantity"] = d.get("quantity", "")
        if d["mode"] == "1d":
            f = {"cm": 10, "m": 1000}.get(d["dims"][d["axis"]]["unit"], 1)
            extra.update(metrics_1d([v * f for v in d["x"]], d["y"]))
    except (ValueError, OSError, IndexError, KeyError):
        extra["kind"] = "other"
    return extra


_INDEX = {}                     # scope -> {"t", "root", "gen", "items"}
_INDEX_LOCK = threading.Lock()


def build_index(scope=""):
    """All result files in the workspace with beam info (from the parameter file that wrote them) and depth-curve metrics.
    Metrics are cached in .workbench/index.json by path, size and modification time; the item list is kept in memory
    for WALK_TTL seconds or until something is written."""
    return _index_refresh(scope)


def index_for_request(scope=""):
    """(items, stale) for the Overview tab: an expired index is answered at once and rebuilt in the background."""
    c = _INDEX.get(scope)
    if c is not None and c["root"] == root_real() and c["gen"] == _GEN[0]:
        if time.time() - c["t"] < WALK_TTL: return c["items"], False
        _background(("index", scope), lambda: _index_refresh(scope))
        return c["items"], True
    return _index_refresh(scope), False


def _index_refresh(scope):
    with _INDEX_LOCK:
        root = root_real()
        c = _INDEX.get(scope)
        if _fresh(c, root): return c["items"]   # built by another request while this one waited
        gen = _GEN[0]
        items = _index_items(scope)
        if gen == _GEN[0]: _INDEX[scope] = {"t": time.time(), "root": root, "gen": gen, "items": items}
        return items


def _index_items(scope):
    w = _walk(scope, sync=True)
    producers = {}   # (folder of the parameter file, scorer output name) -> (parameter file, particle, energy)
    for fp, st in w["txt"]:
        info = file_info(fp, st)
        if info and info["param"]:
            rel = relp(fp)
            for o in info["outs"]:
                producers[(os.path.dirname(rel), o)] = (rel, info["part"], info["energy"], info)
    pc = _pcache(); dcache = pc["data"]
    items, seen = [], set()
    for fp in w["data"]:
        try: st = os.stat(fp)
        except OSError: continue
        rel = relp(fp); fn = os.path.basename(fp); low = fn.lower(); folder = os.path.dirname(rel)
        item = {"path": rel, "name": fn, "folder": folder, "mtime": st.st_mtime, "size": st.st_size,
                "kind": "image" if low.endswith(IMG_EXT) else "data", "tags": file_tags(rel)}
        base = os.path.splitext(fn)[0]
        prod = producers.get((folder, base)) or producers.get((os.path.dirname(folder), base))
        if prod:
            item.update(param=prod[0], particle=prod[1], energy=prod[2])
        if item["kind"] == "data":
            seen.add(rel)
            c = dcache.get(rel)
            if c is not None and c[0] == st.st_mtime_ns and c[1] == st.st_size:
                extra = c[2]
            else:
                extra = index_entry(fp, st.st_size)
                dcache[rel] = [st.st_mtime_ns, st.st_size, extra]; pc["dirty"] = True
            item.update(extra)
            if prod and item.get("code"): item["code"] = with_case(item["code"], prod[3])
        items.append(item)
    if not scope:   # a full walk knows every data file, so entries of deleted files can go
        for rel in [r for r in dcache if r not in seen]:
            del dcache[rel]; pc["dirty"] = True
    _pcache_save()
    items.sort(key=lambda i: natkey(i["path"]))
    return items


def case_outputs(rel):
    """Plottable scorer outputs of a parameter file, in its folder and in result sub folders."""
    fp = resolve(rel)
    info = file_info(fp)
    if not info or not info["outs"]: return []
    d = os.path.dirname(fp)
    here, subs = _case_folder(d, {})
    out = []
    for sub, names in [("", here)] + subs:
        for o in info["outs"]:
            for n in sorted(names, key=natkey):
                if n.startswith(o + ".") and n.lower().endswith(DATA_EXT):
                    out.append(relp(os.path.join(d, sub, n) if sub else os.path.join(d, n)))
    return out


# ------------------------------------------------------------------ beam profile view
def beam_api(what, q):
    """GET /api/beam/<what>: scan (a folder), monitor, spectrum, envelope (a folder and a case), dose (a 3D grid)."""
    mode = "cut" if q.get("mode") == "cut" else "int"
    p = resolve(q.get("p", ""))
    if what == "scan":
        if not os.path.isdir(p): raise ValueError("not a folder")
        return beamprofile.scan(p, relp(p))
    if what == "monitor": return beamprofile.monitor(p, mode)
    if what == "spectrum": return beamprofile._clean(beamprofile.read_spectrum(p))
    if what == "envelope": return beamprofile.envelope(p, relp(p), q.get("case", ""), q.get("kind", "p"), mode)
    if what == "dose": return beamprofile.dose_depth(p, int(q.get("k", 0) or 0), mode)
    raise ValueError("unknown beam request")

# ------------------------------------------------------------------ matRad machines view
def machines_dir():
    return STATE["cfg"].get("machines_dir") or machines.default_dir()


def machine_file(s):
    """A machine file the browser named: any .mat file, also outside the workspace (matRad keeps them in the user folder).
    Read only; nothing of it is copied or logged."""
    p = os.path.realpath(machines.local_path(s))
    if not p.lower().endswith(".mat") or not os.path.isfile(p): raise ValueError("not a .mat file")
    return p


def machines_api(what, q):
    """GET /api/machines/<what>: list (a folder), view, tree (meta and one data element), compare (two files)."""
    if what == "list":
        d = os.path.realpath(machines.local_path(q.get("dir") or machines_dir()))
        return {"dir": d, "default": machines.default_dir(), "files": machines.list_dir(d)}
    try:
        if what == "view": return machines.view(machine_file(q.get("p", "")))
        if what == "tree": return machines.tree(machine_file(q.get("p", "")), int(q.get("i", 0) or 0))
        if what == "compare":
            es = [float(x) for x in q.get("e", "").split(",") if x.strip()]
            return machines.compare(machine_file(q.get("a", "")), machine_file(q.get("b", "")), es,
                                    "abs" if q.get("norm") == "abs" else "max", q.get("offset", "1") != "0")
    except (OSError, ValueError, RuntimeError):
        raise
    except Exception as e:  # scipy and h5py raise many kinds on a file that is not a machine; the view shows the message
        raise ValueError(f"{type(e).__name__}: {e}")
    raise ValueError("unknown machines request")

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
             "fields": getattr(m, "FIELDS", []), "fluka": hasattr(m, "generate_fluka")} for k, m in load_generators().items()]

# ------------------------------------------------------------------ RiFi designer
RIFI_DELTAS = [round(2.0 + 0.25 * i, 2) for i in range(21)]      # mm water, layer spacings of the ripple curves


def rifi_api(what, b):
    """The RiFi designer: templates and library, live evaluation of a definition, saving, SVG masks.

    The definitions, their kernel and the TOPAS block come from the ripple_filter generator and its _rifi.py, so the
    designer shows exactly what a generated case will contain."""
    rf = load_generators().get("ripple_filter")
    if rf is None: raise ValueError("ripple_filter generator not found")
    R = rf.R
    if what == "list":
        return {"templates": [rf.template(v, 1.5) for v in rf.TEMPLATES], "library": list(R.library(wb_dir()).values()),
                "plates": list(R.PLATES), "schema": R.SCHEMA}
    if what == "eval":
        d = R.validate(b.get("def"))
        k = R.kernel(d)
        kt, kw = R.rebin(k["t"], k["w"], 0.05)                   # display bins, mm water
        ref = R.reference()
        rip = R.ripple_table(k["t"], k["w"], RIFI_DELTAS)
        return {"def": d, "mean": round(k["mean"], 4), "sigma": round(k["sigma"], 4),
                "design": round(R.design_spacing(k["t"], k["w"]), 3),
                "kernel": [[round(t, 3), round(w, 6)] for t, w in zip(kt, kw)],
                "deltas": RIFI_DELTAS, "ripple": rip, "lateral": R.lateral(k["mean"]), "lateral_spread": ref["lateral_spread"],
                "hints": R.hints(d), "ridges": [{x: r[x] for x in ("name", "x", "z", "width", "pitch", "tz", "rotz", "shift", "n")} for r in R.ridges(d)],
                "topas": rf.definition_block(d)}
    if what == "save":
        d = R.save(b.get("def"), wb_dir())
        return {"ok": True, "name": d["name"], "variant": rf.LIB + d["name"]}
    if what == "delete":
        R.remove(str(b.get("name", "")), wb_dir()); return {"ok": True}
    if what == "svg":
        d = R.validate(b.get("def"))
        return {"files": [{"name": f"{d['name']}_lage{i + 1}.svg", "svg": R.svg_mask(d, i)} for i in range(len(d["layers"]))]}
    raise ValueError("unknown rifi request")


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


WALK_TTL = 15.0      # seconds; writes through the UI and finished jobs invalidate earlier
WALK_LIMIT = 20000   # files per kind; protects against a workspace pointed at a home folder
TXT_HEAD = 65536     # characters read to tell a parameter file from other text
TXT_MAX = 8_000_000  # parameter files with long value lists stay far below this
NAME_RE = re.compile(r"^\s*((?:d|u|i|b|s|dv|uv|iv|bv|sv):[A-Za-z0-9_/\.\-]+)\s*=", re.M)
PARAM_HINT_RE = re.compile(r"^\s*(d|u|i|b|s|dv|sv|iv|uv|bv):(Ge|So|Sc|Ph|Ts|Gr|Ma|El|Tf|Vr)/", re.M)

_WALKS = {}          # scope -> {"t", "root", "gen", "txt": [(path, stat)], "data": [path], "memo": {}, "params": dict or None}
_DIRS = {}           # folder -> (mtime_ns, names, sub folder names)
_FOLDERS = {}        # (root, depth, limit, exclude) -> (time, generation, folders)
_TXT_CHECKED = {}    # path -> time of the last stat, for callers that accept a slightly older answer
_GEN = [0]           # bumped by invalidate_cache(); a walk that started before a write is not stored
_WALK_LOCK = threading.Lock()
_BG = set()          # keys of background refreshes in progress
_PC = {"root": None, "txt": {}, "data": {}, "dirty": False}   # parsed .txt facts and data metrics, kept in index.json
_PC_LOCK = threading.Lock()


DEFAULT_EXCLUDE = ["__pycache__", "node_modules", "venv", ".venv", "build", "dist", "*.egg-info", "_to_delete"]


def scan_allowed(rel_dir, name):
    """Whether a sub folder is searched for parameter files and results (Settings: include / exclude)."""
    if name.startswith(".") or name == TRASH: return False
    rel = f"{rel_dir}/{name}" if rel_dir else name
    pats = STATE["cfg"].get("exclude", DEFAULT_EXCLUDE)
    if any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel, p) for p in pats): return False
    inc = [i.strip("/") for i in STATE["cfg"].get("include", []) if i.strip("/")]
    if inc and not any(rel == i or rel.startswith(i + "/") or i.startswith(rel + "/") for i in inc): return False
    return True


def list_folders(depth=3, limit=300):
    """Workspace sub folders up to `depth` levels for the folder pickers (Runs, Overview), honoring the exclude list.
    Cached like the walks: every page load asks for it, and deep trees take hundreds of directory reads."""
    pats = STATE["cfg"].get("exclude", DEFAULT_EXCLUDE)
    key = (root_real(), depth, limit, tuple(pats))
    c = _FOLDERS.get(key)
    if c is not None and c[1] == _GEN[0] and time.time() - c[0] < WALK_TTL: return c[2]
    gen, out = _GEN[0], []
    def walk(d, rel, level):
        try:
            with os.scandir(d) as it: entries = sorted(it, key=lambda e: e.name.lower())
        except OSError: return
        for e in entries:
            if len(out) >= limit: return
            if not e.is_dir(follow_symlinks=False) or e.name.startswith(".") or e.name == TRASH: continue
            # a FLUKA case's sub-run folders hold no results of their own, and 16 of them per case would push real
            # folders past the limit, out of the pickers and out of a saved list of searched folders
            if e.name.endswith(FLUKA_RUNS_SUFFIX): continue
            r = f"{rel}/{e.name}" if rel else e.name
            if any(fnmatch.fnmatch(e.name, p) or fnmatch.fnmatch(r, p) for p in pats): continue
            out.append(r)
            if level < depth: walk(e.path, r, level + 1)
    walk(root_real(), "", 1)
    _FOLDERS[key] = (time.time(), gen, out)
    return out


def scan_root_files_allowed():
    """Files directly in the workspace root are skipped when only certain folders are included."""
    return not [i for i in STATE["cfg"].get("include", []) if i.strip("/")]


def invalidate_cache(soft=False):
    """Drop the cached walks, indexes and folder lists. soft: only mark them expired, so the next request is answered
    at once from the old state while a background walk picks up the change (used when a job finishes: rewalking a big
    workspace synchronously after every run is what made the page wait). Writes through the UI use the hard form, so a
    new file shows up in the very next answer."""
    if soft:
        for c in list(_WALKS.values()) + list(_INDEX.values()): c["t"] = 0.0
        _FOLDERS.clear()
        return
    _GEN[0] += 1
    _WALKS.clear(); _INDEX.clear(); _FOLDERS.clear(); _TXT_CHECKED.clear()


def scope_dir(scope):
    """Workspace sub folder a tab searches ('' = whole workspace)."""
    scope = (scope or "").strip("/")
    if scope and not os.path.isdir(resolve(scope)): raise ValueError(f"folder not found: {scope}")
    return scope


# ------------------------------------------------------------------ persistent cache (.workbench/index.json)
def _pcache():
    """Parsed .txt facts (by absolute path) and data metrics (by relative path), loaded once per workspace.
    After a restart the Runs and Overview tabs then only stat files instead of reading them again."""
    root = root_real()
    if _PC["root"] != root:
        txt, data = {}, {}
        try:
            with open(os.path.join(wb_dir(), "index.json")) as fh: raw = json.load(fh)
            if isinstance(raw, dict) and raw.get("v") == 5:   # older formats are rebuilt once
                data = dict(raw.get("data") or {})
                txt = {os.path.join(root, *rel.split("/")): v for rel, v in (raw.get("txt") or {}).items()}
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        _PC.update(root=root, txt=txt, data=data, dirty=False)
    return _PC


def _pcache_save():
    pc = _pcache()
    if not pc["dirty"]: return
    with _PC_LOCK:
        if not pc["dirty"]: return
        pc["dirty"] = False
        pre = pc["root"] + os.sep
        out = {"v": 5, "txt": {k[len(pre):].replace(os.sep, "/"): v for k, v in list(pc["txt"].items()) if k.startswith(pre)},
               "data": dict(pc["data"])}
        try:
            os.makedirs(wb_dir(), exist_ok=True)
            tmp = os.path.join(wb_dir(), "index.json.tmp")
            with open(tmp, "w") as fh: json.dump(out, fh, separators=(",", ":"))
            os.replace(tmp, os.path.join(wb_dir(), "index.json"))   # a crash mid-write must not leave broken JSON
        except OSError:
            pass


def _num(s):
    try: return float(s)
    except (TypeError, ValueError): return None


def file_info(fp, st=None, max_age=0.0):
    """Parsed facts about a .txt file, re-read only when it changed. Of other text files only the start is read, so a
    large export lying in the workspace costs one small read. With max_age, a file stat'ed less than max_age seconds
    ago is not stat'ed again."""
    pc = _pcache(); cache = pc["txt"]
    c = cache.get(fp)
    now = time.time()
    if c is not None and st is None and max_age and now - _TXT_CHECKED.get(fp, 0) < max_age: return c[2]
    try: st = st or os.stat(fp)
    except OSError: return None
    if c is not None and c[0] == st.st_mtime_ns and c[1] == st.st_size:
        _TXT_CHECKED[fp] = now
        return c[2]
    try:
        with open(fp, errors="replace") as fh:
            text = fh.read(TXT_HEAD)
            head = text[:6000]
            is_param = bool(PARAM_HINT_RE.search(head)) or "includeFile" in head
            if is_param and len(text) == TXT_HEAD: text += fh.read(TXT_MAX)
    except OSError:
        return None
    if fp.lower().endswith(".inp"):   # a FLUKA input: in the Runs list, its result is the CSV that fluka_convert writes
        try: rec = fluka_module().read_case(text)
        except (ValueError, KeyError, IndexError, TypeError): rec = None   # an input from elsewhere: no known result
        info = {"param": False, "fluka": True, "outs": [rec["score"]["name"] + "_fluka"] if rec else [], "names": [],
                "part": None, "energy": None, "total": None, "hist": rec["beam"]["histories"] if rec else None, "etot": None}
        cache[fp] = [st.st_mtime_ns, st.st_size, info]; _TXT_CHECKED[fp] = now; pc["dirty"] = True
        return info
    try: part, energy = beam_info(text) if is_param else (None, None)
    except ValueError: part, energy = None, None
    e = ENERGY_TOTAL_RE.search(text)
    etot = _num(e.group(1)) if e else None
    info = {"param": is_param, "outs": scorer_outputs(text) if is_param else [],
            "names": sorted(set(NAME_RE.findall(text))) if is_param else [],
            "part": part, "energy": energy, "total": f"{e.group(1)} {e.group(2)}" if (e and is_param) else None,
            "hist": total_histories(text), "seed": _seed(text), "etot": etot * UNIT_MEV.get(e.group(2), 1) if etot is not None else None}
    cache[fp] = [st.st_mtime_ns, st.st_size, info]; _TXT_CHECKED[fp] = now; pc["dirty"] = True
    return info


def dir_listing(folder):
    """(names, sub folder names) of a folder, cached by the folder's modification time."""
    try: m = os.stat(folder).st_mtime_ns
    except OSError: return [], []
    c = _DIRS.get(folder)
    if c is not None and c[0] == m: return c[1], c[2]
    names, subs = [], []
    try:
        with os.scandir(folder) as it:
            for e in it:
                names.append(e.name)
                try:
                    if e.is_dir(): subs.append(e.name)   # d_type, no extra syscall except for links
                except OSError:
                    pass
    except OSError:
        pass
    _DIRS[folder] = (m, names, subs)
    return names, subs


def dir_names(folder):
    return dir_listing(folder)[0]


# ------------------------------------------------------------------ cached walks
def _fresh(c, root):
    return c is not None and c["root"] == root and c["gen"] == _GEN[0] and time.time() - c["t"] < WALK_TTL


def _background(key, fn):
    """Runs fn in a daemon thread unless the same refresh is already running."""
    if key in _BG: return
    _BG.add(key)
    def run():
        try: fn()
        except Exception as e:  # a failed refresh must not kill the server; the next request tries again
            print(f"background refresh {key} failed: {e}", file=sys.stderr)
        finally: _BG.discard(key)
    threading.Thread(target=run, daemon=True).start()


def _scope_reachable(scope):
    """True if a full walk descends into scope, so a scoped walk can be cut out of it with the same result."""
    rel = ""
    for name in scope.split("/"):
        if not scan_allowed(rel, name): return False
        rel = f"{rel}/{name}" if rel else name
    return True


def _scan(scope):
    """One pass over a folder scope: .txt files with their stat, data and image files by path (stat'ed later, only
    when the Overview needs them). Big binaries such as .mat or DICOM are never opened."""
    root = root_real()
    txt, data = [], []
    def scan(d, rel):
        try:
            with os.scandir(d) as it: entries = sorted(it, key=lambda e: natkey(e.name))
        except OSError: return
        files_ok = bool(rel) or scan_root_files_allowed()
        for e in entries:
            if len(txt) > WALK_LIMIT or len(data) > WALK_LIMIT: return
            try: is_dir = e.is_dir(follow_symlinks=False)
            except OSError: continue
            if is_dir:
                if scan_allowed(rel, e.name): scan(e.path, f"{rel}/{e.name}" if rel else e.name)
                continue
            if not files_ok: continue
            is_txt = e.name.endswith(".txt") or e.name.lower().endswith(".inp")   # .inp: FLUKA inputs for the queue
            if not is_txt and not e.name.lower().endswith(DATA_EXT + IMG_EXT): continue
            if e.is_symlink() and not inside_ws(e.path): continue   # never read through a link that leaves the workspace
            if is_txt:
                try: txt.append((e.path, e.stat()))
                except OSError: pass
            else:
                data.append(e.path)
    scan(os.path.join(root, *scope.split("/")) if scope else root, scope)
    return txt, data


def _build_walk(scope):
    """Called with _WALK_LOCK held."""
    root, gen = root_real(), _GEN[0]
    full = _WALKS.get("")
    if scope and _fresh(full, root) and _scope_reachable(scope):
        pre = os.path.join(root, *scope.split("/")) + os.sep
        w = {"t": full["t"], "root": root, "gen": gen, "txt": [x for x in full["txt"] if x[0].startswith(pre)],
             "data": [p for p in full["data"] if p.startswith(pre)], "memo": full["memo"], "params": None}
    else:
        txt, data = _scan(scope)
        w = {"t": time.time(), "root": root, "gen": gen, "txt": txt, "data": data, "memo": {}, "params": None}
    if gen == _GEN[0]: _WALKS[scope] = w
    return w


def _walk(scope="", sync=False):
    """Cached walk of a folder scope. Younger than WALK_TTL: returned as is. Older: returned at once while a background
    thread walks again (the page asks once more when the answer says it was stale), unless sync is set.
    Writes through the UI and finished jobs drop the cache, so their effect is visible right away."""
    root = root_real()
    w = _WALKS.get(scope)
    if w is not None and w["root"] == root and w["gen"] == _GEN[0]:
        if time.time() - w["t"] < WALK_TTL: return w
        if not sync:
            _background(("walk", scope), lambda: _refresh_walk(scope))
            return w
    with _WALK_LOCK:
        w = _WALKS.get(scope)
        if _fresh(w, root): return w   # built by another request while this one waited
        return _build_walk(scope)


def _refresh_walk(scope):
    with _WALK_LOCK:
        if not _fresh(_WALKS.get(scope), root_real()): _build_walk(scope)
    list_params(scope)   # parse new files and look for results now, so the next request is answered from memory


def _walk_txt(scope=""):
    return _walk(scope)["txt"]


def list_param_files(scope=""):
    out = []
    for fp, st in _walk_txt(scope):
        info = file_info(fp, st)
        if info and info["param"]: out.append(relp(fp))
        if len(out) > 5000: break
    return out


def _case_folder(d, memo):
    """Names in a case folder and in its sub folders that may hold results. Computed once per folder and walk, not
    once per case: plan folders hold a hundred cases next to each other."""
    r = memo.get(d)
    if r is None:
        names, subs = dir_listing(d)
        cands = sorted(x for x in subs if not x.startswith((".", TRASH)) and "." not in x)
        r = memo[d] = (names, [(x, dir_names(os.path.join(d, x))) for x in cands])
    return r


def existing_results(rel, outs, memo=None):
    """Subfolders (relative to the parameter file, "" = same folder) that hold all scorer outputs of the file."""
    if not outs: return []
    here, subs = _case_folder(os.path.dirname(ws_path(rel)), {} if memo is None else memo)
    found = [""] if all(has_output(here, o) for o in outs) else []
    return found + [sub for sub, names in subs if all(has_output(names, o) for o in outs)]


def workspace_names(limit=4000):
    """Parameter names (with type prefix) used anywhere in the workspace, for autocompletion."""
    names = set()
    for fp, st in _walk_txt():
        info = file_info(fp, st)
        if info: names.update(info["names"])
        if len(names) > limit: break
    return sorted(names)


def list_params(scope=""):
    w = _walk(scope)
    stale = time.time() - w["t"] >= WALK_TTL
    p = w["params"]
    if p is None:   # once per walk; afterwards only the job states below are recomputed
        files, results = [], {}
        for fp, st in w["txt"]:
            info = file_info(fp, st)
            if not info or not (info["param"] or info.get("fluka")): continue
            rel = relp(fp)
            files.append(rel)
            r = existing_results(rel, info["outs"], w["memo"])
            if r: results[rel] = r
            if len(files) > 5000: break
        p = w["params"] = {"files": files, "results": results}
        if not scope:   # a full walk knows every .txt file, so facts about deleted files can go
            pc = _pcache(); seen = {fp for fp, _ in w["txt"]}
            for k in [k for k in pc["txt"] if k not in seen]:
                del pc["txt"][k]; pc["dirty"] = True
        _pcache_save()
    last = {}
    for j in STATE["jobs"]:
        last[(j["file"], j["binary"])] = {"status": j["status"], "end": j.get("end", "")}
        if j.get("source"): last[(j["source"], j["binary"])] = last[(j["file"], j["binary"])]   # the TOPAS file it came from
    return {"files": p["files"], "results": p["results"], "last": {f"{k[0]}|{k[1]}": v for k, v in last.items()}, "stale": stale}


def prewarm():
    """Fills the caches right after start, so the first page load does not wait for a full walk."""
    try:
        list_folders(); list_params("")
        if _pcache()["data"]: build_index("")   # only with an earlier index: revalidating it means stat calls, no parsing
    except Exception as e:  # the page builds the caches on demand anyway
        print(f"prewarm failed: {e}", file=sys.stderr)


# ------------------------------------------------------------------ LETd on MRI (letd.py)
LETD_KEEP = ("status", "start", "end", "seconds", "rc", "error", "out", "out_local", "build", "synthseg", "page_local", "html", "t0")


def letd_save(form=None, last=None):
    """The form and the record of the last run in config.json, so the next visit starts from them (also in another
    browser). Only the paths the user typed and the run's state, nothing read from the data."""
    c = STATE["cfg"].setdefault("letd", {})
    if form is not None: c["form"] = {k: str(form.get(k, "") or "")[:1000] for k in letd.FIELDS}
    if last is not None: c["last"] = {k: last[k] for k in LETD_KEEP if k in last}
    save_cfg()


def letd_info():
    c = STATE["cfg"].get("letd", {})
    letd.restore(c.get("last"))
    form = dict(c.get("form") or {"rx": "", "fx": "30", "build": "1", "synthseg": "1"})
    for k, v in letd.default_out(STATE["root"]).items():
        if not form.get(k): form[k] = v
    return {"form": form, "job": letd.status(), "script": os.path.isfile(letd.SCRIPT)}


def letd_post(what, b):
    form = b.get("form") or {}
    viewer = STATE["cfg"].get("viewer_dir")
    if what == "derive": return 200, letd.derive(b.get("mrt", ""))
    if what == "check":
        bad, notes = letd.check(form, STATE["root"], viewer)
        return 200, {"problems": bad, "notes": notes}
    if what == "start":
        bad, notes = letd.check(form, STATE["root"], viewer)
        letd_save(form=form)
        if bad: return 200, {"problems": bad, "notes": notes}
        try: job = letd.start(form, on_end=lambda st: letd_save(last=st), viewer=viewer)
        except RuntimeError: return 409, {"error": "running"}
        letd_save(last=letd.JOB["state"])
        return 200, {"job": job, "notes": notes}
    if what == "stop": return 200, {"stopped": letd.stop()}
    if what == "open": return 200, {"ok": letd.open_html()}
    return 404, {"error": "not found"}


# ------------------------------------------------------------------ HTTP
class Server(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind looks up the host name of the address (socket.getfqdn), a reverse DNS query that can
        # hang for half a minute on macOS before the workbench answers; the name is not used, so skip it
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


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
                return self.send(200, {"root": STATE["root"], "cfg": STATE["cfg"], "version": VERSION, "detected": detected_binaries(),
                                       "default_exclude": DEFAULT_EXCLUDE,
                                       "fluka_dir_default": FLUKA_DIR_DEFAULT,
                                       "viewer_dir_default": letd.VIEWER_DIR_DEFAULT,
                                       "fluka_parts_auto_max": max(1, min(FLUKA_PARTS_MAX, usable_cores() - FLUKA_SPARE_CORES)),
                                       "folders": list_folders()})
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
            if u.path == "/api/index":
                items, stale = index_for_request(scope_dir(q.get("scope", "")))
                return self.send(200, {"items": items, "stale": stale})
            if u.path == "/api/views":
                try:
                    with open(os.path.join(wb_dir(), "views.json")) as fh: return self.send(200, {"views": json.load(fh)})
                except (OSError, ValueError):
                    return self.send(200, {"views": []})
            if u.path == "/api/jobs":
                STATE["last_poll"] = time.time()
                try: hist = max(0, min(int(q.get("hist", 200)), 5000))
                except ValueError: hist = 200
                with STATE["lock"]:
                    alljobs = list(STATE["jobs"]); running = STATE["proc"] is not None; rev = STATE["rev"]
                # every waiting job is sent (a long queue must stay complete for "remove waiting"), history only up to hist
                active = [j for j in alljobs if j["status"] in ("queued", "running", "held")]
                finished = [j for j in alljobs if j["status"] not in ("queued", "running", "held")]
                keep = {id(j) for j in active} | {id(j) for j in finished[max(0, len(finished) - hist):]}
                jobs = [dict(j) for j in alljobs if id(j) in keep]
                counts = {}
                for j in alljobs: counts[j["status"]] = counts.get(j["status"], 0) + 1
                for j in jobs:
                    if j["status"] == "running" and j.get("log"):
                        done = fluka_done(j) if j.get("subruns") else log_progress(ws_path(j["log"]))
                        j["done"] = done
                        j["live"] = live_info(j)
                        lv, done = j["live"], j["done"]
                        if lv.get("rate") and j.get("total"):  # rate from the history counts, excludes the initialisation time
                            j["progress"] = lv["est_done"] / j["total"]
                            j["eta"] = round((j["total"] - lv["est_done"]) / lv["rate"])
                        elif done and j.get("total"):
                            el = time.time() - j.get("t0", time.time())
                            j["progress"] = min(1.0, done / j["total"])
                            j["eta"] = round(el * (j["total"] - done) / done) if done < j["total"] else 0
                est = [dict(j) for j in finished[-200:]] + [j for j in jobs if j["status"] in ("queued", "running")]
                return self.send(200, {"jobs": jobs, "running": running, "awake": AWAKE.status(), "queue_eta": queue_estimate(est),
                                       "counts": counts, "finished": len(finished), "rev": rev, "external": external_runs()})
            if u.path == "/api/log":
                p = resolve(q.get("p", ""))
                with open(p, errors="replace") as fh: fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 12000)); return self.send(200, {"text": fh.read()})
            if u.path == "/api/csv":
                col = q.get("collapse"); sl = q.get("slice")
                res = read_any(resolve(q.get("p", "")), int(q.get("stat", 0) or 0),
                               int(col) if col not in (None, "") else None, int(sl) if sl not in (None, "") else None,
                               int(q.get("scan", 0) or 0))
                if not res.get("measured"):
                    res["beam"] = producer_of(relp(resolve(q.get("p", ""))))
                    res["code"] = with_case(result_code(resolve(q.get("p", ""))), res["beam"])
                return self.send(200, res)
            if u.path.startswith("/api/beam/"): return self.send(200, beam_api(u.path[len("/api/beam/"):], q))
            if u.path.startswith("/api/machines/"): return self.send(200, machines_api(u.path[len("/api/machines/"):], q))
            if u.path == "/api/generators": return self.send(200, gen_schema())
            if u.path == "/api/letd": return self.send(200, letd_info())
            if u.path == "/file":
                p = resolve(q.get("p", ""))
                ext = os.path.splitext(p)[1].lower()
                types = {".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".csv": "text/plain", ".txt": "text/plain"}
                if ext not in types: raise ValueError("type not served")
                with open(p, "rb") as fh: data = fh.read()
                self.send_response(200); self.send_header("Content-Type", types[ext]); self.send_header("Content-Length", str(len(data)))
                if q.get("dl"): self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(p)}"')
                # with a version (the file's mtime) the URL changes when the file does, so the browser may keep it;
                # the token in the URL ties the cached copy to this session
                self.send_header("Cache-Control", "private, max-age=86400" if q.get("v") else "no-store")
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
        writes = u.path in ("/api/write", "/api/mkdir", "/api/newfile", "/api/move", "/api/copy", "/api/trash", "/api/generate", "/api/variants",
                          "/api/rifi/save", "/api/rifi/delete")
        if writes: invalidate_cache()
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
                a, s, r = enqueue(b["files"], b["binary"], b.get("subdir", ""), b.get("skip", False))
                return self.send(200, {"added": a, "skipped": s, "refused": r})
            if u.path == "/api/stop": stop_current(); return self.send(200, {"ok": True})
            if u.path == "/api/release":
                return self.send(200, {"ok": True, "released": release_jobs(None if b.get("all") else set(b.get("ids", [])))})
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
                ids, n = set(b.get("ids", [])), 0
                with STATE["lock"]:
                    for j in STATE["jobs"]:
                        if j["status"] in ("queued", "held") and (b.get("all") or j["id"] in ids): j["status"] = "removed"; n += 1
                    save_jobs()
                return self.send(200, {"ok": True, "removed": n})
            if u.path == "/api/compare":
                return self.send(200, code_compare([str(p) for p in b.get("paths", [])], b.get("ref") or None,
                                                   str(b.get("base", "rifi0") or "").strip(), int(b.get("stat", 0) or 0)))
            if u.path == "/api/outputs":
                out = {}
                for f in b.get("files", [])[:2000]:
                    try: out[f] = case_outputs(f)
                    except (OSError, ValueError): out[f] = []
                return self.send(200, {"outputs": out})
            if u.path == "/api/clearhistory":
                with STATE["lock"]:
                    STATE["jobs"] = [j for j in STATE["jobs"] if j["status"] in ("queued", "running", "held")]; save_jobs()
                return self.send(200, {"ok": True})
            if u.path == "/api/config":
                cfg = STATE["cfg"]
                if "lang" in b: cfg["lang"] = b["lang"] if b["lang"] in ("de", "en") else "en"
                if "keep_awake" in b: cfg["keep_awake"] = bool(b["keep_awake"])
                if "keep_display" in b: cfg["keep_display"] = bool(b["keep_display"])
                if "toast" in b: cfg["toast"] = bool(b["toast"])
                if "machines_dir" in b: cfg["machines_dir"] = str(b["machines_dir"] or "").strip()
                if "fluka_parts" in b:   # before the other settings change: a wrong number saves nothing
                    v = str(b["fluka_parts"] if b["fluka_parts"] is not None else "").strip() or "0"
                    if not v.isdecimal() or int(v) > FLUKA_PARTS_MAX:
                        raise ValueError(f"fluka_parts: a whole number from 1 to {FLUKA_PARTS_MAX}, or empty for automatic")
                    cfg["fluka_parts"] = int(v)
                if "fluka_dir" in b: cfg["fluka_dir"] = str(b["fluka_dir"] or "").strip()
                if "viewer_dir" in b: cfg["viewer_dir"] = str(b["viewer_dir"] or "").strip()
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
            if u.path.startswith("/api/letd/"): return self.send(*letd_post(u.path[len("/api/letd/"):], b))
            if u.path.startswith("/api/rifi/"): return self.send(200, rifi_api(u.path[len("/api/rifi/"):], b))
            if u.path == "/api/generate":
                gens = load_generators()
                mod = gens.get(b["id"])
                if mod is None: raise ValueError("unknown generator")
                if b.get("backend", "topas") == "fluka":
                    if not hasattr(mod, "generate_fluka"): raise ValueError("this generator writes no FLUKA input yet")
                    files = mod.generate_fluka(b.get("values", {}))
                else:
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
        finally:
            if writes: invalidate_cache()   # again after the change: a background walk that ran meanwhile must not stay


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


SERVER_FILE = "server.json"


def write_server_file(url):
    """Address, token and process of this instance in .workbench/server.json, readable by the owner only.
    The MCP adapter (mcp.py) finds a running workbench through it; the token in it grants the same access as the page."""
    os.makedirs(wb_dir(), exist_ok=True)
    path = os.path.join(wb_dir(), SERVER_FILE)
    tmp = f"{path}.{os.getpid()}.tmp"   # own name per process: two starting at once must not replace each other's file
    try: os.unlink(tmp)
    except OSError: pass
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)   # O_EXCL: a leftover file would keep its wider mode
    with os.fdopen(fd, "w") as fh:
        json.dump({"url": url, "token": TOKEN, "pid": os.getpid(), "version": VERSION}, fh)
    os.replace(tmp, path)
    return path


def remove_server_file():
    """Removes server.json unless a later instance on the same workspace has replaced it."""
    path = os.path.join(wb_dir(), SERVER_FILE)
    try:
        with open(path) as fh:
            if json.load(fh).get("pid") != os.getpid(): return
        os.remove(path)
    except (OSError, ValueError):
        pass


def ctrl_c_ends(now):
    """Whether this Ctrl+C ends the server. While a job runs, the first one only asks for a second one within
    CTRL_C_WINDOW seconds, since an accidental Ctrl+C would cost the progress of the run. A signal from outside
    (kill, a closing terminal) and a server without a terminal end at once."""
    try: tty = sys.stdin.isatty()
    except (AttributeError, ValueError): tty = False
    if STATE.get("term") or STATE["proc"] is None or not tty: return True
    if now - STATE.get("ctrl_c", -1e9) <= CTRL_C_WINDOW: return True
    STATE["ctrl_c"] = now
    return False


def shut_down():
    """Stops the running job as the server ends. Further SIGTERM, SIGHUP and Ctrl+C are ignored meanwhile: one would
    break off end_run between SIGTERM and SIGKILL, and sub-runs that ignore SIGTERM would outlive the server."""
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name): signal.signal(getattr(signal, name), signal.SIG_IGN)
    STATE["closing"] = True   # set before the job is stopped, so run_job leaves its record untouched
    stop_current()
    p, mark = STATE["proc"], STATE.get("mark")
    if p is not None and mark: end_run(p.pid, mark)   # the worker thread ends with the server, so not left to it


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] in (["mcp"], ["enqueue"], ["jobs"]):
        from . import mcp
        return {"mcp": mcp.main, "enqueue": mcp.cli_enqueue, "jobs": mcp.cli_jobs}[argv[0]](argv[1:])
    ap = argparse.ArgumentParser(description="OpenTOPAS Workbench. Subcommands: mcp (MCP server), enqueue, jobs; "
                                             "see 'opentopas-workbench mcp --help'")
    ap.add_argument("--workspace", default=os.getcwd(), help="folder with your parameter files (default: current folder)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    STATE["root"] = os.path.realpath(a.workspace)
    if not os.path.isdir(STATE["root"]): sys.exit(f"workspace not found: {STATE['root']}")
    os.environ["OPENTOPAS_WB_DIR"] = wb_dir()    # the ripple_filter generator reads its filter library from here
    load_cfg(); load_jobs(); save_cfg()
    threading.Thread(target=job_worker, daemon=True).start()
    threading.Thread(target=prewarm, daemon=True).start()
    srv = None
    for port in range(a.port, a.port + 20):  # the default port may be taken by an earlier instance
        try:
            srv = Server(("127.0.0.1", port), H); break
        except OSError:
            print(f"port {port} is in use, trying {port + 1}")
    if srv is None: sys.exit(f"no free port between {a.port} and {a.port + 19}")
    port = srv.server_address[1]   # the one the system picked for --port 0
    url = f"http://localhost:{port}/"
    print(f"OpenTOPAS Workbench {VERSION}\n  workspace: {STATE['root']}\n  open:      {url}\n  stop:      Ctrl+C", flush=True)
    write_server_file(url)
    # SIGTERM (kill) and SIGHUP (a closing terminal) end like Ctrl+C, so the running job is stopped and server.json goes away
    def on_term(*_): STATE["term"] = True; raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    if hasattr(signal, "SIGHUP"): signal.signal(signal.SIGHUP, on_term)
    try:
        if not a.no_browser: open_browser(url)
        while True:
            try:
                srv.serve_forever(); break
            except KeyboardInterrupt:
                if ctrl_c_ends(time.time()): raise
                print(f"A job is running. Press Ctrl+C again within {CTRL_C_WINDOW} s to stop it and end the workbench. "
                      "It runs again at the next start; waiting jobs stay in the queue.", flush=True)
    except KeyboardInterrupt:
        shut_down()
    finally:
        remove_server_file()
        AWAKE.stop()


if __name__ == "__main__":
    main()
