#!/usr/bin/env python3
"""OpenTOPAS Workbench: a small local web UI for OpenTOPAS and TOPAS.

Manage parameter files, run them in a queue, look at scorer output, and generate parameter sets from plugins.
Pure Python standard library. Runs on Linux, macOS and WSL.

    opentopas-workbench [--workspace DIR] [--port 8765] [--no-browser]
    python3 -m opentopas_workbench [...]

Security: listens on 127.0.0.1 only, checks the Host header and requires a per-session token for every API call,
and only touches files inside the workspace.
"""
import argparse, base64, csv, fnmatch, io, zipfile, importlib.util, json, os, re, secrets, shutil, signal, socketserver, subprocess, sys, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import __version__ as VERSION
from . import beamprofile
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
    cfg.setdefault("g4_filter", True)   # written to config.json, so the switch for a raw log can be found there
    STATE["cfg"] = cfg


def save_cfg():
    os.makedirs(wb_dir(), exist_ok=True)
    with open(os.path.join(wb_dir(), "config.json"), "w") as fh: json.dump(STATE["cfg"], fh, indent=1)


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
            j["status"] = "queued"; j["restarts"] = j.get("restarts", 0) + 1; j["note"] = NOTE_REQUEUED
            for k in ("start", "t0"): j.pop(k, None)
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


# ------------------------------------------------------------------ run plan: histories and runs of an input
TIME_MS = {"s": 1e3, "ms": 1.0, "us": 1e-3, "mus": 1e-3, "ns": 1e-6, "min": 6e4, "h": 3.6e6}
MAX_RUNS = 200_000   # sequential times up to which the histories of every run are added up
PLAN_HINT_RE = re.compile(r"includeFile|Tf/|PhaseSpace", re.I)
HIST_LITERAL_RE = re.compile(r"^\s*i:So/[^/]+/NumberOfHistoriesInRun\s*=\s*(\d+)\s*(?:#.*)?$", re.M | re.I)


def param_map(text, base_dir=None, depth=0):
    """Parameters of a file and of the files it includes, as {lower-case name: value text}. TOPAS reads the included
    file first, so the including file wins; names do not depend on case."""
    out = {}
    if base_dir is not None and depth < 6:
        for _, _, rel in includes(text, base_dir):
            if rel is None: continue
            try:
                p = ws_path(rel)
                with open(p, errors="replace") as fh: out.update(param_map(fh.read(TXT_MAX), os.path.dirname(p), depth + 1))
            except (OSError, ValueError):
                pass
    for raw in text.splitlines():
        m = PARAM_RE.match(raw.split("#", 1)[0])
        if m and m.group(1): out[m.group(2).lower()] = m.group(3)
    return out


def _vector(val):
    """Numbers and unit of a TOPAS vector value 'n v1 v2 ... [unit]', or (None, None)."""
    toks = (val or "").split()
    if not toks or not toks[0].isdigit(): return None, None
    n = int(toks[0]); nums = []
    for tok in toks[1:1 + n]:
        v = _num(tok)
        if v is None: return None, None
        nums.append(v)
    return (nums, toks[1 + n] if len(toks) > 1 + n else "") if len(nums) == n else (None, None)


def _time_ms(val):
    toks = (val or "").split()
    v = _num(toks[0]) if toks else None
    return None if v is None or len(toks) != 2 or toks[1] not in TIME_MS else v * TIME_MS[toks[1]]


def phsp_histories(name, base_dir, original=False):
    """Histories of a TOPAS phase space file from its header: those that reached the phase space, which is what a source
    replays, or the original ones when the source includes empty histories. None if the header cannot be read."""
    if base_dir is None or not name: return None
    try:
        with open(resolve(relp(os.path.join(base_dir, name + ".header"))), errors="replace") as fh: head = fh.read(4000)
    except (OSError, ValueError):
        return None
    m = re.search(r"^Number of Original Histories:\s*(\d+)" if original else r"^Number of Original Histories that Reached Phase Space:\s*(\d+)", head, re.M)
    return int(m.group(1)) if m else None


def run_plan(text, base_dir=None, per_run=False):
    """How much an input simulates: {"hist": histories of all runs, "runs": number of runs, "kind", "threads", "silent"}.

    TOPAS does one run per sequential time and takes, for each run, the largest history count of all sources
    (TsSourceManager::GetNumberOfHistoriesInRun). kind is "plain" (one run), "steps" (several sequential times, as for
    spots given as time features), "phsp" (a phase space source) or "random" (random time mode: one history per run).
    A count that cannot be worked out is None. silent: the input switches the history lines of the log off.
    With per_run, "per_run" holds the histories of each run."""
    plan = {"hist": None, "runs": 1, "kind": "plain", "threads": None, "silent": False}
    if not PLAN_HINT_RE.search(text):   # the usual file: no include, no time feature, no phase space
        vals = [int(v) for v in HIST_LITERAL_RE.findall(text)]
        pm = {k: v for k, v in ((m.group(2).lower(), m.group(3)) for m in map(PARAM_RE.match, re.findall(r"^\s*i:Ts/[^\n#]+", text, re.M | re.I)) if m)}
        plan["hist"] = max(vals) if vals else None
    else:
        pm = param_map(text, base_dir)
        vals = None
    thr = pm.get("ts/numberofthreads", "1")
    plan["threads"] = int(thr) if re.fullmatch(r"-?\d+", thr) else None
    plan["silent"] = pm.get("ts/showhistorycountatinterval", "1") == "0"
    if vals is not None:
        if per_run: plan["per_run"] = [plan["hist"]] if plan["hist"] is not None else None
        return plan

    steps = {}   # time feature name -> (end times in ms, values, repetition in ms), None if it is no step function

    def step_value(tf, t):
        if tf not in steps:
            times, unit = _vector(pm.get(f"tf/{tf}/times")); values, _ = _vector(pm.get(f"tf/{tf}/values"))
            ok = pm.get(f"tf/{tf}/function", "").strip('"').lower() == "step" and times and values and len(times) == len(values) and unit in TIME_MS
            steps[tf] = ([x * TIME_MS[unit] for x in times], values, _time_ms(pm.get(f"tf/{tf}/repetitioninterval"))) if ok else None
        if steps[tf] is None: return None
        times, values, rep = steps[tf]
        if rep: t = t % rep
        lo, hi = 0, len(times) - 1   # a value holds until its time is reached; after the last time the last value stays
        while lo < hi:
            mid = (lo + hi) // 2
            if t < times[mid] - 1e-9: hi = mid
            else: lo = mid + 1
        return values[lo]

    def integer(expr, t, depth=0):
        """An integer value at time t: a number, another parameter, a time feature value, or number * one of these."""
        expr = expr.strip()
        if re.fullmatch(r"-?\d+", expr): return int(expr)
        m = re.fullmatch(r"(?:(-?[\d.]+)\s*\*\s*)?([A-Za-z0-9_/\.\-]+)", expr)
        if not m or depth > 6: return None
        name = m.group(2).lower(); tf = re.fullmatch(r"tf/(.+)/value", name)
        v = step_value(tf.group(1), t) if tf else (integer(pm[name], t, depth + 1) if name in pm else None)
        return None if v is None else int(round(float(m.group(1) or 1) * v))

    flag = lambda name: pm.get(name, "").strip('"').lower() in ("true", "t", "1")
    sources = sorted({k.split("/")[1] for k in pm if k.startswith("so/") and k.count("/") == 2})
    phsp = [s for s in sources if pm.get(f"so/{s}/type", "").strip('"').lower() == "phasespace"]

    def histories(s, t):
        if s in phsp and pm.get(f"so/{s}/phasespacemultipleuse", "1") != "0":
            n = phsp_histories(pm.get(f"so/{s}/phasespacefilename", "").strip('"'), base_dir, flag(f"so/{s}/phasespaceincludeemptyhistories"))
            use = integer(pm.get(f"so/{s}/phasespacemultipleuse", "1"), t)
            return None if n is None or use is None else n * use
        return integer(pm.get(f"so/{s}/numberofhistoriesinrun", "0"), t)

    if flag("tf/randomizetimedistribution"):
        vals = [integer(pm.get(f"so/{s}/numberofhistoriesinrandomjob", "0"), 0) for s in sources]
        plan["kind"] = "random"
        plan["hist"] = plan["runs"] = max(vals) if vals and None not in vals else None
        return plan
    n = integer(pm.get("tf/numberofsequentialtimes", "1"), 0)
    start = _time_ms(pm.get("tf/timelinestart", "0 s")); end = _time_ms(pm.get("tf/timelineend")) if "tf/timelineend" in pm else start
    plan["kind"] = "phsp" if phsp else "steps" if (n or 1) > 1 else "plain"
    if not n or n < 1 or start is None or end is None: plan["runs"] = n if n and n > 0 else None; return plan
    plan["runs"] = n
    if not sources or n > MAX_RUNS: return plan
    runs = []
    for k in range(n):
        vals = [histories(s, start + k * (end - start) / n) for s in sources]
        if None in vals: return plan
        runs.append(max(vals))
    plan["hist"] = sum(runs)
    if per_run: plan["per_run"] = runs
    return plan


def total_histories(text, base_dir=None):
    """Histories of all runs of an input, None if they cannot be worked out. See run_plan."""
    return run_plan(text, base_dir)["hist"]


ENERGY_TOTAL_RE = re.compile(r"^\s*d:So/[^/]+/BeamEnergy\s*=\s*([\d.eE+\-]+)\s*(\w+)", re.M)
UNIT_MEV = {"eV": 1e-6, "keV": 1e-3, "MeV": 1, "GeV": 1e3}
SIZE_KEYS = ("hist", "etot", "runs", "threads", "kind", "silent")


def job_size(path, max_age=10.0):
    """(histories, total beam energy in MeV) of a parameter file. The queue estimate asks for every waiting job on each
    poll, so a file checked within max_age seconds is not stat'ed again."""
    info = file_info(path, max_age=max_age)
    return (info["hist"], info["etot"]) if info else (None, None)


LIVE_CAP = 0.99   # share that a running job reaches at most between two signs of progress: only its end makes it 1
LIVE = {}   # job id -> what the polls have learned about the running job: its plan, when its first run began, progress samples
TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def proc_usage(pgid):
    """CPU ticks, resident memory and number of threads of all processes in a process group (Linux /proc), or None."""
    if not os.path.isdir("/proc"): return None
    ticks = rss = threads = 0; found = False
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try:
            with open(f"/proc/{d}/stat") as fh: st = fh.read()
            f = st[st.rindex(")") + 2:].split()
            if int(f[2]) != pgid: continue        # field 5 (pgrp)
            ticks += int(f[11]) + int(f[12]); rss += int(f[21]) * PAGE_SIZE; threads += int(f[17]); found = True   # utime, stime, rss, num_threads
        except (OSError, ValueError, IndexError):
            continue
    return (ticks, rss, threads) if found else None


RUN_HIST_RE = re.compile(rb"Run:\s*(\d+),\s*History:\s*(\d+)")
HIST_RE = re.compile(rb"History:\s*(\d+)")
RUN_MARK_RE = re.compile(rb"Begin processing for Run|TimeFeature updated:|Updating Time Features for time interval")
END_MARK_RE = re.compile(rb"run sequence complete|Results have been written|Execution done")


def log_signal(path):
    """What the end of a log tells about progress: the highest run number with the highest history number printed in it
    (TOPAS prints 'Begin processing for Run: R, History: N' at Ts/ShowHistoryCountAtInterval, and for history 0 of every
    run), whether a run has started, whether the run sequence is complete, and the last meaningful line."""
    sig = {"run": None, "hist": None, "started": False, "complete": False, "last": ""}
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 65536)); tail = fh.read()
    except OSError:
        return sig
    pairs = [(int(r), int(h)) for r, h in RUN_HIST_RE.findall(tail)]
    if pairs: sig["run"], sig["hist"] = max(pairs)
    else:
        nums = [int(x) for x in HIST_RE.findall(tail)]   # a program that prints the history without the run
        if nums: sig["hist"] = max(nums)
    sig["complete"] = bool(END_MARK_RE.search(tail))
    sig["started"] = sig["hist"] is not None or sig["complete"] or bool(RUN_MARK_RE.search(tail))
    lines = [l.strip() for l in tail.decode("utf-8", "replace").splitlines() if l.strip() and not l.startswith("# ")]
    sig["last"] = lines[-1][:160] if lines else ""
    return sig


def job_plan(job):
    """The run plan of a running job with the histories before each run, worked out once per job."""
    L = LIVE.setdefault(job["id"], {})
    if "plan" not in L:
        plan = {"hist": job.get("total"), "runs": job.get("runs") or 1, "kind": job.get("kind") or "plain", "per_run": None}
        try:
            p = ws_path(job["file"])
            with open(p, errors="replace") as fh: plan = run_plan(fh.read(TXT_MAX), os.path.dirname(p), per_run=True)
        except (OSError, ValueError):
            pass
        per = plan.get("per_run")
        if per:
            cum = [0]
            for n in per: cum.append(cum[-1] + n)
            plan["cum"] = cum
        L["plan"] = plan
    return L["plan"]


def live_info(job):
    """Live status of the running job: phase, progress, histories per second, CPU cores in use, memory, last log line.

    Progress comes from what the run reports: history lines, and for an input with several runs the run number in them
    or, where Geant4 warns once per run (Run10035), the count of those warnings. "frac" is that measured share and is
    missing when the run reports nothing. The phase changes to "run" with the first of these signs or when the program
    has started its worker threads, which it does at the start of the first run."""
    now = time.time(); L = LIVE.setdefault(job["id"], {})
    sig = log_signal(ws_path(job["log"])) if job.get("log") else {"run": None, "hist": None, "started": False, "complete": False, "last": ""}
    plan = job_plan(job)
    total, runs, cum = plan.get("hist") or job.get("total"), plan.get("runs") or 1, plan.get("cum")
    info = {"last": sig["last"], "elapsed": round(now - job.get("t0", now)), "runs": runs, "kind": plan.get("kind"), "silent": bool(job.get("silent"))}
    proc = STATE.get("proc"); u = None
    if proc is not None and proc.poll() is None: u = proc_usage(proc.pid)
    warned = (job.get("g4_hidden") or {}).get("Run10035", 0)   # one warning per run that has begun
    frac = done = step = None
    if plan.get("kind") == "random":
        if sig["run"] is not None: done = sig["run"]; step = done + 1
    elif runs > 1:
        before = max(sig["run"] or 0, warned - 1, 0)   # runs that have ended
        if sig["run"] is not None or warned: step = min(before + 1, runs)
        if step is None: pass   # no sign yet: nothing is measured, not even zero
        elif cum and before < len(cum): done = cum[before] + (sig["hist"] or 0 if sig["run"] == before else 0)
        else: frac = before / runs
    else:
        done = sig["hist"]
    if done is not None and total: done = min(done, total); frac = done / total
    if sig["started"] or step is not None or (u and u[2] > 1): L.setdefault("run_t0", now)
    info["phase"] = "output" if sig["complete"] else "run" if "run_t0" in L else "init"
    if info["phase"] == "init": L["init_seen"] = True
    if step is not None and runs > 1: info["step"] = step
    if done is not None: info["done"] = done
    if frac is not None:
        info["frac"] = frac
        if frac > 0:
            if "first" not in L: L["first"] = (now, frac)
            if L.get("lastp", (0, -1))[1] != frac: L["lastp"] = (now, frac)
            (ta, fa), (tb, fb) = L["first"], L["lastp"]
            if fb > fa and tb > ta:
                rate = (fb - fa) / (tb - ta)   # share per second since the first sign, so without the initialisation
                info["frac"] = max(fb, min(LIVE_CAP, fb + rate * (now - tb)))   # goes on smoothly between two signs
                info["eta"] = round((1 - info["frac"]) / rate)
                if total: info["rate"] = rate * total; info["est_done"] = info["frac"] * total
            elif frac < 1:
                # one sign only: the share against the time it took. The start of the first run is known only when a
                # poll saw the initialisation before it; else the whole time of the job counts, which errs on the long side.
                span = now - (L["run_t0"] if L.get("init_seen") else job.get("t0", now))
                if span > 0: info["eta"] = round(span * (1 - frac) / frac)
    if "run_t0" in L: info["run_elapsed"] = round(now - L["run_t0"])
    if u:
        if L.get("cpu") and now > L["cpu"][0]:
            info["cores"] = round((u[0] - L["cpu"][1]) / TICK / (now - L["cpu"][0]), 1)
        elif L.get("cores") is not None:
            info["cores"] = L["cores"]
        if L.get("cpu") is None or now - L["cpu"][0] > 1.5:
            L["cpu"] = (now, u[0]); L["cores"] = info.get("cores")
        info["rss_mb"] = round(u[1] / 1e6); info["threads"] = u[2]
    return info


# ------------------------------------------------------------------ runtime model and queue overview
MODEL_JOBS = 40   # finished jobs of one kind that a fit uses
_MODEL = {"key": None, "fits": {}}


def job_measures(j):
    """(key, runs, weight) of a job for the runtime model. The key holds what must be equal for two jobs to share a fit:
    program, thread count, whether the weight is histories x MeV ("e"), histories ("h") or unknown (""), and the folder,
    since the inputs of one study share geometry and physics.
    A record from before these fields existed, and a waiting job, are completed from the parameter file."""
    hist, energy, runs, threads = j.get("total"), j.get("energy"), j.get("runs"), j.get("threads")
    if "runs" not in j or (j["status"] == "queued" and "total" not in j):
        try: info = file_info(ws_path(j["file"]), max_age=10.0)
        except ValueError: info = None
        if info:
            if j["status"] == "queued": hist, energy = info.get("hist"), info.get("etot")
            runs, threads = info.get("runs"), info.get("threads")
    kind = "e" if hist and energy else "h" if hist else ""
    return (j["binary"], threads, kind, os.path.dirname(j["file"])), runs or 1, (hist * energy if kind == "e" else hist or 0)


def model_keys(k):
    """The key of a job and its wider forms, the closest first: the same folder with any thread count, then any folder.
    A job whose histories are unknown is not compared across folders: nothing but the folder says how large it is."""
    keys = [k, (k[0], None, k[2], k[3])]
    if k[2]: keys += [(k[0], k[1], k[2], None), (k[0], None, k[2], None)]
    return list(dict.fromkeys(keys))


def _solve(A, b):
    """Solution of a small linear system by elimination, None if it has none."""
    n = len(b); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12 * max(1.0, max(abs(v) for row in M for v in row)): return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_runtime(rows):
    """Coefficients (a, b, c) of seconds = a + b x runs + c x weight for rows of (seconds, runs, weight), or None.

    Least squares of the relative error, so a long job does not outweigh the short ones. Every choice of the three terms
    is tried; one with a negative coefficient is dropped, since no term can make a run shorter, and a term is only added
    when it brings the error down clearly. With one finished job this is the plain rule of three."""
    rows = [r for r in rows if r[0] > 0]
    if not rows: return None
    has_w = all(r[2] > 0 for r in rows)
    best = None
    for cols in ((2,), (0,), (1,), (0, 2), (1, 2), (0, 1), (0, 1, 2)):
        if (2 in cols and not has_w) or len(rows) < len(cols) + (len(cols) > 1): continue
        X = [[(1.0, r[1], r[2])[c] / r[0] for c in cols] for r in rows]   # each row divided by its seconds: relative error
        A = [[sum(x[i] * x[j] for x in X) for j in range(len(cols))] for i in range(len(cols))]
        sol = _solve(A, [sum(x[i] for x in X) for i in range(len(cols))])
        if sol is None or min(sol) < 0: continue
        err = sum((sum(s * v for s, v in zip(sol, x)) - 1) ** 2 for x in X)
        if best is None or err < best[0] * 0.8 - 1e-12:
            coef = [0.0, 0.0, 0.0]
            for c, s in zip(cols, sol): coef[c] = s
            best = (err, tuple(coef))
    if not best: return None
    ratios = sorted(r[0] / (best[1][0] + best[1][1] * r[1] + best[1][2] * r[2]) for r in rows)
    mid = ratios[len(ratios) // 2] if len(ratios) % 2 else (ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2]) / 2
    return tuple(c * mid for c in best[1])   # half of the jobs took longer than the fit says, half took less


def runtime_model(finished):
    """Fits per kind of job from the finished jobs: {key: (a, b, c)}, also under the wider keys of model_keys for a job
    whose folder or thread count has no finished job yet. Fitted again only when a job has finished."""
    done = [j for j in finished if j["status"] == "done" and j.get("seconds")]
    key = (STATE["root"], len(done), done[-1]["id"] if done else None)
    if _MODEL["key"] == key: return _MODEL["fits"]
    groups = {}
    for j in done:
        k, runs, w = job_measures(j)
        for kk in model_keys(k): groups.setdefault(kk, []).append((j["seconds"], runs, w))
    fits = {k: f for k, f in ((k, fit_runtime(rows[-MODEL_JOBS:])) for k, rows in groups.items()) if f}
    _MODEL.update(key=key, fits=fits)
    return fits


def predict_seconds(fits, j):
    """Expected run time of a job from the model, None when no job of its kind has finished."""
    k, runs, w = job_measures(j)
    f = next((fits[kk] for kk in model_keys(k) if kk in fits), None)
    return None if f is None else f[0] + f[1] * runs + f[2] * w


def queue_overview(jobs, now=None):
    """Progress of the queue as a whole and per folder, and per active job its expected time.

    The queue counts from the moment its oldest waiting or running job was queued: jobs that ended after that belong to
    it. A job's time left is measured when the run reports progress, else it comes from the runtime model and is marked
    as an estimate; a running job past its estimate counts as overdue and adds nothing. Jobs run one after the other,
    so a folder ends when its last job does. Returns ({"total": ..., "groups": [...]}, {job id: {...}})."""
    now = now or time.time()
    active = [j for j in jobs if j["status"] in ("queued", "running")]
    if not active: return None, {}
    since = min(j.get("queued") or j.get("start") or "" for j in active)
    batch = [j for j in jobs if j["status"] not in ("queued", "running") and (j.get("end") or "") >= since and since]
    fits = runtime_model([j for j in jobs if j["status"] not in ("queued", "running")])
    groups, per_job = {}, {}
    total = {"n": 0, "done": 0, "failed": 0, "used": 0, "left": 0, "unknown": 0, "estimated": 0, "share": 0.0, "overdue": 0}

    def group(j):
        name = os.path.dirname(j["file"])
        g = groups.setdefault(name, {"name": name, "n": 0, "done": 0, "failed": 0, "running": 0, "used": 0, "left": 0,
                                     "unknown": 0, "estimated": 0, "share": 0.0, "overdue": 0, "finish": None})
        return g

    def add(g, **kw):
        for d in (g, total):
            for k, v in kw.items(): d[k] += v

    for j in batch:
        ok = j["status"] == "done"
        add(group(j), n=1, done=1 if ok else 0, failed=0 if ok else 1, used=j.get("seconds") or 0, share=1.0)
    ahead = 0.0
    for j in active:
        g = group(j); pred = predict_seconds(fits, j); e = {}
        if pred is not None: e["est"] = round(pred)
        if j["status"] == "running":
            el = now - j.get("t0", now); g["running"] += 1
            if j.get("eta") is not None: left = j["eta"]; share = j.get("progress") or 0.0
            elif pred is not None:
                left = max(0.0, pred - el); share = min(el / pred, 0.99) if pred > 0 else 0.0
                e.update(left=round(left), share=share, estimated=True)
                if el > pred: e["overdue"] = round(el - pred); add(g, overdue=e["overdue"])
            else: left = None; share = 0.0
            add(g, n=1, used=round(el), share=share)
        else:
            left = pred; add(g, n=1)
            if pred is not None: e["estimated"] = True
        if left is None: add(g, unknown=1)
        else:
            add(g, left=left, estimated=1 if e.get("estimated") else 0); ahead += left
        g["finish"] = now + ahead   # the last job of the folder so far ends here, after everything queued before it
        if e: per_job[j["id"]] = e
    for d in list(groups.values()) + [total]:
        d["left"] = round(d["left"]); d["share"] = round(d["share"] / d["n"], 4) if d["n"] else 0.0
    for g in groups.values():   # a folder whose jobs have all ended has no end time ahead
        if g["finish"] is not None: g["finish"] = round(g["finish"])
    total["finish"] = round(now + ahead)
    order = sorted(groups.values(), key=lambda g: (g["finish"] is None, g["finish"] or 0, g["name"]))
    return {"total": total, "groups": order}, per_job


def queue_estimate(jobs):
    """Remaining time of the queue: {"seconds", "unknown" (jobs without an estimate), "finish" (clock time)}, None while
    no active job has an estimate. The short form of queue_overview."""
    ov, _ = queue_overview(jobs)
    if not ov or ov["total"]["unknown"] == sum(1 for j in jobs if j["status"] in ("queued", "running")): return None
    t = ov["total"]
    return {"seconds": t["left"], "unknown": t["unknown"], "finish": time.strftime("%H:%M", time.localtime(t["finish"]))}


ERR_RE = re.compile(r"(?i)(\berror\b|exception|quitting|exiting|segmentation|abort|unknown parameter)")
ERR_IGNORE = re.compile(r"(?i)not necessarily an error")


def log_hints(path, limit=4):
    """Last log lines that look like errors."""
    try:
        with open(path, errors="replace") as fh: lines = fh.readlines()[-3000:]
    except OSError:
        return []
    hits = [l.strip() for l in lines if ERR_RE.search(l) and not ERR_IGNORE.search(l)]
    return hits[-limit:]


# ------------------------------------------------------------------ Geant4 warnings
# Warnings known to be harmless that Geant4 prints once per run, so thousands of times in a job with many short runs.
# An entry names the text for the summary line and the only lines its block may hold between the code and the closing
# "just a warning" line. A block with any other line, another severity, or another origin stays in the log.
# Run10035: TOPAS sets /run/eventModulo before every run, and Geant4 lowers it when a run has fewer histories than threads.
G4_HARMLESS = {b"Run10035": {"text": "Event modulo reduced to 1", "lines": (
    re.compile(rb"\s*issued by : G4(MT|Task)RunManager::InitializeEventLoop\(\)$"),
    re.compile(rb"Event modulo is reduced to \d+( \(was \d+\))? to distribute events to all threads\.$"))}}
G4_START = re.compile(rb"-+ WWWW -+ G4Exception-START -+ WWWW -+$")
G4_END = re.compile(rb"-+ WWWW -+ G4Exception-END -+ WWWW -+$")
G4_CODE = re.compile(rb"\*\*\* G4Exception : (\S+)$")
G4_WARNING = b"*** This is just a warning message. ***"
PUMP_WAIT = 30          # seconds the queue waits, after a run has ended, for the rest of its output
G4_BLOCK_MAX = 12       # lines; a longer block is not one of the short known warnings and is passed on at once
G4_LINE_MAX = 65536     # bytes without a line end that are held back before they are passed on unfiltered
LOG_TAIL = 12000        # characters of a log that the page shows
LOG_WINDOW = 500_000    # bytes read instead when that end holds a warning block, as in a log written without the filter


class G4Filter:
    """Drops complete blocks of the known harmless Geant4 warnings from a stream and counts them per code.

    Output arrives in arbitrary chunks, so lines are put together first and a warning block is held back until its
    end line shows whether it may go. Geant4 puts an empty line before and after a block; both go with it."""

    def __init__(self):
        self.counts = {}
        self.part = b""          # the start of a line whose end has not arrived
        self.held = []           # an empty line that may belong to a block, or the lines of an open block
        self.block = False
        self.skip_blank = False  # the empty line after a dropped block is still to come
        self.long = False        # the start of the current line was passed on unfiltered, its end is still to come

    def feed(self, data):
        lines = (self.part + data).split(b"\n")
        self.part = lines.pop()
        out = []
        if self.long and lines:   # the rest of a line that is no warning line: it goes on as it is, with its line end
            out.append(lines.pop(0) + b"\n"); self.long = False
        for l in lines: self._line(l + b"\n", out)
        if self.long or len(self.part) > G4_LINE_MAX:   # no line of a known block is that long, so nothing is held back for it
            out.extend(self.held); out.append(self.part)
            self.held = []; self.part = b""; self.block = False; self.skip_blank = False; self.long = True
        return b"".join(out)

    def _line(self, raw, out):
        s = raw.rstrip(b"\r\n")
        if self.block:
            if G4_START.match(s):   # the open block never ended: it stays, and this line opens the next one
                out.extend(self.held); self.held = [raw]
                return
            self.held.append(raw)
            if G4_END.match(s): self._close(out)
            elif len(self.held) > G4_BLOCK_MAX: out.extend(self.held); self.held = []; self.block = False
            return
        if self.skip_blank:
            self.skip_blank = False
            if not s: return
        if G4_START.match(s):
            self.held.append(raw); self.block = True
            return
        out.extend(self.held); self.held = []
        if s: out.append(raw)
        else: self.held.append(raw)

    def _close(self, out):
        lines = [l.rstrip(b"\r\n") for l in self.held]
        inner = lines[(0 if G4_START.match(lines[0]) else 1) + 1:-1]
        code = G4_CODE.match(inner[0]) if inner else None
        known = G4_HARMLESS.get(code.group(1)) if code else None
        if known and len(inner) >= 3 and inner[-1] == G4_WARNING and all(any(p.match(l) for p in known["lines"]) for l in inner[1:-1]):
            key = code.group(1).decode()
            self.counts[key] = self.counts.get(key, 0) + 1
            self.skip_blank = True
        else:
            out.extend(self.held)
        self.held = []; self.block = False

    def finish(self):
        """What is still held back when the stream ends: an unfinished block or line stays in the log."""
        out = b"".join(self.held) + self.part
        self.held = []; self.part = b""; self.block = False; self.skip_blank = False; self.long = False
        return out


def g4_summary(counts, lang="en"):
    """One line per hidden warning code for the end of a log. It starts with '# ' like the head lines of a log."""
    form = "# Geant4-Warnung {c} ({t}) {n}-mal ausgeblendet\n" if lang == "de" else "# Geant4 warning {c} ({t}) hidden {n} times\n"
    return "".join(form.format(c=c, t=G4_HARMLESS[c.encode()]["text"], n=n) for c, n in sorted(counts.items()))


def g4_enabled(): return STATE["cfg"].get("g4_filter", True) is not False


class LogPump(threading.Thread):
    """Copies the output of a run into its log through the filter, until the output ends.

    It reads on whatever happens: a pipe that nobody reads would stop TOPAS, and a closed one would end it. So after a
    failed write, or once the queue has gone on without it, the rest of the output is read and thrown away."""

    def __init__(self, src, log, flt, job):
        super().__init__(daemon=True)
        self.src, self.log, self.flt, self.job = src, log, flt, job
        self.lock = threading.Lock()   # the run thread ends the log while this thread may still write
        self.ended = False
        self.last = b"\n"
        self.error = None
        self.raw = None          # why the filter was given up; the output then goes into the log as it comes

    def _write(self, data):
        if not data or self.ended or self.error: return
        try: self.log.write(data); self.log.flush(); self.last = data[-1:]
        except (OSError, ValueError) as e: self.error = str(e)

    def run(self):
        try:
            while True:
                data = os.read(self.src.fileno(), 65536)
                if not data: break
                with self.lock:
                    if self.ended: continue   # the log has its summary line: the count must not move past it
                    n = sum(self.flt.counts.values())
                    if self.raw: out = data
                    else:
                        try: out = self.flt.feed(data)
                        except Exception as e:   # a defect in the filter must not cost the log: what it held goes on with the rest
                            out = b"".join(self.flt.held) + self.flt.part + data
                            self.flt.held = []; self.flt.part = b""; self.raw = f"filter: {e}"
                    self._write(out)
                    if sum(self.flt.counts.values()) != n:
                        with STATE["lock"]: self.job["g4_hidden"] = dict(self.flt.counts)   # the record is saved under this lock
        except (OSError, ValueError):
            pass
        finally:
            self.end()
            try: self.src.close()
            except OSError: pass

    def end(self):
        """Writes what the filter still holds and the summary line, once. Called by the run thread when the output outlives the run."""
        with self.lock:
            if self.ended: return
            rest = self.flt.finish()
            summary = g4_summary(self.flt.counts, STATE["cfg"].get("lang", "en")).encode()
            if summary and not (rest or self.last).endswith(b"\n"): rest += b"\n"
            self._write(rest + summary)
            self.ended = True


def log_tail(path):
    """The end of a log for the page. With the filter on, the known warnings are also left out of a log written without it,
    as far as the last LOG_WINDOW bytes reach."""
    def read(n):
        with open(path, "rb") as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - n))
            return fh.read(), size > n
    raw, cut = read(LOG_TAIL)
    if g4_enabled() and b"G4Exception" in raw:   # a log written with the filter seldom gets here, so the usual poll stays a short read
        raw, cut = read(LOG_WINDOW)
        if cut: raw = raw[raw.find(b"\n") + 1:]
        if cut:   # a window that begins inside a block would leave the end of that block standing
            pos = 0
            for l in raw.split(b"\n", G4_BLOCK_MAX)[:G4_BLOCK_MAX]:
                pos += len(l) + 1
                if G4_START.match(l.rstrip(b"\r")): break
                if G4_END.match(l.rstrip(b"\r")): raw = raw[pos:].lstrip(b"\r\n"); break
        flt = G4Filter(); raw = flt.feed(raw) + flt.finish()
        # the file has no summary line, so the view says what it left out of the part it read; else it could come out empty
        raw = raw[-LOG_TAIL:] + (b"" if raw.endswith(b"\n") or not raw else b"\n") + g4_summary(flt.counts, STATE["cfg"].get("lang", "en")).encode()
    return raw.decode("utf-8", "replace")[-LOG_TAIL:]


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
        size = file_info(param) or {}   # read again if the file was edited right before the start
        job["total"], job["energy"] = size.get("hist"), size.get("etot")
        job["runs"], job["threads"], job["kind"], job["silent"] = size.get("runs"), size.get("threads"), size.get("kind"), bool(size.get("silent"))
        LIVE.pop(job["id"], None)
        t0 = time.time(); job["t0"] = t0
        flt = G4Filter() if g4_enabled() else None   # switched off, TOPAS writes into the log itself, as it always did
        with STATE["lock"]:
            for k in ("g4_hidden", "log_error"): job.pop(k, None)
        with open(resolve(job["log"]), "wb") as log:
            log.write(f"# {job['start']}  {binary['cmd']} {param}\n# cwd: {out}\n".encode("utf-8", "replace")); log.flush()
            proc = subprocess.Popen([os.path.expanduser(binary["cmd"]), param], cwd=out, stdout=subprocess.PIPE if flt else log,
                                    stderr=subprocess.STDOUT, env=env, start_new_session=True)
            with STATE["lock"]: STATE["proc"] = proc
            pump = None
            if flt and getattr(proc, "stdout", None) is not None:
                pump = LogPump(proc.stdout, log, flt, job); pump.start()
            while True:   # this thread is the only one that feeds the keep-awake heartbeat, so it must not block for the whole run
                try: rc = proc.wait(timeout=AWAKE_EVERY); break
                except subprocess.TimeoutExpired: tend_awake()
            # the output ends with the run; a child that outlives it and keeps the pipe open must not hold up the queue
            if pump is not None and not STATE.get("closing"):
                pump.join(PUMP_WAIT); pump.end()
                if pump.error or pump.raw: job["log_error"] = pump.error or pump.raw
        if STATE.get("closing"): return   # the server is ending: the record stays "running", so the next start runs the job again
        job["seconds"] = round(time.time() - t0)
        job["outputs"] = sorted(relp(os.path.join(out, f)) for f, mt in snapshot().items()
                                if before.get(f) != mt and not f.lower().endswith((".stl", ".ply")))
        job["status"] = "stopped" if STATE["stop"] else ("done" if rc == 0 else "failed")
        job["rc"] = rc
        job["hints"] = log_hints(resolve(job["log"]))
        if job["status"] == "failed" and not job["hints"]:
            with open(resolve(job["log"]), errors="replace") as fh:
                job["hints"] = [l.rstrip() for l in [l for l in fh.readlines() if not l.startswith("# Geant4")][-3:] if l.strip()]
    except Exception as e:  # report everything in the job record
        job["status"] = "failed"; job["error"] = str(e)
    finally:
        if not STATE.get("closing"):
            with STATE["lock"]:
                STATE["proc"] = None; STATE["stop"] = False; job["end"] = time.strftime("%Y-%m-%d %H:%M:%S"); save_jobs()
            invalidate_cache(soft=True)
            notify_job(job)


def enqueue(files, binary, subdir, skip_existing):
    added, skipped = [], []
    for f in files:
        p = resolve(f)
        if not os.path.isfile(p): raise ValueError(f"not a file: {f}")
        if skip_existing:
            info = file_info(p)
            outs = info["outs"] if info else []
            od = resolve(os.path.join(os.path.dirname(f), subdir or ""))
            if outs and os.path.isdir(od) and all(has_output(dir_names(od), o) for o in outs):
                skipped.append(f); continue
        with STATE["lock"]:
            STATE["jobs"].append({"id": STATE["next_id"], "file": f, "binary": binary, "subdir": subdir, "status": "queued",
                                  "queued": time.strftime("%Y-%m-%d %H:%M:%S")})
            STATE["next_id"] += 1
        added.append(f)
    with STATE["lock"]: save_jobs()
    return added, skipped


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
        if p is not None and p.poll() is None:
            STATE["stop"] = True
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)

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
                        "total": info["total"]}
    return None


INDEX_READ_MAX = 64_000_000   # larger data files are listed without metrics instead of being parsed


def index_entry(fp, size):
    """Kind, mode, quantity and depth-curve metrics of one data file for the Overview tab."""
    extra = {}
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
                producers[(os.path.dirname(rel), o)] = (rel, info["part"], info["energy"])
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
    try: part, energy = beam_info(text) if is_param else (None, None)
    except ValueError: part, energy = None, None
    e = ENERGY_TOTAL_RE.search(text)
    etot = _num(e.group(1)) if e else None
    info = {"param": is_param, "outs": scorer_outputs(text) if is_param else [],
            "names": sorted(set(NAME_RE.findall(text))) if is_param else [],
            "part": part, "energy": energy, "total": f"{e.group(1)} {e.group(2)}" if (e and is_param) else None,
            "etot": etot * UNIT_MEV.get(e.group(2), 1) if etot is not None else None}
    plan = run_plan(text, os.path.dirname(fp)) if is_param else {}
    info.update(hist=plan.get("hist"), runs=plan.get("runs"), threads=plan.get("threads"), kind=plan.get("kind"), silent=bool(plan.get("silent")))
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
            is_txt = e.name.endswith(".txt")
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
            if not info or not info["param"]: continue
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
    return {"files": p["files"], "results": p["results"], "last": {f"{k[0]}|{k[1]}": v for k, v in last.items()}, "stale": stale}


def prewarm():
    """Fills the caches right after start, so the first page load does not wait for a full walk."""
    try:
        list_folders(); list_params("")
        if _pcache()["data"]: build_index("")   # only with an earlier index: revalidating it means stat calls, no parsing
    except Exception as e:  # the page builds the caches on demand anyway
        print(f"prewarm failed: {e}", file=sys.stderr)


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
                active = [j for j in alljobs if j["status"] in ("queued", "running")]
                finished = [j for j in alljobs if j["status"] not in ("queued", "running")]
                keep = {id(j) for j in active} | {id(j) for j in finished[max(0, len(finished) - hist):]}
                jobs = [dict(j) for j in alljobs if id(j) in keep]
                counts = {}
                for j in alljobs: counts[j["status"]] = counts.get(j["status"], 0) + 1
                for j in jobs:
                    if j["status"] == "running" and j.get("log"):
                        lv = j["live"] = live_info(j)
                        j["done"] = lv.get("done")
                        if lv["phase"] == "output": j["progress"] = 1.0; j["eta"] = 0
                        elif lv.get("frac") is not None:   # measured from what the run reports
                            j["progress"] = lv["frac"]
                            if lv.get("eta") is not None: j["eta"] = lv["eta"]
                overview, per_job = queue_overview([dict(j) for j in finished[-200:]] + [j for j in jobs if j["status"] in ("queued", "running")])
                for j in jobs:
                    if j["id"] in per_job and j["status"] in ("queued", "running"): j["plan"] = per_job[j["id"]]
                t = overview["total"] if overview else None
                eta = {"seconds": t["left"], "unknown": t["unknown"], "finish": time.strftime("%H:%M", time.localtime(t["finish"]))} if t and t["unknown"] < len(active) else None
                return self.send(200, {"jobs": jobs, "running": running, "awake": AWAKE.status(), "queue_eta": eta, "overview": overview, "now": round(time.time()),
                                       "counts": counts, "finished": len(finished), "rev": rev, "external": external_runs()})
            if u.path == "/api/log":
                p = resolve(q.get("p", ""))
                return self.send(200, {"text": log_tail(p)})
            if u.path == "/api/csv":
                col = q.get("collapse"); sl = q.get("slice")
                res = read_any(resolve(q.get("p", "")), int(q.get("stat", 0) or 0),
                               int(col) if col not in (None, "") else None, int(sl) if sl not in (None, "") else None,
                               int(q.get("scan", 0) or 0))
                if not res.get("measured"): res["beam"] = producer_of(relp(resolve(q.get("p", ""))))
                return self.send(200, res)
            if u.path.startswith("/api/beam/"): return self.send(200, beam_api(u.path[len("/api/beam/"):], q))
            if u.path == "/api/generators": return self.send(200, gen_schema())
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
        writes = u.path in ("/api/write", "/api/mkdir", "/api/newfile", "/api/move", "/api/copy", "/api/trash", "/api/generate", "/api/variants")
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
                ids, n = set(b.get("ids", [])), 0
                with STATE["lock"]:
                    for j in STATE["jobs"]:
                        if j["status"] == "queued" and (b.get("all") or j["id"] in ids): j["status"] = "removed"; n += 1
                    save_jobs()
                return self.send(200, {"ok": True, "removed": n})
            if u.path == "/api/outputs":
                out = {}
                for f in b.get("files", [])[:2000]:
                    try: out[f] = case_outputs(f)
                    except (OSError, ValueError): out[f] = []
                return self.send(200, {"outputs": out})
            if u.path == "/api/clearhistory":
                with STATE["lock"]:
                    STATE["jobs"] = [j for j in STATE["jobs"] if j["status"] in ("queued", "running")]; save_jobs()
                return self.send(200, {"ok": True})
            if u.path == "/api/config":
                cfg = STATE["cfg"]
                if "lang" in b: cfg["lang"] = b["lang"] if b["lang"] in ("de", "en") else "en"
                if "keep_awake" in b: cfg["keep_awake"] = bool(b["keep_awake"])
                if "keep_display" in b: cfg["keep_display"] = bool(b["keep_display"])
                if "toast" in b: cfg["toast"] = bool(b["toast"])
                if "g4_filter" in b: cfg["g4_filter"] = bool(b["g4_filter"])
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
        STATE["closing"] = True   # set before the job is stopped, so run_job leaves its record untouched
        stop_current()
    finally:
        remove_server_file()
        AWAKE.stop()


if __name__ == "__main__":
    main()
