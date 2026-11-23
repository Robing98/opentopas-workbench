"""The tab "LETd on MRI": start tools/mrt_overlay/run.sh from one form instead of a start file that asks for ten paths.

The page itself is built by the Timeline Viewer, a repository of its own: the tab runs build_page.py from its clone
(setting viewer_dir).

The paths point outside the workspace (the plan_mc run, the planning CT, the Timeline Viewer's folders), because the data
they hold must not lie in the workspace's public part. So this module takes absolute paths, never serves a file
from them, and only runs the one script. The run's log and results go to the output folder the user names, not to
the workspace's .workbench folder: the log lists exams and numbers derived from the patient's data.

Research only, no diagnosis: the layer is for looking, the clinical RBE stays 1.1.
"""
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(TOOL_DIR), "tools", "mrt_overlay", "run.sh")
VIEWER_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(TOOL_DIR)), "timeline-viewer")   # a clone next to this one
LOG_NAME = "letd_lauf.log"
STEPS = ("check", "export", "synthseg", "register", "exams", "write", "page")
FIELDS = ("mrt", "run", "ct", "rs", "cache", "view", "rx", "fx", "out", "kz", "build", "synthseg")
SEARCH_DEPTH = 4
SEARCH_LIMIT = 20000          # folders visited at most when deriving from the MRI folder
SYNTHSEG_GB = 10              # memory that mri_synthseg --robust --crop 192 needs (9.5 GB measured)

JOB = {"proc": None, "thread": None, "stop": False, "state": None}
LOCK = threading.Lock()


# ------------------------------------------------------------------ paths
def to_local(p):
    """A path as typed: Explorer's "copy as path" adds quotes, and under WSL a Windows path (C:\\...) becomes
    /mnt/c/... . Done here rather than by wslpath so that a test or a plain Linux sees the same result."""
    p = (p or "").strip().strip('"').strip("'").strip()
    if not p: return ""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m and os.path.isdir("/mnt"):
        p = f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")
    # absolute here: run.sh changes into tools/, so a relative path would be checked in one place and written in another
    return os.path.abspath(os.path.expanduser(p))


def to_windows(p):
    """/mnt/c/x -> C:\\x for a Windows program; other paths stay."""
    m = re.match(r"^/mnt/([a-z])/(.*)$", p)
    return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\") if m else p


def in_public_zone(p, root=None):
    """True for a path in workspace/offen, or in offen of the workbench's own root: results from a patient's plan and
    MRIs must not land in the public part. Links are followed, so a link into offen counts as offen."""
    real = os.path.realpath(p)
    parts = [s.lower() for s in real.split(os.sep) if s]
    if any(a == "workspace" and b == "offen" for a, b in zip(parts, parts[1:])): return True
    if root:
        r = os.path.realpath(root)
        for pub in (os.path.join(r, "offen"), os.path.join(r, "workspace", "offen")):
            if os.path.normcase(real) == os.path.normcase(pub) or os.path.normcase(real).startswith(os.path.normcase(pub) + os.sep):
                return True
    return False


def public_ok(form):
    """A case marked public (the demo case, plan.oeffentlich in fall.json) may write into offen: its plan and images
    are synthetic. Set only by the case start of fall_setup, never by the form."""
    return form.get("oeffentlich") is True


def _walk(top, depth=SEARCH_DEPTH, limit=SEARCH_LIMIT):
    """(folder, subfolder names, file names) breadth first, shallow folders first, so the nearest match wins."""
    queue, seen = [(top, 0)], 0
    while queue and seen < limit:
        d, lvl = queue.pop(0); seen += 1
        try: entries = list(os.scandir(d))
        except OSError: continue
        dirs = sorted(e.name for e in entries if e.is_dir(follow_symlinks=False) and not e.name.startswith((".", "_")))
        files = {e.name for e in entries if e.is_file()}
        yield d, dirs, files
        if lvl < depth: queue.extend((os.path.join(d, n), lvl + 1) for n in dirs)


def derive(mrt):
    """From the MRI folder of the viewer: the cacheF folder (log.json and ref_t1.nii.gz) and the data folder of the
    viewer (Verlauf-Daten). Each found value is a suggestion the form shows and the user may change. build_page.py is
    not searched: an old copy next to the images must not win over the viewer's repository."""
    top = to_local(mrt)
    if not top or not os.path.isdir(top): return {"error": "not_found"}
    found = {}
    for d, dirs, files in _walk(top):
        if "cache" not in found and {"log.json", "ref_t1.nii.gz"} <= files: found["cache"] = d
        if "view" not in found and os.path.basename(d).lower() == "verlauf-daten": found["view"] = d
        if len(found) == 2: break
    return found


def default_out(root, today=None):
    """Output and numbers into the private zones of the workspace when it has them: results computed with the
    treatment plan are MIT data (lokal), kennzahlen.json is Robin's (robin)."""
    out = {}
    base = root if os.path.isdir(os.path.join(root, "lokal")) else os.path.join(root, "workspace")
    if os.path.isdir(os.path.join(base, "lokal")):
        out["out"] = os.path.join(base, "lokal", "letd_mrt", today or time.strftime("%Y-%m-%d"))
    if os.path.isdir(os.path.join(base, "robin")):
        out["kz"] = os.path.join(base, "robin", "letd_mrt", "kennzahlen.json")
    return out


# ------------------------------------------------------------------ checks
def _num(v, kind):
    try:
        x = kind(str(v).replace(",", ".").strip())
        return x if x > 0 else None
    except ValueError:
        return None


def page_script(viewer=None):
    """build_page.py in the clone of the viewer's repository."""
    return os.path.join(to_local(viewer) or VIEWER_DIR_DEFAULT, "build_page.py")


def page_inputs(cache, view):
    """What build_page.py needs beside the form's folders, at its default places."""
    case, work = os.path.join(os.path.dirname(view), "Skripte"), os.path.join(os.path.dirname(cache), "cacheG")
    return [os.path.join(case, "viewer.json"), os.path.join(case, "befunde.json"),
            os.path.join(work, "log.json"), os.path.join(work, "vent.json")]


def check(form, root=None, viewer=None):
    """(problems, notes) for a form. A problem stops the start and names the field; a note does not."""
    f = {k: str(form.get(k, "") or "") for k in FIELDS}
    p = {k: to_local(f[k]) for k in ("run", "ct", "rs", "cache", "view", "out", "kz")}
    bad, notes = [], []
    need = lambda k, msg, *a: bad.append({"field": k, "msg": msg, "args": list(a)})
    if not p["run"]: need("run", "empty")
    elif not os.path.isdir(p["run"]): need("run", "not_found")
    else:
        miss = [n for n in ("letd.npy", "rbe_dose_1_1.npy", "rbe_dose_1_1.dcm") if not os.path.isfile(os.path.join(p["run"], n))]
        if miss: need("run", "missing", ", ".join(miss))
    if not p["ct"]: need("ct", "empty")
    elif os.path.isdir(p["ct"]):
        try: names = os.listdir(p["ct"])
        except OSError: names = None
        if names is None: need("ct", "not_found")
        elif not any(n.lower().endswith(".dcm") or "." not in n for n in names): need("ct", "no_dicom")
    elif not (os.path.isfile(p["ct"]) and p["ct"].lower().endswith((".nii", ".nii.gz", ".nrrd", ".mha"))): need("ct", "not_found")
    if p["rs"] and not os.path.isfile(p["rs"]): need("rs", "not_found")
    if not p["cache"]: need("cache", "empty")
    elif not os.path.isdir(p["cache"]): need("cache", "not_found")
    else:
        miss = [n for n in ("ref_t1.nii.gz", "log.json") if not os.path.isfile(os.path.join(p["cache"], n))]
        if miss: need("cache", "missing", ", ".join(miss))
    if not p["view"]: need("view", "empty")
    elif not os.path.isdir(p["view"]): need("view", "not_found")
    elif in_public_zone(p["view"], root) and not public_ok(form): need("view", "public_zone")
    rx, fx = _num(f["rx"], float), _num(f["fx"], int)
    if f["rx"].strip() and rx is None: need("rx", "number")
    if not f["rx"].strip() and not p["rs"]: need("rx", "rx_or_rs")
    if fx is None: need("fx", "number")
    for k in ("out", "kz"):
        if not p[k]: need(k, "empty")
        elif in_public_zone(p[k], root) and not public_ok(form): need(k, "public_zone")
    if p["kz"] and not p["kz"].lower().endswith(".json"): need("kz", "json")
    if f["build"] in ("1", "true", "True"):
        # named here: otherwise the page fails only after the whole run, which takes up to an hour
        script = page_script(viewer)
        if not os.path.isfile(script): need("build", "no_viewer", script)
        elif p["cache"] and p["view"]:
            miss = [n for n in page_inputs(p["cache"], p["view"]) if not os.path.isfile(n)]
            if miss: need("build", "no_case", ", ".join(miss))
    if not os.path.isfile(SCRIPT): need("", "no_script", SCRIPT)
    if f["synthseg"] in ("1", "true", "True"):
        if not synthseg_found(): notes.append({"field": "synthseg", "msg": "synthseg_missing"})
        gb = memory_gb()
        if gb and gb < SYNTHSEG_GB: notes.append({"field": "synthseg", "msg": "synthseg_memory", "args": [round(gb, 1), SYNTHSEG_GB]})
    if p["out"] and os.path.isdir(p["out"]) and os.path.exists(os.path.join(p["out"], "viewer_grid", "overlay.json")):
        notes.append({"field": "out", "msg": "out_exists"})
    return bad, notes


def synthseg_found():
    fs = os.path.expanduser("~/opt/freesurfer-7.4.1/freesurfer/bin/mri_synthseg")
    env = os.environ.get("SYNTHSEG", "")
    if env == "none": return False
    return bool(env or shutil.which("mri_synthseg") or os.path.exists(fs))


def memory_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"): return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


# ------------------------------------------------------------------ the run
def page_command(page, args=()):
    """How build_page.py runs: under WSL with the Windows Python launcher when there is one and it can reach every
    path, since the viewer's scripts are run on Windows otherwise; else with this Python."""
    py = shutil.which("py.exe")
    if py and all(re.match(r"^/mnt/[a-z]/", a) for a in (page, *args) if a.startswith("/")):
        return [py, "-3", to_windows(page), *(to_windows(a) for a in args)]
    return [shutil.which("python3") or "python3", page, *args]


def _progress(lines, build, synthseg):
    """Step states from the "## STEP" and "## EXAM" lines of the log: {step: "done"|"now"|"todo"|"skip"}, exams i/n."""
    seen, exam = [], None
    for line in lines:
        m = re.match(r"^## STEP (\w+)", line)
        if m: seen.append(m.group(1)); continue
        m = re.match(r"^## EXAM (\d+)/(\d+)", line)
        if m:
            exam = [int(m.group(1)), int(m.group(2))]
            if "exams" not in seen: seen.append("exams")
    steps = {s: "todo" for s in STEPS}
    steps["check"] = "done"
    if not synthseg: steps["synthseg"] = "skip"
    if not build: steps["page"] = "skip"
    for s in seen[:-1]: steps[s] = "done"
    if seen: steps[seen[-1]] = "now"
    order = [s for s in STEPS if steps[s] != "skip"]
    if seen:   # a skipped or silent step before the current one is done (SynthSeg found no command, say)
        cur = order.index(seen[-1]) if seen[-1] in order else 0
        for s in order[:cur]:
            if steps[s] == "todo": steps[s] = "done"
    return steps, exam


def _tail(path, size=12000):
    try:
        with open(path, errors="replace") as fh:
            fh.seek(0, 2); n = fh.tell(); fh.seek(max(0, n - size)); return fh.read()
    except OSError:
        return ""


def _all_lines(path):
    try:
        with open(path, errors="replace") as fh: return [l.rstrip("\n") for l in fh if l.startswith("## ")]
    except OSError:
        return []


def result(out):
    """What the run found, from OUT/viewer_grid/overlay.json: registration, exams with their check, the look-list."""
    path = os.path.join(out, "viewer_grid", "overlay.json")
    try:
        with open(path, encoding="utf-8") as fh: info = json.load(fh)
    except (OSError, ValueError):
        return None
    return {k: info.get(k) for k in ("registration", "exams", "look_list", "prescription_gy_rbe", "prescription_from",
                                     "ptv", "margin", "hinweis")}


def find_page_html(start):
    """Verlauf.html next to the viewer's data folder (`start`) or up to two folders above, the newest one."""
    if not start: return ""
    d, best = os.path.dirname(start), None
    for _ in range(3):
        try: names = os.listdir(d)
        except OSError: names = []
        for n in names:
            if n.lower() == "verlauf.html":
                fp = os.path.join(d, n)
                if best is None or os.path.getmtime(fp) > os.path.getmtime(best): best = fp
        d = os.path.dirname(d)
    return best or ""


def status():
    with LOCK:
        st = dict(JOB["state"] or {})
        running = JOB["proc"] is not None or (JOB["thread"] is not None and JOB["thread"].is_alive())
    if not st: return {"running": False}
    log = os.path.join(st["out_local"], LOG_NAME)
    steps, exam = _progress(_all_lines(log), st["build"], st["synthseg"])
    if st.get("status") == "done":
        steps = {k: ("skip" if v == "skip" else "done") for k, v in steps.items()}
    st.update(running=running, steps=steps, exam=exam, log=_tail(log))
    if not running: st["result"] = result(st["out_local"])
    return st


def start(form, on_end=None, viewer=None):
    """Checks the form and starts run.sh, then build_page.py. Raises ValueError with the problems when the form
    does not pass, RuntimeError when a run is going."""
    bad, _ = check(form, viewer=viewer)
    if bad: raise ValueError(json.dumps(bad))
    with LOCK:
        if JOB["proc"] is not None or (JOB["thread"] is not None and JOB["thread"].is_alive()):
            raise RuntimeError("running")
        f = {k: str(form.get(k, "") or "") for k in FIELDS}
        build, synthseg = f["build"] in ("1", "true", "True"), f["synthseg"] in ("1", "true", "True")
        out = to_local(f["out"])
        os.makedirs(out, exist_ok=True)
        JOB["stop"] = False
        JOB["state"] = {"status": "running", "start": time.strftime("%Y-%m-%d %H:%M:%S"), "t0": time.time(),
                        "out": f["out"], "out_local": out, "build": build, "synthseg": synthseg,
                        "page_local": page_script(viewer) if build else "",
                        "cache_local": to_local(f["cache"]), "view_local": to_local(f["view"])}
        args = [to_local(f[k]) for k in ("run", "ct", "rs", "cache", "view")]
        args += [str(_num(f["rx"], float) or ""), str(_num(f["fx"], int)), out, to_local(f["kz"])]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if not synthseg: env["SYNTHSEG"] = "none"
        # a start from a case (fall_setup.letd_form) names its folder: the cli writes its results back into fall.json
        if form.get("fall"): env["FALL"] = to_local(str(form["fall"]))
        t = threading.Thread(target=_run, args=(["bash", SCRIPT] + args, env, on_end), daemon=True)
        JOB["thread"] = t
    t.start()
    return status()


def _spawn(cmd, log, env, cwd=None, job=None):
    """Runs one process of a job and waits. `job` is JOB here; the PET run (pet_run.py) passes its own."""
    job = JOB if job is None else job
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True,
                            stdin=subprocess.DEVNULL)
    with LOCK:
        job["proc"] = proc; stopped = job["stop"]
    # a Stop between the start and this point found no process to end: end it here
    if stopped: _kill(proc)
    rc = proc.wait()
    with LOCK: job["proc"] = None
    return rc


def _run(cmd, env, on_end):
    st = JOB["state"]
    log_path = os.path.join(st["out_local"], LOG_NAME)
    try:
        with open(log_path, "w") as log:
            log.write(f"# {st['start']}  {' '.join(shlex.quote(c) for c in cmd)}\n"); log.flush()
            rc = _spawn(cmd, log, env) if not JOB["stop"] else -signal.SIGTERM
            if rc == 0 and st["build"] and not JOB["stop"]:
                page = st["page_local"]
                log.write("## STEP page\n"); log.flush()
                pc = page_command(page, ("--cache", st["cache_local"], "--view", st["view_local"]))
                log.write(f"# {' '.join(shlex.quote(c) for c in pc)}\n"); log.flush()
                rc = _spawn(pc, log, env, cwd=os.path.dirname(page))
                if rc == 0: st["html"] = find_page_html(st["view_local"])
        st["rc"] = rc
        st["status"] = "stopped" if JOB["stop"] else ("done" if rc == 0 else "failed")
    except Exception as e:  # the record says what went wrong; the page shows it
        st["status"] = "failed"; st["error"] = str(e)
    finally:
        st["end"] = time.strftime("%Y-%m-%d %H:%M:%S"); st["seconds"] = round(time.time() - st["t0"])
        if on_end:
            try: on_end(st)
            except Exception: pass


def _kill(proc):
    try: os.killpg(proc.pid, signal.SIGTERM)   # run.sh starts Python and SynthSeg: the whole group must end
    except OSError: pass


def stop(job=None):
    """Ends the run. True while a run is going, even between two of its processes: _spawn sees the flag."""
    job = JOB if job is None else job
    with LOCK:
        proc, t = job["proc"], job["thread"]
        running = t is not None and t.is_alive()
        if running: job["stop"] = True
    if proc is not None: _kill(proc)
    return running


def restore(last):
    """The record of the last run after a server start, so the tab shows its result and log again."""
    if not last or JOB["state"] is not None: return
    st = dict(last)
    if st.get("status") == "running":
        st["status"] = "stopped"; st["note"] = "server_restart"
        st.setdefault("seconds", None)   # unknown: else the page counts the time since the start for ever
    with LOCK: JOB["state"] = st


def open_html(job=None):
    """Opens the Verlauf.html of the last run in the default browser; under WSL the Windows one, which cannot read a
    WSL path. Only that file: the page does not choose what is opened. explorer.exe, not cmd /c start, because cmd
    reads & ^ % in a folder name as commands."""
    st = (JOB if job is None else job)["state"] or {}
    p = st.get("html") or ""
    if not p or not os.path.isfile(p) or not p.lower().endswith(".html"): raise ValueError("not_found")
    if shutil.which("explorer.exe") and p.startswith("/mnt/"):
        subprocess.Popen(["explorer.exe", to_windows(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="/mnt/c")
    else:
        import webbrowser
        webbrowser.open("file://" + p)
    return True
