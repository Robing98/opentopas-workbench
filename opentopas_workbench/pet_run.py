"""The PET part of the tab "LETd/PET on MRI": start tools/mrt_overlay/run_pet.sh from one form, then build_page.py
from the clone of the Timeline Viewer (setting viewer_dir), as the LETd part does.

Same pattern as letd.py, whose path helpers and process handling it uses: absolute paths outside the workspace, one
script, the log and results in the output folder the user names. The one file it serves is the control sheet the
run wrote, by the path in the job's own record, never by a path the page sends.

Display only, no reading of the images: the layer is a ratio to a reference region, no SUV.
"""
import glob
import json
import os
import threading
import time

from . import letd
from .letd import to_local, in_public_zone, public_ok

SCRIPT = os.path.join(os.path.dirname(letd.TOOL_DIR), "tools", "mrt_overlay", "run_pet.sh")
LOG_NAME = "pet_lauf.log"
STEPS = ("check", "read", "register", "headfoot", "side", "write", "page")
FIELDS = ("mrt", "pet", "cache", "view", "ref", "side", "labels", "out", "build")
REFS = ("brain", "striatum", "contralateral")
SIDES = ("links", "rechts")

JOB = {"proc": None, "thread": None, "stop": False, "state": None}


def default_out(root, today=None):
    """PET results computed from Robin's own images with public tools belong in the zone eigen; without that folder
    they go to lokal, which is closed to sessions as well."""
    base = root if os.path.isdir(os.path.join(root, "eigen")) or os.path.isdir(os.path.join(root, "lokal")) \
        else os.path.join(root, "workspace")
    day = today or time.strftime("%Y-%m-%d")
    for zone in ("eigen", "lokal"):
        if os.path.isdir(os.path.join(base, zone)): return {"out": os.path.join(base, zone, "pet_mrt", day)}
    return {}


def _has_files(d):
    try: return any(os.path.isfile(os.path.join(r, n)) for r, _, ns in os.walk(d) for n in ns)
    except OSError: return False


def check(form, root=None, viewer=None):
    """(problems, notes), as letd.check: a problem names the field and stops the start."""
    f = {k: str(form.get(k, "") or "") for k in FIELDS}
    p = {k: to_local(f[k]) for k in ("pet", "cache", "view", "labels", "out")}
    bad, notes = [], []
    need = lambda k, msg, *a: bad.append({"field": k, "msg": msg, "args": list(a)})
    if not p["pet"]: need("pet", "empty")
    elif not os.path.isdir(p["pet"]): need("pet", "not_found")
    elif not _has_files(p["pet"]): need("pet", "no_dicom")
    if not p["cache"]: need("cache", "empty")
    elif not os.path.isdir(p["cache"]): need("cache", "not_found")
    else:
        miss = [n for n in ("ref_t1.nii.gz", "log.json", "mask.npy") if not os.path.isfile(os.path.join(p["cache"], n))]
        if miss: need("cache", "missing", ", ".join(miss))
        elif not glob.glob(os.path.join(glob.escape(p["cache"]), "t1_*.npy")): need("cache", "no_t1")
    if not p["view"]: need("view", "empty")
    elif not os.path.isdir(p["view"]): need("view", "not_found")
    elif in_public_zone(p["view"], root) and not public_ok(form): need("view", "public_zone")
    ref = f["ref"] or "brain"
    if ref not in REFS: need("ref", "choice")
    if ref == "striatum":
        if not p["labels"]: need("labels", "empty")
        elif not os.path.isfile(p["labels"]): need("labels", "not_found")
    if ref == "contralateral" and f["side"] not in SIDES: need("side", "choice")
    if not p["out"]: need("out", "empty")
    elif in_public_zone(p["out"], root) and not public_ok(form): need("out", "public_zone")
    if f["build"] in ("1", "true", "True"):
        # named before the run, as in letd.check: the page needs the case files beside the form's folders
        script = letd.page_script(viewer)
        if not os.path.isfile(script): need("build", "no_viewer", script)
        elif p["cache"] and p["view"]:
            miss = [n for n in letd.page_inputs(p["cache"], p["view"]) if not os.path.isfile(n)]
            if miss: need("build", "no_case", ", ".join(miss))
    if not os.path.isfile(SCRIPT): need("", "no_script", SCRIPT)
    if p["view"] and os.path.isfile(os.path.join(p["view"], "pet.json")):
        notes.append({"field": "view", "msg": "pet_adds"})
    return bad, notes


def progress(lines, build):
    """Step states from the "## STEP" lines, and [done, total] from the last "## START i/n" line."""
    seen, start = [], None
    for line in lines:
        if line.startswith("## STEP "):
            s = line.split()[2]
            if s in STEPS: seen.append(s)
        elif line.startswith("## START "):
            try: start = [int(x) for x in line.split()[2].split("/")]
            except (IndexError, ValueError): pass
    steps = {s: "todo" for s in STEPS}
    steps["check"] = "done"
    if not build: steps["page"] = "skip"
    order = [s for s in STEPS if steps[s] != "skip"]
    if seen:
        cur = order.index(seen[-1])
        for s in order[:cur]: steps[s] = "done"
        steps[seen[-1]] = "now"
    return steps, start


def result(out, since=None):
    """The newest pet_<day>.json of OUT written by this run (not older than `since`): date, tracer, registration with
    the MI as read, upside down and mirrored, and whether the control sheet is there."""
    files = [p for p in glob.glob(os.path.join(glob.escape(out), "pet_????-??-??.json"))
             if since is None or os.path.getmtime(p) >= since - 2]
    if not files: return None
    path = max(files, key=os.path.getmtime)
    try:
        with open(path, encoding="utf-8") as fh: e = json.load(fh)
    except (OSError, ValueError):
        return None
    r = {k: e.get(k) for k in ("day", "tracer", "reference", "fixed_t1", "registration", "units", "hinweis")}
    sheet = e.get("sheet") or ""
    r["sheet"] = bool(sheet) and os.path.isfile(sheet)
    return r, (sheet if r["sheet"] else "")


def status():
    with letd.LOCK:
        st = dict(JOB["state"] or {})
        running = JOB["proc"] is not None or (JOB["thread"] is not None and JOB["thread"].is_alive())
    if not st: return {"running": False}
    log = os.path.join(st["out_local"], LOG_NAME)
    steps, start = progress(letd._all_lines(log), st["build"])
    if st.get("status") == "done":
        steps = {k: ("skip" if v == "skip" else "done") for k, v in steps.items()}
    st.update(running=running, steps=steps, start_n=start, log=letd._tail(log))
    if not running:
        got = result(st["out_local"], st.get("t0"))
        st["result"] = got[0] if got else None
    st.pop("sheet", None)
    return st


def sheet_path():
    """The control sheet of the last run, or "" when there is none. Only a PNG inside that run's output folder."""
    st = JOB["state"] or {}
    if not st.get("out_local") or st.get("status") == "running": return ""
    got = result(st["out_local"], st.get("t0"))
    p = got[1] if got else ""
    out = os.path.realpath(st["out_local"])
    if not p or not p.lower().endswith(".png") or not os.path.realpath(p).startswith(out + os.sep): return ""
    return p


def start(form, on_end=None, viewer=None):
    """Checks the form and starts run_pet.sh, then build_page.py. ValueError with the problems when the form does
    not pass, RuntimeError while a PET run is going."""
    bad, _ = check(form, viewer=viewer)
    if bad: raise ValueError(json.dumps(bad))
    with letd.LOCK:
        if JOB["proc"] is not None or (JOB["thread"] is not None and JOB["thread"].is_alive()):
            raise RuntimeError("running")
        f = {k: str(form.get(k, "") or "") for k in FIELDS}
        build = f["build"] in ("1", "true", "True")
        out = to_local(f["out"])
        os.makedirs(out, exist_ok=True)
        JOB["stop"] = False
        JOB["state"] = {"status": "running", "start": time.strftime("%Y-%m-%d %H:%M:%S"), "t0": time.time(),
                        "out": f["out"], "out_local": out, "build": build,
                        "page_local": letd.page_script(viewer) if build else "",
                        "cache_local": to_local(f["cache"]), "view_local": to_local(f["view"])}
        ref = f["ref"] or "brain"
        args = [to_local(f["pet"]), to_local(f["cache"]), to_local(f["view"]), out, ref]
        if ref == "contralateral": args.append(f["side"])
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if ref == "striatum": env["LABELS"] = to_local(f["labels"])
        t = threading.Thread(target=_run, args=(["bash", SCRIPT] + args, env, on_end), daemon=True)
        JOB["thread"] = t
    t.start()
    return status()


def _run(cmd, env, on_end):
    st = JOB["state"]
    log_path = os.path.join(st["out_local"], LOG_NAME)
    try:
        with open(log_path, "w") as log:
            log.write(f"# {st['start']}  {' '.join(letd.shlex.quote(c) for c in cmd)}\n"); log.flush()
            rc = letd._spawn(cmd, log, env, job=JOB) if not JOB["stop"] else -letd.signal.SIGTERM
            if rc == 0 and st["build"] and not JOB["stop"]:
                page = st["page_local"]
                log.write("## STEP page\n"); log.flush()
                pc = letd.page_command(page, ("--cache", st["cache_local"], "--view", st["view_local"]))
                log.write(f"# {' '.join(letd.shlex.quote(c) for c in pc)}\n"); log.flush()
                rc = letd._spawn(pc, log, env, cwd=os.path.dirname(page), job=JOB)
                if rc == 0: st["html"] = letd.find_page_html(st["view_local"])
        st["rc"] = rc
        st["status"] = "stopped" if JOB["stop"] else ("done" if rc == 0 else "failed")
    except Exception as e:  # the record says what went wrong; the page shows it
        st["status"] = "failed"; st["error"] = str(e)
    finally:
        st["end"] = time.strftime("%Y-%m-%d %H:%M:%S"); st["seconds"] = round(time.time() - st["t0"])
        if on_end:
            try: on_end(st)
            except Exception: pass


def stop():
    return letd.stop(JOB)


def restore(last):
    if not last or JOB["state"] is not None: return
    st = dict(last)
    if st.get("status") == "running":
        st["status"] = "stopped"; st["note"] = "server_restart"; st.setdefault("seconds", None)
    with letd.LOCK: JOB["state"] = st


def open_html():
    return letd.open_html(JOB)
