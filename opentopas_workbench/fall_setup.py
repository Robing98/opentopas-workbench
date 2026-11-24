"""Set up a case once instead of typing ten paths per run: the search behind the assistant of the tab "LETd/PET on MRI",
the demo case, and the step from a case to the forms that letd.py and pet_run.py already start.

The search runs as code of the workbench server on the user's computer when the user starts it in the web UI. It reads
folder names and DICOM headers (Modality, dates, from the RT plan the fractions and the prescription), never pixel
data, and it returns paths and those few facts to the page, nothing else.
"""
import gzip
import hashlib
import json
import os
import re
import shlex
import struct
import threading
import time

from . import dicomkopf, fall, letd, pet_run

DEPTH = 6
LIMIT = 40000                    # folders visited at most per search root
RUN_FILES = ("letd.npy", "rbe_dose_1_1.npy", "rbe_dose_1_1.dcm")
RUN_INPUTS = ("ct", "rtstruct", "rtplan")


# ------------------------------------------------------------------ search
# folders no search enters: system and program folders, synced clouds (cacheF lies outside OneDrive by design), caches
SKIP = re.compile(r"^(onedrive.*|appdata|windows|program files.*|programdata|\$recycle\.bin|system volume information|"
                  r"recovery|node_modules|site-packages|__pycache__|\.git|wsl.*|perflogs)$", re.I)
CACHE_DEPTH = 4
CACHE_LIMIT = 30000              # folders visited at most by one cacheF search over all its roots
CACHE_SECONDS = 60
EXAM = re.compile(r"^\S+ \d{4}-\d{2}-\d{2}$")   # a series of one exam in log.json: "t1 2020-01-01"


def _walk(top, depth=DEPTH, limit=LIMIT):
    yield from letd._walk(top, depth, limit)


class Unreadable:
    """Files a search found but could not read. A file that OneDrive keeps only in the cloud is a placeholder: under
    WSL it lists like any file and fails or comes back empty when read. The assistant names them instead of
    reporting "nothing found"."""
    def __init__(self): self.n, self.folders = 0, []

    def __call__(self, path):
        self.n += 1
        d = os.path.dirname(path)
        if d not in self.folders and len(self.folders) < 5: self.folders.append(d)

    def __enter__(self):
        self.prev, dicomkopf.ON_ERROR = dicomkopf.ON_ERROR, self
        return self

    def __exit__(self, *a): dicomkopf.ON_ERROR = self.prev

    def report(self):
        return {"anzahl": self.n, "ordner": self.folders} if self.n else None


def _dicom_head(d, files):
    """The header of the first readable DICOM file in a folder, or None. Two tries at most: a folder of 300 slices
    must not cost 300 reads."""
    tried = 0
    for n in sorted(files):
        if not (n.lower().endswith(".dcm") or "." not in n): continue
        h = dicomkopf.read(os.path.join(d, n), limit=64 << 10)
        if h: return h
        tried += 1
        if tried >= 2: break
    return None


def viewer_root(start):
    """The folder of the follow-up viewer that holds `start`: the nearest folder at or above it with Analyse/Verlauf.html
    or Analyse/Verlauf-Daten, else the nearest with a bare Verlauf.html. The user may pick an exam folder deep inside,
    or Analyse itself; the search then still sees the whole viewer. None when no folder above has one."""
    def up(test):
        d = start
        for _ in range(12):
            if test(d): return d
            parent = os.path.dirname(d)
            if parent == d: return None
            d = parent
        return None
    # Analyse holds a bare Verlauf.html too: the Analyse/... forms must win over all levels, else picking Analyse or a
    # folder in it stops one level short of the viewer
    return (up(lambda d: any(os.path.exists(os.path.join(d, "Analyse", n)) for n in ("Verlauf.html", "Verlauf-Daten")))
            or up(lambda d: os.path.exists(os.path.join(d, "Verlauf.html"))))


def find_viewer(mrt):
    """From any folder of the viewer: Verlauf.html (the newest), Verlauf-Daten, Skripte beside it, and cacheF when it
    lies inside. The search goes up to the viewer's own folder first."""
    top = letd.to_local(mrt)
    if not top or not os.path.isdir(top): return {"fehler": "mrt_nicht_gefunden"}
    top = viewer_root(top) or top
    found, pages = {}, []
    for d, dirs, files in _walk(top, 5):
        for n in files:
            if n.lower() == "verlauf.html": pages.append(os.path.join(d, n))
        base = os.path.basename(d).lower()
        if "daten" not in found and base == "verlauf-daten":
            found["daten"] = d
            sk = os.path.join(os.path.dirname(d), "Skripte")
            if os.path.isdir(sk): found["skripte"] = sk
        if "cache" not in found and _cache_log(d, files) is not None: found["cache"] = d
    if pages: found["seite"] = max(pages, key=os.path.getmtime)
    found["bilder"] = top
    return found


def _cache_log(d, files):
    """The exams a cacheF folder holds (keys of its log.json), or None when `d` is no cacheF: log.json with size and
    spacing, in a folder named cacheF or beside ref_t1.nii.gz (cacheG has a log.json too)."""
    if "log.json" not in files: return None
    if os.path.basename(d).lower() != "cachef" and "ref_t1.nii.gz" not in files: return None
    try:
        with open(os.path.join(d, "log.json"), encoding="utf-8") as fh: log = json.load(fh)
    except OSError:
        if dicomkopf.ON_ERROR: dicomkopf.ON_ERROR(os.path.join(d, "log.json"))
        return None
    except ValueError:
        return None
    if not isinstance(log, dict) or "size" not in log or "spacing" not in log: return None
    return {k for k in log if EXAM.match(k)}


def stand_exams(skripte):
    """The exams of the viewer's own copy of the cacheF log (Skripte/stand/cacheF_log.json): what the right cacheF
    must hold. Empty when the viewer has no copy."""
    try:
        with open(os.path.join(skripte, "stand", "cacheF_log.json"), encoding="utf-8") as fh: log = json.load(fh)
        return {k for k in log if EXAM.match(k)} if isinstance(log, dict) else set()
    except (OSError, ValueError, TypeError):
        return set()


def cache_roots_default(mnt="/mnt", home=None):
    """Where cacheF may lie on this computer, nearest first: the WSL home, C:\\dev, each Windows user folder (its
    OneDrive is skipped), C:\\Uni, then every drive."""
    out = [home or os.path.expanduser("~")]
    c = os.path.join(mnt, "c")
    out.append(os.path.join(c, "dev"))
    users = os.path.join(c, "Users")
    try:
        out += sorted(e.path for e in os.scandir(users) if e.is_dir() and e.name not in ("Public", "Default", "Default User", "All Users"))
    except OSError:
        pass
    out.append(os.path.join(c, "Uni"))
    try:
        out += sorted(e.path for e in os.scandir(mnt) if len(e.name) == 1 and e.is_dir())
    except OSError:
        pass
    return [p for p in out if os.path.isdir(p)]


def find_cache(roots, want=(), depth=CACHE_DEPTH, limit=CACHE_LIMIT, seconds=CACHE_SECONDS):
    """The cacheF folder under the roots that holds the most exams of `want` (the viewer's copy of its log); without
    `want`, the newest cacheF. With `want`, a folder that holds none of its exams belongs to another viewer and is
    no candidate. Stops at the first folder holding every exam of `want`."""
    want, seen, best, end, n = set(want), set(), None, time.monotonic() + seconds, 0
    for r in roots:
        r = letd.to_local(r)
        if not r or not os.path.isdir(r): continue
        queue = [(r, 0)]
        while queue and n < limit and time.monotonic() < end:
            d, lvl = queue.pop(0)
            real = os.path.realpath(d)
            if real in seen: continue
            seen.add(real); n += 1
            try: entries = list(os.scandir(d))
            except OSError: continue
            files = {e.name for e in entries if e.is_file()}
            exams = _cache_log(d, files)
            if exams is not None:
                hits = len(exams & want)
                if hits or not want:
                    key = (hits, os.path.getmtime(os.path.join(d, "log.json")))
                    if best is None or key > best[0]: best = (key, d)
                    if want and want <= exams: return d
                continue
            if lvl < depth:
                queue.extend((e.path, lvl + 1) for e in entries
                             if e.is_dir(follow_symlinks=False) and not SKIP.match(e.name) and not e.name.startswith("."))
    return best[1] if best else None


def run_inputs(run):
    """CT, RT structure set and RT plan a plan_mc run used, from its run.json; for a run older than run.json, from
    chain.json when it names them. Missing ones stay out: complete_inputs looks for them beside the CT."""
    out = {}
    for name in ("run.json", "chain.json"):
        try:
            with open(os.path.join(run, name), encoding="utf-8") as fh: d = json.load(fh)
        except (OSError, ValueError):
            continue
        src = d.get("eingaben", d) if isinstance(d, dict) else {}
        for k in RUN_INPUTS:
            v = src.get(k)
            if k not in out and isinstance(v, str) and v:
                p = v if os.path.isabs(v) else os.path.join(run, v)
                if os.path.exists(p): out[k] = os.path.normpath(p)
    return out


MODALITY = {"ct": "CT", "rtstruct": "RTSTRUCT", "rtplan": "RTPLAN"}
EXPORT_FILES = 4000              # DICOM files read at most when looking through an export folder


def _dicom_files(top, depth=3, limit=EXPORT_FILES):
    """(path, header) of the DICOM files under `top` (a file: itself). An image series counts once, as its folder.
    Each file is read only as far as its Modality (8 KB); an RT file then in full, for its frame of reference."""
    if os.path.isfile(top):
        h = dicomkopf.read(top, limit=256 << 10)
        return [(top, h)] if h else []
    out, n, series = [], 0, set()
    for d, _, files in _walk(top, depth, 2000):
        for f in sorted(files):
            if not (f.lower().endswith(".dcm") or "." not in f): continue
            if n >= limit: return out
            p = os.path.join(d, f); n += 1
            h = dicomkopf.read(p, limit=8 << 10)
            if not h: continue
            if h.get("Modality") in ("CT", "MR", "PT"):
                uid = (d, h.get("SeriesInstanceUID"))
                if uid not in series:
                    series.add(uid); out.append((d, dicomkopf.read(p, limit=64 << 10) or h))
            elif h.get("Modality") in ("RTSTRUCT", "RTPLAN"):
                out.append((p, dicomkopf.read(p, limit=256 << 10) or h))
    return out


def _is(kind, path):
    """The header of `path` when it is the right thing for row `kind` (a CT folder, an RTSTRUCT or RTPLAN file)."""
    if not path: return None
    if kind == "ct":
        if not os.path.isdir(path): return None
        h = _dicom_head(path, os.listdir(path))
    else:
        h = dicomkopf.read(path, limit=256 << 10) if os.path.isfile(path) else None
    return h if h and h.get("Modality") == MODALITY[kind] else None


def complete_inputs(inp):
    """CT, RTSTRUCT and RTPLAN from what is known: a row may name the export folder instead of the file. A missing or
    folder row is looked up by DICOM Modality in the export folders of the known rows (the folder of the CT and the
    one above it), and only a file of the CT's frame of reference fits. Returns {row: (path, source)}."""
    out, heads = {}, {}
    for k in RUN_INPUTS:
        p = letd.to_local(inp.get(k)) if inp.get(k) else ""
        h = _is(k, p)
        if h: out[k], heads[k] = (p, "lauf"), h
    if len(out) == 3: return out
    places = []
    for k in RUN_INPUTS:
        p = letd.to_local(inp.get(k)) if inp.get(k) else ""
        if not p or not os.path.exists(p): continue
        base = p if os.path.isdir(p) and k != "ct" else os.path.dirname(p)
        for q in (base, os.path.dirname(base)):
            if q not in places: places.append(q)
    if not places: return out
    frame = next((dicomkopf.frame_uid(heads[k]) for k in RUN_INPUTS if k in heads and dicomkopf.frame_uid(heads[k])), None)
    found = {k: [] for k in RUN_INPUTS}
    for place in places:
        for p, h in _dicom_files(place):
            kind = next((k for k, m in MODALITY.items() if h.get("Modality") == m), None)
            if kind and kind not in out and (p, h) not in found[kind]: found[kind].append((p, h))
    if not frame:   # no known row gives the frame: one CT, or else one RT file alone, sets it
        for k in RUN_INPUTS:
            frames = {dicomkopf.frame_uid(h) for _, h in found[k]} - {None}
            if len(frames) == 1: frame = frames.pop(); break
    for k in RUN_INPUTS:
        if k in out: continue
        fit = [(p, h) for p, h in found[k] if frame and dicomkopf.frame_uid(h) == frame]
        if fit: out[k] = (max(fit, key=lambda x: os.path.getmtime(x[0]))[0], "export")
    return out


def find_runs(roots):
    """plan_mc run folders (letd.npy and its doses) under the roots, newest first, with the inputs each one names."""
    runs, seen = [], set()
    for r in roots:
        r = letd.to_local(r)
        if not r or not os.path.isdir(r): continue
        for d, _, files in _walk(r):
            if set(RUN_FILES) <= files and d not in seen:
                seen.add(d)
                runs.append({"ordner": d, "zeit": os.path.getmtime(os.path.join(d, "letd.npy")),
                             "eingaben": run_inputs(d)})
    runs.sort(key=lambda x: -x["zeit"])
    for x in runs: x["zeit"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(x["zeit"]))
    return runs


def find_pet(roots):
    """Folders of DICOM series with Modality PT, one entry per series, with its date, oldest first. A root may be
    (folder, depth)."""
    out, seen = [], set()
    for r in roots:
        r, depth = r if isinstance(r, tuple) else (r, DEPTH)
        r = letd.to_local(r)
        if not r or not os.path.isdir(r): continue
        for d, _, files in _walk(r, depth):
            h = _dicom_head(d, files)
            if not h or h.get("Modality") != "PT": continue
            uid = h.get("SeriesInstanceUID") or d
            if uid in seen: continue
            seen.add(uid)
            date = h.get("SeriesDate") or h.get("StudyDate") or ""
            if re.match(r"^\d{8}$", date): date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            out.append({"ordner": d, "datum": date or None, "beschreibung": h.get("SeriesDescription") or ""})
    out.sort(key=lambda x: x["datum"] or "")
    return out


def plan_facts(path):
    """Fractions and prescription from the RT plan; {} when the file is no readable RT plan."""
    h = dicomkopf.read(letd.to_local(path))
    return dicomkopf.plan_facts(h) or {}


def search_roots(root):
    """Where plan_mc runs are searched: the private zones of the workspace (lokal, eigen); a run of a patient's plan
    never lies in offen."""
    base = root if os.path.isdir(os.path.join(root, "lokal")) else os.path.join(root, "workspace")
    return [os.path.join(base, z) for z in ("lokal", "eigen") if os.path.isdir(os.path.join(base, z))]


# rows the LETd layer cannot start without; the prescription only when no RTSTRUCT gives the PTV. Every other row is
# optional: shown grey when nothing is found, never red.
REQUIRED = ("lauf", "ct", "cache", "daten", "fraktionen")


def _mark(rows):
    for k, r in rows.items():
        need = k in REQUIRED or (k == "verschreibung" and not (rows.get("rtstruct") or {}).get("wert"))
        r["pflicht"] = need
        if r.get("quelle") == "fehlt" and not need: r["quelle"] = "optional"
    return rows


def scan(mrt, root, cache_roots=(), pet_dir=None, extra_runs=(), default_roots=None):
    with Unreadable() as bad:
        r = _scan(mrt, root, cache_roots, pet_dir, extra_runs, default_roots)
    if "zeilen" in r and bad.report(): r["unlesbar"] = bad.report()
    return r


def _scan(mrt, root, cache_roots=(), pet_dir=None, extra_runs=(), default_roots=None):
    """The proposal of the assistant: one value per row, with candidates where several fit. Each row says where it
    came from (gefunden, gemerkt, lauf, export, plan, fehlt, optional), so the page can mark what the user must
    still choose. `default_roots` (tests) replaces the places of this computer where cacheF is looked for."""
    v = find_viewer(mrt) if mrt else {}
    if v.get("fehler"): return {"fehler": v["fehler"]}
    rows = {k: {"wert": v.get(k), "quelle": "gefunden" if v.get(k) else "fehlt"} for k in ("seite", "daten", "skripte", "cache")}
    rows["bilder"] = {"wert": v.get("bilder"), "quelle": "eingegeben"}
    if not rows["cache"]["wert"]:
        want = stand_exams(v["skripte"]) if v.get("skripte") else set()
        # without the viewer's copy of its log any cacheF of the computer could be taken: only a proposal to confirm
        quelle = "gefunden" if want else "vorschlag"
        mine = cachef_target(root, v.get("seite"))      # one made from this viewer's page earlier lies here
        held = _cache_log(mine, set(os.listdir(mine))) if mine and os.path.isdir(mine) else None
        # the viewer may have gained an exam since: an own cacheF that lacks one is made again, not kept
        if held is not None and want <= held:
            rows["cache"] = {"wert": mine, "quelle": "gemerkt"}
        else:
            c = find_cache(cache_roots, want, seconds=10) if cache_roots else None
            # a root named in the settings may hold several patients' cacheF: without `want` only a proposal
            if c: rows["cache"] = {"wert": c, "quelle": "gemerkt" if want else "vorschlag"}
            else:
                c = find_cache(cache_roots_default() if default_roots is None else default_roots, want)
                if c: rows["cache"] = {"wert": c, "quelle": quelle}
        src = cachef_sources({k: v.get(k) for k in ("seite", "daten", "skripte")})
        if not rows["cache"]["wert"] and src and mine:
            rows["cache"].update(erzeugbar=True, ziel=mine)
    runs = find_runs(list(search_roots(root)) + list(extra_runs))
    rows["lauf"] = {"wert": runs[0]["ordner"] if runs else None, "quelle": "gefunden" if runs else "fehlt",
                    "kandidaten": [{"wert": r["ordner"], "info": r["zeit"]} for r in runs[:20]]}
    rows.update(input_rows(runs[0]["eingaben"] if runs else {}))
    vb = v.get("bilder")
    pets = find_pet([p for p in (pet_dir, vb) if p] + ([(os.path.dirname(vb), 3)] if vb else []))
    rows["pet"] = {"wert": [pets[-1]["ordner"]] if pets else [], "quelle": "gefunden" if pets else "keins",
                   "kandidaten": [{"wert": p["ordner"], "info": " ".join(x for x in (p["datum"], p["beschreibung"]) if x)}
                                  for p in reversed(pets)]}
    outs = dict(letd.default_out(root)); outs.update({"pet_" + k: x for k, x in pet_run.default_out(root).items()})
    rows["ausgabe"] = {"wert": outs.get("out"), "quelle": "zone"}
    rows["kennzahlen"] = {"wert": outs.get("kz"), "quelle": "zone"}
    rows["pet_ausgabe"] = {"wert": outs.get("pet_out"), "quelle": "zone"}
    rows["synthseg"] = {"wert": True, "quelle": "standard"}
    rows["seite_neu"] = {"wert": True, "quelle": "standard"}
    return {"zeilen": _mark(rows)}


def input_rows(inp):
    """The rows CT, RTSTRUCT, RTPLAN, fractions and prescription from what a run names (or the user gave), completed
    from the export folder."""
    found = complete_inputs(inp)
    rows = {k: {"wert": found[k][0], "quelle": found[k][1]} if k in found else {"wert": None, "quelle": "fehlt"}
            for k in RUN_INPUTS}
    rows.update(plan_rows(rows["rtplan"]["wert"]) if rows["rtplan"]["wert"] else plan_rows(None))
    rows["ptv"] = ptv_row(rows["rtstruct"]["wert"])
    return rows


# as tools/mrt_overlay/geometry.PTV_AID (the server has no numpy to import it; a test keeps the two equal).
# helper structures that carry "PTV" in their name but are not the target: rings, optimisation and help volumes,
# "Brain-PTV" (something minus the PTV), and names starting with z, the common mark of a planning aid
PTV_AID = re.compile(r"ring|opt|hilf|help|avoid|dummy|eval|^z|[-\u2013]\s*ptv", re.I)


def roi_names(rtstruct):
    """The ROI names of an RT structure set, in file order; [] when it is not readable here."""
    h = dicomkopf.read(letd.to_local(rtstruct or "")) if rtstruct else None
    if not h or h.get("Modality") != "RTSTRUCT": return []
    return [str(r["ROIName"]) for r in h.get("StructureSetROISequence") or [] if r.get("ROIName")]


def ptv_proposal(names):
    """The ROIs that may be the PTV, best first: names starting with PTV before others that contain it, helper
    structures left out. The tool picks the largest by volume when the case names none (geometry.pick_ptv)."""
    cand = [n for n in names if "ptv" in n.lower() and not PTV_AID.search(n.strip())]
    return sorted(cand, key=lambda n: not n.strip().lower().startswith("ptv"))


def ptv_row(rtstruct):
    """The row PTV: the ROI names of the structure set to choose from (shown on this computer only, never stored
    outside the case). One clear PTV is taken; several are a proposal the user confirms; none leaves it to the tool."""
    names = roi_names(rtstruct)
    cand = ptv_proposal(names)
    row = {"wert": cand[0] if cand else None, "kandidaten": [{"wert": n, "info": ""} for n in names]}
    # never required: without a choice the tool picks the largest PTV itself
    row["quelle"] = "gefunden" if len(cand) == 1 else "vorschlag" if cand else "optional"
    return row


def plan_rows(rtplan):
    """Fractions and prescription rows after the user picked another RT plan."""
    f = plan_facts(rtplan) if rtplan else {}
    return {"fraktionen": {"wert": f.get("fraktionen"), "quelle": "plan" if f.get("fraktionen") else "fehlt"},
            "verschreibung": {"wert": f.get("verschreibung_gy_rbe"), "quelle": "plan" if f.get("verschreibung_gy_rbe") else "fehlt"}}


# ------------------------------------------------------------------ cacheF from the viewer's own page
# The aligned volumes of a case (cacheF) may be gone while the viewer still shows them: the page holds each one as a
# mosaic. tools/mrt_overlay/cachef.py decodes them; this runs it as a job of the server, as the LETd and PET runs.
CACHEF_SCRIPT = os.path.join(os.path.dirname(letd.TOOL_DIR), "tools", "mrt_overlay", "run_cachef.sh")
CACHEF_LOG = "cachef_lauf.log"
CACHEF_STEPS = ("read", "volumes", "mask", "write")
CACHEF_JOB = {"proc": None, "thread": None, "stop": False, "state": None}


def cachef_target(root, page):
    """Where a cacheF made from the viewer's page goes: the zone lokal, one folder per page, so that a second
    patient's viewer never overwrites the first one's. None without that zone or without a page."""
    page = letd.to_local(page or "")
    if not page: return None
    base = root if os.path.isdir(os.path.join(root, "lokal")) else os.path.join(root, "workspace")
    if not os.path.isdir(os.path.join(base, "lokal")): return None
    key = hashlib.sha1(os.path.realpath(page).encode("utf-8")).hexdigest()[:10]
    return os.path.join(base, "lokal", "viewer", key, "cacheF")


def cachef_sources(values):
    """Page, data folder and the viewer's copy of the cacheF log, when all three are there; else None."""
    page, daten, sk = (letd.to_local(values.get(k) or "") for k in ("seite", "daten", "skripte"))
    log = os.path.join(sk, "stand", "cacheF_log.json") if sk else ""
    if page and os.path.isfile(page) and daten and os.path.isdir(daten) and log and os.path.isfile(log):
        return {"seite": page, "daten": daten, "log": log}
    return None


def _cachef_running():
    j = CACHEF_JOB
    return j["proc"] is not None or (j["thread"] is not None and j["thread"].is_alive())


def cachef_status():
    with letd.LOCK:
        st = dict(CACHEF_JOB["state"] or {})
        running = _cachef_running()
    if not st: return {"running": False}
    lines = letd._all_lines(st["log_local"])
    steps, vol = {s: "todo" for s in CACHEF_STEPS}, None
    for line in lines:
        m = re.match(r"^## STEP (\w+)", line)
        if m and m.group(1) in steps:
            for s in steps:
                if steps[s] == "now": steps[s] = "done"
            steps[m.group(1)] = "now"
        m = re.match(r"^## VOL (\d+)/(\d+)", line)
        if m: vol = [int(m.group(1)), int(m.group(2))]
    if st.get("status") == "done": steps = {s: "done" for s in steps}
    tail = letd._tail(st["log_local"], 4000)
    st.update(running=running, steps=steps, vol=vol, log=tail)
    if st.get("error"): st["meldung"] = st["error"]
    elif st.get("status") == "failed":
        st["meldung"] = next((l for l in reversed(tail.splitlines()) if l.strip() and not l.startswith("#")), "")
    return st


def cachef_start(values, root, on_end=None):
    """Starts run_cachef.sh for the viewer of the case rows. Raises ValueError when page, data folder or the copy of
    the log is missing or there is no zone lokal, RuntimeError when a run is going."""
    src = cachef_sources(values)
    out = cachef_target(root, src["seite"]) if src else None
    if not src: raise ValueError("quellen")
    if not out: raise ValueError("kein_lokal")
    with letd.LOCK:
        if _cachef_running(): raise RuntimeError("running")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        CACHEF_JOB["stop"] = False
        CACHEF_JOB["state"] = {"status": "running", "start": time.strftime("%Y-%m-%d %H:%M:%S"), "t0": time.time(),
                               "cache": out, "log_local": os.path.join(os.path.dirname(out), CACHEF_LOG)}
        cmd = ["bash", CACHEF_SCRIPT, src["seite"], src["daten"], src["log"], out]
        t = threading.Thread(target=_cachef_run, args=(cmd, on_end), daemon=True)
        CACHEF_JOB["thread"] = t
        t.start()       # inside the lock: a second request must see the thread alive, not one not yet started
    return cachef_status()


def cachef_stop():
    return letd.stop(CACHEF_JOB)


def _cachef_run(cmd, on_end):
    st = CACHEF_JOB["state"]
    try:
        with open(st["log_local"], "w") as log:
            log.write(f"# {st['start']}  {' '.join(shlex.quote(c) for c in cmd)}\n"); log.flush()
            rc = letd._spawn(cmd, log, dict(os.environ, PYTHONUNBUFFERED="1"), job=CACHEF_JOB)
        ok = rc == 0 and os.path.isdir(st["cache"]) and _cache_log(st["cache"], set(os.listdir(st["cache"]))) is not None
        st["rc"] = rc
        st["status"] = "stopped" if CACHEF_JOB["stop"] else ("done" if ok else "failed")
    except Exception as e:  # the record says what went wrong; the page shows it
        st["status"] = "failed"; st["error"] = str(e)
    finally:
        st["end"] = time.strftime("%Y-%m-%d %H:%M:%S"); st["seconds"] = round(time.time() - st["t0"])
        if on_end:
            try: on_end(st)
            except Exception: pass


# ------------------------------------------------------------------ the start check of the assistant
# The rows can all be filled while the start would still stop: build_page.py needs Skripte/viewer.json beside the page
# and cacheG beside cacheF. The assistant runs the same checks as the start, so that "save" is offered only for a case
# that starts, and it fills what it can fill itself from the user's own copies.
def _base(root):
    return root if os.path.isdir(os.path.join(root, "lokal")) else os.path.join(root, "workspace")


def viewer_json_source(root):
    """The user's own copy of the viewer's settings (zone eigen), from which a viewer without one is completed."""
    p = os.path.join(_base(root), "eigen", "mrt-viewer", "viewer.json")
    return p if os.path.isfile(p) else None


def _forms(values):
    v = lambda k: values.get(k) if values.get(k) not in (None, "") else ""
    flag = lambda k: "1" if values.get(k, True) not in (False, "", "0", "false") else ""
    letd_f = {"mrt": v("bilder"), "run": v("lauf"), "ct": v("ct"), "rs": v("rtstruct"), "cache": v("cache"),
              "view": v("daten"), "rx": str(v("verschreibung")), "fx": str(v("fraktionen")), "out": v("ausgabe"),
              "kz": v("kennzahlen"), "build": flag("seite_neu"), "synthseg": flag("synthseg")}
    pets = values.get("pet") or []
    pet_f = {"mrt": v("bilder"), "pet": pets[0] if pets else "", "cache": v("cache"), "view": v("daten"),
             "ref": "brain", "side": "", "labels": "", "out": v("pet_ausgabe"), "build": flag("seite_neu")} if pets else None
    return letd_f, pet_f


def fixes(values, root):
    """What the assistant can fill itself before the start: viewer.json from the zone eigen into the viewer's Skripte,
    cacheG from the viewer's stand into the cacheF it made. Never into a folder it did not make, never over a file."""
    out = []
    cache, daten = (letd.to_local(values.get(k) or "") for k in ("cache", "daten"))
    if not daten: return out
    sk = os.path.join(os.path.dirname(daten), "Skripte")
    vj, src = os.path.join(sk, "viewer.json"), viewer_json_source(root)
    if os.path.isdir(sk) and not os.path.isfile(vj) and src:
        out.append({"art": "viewer_json", "ziel": vj, "quelle": src})
    mine = cachef_target(root, values.get("seite"))
    if cache and mine and os.path.isdir(cache) and os.path.realpath(cache) == os.path.realpath(mine):
        g = os.path.join(os.path.dirname(mine), "cacheG")
        names = [n for s, n in CACHEG if not os.path.isfile(os.path.join(g, n)) and os.path.isfile(os.path.join(sk, "stand", s))]
        if names: out.append({"art": "cacheg", "ziel": g, "dateien": names})
    return out


CACHEG = (("cacheG_log.json", "log.json"), ("cacheG_vent.json", "vent.json"))   # as tools/mrt_overlay/cachef.py


def start_check(values, root, viewer=None):
    """The checks of the start on the rows: {"probleme": {"letd": [...], "pet": [...]}, "behebbar": [...], "ok"}.
    PET is checked when a PET series is chosen."""
    letd_f, pet_f = _forms(values)
    probs = {"letd": letd.check(letd_f, root, viewer)[0]}
    if pet_f: probs["pet"] = pet_run.check(pet_f, root, viewer)[0]
    probs = {k: x for k, x in probs.items() if x}
    return {"probleme": probs, "behebbar": fixes(values, root), "ok": not probs}


def prepare(values, root, viewer=None):
    """Fills what fixes() names, then checks again."""
    for f in fixes(values, root):
        if f["art"] == "viewer_json":
            tmp = f["ziel"] + ".part"
            with open(f["quelle"], "rb") as a, open(tmp, "wb") as b: b.write(a.read())
            os.replace(tmp, f["ziel"])
        elif f["art"] == "cacheg":
            sk = os.path.join(os.path.dirname(letd.to_local(values["daten"])), "Skripte", "stand")
            os.makedirs(f["ziel"], exist_ok=True)
            for s, n in CACHEG:
                if n in f["dateien"]:
                    with open(os.path.join(sk, s), "rb") as a, open(os.path.join(f["ziel"], n + ".part"), "wb") as b:
                        b.write(a.read())
                    os.replace(os.path.join(f["ziel"], n + ".part"), os.path.join(f["ziel"], n))
    return start_check(values, root, viewer)


# what a pasted or chosen path must be, per row: a folder, a file, and what it holds
FOLDER_ROWS = {"daten", "skripte", "cache", "lauf", "ct", "pet", "bilder", "mrt", "ausgabe", "pet_ausgabe"}
FILE_ROWS = {"seite", "rtstruct", "rtplan", "kennzahlen"}
NEW_OK = {"ausgabe", "pet_ausgabe", "kennzahlen"}      # written by the run: need not exist yet


def check_path(key, text):
    """A path the user pasted or chose for row `key`, in any form (Windows with or without quotes, \\ or /, /mnt/...),
    as the local path, its Windows form, and a verdict: ok, fehlt (does not exist) or falsch (exists, wrong content).
    A folder given for a file row (RTSTRUCT, RTPLAN) is looked through for the file by DICOM Modality."""
    p = letd.to_local(text)
    if not p: return {"wert": None, "anzeige": "", "befund": "fehlt"}
    res = lambda q, b: {"wert": q, "anzeige": letd.to_windows(q), "befund": b}
    if key in NEW_OK: return res(p, "ok" if os.path.exists(p) or os.path.isdir(os.path.dirname(p)) else "fehlt")
    if not os.path.exists(p): return res(p, "fehlt")
    if key in ("rtstruct", "rtplan"):
        if _is(key, p): return res(p, "ok")
        if os.path.isdir(p):
            hits = [q for q, h in _dicom_files(p) if h.get("Modality") == MODALITY[key]]
            if hits: return res(max(hits, key=os.path.getmtime), "ok")
        return res(p, "falsch")
    if key == "ct": return res(p, "ok" if _is("ct", p) else "falsch")
    if key == "cache":
        ok = os.path.isdir(p) and _cache_log(p, set(os.listdir(p))) is not None
        return res(p, "ok" if ok else "falsch")
    if key == "lauf": return res(p, "ok" if all(os.path.isfile(os.path.join(p, n)) for n in RUN_FILES) else "falsch")
    if key == "pet":
        return res(p, "ok" if find_pet([(p, 3)]) else "falsch")
    if key in FILE_ROWS: return res(p, "ok" if os.path.isfile(p) else "falsch")
    return res(p, "ok" if os.path.isdir(p) else "falsch")


# ------------------------------------------------------------------ save a case
def missing(values):
    m = [k for k in REQUIRED if not values.get(k)]
    if not values.get("verschreibung") and not values.get("rtstruct"): m.append("verschreibung")
    return m


def save_case(folder, fall_id, values):
    """Writes fall.json and orte.json of a case from the confirmed rows. `values` maps a row to its local path or
    value. The folder lies in the zone lokal: the case names a treatment plan."""
    if not fall._ID.match(fall_id or ""): raise fall.FallError("fall_id: Buchstaben, Ziffern, . _ -")
    m = missing(values)
    if m: raise fall.FallError("fehlt: " + ", ".join(m))
    os.makedirs(folder, exist_ok=True)
    orte = fall.load_orte(folder)
    cp = lambda p: fall.to_case_path(folder, letd.to_local(p), orte) if p else None

    def change(d):
        lp = {k: letd.to_local(values[k]) for k in ("rtplan", "rtstruct") if values.get(k)}
        plan = d.get("plan") or {}
        if lp.get("rtplan"):
            plan.update(datei=cp(lp["rtplan"]), sha256=fall.sha256(lp["rtplan"]))
        if plan or values.get("fraktionen"):
            plan["fraktionen"] = int(values["fraktionen"])
            rx = values.get("verschreibung")
            plan["verschreibung_gy_rbe"] = float(str(rx).replace(",", ".")) if rx not in (None, "") else None
            if not plan.get("datei"):   # a case without a readable RT plan keeps the numbers the user confirmed
                plan["datei"] = None
        d["plan"] = {k: x for k, x in plan.items() if x is not None} or None
        if d["plan"] and "datei" not in d["plan"]:
            d["plan_ohne_datei"] = {"fraktionen": d["plan"].get("fraktionen"),
                                    "verschreibung_gy_rbe": d["plan"].get("verschreibung_gy_rbe")}
            d["plan"] = None
        if lp.get("rtstruct"):
            # the ROI name stays in the case (zone lokal) and reaches the tool as --ptv
            d["strukturen"] = {"datei": cp(lp["rtstruct"]), "sha256": fall.sha256(lp["rtstruct"]), "ptv": values.get("ptv") or None}
        ct = {"serie": "ct_plan", "modalitaet": "CT", "datum": None, "ordner": cp(values["ct"]), "nach_bezug": None, "guete": None}
        keep = [b for b in d.get("bilder", []) if b.get("modalitaet") not in ("CT", "PT")]
        pets = []
        for i, p in enumerate(values.get("pet") or []):
            h = _dicom_head(letd.to_local(p), os.listdir(letd.to_local(p))) or {}
            date = h.get("SeriesDate") or h.get("StudyDate") or ""
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}" if re.match(r"^\d{8}$", date) else None
            pets.append({"serie": f"pet_{date or i + 1}", "modalitaet": "PT", "datum": date, "ordner": cp(p),
                         "nach_bezug": None, "guete": None, "pet": {"tracer": "aminosaeure", "referenz": None, "suv_eingabe": None}})
        d["bilder"] = [ct] + keep + pets
        d["bezug"] = {"serie": "ct_plan"}
        d["viewer"] = {k: cp(values[k]) for k in ("seite", "daten", "skripte", "cache", "bilder") if values.get(k)}
        d["einstellungen"] = {"ausgabe": cp(values.get("ausgabe")), "kennzahlen": cp(values.get("kennzahlen")),
                              "pet_ausgabe": cp(values.get("pet_ausgabe")),
                              "synthseg": bool(values.get("synthseg", True)), "seite_neu": bool(values.get("seite_neu", True)),
                              "lauf": "mc-1"}
        d["einstellungen"] = {k: x for k, x in d["einstellungen"].items() if x is not None}
        run = letd.to_local(values["lauf"])
        wb = [r for r in d["ergebnisse"].get("workbench", []) if r.get("id") != "mc-1"]
        eing = {}
        if d.get("plan") and d["plan"].get("sha256"): eing["plan"] = d["plan"]["sha256"]
        if d.get("strukturen"): eing["strukturen"] = d["strukturen"]["sha256"]
        wb.insert(0, {"id": "mc-1", "rifi": None, "ordner": cp(run), "version": None, "eingaben": eing,
                      "dateien": {n: None for n in RUN_FILES}})
        d["ergebnisse"]["workbench"] = wb
        fall.stamp(d)

    # orte.json first: fall.json names places that only it resolves
    probe = {k: values.get(k) for k in ("rtplan", "rtstruct", "ct", "seite", "daten", "skripte", "cache", "bilder",
                                         "ausgabe", "kennzahlen", "pet_ausgabe", "lauf")}
    for p in list(probe.values()) + list(values.get("pet") or []):
        if p: cp(p)
    fall.save_orte(folder, orte)
    return fall.update(folder, change, create=fall_id)


# ------------------------------------------------------------------ a case as the forms of letd.py and pet_run.py
def _get(folder, d, orte, sec, key):
    v = (d.get(sec) or {}).get(key)
    return fall.resolve(folder, v, orte) if v else ""


def letd_form(folder):
    """The form letd.start takes, filled from the case."""
    d, orte = fall.load(folder), fall.load_orte(folder)
    ein = d.get("einstellungen") or {}
    run = next((r for r in d["ergebnisse"].get("workbench", []) if r["id"] == ein.get("lauf")), None) \
        or (d["ergebnisse"].get("workbench") or [None])[0]
    if not run or not run.get("ordner"): raise fall.FallError("ergebnisse.workbench: kein plan_mc-Lauf im Fall")
    ct = next((b for b in d.get("bilder", []) if b["serie"] == (d.get("bezug") or {}).get("serie")), None)
    plan = d.get("plan") or d.get("plan_ohne_datei") or {}
    rx = plan.get("verschreibung_gy_rbe")
    return {"mrt": _get(folder, d, orte, "viewer", "bilder"), "run": fall.resolve(folder, run["ordner"], orte),
            "ct": fall.resolve(folder, ct["ordner"], orte) if ct else "",
            "rs": _get(folder, d, orte, "strukturen", "datei"),
            "ptv": (d.get("strukturen") or {}).get("ptv") or "",
            "cache": _get(folder, d, orte, "viewer", "cache"), "view": _get(folder, d, orte, "viewer", "daten"),
            "rx": "" if rx is None else f"{rx:g}", "fx": str(plan.get("fraktionen") or ""),
            "out": _get(folder, d, orte, "einstellungen", "ausgabe"),
            "kz": _get(folder, d, orte, "einstellungen", "kennzahlen"),
            "build": "1" if ein.get("seite_neu", True) else "", "synthseg": "1" if ein.get("synthseg", True) else "",
            "oeffentlich": bool((d.get("plan") or {}).get("oeffentlich")), "fall": folder}


def pet_form(folder, serie=None):
    """The form pet_run.start takes, for the case's PET series `serie` (default: the newest)."""
    d, orte = fall.load(folder), fall.load_orte(folder)
    pets = [b for b in d.get("bilder", []) if b.get("modalitaet") == "PT"]
    if serie: pets = [b for b in pets if b["serie"] == serie]
    if not pets: raise fall.FallError("bilder: keine PET-Serie im Fall")
    b = sorted(pets, key=lambda x: x.get("datum") or "")[-1]
    rz = (b.get("pet") or {}).get("referenz") or {}
    ref = rz.get("art") or "brain"
    ein = d.get("einstellungen") or {}
    return {"mrt": _get(folder, d, orte, "viewer", "bilder"), "pet": fall.resolve(folder, b["ordner"], orte),
            "cache": _get(folder, d, orte, "viewer", "cache"), "view": _get(folder, d, orte, "viewer", "daten"),
            "ref": ref if ref in pet_run.REFS else "brain",
            "side": (rz.get("seite") or "") if ref in pet_run.REFS else "",
            "labels": fall.resolve(folder, rz["maske"], orte) if rz.get("maske") and ref in pet_run.REFS else "",
            "out": _get(folder, d, orte, "einstellungen", "pet_ausgabe"),
            "build": "1" if ein.get("seite_neu", True) else "", "oeffentlich": bool((d.get("plan") or {}).get("oeffentlich")),
            "fall": folder, "serie": b["serie"]}


# ------------------------------------------------------------------ folder chooser of the server
DRIVES = ":laufwerke"            # the chooser's top under WSL: the drives as C:, D: instead of /mnt


def _drives(mnt):
    try: return sorted(e.name for e in os.scandir(mnt) if len(e.name) == 1 and e.is_dir())
    except OSError: return []


def browse(path, root, mnt="/mnt"):
    """One folder for the chooser: its subfolders and files, each with its path, and the folder in Windows form.
    Starts at the workspace root when no path is given; above a drive (/mnt/c) come the drives as C:, D:. A pasted
    path in any form is taken. Names only, nothing opened."""
    if path == DRIVES and _drives(mnt):
        return {"pfad": DRIVES, "anzeige": "", "oben": None, "ordner": [], "dateien": [], "lauf": False,
                "ziele": [{"name": d.upper() + ":", "pfad": os.path.join(mnt, d)} for d in _drives(mnt)], "datei_ziele": []}
    p = letd.to_local(path) if path else root
    if not os.path.isdir(p): p = os.path.dirname(p) if os.path.isdir(os.path.dirname(p)) else root
    try: entries = sorted(os.scandir(p), key=lambda e: e.name.lower())
    except OSError: entries = []
    dirs = []
    for e in entries[:2000]:
        if not e.is_dir(follow_symlinks=True) or e.name.startswith("."): continue
        dirs.append(e.name)
        if len(dirs) >= 500: break
    files = [e.name for e in entries if e.is_file() and (e.name.lower().endswith((".dcm", ".html", ".json", ".npy"))
                                                          or "." not in e.name)][:300]
    parent = os.path.dirname(p.rstrip(os.sep)) or os.sep
    if os.path.dirname(p.rstrip(os.sep)) == mnt.rstrip(os.sep) and _drives(mnt): parent = DRIVES
    j = lambda n: os.path.join(p, n)
    return {"pfad": p, "anzeige": letd.to_windows(p), "oben": parent if parent != p else None, "ordner": dirs,
            "dateien": files, "lauf": all(os.path.isfile(j(n)) for n in RUN_FILES),
            "ziele": [{"name": n, "pfad": j(n)} for n in dirs], "datei_ziele": [{"name": n, "pfad": j(n)} for n in files]}


# ------------------------------------------------------------------ demo case (public, synthetic)
def _npy(path, shape, values):
    """A float32 .npy in C order, standard library only."""
    hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }" % "".join(f"{s}, " for s in shape)
    hdr += " " * (63 - (10 + len(hdr)) % 64) + "\n"
    with open(path, "wb") as fh:
        fh.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode())
        fh.write(struct.pack(f"<{len(values)}f", *values))


def _nifti_gz(path, nx, ny, nz, values):
    """A uint8 NIfTI-1 volume, 1 mm voxels, gzipped."""
    h = bytearray(348)
    struct.pack_into("<i", h, 0, 348)
    struct.pack_into("<8h", h, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    struct.pack_into("<hh", h, 70, 2, 8)                 # datatype uint8, bitpix
    struct.pack_into("<8f", h, 76, 1, 1, 1, 1, 1, 1, 1, 1)
    struct.pack_into("<f", h, 108, 352.0)
    h[344:348] = b"n+1\x00"
    with gzip.open(path, "wb") as fh: fh.write(bytes(h) + b"\x00" * 4 + bytes(values))


def _tg119_c(x, y):
    """The C-shaped target of the AAPM TG-119 test suite, in mm about the phantom centre (public geometry)."""
    r = (x * x + y * y) ** 0.5
    return 15 <= r <= 37 and not (x > 0 and abs(y) < 15)


def demo(folder):
    """The demo case: TG-119-like phantom with a C target, a synthetic RT plan (marked public), a plan_mc-like run, a
    viewer folder with a small synthetic MRI cache and one PET series. Every value is made up; small enough to build
    in a test. Returns the case folder."""
    j = lambda *a: os.path.join(folder, *a)
    for d in ("plan", "bilder/ct_plan", "bilder/pet_2001-04-01", "laeufe/mc-ohne-rifi", "viewer/Analyse/Verlauf-Daten",
              "viewer/Analyse/Skripte", "cacheF"):
        os.makedirs(j(*d.split("/")), exist_ok=True)
    n, nz, sp = 16, 4, 5.0
    pos = lambda i: (i - (n - 1) / 2) * sp
    for k in range(nz):
        px = b"".join(struct.pack("<h", 0 if (pos(a) ** 2 + pos(b) ** 2) ** 0.5 <= 40 else -1000) for b in range(n) for a in range(n))
        dicomkopf.write(j("bilder", "ct_plan", f"CT{k + 1:04d}.dcm"), {
            "Modality": "CT", "SOPInstanceUID": f"2.25.119.1.{k + 1}", "SeriesInstanceUID": "2.25.119.1",
            "FrameOfReferenceUID": "2.25.119.9", "StudyDate": "20010101", "InstanceNumber": k + 1,
            "Rows": n, "Columns": n, "PixelSpacing": [sp, sp], "SliceThickness": sp,
            "ImagePositionPatient": [pos(0), pos(0), k * sp]}, px)
        dicomkopf.write(j("bilder", "pet_2001-04-01", f"PT{k + 1:04d}.dcm"), {
            "Modality": "PT", "SOPInstanceUID": f"2.25.119.3.{k + 1}", "SeriesInstanceUID": "2.25.119.3",
            "SeriesDate": "20010401", "SeriesDescription": "Demo PET", "InstanceNumber": k + 1, "Rows": n, "Columns": n},
            bytes(2 * n * n))
    dicomkopf.write(j("plan", "rtstruct.dcm"), {
        "Modality": "RTSTRUCT", "SOPInstanceUID": "2.25.119.4", "FrameOfReferenceUID": "2.25.119.9",
        "StructureSetROISequence": [{"ROIName": "PTV"}, {"ROIName": "Core"}]})
    dicomkopf.write(j("plan", "rtplan.dcm"), {
        "Modality": "RTPLAN", "SOPInstanceUID": "2.25.119.5", "FrameOfReferenceUID": "2.25.119.9", "RTPlanLabel": "DEMO",
        "FractionGroupSequence": [{"NumberOfFractionsPlanned": 30}],
        "DoseReferenceSequence": [{"DoseReferenceType": "TARGET", "TargetPrescriptionDose": 60.0}],
        "IonBeamSequence": [{"RadiationType": "PROTON"}, {"RadiationType": "PROTON"}],
        "ReferencedStructureSetSequence": [{"ReferencedSOPInstanceUID": "2.25.119.4"}]})
    run = j("laeufe", "mc-ohne-rifi")
    letd_v = [3.0 if _tg119_c(pos(a), pos(b)) else 1.0 for _ in range(nz) for b in range(n) for a in range(n)]
    _npy(os.path.join(run, "letd.npy"), (nz, n, n), letd_v)
    _npy(os.path.join(run, "rbe_dose_1_1.npy"), (nz, n, n), [2.0 if v > 1 else 0.2 for v in letd_v])
    dicomkopf.write(os.path.join(run, "rbe_dose_1_1.dcm"), {"Modality": "RTDOSE", "SOPInstanceUID": "2.25.119.6",
                                                             "FrameOfReferenceUID": "2.25.119.9"})
    with open(os.path.join(run, "run.json"), "w", encoding="utf-8") as fh:
        json.dump({"eingaben": {"ct": "../../bilder/ct_plan", "rtstruct": "../../plan/rtstruct.dcm",
                                "rtplan": "../../plan/rtplan.dcm"}}, fh, indent=1)
    with open(j("viewer", "Verlauf.html"), "w", encoding="utf-8") as fh: fh.write("<!doctype html><title>Demo</title>\n")
    with open(j("viewer", "Analyse", "Skripte", "viewer.json"), "w", encoding="utf-8") as fh: json.dump({"demo": True}, fh)
    with open(j("cacheF", "log.json"), "w", encoding="utf-8") as fh:
        json.dump({"size": [8, 8, 8], "spacing": 1.0, "t1 2001-02-01": {"series": "demo", "kind": "3D", "native": [1.0, 1.0, 1.0]}}, fh)
    _nifti_gz(j("cacheF", "ref_t1.nii.gz"), 8, 8, 8, [int(100 + 50 * ((i // 64) % 2)) for i in range(512)])
    values = {"lauf": run, "ct": j("bilder", "ct_plan"), "rtstruct": j("plan", "rtstruct.dcm"), "rtplan": j("plan", "rtplan.dcm"),
              "seite": j("viewer", "Verlauf.html"), "daten": j("viewer", "Analyse", "Verlauf-Daten"),
              "skripte": j("viewer", "Analyse", "Skripte"), "cache": j("cacheF"), "bilder": j("viewer"),
              "fraktionen": 30, "verschreibung": 60.0, "pet": [j("bilder", "pet_2001-04-01")],
              "ausgabe": j("ergebnisse", "letd"), "kennzahlen": j("ergebnisse", "kennzahlen.json"),
              "pet_ausgabe": j("ergebnisse", "pet")}
    save_case(folder, "demo-tg119", values)

    def finish(d):
        d["plan"]["oeffentlich"] = True; d["plan"]["teilchen"] = "proton"
        d["bezug"].update(frame_of_reference_uid="2.25.119.9", raster={"size": [n, n, nz], "spacing_mm": [sp, sp, sp]})
        d["bilder"][0]["datum"] = "2001-01-01"
        d["zeitleiste"] = [{"datum": "2000-12-01", "art": "biopsie", "ausgangszustand": True, "befund": None, "ort": None}]
        wb = d["ergebnisse"]["workbench"][0]
        wb.update(rifi=False, version="demo", dateien={n_: fall.sha256(os.path.join(run, n_)) for n_ in RUN_FILES})
    fall.update(folder, finish)
    return folder
