"""Set up a case once instead of typing ten paths per run: the search behind the assistant of the tab "LETd/PET on MRI",
the demo case, and the step from a case to the forms that letd.py and pet_run.py already start.

The search runs as code of the workbench server on the user's computer when the user starts it in the web UI. It reads
folder names and DICOM headers (Modality, dates, from the RT plan the fractions and the prescription), never pixel
data, and it returns paths and those few facts to the page, nothing else.
"""
import gzip
import json
import os
import re
import struct
import time

from . import dicomkopf, fall, letd, pet_run

DEPTH = 6
LIMIT = 40000                    # folders visited at most per search root
RUN_FILES = ("letd.npy", "rbe_dose_1_1.npy", "rbe_dose_1_1.dcm")
RUN_INPUTS = ("ct", "rtstruct", "rtplan")


# ------------------------------------------------------------------ search
def _walk(top, depth=DEPTH, limit=LIMIT):
    yield from letd._walk(top, depth, limit)


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


def find_viewer(mrt):
    """From the MRI folder: Verlauf.html (the newest), Verlauf-Daten, Skripte beside it, and cacheF when it lies inside."""
    top = letd.to_local(mrt)
    if not top or not os.path.isdir(top): return {"fehler": "mrt_nicht_gefunden"}
    found, pages = {}, []
    for d, dirs, files in _walk(top, 5):
        for n in files:
            if n.lower() == "verlauf.html": pages.append(os.path.join(d, n))
        base = os.path.basename(d).lower()
        if "daten" not in found and base == "verlauf-daten":
            found["daten"] = d
            sk = os.path.join(os.path.dirname(d), "Skripte")
            if os.path.isdir(sk): found["skripte"] = sk
        if "cache" not in found and {"log.json", "ref_t1.nii.gz"} <= files: found["cache"] = d
    if pages: found["seite"] = max(pages, key=os.path.getmtime)
    found["bilder"] = top
    return found


def find_cache(roots):
    """cacheF (log.json and ref_t1.nii.gz) under the remembered roots: it lies outside OneDrive, so the MRI folder
    does not hold it."""
    for r in roots:
        r = letd.to_local(r)
        if not r or not os.path.isdir(r): continue
        for d, _, files in _walk(r, 4):
            if {"log.json", "ref_t1.nii.gz"} <= files: return d
    return None


def run_inputs(run):
    """CT, RT structure set and RT plan a plan_mc run used, from its run.json; for a run older than run.json, from
    chain.json when it names them. Missing ones stay out: the assistant asks for them once."""
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
    """Folders of DICOM series with Modality PT, one entry per series, with its date."""
    out, seen = [], set()
    for r in roots:
        r = letd.to_local(r)
        if not r or not os.path.isdir(r): continue
        for d, _, files in _walk(r):
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


def scan(mrt, root, cache_roots=(), pet_dir=None, extra_runs=()):
    """The proposal of the assistant: one value per row, with candidates where several fit. Each row says where it
    came from (gefunden, gemerkt, Plan, fehlt), so the page can mark what the user must still choose."""
    v = find_viewer(mrt) if mrt else {}
    if v.get("fehler"): return {"fehler": v["fehler"]}
    rows = {k: {"wert": v.get(k), "quelle": "gefunden" if v.get(k) else "fehlt"} for k in ("seite", "daten", "skripte", "cache")}
    rows["bilder"] = {"wert": v.get("bilder"), "quelle": "eingegeben"}
    if not rows["cache"]["wert"]:
        c = find_cache(cache_roots)
        if c: rows["cache"] = {"wert": c, "quelle": "gemerkt"}
    runs = find_runs(list(search_roots(root)) + list(extra_runs))
    rows["lauf"] = {"wert": runs[0]["ordner"] if runs else None, "quelle": "gefunden" if runs else "fehlt",
                    "kandidaten": [{"wert": r["ordner"], "info": r["zeit"]} for r in runs[:20]]}
    inp = runs[0]["eingaben"] if runs else {}
    for k in RUN_INPUTS:
        rows[k] = {"wert": inp.get(k), "quelle": "lauf" if inp.get(k) else "fehlt"}
    facts = plan_facts(rows["rtplan"]["wert"]) if rows["rtplan"]["wert"] else {}
    rows["fraktionen"] = {"wert": facts.get("fraktionen"), "quelle": "plan" if facts.get("fraktionen") else "fehlt"}
    rows["verschreibung"] = {"wert": facts.get("verschreibung_gy_rbe"),
                             "quelle": "plan" if facts.get("verschreibung_gy_rbe") else "fehlt"}
    pets = find_pet([p for p in (pet_dir, v.get("bilder")) if p])
    rows["pet"] = {"wert": [p["ordner"] for p in pets], "quelle": "gefunden" if pets else "keins",
                   "kandidaten": [{"wert": p["ordner"], "info": " ".join(x for x in (p["datum"], p["beschreibung"]) if x)} for p in pets]}
    outs = dict(letd.default_out(root)); outs.update({"pet_" + k: x for k, x in pet_run.default_out(root).items()})
    rows["ausgabe"] = {"wert": outs.get("out"), "quelle": "zone"}
    rows["kennzahlen"] = {"wert": outs.get("kz"), "quelle": "zone"}
    rows["synthseg"] = {"wert": True, "quelle": "standard"}
    rows["seite_neu"] = {"wert": True, "quelle": "standard"}
    return {"zeilen": rows}


def plan_rows(rtplan):
    """Fractions and prescription rows after the user picked another RT plan."""
    f = plan_facts(rtplan)
    return {"fraktionen": {"wert": f.get("fraktionen"), "quelle": "plan" if f.get("fraktionen") else "fehlt"},
            "verschreibung": {"wert": f.get("verschreibung_gy_rbe"), "quelle": "plan" if f.get("verschreibung_gy_rbe") else "fehlt"}}


# ------------------------------------------------------------------ save a case
REQUIRED = ("lauf", "ct", "cache", "daten", "fraktionen")


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
            "cache": _get(folder, d, orte, "viewer", "cache"), "view": _get(folder, d, orte, "viewer", "daten"),
            "rx": "" if rx is None else f"{rx:g}", "fx": str(plan.get("fraktionen") or ""),
            "out": _get(folder, d, orte, "einstellungen", "ausgabe"),
            "kz": _get(folder, d, orte, "einstellungen", "kennzahlen"),
            "build": "1" if ein.get("seite_neu", True) else "", "synthseg": "1" if ein.get("synthseg", True) else "",
            "oeffentlich": bool((d.get("plan") or {}).get("oeffentlich"))}


def pet_form(folder, serie=None):
    """The form pet_run.start takes, for the case's PET series `serie` (default: the newest)."""
    d, orte = fall.load(folder), fall.load_orte(folder)
    pets = [b for b in d.get("bilder", []) if b.get("modalitaet") == "PT"]
    if serie: pets = [b for b in pets if b["serie"] == serie]
    if not pets: raise fall.FallError("bilder: keine PET-Serie im Fall")
    b = sorted(pets, key=lambda x: x.get("datum") or "")[-1]
    ref = ((b.get("pet") or {}).get("referenz") or {}).get("art") or "brain"
    ein = d.get("einstellungen") or {}
    return {"mrt": _get(folder, d, orte, "viewer", "bilder"), "pet": fall.resolve(folder, b["ordner"], orte),
            "cache": _get(folder, d, orte, "viewer", "cache"), "view": _get(folder, d, orte, "viewer", "daten"),
            "ref": ref if ref in pet_run.REFS else "brain", "side": "", "labels": "",
            "out": _get(folder, d, orte, "einstellungen", "pet_ausgabe"),
            "build": "1" if ein.get("seite_neu", True) else "", "oeffentlich": bool((d.get("plan") or {}).get("oeffentlich"))}


# ------------------------------------------------------------------ folder chooser of the server
def browse(path, root):
    """One folder for the chooser: its subfolders and a hint per folder of what it holds (a plan_mc run, a viewer
    data folder). Starts at the workspace root when no path is given. Names only, nothing opened."""
    p = letd.to_local(path) if path else root
    if not os.path.isdir(p): p = os.path.dirname(p) if os.path.isdir(os.path.dirname(p)) else root
    try: entries = sorted(os.scandir(p), key=lambda e: e.name.lower())
    except OSError: entries = []
    dirs = []
    for e in entries[:2000]:
        if not e.is_dir(follow_symlinks=True) or e.name.startswith("."): continue
        dirs.append(e.name)
        if len(dirs) >= 500: break
    files = [e.name for e in entries if e.is_file() and e.name.lower().endswith((".dcm", ".html", ".json", ".npy"))][:200]
    parent = os.path.dirname(p.rstrip(os.sep)) or os.sep
    return {"pfad": p, "anzeige": letd.to_windows(p), "oben": parent if parent != p else None, "ordner": dirs,
            "dateien": files, "lauf": all(os.path.isfile(os.path.join(p, n)) for n in RUN_FILES)}


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
        json.dump({"demo": True, "serien": ["t1_2001-02-01"]}, fh)
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
