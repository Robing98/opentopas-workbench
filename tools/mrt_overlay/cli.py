"""Command line: export, series, onto (any MRI), viewer (the grid of the MRI viewer). Research only, no diagnosis.

  python -m mrt_overlay.cli export RUN OUT --fractions N
  python -m mrt_overlay.cli series EXAMDIR
  python -m mrt_overlay.cli onto MRI CT CTFRAME OUT [--rtstruct RS] [--labels SEG] [--prescription GY]
  python -m mrt_overlay.cli viewer CACHEF CT CTFRAME OUT [--rtstruct RS] [--labels SEG] [--viewer-data DIR]
  python -m mrt_overlay.cli pet PETDIR CACHEF VIEWERDATA [--ref brain|striatum|contralateral] [--out DIR]
  python -m mrt_overlay.cli viewer --fall CASEDIR      every path from fall.json; a path given beside it wins
  python -m mrt_overlay.cli pet --fall CASEDIR [--serie pet_2001-04-01]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import SimpleITK as sitk

from . import export as ex, geometry as geo, layer as ly, pet as pe, register as rg, series as se, ventricles as ve


def _ptv(a, grid_ct):
    if not a.rtstruct: return None, None
    st = geo.read_rtstruct(a.rtstruct)
    name = geo.pick_ptv(st, a.ptv)
    if not name: return None, None
    return geo.like(geo.contour_mask(st[name][1], grid_ct).astype(np.uint8), grid_ct), name


def _margin_on(grid, labels_path):
    vent, rim = ve.margin(sitk.ReadImage(labels_path))
    # the segmentation lies in the physical space of the T1 it came from: identity onto the grid
    return ly.resample(rim, grid, sitk.Transform(), label=True).astype(bool)


def _prescription(a, layers):
    if a.prescription: return float(a.prescription), "angegeben"
    if "ptv" in layers and layers["ptv"].any():
        return float(np.median(layers["dose_rbe_1_1"][layers["ptv"]])), "Median 1,1 x Dosis im PTV"
    raise SystemExit("--prescription is needed when no PTV is given")


def _common(a, mri, ct, transform, quality, out):
    ref = sitk.ReadImage(os.path.join(a.ctframe, "dose_rbe_1_1.nii.gz"))
    ptv_ct, ptv_name = _ptv(a, ref)
    layers = ly.to_grid(a.ctframe, mri, transform, ptv_ct)
    rx, rx_how = _prescription(a, layers)
    margin = _margin_on(mri, a.labels) if a.labels else np.zeros(layers["letd"].shape, bool)
    look, hit = ly.look_list(layers["letd"], layers["dose_rbe_1_1"], margin, rx, mri.GetSpacing())
    mk = ly.marks(layers, margin, rx, hit)
    os.makedirs(out, exist_ok=True)
    for k in ("letd", "dose_phys", "dose_rbe_1_1", "dose_rbe_bahn_linear"):
        if k in layers: sitk.WriteImage(geo.like(layers[k].astype(np.float32), mri), os.path.join(out, f"{k}.nii.gz"))
    sitk.WriteImage(geo.like(mk, mri), os.path.join(out, "marks.nii.gz"))
    sitk.WriteTransform(transform, os.path.join(out, "mri_to_ct.tfm"))
    info = {"prescription_gy_rbe": rx, "prescription_from": rx_how, "ptv": bool(ptv_name), "margin": bool(a.labels),
            "registration": quality, "look_list": look,
            "hinweis": "Nur Anzeige, keine Diagnose. Deutung bei Neuroradiologie und Strahlentherapie. "
                       "Variable RBE nur Forschung, klinisch gilt 1,1."}
    return layers, mk, info


# ------------------------------------------------------------------ the case file (fall.json)
def _case_modules():
    """fall.py and fall_setup.py of the workbench: standard library only, so no install; the scripts run from tools/."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path: sys.path.insert(0, root)
    from opentopas_workbench import fall, fall_setup
    return fall, fall_setup


def _version():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip(): return "git:" + r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _case_error(a, msg):
    raise SystemExit(f"Fall {a.fall}: {msg}")


def _viewer_from_case(a):
    """Fills what the command line left empty from the case; exports the plan_mc run into ct_frame when that has not
    happened yet, so that one call with --fall does what run.sh does with nine arguments."""
    fall, fs = _case_modules()
    try: f = fs.letd_form(a.fall)
    except fall.FallError as e: _case_error(a, e)
    out = f["out"]
    a.cache = a.cache or f["cache"]; a.ct = a.ct or f["ct"]
    a.ctframe = a.ctframe or (os.path.join(out, "ct_frame") if out else "")
    a.out = a.out or (os.path.join(out, "viewer_grid") if out else "")
    a.rtstruct = a.rtstruct or f["rs"] or None
    if a.prescription is None and f["rx"]: a.prescription = float(f["rx"])
    a.viewer_data = a.viewer_data or f["view"] or None
    a.kennzahlen = a.kennzahlen or f["kz"] or None
    if f["synthseg"] and not (a.synthseg or a.labels) and shutil.which("mri_synthseg"): a.synthseg = "mri_synthseg"
    gaps = [n for n, v in (("viewer.cache", a.cache), ("Planungs-CT (bilder, bezug)", a.ct),
                           ("einstellungen.ausgabe", a.out)) if not v]
    if gaps: _case_error(a, "fehlt: " + ", ".join(gaps))
    if not os.path.isfile(os.path.join(a.ctframe, "dose_rbe_1_1.nii.gz")):
        # one fraction by default would give the dose of one fraction against a total prescription, without a word
        if not f["fx"]: _case_error(a, "fehlt: plan.fraktionen (für den Export des plan_mc-Laufs)")
        _step("export")
        ex.export(f["run"], a.ctframe, int(f["fx"]))


def _viewer_to_case(a, q, exams):
    """bilder[].nach_bezug and bilder[].guete per exam on the viewer grid, and ergebnisse.viewer, back into the case.
    The viewer grid has one transform onto the CT; each exam's own check (its distance to that transform) is its
    quality, as the viewer shows every exam on that one grid."""
    fall, _ = _case_modules()
    orte = fall.load_orte(a.fall)
    cp = lambda p: fall.to_case_path(a.fall, p, orte)
    tfm, cache = cp(os.path.join(a.out, "mri_to_ct.tfm")), cp(a.cache)
    spread = q.get("spread_rms_mm")
    files = {}
    if a.viewer_data:
        for n in ("letd_all.js", "marks_all.js", "letd.json"):
            fp = os.path.join(a.viewer_data, n)
            if os.path.isfile(fp): files[n] = fall.sha256(fp)
    data = cp(a.viewer_data) if a.viewer_data else None
    overlay = cp(os.path.join(a.out, "overlay.json"))
    fall.save_orte(a.fall, orte)     # first: fall.json may name a place only orte.json resolves

    def change(d):
        mr = {}
        for r in exams:
            g = {"verfahren": "mattes_mi", "streuung_mm": None if spread != spread else round(spread, 2),
                 "geeignet": r["suitable"], "native_mm": r["native_mm"]}
            if "check_ok" in r: g.update(rms_mm=r["check_rms_mm"], max_mm=r["check_max_mm"], ok=r["check_ok"])
            serie = f"{r['track']}_{r['day']}".replace(" ", "_")
            mr[serie] = {"serie": serie, "modalitaet": "MR", "art": r["track"],
                         "datum": r["day"] if fall._DATE.match(r["day"]) else None, "ordner": cache,
                         "volumen": f"{r['track']}_{r['day']}.npy", "nach_bezug": tfm, "guete": g if "ok" in g else None}
        kept = [b for b in d.get("bilder", []) if b.get("modalitaet") != "MR" or b["serie"] not in mr]
        d["bilder"] = kept + [mr[k] for k in sorted(mr)]
        runs = [x["id"] for x in d["ergebnisse"].get("workbench", [])]
        lauf = (d.get("einstellungen") or {}).get("lauf")
        entry = {"id": "verlauf", "aus": [lauf] if lauf in runs else runs[:1], "ordner": data, "datei": overlay,
                 "version": _version(), "dateien": files}
        d["ergebnisse"]["viewer"] = [x for x in d["ergebnisse"].get("viewer", []) if x.get("id") != "verlauf"] + [entry]
        fall.stamp(d)
    fall.update(a.fall, change)


def _pet_from_case(a):
    fall, fs = _case_modules()
    try: f = fs.pet_form(a.fall, a.serie)
    except fall.FallError as e: _case_error(a, e)
    a.pet = a.pet or f["pet"]; a.cache = a.cache or f["cache"]; a.viewer_data = a.viewer_data or f["view"]
    a.out = a.out or f["out"] or None
    if a.ref is None: a.ref = f["ref"]
    if a.ref == f["ref"]:    # the case's side and label map belong to the case's reference only
        a.side = a.side or f["side"] or None; a.labels = a.labels or f["labels"] or None
    gaps = [n for n, v in (("PET-Serie", a.pet), ("viewer.cache", a.cache), ("viewer.daten", a.viewer_data)) if not v]
    if gaps: _case_error(a, "fehlt: " + ", ".join(gaps))


def _pet_to_case(a, e):
    """The PET's transform (viewer grid onto the PET) and its side check into its entry in bilder."""
    fall, _ = _case_modules()
    orte = fall.load_orte(a.fall)
    here = os.path.normpath(os.path.abspath(a.pet))
    tfm = os.path.join(a.out, f"grid_to_pet_{e['day']}.tfm") if a.out else None
    tfm = fall.to_case_path(a.fall, tfm, orte) if tfm and os.path.isfile(tfm) else None
    labels = fall.to_case_path(a.fall, a.labels, orte) if a.labels else None
    fall.save_orte(a.fall, orte)
    q = e["registration"]

    def change(d):
        for b in d.get("bilder", []):
            if b.get("modalitaet") != "PT": continue
            try: same = os.path.normpath(fall.resolve(a.fall, b["ordner"], orte)) == here
            except fall.FallError: same = False
            if not same: continue
            b["nach_bezug"] = tfm
            # passing the run means passing both checks: pet.run ends without writing when either fails
            b["guete"] = {"verfahren": "mattes_mi", "raster": "viewer", "fest": e["fixed_t1"], "ok": True,
                          "seitenpruefung": {"mi": q["mi"], "mi_gespiegelt": q["mi_gespiegelt"], "ok": True},
                          "kopf_fuss": {"mi_kopfstehend": q["mi_kopfstehend"], "ok": True}}
            pet = b.get("pet") or {}
            # side and label map too: pet_form gives them back, else the next start from the case fails its check
            pet["referenz"] = {"art": a.ref, "seite": a.side, "maske": labels or (pet.get("referenz") or {}).get("maske"),
                               "region": e["reference"]}
            b["pet"] = pet
            if not b.get("datum") and fall._DATE.match(e["day"] or ""): b["datum"] = e["day"]
            break
        else:
            return
        fall.stamp(d)
    fall.update(a.fall, change)


def _step(name):
    """Progress line for the workbench tab; run.sh prints the first one (export)."""
    print(f"## STEP {name}", flush=True)


def cmd_export(a):
    print(json.dumps(ex.export(a.run, a.out, a.fractions), indent=1))


def cmd_series(a):
    rows = se.scan(a.exam)
    for r in rows:
        print(f"{r['date']}  Serie {r['series_number']:>4}  {r['kind'] or '-':6}  {r['slice_thickness_mm']:.1f} mm  "
              f"{r['images']:>4} Bilder  {'geeignet' if r['suitable'] else 'ungeeignet'}")
    if a.json: json.dump([{k: v for k, v in r.items() if k != "dir"} for r in rows], open(a.json, "w"), indent=1)


def cmd_onto(a):
    mri, ct = geo.read_image(a.mri), geo.read_image(a.ct)
    t, q = rg.rigid(mri, ct)
    _, _, info = _common(a, mri, ct, t, q, a.out)
    json.dump(info, open(os.path.join(a.out, "overlay.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: info[k] for k in ("registration", "prescription_gy_rbe", "look_list")}, indent=1))


def cmd_viewer(a):
    if a.fall: _viewer_from_case(a)
    elif not all((a.cache, a.ct, a.ctframe, a.out)): raise SystemExit("viewer: CACHEF CT CTFRAME OUT, or --fall CASEDIR")
    log = json.load(open(os.path.join(a.cache, "log.json")))
    grid = sitk.ReadImage(os.path.join(a.cache, "ref_t1.nii.gz"))
    ct = geo.read_image(a.ct)
    os.makedirs(a.out, exist_ok=True)
    if a.synthseg and not a.labels:
        _step("synthseg")
        a.labels = ve.run_synthseg(os.path.join(a.cache, "ref_t1.nii.gz"), os.path.join(a.out, "synthseg.nii.gz"), a.synthseg)
    head = sitk.GetArrayFromImage(grid) > 0
    pts = geo.brain_points(geo.like(head.astype(np.uint8), grid), step=8)
    _step("register")
    t, q = rg.rigid(grid, ct, mask_points=pts)
    layers, mk, info = _common(a, grid, ct, t, q, a.out)
    exams = []
    todo = [(k, e) for k, e in sorted(log.items()) if isinstance(e, dict) and e.get("series") and "native" in e]
    for i, (key, e) in enumerate(todo, 1):
        print(f"## EXAM {i}/{len(todo)}", flush=True)
        track, day = key.split(" ", 1)
        row = {"track": track, "day": day, "native_mm": e["native"], "suitable": ly.exam_suitable(e["native"])}
        vol = os.path.join(a.cache, f"{track}_{day}.npy")
        if a.check and os.path.exists(vol):
            # each exam on its own: CT onto the exam as the viewer shows it, from the reference result; the
            # distance to the reference result is the exam's alignment error with respect to the CT
            img = geo.like(np.load(vol).astype(np.float32), grid)
            te, qe = rg.rigid(img, ct, initial=t, perturb=((0, 0, 0, 0, 0, 0),), mask_points=pts)
            rms, mx = geo.displacement_mm(t, te, pts)
            row.update(check_rms_mm=round(rms, 2), check_max_mm=round(mx, 2), check_ok=rms <= rg.ACCEPT_MM)
        exams.append(row)
        print(track, day, row)
    info["exams"] = exams
    _step("write")
    json.dump(info, open(os.path.join(a.out, "overlay.json"), "w"), indent=1, ensure_ascii=False)
    if a.viewer_data:
        ly.write_viewer(a.viewer_data, layers, mk, {"exams": exams, "look_list": info["look_list"],
                                                    "prescription_gy_rbe": info["prescription_gy_rbe"],
                                                    "ptv": info["ptv"], "margin": info["margin"],
                                                    "registration": q, "hinweis": info["hinweis"]})
    if a.kennzahlen:
        k = {"registration_spread_mm": q["spread_rms_mm"], "registration_metric": q["metric"],
             "exams": len(exams), "exams_unsuitable": sum(not r["suitable"] for r in exams),
             "exams_check_over_1_5mm": sum(1 for r in exams if r.get("check_ok") is False),
             "look_regions": len(info["look_list"]),
             "look_volume_ml": round(sum(r["volume_ml"] for r in info["look_list"]), 3)}
        os.makedirs(os.path.dirname(os.path.abspath(a.kennzahlen)), exist_ok=True)
        json.dump(k, open(a.kennzahlen, "w"), indent=1)
    if a.fall: _viewer_to_case(a, q, exams)
    print(json.dumps({k: info[k] for k in ("registration", "prescription_gy_rbe", "look_list")}, indent=1))


def cmd_pet(a):
    if a.fall: _pet_from_case(a)
    elif not all((a.pet, a.cache, a.viewer_data)): raise SystemExit("pet: PETDIR CACHEF VIEWERDATA, or --fall CASEDIR")
    a.ref = a.ref or "brain"
    e = pe.run(a.pet, a.cache, a.viewer_data, out=a.out, day=a.day, ref=a.ref, labels=a.labels, side=a.side,
               fixed_day=a.fixed_day, weight_kg=a.weight_kg, activity_mbq=a.activity_mbq, margin=a.mirror_margin,
               start_deg=tuple(float(x) for x in a.starts.split(",")))
    print(json.dumps({k: e[k] for k in ("day", "tracer", "fixed_t1", "reference", "registration")}, indent=1,
                     ensure_ascii=False))
    if a.fall: _pet_to_case(a, e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="plan_mc run folder to NIfTI in the CT frame")
    p.add_argument("run"); p.add_argument("out"); p.add_argument("--fractions", type=int, default=1)
    p.set_defaults(f=cmd_export)
    p = sub.add_parser("series", help="list the MRI series of an exam folder by kind")
    p.add_argument("exam"); p.add_argument("--json"); p.set_defaults(f=cmd_series)
    for name, f in (("onto", cmd_onto), ("viewer", cmd_viewer)):
        p = sub.add_parser(name)
        # viewer takes its paths from a case (--fall); then the positionals may be left out
        opt = {} if name == "onto" else {"nargs": "?", "default": ""}
        if name == "onto": p.add_argument("mri")
        else: p.add_argument("cache", help="cacheF folder of the viewer (log.json, ref_t1.nii.gz, *.npy)", **opt)
        p.add_argument("ct", help="planning CT: DICOM folder or NIfTI", **opt)
        p.add_argument("ctframe", help="output folder of export", **opt)
        p.add_argument("out", **opt)
        p.add_argument("--rtstruct"); p.add_argument("--ptv", help="ROI name of the PTV (default: found by type/name)")
        p.add_argument("--labels", help="SynthSeg label map of the MRI's T1")
        p.add_argument("--synthseg", help="SynthSeg command, when --labels is not given")
        p.add_argument("--prescription", type=float, help="total Gy(RBE); default the median 1.1 dose in the PTV")
        if name == "viewer":
            p.add_argument("--viewer-data", help="Verlauf-Daten folder of the viewer")
            p.add_argument("--no-check", dest="check", action="store_false", help="skip the check per exam")
            p.add_argument("--kennzahlen", help="kennzahlen.json path (numbers only)")
            p.add_argument("--fall", help="case folder with fall.json: every path from there, written back after the run")
        p.set_defaults(f=f)
    p = sub.add_parser("pet", help="PET series onto the viewer grid as a ratio layer, with the side check")
    p.add_argument("pet", nargs="?", default="", help="folder of the PET DICOM series (Modality PT)")
    p.add_argument("cache", nargs="?", default="", help="cacheF folder of the viewer (log.json, ref_t1.nii.gz, mask.npy, t1_*.npy)")
    p.add_argument("viewer_data", nargs="?", default="", help="Verlauf-Daten folder of the viewer: pet_<day>.js and pet.json land there")
    p.add_argument("--fall", help="case folder with fall.json: paths and reference from there, the checks written back")
    p.add_argument("--serie", help="with --fall: the PET series of the case (default: the newest)")
    p.add_argument("--out", help="folder for the ratio NIfTI, the transform and the run's JSON")
    p.add_argument("--day", help="date of the PET, YYYY-MM-DD (default: from the header)")
    p.add_argument("--fixed-day", help="date of the 3D T1 to register on (default: the nearest in time)")
    p.add_argument("--ref", choices=("brain", "striatum", "contralateral"), help="default brain, or the case's")
    p.add_argument("--labels", help="SynthSeg label map of the viewer's reference T1 (for --ref striatum)")
    p.add_argument("--side", choices=("links", "rechts"), help="for --ref contralateral: the side NOT to use")
    p.add_argument("--weight-kg", type=float); p.add_argument("--activity-mbq", type=float)
    p.add_argument("--mirror-margin", type=float, default=pe.MIRROR_MARGIN,
                   help="how much better (relative MI) the PET as read must fit than flipped or upside down")
    p.add_argument("--starts", default=",".join(f"{d:g}" for d in pe.START_DEG),
                   help="start rotations about x and z in degrees, comma separated; each pair is one start")
    p.set_defaults(f=cmd_pet)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
