"""Command line: export, series, onto (any MRI), viewer (the grid of the MRI viewer). Research only, no diagnosis.

  python -m mrt_overlay.cli export RUN OUT --fractions N
  python -m mrt_overlay.cli series EXAMDIR
  python -m mrt_overlay.cli onto MRI CT CTFRAME OUT [--rtstruct RS] [--labels SEG] [--prescription GY]
  python -m mrt_overlay.cli viewer CACHEF CT CTFRAME OUT [--rtstruct RS] [--labels SEG] [--viewer-data DIR]
  python -m mrt_overlay.cli pet PETDIR CACHEF VIEWERDATA [--ref brain|striatum|contralateral] [--out DIR]
"""
import argparse
import json
import os
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
    log = json.load(open(os.path.join(a.cache, "log.json")))
    grid = sitk.ReadImage(os.path.join(a.cache, "ref_t1.nii.gz"))
    ct = geo.read_image(a.ct)
    os.makedirs(a.out, exist_ok=True)
    if a.synthseg and not a.labels:
        a.labels = ve.run_synthseg(os.path.join(a.cache, "ref_t1.nii.gz"), os.path.join(a.out, "synthseg.nii.gz"), a.synthseg)
    head = sitk.GetArrayFromImage(grid) > 0
    pts = geo.brain_points(geo.like(head.astype(np.uint8), grid), step=8)
    t, q = rg.rigid(grid, ct, mask_points=pts)
    layers, mk, info = _common(a, grid, ct, t, q, a.out)
    exams = []
    for key, e in sorted(log.items()):
        if not isinstance(e, dict) or not e.get("series") or "native" not in e: continue
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
    print(json.dumps({k: info[k] for k in ("registration", "prescription_gy_rbe", "look_list")}, indent=1))


def cmd_pet(a):
    e = pe.run(a.pet, a.cache, a.viewer_data, out=a.out, day=a.day, ref=a.ref, labels=a.labels, side=a.side,
               fixed_day=a.fixed_day, weight_kg=a.weight_kg, activity_mbq=a.activity_mbq, margin=a.mirror_margin)
    print(json.dumps({k: e[k] for k in ("day", "tracer", "fixed_t1", "reference", "registration")}, indent=1,
                     ensure_ascii=False))


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
        if name == "onto": p.add_argument("mri")
        else: p.add_argument("cache", help="cacheF folder of the viewer (log.json, ref_t1.nii.gz, *.npy)")
        p.add_argument("ct", help="planning CT: DICOM folder or NIfTI")
        p.add_argument("ctframe", help="output folder of export")
        p.add_argument("out")
        p.add_argument("--rtstruct"); p.add_argument("--ptv", help="ROI name of the PTV (default: found by type/name)")
        p.add_argument("--labels", help="SynthSeg label map of the MRI's T1")
        p.add_argument("--synthseg", help="SynthSeg command, when --labels is not given")
        p.add_argument("--prescription", type=float, help="total Gy(RBE); default the median 1.1 dose in the PTV")
        if name == "viewer":
            p.add_argument("--viewer-data", help="Verlauf-Daten folder of the viewer")
            p.add_argument("--no-check", dest="check", action="store_false", help="skip the check per exam")
            p.add_argument("--kennzahlen", help="kennzahlen.json path (numbers only)")
        p.set_defaults(f=f)
    p = sub.add_parser("pet", help="PET series onto the viewer grid as a ratio layer, with the side check")
    p.add_argument("pet", help="folder of the PET DICOM series (Modality PT)")
    p.add_argument("cache", help="cacheF folder of the viewer (log.json, ref_t1.nii.gz, mask.npy, t1_*.npy)")
    p.add_argument("viewer_data", help="Verlauf-Daten folder of the viewer: pet_<day>.js and pet.json land there")
    p.add_argument("--out", help="folder for the ratio NIfTI, the transform and the run's JSON")
    p.add_argument("--day", help="date of the PET, YYYY-MM-DD (default: from the header)")
    p.add_argument("--fixed-day", help="date of the 3D T1 to register on (default: the nearest in time)")
    p.add_argument("--ref", choices=("brain", "striatum", "contralateral"), default="brain")
    p.add_argument("--labels", help="SynthSeg label map of the viewer's reference T1 (for --ref striatum)")
    p.add_argument("--side", choices=("links", "rechts"), help="for --ref contralateral: the side NOT to use")
    p.add_argument("--weight-kg", type=float); p.add_argument("--activity-mbq", type=float)
    p.add_argument("--mirror-margin", type=float, default=pe.MIRROR_MARGIN,
                   help="how much better (relative MI) the unflipped PET must fit than the flipped one")
    p.set_defaults(f=cmd_pet)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.exit(main())
