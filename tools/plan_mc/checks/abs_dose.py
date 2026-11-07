"""Absolute dose reference (part A3): TOPAS dose per MU in a reference field, against a measured value, gives the
beam model's dose_scale.

    python3 abs_dose.py write OUTDIR MODEL.json [--energy 150] [--size 100] [--spot-spacing 2.5] [--depth 20]
                              [--mu-per-spot 0.01] [--histories 2000000]
    python3 abs_dose.py check OUTDIR --measured GY [--update MODEL.json]

Reference field: one energy layer, square of `size` mm with spots every `spot-spacing` mm, each spot `mu-per-spot` MU,
gantry 0, into a water box whose surface sits `depth` mm upstream of the isocenter (the reference point). The dose is
the mean over the central 10 x 10 mm and 1 mm along the beam at the reference depth. The measured value must belong to
the same field (energy, size, spacing, MU, depth); with it, dose_scale = measured / TOPAS, which the start script then
applies to every plan. A model without a measured value keeps dose_scale 1 and gives dose per particles_per_mu only.
Queue the case through the workbench (wbqueue.py); nothing here starts TOPAS.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plan_mc import beam_model, plan as pl, topas_io, topas_writer as tw  # noqa: E402

LAT_BINS, LAT_MM = 31, 310.0          # 10 mm lateral bins, the central one centred on the beam axis
DEPTH_MM, DEPTH_BIN = 300.0, 0.5


def field(size, spacing, mu):
    k = int(round(size / spacing / 2))
    pos = [spacing * (i - k) for i in range(2 * k + 1)]
    xs = [x for x in pos for _ in pos]; ys = [y for _ in pos for y in pos]
    return xs, ys, [mu] * len(xs)


def write(out, model_path, energy, size, spacing, depth, mu, histories):
    out = os.path.abspath(out); os.makedirs(out, exist_ok=True)
    model = beam_model.BeamModel.load(model_path)
    xs, ys, ws = field(size, spacing, mu)
    # gantry 0, HFS: the beam runs along DICOM +y (checks/ct_range.py); the surface sits at y = -depth
    box = {"center": (0.0, -depth + DEPTH_MM / 2, 0.0), "half": (LAT_MM / 2, DEPTH_MM / 2, LAT_MM / 2),
           "bins": (LAT_BINS, int(DEPTH_MM / DEPTH_BIN), LAT_BINS)}
    pat = os.path.join(out, "patient.txt")
    open(pat, "w").write(tw.patient_block("box", "Patient", box=box, let=False))
    f = pl.Field("absdose", 1, 0.0, 0.0, (0.0, 0.0, 0.0), [pl.Layer(energy, xs, ys, ws)], "MU")
    text, meta = tw.field_file(pl.Plan("absdose", "HFS", [f]), f, model, {}, pat, histories, out_name="absdose")
    open(os.path.join(out, "absdose.txt"), "w").write(text)
    json.dump({"model": os.path.abspath(model_path), "energy": energy, "size_mm": size, "spot_spacing_mm": spacing,
               "depth_mm": depth, "mu_per_spot": mu, "spots": len(xs), "mu_total": mu * len(xs), "field": meta},
              open(os.path.join(out, "absdose.json"), "w"), indent=1)
    print(f"{len(xs)} spots, {mu * len(xs):g} MU, {histories} histories: {os.path.join(out, 'absdose.txt')}")


def reference_dose(out):
    """TOPAS dose [Gy] at the reference point for the MU of the field, before dose_scale."""
    c = json.load(open(os.path.join(out, "absdose.json")))
    _, a = topas_io.read(os.path.join(out, "absdose_dose.bin"))
    s = np.asarray(a["Sum"], float) * c["field"]["particles"] / c["field"]["histories"]   # [x, y, z]
    mid = LAT_BINS // 2
    depth = DEPTH_BIN * (np.arange(s.shape[1]) + 0.5)
    sel = np.abs(depth - c["depth_mm"]) <= 0.5 + 1e-9
    return float(s[mid, sel, mid].mean()), c


def check(out, measured, update=None):
    d, c = reference_dose(os.path.abspath(out))
    print(f"TOPAS {d:.5g} Gy for {c['mu_total']:g} MU at {c['depth_mm']:g} mm ({d / c['mu_total']:.5g} Gy/MU)")
    if measured is None: return d
    scale = measured / d
    print(f"measured {measured:.5g} Gy: dose_scale = {scale:.5f}")
    if update:
        m = json.load(open(update))
        m["dose_scale"] = scale
        m.setdefault("sources", {})["dose_scale"] = (f"abs_dose.py: {c['energy']:g} MeV, {c['size_mm']:g} mm field, "
                                                     f"{c['depth_mm']:g} mm depth, measured {measured:g} Gy")
        json.dump(m, open(update, "w"), indent=1)
        print(f"written to {update}")
    return scale


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.add_argument("out"); w.add_argument("model")
    w.add_argument("--energy", type=float, default=150.0); w.add_argument("--size", type=float, default=100.0)
    w.add_argument("--spot-spacing", type=float, default=2.5); w.add_argument("--depth", type=float, default=20.0)
    w.add_argument("--mu-per-spot", type=float, default=0.01); w.add_argument("--histories", type=int, default=2000000)
    k = sub.add_parser("check"); k.add_argument("out"); k.add_argument("--measured", type=float)
    k.add_argument("--update")
    a = ap.parse_args()
    if a.cmd == "write":
        write(a.out, a.model, a.energy, a.size, a.spot_spacing, a.depth, a.mu_per_spot, a.histories)
    else:
        check(a.out, a.measured, a.update)
