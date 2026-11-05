"""Alignment check in TOPAS: does a spot land where the IEC model (iec.py) says, for several gantry, couch and patient
position settings?

    python3 alignment.py write OUTDIR      # TOPAS cases, one per setting (100 MeV, 20000 protons, 2 to 3 min each)
    python3 alignment.py check OUTDIR      # after the runs: distance of the dose axis from the predicted ray

A water box (160 mm, 2 mm voxels) is centred on an off-centre isocenter in DICOM coordinates; one spot at (20, -15) mm
at the isocenter plane (inside the box along the whole path in every case). The check fits the beam axis through the
dose centroids of slabs across the beam (slab_axis) and compares it with the ray of iec.spot_ray: pass when the axis
lies within 0.5 mm of the ray at the isocenter plane, the angle is below 5 mrad (a wrong sign or axis gives degrees),
and the Bragg peak lies beyond the dose centroid along the beam.
Queue the cases through the workbench; nothing here starts TOPAS.
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plan_mc import beam_model, iec, plan as pl, topas_io, topas_writer as tw  # noqa: E402

CASES = [(0, 0, "HFS"), (90, 0, "HFS"), (270, 0, "HFS"), (45, 310, "HFS"), (300, 240, "HFS"), (60, 0, "FFS"), (120, 20, "HFP")]
ISO = (25.0, -40.0, 60.0)
HALF, BIN, SPOT = 80.0, 2.0, (20.0, -15.0)


def name(c): return f"align_g{c[0]}_c{c[1]}_{c[2]}"


def write(out):
    os.makedirs(out, exist_ok=True)
    nb = int(2 * HALF / BIN)
    pat = os.path.join(os.path.abspath(out), "patient_box.txt")
    text = tw.patient_block("box", "Patient", box={"center": ISO, "half": (HALF,) * 3, "bins": (nb,) * 3}, let=False)
    text = text.replace('s:Sc/DoseMed/OutputType = "binary"', 's:Sc/DoseMed/OutputType = "csv"').replace('s:Sc/Dose/OutputType = "binary"', 's:Sc/Dose/OutputType = "csv"')
    open(pat, "w").write(text)
    m = beam_model.generic()
    for c in CASES:
        f = pl.Field(name(c), 1, c[0], c[1], ISO, [pl.Layer(100.0, [SPOT[0]], [SPOT[1]], [1.0])], "particles")
        t, _ = tw.field_file(pl.Plan("align", c[2], [f]), f, m, {}, pat, 20000, out_name=name(c))
        open(os.path.join(out, name(c) + ".txt"), "w").write(t)
    print(f"{len(CASES)} cases in {out}")


def slab_axis(a, centres, p, u, slab=2.0, margin=12.0):
    """Beam axis as the line through the dose centroids of slabs perpendicular to u, in the plateau between the
    entrance and the Bragg peak. A principal axis of the whole dose tilts by about 10 mrad when the beam enters the
    box obliquely, since the peak region then lies oblique too (seen on TOPAS and synthetic doses); slab centroids in
    the plateau stay on the axis within about 2 mrad. Returns (unit direction, distance of the line from p, slabs)."""
    X = np.stack(np.meshgrid(*centres, indexing="ij"), -1).reshape(-1, 3); w = a.ravel()
    keep = w > 0.01 * w.max(); X, w = X[keep], w[keep]
    t = (X - p) @ u
    ts, cs = [], []
    for s in np.arange(t.min() + margin, t[np.argmax(w)] - margin, slab):
        m = (t >= s) & (t < s + slab)
        if w[m].sum() <= 0: continue
        c = (X[m] * w[m, None]).sum(0) / w[m].sum(); ts.append((c - p) @ u); cs.append(c)
    if len(ts) < 2: return u * 0 + np.nan, np.inf, len(ts)
    coef = np.linalg.lstsq(np.stack([np.ones(len(ts)), ts], 1), np.array(cs), rcond=None)[0]   # c(t) = c0 + t * d
    d, c0 = coef[1] / np.linalg.norm(coef[1]), coef[0]
    return d, float(np.linalg.norm(c0 + np.dot(p - c0, d) * d - p)), len(ts)


def check(out):
    nb = int(2 * HALF / BIN)
    centres = np.asarray(ISO)[:, None] + (np.arange(nb) + 0.5) * BIN - HALF       # per axis, DICOM mm
    ok_all, rows_out = True, []
    for c in CASES:
        path = os.path.join(out, name(c) + "_dose.csv")
        if not os.path.exists(path): rows_out.append((name(c), "missing")); ok_all = False; continue
        with open(path) as fh: rows = sum(1 for l in fh if l.strip() and not l.startswith("#"))
        if rows != nb ** 3:
            rows_out.append((name(c), f"incomplete ({rows} of {nb ** 3} rows): run still writing?")); ok_all = False; continue
        d = topas_io.read(path)[1]["Sum"]
        p, u = iec.spot_ray(*SPOT, c[0], c[1], ISO, position=c[2])
        axis, dist, n = slab_axis(d, centres, p, u)
        ang = 1000 * np.arccos(min(1.0, float(np.dot(axis, u))))   # mrad
        pts = np.stack(np.meshgrid(*centres, indexing="ij"), -1).reshape(-1, 3)
        mean = (pts * d.reshape(-1, 1)).sum(0) / d.sum()
        peak = pts[d.argmax()]
        downstream = np.dot(peak - mean, u) > 0   # Bragg peak beyond the dose centroid: beam runs along +u
        ok = dist < 0.5 and ang < 5.0 and downstream and n >= 10
        ok_all &= ok
        rows_out.append((name(c), f"{'pass' if ok else 'FAIL'} distance {dist:.2f} mm, angle {ang:.2f} mrad, peak downstream {downstream}"))
    for r in rows_out: print(*r)
    print("all pass" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("write", "check"): sys.exit(__doc__)
    if sys.argv[1] == "write": write(sys.argv[2])
    else: sys.exit(0 if check(sys.argv[2]) else 1)
