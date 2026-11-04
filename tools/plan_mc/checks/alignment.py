"""Alignment check in TOPAS: does a spot land where the IEC model (iec.py) says, for several gantry, couch and patient
position settings?

    python3 alignment.py write OUTDIR      # TOPAS cases, one per setting (100 MeV, 20000 protons, a few seconds each)
    python3 alignment.py check OUTDIR      # after the runs: distance of the dose axis from the predicted ray

A water box (120 mm, 2 mm voxels) is centred on an off-centre isocenter in DICOM coordinates; one spot at (60, -45) mm
at the isocenter plane. The check fits the dose-weighted axis of the voxels above 20 % of the maximum and compares it
with the ray of iec.spot_ray: pass when the axis lies within 0.5 mm of the ray at the isocenter plane, the angle is
below 2 mrad, and the Bragg peak lies downstream. Queue the cases through the workbench; nothing here starts TOPAS.
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plan_mc import beam_model, iec, plan as pl, topas_writer as tw  # noqa: E402

CASES = [(0, 0, "HFS"), (90, 0, "HFS"), (270, 0, "HFS"), (45, 310, "HFS"), (300, 240, "HFS"), (60, 0, "FFS"), (120, 20, "HFP")]
ISO = (25.0, -40.0, 60.0)
HALF, BIN, SPOT = 60.0, 2.0, (60.0, -45.0)


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


def read_csv(path, n):
    d = np.zeros((n, n, n))
    for line in open(path):
        if line.startswith("#") or not line.strip(): continue
        i, j, k, v = line.split(",")[:4]
        d[int(i), int(j), int(k)] = float(v)
    return d


def check(out):
    nb = int(2 * HALF / BIN)
    centres = np.asarray(ISO)[:, None] + (np.arange(nb) + 0.5) * BIN - HALF       # per axis, DICOM mm
    ok_all, rows = True, []
    for c in CASES:
        path = os.path.join(out, name(c) + "_dose.csv")
        if not os.path.exists(path): rows.append((name(c), "missing")); ok_all = False; continue
        d = read_csv(path, nb)
        idx = np.argwhere(d > 0.2 * d.max()); w = d[d > 0.2 * d.max()]
        pts = np.stack([centres[a][idx[:, a]] for a in range(3)], axis=1)
        mean = (pts * w[:, None]).sum(0) / w.sum()
        cov = ((pts - mean).T * w) @ (pts - mean) / w.sum()
        axis = np.linalg.eigh(cov)[1][:, -1]
        p, u = iec.spot_ray(*SPOT, c[0], c[1], ISO, position=c[2])
        if np.dot(axis, u) < 0: axis = -axis
        ang = 1000 * np.arccos(min(1.0, abs(np.dot(axis, u))))   # mrad
        # distance of the fitted axis from p (closest point of the axis to the predicted isocenter-plane point)
        q = mean + np.dot(p - mean, axis) * axis
        dist = np.linalg.norm(q - p)
        peak = np.stack([centres[a][np.unravel_index(d.argmax(), d.shape)[a]] for a in range(3)])
        downstream = np.dot(peak - p, u) > 0
        ok = dist < 0.5 and ang < 2.0 and downstream
        ok_all &= ok
        rows.append((name(c), f"{'pass' if ok else 'FAIL'} distance {dist:.2f} mm, angle {ang:.2f} mrad, peak downstream {downstream}"))
    for r in rows: print(*r)
    print("all pass" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("write", "check"): sys.exit(__doc__)
    if sys.argv[1] == "write": write(sys.argv[2])
    else: sys.exit(0 if check(sys.argv[2]) else 1)
