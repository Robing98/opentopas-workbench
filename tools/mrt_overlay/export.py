"""plan_mc run folder to NIfTI in the frame of the planning CT: LETd, physical dose, RBE weighted doses.

plan_mc writes letd.npy and rbe_dose_*.npy [z, y, x] per fraction on its scoring grid (the TPS dose grid, or the
CT), and the same grid as RT Dose in rbe_dose_1_1.dcm. That RT Dose carries the geometry; the CT need not share
the grid, so nothing here assumes it does.
"""
import glob
import json
import os

import numpy as np
import SimpleITK as sitk

from . import geometry as geo

BAHN_SLOPE = 0.11          # 1 + 0.11 LETd: line through RBE 1.22 at 2 and about 1.56 at 5 keV/um (Bahn 2020)
LET_FLOOR = 0.05           # LETd shown only where the physical dose is at least 5 % of its maximum
CONSTANT = 1.1


def bahn_linear(letd):
    """Linear RBE approximation from the two values of Bahn et al. 2020 (IJROBP 107:571). An approximation for
    orientation, not a fitted model."""
    return 1.0 + BAHN_SLOPE * np.asarray(letd, float)


def export(run, out, fractions=1, body=None):
    """Write the layers of a plan_mc run as NIfTI (total over `fractions`). Returns the summary dict."""
    ref_path = os.path.join(run, "rbe_dose_1_1.dcm")
    if not os.path.exists(ref_path): raise SystemExit(f"{ref_path} missing: run plan_mc rbe on this run first")
    ref = geo.read_rtdose(ref_path)
    letd = np.load(os.path.join(run, "letd.npy")).astype(np.float32)
    eff11 = np.load(os.path.join(run, "rbe_dose_1_1.npy")).astype(np.float64)
    if letd.shape != eff11.shape or letd.shape[::-1] != ref.GetSize():
        raise SystemExit(f"grids differ: letd {letd.shape}, rbe_dose_1_1 {eff11.shape}, RT Dose {ref.GetSize()[::-1]}")
    phys = eff11 / CONSTANT
    inside = np.ones(phys.shape, bool) if body is None else body
    dmax = float(phys[inside].max())
    valid = inside & (phys >= LET_FLOOR * dmax)
    os.makedirs(out, exist_ok=True)
    layers = {"letd": np.where(valid, letd, 0.0), "dose_phys": phys * fractions}
    for p in sorted(glob.glob(os.path.join(run, "rbe_dose_*.npy"))):
        tag = os.path.basename(p)[len("rbe_dose_"):-4]
        layers[f"dose_rbe_{tag}"] = np.load(p).astype(np.float64) * fractions
    # the same rule as plan_mc: model RBE only where the LETd is trusted, 1.1 elsewhere
    layers["dose_rbe_bahn_linear"] = np.where(valid, bahn_linear(letd), CONSTANT) * phys * fractions
    for name, arr in layers.items():
        sitk.WriteImage(geo.like(arr.astype(np.float32), ref), os.path.join(out, f"{name}.nii.gz"))
    summary = {"fractions": fractions, "grid_size": list(ref.GetSize()), "spacing_mm": list(ref.GetSpacing()),
               "dose_max_phys_gy": dmax * fractions, "letd_floor_fraction": LET_FLOOR,
               "letd_shown_voxels": int(valid.sum()), "layers": sorted(layers),
               "note": "dose_rbe_bahn_linear: 1 + 0.11 LETd, Naeherung durch zwei Werte aus Bahn 2020, nur Forschung; "
                       "klinisch gilt 1,1."}
    json.dump(summary, open(os.path.join(out, "export.json"), "w"), indent=1)
    return summary
