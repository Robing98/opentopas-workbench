"""Layers on an MRI grid: resampling from the CT frame, the look-list, and the files of the MRI viewer."""
import base64
import io
import json
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from . import geometry as geo

LET_STEPS = (2.0, 3.0, 5.0)        # colour thresholds of the LETd layer [keV/um]
LOOK_LET = 4.0                     # look-list: LETd at least 4 keV/um,
LOOK_DOSE = 0.5                    # dose at least 50 % of the prescription, inside the ventricle margin
LET_SCALE = 25.0                   # viewer byte = LETd x 25: 0.04 keV/um steps up to 10.2
THICK_MM = 3.0                     # series with thicker slices are marked unsuitable
MARK = {"margin": 1, "iso50": 2, "iso95": 4, "ptv": 8, "look": 16}


def resample(img, grid, transform, label=False):
    interp = sitk.sitkNearestNeighbor if label else sitk.sitkLinear
    out = sitk.Resample(img, grid, transform, interp, 0.0, img.GetPixelID() if label else sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out)


def to_grid(ct_frame, grid, transform, ptv=None):
    """{name: array on `grid`} of the CT frame layers in folder `ct_frame` (NIfTI of export.py) and the PTV mask."""
    out = {}
    for name in sorted(os.listdir(ct_frame)):
        if name.endswith(".nii.gz"):
            out[name[:-7]] = resample(sitk.ReadImage(os.path.join(ct_frame, name)), grid, transform)
    if ptv is not None:
        out["ptv"] = resample(ptv, grid, transform, label=True).astype(bool)
    return out


def shell(mask):
    """The outer voxel layer of a 3D region: it shows as an outline in every plane that cuts it."""
    return mask & ~ndi.binary_erosion(mask)


def look_list(letd, dose, margin, prescription, spacing, min_voxels=3):
    """Connected regions with LETd >= 4 keV/um, dose >= 50 % of the prescription and inside the margin.
    Slices count from 1 along the third grid axis, as the viewer numbers them."""
    hit = (letd >= LOOK_LET) & (dose >= LOOK_DOSE * prescription) & margin
    lab, n = ndi.label(hit, structure=np.ones((3, 3, 3)))
    vox = float(np.prod(spacing)) / 1000.0
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab[sl] == i
        if m.sum() < min_voxels: continue
        z, y, x = (np.nonzero(m)[k] + sl[k].start for k in range(3))
        out.append({"volume_ml": round(float(m.sum()) * vox, 3), "slices": [int(z.min()) + 1, int(z.max()) + 1],
                    "centre_voxel_xyz": [round(float(x.mean()), 1), round(float(y.mean()), 1), round(float(z.mean()), 1)],
                    "letd_max": round(float(letd[sl][m].max()), 2), "dose_mean_gy": round(float(dose[sl][m].mean()), 2)})
    out.sort(key=lambda r: -r["volume_ml"])
    return out, hit


def marks(layers, margin, prescription, hit, rbe_key="dose_rbe_1_1"):
    """Bit mask per voxel: margin, 50 % and 95 % isodose shells, PTV shell, look-list regions."""
    d = layers[rbe_key]
    m = np.zeros(d.shape, np.uint8)
    m[margin] |= MARK["margin"]
    m[shell(d >= 0.5 * prescription)] |= MARK["iso50"]
    m[shell(d >= 0.95 * prescription)] |= MARK["iso95"]
    if "ptv" in layers: m[shell(layers["ptv"])] |= MARK["ptv"]
    m[hit] |= MARK["look"]
    return m


def montage_js(vol_u8, key, day="all", cols=12):
    """The viewer's volume format: all axial slices side by side in one lossless PNG, wrapped as MRT.vol(...)."""
    from PIL import Image
    nz, ny, nx = vol_u8.shape
    rows = -(-nz // cols)
    m = np.zeros((rows * ny, cols * nx), np.uint8)
    for z in range(nz):
        m[(z // cols) * ny:(z // cols + 1) * ny, (z % cols) * nx:(z % cols + 1) * nx] = vol_u8[z]
    b = io.BytesIO()
    Image.fromarray(m).save(b, "PNG", optimize=True)
    return f'MRT.vol("{key}","{day}","data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}");'


def write_viewer(folder, layers, mk, info, cols=12):
    """letd_all.js, marks_all.js and letd.json into the viewer's data folder."""
    os.makedirs(folder, exist_ok=True)
    letd_u8 = np.clip(np.rint(layers["letd"] * LET_SCALE), 0, 255).astype(np.uint8)
    open(os.path.join(folder, "letd_all.js"), "w").write(montage_js(letd_u8, "letd", cols=cols))
    open(os.path.join(folder, "marks_all.js"), "w").write(montage_js(mk, "marks", cols=cols))
    meta = {"let_scale": LET_SCALE, "let_steps": list(LET_STEPS), "marks": MARK, **info}
    json.dump(meta, open(os.path.join(folder, "letd.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return meta


def exam_suitable(native_spacing):
    return max(native_spacing) <= THICK_MM
