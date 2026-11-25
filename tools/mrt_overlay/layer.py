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


PTV_IN_ISO50 = 0.90               # a PTV with less of its volume inside the 50 % isodose is most likely the wrong ROI


def ptv_check(ptv, dose, reference):
    """Share of the PTV inside the 50 % isodose of `reference` (Gy). A PTV the plan treats lies almost wholly in it;
    a low share points at the wrong ROI or a broken contour fill."""
    n = int(ptv.sum())
    if not n or reference <= 0: return {"anteil_iso50": None, "ok": False, "voxel": n}
    share = float((dose[ptv] >= 0.5 * reference).sum()) / n
    return {"anteil_iso50": round(share, 3), "ok": share >= PTV_IN_ISO50, "voxel": n}


def _edge2d(m):
    return m & ~ndi.binary_erosion(m)


def ptv_sheet(t1, ptv, dose, prescription, grid, path, title=""):
    """PNG for checking the PTV by eye: the T1 with the PTV outline (cyan) and the 95 % isodose outline (orange),
    axial, coronal and sagittal through the centre of the PTV. Arrays are [z, y, x] on `grid`, turned to LPS first
    as in pet.control_sheet, so that the patient's right is on the image's left."""
    from PIL import Image, ImageDraw
    lps = lambda a: sitk.GetArrayFromImage(sitk.DICOMOrient(geo.like(np.asarray(a, np.float32), grid), "LPS"))
    sp = sitk.DICOMOrient(grid, "LPS").GetSpacing()
    t1, ptv, iso = lps(t1), lps(ptv) > 0.5, lps(dose) >= 0.95 * prescription
    z, y, x = (int(round(c)) for c in np.mean(np.nonzero(ptv), 1)) if ptv.any() else (s // 2 for s in t1.shape)
    cuts = [(lambda a: a[z], (sp[1], sp[0]), False, "R L"), (lambda a: a[:, y], (sp[2], sp[0]), True, "R L"),
            (lambda a: a[:, :, x], (sp[2], sp[1]), True, "A P")]
    px, gap, head = 300, 6, 24
    panels = []
    for cut, (sh, sw), flip, marks in cuts:
        g, p, d = cut(t1), cut(ptv), cut(iso)
        if flip: g, p, d = g[::-1], p[::-1], d[::-1]
        lo, hi = np.percentile(g[g > 0], (1, 99.5)) if (g > 0).any() else (0.0, 1.0)
        rgb = np.repeat((np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
        rgb[_edge2d(d)] = (255, 140, 0)
        rgb[_edge2d(p)] = (0, 230, 255)
        im = Image.fromarray(rgb)
        h, w = g.shape
        panels.append((im.resize((max(1, round(w * sw * px / (h * sh))), px), Image.NEAREST), marks.split()))
    sheet = Image.new("RGB", (sum(p.width for p, _ in panels) + gap * (len(panels) + 1), head + px + 2 * gap), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 6), title, fill=(230, 230, 230))
    left = gap
    for p, (a, b) in panels:
        sheet.paste(p, (left, head + gap))
        d.text((left + 4, head + gap + px // 2), a, fill=(255, 255, 0))
        d.text((left + p.width - 12, head + gap + px // 2), b, fill=(255, 255, 0))
        left += p.width + gap
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sheet.save(path)
    return path


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
