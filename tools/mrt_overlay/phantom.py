"""Synthetic head: CT, a pseudo T1 MRI with a known rigid transform, SynthSeg-like labels, and a plan_mc run.

Tissue classes become CT numbers on one side and T1 intensities on the other, so mutual information must find the
transform without a shared intensity scale. Asymmetric parts (nose, one larger eye, a thicker skull plate) fix the
rotation.
"""
import os

import numpy as np
import SimpleITK as sitk

from . import geometry as geo

# class: (CT number, T1 intensity, SynthSeg label)
CLASSES = {"air": (-1000, 0, 0), "scalp": (40, 700, 0), "skull": (1100, 60, 0), "brain": (35, 520, 2),
           "white": (30, 640, 41), "ventricle": (5, 120, 4), "ventricle_r": (5, 120, 43), "third": (5, 120, 14),
           "eye": (10, 160, 0)}
TRUE_ROT_DEG = (3.0, -2.0, 5.0)
TRUE_SHIFT_MM = (6.0, -4.0, 3.0)


def _ell(x, y, z, c, r):
    return ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2 <= 1.0


def classes_at(x, y, z):
    """Class name per point (mm, LPS, head centre at the origin, nose towards -y)."""
    out = np.full(x.shape, "air", dtype=object)
    head = _ell(x, y, z, (0, 0, 0), (72, 90, 80))
    out[head] = "scalp"
    out[_ell(x, y, z, (0, 0, 0), (66, 84, 74)) | (_ell(x, y, z, (0, 0, 40), (60, 70, 40)) & head)] = "skull"
    brain = _ell(x, y, z, (0, 0, 2), (60, 78, 68))
    out[brain] = "brain"
    out[brain & _ell(x, y, z, (0, 6, 8), (44, 58, 40))] = "white"
    out[_ell(x, y, z, (-11, 4, 12), (6, 26, 9))] = "ventricle"
    out[_ell(x, y, z, (11, 4, 12), (6, 26, 9))] = "ventricle_r"
    out[_ell(x, y, z, (0, 6, -4), (2.5, 10, 7))] = "third"
    out[_ell(x, y, z, (-30, -78, -20), (11, 11, 11))] = "eye"
    out[_ell(x, y, z, (30, -78, -20), (14, 14, 14))] = "eye"
    out[(np.abs(x) < 9) & (y < -86) & (y > -104) & (z > -42) & (z < -6)] = "scalp"   # nose
    return out


def _grid_points(img):
    nx, ny, nz = img.GetSize()
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    o, s = np.array(img.GetOrigin()), np.array(img.GetSpacing())
    d = np.array(img.GetDirection()).reshape(3, 3)
    idx = np.stack([i, j, k], -1) * s
    pts = idx @ d.T + o
    return pts[..., 0], pts[..., 1], pts[..., 2]


def _render(img, column, transform=None):
    x, y, z = _grid_points(img)
    if transform is not None:
        flat = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
        flat = _apply(transform, flat)
        x, y, z = (flat[:, k].reshape(x.shape) for k in range(3))
    cls = classes_at(x, y, z)
    lut = {n: v[column] for n, v in CLASSES.items()}
    return np.vectorize(lut.get, otypes=[float])(cls)


def _apply(t, pts):
    m = np.array(t.GetMatrix()).reshape(3, 3)
    c, tr = np.array(t.GetCenter()), np.array(t.GetTranslation())
    return (pts - c) @ m.T + c + tr


def true_transform():
    """MRI point -> CT point."""
    t = sitk.Euler3DTransform()
    t.SetRotation(*np.radians(TRUE_ROT_DEG)); t.SetTranslation(TRUE_SHIFT_MM)
    return t


def make(ct_spacing=(2.0, 2.0, 2.5), mr_spacing=(1.6, 1.6, 2.0), seed=0):
    """(ct, mri, labels on the MRI grid, true MRI->CT transform)."""
    rng = np.random.default_rng(seed)
    ct = sitk.Image([int(round(170 / ct_spacing[0])), int(round(220 / ct_spacing[1])), int(round(180 / ct_spacing[2]))],
                    sitk.sitkFloat32)
    ct.SetSpacing(ct_spacing); ct.SetOrigin((-85.0, -110.0, -90.0))
    ct_arr = _render(ct, 0) + rng.normal(0, 8, size=ct.GetSize()[::-1])
    ct = geo.like(ct_arr.astype(np.float32), ct)
    mri = sitk.Image([int(round(160 / mr_spacing[0])), int(round(200 / mr_spacing[1])), int(round(170 / mr_spacing[2]))],
                     sitk.sitkFloat32)
    mri.SetSpacing(mr_spacing); mri.SetOrigin((-78.0, -96.0, -80.0))
    t = true_transform()
    t1 = _render(mri, 1, t)
    zz = np.linspace(-1, 1, t1.shape[0])[:, None, None]
    t1 = t1 * (1 + 0.12 * zz) + rng.normal(0, 15, size=t1.shape)   # a smooth bias field and noise
    labels = _render(mri, 2, t).astype(np.int16)
    mri = geo.like(np.clip(t1, 0, None).astype(np.float32), mri)
    return ct, mri, geo.like(labels, mri), t


def write_run(folder, centre_mm=(-11.0, 4.0, 26.0), sigma_mm=10.0, dose_gy=2.0):
    """A plan_mc-like run folder on a dose grid that is not the CT grid: a Gaussian dose blob (per fraction) with a
    LETd rising towards one side, as letd.npy, rbe_dose_1_1.npy and rbe_dose_1_1.dcm."""
    grid = sitk.Image([60, 70, 56], sitk.sitkFloat32)
    grid.SetSpacing((2.5, 2.5, 3.0)); grid.SetOrigin((-75.0, -87.0, -84.0))
    x, y, z = _grid_points(grid)
    r2 = (x - centre_mm[0]) ** 2 + (y - centre_mm[1]) ** 2 + (z - centre_mm[2]) ** 2
    phys = dose_gy * np.exp(-r2 / (2 * sigma_mm ** 2))
    letd = 1.5 + 0.12 * np.clip(x - centre_mm[0] + 20, 0, None)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "letd.npy"), letd.astype(np.float32))
    np.save(os.path.join(folder, "rbe_dose_1_1.npy"), (1.1 * phys).astype(np.float32))
    geo.write_rtdose(os.path.join(folder, "rbe_dose_1_1.dcm"), 1.1 * phys, grid.GetOrigin(), grid.GetSpacing())
    return grid
