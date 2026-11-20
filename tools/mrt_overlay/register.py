"""Rigid CT to MRI registration with Mattes mutual information, and its quality measures.

The transform maps points of the fixed image (the MRI) to the moving image (the CT), so
sitk.Resample(ct_layer, mri, transform) puts a CT frame layer onto the MRI grid.
"""
import numpy as np
import SimpleITK as sitk

from . import geometry as geo

ACCEPT_MM = 1.5            # target: registration error at most 1.5 mm
SPREAD_TOL = 0.005         # starts within 0.5 % of the best metric count as the same optimum (phantom: those lay
                           # 0.3 to 0.6 mm from the truth, the starts 0.7 % and more off lay 1.4 mm and more off)


def _prep_ct(ct):
    # clamp the CT so that metal and the couch do not dominate the joint histogram
    return sitk.Clamp(sitk.Cast(ct, sitk.sitkFloat32), sitk.sitkFloat32, -1000.0, 2000.0)


def _one(fixed, moving, init, levels=((4, 2.0), (2, 1.0), (1, 0.0)), iterations=200, seed=1):
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(50)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.10, seed)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=iterations,
                                               relaxationFactor=0.6, gradientMagnitudeTolerance=1e-8)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([s for s, _ in levels])
    r.SetSmoothingSigmasPerLevel([g for _, g in levels])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(init, inPlace=False)
    t = r.Execute(fixed, moving)
    # the final value on all voxels, so starts compare without the sampling noise
    e = sitk.ImageRegistrationMethod()
    e.SetMetricAsMattesMutualInformation(50)
    e.SetInterpolator(sitk.sitkLinear)
    e.SetInitialTransform(t)
    return t, float(e.MetricEvaluate(fixed, moving))


def _euler(t):
    """An Euler3DTransform equal to a composite of rigid transforms."""
    e = sitk.Euler3DTransform()
    if isinstance(t, sitk.CompositeTransform) and t.GetNumberOfTransforms() == 1: t = t.GetNthTransform(0)
    try:
        e.SetMatrix(t.GetMatrix()); e.SetTranslation(t.GetTranslation()); e.SetCenter(t.GetCenter())
        return e
    except (AttributeError, RuntimeError):
        return t


def starts(fixed, moving, initial=None, perturb=((0, 0, 0, 0, 0, 0),)):
    """Initial transforms: from `initial`, else the geometric centres and the centres of mass, each perturbed by
    (rx, ry, rz in degrees, tx, ty, tz in mm)."""
    bases = [initial] if initial is not None else [
        sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), m)
        for m in (sitk.CenteredTransformInitializerFilter.GEOMETRY, sitk.CenteredTransformInitializerFilter.MOMENTS)]
    out = []
    for b in bases:
        b = _euler(b)
        for rx, ry, rz, tx, ty, tz in perturb:
            p = sitk.Euler3DTransform(b)
            p.SetRotation(b.GetAngleX() + np.radians(rx), b.GetAngleY() + np.radians(ry), b.GetAngleZ() + np.radians(rz))
            p.SetTranslation(np.add(b.GetTranslation(), (tx, ty, tz)).tolist())
            out.append(p)
    return out


PERTURB = ((0, 0, 0, 0, 0, 0), (4, -3, 3, 6, -6, 5), (-4, 3, -3, -6, 6, -5))


def rigid(mri, ct, initial=None, perturb=PERTURB, mask_points=None, levels=None):
    """Register CT (moving) to MRI (fixed). Returns (transform, quality dict).

    Quality: the metric of the best start, and the spread: the largest RMS distance over `mask_points` (brain
    points of the MRI; the whole image when None) between the best result and every other start that reached the
    same optimum. A spread near zero shows a reproducible optimum, not that the optimum is right."""
    f = sitk.Cast(mri, sitk.sitkFloat32)
    m = _prep_ct(ct)
    kw = {"levels": levels} if levels else {}
    results = []
    for i, s in enumerate(starts(f, m, initial, perturb)):
        try: results.append(_one(f, m, s, seed=i + 1, **kw))
        except RuntimeError: continue
    if not results: raise SystemExit("registration failed from every start")
    results.sort(key=lambda r: r[1])
    best_t, best_v = results[0]
    if mask_points is None:
        sx, sy, sz = mri.GetSize()
        g = [np.linspace(0, n - 1, 6) for n in (sx, sy, sz)]
        mask_points = np.array([mri.TransformContinuousIndexToPhysicalPoint((float(a), float(b), float(c)))
                                for a in g[0] for b in g[1] for c in g[2]])
    same = [t for t, v in results[1:] if v <= best_v * (1 - SPREAD_TOL)] if best_v < 0 else []
    spread = max((geo.displacement_mm(best_t, t, mask_points)[0] for t in same), default=float("nan"))
    lost = len(results) - 1 - len(same)
    return best_t, {"metric": best_v, "starts": len(results), "starts_same_optimum": len(same) + 1,
                    "starts_elsewhere": lost, "spread_rms_mm": spread}


def metric(mri, ct, transform):
    """Mattes MI of CT on MRI at `transform`, on all voxels (lower is better)."""
    e = sitk.ImageRegistrationMethod()
    e.SetMetricAsMattesMutualInformation(50)
    e.SetInterpolator(sitk.sitkLinear)
    e.SetInitialTransform(transform)
    return float(e.MetricEvaluate(sitk.Cast(mri, sitk.sitkFloat32), _prep_ct(ct)))


def gold_from_points(src, dst):
    """Least squares rigid transform (Kabsch) that maps src points onto dst points."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    t = sitk.Euler3DTransform()
    t.SetMatrix(rot.ravel().tolist()); t.SetTranslation((cd - rot @ cs).tolist())
    return t
