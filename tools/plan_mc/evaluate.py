"""Compare two dose distributions on the same grid (part E): gamma index, DVH figures per structure, profiles.

    python3 evaluate.py REF EVAL OUT.json [--matrad FILE.mat] [--dta 2 --dd 2 --cutoff 10]

REF and EVAL: RT Dose files (synthetic.read_rtdose) or .npy cubes [z, y, x] in Gy with --origin/--spacing. With
--matrad the structures come from that matRad file (matrad.read_cst on its CT grid, which the doses must share).

Gamma (Low et al. 1998): global, the dose criterion is a share of the maximum reference dose; points below `cutoff`
per cent of that maximum are not evaluated. The search runs over offsets on a grid of `step` (default a third of the
DTA) up to `search` times the DTA (default 1.5), with the evaluated dose linearly interpolated at each offset; the
minimum found is an upper bound of the true gamma that is tight to about step/DTA/2 in gamma units.
DVH figures: D98, D95, D50, D2 (dose that the hottest x per cent of the volume reach), mean, V95 and V107 relative to a
prescription (default the median reference dose in the first TARGET).
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gamma(ref, ev, spacing, dta=2.0, dd=2.0, cutoff=10.0, step=None, search=1.5, norm=None):
    """Gamma map [z, y, x] (nan where not evaluated) and the pass rate in per cent. spacing (dx, dy, dz) in mm; dd and
    cutoff in per cent of `norm` (default max of ref)."""
    from scipy.ndimage import map_coordinates
    ref = np.asarray(ref, float); ev = np.asarray(ev, float)
    if ref.shape != ev.shape: raise ValueError(f"grids differ: {ref.shape} vs {ev.shape}")
    norm = float(ref.max() if norm is None else norm)
    dose_tol = dd / 100.0 * norm
    sel = ref >= cutoff / 100.0 * norm
    idx = np.array(np.nonzero(sel), float)          # rows z, y, x
    r = ref[sel]
    step = dta / 3.0 if step is None else step
    n = int(np.ceil(search * dta / step))
    g = np.arange(-n, n + 1) * step
    sz = np.array([spacing[2], spacing[1], spacing[0]], float)
    best = np.full(r.shape, np.inf)
    for oz in g:
        for oy in g:
            for ox in g:
                d2 = oz * oz + oy * oy + ox * ox
                if d2 > (search * dta) ** 2: continue
                c = idx + (np.array([oz, oy, ox]) / sz)[:, None]
                e = map_coordinates(ev, c, order=1, mode="nearest")
                best = np.minimum(best, d2 / dta ** 2 + ((e - r) / dose_tol) ** 2)
    out = np.full(ref.shape, np.nan)
    out[sel] = np.sqrt(best)
    return out, (100.0 * float(np.mean(out[sel] <= 1.0)) if sel.any() else float("nan"))


def dvh_figures(dose, mask, prescription):
    d = np.sort(np.asarray(dose, float)[mask])[::-1]
    if d.size == 0: return None
    at = lambda pct: float(d[min(d.size - 1, int(np.ceil(pct / 100.0 * d.size)) - 1)]) if pct > 0 else float(d[0])
    return {"voxels": int(d.size), "D98": at(98), "D95": at(95), "D50": at(50), "D2": at(2), "mean": float(d.mean()),
            "V95_pct": 100.0 * float(np.mean(d >= 0.95 * prescription)),
            "V107_pct": 100.0 * float(np.mean(d >= 1.07 * prescription))}


def profiles(ref, ev, origin, spacing, point):
    """Dose along x, y and z through `point` (mm, DICOM): {axis: {"pos", "ref", "eval"}}."""
    o, s = np.asarray(origin, float), np.asarray(spacing, float)
    i = np.clip(np.rint((np.asarray(point, float) - o) / s).astype(int), 0, np.array(ref.shape[::-1]) - 1)
    out = {}
    for ax, sl in (("x", (i[2], i[1], slice(None))), ("y", (i[2], slice(None), i[0])), ("z", (slice(None), i[1], i[0]))):
        k = "xyz".index(ax)
        out[ax] = {"pos": (o[k] + s[k] * np.arange(ref.shape[2 - k])).tolist(),
                   "ref": np.asarray(ref[sl], float).tolist(), "eval": np.asarray(ev[sl], float).tolist()}
    return out


def compare(ref, ev, origin, spacing, structures=(), criteria=((2.0, 2.0), (3.0, 3.0)), cutoff=10.0,
            prescription=None, point=None):
    """Summary dict: gamma pass rates, DVH figures of both doses per structure, profiles, global dose ratio."""
    res = {"grid": list(ref.shape), "spacing_mm": list(spacing), "ref_max": float(ref.max()), "eval_max": float(ev.max()),
           "gamma": {}, "structures": {}}
    for dd, dta in criteria:
        gm, rate = gamma(ref, ev, spacing, dta=dta, dd=dd, cutoff=cutoff)
        v = gm[np.isfinite(gm)]
        res["gamma"][f"{dd:g}%/{dta:g}mm"] = {"pass_pct": rate, "mean": float(v.mean()) if v.size else None,
                                             "p95": float(np.percentile(v, 95)) if v.size else None, "points": int(v.size)}
    targets = [m for _, t, m in structures if t.upper() == "TARGET"]
    if prescription is None and targets: prescription = float(np.median(ref[targets[0]]))
    res["prescription"] = prescription
    for name, typ, mask in structures:
        res["structures"][name] = {"type": typ, "ref": dvh_figures(ref, mask, prescription or ref.max()),
                                   "eval": dvh_figures(ev, mask, prescription or ref.max())}
    if targets:
        t = targets[0]
        # A plan recomputed in a water phantom can leave the target without dose (E2): no ratio then.
        r = float(ref[t].mean()) if t.any() else 0.0
        res["target_mean_ratio_eval_over_ref"] = float(ev[t].mean()) / r if r > 0 else None
    hi = ref >= 0.5 * ref.max()
    res["high_dose_mean_ratio_eval_over_ref"] = float(ev[hi].mean() / ref[hi].mean()) if ref.max() > 0 else None
    if point is not None: res["profiles"] = profiles(ref, ev, origin, spacing, point)
    return res


def centre_of_mass(dose, mask, origin, spacing):
    """Dose weighted centre of the voxels in `mask`, (x, y, z) in DICOM mm."""
    w = np.where(mask, np.asarray(dose, float), 0.0)
    tot = w.sum()
    if tot <= 0: return None
    z, y, x = np.indices(w.shape)
    return [float(origin[k] + spacing[k] * (w * c).sum() / tot) for k, c in enumerate((x, y, z))]


def rigid_shift(ref, ev, region, spacing, max_shift=3):
    """Integer voxel shift of `ev` that fits `ref` best in `region` (rms of the difference over the mean of ref there).

    The shift d (mm, x y z) means ev(p + d) matches ref(p): TOPAS lies displaced by +d against the TPS. Indices beyond
    the grid are clamped to its edge.
    """
    idx = np.array(np.nonzero(region))
    r = ref[region]; m = float(r.mean())
    best, zero = None, None
    hi = np.array(ref.shape)[:, None] - 1
    rng = range(-max_shift, max_shift + 1)
    for dz in rng:
        for dy in rng:
            for dx in rng:
                j = np.clip(idx + np.array([dz, dy, dx])[:, None], 0, hi)
                rms = float(np.sqrt(np.mean((ev[tuple(j)] - r) ** 2))) / m
                if dz == dy == dx == 0: zero = rms
                if best is None or rms < best[0] - 1e-12: best = (rms, (dx, dy, dz))
    return {"shift_mm": [float(best[1][k] * spacing[k]) for k in range(3)], "shift_voxels": list(best[1]),
            "rms_rel_best": best[0], "rms_rel_unshifted": zero, "max_shift_voxels": max_shift}


def diagnostics(ref, ev, origin, spacing, cutoff=10.0, max_shift=3, body=None):
    """Where TOPAS and TPS differ: voxel ratio in the high dose region (ref >= 50 % of its maximum) with the 5/50/95 %
    quantiles, centre of mass of each dose over its own voxels above cutoff % of its maximum (a common mask would cut
    off the part of a shifted dose that lies outside it), and the best rigid shift (rigid_shift).

    `body` (bool [z, y, x]) limits both centres to the patient: TOPAS scores dose in the air and the couch where the TPS
    has none, and those voxels would pull the TOPAS centre away although the dose in the body lies in place."""
    body = np.ones(ref.shape, bool) if body is None else np.asarray(body, bool)
    hi = ref >= 0.5 * ref.max()
    q = np.percentile(ev[hi] / ref[hi], [5, 50, 95])
    cr = centre_of_mass(ref, body & (ref >= cutoff / 100.0 * ref[body].max()), origin, spacing) if body.any() else None
    ce = centre_of_mass(ev, body & (ev >= cutoff / 100.0 * ev[body].max()), origin, spacing) if body.any() else None
    return {"high_dose_mean_ratio": float(ev[hi].mean() / ref[hi].mean()), "ratio_q05_q50_q95": [float(v) for v in q],
            "high_dose_voxels": int(hi.sum()),
            "centre_of_mass_shift_mm": [b - a for a, b in zip(cr, ce)] if cr and ce else None,
            "rigid": rigid_shift(ref, ev, hi, spacing, max_shift)}


def _load(path, origin=None, spacing=None):
    if path.endswith(".npy"):
        if origin is None or spacing is None: raise SystemExit(f"{path}: give --origin and --spacing for .npy")
        return np.load(path), tuple(origin), tuple(spacing)
    from plan_mc import synthetic
    return synthetic.read_rtdose(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref"); ap.add_argument("eval"); ap.add_argument("out")
    ap.add_argument("--matrad"); ap.add_argument("--origin", type=float, nargs=3); ap.add_argument("--spacing", type=float, nargs=3)
    ap.add_argument("--dta", type=float, default=None); ap.add_argument("--dd", type=float, default=None)
    ap.add_argument("--cutoff", type=float, default=10.0); ap.add_argument("--prescription", type=float)
    a = ap.parse_args()
    ref, o, s = _load(a.ref, a.origin, a.spacing)
    ev, o2, s2 = _load(a.eval, a.origin, a.spacing)
    if ref.shape != ev.shape or not np.allclose(o, o2, atol=1e-3) or not np.allclose(s, s2, atol=1e-4):
        raise SystemExit("the two doses are not on the same grid")
    st = []
    if a.matrad:
        from plan_mc import matrad
        st = matrad.read_cst(matrad._load(a.matrad), ref.shape)
    crit = ((a.dd, a.dta),) if a.dd and a.dta else ((2.0, 2.0), (3.0, 3.0))
    res = compare(ref, ev, o, s, st, crit, a.cutoff, a.prescription)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res["gamma"].items(): print(f"gamma {k}: {v['pass_pct']:.1f} % pass ({v['points']} points)")
