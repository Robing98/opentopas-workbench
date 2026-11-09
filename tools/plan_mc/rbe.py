"""Variable proton RBE from the physical dose and the dose averaged LET of TOPAS, against the constant 1.1.

Research only, never for a treatment decision: variable RBE models carry large uncertainties; clinically 1.1 applies.

All three models are phenomenological fits of the linear quadratic model to in vitro cell survival data. Each gives
RBEmax (alpha / alpha_x, the RBE at dose 0) and RBEmin (sqrt(beta / beta_x), the RBE at very high dose) as functions of
LETd [keV/um] and the alpha/beta ratio of the reference photons [Gy]. With D the physical proton dose per fraction:

    RBE(D) = ( sqrt( (a/b)^2 + 4 D (a/b) RBEmax + 4 D^2 RBEmin^2 ) - (a/b) ) / (2 D)

(McNamara et al. 2015, eq. 2; Wedenberg et al. 2013, eq. 3 with RBEmin = 1). The limit D -> 0 is RBEmax.

LETd is the dose averaged LET in water of TOPAS ProtonLET (protons only, primaries and secondaries; see read_let).
The models were fitted for LETd up to about 20 keV/um (McNamara) and doses of 1 to 10 Gy; outside they extrapolate.

The OpenTOPAS RBE extension (Polster et al. 2015, doi:10.1088/0031-9155/60/13/5053) scores the same models and more
(Chen, Wilkens, MKM, ...) during the run when compiled into OpenTOPAS; this module works on the dose and LET files a
standard OpenTOPAS writes, so no extension is needed.
"""
import numpy as np

CONSTANT = 1.1


def mcnamara(letd, ab):
    """McNamara, Schuemann, Paganetti 2015, Phys Med Biol 60:8399, doi:10.1088/0031-9155/60/21/8399, eq. 4 with the
    fit of sec. 3.1: RBEmax = 0.99064 + 0.35605 LETd / (a/b), RBEmin = 1.1012 - 0.0038703 sqrt(a/b) LETd.
    Checked: RBEmax at LETd 2.5 keV/um is 1.08 for (a/b) 10 Gy and 1.05 for 15 Gy, the values sec. 3.2 names."""
    letd, ab = np.asarray(letd, float), np.asarray(ab, float)
    return 0.99064 + 0.35605 * letd / ab, 1.1012 - 0.0038703 * np.sqrt(ab) * letd


def wedenberg(letd, ab):
    """Wedenberg, Lind, Hardemark 2013, Acta Oncol 52:580, doi:10.3109/0284186X.2012.705892, eq. 2 and 3:
    alpha / alpha_x = 1 + q LETd / (a/b) with q = 0.434 Gy um/keV, beta = beta_x, so RBEmax = 1 + 0.434 LETd / (a/b)
    and RBEmin = 1. Checked: the slope q / (a/b) is the k of their Table II (0.14 to 0.16 per keV/um for (a/b) 2.7 to
    3.1 Gy, 0.057 for 7.7 Gy, 0.024 to 0.029 for 15 to 18 Gy)."""
    letd, ab = np.asarray(letd, float), np.asarray(ab, float)
    return 1.0 + 0.434 * letd / ab, np.ones(np.broadcast(letd, ab).shape)


def carabe(letd, ab):
    """Carabe, Moteabbed, Depauw, Schuemann, Paganetti 2012, Phys Med Biol 57:1159, doi:10.1088/0031-9155/57/5/1159:
    RBEmax = 0.843 + 0.154 (2.686 / (a/b)) LETd, RBEmin = 1.09 + 0.006 (2.686 / (a/b)) LETd, with 2.686 Gy the
    alpha/beta of the V79 cells the fit was made on. Coefficients as McNamara 2015 (eq. 6) quotes them; the primary
    paper was not open to check them."""
    letd, ab = np.asarray(letd, float), np.asarray(ab, float)
    r = 2.686 / ab * letd
    return 0.843 + 0.154 * r, 1.09 + 0.006 * r


MODELS = {"mcnamara": mcnamara, "wedenberg": wedenberg, "carabe": carabe}


def rbe_lq(dose, ab, rbemax, rbemin):
    """RBE of the linear quadratic model at the physical dose per fraction `dose` [Gy]; RBEmax where the dose is 0."""
    d = np.asarray(dose, float)
    ab, rbemax, rbemin = (np.broadcast_to(np.asarray(v, float), np.broadcast(d, ab, rbemax, rbemin).shape)
                          for v in (ab, rbemax, rbemin))
    d = np.broadcast_to(d, ab.shape)
    out = np.array(rbemax, float, copy=True)
    pos = d > 1e-9
    a, D, mx, mn = ab[pos], d[pos], rbemax[pos], rbemin[pos]
    out[pos] = (np.sqrt(a * a + 4 * D * a * mx + 4 * D * D * mn * mn) - a) / (2 * D)
    return out


def rbe(model, dose, letd, ab, valid=None):
    """RBE per voxel of `model` ("mcnamara", "wedenberg", "carabe" or "1.1"). Outside `valid` (bool, e.g. voxels with
    too little dose for a trustworthy LET) the RBE is the constant 1.1, so the RBE weighted dose stays continuous."""
    if model in ("1.1", "constant"): return np.full(np.shape(dose), CONSTANT)
    mx, mn = MODELS[model](letd, ab)
    r = rbe_lq(dose, ab, mx, mn)
    if valid is not None: r = np.where(valid, r, CONSTANT)
    return r


# ---------------------------------------------------------------- LET of TOPAS
def read_let(path, histories):
    """LETd [keV/um] [x, y, z] of a TOPAS ProtonLET binary file and how it was scaled.

    ProtonLET (OpenTOPAS scoring/TsScoreProtonLET.cc) is dose weighted by default (WeightBy "dose"), scores protons
    only, counts the energy of secondary electrons made in the step (in material above 0.1 g/cm3), leaves out steps
    above MaxScoredLET (100 keV/um), and gives MeV/mm/(g/cm3), which is keV/um in water. The scorer divides the
    weighted sum by its denominator sub-scorer at the end of the run. The report "Mean" then divides that ratio once
    more by the number of histories: a "Mean" file is multiplied by the histories here, a "Sum" file is the LETd.
    Checked on the TG119 run (200k histories): Mean x histories gives a median 1.0 keV/um in the field and a maximum
    of 97 keV/um, just under the cut of 100.
    """
    from plan_mc import topas_io
    h, arr = topas_io.read(path)
    if "Sum" in arr: return np.asarray(arr["Sum"], float), "Sum"
    if "Mean" in arr: return np.asarray(arr["Mean"], float) * histories, f"Mean x {histories} Historien"
    raise ValueError(f"{path}: no Sum or Mean report in the LET file")


def combine_letd(doses, lets):
    """LETd of several fields: sum of D_f LETd_f over the sum of D_f (dose weighting, as within one field)."""
    num = den = 0.0
    for d, l in zip(doses, lets):
        num = num + np.asarray(d, float) * np.asarray(l, float)
        den = den + np.asarray(d, float)
    return np.divide(num, den, out=np.zeros(np.shape(den)), where=den > 0)


def let_mask(dose, threshold=0.05, dmax=None):
    """Voxels whose LETd is trusted: physical dose at least `threshold` of the maximum. Below, few protons stop or
    pass and the LET is noise (TG119 run: 99.9 % quantile 41 keV/um above 1 % of the maximum, 11 above 5 %).
    `dmax` replaces the maximum of the grid, e.g. the maximum inside the body, so one noisy air voxel cannot raise it."""
    d = np.asarray(dose, float)
    m = float(d.max()) if dmax is None else float(dmax)
    return d >= threshold * m if m > 0 else np.zeros(d.shape, bool)


# ---------------------------------------------------------------- range along a beam
def rays(arrays, origin, spacing, direction, min_ray=0.8, step=None, offsets=False):
    """Depth profiles of each array along rays parallel to the unit vector `direction` (beam travel, DICOM).

    The rays run through every voxel where arrays[0] >= min_ray x its maximum, one ray per voxel width in the plane
    across the beam, from before that region to the edge of the grid. Yields (t [mm along the beam], [profile of each
    array]) per ray, linear between voxel centres; with `offsets` also the point where the ray crosses the plane
    through the world origin across the beam (t = 0), so a caller can tell how far a ray runs from a given axis.
    """
    from scipy.ndimage import map_coordinates
    arrays = [np.asarray(a, float) for a in arrays]
    ref = arrays[0]
    o, s = np.asarray(origin, float), np.asarray(spacing, float)
    u = np.asarray(direction, float); u = u / np.linalg.norm(u)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    v1 = np.cross(u, a); v1 /= np.linalg.norm(v1); v2 = np.cross(u, v1)
    top = ref.max()
    if top <= 0: return
    z, y, x = np.nonzero(ref >= min_ray * top)
    pts = o + np.stack([x, y, z], 1) * s
    h = float(s.min())
    step = step or h / 10.0
    pu, p1, p2 = pts @ u, pts @ v1, pts @ v2
    k1, k2 = np.unique(np.rint(np.stack([p1, p2], 1) / h).astype(int), axis=0).T
    smin = pu.min() - 3 * float(np.abs(s).max())
    corners = o + np.array([[i, j, k] for i in (0, ref.shape[2] - 1) for j in (0, ref.shape[1] - 1)
                            for k in (0, ref.shape[0] - 1)]) * s
    ts = np.arange(smin, (corners @ u).max() + step, step)
    base = k1[:, None] * h * v1 + k2[:, None] * h * v2                      # ray offsets in the plane
    for i0 in range(0, len(base), 256):
        b = base[i0:i0 + 256]
        p = b[:, None, :] + ts[None, :, None] * u                           # [ray, t, xyz]
        idx = ((p - o) / s)[..., ::-1].reshape(-1, 3).T                     # z, y, x indices
        prof = [map_coordinates(c, idx, order=1, mode="constant", cval=0.0).reshape(len(b), len(ts)) for c in arrays]
        for j in range(len(b)):
            if prof[0][j].max() >= min_ray * top:
                yield (ts, [pr[j] for pr in prof], b[j]) if offsets else (ts, [pr[j] for pr in prof])


def distal_shift(ref, ev, origin, spacing, direction, level=0.8, min_ray=0.8, step=None, body=None, falloff_mm=15.0,
                 low=0.2, centre=None, radius_mm=5.0):
    """Shift [mm] of the distal falloff of `ev` against `ref` along the unit vector `direction` (beam travel, DICOM).

    On each ray (see rays) two crossings are compared, ev minus ref (positive: ev reaches deeper):
    "absolute": both at `level` x the ray's maximum of ref, the same dose in Gy(RBE): where the isodose planned with
    1.1 really lands, so a higher RBE everywhere moves it too;
    "shape": each at `level` x its own maximum on the ray (R80 of each depth dose), so only a change of the shape of
    the falloff counts.

    Only rays with a clear distal falloff of ref count: after its last point above `level` x the ray maximum, ref
    must fall to `low` x that maximum within `falloff_mm` (None: no limit), and with a `body` mask that stretch must
    lie inside the body. A ray that runs obliquely along the isodose falls slowly, and there a small change of the
    dose moves the crossing far; a ray that leaves the body meets the edge of a masked dose, not the range.
    Dropped rays are counted by reason: "no_edge" (ref never falls below the level on the grid), "falloff", "body".

    For the kept rays: median, 5 and 95 % quantiles [mm] of both shifts and of the distal falloff width (`level` to
    `low`, each dose at its own maximum) of ref ("width_ref") and ev ("width"). With `centre` (a point, DICOM mm),
    "axis" gives the medians over the kept rays within `radius_mm` of the beam axis through it.
    Returns None when no ray passes the region above `min_ray` of the maximum.
    """
    arrays = [ref, ev] + ([np.asarray(body, float)] if body is not None else [])
    u = np.asarray(direction, float); u = u / np.linalg.norm(u)
    if centre is not None:
        c = np.asarray(centre, float); c = c - (c @ u) * u                 # the axis crosses the plane t = 0 here
    got = {k: [] for k in ("absolute", "shape", "width_ref", "width")}
    axis = {"absolute": [], "shape": []}
    dropped = {"no_edge": 0, "falloff": 0, "body": 0}
    seen = False
    for ts, profs, b in rays(arrays, origin, spacing, direction, min_ray, step, offsets=True):
        seen = True
        r, e = profs[0], profs[1]
        edge = _falloff(ts, r, level, low)
        if edge is None: dropped["no_edge"] += 1; continue
        cr, c20, i, j = edge
        if c20 is None or (falloff_mm is not None and c20 - cr > falloff_mm): dropped["falloff"] += 1; continue
        if body is not None and (profs[2][i:j + 1] < 0.5).any(): dropped["body"] += 1; continue
        ca, cs = _distal(ts, e, level * r.max()), distal_edge(ts, e, level)
        if ca is not None: got["absolute"].append(ca - cr)
        if cs is not None: got["shape"].append(cs - cr)
        got["width_ref"].append(c20 - cr)
        we = _falloff(ts, e, level, low) if e.max() > 0 else None
        if we is not None and we[1] is not None: got["width"].append(we[1] - we[0])
        if centre is not None and np.linalg.norm(b - c) <= radius_mm:
            if ca is not None: axis["absolute"].append(ca - cr)
            if cs is not None: axis["shape"].append(cs - cr)
    if not seen: return None
    kept = len(got["width_ref"])
    out = {"level": level, "low": low, "falloff_mm": falloff_mm,
           "rays": {"kept": kept, "dropped": sum(dropped.values()), "dropped_by": dropped}}
    for k, v in got.items(): out[k] = _quantiles(v)
    out["axis"] = None if centre is None else {
        "radius_mm": radius_mm, "rays": len(axis["shape"]),
        **{f"{k}_median_mm": float(np.median(v)) if v else None for k, v in axis.items()}}
    return out


def _quantiles(v):
    if not v: return {"median_mm": None, "q05_mm": None, "q95_mm": None, "rays": 0}
    q = np.percentile(v, [5, 50, 95])
    return {"median_mm": float(q[1]), "q05_mm": float(q[0]), "q95_mm": float(q[2]), "rays": len(v)}


def _falloff(t, p, level, low):
    """(distal crossing of `level` x max(p), first crossing of `low` x max(p) after it or None, index of the last
    sample above the level, index of the first sample at or below `low`), or None without a distal crossing."""
    top = p.max()
    if top <= 0: return None
    above = p >= level * top
    i = int(np.nonzero(above)[0][-1])
    if i + 1 >= len(p): return None
    cr = float(t[i] + (p[i] - level * top) / (p[i] - p[i + 1]) * (t[i + 1] - t[i]))
    below = np.nonzero(p[i + 1:] <= low * top)[0]
    if not len(below): return cr, None, i, len(p) - 1
    j = i + 1 + int(below[0])
    L = low * top
    c20 = float(t[j - 1] + (p[j - 1] - L) / (p[j - 1] - p[j]) * (t[j] - t[j - 1])) if p[j - 1] != p[j] else float(t[j])
    return cr, c20, i, j


def distal_edge(t, p, level=0.8):
    """Distal crossing of `level` x max(p) on a depth profile, or None."""
    return _distal(t, p, level * p.max()) if p.max() > 0 else None


def _distal(t, p, L):
    """Last position where the profile p falls from >= L to < L, linear between samples."""
    above = p >= L
    if not above.any(): return None
    i = int(np.nonzero(above)[0][-1])
    if i + 1 >= len(p): return None
    return float(t[i] + (p[i] - L) / (p[i] - p[i + 1]) * (t[i + 1] - t[i]))
