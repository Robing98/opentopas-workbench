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


def let_mask(dose, threshold=0.05):
    """Voxels whose LETd is trusted: physical dose at least `threshold` of the maximum. Below, few protons stop or
    pass and the LET is noise (TG119 run: 99.9 % quantile 41 keV/um above 1 % of the maximum, 11 above 5 %)."""
    d = np.asarray(dose, float)
    return d >= threshold * d.max() if d.max() > 0 else np.zeros(d.shape, bool)


# ---------------------------------------------------------------- range along a beam
def distal_shift(ref, ev, origin, spacing, direction, level=0.8, min_ray=0.8, step=None):
    """Shift [mm] of the distal isodose of `ev` against `ref` along the unit vector `direction` (beam travel, DICOM).

    Rays parallel to the beam run through every voxel where ref >= min_ray x its maximum. On each ray the distal
    crossing of the absolute dose level x max(ref) is found for both doses, linearly between samples; the shift is
    ev minus ref (positive: ev reaches deeper). Returns median, 5 and 95 % quantiles and the number of rays, or None.
    """
    from scipy.ndimage import map_coordinates
    ref, ev = np.asarray(ref, float), np.asarray(ev, float)
    o, s = np.asarray(origin, float), np.asarray(spacing, float)
    u = np.asarray(direction, float); u = u / np.linalg.norm(u)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    v1 = np.cross(u, a); v1 /= np.linalg.norm(v1); v2 = np.cross(u, v1)
    top = ref.max()
    if top <= 0: return None
    L = level * top
    z, y, x = np.nonzero(ref >= min_ray * top)
    pts = o + np.stack([x, y, z], 1) * s
    h = float(s.min())
    step = step or h / 10.0
    pu, p1, p2 = pts @ u, pts @ v1, pts @ v2
    k1, k2 = np.unique(np.rint(np.stack([p1, p2], 1) / h).astype(int), axis=0).T
    smin, smax = pu.min() - 3 * float(np.abs(s).max()), None
    # how far the dose can reach behind the high dose region: to the edge of the grid
    corners = o + np.array([[i, j, k] for i in (0, ref.shape[2] - 1) for j in (0, ref.shape[1] - 1)
                            for k in (0, ref.shape[0] - 1)]) * s
    smax = (corners @ u).max()
    ts = np.arange(smin, smax + step, step)
    base = k1[:, None] * h * v1 + k2[:, None] * h * v2                      # ray offsets in the plane
    shifts = []
    for i0 in range(0, len(base), 256):
        b = base[i0:i0 + 256]
        p = b[:, None, :] + ts[None, :, None] * u                           # [ray, t, xyz]
        idx = ((p - o) / s)[..., ::-1].reshape(-1, 3).T                     # z, y, x indices
        prof = [map_coordinates(c, idx, order=1, mode="constant", cval=0.0).reshape(len(b), len(ts)) for c in (ref, ev)]
        for r, e in zip(*prof):
            if r.max() < min_ray * top: continue
            cr, ce = _distal(ts, r, L), _distal(ts, e, L)
            if cr is not None and ce is not None: shifts.append(ce - cr)
    if not shifts: return None
    q = np.percentile(shifts, [5, 50, 95])
    return {"median_mm": float(q[1]), "q05_mm": float(q[0]), "q95_mm": float(q[2]), "rays": len(shifts), "level": level}


def _distal(t, p, L):
    """Last position where the profile p falls from >= L to < L, linear between samples."""
    above = p >= L
    if not above.any(): return None
    i = int(np.nonzero(above)[0][-1])
    if i + 1 >= len(p): return None
    return float(t[i] + (p[i] - L) / (p[i] - p[i + 1]) * (t[i + 1] - t[i]))
