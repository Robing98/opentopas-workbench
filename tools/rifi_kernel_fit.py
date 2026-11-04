#!/usr/bin/env python3
"""Fits the ripple filter kernel from depth dose curves measured or simulated with and without the filter.

Model (Weber and Kraft 1999): every particle crosses one water equivalent thickness t of the filter and its Bragg curve
moves t towards the surface, so the curve with filter is a weighted mean of shifted copies of the curve without:

    IDD_with(z) = A * sum_t K(t) * IDD_without(z + t)        (a convolution, K sums to 1)

K is the kernel: the fraction of the filter area (one period, 1D grooves or 2D pins alike) that has thickness t. The
beam profile drops out as long as the spot is wider than about half a period. A absorbs normalisation (both curves are
scaled to their peak) and nuclear losses in the filter. The fit returns the mean thickness t_mean (pull-back) and the
spread sigma_t of K per energy, for these kernel shapes:
  gauss    Gaussian K (Weber's optimised grooves aim at this)
  box      uniform K (1D triangle, 2D pins shaped for a uniform distribution)
  pyramid  K falling linearly from the thinnest to the thickest part (straight 2D pyramids)
  free     K without an assumed shape: non-negative weights on a 0.2 mm grid with a smoothness penalty
The best parametric shape (smallest residual) is reported per curve.

True kernel for comparison, taken automatically from the TOPAS parameter file that wrote the curve with filter:
TsRidgeFilter (several filters with the same orientation add up, crossed filters convolve) and TsCAD (binary or ASCII
STL, thickness by casting rays through the mesh). Or given by hand: --ridge, --pin, --stl, --heightmap. With
--show-kernel only the true kernel is computed and plotted (no curves needed), e.g. for the STL of a printed filter.
The script also reports the shortcut sigma_t^2 ~ BPW80_with^2 - BPW80_without^2, which does not hold for Bragg peaks.

Usage:
  python3 rifi_kernel_fit.py FOLDER [--variants t4,pin4] [--ref REF_FOLDER] [--out DIR]
      every curve with "rifi0" in its name (in FOLDER, else REF_FOLDER or the parent folder) is paired with all curves
      whose name differs only there, e.g. IDD_planF1_p063_rifi0.csv with IDD_planF1_p063_t4.csv
  python3 rifi_kernel_fit.py --pair WITHOUT WITH [--pair WITHOUT WITH ...] [--energy E ...]
  python3 rifi_kernel_fit.py MEASURED_FOLDER      (pairs named like 100MeV_ohne.mcc and 100MeV_mit.mcc)
  python3 rifi_kernel_fit.py --show-kernel --pin pyramid:h=4,base=0.1,p=1.5 [--stl FILE] [--out DIR]
Input files: TOPAS csv (laterally integrated, one X and Y bin), PTW .mcc, or two columns depth [mm] and dose.
Options for tests with simulated data: --noise PCT (Gaussian noise relative to the peak), --coarse (1 mm steps in the
plateau, 0.1 mm around the peak, like a typical water column scan). Results: kernfit.csv and plots in --out.
"""
import argparse, glob, os, re, struct, sys
import numpy as np
from scipy.optimize import least_squares, nnls

SHAPES = ("gauss", "box", "pyramid")
SPR_BY_MATERIAL = {"G4_PLEXIGLASS": None, "LUCITE": None, "PMMA": None, "G4_WATER": 1.0, "WATER": 1.0}   # None = --spr
NUCLEONS = {"proton": 1, "deuteron": 2, "triton": 3, "he3": 3, "alpha": 4}


# ------------------------------------------------------------------ reading curves and parameter files
def read_curve(path):
    """Depth [mm] and dose for TOPAS csv, PTW mcc, or two-column text. TOPAS dose is per history when known."""
    txt = open(path, errors="replace").read()
    if "# Z in" in txt and "bins" in txt:                       # TOPAS
        m = re.search(r"#\s*Z\s+in\s+(\d+)\s+bins?\s+of\s+([\d.eE+\-]+)\s*(\w+)", txt)
        nz, w = int(m.group(1)), float(m.group(2)) * {"mm": 1, "cm": 10, "m": 1000}[m.group(3)]
        d = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
        dose = np.zeros(nz); dose[d[:, 2].astype(int)] = d[:, -1]
        z = (np.arange(nz) + 0.5) * w
        info = topas_info(path)
        if info.get("n"): dose = dose / info["n"]
        return z, dose, info
    if "BEGIN_DATA" in txt:                                     # PTW mcc: first scan only
        block = txt.split("BEGIN_DATA", 1)[1].split("END_DATA", 1)[0]
        rows = [l.split() for l in block.strip().splitlines() if l.strip()]
        a = np.array([[float(r[0].replace(",", ".")), float(r[1].replace(",", "."))] for r in rows if len(r) >= 2])
    else:                                                       # two columns, decimal comma allowed
        rows = []
        for l in txt.splitlines():
            l = l.strip()
            if not l or l[0] in "#;%": continue
            p = re.split(r"[\s;\t]+|,(?=\s)", l)
            p = [x.replace(",", ".") for x in p if x]
            try: rows.append([float(p[0]), float(p[1])])
            except (ValueError, IndexError): continue
        a = np.array(rows)
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1], {}


UNIT = {"um": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1000.0, "deg": 1.0, "rad": 180 / np.pi, "keV": 1e-3, "MeV": 1.0, "GeV": 1e3}


def parse_params(text):
    """TOPAS parameters as {(component, name): (type, value string)} for Ge/ and So/ entries (no include files)."""
    out = {}
    for l in text.splitlines():
        m = re.match(r'^\s*([a-z]+):(Ge|So)/([^/\s]+)/(\w+)\s*=\s*(.+?)\s*(#.*)?$', l)
        if m: out[(m.group(3), m.group(4))] = (m.group(1), m.group(5))
    return out


def num(v):
    """'80.0 cm' -> 800.0 (mm, deg, MeV); vectors 'N v1 ... vN unit' -> array."""
    p = v.replace('"', "").split()
    u = UNIT.get(p[-1], None)
    vals = p[:-1] if u is not None else p
    u = 1.0 if u is None else u
    if len(vals) > 1:                                           # vector: first entry is the count
        return np.array(vals[1:], float) * u
    return float(vals[0]) * u


def topas_info(path):
    """Energy, histories, particle, and filter geometry from the TOPAS parameter file that writes this output."""
    base = os.path.splitext(os.path.basename(path))[0]
    for d in (os.path.dirname(path), os.path.dirname(os.path.dirname(path))):
        for f in sorted(glob.glob(os.path.join(d or ".", "*.txt"))):
            t = open(f, errors="replace").read()
            if not re.search(r'OutputFile\s*=\s*"' + re.escape(base) + '"', t): continue
            P = parse_params(t); out = {"param": f, "ridges": [], "cads": []}
            for (comp, key), (typ, val) in P.items():
                if comp == "BeamPosition": continue
                if key == "BeamEnergy": out["energy"] = num(val)
                elif key == "NumberOfHistoriesInRun": out["n"] = int(float(val.split()[0]))
                elif key == "BeamParticle":
                    pn = val.strip('"'); out["particle"] = pn
                    g = re.match(r"GenericIon\((\d+),(\d+)", pn)
                    out["A"] = int(g.group(2)) if g else NUCLEONS.get(pn.lower(), 1)
            for comp in sorted({c for c, _ in P}):
                typ = P.get((comp, "Type"), ("", ""))[1].strip('"')
                mat = P.get((comp, "Material"), ("", '""'))[1].strip('"')
                if typ == "TsRidgeFilter" and (comp, "XPoints") in P:
                    out["ridges"].append({"name": comp, "x": num(P[(comp, "XPoints")][1]), "z": num(P[(comp, "ZPoints")][1]),
                                          "width": num(P[(comp, "Width")][1]) if (comp, "Width") in P else None,
                                          "rotz": num(P[(comp, "RotZ")][1]) if (comp, "RotZ") in P else 0.0, "material": mat})
                elif typ == "TsCAD" and (comp, "InputFile") in P:
                    fn = P[(comp, "InputFile")][1].strip('"')
                    fn = fn if os.path.isabs(fn) else os.path.join(os.path.dirname(f), fn)
                    fmt = P.get((comp, "FileFormat"), ("s", '"stl"'))[1].strip('"').lower()
                    out["cads"].append({"name": comp, "file": fn + "." + fmt, "format": fmt, "material": mat,
                                        "units": num(P[(comp, "Units")][1]) if (comp, "Units") in P else 1.0})
            return out
    return {}


def energy_from_name(path):
    b = os.path.basename(path)
    m = re.search(r"p(\d+)p(\d+)", b)
    if m: return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*MeV", b, re.I)
    if m: return float(m.group(1).replace(",", "."))
    m = re.search(r"(?:^|[_\-\s])E(\d+(?:[.,]\d+)?)", b)
    if m: return float(m.group(1).replace(",", "."))
    m = re.search(r"_[pac](\d{2,3})_", b)                       # planF1_p099_... / mess_a058_...: whole MeV(/u) only
    return float(m.group(1)) if m else np.nan


# ------------------------------------------------------------------ true kernels from geometry
def hist_kernel(samples, step=0.02):
    """Grid, weights (sum 1), mean, and standard deviation of a sample of water equivalent thicknesses."""
    s = np.asarray(samples, float).ravel()
    edges = np.arange(np.floor(s.min() / step) * step, s.max() + 2 * step, step)
    w, _ = np.histogram(s, edges)
    return 0.5 * (edges[1:] + edges[:-1]), w / w.sum(), float(s.mean()), float(s.std())


def convolve_kernels(a, b):
    """Kernel of the sum of two independent thicknesses (crossed filters, or filter plus plate)."""
    ta, wa = a[0], a[1]; tb, wb = b[0], b[1]
    step = ta[1] - ta[0]
    wb2 = np.interp(np.arange(tb[0], tb[-1] + step / 2, step), tb, wb); wb2 /= wb2.sum()
    w = np.convolve(wa, wb2); t = ta[0] + np.arange(len(w)) * step + tb[0]
    m = float((w * t).sum()); s = float(np.sqrt((w * (t - m) ** 2).sum()))
    return t, w, m, s


def ridge_samples(x, z, width=None, n=200001):
    """Thickness along one period of a TsRidgeFilter profile (TOPAS closes the profile at 0 on both ends)."""
    x = np.asarray(x, float); z = np.asarray(z, float)
    width = width or x[-1]
    if x[0] > 0: x = np.r_[0.0, x]; z = np.r_[0.0, z]
    if x[-1] < width: x = np.r_[x, width]; z = np.r_[z, 0.0]
    xs = np.linspace(0, width, n + 1)[:-1] + width / (2 * n)
    return np.interp(xs, x, z)


def read_stl(path):
    """Triangles (N, 3, 3) from a binary or ASCII STL file."""
    raw = open(path, "rb").read()
    if len(raw) >= 84:
        n = struct.unpack("<I", raw[80:84])[0]
        if len(raw) == 84 + 50 * n:
            a = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
            return a["v"].astype(float)
    v = [float(x) for x in re.findall(r"vertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)", raw.decode(errors="replace")) for x in x]
    return np.array(v, float).reshape(-1, 3, 3)


def stl_thickness(tris, region=12.0, step=0.01):
    """Thickness along z through a closed mesh on a grid in the centre of its footprint.

    Every facet crossed by a ray parallel to z adds sign(n_z) * z at the crossing (entry through an upward facing
    facet, exit through a downward facing one), so the sum over facets is the material length. The grid is offset
    differently in x and y so that no ray hits an edge or a pin diagonal exactly."""
    lo, hi = tris[:, :, :2].min(axis=(0, 1)), tris[:, :, :2].max(axis=(0, 1))
    c = 0.5 * (lo + hi); half = min(region / 2, 0.45 * float((hi - lo).min()))
    n = int(round(2 * half / step))
    gx = c[0] - half + (np.arange(n) + 0.2137) * step; gy = c[1] - half + (np.arange(n) + 0.6719) * step
    T = np.zeros((n, n))
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    nz = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    tmin, tmax = tris[:, :, :2].min(axis=1), tris[:, :, :2].max(axis=1)
    use = (np.abs(nz) > 1e-12) & (tmax[:, 0] >= gx[0]) & (tmin[:, 0] <= gx[-1]) & (tmax[:, 1] >= gy[0]) & (tmin[:, 1] <= gy[-1])
    for k in np.nonzero(use)[0]:
        i0 = max(int(np.ceil((tmin[k, 0] - gx[0]) / step)), 0); i1 = min(int(np.floor((tmax[k, 0] - gx[0]) / step)), n - 1)
        j0 = max(int(np.ceil((tmin[k, 1] - gy[0]) / step)), 0); j1 = min(int(np.floor((tmax[k, 1] - gy[0]) / step)), n - 1)
        if i1 < i0 or j1 < j0: continue
        X, Y = np.meshgrid(gx[i0:i1 + 1], gy[j0:j1 + 1])
        a, b, cc = v0[k], v1[k], v2[k]
        d = nz[k]
        l1 = ((b[0] - X) * (cc[1] - Y) - (b[1] - Y) * (cc[0] - X)) / d
        l2 = ((cc[0] - X) * (a[1] - Y) - (cc[1] - Y) * (a[0] - X)) / d
        l3 = 1 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not inside.any(): continue
        zz = l1 * a[2] + l2 * b[2] + l3 * cc[2]
        T[j0:j1 + 1, i0:i1 + 1] += np.where(inside, np.sign(d) * zz, 0.0)
    if T.mean() < 0: T = -T                                    # mesh oriented inwards
    return T


def profile_samples(kind, h, base=0.0, p=1.5, sigma=None, dim=2, n=400000):
    """Thickness samples of one cell of a pin (dim 2, square, profile over max(|x|,|y|)) or ridge (dim 1).

    u in [0, 1] is the distance from the pin axis in units of half the period; its density is dim * u^(dim-1).
    pyramid: f(u) = h (1 - u); uniform: f(u) = h (1 - u^dim), every thickness equally frequent;
    gauss: f(u) = F^-1(1 - u^dim) with F the Gaussian (mean h/2, sigma) truncated to [0, h] (pin shaped for that kernel)."""
    u = ((np.arange(n) + 0.5) / n) ** (1.0 / dim)
    if kind == "pyramid": f = h * (1 - u)
    elif kind == "uniform": f = h * (1 - u ** dim)
    elif kind == "gauss":
        from scipy.stats import truncnorm
        s = sigma or h / 4
        a_, b_ = (0 - h / 2) / s, (h - h / 2) / s
        f = truncnorm.ppf(np.clip(1 - u ** dim, 1e-12, 1 - 1e-12), a_, b_, loc=h / 2, scale=s)
    else: raise ValueError(f"unknown pin shape {kind}")
    return base + f


def parse_pin(spec):
    """'pyramid:h=4,base=0.1,p=1.5' -> kwargs for profile_samples (lengths in mm of filter material)."""
    kind, _, rest = spec.partition(":")
    kw = {"kind": kind.strip().lower()}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        k = k.strip().lower(); kw[{"height": "h", "period": "p", "d": "dim"}.get(k, k)] = float(v)
    if "dim" in kw: kw["dim"] = int(kw["dim"])
    if "h" not in kw: raise ValueError(f"--pin {spec}: height h missing")
    return kw


def material_spr(mat, spr):
    v = SPR_BY_MATERIAL.get((mat or "").upper(), None)
    return spr if v is None else v


def true_kernel(info, a):
    """True kernel (grid, weights, mean, sigma, description) from the command line or the parameter file, else None."""
    parts, desc = [], []
    if a.pin:
        kw = parse_pin(a.pin); parts.append(hist_kernel(profile_samples(**kw) * a.spr)); desc.append(f"pin {a.pin}")
    if a.stl:
        parts.append(hist_kernel(stl_thickness(read_stl(a.stl) * a.stl_units) * a.spr)); desc.append(f"STL {os.path.basename(a.stl)}")
    if a.heightmap:
        hm = np.load(a.heightmap) if a.heightmap.endswith(".npy") else np.loadtxt(a.heightmap, delimiter="," if a.heightmap.endswith(".csv") else None)
        parts.append(hist_kernel(hm * a.spr)); desc.append(f"heightmap {os.path.basename(a.heightmap)}")
    if a.ridge:
        xs, hs = a.ridge.split(":")
        parts.append(hist_kernel(ridge_samples(np.array(xs.split(","), float), np.array(hs.split(","), float)) * a.spr)); desc.append("ridge (CLI)")
    if not parts and info:
        groups = {}
        for r in info.get("ridges", []):                       # same orientation: aligned, profiles add; crossed: independent
            groups.setdefault(round(r["rotz"] % 180), []).append(r)
        for rot, rs in sorted(groups.items()):
            w = rs[0]["width"] or rs[0]["x"][-1]
            if all(abs((r["width"] or r["x"][-1]) - w) < 1e-9 for r in rs):
                s = sum(ridge_samples(r["x"], r["z"], r["width"]) * material_spr(r["material"], a.spr) for r in rs)
                parts.append(hist_kernel(s)); desc.append("+".join(r["name"] for r in rs) + f" (RotZ {rot}°)")
            else:
                for r in rs:
                    parts.append(hist_kernel(ridge_samples(r["x"], r["z"], r["width"]) * material_spr(r["material"], a.spr))); desc.append(r["name"])
        for c in info.get("cads", []):
            if c["format"] != "stl" or not os.path.exists(c["file"]): continue
            parts.append(hist_kernel(stl_thickness(read_stl(c["file"]) * c["units"]) * material_spr(c["material"], a.spr)))
            desc.append(f"{c['name']} ({os.path.basename(c['file'])})")
    if not parts: return None
    k = parts[0]
    for q in parts[1:]: k = convolve_kernels(k, q)
    return k[0], k[1], k[2], k[3], " ⊗ ".join(desc)


# ------------------------------------------------------------------ kernel models
def kernel(shape, tm, sig, n=161):
    """Sample points and weights of a kernel with mean tm and standard deviation sig."""
    sig = max(sig, 1e-4); q = (np.arange(n) + 0.5) / n
    if shape == "gauss":
        t = np.linspace(tm - 5 * sig, tm + 5 * sig, n); w = np.exp(-0.5 * ((t - tm) / sig) ** 2)
    elif shape == "box":                                        # half width sqrt(3) sigma
        hw = np.sqrt(3) * sig; t = tm - hw + 2 * hw * q; w = np.ones(n)
    elif shape == "pyramid":                                    # falling linearly over h = sqrt(18) sigma, mean b + h/3
        h = np.sqrt(18) * sig; t = tm - h / 3 + h * q; w = 1 - q
    else: raise ValueError(shape)
    return t, w / w.sum()


def kernel_density(shape, tm, sig, t):
    """Kernel as density per mm on the points t (for plots)."""
    if shape == "gauss": return np.exp(-0.5 * ((t - tm) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    if shape == "box": hw = np.sqrt(3) * sig; return np.where(abs(t - tm) <= hw, 1 / (2 * hw), 0.0)
    h = np.sqrt(18) * sig; b = tm - h / 3
    return np.where((t >= b) & (t <= b + h), 2 * (1 - (t - b) / h) / h, 0.0)


def shifted(zq, z, d, t):
    """IDD_without(zq + t), zero beyond the curve, first value before it."""
    return np.interp(zq + t, z, d, left=d[0], right=0.0)


def model(zq, z, d, A, tm, sig, shape):
    t, w = kernel(shape, tm, sig)
    return A * sum(wi * shifted(zq, z, d, ti) for ti, wi in zip(t, w))


# ------------------------------------------------------------------ curve metrics
def area(y, x):
    """Area under a sampled curve by the trapezoid rule (np.trapezoid only exists from numpy 2.0 on)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def metrics(z, d):
    i = int(np.argmax(d)); dm = d[i]
    def after(level):
        for k in range(i, len(d) - 1):
            if d[k] >= level * dm > d[k + 1]: return z[k] + (z[k + 1] - z[k]) * (d[k] - level * dm) / (d[k] - d[k + 1])
        return np.nan
    def before(level):
        for k in range(i - 1, -1, -1):
            if d[k] < level * dm <= d[k + 1]: return z[k] + (z[k + 1] - z[k]) * (level * dm - d[k]) / (d[k + 1] - d[k])
        return np.nan
    r80, r20, r05 = after(0.8), after(0.2), after(0.05)
    return {"peak": z[i], "R80": r80, "R05": r05, "BPW80": r80 - before(0.8), "D8020": r20 - r80}


def smooth(d, n=3):
    k = n // 2
    return np.convolve(np.pad(d, (k, n - 1 - k), mode="edge"), np.ones(n) / n, mode="valid")


# ------------------------------------------------------------------ fits
def fit_param(z0, d0, z1, d1, shape, start):
    """Least squares for A, t_mean, sigma_t; residuals relative to the peak of the curve with filter."""
    m1 = metrics(z1, d1); mask = z1 <= m1["R05"]
    zq, y = z1[mask], d1[mask]; s = y.max()
    res = lambda p: (model(zq, z0, d0, p[0], p[1], p[2], shape) - y) / s
    r = least_squares(res, start, bounds=([0, -5, 0.01], [100, 40, 15]), x_scale=[0.1, 1, 0.3])
    J = r.jac; dof = max(len(y) - 3, 1); s2 = (r.fun ** 2).sum() / dof
    try: err = np.sqrt(np.diag(np.linalg.inv(J.T @ J)) * s2)
    except np.linalg.LinAlgError: err = np.full(3, np.nan)
    return r.x, err, float(np.sqrt((r.fun ** 2).mean())) * 100


def fit_free(z0, d0, z1, d1, tmax, step=0.2, alpha=3e-3):
    """Non-negative kernel on a grid with a second-difference penalty (shape without assumption)."""
    m1 = metrics(z1, d1); mask = z1 <= m1["R05"]
    zq, y = z1[mask], d1[mask]
    t = np.arange(0, tmax + step / 2, step)
    M = np.stack([shifted(zq, z0, d0, ti) for ti in t], axis=1)
    D = np.diff(np.eye(len(t)), 2, axis=0)
    lam = alpha * np.trace(M.T @ M) / len(t)
    w, _ = nnls(np.vstack([M, np.sqrt(lam) * D]), np.concatenate([y, np.zeros(D.shape[0])]), maxiter=50 * len(t))
    A = w.sum(); k = w / A if A > 0 else w
    tm = float((k * t).sum()); sig = float(np.sqrt(max((k * (t - tm) ** 2).sum(), 0)))
    rms = float(np.sqrt((((M @ w) - y) / y.max()) ** 2).mean()) * 100
    return t, k, A, tm, sig, rms


def coarse_grid(z, d):
    m = metrics(z, d)
    g = np.concatenate([np.arange(z[0], m["peak"] - 10, 1.0), np.arange(m["peak"] - 10, m["R05"] + 5, 0.1), np.arange(m["R05"] + 5, z[-1], 1.0)])
    return g, np.interp(g, z, d)


# ------------------------------------------------------------------ pairing
def discover(folder, ref, tag, variants):
    """Pairs (without, with, variant): every curve containing `tag` in REF (default FOLDER, else its parent) with the
    curves in FOLDER whose names differ only there."""
    pairs = []
    for refdir in ([ref] if ref else [folder, os.path.dirname(os.path.abspath(folder))]):
        for f0 in sorted(glob.glob(os.path.join(refdir, f"*{tag}*"))):
            if os.path.splitext(f0)[1].lower() not in (".csv", ".mcc", ".txt", ".dat") or f0.endswith(".txt") and "IDD" not in os.path.basename(f0):
                continue
            b = os.path.basename(f0); i = b.index(tag); pre, suf = b[:i], b[i + len(tag):]
            for f1 in sorted(glob.glob(os.path.join(folder, glob.escape(pre) + "*" + glob.escape(suf)))):
                v = os.path.basename(f1)[len(pre):len(os.path.basename(f1)) - len(suf)]
                if not v or v == tag or (variants and v not in variants): continue
                pairs.append((f0, f1, v))
        if pairs: break
    if not pairs:                                               # measured curves named ..._ohne.* / ..._mit.*
        for f in sorted(glob.glob(os.path.join(folder, "*ohne*"))):
            g = os.path.join(os.path.dirname(f), os.path.basename(f).replace("ohne", "mit"))
            if os.path.exists(g): pairs.append((f, g, "mit"))
    return pairs


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", help="folder with curves (TOPAS output or measured pairs)")
    ap.add_argument("--ref", help="folder with the curves without filter (default: FOLDER, else its parent)")
    ap.add_argument("--tag-without", default="rifi0", help="name part of the curves without filter (default rifi0)")
    ap.add_argument("--variants", help="comma separated variants to fit, e.g. t4,pin4 (default: all found)")
    ap.add_argument("--match", help="only pairs whose curve with filter matches this regular expression, e.g. _p[0-9]+_")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("WITHOUT", "WITH"), help="explicit pair, repeatable")
    ap.add_argument("--energy", type=float, action="append", help="energy per --pair if not in the file name")
    ap.add_argument("--shapes", default=",".join(SHAPES), help="parametric kernel shapes to fit (default gauss,box,pyramid)")
    ap.add_argument("--spr", type=float, default=1.16, help="stopping power ratio of the filter material to water (PMMA 1.16)")
    ap.add_argument("--ridge", help="true kernel from a 1D profile: 'x1,x2,...:h1,h2,...' in mm, one period")
    ap.add_argument("--pin", help="true kernel from a pin shape, e.g. pyramid:h=4,base=0.1 | uniform:h=4 | gauss:h=4,sigma=1 (dim=1 for ridges)")
    ap.add_argument("--stl", help="true kernel from an STL file of the filter (beam along z)")
    ap.add_argument("--stl-units", type=float, default=1.0, help="mm per STL unit (default 1)")
    ap.add_argument("--heightmap", help="true kernel from a thickness map (.npy, .csv, or .txt, mm)")
    ap.add_argument("--show-kernel", action="store_true", help="only compute and plot the true kernel")
    ap.add_argument("--noise", type=float, default=0.0, help="add Gaussian noise [%% of the peak] to both curves (test)")
    ap.add_argument("--coarse", action="store_true", help="resample both curves like a water column scan (test)")
    ap.add_argument("--alpha", type=float, default=3e-3, help="smoothness of the free kernel (larger = smoother)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="output folder (default FOLDER/kernfit)")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    shapes = [s.strip() for s in a.shapes.split(",") if s.strip() in SHAPES]

    if a.show_kernel:
        tk = true_kernel({}, a)
        if tk is None: sys.exit("--show-kernel needs --pin, --stl, --heightmap, or --ridge")
        out = a.out or "."; os.makedirs(out, exist_ok=True)
        print(f"true kernel ({tk[4]}): t_mean {tk[2]:.3f} mm, sigma_t {tk[3]:.3f} mm water, range {tk[0][tk[1] > 0][0]:.2f} to {tk[0][tk[1] > 0][-1]:.2f} mm,"
              f" modulation power sigma^2/t_mean {tk[3] ** 2 / tk[2]:.3f} mm (SPR {a.spr})")
        if not a.no_plots: plot_true(tk, os.path.join(out, "kern_geometrie.png"))
        return

    pairs = [(p[0], p[1], "mit") for p in (a.pair or [])]
    if a.folder: pairs += discover(a.folder, a.ref, a.tag_without, set(a.variants.split(",")) if a.variants else None)
    if a.match: pairs = [p for p in pairs if re.search(a.match, os.path.basename(p[1]))]
    if not pairs: sys.exit("no pairs found (FOLDER with curves *rifi0* and variants, measured *_ohne/_mit, or --pair)")
    out = a.out or os.path.join(a.folder or ".", "kernfit"); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    truth_cache = {}

    rows, keep = [], []
    for k, (f0, f1, var) in enumerate(pairs):
        z0, d0, i0 = read_curve(f0); z1, d1, i1 = read_curve(f1)
        E = i1.get("energy") or i0.get("energy") or (a.energy[k] if a.energy and k < len(a.energy) else energy_from_name(f1))
        A_nuc = i1.get("A") or i0.get("A") or 1
        if a.coarse: (z0, d0), (z1, d1) = coarse_grid(z0, d0), coarse_grid(z1, d1)
        d0, d1 = d0 / d0.max(), d1 / d1.max()                    # each curve relative to its peak; A carries the ratio
        if a.noise > 0:
            d0 = d0 + rng.normal(0, a.noise / 100, d0.size); d1 = d1 + rng.normal(0, a.noise / 100, d1.size)
        d0s = smooth(d0) if not a.coarse else d0                 # the curve that gets shifted: light smoothing against noise
        m0, m1 = metrics(z0, smooth(d0)), metrics(z1, smooth(d1))
        geo = repr([(r["x"].tolist(), r["z"].tolist(), r["rotz"], r["width"], r["material"]) for r in i1.get("ridges", [])]) + \
              repr([(c["file"], c["units"], c["material"]) for c in i1.get("cads", [])])
        key = (geo, a.pin, a.stl, a.ridge, a.heightmap)            # same geometry, same true kernel
        if key not in truth_cache: truth_cache[key] = true_kernel(i1, a)
        tk = truth_cache[key]
        dR80 = m0["R80"] - m1["R80"]; q2 = m1["BPW80"] ** 2 - m0["BPW80"] ** 2
        A0 = area(d1, z1) / area(d0, z0)          # a convolution keeps the area
        start = [float(np.clip(A0, 0.01, 99)), float(np.clip(dR80 - 0.3, 0.1, 39)), float(np.clip(np.sqrt(max(q2, 0.01)) / 1.336, 0.3, 14))]
        row = {"variant": var, "particle": i1.get("particle", ""), "energy_MeV": E, "energy_MeV_u": E / A_nuc,
               "file_with": os.path.basename(f1), "R80_without": m0["R80"], "R80_with": m1["R80"], "dR80": dR80,
               "BPW80_without": m0["BPW80"], "BPW80_with": m1["BPW80"], "D8020_without": m0["D8020"], "D8020_with": m1["D8020"],
               "sigma_quad_raw": np.sqrt(max(q2, 0)), "sigma_quad_gauss": np.sqrt(max(q2, 0)) / 1.336}
        fits = {}
        for shape in shapes:
            p, e, rms = fit_param(z0, d0s, z1, d1, shape, start)
            fits[shape] = p
            row.update({f"A_{shape}": p[0], f"t_mean_{shape}": p[1], f"t_mean_{shape}_err": e[1], f"sigma_t_{shape}": p[2],
                        f"sigma_t_{shape}_err": e[2], f"rms_{shape}_pct": rms})
        best = min(shapes, key=lambda s: row[f"rms_{s}_pct"])
        row.update({"best_shape": best, "t_mean_best": row[f"t_mean_{best}"], "sigma_t_best": row[f"sigma_t_{best}"]})
        ref = fits.get("box", fits[best])
        tf, kf, Af, tmf, sgf, rmsf = fit_free(z0, d0s, z1, d1, tmax=ref[1] + 3 * ref[2] + 1.5, alpha=a.alpha)
        row.update({"A_free": Af, "t_mean_free": tmf, "sigma_t_free": sgf, "rms_free_pct": rmsf})
        if tk is not None: row.update({"t_mean_true": tk[2], "sigma_t_true": tk[3], "truth": tk[4]})
        rows.append(row); keep.append((var, E / A_nuc, z0, d0s, z1, d1, fits, (tf, kf), tk, row))
        print(f"{var:>6s} {E / A_nuc:7.2f} MeV/u  t_mean " + " ".join(f"{s} {row[f't_mean_{s}']:5.2f}" for s in shapes) + f" free {tmf:5.2f}"
              + (f" true {tk[2]:5.2f}" if tk else "") + " | sigma_t " + " ".join(f"{s} {row[f'sigma_t_{s}']:4.2f}" for s in shapes)
              + f" free {sgf:4.2f}" + (f" true {tk[3]:4.2f}" if tk else "") + f" | best {best} ({row[f'rms_{best}_pct']:.2f} %)"
              + f" | quadrature {row['sigma_quad_raw']:4.2f} | dR80 {dR80:4.2f}")

    cols = []
    for r in rows:
        for c in r:
            if c not in cols: cols.append(c)
    with open(os.path.join(out, "kernfit.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: (r["variant"], r["energy_MeV_u"])):
            fh.write(",".join(f"{r[c]:.4f}" if isinstance(r.get(c), (float, np.floating)) else '"' + str(r.get(c, "")).replace('"', "'") + '"'
                              if isinstance(r.get(c), str) else str(r.get(c, "")) for c in cols) + "\n")
    variants = sorted({r["variant"] for r in rows})
    for v in variants: summary([r for r in rows if r["variant"] == v], v, shapes)
    if not a.no_plots:
        for v in variants:
            suffix = "" if len(variants) == 1 else f"_{v}"
            plots([k for k in keep if k[0] == v], out, suffix, shapes, v)
        if len(variants) > 1: plot_overview(rows, os.path.join(out, "kernfit_uebersicht.png"))
    print(f"-> {os.path.join(out, 'kernfit.csv')}" + ("" if a.no_plots else " and plots"))


def summary(rows, v, shapes):
    get = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    best = max({r["best_shape"] for r in rows}, key=lambda s: sum(r["best_shape"] == s for r in rows))
    print(f"\n{v}: mean over {len(rows)} curves (standard deviation), mostly best shape: {best}")
    for k in [f"t_mean_{s}" for s in shapes] + ["t_mean_free", "t_mean_true", "dR80"] + [f"sigma_t_{s}" for s in shapes] + \
             ["sigma_t_free", "sigma_t_true", "sigma_quad_raw", "sigma_quad_gauss"] + [f"rms_{s}_pct" for s in shapes]:
        x = get(k)
        if np.isfinite(x).any(): print(f"  {k:18s} {np.nanmean(x):6.3f} ({np.nanstd(x):.3f})")


# ------------------------------------------------------------------ plots
COL = {"free": "#2a78d6", "gauss": "#eb6834", "box": "#1baf7a", "pyramid": "#eda100"}   # categorical slots 1 to 4, fixed order


def plot_true(tk, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    st = tk[0][1] - tk[0][0]
    ax.bar(tk[0], tk[1] / st, width=st, color="0.6")
    ax.set_xlabel("wasseräquivalente Dicke t [mm]"); ax.set_ylabel("Anteil pro mm")
    ax.set_title(f"Kern aus Geometrie: t̄ {tk[2]:.2f} mm, σ_t {tk[3]:.2f} mm", fontsize=10); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plots(keep, out, suffix, shapes, v):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keep = sorted(keep, key=lambda k: k[1])
    pick = [keep[0], keep[len(keep) // 2], keep[-1]] if len(keep) >= 3 else keep
    # 1) curves, fits, residuals
    fig, ax = plt.subplots(2, len(pick), figsize=(4.6 * len(pick), 6), sharex="col", gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for j, (_, E, z0, d0, z1, d1, fits, free, tk, row) in enumerate(pick):
        m1 = metrics(z1, smooth(d1)); lo, hi = m1["peak"] - 25, m1["R05"] + 6
        s = d1.max(); sel = (z1 >= lo) & (z1 <= hi)
        ax[0, j].plot(z0, d0 / s, color="0.6", lw=1.2, label="ohne RiFi")
        ax[0, j].plot(z1[sel], d1[sel] / s, ".", color="k", ms=2, label="mit RiFi")
        for shape in shapes:
            p = fits[shape]; mm = model(z1[sel], z0, d0, *p, shape)
            ax[0, j].plot(z1[sel], mm / s, color=COL[shape], lw=1.3, label=f"Faltung {shape}: t̄ {p[1]:.2f}, σ {p[2]:.2f} mm")
            ax[1, j].plot(z1[sel], (mm - d1[sel]) / s * 100, color=COL[shape], lw=1)
        ax[0, j].set_xlim(lo, hi); ax[0, j].set_title(f"{v}, {E:.1f} MeV/u"); ax[0, j].legend(fontsize=7); ax[0, j].grid(alpha=.3)
        ax[1, j].axhline(0, color="0.5", lw=.7); ax[1, j].set_xlabel("Tiefe in Wasser [mm]"); ax[1, j].grid(alpha=.3)
    ax[0, 0].set_ylabel("Dosis (relativ)"); ax[1, 0].set_ylabel("Modell − Kurve [%]")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"kernfit_kurven{suffix}.png"), dpi=130); plt.close(fig)
    # 2) kernel shapes at the middle energy
    _, E, z0, d0, z1, d1, fits, (tf, kf), tk, row = pick[len(pick) // 2]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if tk is not None:                                          # shown on 0.1 mm bins (fine structure of the mesh is real but noisy)
        e = np.arange(np.floor(tk[0][0] * 10) / 10, tk[0][-1] + 0.2, 0.1)
        wb, _ = np.histogram(tk[0], e, weights=tk[1])
        ax.bar(0.5 * (e[1:] + e[:-1]), wb / 0.1, width=0.095, color="0.82", label=f"wahr aus Geometrie (t̄ {tk[2]:.2f}, σ {tk[3]:.2f})")
    step = tf[1] - tf[0]
    ax.step(tf, kf / step, where="mid", color=COL["free"], lw=2, label=f"frei gefittet (t̄ {row['t_mean_free']:.2f}, σ {row['sigma_t_free']:.2f})")
    tt = np.linspace(min(0, tf[0]), tf[-1] + 1, 800)
    for shape in shapes:
        p = fits[shape]
        ax.plot(tt, kernel_density(shape, p[1], p[2], tt), color=COL[shape], lw=2, label=f"{shape} (t̄ {p[1]:.2f}, σ {p[2]:.2f})")
    ymax = max(1.2 * max(kernel_density(s, fits[s][1], fits[s][2], tt).max() for s in shapes), 1.2 * (kf / step).max(),
               1.1 * (wb / 0.1).max() if tk is not None else 0)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("wasseräquivalente Dicke t [mm]"); ax.set_ylabel("Anteil pro mm"); ax.set_title(f"RiFi-Kern {v} bei {E:.1f} MeV/u")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, f"kernfit_kern{suffix}.png"), dpi=130); plt.close(fig)
    # 3) t_mean and sigma_t over energy
    Es = np.array([k[1] for k in keep]); R = [k[-1] for k in keep]; g = lambda key: np.array([r.get(key, np.nan) for r in R], float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for key in list(shapes) + ["free"]:
        lab = "frei" if key == "free" else key
        ax[0].plot(Es, g(f"t_mean_{key}"), "o", ms=4, color=COL[key], label=lab); ax[1].plot(Es, g(f"sigma_t_{key}"), "o", ms=4, color=COL[key], label=lab)
    ax[0].plot(Es, g("dR80"), "x", color="0.4", label="R80-Verschiebung")
    ax[1].plot(Es, g("sigma_quad_raw"), "x", color="0.4", label="√(BPW80²−BPW80²)")
    ax[1].plot(Es, g("sigma_quad_gauss"), "+", color="0.4", label="dieselbe / 1,336")
    if np.isfinite(g("t_mean_true")).any():
        ax[0].axhline(np.nanmean(g("t_mean_true")), color="k", lw=1, ls="--", label="wahr")
        ax[1].axhline(np.nanmean(g("sigma_t_true")), color="k", lw=1, ls="--", label="wahr")
    ax[0].set_ylabel("t̄ [mm Wasser]"); ax[1].set_ylabel("σ_t [mm Wasser]")
    for x in ax: x.set_xlabel("Energie [MeV/u]"); x.grid(alpha=.3); x.legend(fontsize=7)
    fig.suptitle(f"RiFi-Kern je Energie: {v}"); fig.tight_layout(); fig.savefig(os.path.join(out, f"kernfit_energie{suffix}.png"), dpi=130); plt.close(fig)


def plot_overview(rows, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
    vs = sorted({r["variant"] for r in rows})
    for k, v in enumerate(vs):
        R = [r for r in rows if r["variant"] == v and np.isfinite(r.get("t_mean_true", np.nan))]
        if not R: continue
        for j, q in enumerate(("t_mean", "sigma_t")):
            ax[j].plot([r[f"{q}_true"] for r in R], [r[f"{q}_best"] for r in R], "o", ms=5, color=f"C{k % 10}", label=v if j == 0 else None)
    for j, (q, lab) in enumerate((("t_mean", "t̄"), ("sigma_t", "σ_t"))):
        lim = ax[j].get_xlim(); lo, hi = min(lim[0], ax[j].get_ylim()[0]), max(lim[1], ax[j].get_ylim()[1])
        ax[j].plot([lo, hi], [lo, hi], "k--", lw=.8); ax[j].set_xlabel(f"{lab} aus Geometrie [mm Wasser]"); ax[j].set_ylabel(f"{lab} gefittet (beste Form) [mm Wasser]")
        ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=8, ncol=2); fig.suptitle("Gefitteter Kern gegen Geometrie, alle Varianten und Energien")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
