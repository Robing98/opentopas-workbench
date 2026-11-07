"""Beam profile monitors and lateral dose profiles: readers, fits, and the data behind the Beam profile view.

Monitor outputs follow the names the generators write (and the OTPAS-1866 cases used):
    mon_<position in cm>_<case>_<p|e|o|all>.bin|.csv   2D fluence map of one plane (p protons, e electrons, o other)
    mon_<position in cm>s_<case>_<spec|espec|ospec>.csv energy spectrum of the same plane
The position is in cm before the isocenter, with "p" for the decimal point (19p52 = 19.52 cm).
Lateral dose profiles come from any 3D grid (TOPAS binary with its .binheader, or TOPAS CSV) with X, Y and Z binned;
Z is taken as depth, bin 0 first. Lateral coordinates are relative to the center of the scoring box.

Standard library only.
"""
import array, math, os, re, sys, threading

UNIT_MM = {"um": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1000.0}
MON_RE = re.compile(r"^mon_(\d+(?:p\d+)?)(s?)_(.+)_(p|e|o|all|spec|espec|ospec)$")
KINDS = ("p", "e", "o", "all")
SPECS = ("spec", "espec", "ospec")

_CACHE, _LOCK, _MAX_BYTES = {}, threading.Lock(), 256 * 2 ** 20   # a 160 x 160 x 150 dose grid is 31 MB


# ------------------------------------------------------------------ reading
def _header(path):
    from .server import read_topas_header  # late: server imports this module
    return read_topas_header(path)


def _dims_mm(dims):
    out = []
    for d in dims:
        f = UNIT_MM.get(d["unit"])
        if f is None: raise ValueError(f"unknown length unit {d['unit']}")
        out.append({"name": d["name"], "n": d["n"], "w": d["width"] * f})
    return out


class Grid:
    """A scored X/Y/Z grid with x fastest in `data`, bin widths in mm."""
    def __init__(self, dims, quantity, data):
        if len(dims) != 3: raise ValueError("not a 3D binned TOPAS output (X, Y, Z)")
        self.dims, self.quantity, self.data = dims, quantity, data
        self.nx, self.ny, self.nz = (d["n"] for d in dims)
        self.dx, self.dy, self.dz = (d["w"] for d in dims)
        if len(data) != self.nx * self.ny * self.nz: raise ValueError("data size does not match the header")

    def axis(self, a):
        """Bin centers of a lateral axis (0 = x, 1 = y), centered on the box, in mm."""
        n, w = (self.nx, self.dx) if a == 0 else (self.ny, self.dy)
        return [(i + 0.5) * w - n * w / 2 for i in range(n)]

    def plane(self, k):
        n = self.nx * self.ny
        return self.data[k * n:(k + 1) * n]


def read_grid(path):
    """Grid of a TOPAS binary (.bin with .binheader) or CSV output, cached by modification time and size."""
    st = os.stat(path)
    key = (path, st.st_mtime, st.st_size)
    with _LOCK:
        if key in _CACHE: return _CACHE[key]
    g = _read_bin(path) if path.lower().endswith(".bin") else _read_csv(path)
    with _LOCK:
        for k in [k for k in _CACHE if k[0] == path]: del _CACHE[k]
        while _CACHE and sum(8 * len(v.data) for v in _CACHE.values()) + 8 * len(g.data) > _MAX_BYTES:
            del _CACHE[next(iter(_CACHE))]
        _CACHE[key] = g
    return g


def _read_bin(path):
    hdr = path[:-4] + ".binheader"
    if not os.path.exists(hdr): raise ValueError(f"{os.path.basename(hdr)} is missing next to the binary file")
    dims, quantity, stats = _header(hdr)
    dims = _dims_mm(dims)
    n = 1
    for d in dims: n *= d["n"]
    size = os.path.getsize(path)
    if size != 8 * n:
        raise ValueError("binary file holds several statistics or does not match its header; score one statistic (Sum)")
    data = array.array("d")
    with open(path, "rb") as fh: data.fromfile(fh, n)
    if sys.byteorder == "big": data.byteswap()   # TOPAS writes little-endian doubles on the usual platforms
    return Grid(dims, quantity, data)


def _read_csv(path):
    with open(path, errors="replace") as fh: lines = fh.readlines()
    from .server import topas_header
    dims, quantity, stats = topas_header(lines)
    dims = _dims_mm(dims)
    if len(dims) != 3: raise ValueError("not a 3D binned TOPAS output (X, Y, Z)")
    nx, ny, nz = (d["n"] for d in dims)
    data = array.array("d", bytes(8 * nx * ny * nz))
    for l in lines:
        if l.startswith("#") or not l.strip(): continue
        p = l.replace(",", " ").split()
        if len(p) < 4: continue
        i, j, k = int(p[0]), int(p[1]), int(p[2])
        data[i + nx * (j + ny * k)] = float(p[3])
    return Grid(dims, quantity, data)


def grid_dims(path):
    """Header dims in mm without reading the data, or None if the file is no 3D grid."""
    try:
        hdr = path[:-4] + ".binheader" if path.lower().endswith(".bin") else path
        dims = _dims_mm(_header(hdr)[0])
    except (OSError, ValueError):
        return None
    return dims if len(dims) == 3 else None


def read_spectrum(path):
    """Energy spectrum from a TOPAS CSV binned by incident track energy: bin centers [MeV], values, under/overflow."""
    with open(path, errors="replace") as fh: lines = fh.readlines()
    m, quantity = None, ""
    for l in lines:
        if not l.startswith("#"): continue
        m = m or re.search(r"in\s+(\d+)\s+bins\s+of\s+([\d.eE+\-]+)\s*(\w+)\s+from\s+([\d.eE+\-]+)\s*\w*\s+to", l)
        q = re.match(r"#\s*(.+?)\s*:\s*(Sum|Mean)", l)
        if q: quantity = q.group(1)
    if not m: raise ValueError("no energy binning in the header")
    n, w, e0 = int(m.group(1)), float(m.group(2)), float(m.group(4))
    if m.group(3) == "keV": w, e0 = w / 1000, e0 / 1000
    rows = [l for l in lines if l.strip() and not l.startswith("#")]
    if len(rows) != 1: raise ValueError("expected one row (a spectrum of one unbinned plane)")
    vals = [float(v) for v in rows[0].replace(",", " ").split()]
    if len(vals) == n + 6: vals = vals[3:]      # leading bin indices i, j, k
    if len(vals) != n + 3: raise ValueError(f"expected {n + 3} values, found {len(vals)}")
    e = [e0 + (i + 0.5) * w for i in range(n)]
    y = vals[1:n + 1]
    res = {"e": e, "y": y, "under": vals[0], "over": vals[n + 1], "quantity": quantity, "width": w}
    tot = sum(y)
    if tot > 0:
        res["mean"] = sum(a * b for a, b in zip(e, y)) / tot
        i = max(range(n), key=y.__getitem__)
        res["peak"] = e[i]
        res["fwhm"] = fwhm(e, y)
    return res


# ------------------------------------------------------------------ fits
def _solve(a, b):
    """Gaussian elimination with partial pivoting; a is n x n, b length n."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-300: raise ZeroDivisionError("singular")
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            if f:
                for k in range(c, n + 1): m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def _lm(model, p, x, y, w, iters=200):
    """Levenberg-Marquardt for a model(p, x) -> (value, gradient); returns the parameters and the weighted chi2."""
    def chi(q):
        return sum(wi * (yi - model(q, xi)[0]) ** 2 for xi, yi, wi in zip(x, y, w))
    c0, lam, n = chi(p), 1e-3, len(p)
    for _ in range(iters):
        jtj = [[0.0] * n for _ in range(n)]; jtr = [0.0] * n
        for xi, yi, wi in zip(x, y, w):
            f, g = model(p, xi); r = yi - f
            for a in range(n):
                jtr[a] += wi * g[a] * r
                for b in range(a, n): jtj[a][b] += wi * g[a] * g[b]
        for a in range(n):
            for b in range(a): jtj[a][b] = jtj[b][a]
        improved = False
        while lam < 1e12:
            m = [[jtj[a][b] * (1 + lam if a == b else 1) for b in range(n)] for a in range(n)]
            try: dp = _solve(m, jtr)
            except ZeroDivisionError: lam *= 10; continue
            q = [pa + d for pa, d in zip(p, dp)]
            c1 = chi(q)
            if c1 <= c0:
                done = c0 - c1 <= 1e-12 * max(c0, 1e-300)
                p, c0, lam, improved = q, c1, max(lam / 10, 1e-12), True
                if done: return p, c0
                break
            lam *= 10
        if not improved: break
    return p, c0


def _gauss(p, x):
    a, mu, s = p
    s = s if abs(s) > 1e-12 else 1e-12
    e = math.exp(-0.5 * ((x - mu) / s) ** 2)
    return a * e, (e, a * e * (x - mu) / s ** 2, a * e * (x - mu) ** 2 / s ** 3)


def _gauss2(p, x):
    a1, mu, s1, a2, s2 = p
    f1, g1 = _gauss((a1, mu, s1), x)
    f2, g2 = _gauss((a2, mu, s2), x)
    return f1 + f2, (g1[0], g1[1] + g2[1], g1[2], g2[0], g2[2])


def gauss_core(x, y, frac=0.1):
    """Gaussian fit to the points above `frac` of the maximum: (amplitude, mean, sigma) or None."""
    if len(x) < 3: return None
    top = max(y)
    if top <= 0: return None
    pts = [(xi, yi) for xi, yi in zip(x, y) if yi >= frac * top]
    if len(pts) < 3: return None
    sw = sum(v for _, v in pts)
    mu = sum(a * v for a, v in pts) / sw
    s = math.sqrt(max(sum(v * (a - mu) ** 2 for a, v in pts) / sw, 1e-12))
    # a Gaussian cut at 10 % has an rms of about 0.8 sigma; start a bit wide so the fit converges from above
    p, _ = _lm(_gauss, [top, mu, s * 1.2], [a for a, _ in pts], [v for _, v in pts], [1.0] * len(pts))
    if not (p[2] and math.isfinite(p[2])): return None
    return p[0], p[1], abs(p[2])


def double_gauss(x, y, core):
    """Core plus halo Gaussian on a common center, fitted to all points with Poisson-like weights so the tails count.

    Returns (sigma1, sigma2, halo weight) with sigma2 > sigma1, the weight being the halo's share of the area."""
    a, mu, s = core
    top = max(y)
    floor = 1e-4 * top
    w = [1.0 / max(v, floor) for v in y]
    p, _ = _lm(_gauss2, [a * 0.98, mu, s * 0.97, a * 0.02, s * 3], x, y, w)
    a1, mu, s1, a2, s2 = p[0], p[1], abs(p[2]), p[3], abs(p[4])
    if s2 < s1: a1, s1, a2, s2 = a2, s2, a1, s1
    area1, area2 = a1 * s1, a2 * s2
    if not all(math.isfinite(v) for v in (a1, a2, s1, s2)) or area1 <= 0 or area2 < 0 or area1 + area2 <= 0: return None
    halo = area2 / (area1 + area2)
    # On a profile without a halo the fit splits the one Gaussian into two of similar width (seen on a real monitor
    # plane: sigma1 1.88, sigma2 2.97, halo 99.9 %). Such a split says nothing about a halo, so report none.
    if s2 < 1.2 * s1 or halo > 0.5: return None
    return s1, s2, halo


def fwhm(x, y):
    if not y: return None
    i = max(range(len(y)), key=y.__getitem__); h = y[i] / 2
    if h <= 0: return None
    # Search from the ends inward: just behind a ripple filter the profile carries the groove pattern, and a walk
    # from the peak outward stops in the first dip below half maximum (7.2 mm instead of 7.9 mm on the 10 cm plane
    # 20 cm behind the filter). A single spike at the peak still raises the half maximum; the fitted sigma does not.
    def edge(step):
        k = 0 if step > 0 else len(y) - 1
        while k != i:
            if y[k + step] >= h > y[k]:
                return x[k] + (x[k + step] - x[k]) * (h - y[k]) / (y[k + step] - y[k])
            k += step
        return None
    lo, hi = edge(1), edge(-1)
    return hi - lo if lo is not None and hi is not None else None


def profile_stats(x, y):
    """Centroid, rms, FWHM, Gaussian core sigma and mean, halo share outside 3 sigma, and the double Gaussian."""
    tot = sum(y)
    if tot <= 0 or len(x) < 3: return {}
    c = sum(a * b for a, b in zip(x, y)) / tot
    out = {"centroid": c, "rms": math.sqrt(max(sum(b * (a - c) ** 2 for a, b in zip(x, y)) / tot, 0)), "fwhm": fwhm(x, y),
           "total": tot}
    core = gauss_core(x, y)
    if core:
        out["sigma"], out["mu"] = core[2], core[1]
        out["halo3s"] = sum(b for a, b in zip(x, y) if abs(a - core[1]) > 3 * core[2]) / tot
        try: dg = double_gauss(x, y, core)
        except (OverflowError, ValueError): dg = None
        if dg: out["sigma1"], out["sigma2"], out["halo_dg"] = dg
    return out


def _clean(d):
    """JSON has no NaN or infinity; round to keep the answer small."""
    if isinstance(d, dict): return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)): return [_clean(v) for v in d]
    if isinstance(d, float): return float(f"{d:.6g}") if math.isfinite(d) else None
    return d


# ------------------------------------------------------------------ profiles of one plane
def plane_profiles(g, k=0, mode="int"):
    """x and y profiles of plane k: integrated over the other axis (value times bin width) or the cut through the
    row and column nearest the 2D centroid."""
    pl = g.plane(k); nx, ny = g.nx, g.ny
    px = [sum(pl[i::nx]) * g.dy for i in range(nx)]
    py = [sum(pl[j * nx:(j + 1) * nx]) * g.dx for j in range(ny)]
    if mode == "cut":
        xs, ys = g.axis(0), g.axis(1)
        tx, ty = sum(px), sum(py)
        cx = sum(a * b for a, b in zip(xs, px)) / tx if tx > 0 else 0.0
        cy = sum(a * b for a, b in zip(ys, py)) / ty if ty > 0 else 0.0
        i0 = min(range(nx), key=lambda i: abs(xs[i] - cx)); j0 = min(range(ny), key=lambda j: abs(ys[j] - cy))
        px = list(pl[j0 * nx:(j0 + 1) * nx]); py = list(pl[i0::nx])
    return px, py


def plane_map(g, k=0):
    pl = g.plane(k); nx = g.nx
    return [[float(f"{v:.4g}") for v in pl[j * nx:(j + 1) * nx]] for j in range(g.ny)]


def monitor(path, mode="int"):
    g = read_grid(path)
    if g.nz != 1: raise ValueError("a monitor plane has one Z bin")
    xs, ys = g.axis(0), g.axis(1)
    px, py = plane_profiles(g, 0, mode)
    return _clean({"quantity": g.quantity, "dims": g.dims, "x": xs, "y": ys, "px": px, "py": py, "map": plane_map(g),
                   "fx": profile_stats(xs, px), "fy": profile_stats(ys, py)})


# ------------------------------------------------------------------ dose grids
def dose_depth(path, k, mode="int"):
    g = read_grid(path)
    k = max(0, min(int(k), g.nz - 1))
    xs, ys = g.axis(0), g.axis(1)
    px, py = plane_profiles(g, k, mode)
    return _clean({"quantity": g.quantity, "dims": g.dims, "k": k, "depth": (k + 0.5) * g.dz, "x": xs, "y": ys,
                   "px": px, "py": py, "map": plane_map(g, k), "fx": profile_stats(xs, px), "fy": profile_stats(ys, py),
                   "sigma_z": sigma_vs_depth(path, mode)})


_SIGMA = {}


def sigma_vs_depth(path, mode="int", min_share=0.02):
    """Fits per depth bin, for the bins that hold at least `min_share` of the largest plane's integral."""
    st = os.stat(path); key = (path, st.st_mtime, st.st_size, mode)
    with _LOCK:
        if key in _SIGMA: return _SIGMA[key]
    g = read_grid(path)
    xs, ys = g.axis(0), g.axis(1)
    n = g.nx * g.ny
    sums = [sum(g.data[k * n:(k + 1) * n]) for k in range(g.nz)]
    top = max(sums) if sums else 0
    rows = []
    for k in range(g.nz):
        if top <= 0 or sums[k] < min_share * top: continue
        px, py = plane_profiles(g, k, mode)
        fx, fy = profile_stats(xs, px), profile_stats(ys, py)
        rows.append({"k": k, "depth": (k + 0.5) * g.dz, "integral": sums[k] * g.dx * g.dy,
                     "fx": {a: fx.get(a) for a in ("sigma", "centroid", "fwhm", "halo3s", "halo_dg", "sigma2")},
                     "fy": {a: fy.get(a) for a in ("sigma", "centroid", "fwhm", "halo3s", "halo_dg", "sigma2")}})
    rows = _clean(rows)
    with _LOCK:
        for k in [k for k in _SIGMA if k[0] == path]: del _SIGMA[k]
        _SIGMA[key] = rows
    return rows


# ------------------------------------------------------------------ folder scan and envelope
def pos_cm(token):
    return float(token.replace("p", "."))


def scan(folder, rel):
    """Cases with their monitor planes and the 3D dose grids of one folder (not recursive)."""
    cases, dose = {}, []
    try: names = sorted(os.listdir(folder))
    except OSError as e: raise ValueError(str(e))
    for f in names:
        base, ext = os.path.splitext(f)
        if ext.lower() not in (".bin", ".csv"): continue
        m = MON_RE.match(base)
        relf = f"{rel}/{f}" if rel else f
        if m:
            pos, spec, case, kind = pos_cm(m.group(1)), m.group(2), m.group(3), m.group(4)
            if bool(spec) != (kind in SPECS): continue
            c = cases.setdefault(case, {})
            mon = c.setdefault(pos, {"pos": pos, "maps": {}, "spectra": {}})
            (mon["spectra"] if spec else mon["maps"])[kind] = relf
            continue
        if base.startswith("mon_"): continue
        dims = grid_dims(os.path.join(folder, f))
        if dims and all(d["n"] > 1 for d in dims):
            dose.append({"path": relf, "name": base, "nz": dims[2]["n"], "dz": dims[2]["w"]})
    out = []
    for name in sorted(cases):
        mons = sorted(cases[name].values(), key=lambda m: -m["pos"])   # upstream first
        out.append({"name": name, "monitors": mons, "kinds": sorted({k for m in mons for k in m["maps"]}, key=KINDS.index),
                    "dose": [d["path"] for d in dose if d["name"] == name or d["name"].endswith("_" + name)]})
    return {"cases": out, "dose": dose}


def envelope(folder, rel, case, kind="p", mode="int"):
    """Fits per monitor plane of one case and particle kind, upstream first."""
    found = next((c for c in scan(folder, rel)["cases"] if c["name"] == case), None)
    if not found: raise ValueError(f"no monitors for case {case}")
    rows = []
    for m in found["monitors"]:
        relf = m["maps"].get(kind)
        if not relf: continue
        g = read_grid(os.path.join(folder, os.path.basename(relf)))
        xs, ys = g.axis(0), g.axis(1)
        px, py = plane_profiles(g, 0, mode)
        fx, fy = profile_stats(xs, px), profile_stats(ys, py)
        rows.append({"pos": m["pos"], "fx": fx, "fy": fy})
    return _clean({"case": case, "kind": kind, "rows": rows})
