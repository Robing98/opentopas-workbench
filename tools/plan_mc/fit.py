"""Beam model fit (part A): mean energy and energy spread from depth doses, spot optics from spots in air.

Energy: no range-energy formula is assumed. A library of TOPAS depth doses in water at known (mean energy, spread)
gives R80 and the distal falloff D80-D20 on a grid; a measured (or simulated) curve is inverted on that grid. So the
fit uses the same physics as the recalculation that later runs with the fitted values.

    python3 fit.py library-write OUTDIR [--histories N]    # TOPAS cases of the library (queue them in the workbench)
    python3 fit.py library-read OUTDIR                     # -> OUTDIR/library.json (R80, D80-D20 per grid point)
    python3 fit.py fit INPUT.json LIBRARY.json MODEL.json  # fitted beam model (format of beam_model.py)

INPUT.json: {"name", "source_distance_mm", "energies": [{"nominal", "depth_dose": PATH or [[z_mm, dose], ...],
"planes": [{"z_mm": distance from the isocenter, + towards the source, "sigma_x", "sigma_y"}, ...] or "monitors": {...},
"particles_per_mu"}]}. A PATH is a TOPAS csv/binary scorer binned in Z (for example `dd_<tag>_voll` of messpunkte_v3)
or a two-column text file.

Optics: sigma^2(s) = sigma0^2 + 2 rho sigma0 theta s + theta^2 s^2 at the distance s downstream of the source plane
(Fermi-Eyges in vacuum). Air between source and plane adds spread; `air_sigma2` (mm^2 per plane, from a TOPAS spot run
with the fitted values in air minus the same in vacuum) is subtracted before the fit, else the model would count the
air twice.
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan_mc import topas_io  # noqa: E402

LIB_ENERGIES = [60.0 + 20.0 * i for i in range(9)]       # 60 to 220 MeV
LIB_SPREADS = [0.0, 0.4, 0.8, 1.2]                         # % of the mean energy
BIN_MM = 0.1


# ---------------------------------------------------------------- depth dose metrics
def distal(z, d, level):
    """Depth where the dose falls to `level` x maximum behind the peak (linear interpolation)."""
    z, d = np.asarray(z, float), np.asarray(d, float)
    i = int(np.argmax(d)); m = d[i]
    after = np.nonzero(d[i:] < level * m)[0]
    if not after.size: raise ValueError("curve does not fall below the level behind the peak")
    j = i + after[0]
    return z[j - 1] + (level * m - d[j - 1]) * (z[j] - z[j - 1]) / (d[j] - d[j - 1])


def smooth_peak(z, d):
    """Peak height from a parabola through the top bins: the raw maximum of a noisy curve biases R80 upwards."""
    i = int(np.argmax(d)); lo, hi = max(0, i - 5), min(len(d), i + 6)
    c = np.polyfit(z[lo:hi], d[lo:hi], 2)
    return float(c[2] - c[1] ** 2 / (4 * c[0])) if c[0] < 0 else float(d[i])


def metrics(z, d):
    z, d = np.asarray(z, float), np.asarray(d, float)
    m = smooth_peak(z, d); dd = d / m
    r80, r20 = distal(z, dd, 0.8), distal(z, dd, 0.2)
    return {"r80": float(r80), "dfo": float(r20 - r80), "peak": float(z[int(np.argmax(d))])}


def orient(z, d):
    """TOPAS bins a box from -HL to +HL whatever side the beam enters. The entrance side carries the plateau, the far
    side (beyond the range) nearly nothing: reverse when the first bin is the far one, so depth runs from the surface."""
    if d[0] < d[-1]: d = d[::-1]
    return z, np.asarray(d, float)


def load_curve(src):
    if isinstance(src, list): a = np.asarray(src, float); return a[:, 0], a[:, 1]
    if src.endswith((".csv", ".bin")) or os.path.exists(src + ".bin"):
        with open(src if not src.endswith(".bin") else src[:-4] + ".binheader") as fh:
            topas = fh.readline().startswith("# TOPAS")
        if topas: return orient(*topas_io.depth_curve(src))
    a = np.loadtxt(src, delimiter="," if src.endswith(".csv") else None, comments="#")
    return a[:, 0], a[:, -1]


# ---------------------------------------------------------------- library
def library_case(e, s, histories):
    tag = f"lib_e{e:g}_s{s:g}".replace(".", "p")
    return tag, f"""# Library depth dose for fit.py: {e:g} MeV, spread {s:g} %, water 20 x 20 cm, surface at z = 0, {BIN_MM} mm bins.
i:Ts/NumberOfThreads = 0
i:Ts/Seed = 1
b:Ts/PauseBeforeQuit = "False"
i:Ts/ShowHistoryCountAtInterval = 0
s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 0.5 m
d:Ge/World/HLY = 0.5 m
d:Ge/World/HLZ = 0.5 m
s:Ge/Water/Type = "TsBox"
s:Ge/Water/Parent = "World"
s:Ge/Water/Material = "G4_WATER"
d:Ge/Water/HLX = 10 cm
d:Ge/Water/HLY = 10 cm
d:Ge/Water/HLZ = 20 cm
d:Ge/Water/TransZ = -20 cm
i:Ge/Water/ZBins = {int(400 / BIN_MM)}
s:Sc/DD/Quantity = "DoseToMedium"
s:Sc/DD/Component = "Water"
s:Sc/DD/OutputType = "csv"
s:Sc/DD/OutputFile = "{tag}"
s:Sc/DD/IfOutputFileAlreadyExists = "Overwrite"
b:Sc/DD/OutputToConsole = "False"
s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPosition"
s:So/Beam/BeamParticle = "proton"
d:So/Beam/BeamEnergy = {e:g} MeV
u:So/Beam/BeamEnergySpread = {s:g}
s:So/Beam/BeamPositionDistribution = "Gaussian"
s:So/Beam/BeamPositionCutoffShape = "Ellipse"
d:So/Beam/BeamPositionCutoffX = 2 cm
d:So/Beam/BeamPositionCutoffY = 2 cm
d:So/Beam/BeamPositionSpreadX = 3 mm
d:So/Beam/BeamPositionSpreadY = 3 mm
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = {histories}
d:Ge/BeamPosition/TransZ = 1 mm
d:Ge/BeamPosition/RotX = 180 deg
"""


def library_write(out, histories=50000, energies=None, spreads=None):
    os.makedirs(out, exist_ok=True)
    cases = []
    for e in energies or LIB_ENERGIES:
        for s in spreads or LIB_SPREADS:
            tag, text = library_case(e, s, histories)
            open(os.path.join(out, tag + ".txt"), "w").write(text)
            cases.append({"tag": tag, "mean": e, "spread_pct": s})
    json.dump({"histories": histories, "cases": cases}, open(os.path.join(out, "cases.json"), "w"), indent=1)
    return cases


def library_read(out):
    meta = json.load(open(os.path.join(out, "cases.json")))
    rows = []
    for c in meta["cases"]:
        z, d = load_curve(os.path.join(out, c["tag"] + ".csv"))
        rows.append({**c, **metrics(z, d)})
    lib = {"histories": meta["histories"], "rows": rows}
    json.dump(lib, open(os.path.join(out, "library.json"), "w"), indent=1)
    return lib


class Library:
    """Bilinear interpolation of R80 and D80-D20 on the (mean energy, spread) grid; inversion by least squares."""

    def __init__(self, lib):
        rows = lib["rows"]
        self.e = np.array(sorted({r["mean"] for r in rows})); self.s = np.array(sorted({r["spread_pct"] for r in rows}))
        self.r80 = np.full((len(self.e), len(self.s)), np.nan); self.dfo = self.r80.copy()
        for r in rows:
            i, j = int(np.searchsorted(self.e, r["mean"])), int(np.searchsorted(self.s, r["spread_pct"]))
            self.r80[i, j], self.dfo[i, j] = r["r80"], r["dfo"]
        if np.isnan(self.r80).any(): raise ValueError("library grid incomplete")
        from scipy.interpolate import RectBivariateSpline
        ky, kx = min(3, len(self.s) - 1), min(3, len(self.e) - 1)
        self._r = RectBivariateSpline(self.e, self.s, self.r80, kx=kx, ky=ky)
        self._d = RectBivariateSpline(self.e, self.s, self.dfo, kx=kx, ky=ky)

    def predict(self, e, s): return float(self._r(e, s)[0, 0]), float(self._d(e, s)[0, 0])

    def invert(self, r80, dfo):
        from scipy.optimize import least_squares
        # R80 sets the energy almost alone; D80-D20 then sets the spread. Scale the residuals by their typical errors.
        e0 = float(np.interp(r80, self.r80[:, 0], self.e))
        f = lambda p: [(self.predict(*p)[0] - r80) / 0.05, (self.predict(*p)[1] - dfo) / 0.05]
        res = least_squares(f, [e0, float(self.s.mean())], bounds=([self.e[0], self.s[0]], [self.e[-1], self.s[-1]]))
        e, s = res.x
        at_edge = s <= self.s[0] + 1e-6 or s >= self.s[-1] - 1e-6
        return {"mean": float(e), "spread_pct": float(s), "spread_at_grid_edge": bool(at_edge),
                "residual_r80_mm": float(self.predict(e, s)[0] - r80), "residual_dfo_mm": float(self.predict(e, s)[1] - dfo)}


# ---------------------------------------------------------------- optics
def fit_optics(s_mm, sigma_mm, air_sigma2=None):
    """Fit sigma0 [mm], divergence [mrad], correlation at the source plane from sigma at distances s from it.

    Needs 3 or more planes. Returns also the fit residuals per plane [mm]."""
    s, sig = np.asarray(s_mm, float), np.asarray(sigma_mm, float)
    if s.size < 3: raise ValueError("optics fit needs 3 or more planes")
    y = sig ** 2 - (np.asarray(air_sigma2, float) if air_sigma2 is not None else 0.0)
    w = 1.0 / np.maximum(sig, 1e-6) ** 2          # equal relative weight per plane
    A = np.stack([np.ones_like(s), s, s ** 2], axis=1)
    a, b, c = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    if a <= 0 or c <= 0: raise ValueError(f"optics fit gave sigma0^2 = {a:.3g}, theta^2 = {c:.3g}: planes too close or noisy")
    sigma0, theta = np.sqrt(a), np.sqrt(c)
    rho = float(np.clip(b / (2 * sigma0 * theta), -1, 1))
    model = np.sqrt(np.maximum(a + b * s + c * s ** 2, 0))
    return {"sigma": float(sigma0), "div_mrad": float(1000 * theta), "corr": rho,
            "residual_mm": [float(v) for v in np.sqrt(np.maximum(y, 0)) - model]}


def sigma_profile(x_mm, f):
    """Gaussian core sigma of a 1D profile: fit to the bins above 10 % of the maximum (the halo stays out)."""
    from scipy.optimize import curve_fit
    x, f = np.asarray(x_mm, float), np.asarray(f, float)
    keep = f > 0.1 * f.max()
    g = lambda x, a, mu, sg: a * np.exp(-0.5 * ((x - mu) / sg) ** 2)
    mu0 = float((x[keep] * f[keep]).sum() / f[keep].sum())
    p, _ = curve_fit(g, x[keep], f[keep], p0=[f.max(), mu0, max(1.0, float(np.sqrt((f[keep] * (x[keep] - mu0) ** 2).sum() / f[keep].sum())))])
    return abs(float(p[2])), float(p[1])


def fit(inp, lib):
    L = Library(lib)
    zs = float(inp.get("source_distance_mm", 1150.0))
    rows, report = [], []
    for e in sorted(inp["energies"], key=lambda r: r["nominal"]):
        row = {"nominal": e["nominal"], "particles_per_mu": e.get("particles_per_mu", 1.0e8)}
        rep = {"nominal": e["nominal"]}
        if "depth_dose" in e:
            m = metrics(*load_curve(e["depth_dose"])); en = L.invert(m["r80"], m["dfo"])
            row.update(mean=en["mean"], spread_pct=en["spread_pct"]); rep.update(m, **en)
        else:
            row.update(mean=e["nominal"], spread_pct=e.get("spread_pct", 0.2)); rep["energy"] = "nominal (no depth dose)"
        pl = e.get("planes", [])
        for ax in ("x", "y"):
            if len(pl) >= 3:
                o = fit_optics([zs - p["z_mm"] for p in pl], [p["sigma_" + ax] for p in pl], [p.get("air_sigma2_" + ax, 0.0) for p in pl])
                row.update({"sigma_" + ax: o["sigma"], "div_" + ax: o["div_mrad"], "corr_" + ax: o["corr"]}); rep["optics_" + ax] = o
            else:
                row.update({"sigma_" + ax: e.get("sigma_" + ax, 2.97), "div_" + ax: e.get("div_" + ax, 1.0), "corr_" + ax: 0.0})
                rep["optics_" + ax] = "placeholder (fewer than 3 planes)"
        rows.append(row); report.append(rep)
    model = {"name": inp.get("name", "fitted"), "particle": "proton", "source_distance_mm": zs,
             "vsad_x_mm": inp.get("vsad_x_mm", 7549.0), "vsad_y_mm": inp.get("vsad_y_mm", 8219.0),
             "dose_scale": inp.get("dose_scale", 1.0), "energies": rows,
             "sources": {"energy": "fit.py library inversion (R80, D80-D20)", "optics": "fit.py sigma^2(s) quadratic"}}
    return model, report


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["library-write"] and len(a) >= 2:
        n = int(a[a.index("--histories") + 1]) if "--histories" in a else 50000
        print(len(library_write(a[1], n)), "library cases in", a[1])
    elif a[:1] == ["library-read"] and len(a) == 2:
        lib = library_read(a[1]); print(len(lib["rows"]), "rows ->", os.path.join(a[1], "library.json"))
    elif a[:1] == ["fit"] and len(a) == 4:
        m, rep = fit(json.load(open(a[1])), json.load(open(a[2])))
        json.dump(m, open(a[3], "w"), indent=1); json.dump(rep, open(a[3][:-5] + "_report.json", "w"), indent=1)
        print("beam model ->", a[3])
    else: sys.exit(__doc__)
