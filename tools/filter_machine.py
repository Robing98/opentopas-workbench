#!/usr/bin/env python3
"""Builds matRad proton machines for plan comparisons from a template machine: a constant WET layer grid, optionally
with a ripple filter folded into every depth dose and its extra spot size added in air.

Layer grid: the template's energies are about 3 mm WET apart (protons_Generic), too coarse for a 2 mm grid. So the
machine gets one energy per target peak position p_k = p_deep - k * spacing, made from the template entry with the
nearest peak, its depth tables shifted by p_k - peakPos (at most half the template step; over 1.5 mm the Bragg peak
width barely changes). The energy is interpolated over the peak position. With matRad's longitudinal spot spacing set
below the grid (e.g. 0.5 mm), every energy of the machine stays available and the grid is exactly the one built here.

Filter: Weber and Kraft 1999 as in rifi_kernel_fit.py and the convolution model of rifi_formvergleich:
    Z_with(z) = sum_i w_i * Z_without(z + t_i)
with the kernel (t_i, w_i) of the filter in mm WET. Lateral tables and LET move with the mean thickness. The peak
position and range are taken from the folded curve, so the grid of a filter machine is the same set of nominal
energies with peaks pulled back by about the mean thickness.

Extra spot size: the filter's angular spread adds sigma_x(E) in quadrature to the spot size in air at the isocentre,
sigma_x(E) = a * (E / 100 MeV)^b, fitted to TOPAS values behind the nozzle model (rifi_position, rifi_treppe).
The broadening that this spread keeps adding inside the patient (sigma(z)) is not modelled: matRad's focus table only
knows the spot size in air.

Kernels: steps4:MEAN:SIGMA (four equal steps, stp4 and stp3), box:MEAN:SIGMA (t4), gauss:MEAN:SIGMA, none.
Usage:
  python3 filter_machine.py --template protons_Generic.mat --name PV_stp4_464 --spacing 4.64 \
      --kernel steps4:1.74:1.30 --extra-sigma 4.40:-1.07 --out protons_PV_stp4_464.mat
"""
import argparse, copy, datetime, os, sys
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rifi_kernel_fit import kernel as smooth_kernel  # noqa: E402

DEPTH_KEYS = ("sigma", "sigma1", "sigma2", "weight", "LET")


def filter_kernel(spec):
    """'steps4:1.74:1.30' -> thicknesses [mm WET] and weights (sum 1); 'none' -> no filter."""
    if spec in (None, "", "none"): return np.array([0.0]), np.array([1.0])
    shape, mean, sig = spec.split(":"); mean, sig = float(mean), float(sig)
    if shape == "steps4":                        # four steps of equal area, step height h: sd = h * sqrt(1.25)
        h = sig / np.sqrt(1.25)
        return mean + h * np.array([-1.5, -0.5, 0.5, 1.5]), np.full(4, 0.25)
    return smooth_kernel(shape, mean, sig)


def extra_sigma(spec, energy):
    """Added spot sigma in air at the isocentre [mm] for 'a:b' (a * (E/100)^b), 0 for none."""
    if spec in (None, "", "none"): return 0.0
    a, b = (float(x) for x in spec.split(":"))
    return a * (energy / 100.0) ** b


def r80_after_peak(z, y):
    i = int(np.argmax(y)); lvl = 0.8 * y[i]
    for k in range(i, len(y) - 1):
        if y[k] >= lvl > y[k + 1]: return z[k] + (z[k + 1] - z[k]) * (y[k] - lvl) / (y[k] - y[k + 1])
    return float("nan")


def fine_peak(z, y):
    """Peak position from a parabola through the maximum and its neighbours (the tables are 0.2 mm near the peak)."""
    i = int(np.argmax(y))
    if 0 < i < len(y) - 1 and z[i + 1] - z[i] == z[i] - z[i - 1]:
        d = y[i - 1] - 2 * y[i] + y[i + 1]
        if d < 0: return z[i] + 0.5 * (z[i] - z[i - 1]) * (y[i - 1] - y[i + 1]) / d
    return z[i]


def dense(z, y, step=0.05):
    """Curve on a uniform grid (the template grid is irregular, which biases the parabola of fine_peak)."""
    zz = np.arange(0.0, float(z[-1]) + 1e-9, step)
    return zz, np.interp(zz, np.asarray(z, float), np.asarray(y, float))


def col(x): return np.asarray(x, float).reshape(-1, 1)


def build(template, spacing, kernel_spec="none", sigma_spec="none", depth_min=0.0, depth_max=None):
    """New data array (1 x N record array like the template's) and a list of (energy, peak, range, sigma_air)."""
    data = template["machine"]["data"][0, 0]
    ent = [data[0, j] for j in range(data.shape[1])]
    E = np.array([float(e["energy"].ravel()[0]) for e in ent])
    # peaks from the curves: the template's peakPos differs from the maximum of its own Z by up to a few tenths of a mm
    P = np.array([fine_peak(*dense(e["depths"].ravel(), e["Z"].ravel())) for e in ent])
    t, w = filter_kernel(kernel_spec)
    tmean = float(np.sum(t * w))
    hi = P[-1] if depth_max is None else min(depth_max, P[-1])
    targets = np.arange(hi, max(P[0], depth_min) - 1e-9, -spacing)[::-1]
    out, info = [], []
    for p in targets:
        j = int(np.argmin(abs(P - p))); s = p - P[j]; e0 = ent[j]
        z = e0["depths"].ravel().astype(float)
        energy = float(np.interp(p, P, E))
        # shifted template curve: value at depth z is the template's at z - s (deeper peak for s > 0)
        Z0 = np.interp(z - s, z, e0["Z"].ravel(), left=e0["Z"].ravel()[0], right=0.0)
        # dense uniform grid so that folding and the peak search do not depend on the irregular template grid
        zz = np.arange(0.0, z[-1] + s + 1e-9, 0.05)
        Zd = np.interp(zz, z, Z0, left=Z0[0], right=0.0)
        Zf = sum(wi * np.interp(zz + ti, zz, Zd, left=Zd[0], right=0.0) for ti, wi in zip(t, w))
        # the folded peak relative to the unfolded one: both are quantised alike by the linear template grid, so the
        # difference is clean and an unfiltered machine keeps exactly the grid p_k
        peak = p + fine_peak(zz, Zf) - fine_peak(zz, Zd)
        r80 = r80_after_peak(zz, Zf)
        # output depth grid as in the template: coarse plateau, 0.1 mm around the peak, 0.5 mm in the tail
        last = zz[np.where(Zf > 1e-4 * Zf.max())[0][-1]]
        g = np.unique(np.round(np.concatenate([np.arange(0, max(peak - 15, 0), 1.0), np.arange(max(peak - 15, 0), peak + 8, 0.1),
                                               np.arange(peak + 8, last + 0.5, 0.5)]), 3))
        e = copy.deepcopy(e0)
        e["energy"] = np.array([[energy]]); e["peakPos"] = np.array([[peak]]); e["range"] = np.array([[r80]])
        e["depths"] = col(g); e["Z"] = col(np.interp(g, zz, Zf))
        for k in DEPTH_KEYS:
            if k in e.dtype.names:
                v = e0[k].ravel().astype(float)
                e[k] = col(np.interp(g - s + tmean, z, v, left=v[0], right=v[-1]))
        if "sigma1" in e.dtype.names: e["sigma1"][0, 0] = 0.0      # as in the template: no spread at the surface
        if "sigma" in e.dtype.names: e["sigma"][0, 0] = 0.0
        sp = e["energySpectrum"][0, 0]
        rel = float(sp["sigma"].ravel()[0]) / float(sp["mean"].ravel()[0])
        sp["mean"] = np.array([[energy]]); sp["sigma"] = np.array([[energy * rel]])
        f = e["initFocus"][0, 0]
        add2 = extra_sigma(sigma_spec, energy) ** 2
        sig = np.sqrt(f["sigma"].astype(float) ** 2 + add2)
        f["sigma"] = sig
        f["SisFWHMAtIso"] = np.array([[float(np.sqrt((float(f["SisFWHMAtIso"].ravel()[0]) / 2.3548) ** 2 + add2) * 2.3548)]])
        em = f["emittance"][0, 0]
        for k in ("sigmaX", "sigmaY"):
            em[k] = np.array([[float(np.sqrt(float(em[k].ravel()[0]) ** 2 + add2))]])
        out.append(e)
        info.append((energy, peak, r80, float(f["SisFWHMAtIso"].ravel()[0]) / 2.3548))
    arr = np.empty((1, len(out)), dtype=data.dtype)
    for j, e in enumerate(out): arr[0, j] = e
    return arr, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--template", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--spacing", type=float, required=True, help="constant peak distance [mm WET]")
    ap.add_argument("--kernel", default="none", help="filter kernel, e.g. steps4:1.74:1.30 or box:2.58:1.32")
    ap.add_argument("--extra-sigma", default="none", help="a:b for sigma_x = a * (E/100 MeV)^b [mm], added in quadrature")
    ap.add_argument("--depth-min", type=float, default=0.0, help="shallowest peak to keep [mm]")
    ap.add_argument("--depth-max", type=float, help="deepest peak, start of the grid [mm] (default: the template's)")
    ap.add_argument("--out")
    a = ap.parse_args()
    tpl = sio.loadmat(a.template)
    arr, info = build(tpl, a.spacing, a.kernel, a.extra_sigma, a.depth_min, a.depth_max)
    mach = copy.deepcopy(tpl["machine"])
    mach["data"][0, 0] = arr
    meta = mach["meta"][0, 0]
    meta["name"] = np.array([a.name]); meta["created_by"] = np.array(["filter_machine.py (OpenTOPAS Workbench)"])
    meta["created_on"] = np.array([datetime.date.today().strftime("%d-%b-%Y")])
    meta["description"] = np.array([f"{len(info)} energies {a.spacing} mm WET apart from {os.path.basename(a.template)}, "
                                    f"filter kernel {a.kernel}, extra spot sigma {a.extra_sigma}"])
    out = a.out or f"protons_{a.name}.mat"
    sio.savemat(out, {"machine": mach}, do_compression=True)
    gaps = np.diff([i[1] for i in info])
    print(f"-> {out}: {len(info)} energies {info[0][0]:.1f}-{info[-1][0]:.1f} MeV, peaks {info[0][1]:.1f}-{info[-1][1]:.1f} mm, "
          f"gaps {gaps.min():.2f}-{gaps.max():.2f} mm, spot sigma {info[0][3]:.2f}-{info[-1][3]:.2f} mm")


if __name__ == "__main__":
    main()
