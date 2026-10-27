#!/usr/bin/env python3
"""Builds a matRad proton machine file (base data) from TOPAS integrated depth-dose scorers.

Depth dose (Z), peak position and range come from TOPAS; the lateral double-Gaussian kernels (sigma1, sigma2, weight)
and LET are taken from a matRad template machine (e.g. protons_Generic.mat) at the same range, because a TOPAS IDD
scorer does not resolve the lateral profile. The spot size in air is set from --fwhm.

Input: CSV output of DoseToWater/DoseToMedium scorers on a water box (one Z binning, X/Y one bin each) and the
TOPAS parameter files that wrote them (beam energy and number of histories are read from there).
Z [MeV cm2/g per primary] = D[Gy] * A[cm2] / (N * 1.602e-10), A = lateral area of the scoring box.

Usage:
  python3 topas2matrad.py --template protons_Generic.mat --name TOPAS_noRiFi --fwhm 7 \
      --out protons_TOPAS_noRiFi.mat "cases/**/IDD_*_rifi0.csv"
Copy the result to <home>/matRad/machines/ (standalone matRad) or matRad/userdata/machines/ and select the machine
name in matRad (radiation mode protons).
"""
import argparse, glob, os, re, sys, datetime
import numpy as np
import scipy.io as sio

MEV_J = 1.602176634e-13


def read_idd(path):
    dims = []
    with open(path) as fh:
        for l in fh:
            if not l.startswith("#"): break
            m = re.match(r"#\s*(\w+)\s+in\s+(\d+)\s+bins?\s+of\s+([\d.eE+\-]+)\s*(\w+)", l)
            if m: dims.append((m.group(1), int(m.group(2)), float(m.group(3)) * {"mm": 0.1, "cm": 1, "m": 100}[m.group(4)]))
    if len(dims) != 3 or dims[0][1] != 1 or dims[1][1] != 1: raise ValueError(f"{path}: needs X and Y with one bin each")
    d = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    nz, wz = dims[2][1], dims[2][2]
    dose = np.zeros(nz); dose[d[:, 2].astype(int)] = d[:, 3]
    area = dims[0][2] * dims[1][2]                      # cm2
    z_mm = (np.arange(nz) + 0.5) * wz * 10
    return z_mm, dose, area


def producer(path):
    """Beam energy [MeV] and histories from the parameter file whose scorer wrote this output."""
    base = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.dirname(path)
    for d in (folder, os.path.dirname(folder)):
        for f in sorted(glob.glob(os.path.join(d, "*.txt"))):
            t = open(f, errors="replace").read()
            if re.search(r'OutputFile\s*=\s*"' + re.escape(base) + '"', t):
                if 'BeamParticle' in t and '"proton"' not in t: raise ValueError(f"{f}: only protons are supported")
                e = re.search(r"^\s*d:So/[^/]+/BeamEnergy\s*=\s*([\d.eE+\-]+)\s*(\w+)", t, re.M)
                n = re.search(r"^\s*i:So/[^/]+/NumberOfHistoriesInRun\s*=\s*(\d+)", t, re.M)
                spread = re.search(r"^\s*u:So/[^/]+/BeamEnergySpread\s*=\s*([\d.eE+\-]+)", t, re.M)
                return float(e.group(1)) * {"keV": 1e-3, "MeV": 1, "GeV": 1e3}[e.group(2)], int(n.group(1)), float(spread.group(1)) if spread else 0.0
    raise ValueError(f"{path}: no parameter file found that writes '{base}'")


def cross_after(z, y, level, i):
    for k in range(i, len(y) - 1):
        if y[k] >= level > y[k + 1]: return z[k] + (z[k + 1] - z[k]) * (y[k] - level) / (y[k] - y[k + 1])
    return np.nan


def depth_grid(peak, zmax):
    """Irregular grid: 1 mm in the plateau, 0.1 mm around the peak, 0.5 mm in the tail (like the generic data)."""
    g = np.concatenate([np.arange(0, max(peak - 15, 0), 1.0), np.arange(max(peak - 15, 0), peak + 8, 0.1), np.arange(peak + 8, zmax, 0.5)])
    return np.unique(np.round(g, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idd", nargs="+", help="IDD csv files or glob patterns")
    ap.add_argument("--template", required=True, help="matRad proton machine used for lateral kernels and LET")
    ap.add_argument("--name", required=True, help="machine name, the file is protons_<name>.mat")
    ap.add_argument("--fwhm", type=float, default=7.0, help="spot FWHM in air at the isocentre [mm]")
    ap.add_argument("--sigma-table", help="csv with energy_MeV, sigma_noRiFi_mm, sigma_<variant>_mm (from fit_spots.py): "
                    "the increase of the spot size over the reference column is added in quadrature to --fwhm")
    ap.add_argument("--sigma-column", help="column of --sigma-table for this machine, e.g. sigma_RiFi4_mm")
    ap.add_argument("--smooth", type=int, default=3, help="moving average over n bins against MC noise (1 = off)")
    ap.add_argument("--out")
    a = ap.parse_args()
    files = sorted({f for p in a.idd for f in glob.glob(p, recursive=True)})
    if not files: sys.exit("no IDD files found")
    extra = None
    if a.sigma_table:
        tab = np.genfromtxt(a.sigma_table, delimiter=",", names=True)
        col = a.sigma_column or "sigma_noRiFi_mm"
        if col not in tab.dtype.names: sys.exit(f"column {col} not in {a.sigma_table}: {tab.dtype.names}")
        inc2 = np.clip(tab[col] ** 2 - tab["sigma_noRiFi_mm"] ** 2, 0, None)
        extra = lambda e: float(np.interp(e, tab["energy_MeV"], inc2))     # added variance [mm2]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from matload import load_struct
    tpl = load_struct(a.template, "machine")           # v5/v7 or v7.3 (h5py)
    T = tpl.data if isinstance(tpl.data, list) else list(np.atleast_1d(tpl.data))
    T_range = np.array([float(np.ravel(x.range)[0]) for x in T])
    entries, seen = [], set()
    for f in files:
        z, dose, area = read_idd(f)
        energy, n, spread = producer(f)
        if round(energy, 3) in seen: print(f"skip duplicate energy {energy} MeV: {f}"); continue
        seen.add(round(energy, 3))
        if a.smooth > 1:  # edge-padded moving average, keeps the entrance value
            k = a.smooth // 2
            dose = np.convolve(np.pad(dose, (k, a.smooth - 1 - k), mode="edge"), np.ones(a.smooth) / a.smooth, mode="valid")
        Z = dose * area / (n * MEV_J * 1e3)            # MeV cm2/g per primary
        i = int(np.argmax(Z)); peak = z[i]
        r80 = cross_after(z, Z / Z[i], 0.8, i)
        nz = np.where(Z > 1e-4 * Z[i])[0][-1]
        depths = depth_grid(peak, min(z[nz] + 5, z[-1]))
        Zg = np.interp(depths, z, Z)
        t = T[int(np.argmin(abs(T_range - r80)))]          # template entry with the same range
        s = float(np.ravel(t.range)[0]) / r80                         # lateral kernels scaled with relative depth
        td = np.ravel(np.asarray(t.depths, float))
        lat = {k: np.interp(depths * s, td, np.ravel(np.asarray(getattr(t, k), float))) for k in ("sigma1", "sigma2", "weight", "LET", "sigma") if hasattr(t, k)}
        if "sigma1" in lat: lat["sigma1"][0] = 0.0
        if "sigma" in lat: lat["sigma"][0] = 0.0
        sig_air = float(np.sqrt((a.fwhm / 2.3548) ** 2 + (extra(energy) if extra else 0.0)))
        f0 = t.initFocus
        initFocus = {"dist": np.ravel(np.asarray(f0.dist, float)), "sigma": np.full(np.size(f0.dist), sig_air), "SisFWHMAtIso": 2.3548 * sig_air,
                     "emittance": {"type": "bigaussian", "sigmaX": sig_air, "divX": 1e-12, "corrX": 1e-12, "sigmaY": sig_air, "divY": 1e-12, "corrY": 1e-12}}
        e = {"energy": energy, "range": r80, "peakPos": peak,
             "energySpectrum": {"type": "gaussian", "mean": energy, "sigma": energy * spread / 100},
             "initFocus": initFocus, "offset": 0.0, "depths": depths, "Z": Zg}
        e.update(lat)
        entries.append(e)
        print(f"{os.path.basename(f)}: {energy:7.2f} MeV  peak {peak:6.1f} mm  R80 {r80:6.1f} mm  Z(0) {Zg[0]:.2f} MeV cm2/g  spot FWHM {2.3548 * sig_air:4.1f} mm  (template {float(np.ravel(t.energy)[0]):.1f} MeV)")
    entries.sort(key=lambda e: e["energy"])
    order = ["energy", "range", "peakPos", "energySpectrum", "initFocus", "offset", "depths", "Z", "sigma", "sigma1", "sigma2", "weight", "LET"]
    dt = [(k, object) for k in order if k in entries[0]]
    data = np.zeros((1, len(entries)), dtype=dt)
    for j, e in enumerate(entries):
        for k, _ in dt: data[0, j][k] = e[k]
    meta = {"name": a.name, "radiationMode": "protons", "dataType": str(tpl.meta.dataType),
            "created_on": datetime.date.today().strftime("%d-%b-%Y"), "created_by": "topas2matrad.py (OpenTOPAS Workbench)",
            "description": f"Depth dose from TOPAS ({len(entries)} energies), lateral kernels and LET from template {tpl.meta.name}, spot FWHM {a.fwhm} mm" + (f" plus TOPAS spot broadening ({a.sigma_column})" if extra else ""),
            "SAD": float(tpl.meta.SAD), "BAMStoIsoDist": float(tpl.meta.BAMStoIsoDist)}
    for k in ("LUTspotSize", "LUT_bxWidthminFWHM"):
        if hasattr(tpl.meta, k): meta[k] = getattr(tpl.meta, k)
    out = a.out or f"protons_{a.name}.mat"
    sio.savemat(out, {"machine": {"meta": meta, "data": data}}, do_compression=True)
    print(f"-> {out}: {len(entries)} energies, {entries[0]['energy']:.1f}-{entries[-1]['energy']:.1f} MeV, "
          f"ranges {entries[0]['range']:.1f}-{entries[-1]['range']:.1f} mm")


if __name__ == "__main__":
    main()
