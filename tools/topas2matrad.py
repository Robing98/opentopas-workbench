#!/usr/bin/env python3
"""Builds a matRad proton machine file (base data) from TOPAS integrated depth-dose scorers.

Depth dose (Z), peak position and range come from TOPAS; the lateral double-Gaussian kernels (sigma1, sigma2, weight)
and LET are taken from a matRad template machine (e.g. protons_Generic.mat) at the same range, because a TOPAS IDD
scorer does not resolve the lateral profile. The spot size in air is set from --fwhm.

A ripple filter widens the spot. --sigma-table adds that to the spot size in air, which is constant in depth. With
--filter-distance the filter counts as an angular spread at that distance, so its share grows with depth and is added
to the kernels of the template. --filter-wet and --open-idd weight it with the dose: protons through the thick parts
of the filter scatter more and stop earlier, so the spot widens towards the peak and narrows at its distal edge. The
kernels never get narrower than zero: where the share taken back exceeds the variance of the template, they end at 0.
With a range shifter in the beam, --sigma-base names the column with the range shifter and without the filter: only
the increase of --sigma-column over it then grows from the filter's distance, and the range shifter's own share stays
constant in depth unless --base-distance lets it grow from its own distance.

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


def filter_variance(depths, inc2, dist, wet=None, open_z=None, open_dose=None):
    """Variance [mm2] that a ripple filter adds to the lateral kernel at the depths [mm], beyond its share inc2 [mm2]
    in the spot size at the isocentre (the phantom surface, depth 0).

    The filter is an angular spread at dist [mm] before the isocentre: its variance grows as (dist + z)^2, so the
    kernel gets inc2 * ((1 + z / dist)^2 - 1), zero at the surface.

    wet = (min, max) water equivalent thickness [mm] of a filter whose thickness is evenly distributed between the two
    (triangular ridges), with the depth dose without the filter (open_z, open_dose): the beam is a sum of sub-beams,
    one per thickness w, each with the open depth dose shifted by w and an angular variance proportional to w. The
    dose of the sub-beams weights their variance, c(z) = sum(w D(z + w)) / (mean(w) sum(D(z + w))), and the kernel
    gets inc2 * (c(z) / c(0) * (1 + z / dist)^2 - 1). It can be negative behind the range of the thick sub-beams;
    the caller clips the sum with the template variance at zero.

    Behind the peak, c is held from the first depth where the summed dose is below 1 % of its maximum: further out
    the dose is a tail that is the same for every thickness and would bring the whole share back. The open depth
    dose must reach every sub-beam up to that depth or end in such a tail itself (below 1 % of its maximum), else
    ValueError: a grid that ends inside the range would count the thick sub-beams as stopped.
    """
    z = np.asarray(depths, float)
    lever = (1.0 + z / dist) ** 2
    if wet is None: return inc2 * (lever - 1.0)
    w = np.linspace(wet[0], wet[1], 101)
    d = np.interp(z[:, None] + w[None, :], open_z, open_dose, right=0.0)     # dose of sub-beam w at depth z
    tot = d.sum(axis=1)
    peak = int(np.argmax(tot))
    low = np.where(tot[peak:] < 1e-2 * tot[peak])[0]
    end = peak + int(low[0]) if len(low) else len(z)                         # first depth behind the range
    if end and z[end - 1] + wet[1] > open_z[-1] + 1e-9 and open_dose[-1] >= 1e-2 * np.max(open_dose):
        raise ValueError(f"the depth dose without the filter ends at {open_z[-1]:.1f} mm, but the dose with the filter reaches "
                         f"{z[end - 1]:.1f} mm and the thickest part of the filter adds {wet[1]:g} mm: it must cover both")
    c = np.ones(len(z))
    c[:end] = (d[:end] * w).sum(axis=1) / (w.mean() * tot[:end])
    if end: c[end:] = c[end - 1]                                             # behind the range: hold the last value
    c0 = float(np.interp(0.0, z, c))
    return inc2 * (c / c0 * lever - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idd", nargs="+", help="IDD csv files or glob patterns")
    ap.add_argument("--template", required=True, help="matRad proton machine used for lateral kernels and LET")
    ap.add_argument("--name", required=True, help="machine name, the file is protons_<name>.mat")
    ap.add_argument("--fwhm", type=float, default=7.0, help="spot FWHM in air at the isocentre [mm]")
    ap.add_argument("--fwhm-table", help="energy dependent spot FWHM at the isocentre instead of --fwhm: json with a "
                    "'focus' object {energy: fwhm_mm} (as exported from the cycle data) or csv energy_MeV,fwhm_mm. "
                    "Energies outside the table use a fit fwhm = a / E + b to it")
    ap.add_argument("--sigma-table", help="csv with energy_MeV, sigma_noRiFi_mm, sigma_<variant>_mm (from fit_spots.py): "
                    "the increase of the spot size over the reference column is added in quadrature to --fwhm")
    ap.add_argument("--sigma-column", help="column of --sigma-table for this machine, e.g. sigma_RiFi4_mm")
    ap.add_argument("--filter-distance", type=float, help="distance of the ripple filter before the isocentre [cm], the one "
                    "--sigma-column was simulated for: its spot broadening then grows with depth as an angular spread "
                    "and is added to the lateral kernels of the template. Depth counts from the isocentre, which is "
                    "taken to lie at the surface of the phantom")
    ap.add_argument("--filter-wet", type=float, nargs=2, metavar=("MIN", "MAX"), help="smallest and largest water "
                    "equivalent thickness of the filter [mm] (triangular ridges): weights the broadening in depth with "
                    "the dose of each thickness; needs --filter-distance and --open-idd")
    ap.add_argument("--open-idd", action="append", metavar="PATTERN", help="IDD csv file or glob pattern of the same energies "
                    "without the filter, for --filter-wet (may be given several times)")
    ap.add_argument("--sigma-base", help="column of --sigma-table without the filter but with everything else in the beam, "
                    "e.g. sigma_rs_mm for a range shifter (default sigma_noRiFi_mm): --filter-distance lets only the increase "
                    "of --sigma-column over this column grow")
    ap.add_argument("--base-distance", type=float, help="distance [cm] before the isocentre of what --sigma-base holds beyond "
                    "sigma_noRiFi_mm (the middle of a range shifter): that share grows with depth as an angular spread too")
    ap.add_argument("--smooth", type=int, default=3, help="moving average over n bins against MC noise (1 = off)")
    ap.add_argument("--spacing-bpw", type=float, default=0.0,
                    help="keep energies whose peaks are about this factor times BPW80 apart (BPW80 of the deeper peak), "
                         "counted from the deepest one: a depth dependent layer spacing such as the RiFi allows")
    ap.add_argument("--max-spacing", type=float, default=0.0,
                    help="upper limit [mm] for the peak distance with --spacing-bpw (deep peaks are wide, large gaps ripple)")
    ap.add_argument("--min-spacing", type=float, default=0.0,
                    help="keep only energies whose peaks are at least this far apart [mm], counted from the deepest one; "
                         "with --spacing-bpw the lower limit for the peak distance")
    ap.add_argument("--out")
    a = ap.parse_args()
    files = sorted({f for p in a.idd for f in glob.glob(p, recursive=True)})
    if not files: sys.exit("no IDD files found")
    if a.filter_distance is not None and not (a.sigma_table and a.sigma_column): sys.exit("--filter-distance needs --sigma-table and --sigma-column")
    if a.filter_distance is not None and a.filter_distance <= 0: sys.exit("--filter-distance must be positive")
    if (a.filter_wet is None) != (a.open_idd is None): sys.exit("--filter-wet and --open-idd belong together")
    if a.filter_wet and a.filter_distance is None: sys.exit("--filter-wet needs --filter-distance")
    if a.filter_wet and not 0 <= a.filter_wet[0] < a.filter_wet[1]: sys.exit("--filter-wet needs 0 <= MIN < MAX")
    if a.sigma_base and a.filter_distance is None: sys.exit("--sigma-base needs --filter-distance")
    if a.base_distance is not None and not a.sigma_base: sys.exit("--base-distance needs --sigma-base")
    if a.base_distance is not None and a.base_distance <= 0: sys.exit("--base-distance must be positive")
    smooth = lambda dose: dose if a.smooth <= 1 else np.convolve(  # edge-padded moving average, keeps the entrance value
        np.pad(dose, (a.smooth // 2, a.smooth - 1 - a.smooth // 2), mode="edge"), np.ones(a.smooth) / a.smooth, mode="valid")
    opened = {}
    for f in sorted({f for p in a.open_idd or [] for f in glob.glob(p, recursive=True)}):
        zo, do, _ = read_idd(f)
        eo = producer(f)[0]
        if round(eo, 3) in opened: print(f"skip duplicate energy {eo} MeV of --open-idd: {f}"); continue
        opened[round(eo, 3)] = (zo, smooth(do))
    extra = None
    if a.sigma_table:
        tab = np.genfromtxt(a.sigma_table, delimiter=",", names=True)
        col = a.sigma_column or "sigma_noRiFi_mm"
        if col not in tab.dtype.names: sys.exit(f"column {col} not in {a.sigma_table}: {tab.dtype.names}")
        inc2 = np.clip(tab[col] ** 2 - tab["sigma_noRiFi_mm"] ** 2, 0, None)
        extra = lambda e: float(np.interp(e, tab["energy_MeV"], inc2))     # added variance [mm2]
        if a.sigma_base and a.sigma_base not in tab.dtype.names: sys.exit(f"column {a.sigma_base} not in {a.sigma_table}: {tab.dtype.names}")
        ref = tab[a.sigma_base or "sigma_noRiFi_mm"]
        of_filter = lambda e: float(np.interp(e, tab["energy_MeV"], np.clip(tab[col] ** 2 - ref ** 2, 0, None)))
        # the range shifter's share is at most the whole increase in air: noise can put ref above the column
        of_base = lambda e: float(np.interp(e, tab["energy_MeV"], np.clip(np.minimum(ref, tab[col]) ** 2 - tab["sigma_noRiFi_mm"] ** 2, 0, None)))
    base_fwhm = lambda e: a.fwhm
    if a.fwhm_table:
        if a.fwhm_table.endswith(".json"):
            import json
            ft = json.load(open(a.fwhm_table))["focus"]; fe = np.array([float(k) for k in ft]); fw = np.array([float(v) for v in ft.values()])
        else:
            t2 = np.genfromtxt(a.fwhm_table, delimiter=",", names=True); fe, fw = t2[t2.dtype.names[0]], t2[t2.dtype.names[1]]
        o = np.argsort(fe); fe, fw = fe[o], fw[o]
        ca, cb = np.polyfit(1.0 / fe, fw, 1)                               # fwhm = ca / E + cb, used outside the table
        base_fwhm = lambda e: float(np.interp(e, fe, fw)) if fe[0] <= e <= fe[-1] else float(ca / e + cb)
        print(f"spot FWHM from {a.fwhm_table}: {fw.min():.1f} to {fw.max():.1f} mm ({len(fe)} energies), outside: {ca:.0f}/E + {cb:.2f} mm")
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
        dose = smooth(dose)
        Z = dose * area / (n * MEV_J * 1e3)            # MeV cm2/g per primary
        i = int(np.argmax(Z)); peak = z[i]
        r80 = cross_after(z, Z / Z[i], 0.8, i)
        p80 = next((z[k] + (z[k + 1] - z[k]) * (0.8 * Z[i] - Z[k]) / (Z[k + 1] - Z[k]) for k in range(i - 1, -1, -1) if Z[k] < 0.8 * Z[i] <= Z[k + 1]), np.nan)
        nz = np.where(Z > 1e-4 * Z[i])[0][-1]
        depths = depth_grid(peak, min(z[nz] + 5, z[-1]))
        Zg = np.interp(depths, z, Z)
        t = T[int(np.argmin(abs(T_range - r80)))]          # template entry with the same range
        s = float(np.ravel(t.range)[0]) / r80                         # lateral kernels scaled with relative depth
        td = np.ravel(np.asarray(t.depths, float))
        lat = {k: np.interp(depths * s, td, np.ravel(np.asarray(getattr(t, k), float))) for k in ("sigma1", "sigma2", "weight", "LET", "sigma") if hasattr(t, k)}
        if "sigma1" in lat: lat["sigma1"][0] = 0.0
        if "sigma" in lat: lat["sigma"][0] = 0.0
        if a.filter_distance is not None:
            if a.filter_wet and round(energy, 3) not in opened: sys.exit(f"{f}: no file of --open-idd has {energy} MeV")
            try: add = filter_variance(depths, of_filter(energy), a.filter_distance * 10.0, a.filter_wet, *opened.get(round(energy, 3), ()))
            except ValueError as err: sys.exit(f"{f}: {err}")
            if a.base_distance is not None: add = add + filter_variance(depths, of_base(energy), a.base_distance * 10.0)
            for k in ("sigma", "sigma1", "sigma2"):
                if k in lat: lat[k] = np.sqrt(np.clip(lat[k] ** 2 + add, 0.0, None))
        sig_air = float(np.sqrt((base_fwhm(energy) / 2.3548) ** 2 + (extra(energy) if extra else 0.0)))
        f0 = t.initFocus
        initFocus = {"dist": np.ravel(np.asarray(f0.dist, float)), "sigma": np.full(np.size(f0.dist), sig_air), "SisFWHMAtIso": 2.3548 * sig_air,
                     "emittance": {"type": "bigaussian", "sigmaX": sig_air, "divX": 1e-12, "corrX": 1e-12, "sigmaY": sig_air, "divY": 1e-12, "corrY": 1e-12}}
        e = {"energy": energy, "range": r80, "peakPos": peak,
             "energySpectrum": {"type": "gaussian", "mean": energy, "sigma": energy * spread / 100},
             "initFocus": initFocus, "offset": 0.0, "depths": depths, "Z": Zg}
        e.update(lat)
        # matRad expects depth tables as columns (N x 1) and the focus tables as rows (1 x N), as in protons_Generic
        for k in ("depths", "Z", "sigma", "sigma1", "sigma2", "weight", "LET"):
            if k in e: e[k] = np.asarray(e[k], float).reshape(-1, 1)
        for k in ("dist", "sigma"): initFocus[k] = np.asarray(initFocus[k], float).reshape(1, -1)
        e["_bpw80"] = float(r80 - p80)       # helper for --spacing-bpw, not written to the machine
        entries.append(e)
        print(f"{os.path.basename(f)}: {energy:7.2f} MeV  peak {peak:6.1f} mm  R80 {r80:6.1f} mm  Z(0) {Zg[0]:.2f} MeV cm2/g  spot FWHM {2.3548 * sig_air:4.1f} mm  (template {float(np.ravel(t.energy)[0]):.1f} MeV)")
    entries.sort(key=lambda e: e["energy"])
    if a.min_spacing > 0 and a.spacing_bpw <= 0:
        kept = [entries[-1]]
        for e in reversed(entries[:-1]):
            if kept[-1]["peakPos"] - e["peakPos"] >= a.min_spacing - 0.05: kept.append(e)
        print(f"--min-spacing {a.min_spacing} mm: {len(kept)} of {len(entries)} energies kept")
        entries = sorted(kept, key=lambda e: e["energy"])
    if a.spacing_bpw > 0:
        kept = [entries[-1]]
        for e in reversed(entries[:-1]):
            need = a.spacing_bpw * kept[-1]["_bpw80"]
            if a.max_spacing > 0: need = min(need, a.max_spacing)
            need = max(need, a.min_spacing)          # with --min-spacing: never closer than today's constant spacing
            if kept[-1]["peakPos"] - e["peakPos"] >= need - 0.05: kept.append(e)
        gaps = [kept[j]["peakPos"] - kept[j + 1]["peakPos"] for j in range(len(kept) - 1)]
        print(f"--spacing-bpw {a.spacing_bpw}" + (f" (max {a.max_spacing} mm)" if a.max_spacing > 0 else "") + (f" (min {a.min_spacing} mm)" if a.min_spacing > 0 else "") + ": "
              f"{len(kept)} of {len(entries)} energies kept, peak gaps {min(gaps):.1f} to {max(gaps):.1f} mm"
              + f" (deep {gaps[0]:.1f} mm, shallow {gaps[-1]:.1f} mm)" if gaps else "")
        entries = sorted(kept, key=lambda e: e["energy"])
    order = ["energy", "range", "peakPos", "energySpectrum", "initFocus", "offset", "depths", "Z", "sigma", "sigma1", "sigma2", "weight", "LET"]
    dt = [(k, object) for k in order if k in entries[0]]
    data = np.zeros((1, len(entries)), dtype=dt)
    for j, e in enumerate(entries):
        for k, _ in dt: data[0, j][k] = e[k]
    meta = {"name": a.name, "radiationMode": "protons", "dataType": str(tpl.meta.dataType),
            "created_on": datetime.date.today().strftime("%d-%b-%Y"), "created_by": "topas2matrad.py (OpenTOPAS Workbench)",
            "description": f"Depth dose from TOPAS ({len(entries)} energies), lateral kernels and LET from template {tpl.meta.name}, spot FWHM " + (f"from {os.path.basename(a.fwhm_table)}" if a.fwhm_table else f"{a.fwhm} mm") + (f" plus TOPAS spot broadening ({a.sigma_column})" if extra else "") + (f", growing in depth from a filter {a.filter_distance:g} cm before the isocentre" if a.filter_distance is not None else "") + (f" and from {a.sigma_base} at {a.base_distance:g} cm" if a.base_distance is not None else ""),
            "SAD": float(tpl.meta.SAD), "BAMStoIsoDist": float(tpl.meta.BAMStoIsoDist)}
    for k in ("LUTspotSize", "LUT_bxWidthminFWHM"):
        if hasattr(tpl.meta, k): meta[k] = getattr(tpl.meta, k)
    out = a.out or f"protons_{a.name}.mat"
    sio.savemat(out, {"machine": {"meta": meta, "data": data}}, do_compression=True)
    print(f"-> {out}: {len(entries)} energies, {entries[0]['energy']:.1f}-{entries[-1]['energy']:.1f} MeV, "
          f"ranges {entries[0]['range']:.1f}-{entries[-1]['range']:.1f} mm")


if __name__ == "__main__":
    main()
