#!/usr/bin/env python3
"""Compares TOPAS integrated depth doses with the depth doses of a matRad proton machine (base data).

For every TOPAS curve the machine entry with the nearest energy is used. Compared: peak position, R80, BPW80,
distal fall-off D80-D20, peak-to-entrance ratio and the absolute entrance value Z(0) in MeV cm2/g per primary.
The machine depths are shifted by its 'offset' field (water depth = depths + offset, as in matRad's dose engine).
Results go to the output folder only (nothing is printed except the folder), so the machine data stay local.

Usage:
  python3 compare_idd_matrad.py --machine PATH/protons_X.mat --out OUTDIR "cases/IDD_*_rifi0.csv"
"""
import argparse, glob, os, re, sys
import numpy as np

MEV_J = 1.602176634e-13


def load_machine(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from matload import load_struct
    m = load_struct(path, "machine")
    data = m.data if isinstance(m.data, list) else list(np.atleast_1d(m.data))
    def zvec(x):
        z = x.Z
        if hasattr(z, "_fieldnames"):  # some machines store several Z models in a struct
            for k in ("doubleGauss", "profileORG", "singleGauss"):
                if k in z._fieldnames: return np.ravel(np.asarray(getattr(z, k), float))
            return np.ravel(np.asarray(getattr(z, z._fieldnames[0]), float))
        return np.ravel(np.asarray(z, float))
    off = lambda x: float(np.ravel(getattr(x, "offset", 0.0) or 0.0)[0]) if np.size(getattr(x, "offset", 0.0)) else 0.0
    return [{"energy": float(np.ravel(x.energy)[0]), "depths": np.ravel(np.asarray(x.depths, float)), "Z": zvec(x), "offset": off(x)} for x in data]


def read_idd(path):
    dims = []
    for l in open(path):
        if not l.startswith("#"): break
        m = re.match(r"#\s*(\w+)\s+in\s+(\d+)\s+bins?\s+of\s+([\d.eE+\-]+)\s*(\w+)", l)
        if m: dims.append((int(m.group(2)), float(m.group(3)) * {"mm": 0.1, "cm": 1, "m": 100}[m.group(4)]))
    d = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    nz, wz = dims[2]; dose = np.zeros(nz); dose[d[:, 2].astype(int)] = d[:, 3]
    return (np.arange(nz) + 0.5) * wz * 10, dose, dims[0][0] * dims[0][1] * dims[1][0] * dims[1][1]


def producer(path):
    base = os.path.splitext(os.path.basename(path))[0]
    for d in (os.path.dirname(path), os.path.dirname(os.path.dirname(path))):
        for f in glob.glob(os.path.join(d, "*.txt")):
            t = open(f, errors="replace").read()
            if re.search(r'OutputFile\s*=\s*"' + re.escape(base) + '"', t):
                e = re.search(r"^\s*d:So/[^/]+/BeamEnergy\s*=\s*([\d.eE+\-]+)\s*(\w+)", t, re.M)
                n = re.search(r"^\s*i:So/[^/]+/NumberOfHistoriesInRun\s*=\s*(\d+)", t, re.M)
                return float(e.group(1)) * {"keV": 1e-3, "MeV": 1, "GeV": 1e3}[e.group(2)], int(n.group(1))
    raise ValueError(f"no parameter file for {path}")


def metrics(z, y):
    i = int(np.argmax(y)); m = y[i]; n = y / m
    def cross(level, step):
        k = i
        while 0 <= k + step < len(n):
            if n[k] >= level > n[k + step]: return z[k] + (z[k + step] - z[k]) * (n[k] - level) / (n[k] - n[k + step])
            k += step
        return np.nan
    r80, r20, p80 = cross(.8, 1), cross(.2, 1), cross(.8, -1)
    k0 = max(1, int(np.searchsorted(z, 2.0)))                     # entrance: mean over the first 2 mm
    return {"peak": z[i], "r80": r80, "bpw80": r80 - p80, "dfo": r20 - r80, "pe": m / y[:k0].mean(), "z0": y[:k0].mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idd", nargs="+"); ap.add_argument("--machine", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=3)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    files = sorted({f for p in a.idd for f in glob.glob(p, recursive=True)})
    if not files: sys.exit("no TOPAS IDD files found")
    M = load_machine(a.machine); ME = np.array([x["energy"] for x in M])
    os.makedirs(a.out, exist_ok=True)
    rows, curves = [], []
    for f in files:
        z, dose, area = read_idd(f); e, n = producer(f)
        if a.smooth > 1:
            k = a.smooth // 2; dose = np.convolve(np.pad(dose, (k, a.smooth - 1 - k), mode="edge"), np.ones(a.smooth) / a.smooth, mode="valid")
        Zt = dose * area / (n * MEV_J * 1e3)
        mi = int(np.argmin(abs(ME - e))); x = M[mi]
        zm = x["depths"] + x["offset"]; Zm = x["Z"]     # matRad: water depth = depths + offset (matRad_ParticlePencilBeamEngineAbstract)
        keep = zm >= 0
        mt, mm = metrics(z, Zt), metrics(zm[keep], Zm[keep])
        rows.append((os.path.basename(f), e, x["energy"], mt, mm)); curves.append((e, z, Zt, zm, Zm))
    keys = [("peak", "Peak [mm]"), ("r80", "R80 [mm]"), ("bpw80", "BPW80 [mm]"), ("dfo", "D80-D20 [mm]"), ("pe", "Peak/Eingang"), ("z0", "Z(0) [MeV cm2/g]")]
    with open(os.path.join(a.out, "vergleich_topas_matrad.csv"), "w") as fh:
        fh.write("file;E_topas_MeV;E_machine_MeV;" + ";".join(f"{k}_topas;{k}_machine;{k}_diff" for k, _ in keys) + "\n")
        for name, e, em, mt, mm in rows:
            fh.write(f"{name};{e:.2f};{em:.2f};" + ";".join(f"{mt[k]:.4f};{mm[k]:.4f};{mt[k]-mm[k]:.4f}" for k, _ in keys) + "\n")
    lines = ["# TOPAS vs. matRad-Maschine", "", f"Maschine: `{os.path.basename(a.machine)}`, {len(rows)} TOPAS-Kurven.", "",
             "| E [MeV] | ΔE Maschine | ΔR80 [mm] | BPW80 TOPAS/Masch. | D80–D20 TOPAS/Masch. | Peak/Eingang TOPAS/Masch. | Z(0) TOPAS/Masch. |", "|---|---|---|---|---|---|---|"]
    for name, e, em, mt, mm in rows:
        lines.append(f"| {e:.2f} | {em - e:+.2f} | {mt['r80'] - mm['r80']:+.2f} | {mt['bpw80']:.2f}/{mm['bpw80']:.2f} | {mt['dfo']:.2f}/{mm['dfo']:.2f} | {mt['pe']:.2f}/{mm['pe']:.2f} | {mt['z0']:.2f}/{mm['z0']:.2f} |")
    r = np.array([[mt["bpw80"] / mm["bpw80"], mt["dfo"] / mm["dfo"], mt["r80"] - mm["r80"]] for _, _, _, mt, mm in rows])
    lines += ["", f"Median BPW80-Verhältnis TOPAS/Maschine: {np.nanmedian(r[:, 0]):.2f}, D80–D20-Verhältnis: {np.nanmedian(r[:, 1]):.2f}, "
              f"medianes ΔR80: {np.nanmedian(r[:, 2]):+.2f} mm.",
              "Ein Verhältnis unter 1 heißt: die TOPAS-Peaks sind schmaler, die angenommene Energiebreite (0,2 %) ist kleiner als in der Maschinendatei."]
    open(os.path.join(a.out, "vergleich_topas_matrad.md"), "w").write("\n".join(lines) + "\n")
    # plots: selected curves and metrics vs energy
    sel = np.unique(np.linspace(0, len(curves) - 1, min(6, len(curves))).astype(int))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8)); axs = axs.ravel()
    for ax, i in zip(axs, sel):
        e, z, Zt, zm, Zm = curves[i]
        ax.plot(z, Zt / Zt.max(), lw=1.8, label="TOPAS"); ax.plot(zm, Zm / Zm.max(), lw=1.4, ls="--", label="matRad-Maschine")
        pk = z[np.argmax(Zt)]; ax.set_xlim(max(0, pk - 25), pk + 10); ax.set_title(f"{e:.2f} MeV", fontsize=10); ax.grid(alpha=.3); ax.set_xlabel("Tiefe [mm]")
    for ax in axs[len(sel):]: ax.axis("off")
    axs[0].legend(); fig.suptitle("Tiefendosis normiert auf das Maximum, Peakbereich"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "vergleich_kurven.png"), dpi=130); plt.close(fig)
    E = np.array([row[1] for row in rows])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    axs[0].plot(E, r[:, 2], "o-"); axs[0].set_ylabel("ΔR80 TOPAS − Maschine [mm]")
    axs[1].plot(E, [mt["bpw80"] for *_, mt, mm in rows], "o-", label="TOPAS"); axs[1].plot(E, [mm["bpw80"] for *_, mt, mm in rows], "s--", label="Maschine"); axs[1].set_ylabel("BPW80 [mm]"); axs[1].legend()
    axs[2].plot(E, [mt["pe"] for *_, mt, mm in rows], "o-", label="TOPAS"); axs[2].plot(E, [mm["pe"] for *_, mt, mm in rows], "s--", label="Maschine"); axs[2].set_ylabel("Peak/Eingang"); axs[2].legend()
    for ax in axs: ax.set_xlabel("Energie [MeV]"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "vergleich_kennzahlen.png"), dpi=130); plt.close(fig)
    # ---- calibration: TOPAS energy that reproduces the machine R80, and the energy spread that reproduces D80-D20
    T = sorted((e, mt["r80"], mt["dfo"]) for _, e, _, mt, _ in rows if np.isfinite(mt["r80"]))
    if len(T) >= 2:
        Et, Rt, Dt = map(np.array, zip(*T))
        def e_for_r(r):  # log-log interpolation, linear extrapolation in log space at the ends
            return float(np.exp(np.interp(np.log(r), np.log(Rt), np.log(Et), left=np.nan, right=np.nan))) if Rt[0] <= r <= Rt[-1] else \
                float(np.exp(np.log(Et[0 if r < Rt[0] else -1]) + (np.log(r) - np.log(Rt[0 if r < Rt[0] else -1])) / np.polyfit(np.log(Et), np.log(Rt), 1)[0]))
        slope = np.polyfit(np.log(Et), np.log(Rt), 1)[0]          # R ~ E^slope
        cal = []
        for name, e, em, mt, mm in rows:
            r_m = mm["r80"] + slope * mm["r80"] / em * (e - em)      # machine R80 moved to the label energy e
            e_need = e_for_r(r_m)
            drde = slope * r_m / e_need                               # mm per MeV
            sr_m = mm["dfo"] / 1.3                                    # D80-D20 ~ 1.3 sigma_R
            sr_t = float(np.interp(e_need, Et, Dt)) / 1.3
            extra = np.sqrt(max(sr_m ** 2 - sr_t ** 2, 0.0)) / drde   # MeV
            spread = 100 * np.sqrt((0.002 * e_need) ** 2 + extra ** 2) / e_need
            cal.append((e, e_need, spread))
        with open(os.path.join(a.out, "kalibrierung.csv"), "w") as fh:
            fh.write("energy_label_MeV,energy_topas_MeV,energy_spread_percent\n")
            for e, en, sp in cal: fh.write(f"{e:.3f},{en:.3f},{sp:.4f}\n")
        d = np.array([(en - e, sp) for e, en, sp in cal])
        with open(os.path.join(a.out, "vergleich_topas_matrad.md"), "a") as fh:
            fh.write(f"\n## Kalibrierung\n\nTOPAS-Energie für gleiche R80: median {np.median(d[:, 0]):+.2f} MeV gegenüber der Bezeichnung "
                     f"(Spanne {d[:, 0].min():+.2f} bis {d[:, 0].max():+.2f}). Energiebreite für gleiches D80–D20: median {np.median(d[:, 1]):.2f} % "
                     f"(Spanne {d[:, 1].min():.2f} bis {d[:, 1].max():.2f} %). Annahme D80–D20 ≈ 1,3 σ_R. Werte je Energie in kalibrierung.csv.\n")
    print("results in", a.out)


if __name__ == "__main__":
    main()
