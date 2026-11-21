#!/usr/bin/env python3
"""Writes opentopas_workbench/generators/_rifi_ref.json, the reference data of the RiFi designer.

Usage: python3 -I tools/make_rifi_ref.py FORMVERGLEICH_DIR [TREPPE_DIR ...]
FORMVERGLEICH_DIR is the shape comparison study (task 3176) with faelle/IDD_form_p{063,080,094,110}_rifi0.csv and
auswertung.csv; further study folders add their auswertung.csv rows to the lateral calibration (task 3141 staircase).
Curves: proton depth doses without filter (TOPAS, synthetic beam), smoothed as in auswertung.py (5 bins of 0.1 mm)
and scaled to a peak of 1. Lateral: extra sigma at the isocentre (filter 80 cm upstream) over the square root of the
mean water equivalent thickness, averaged over all filter variants per energy.
"""
import csv, json, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rifi_kernel_fit as kf                                    # noqa: E402

ENERGIES = (63, 80, 94, 110)


def main():
    form, more = sys.argv[1], sys.argv[2:]
    curves = {}
    for e in ENERGIES:
        z, d, _ = kf.read_curve(os.path.join(form, "faelle", f"IDD_form_p{e:03d}_rifi0.csv"))
        ds = kf.smooth(d, 5); ds = ds / ds.max()
        dz = float(round(z[1] - z[0], 6))
        curves[str(e)] = {"z0": float(round(z[0], 6)), "dz": dz, "d": [float(f"{v:.5g}") for v in ds]}
    coef = defaultdict(list)
    for folder in [form] + more:
        for r in csv.DictReader(open(os.path.join(folder, "auswertung.csv"))):
            if r.get("ion", "p") != "p" or not r.get("kernel_mean") or r["variant"] == "rifi0": continue
            e = int(float(r["energy"]))
            if e in ENERGIES: coef[str(e)].append(float(r["extra_lat"]) / math.sqrt(float(r["kernel_mean"])))
    out = {"note": "Reference data of the RiFi designer, written by tools/make_rifi_ref.py from public TOPAS runs "
                   "(tasks 3141 and 3176). curves: proton IDD without filter, peak 1, depth in mm water. lateral_coef: "
                   "extra lateral sigma [mm] at the isocentre per sqrt(mean thickness [mm water]), filter 80 cm upstream.",
           "curves": curves,
           "lateral_coef": {e: round(sum(v) / len(v), 3) for e, v in sorted(coef.items(), key=lambda kv: int(kv[0]))},
           "lateral_spread": {e: round((max(v) - min(v)) / 2 / (sum(v) / len(v)), 3) for e, v in coef.items()},
           "lateral_n": {e: len(v) for e, v in coef.items()}}
    path = os.path.join(os.path.dirname(HERE), "opentopas_workbench", "generators", "_rifi_ref.json")
    with open(path, "w") as fh: json.dump(out, fh, separators=(",", ":"))
    print(path, os.path.getsize(path), out["lateral_coef"], out["lateral_spread"])


if __name__ == "__main__":
    main()
