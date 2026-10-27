#!/usr/bin/env python3
"""Writes copies of TOPAS parameter files with calibrated beam energy and energy spread.

The calibration table (kalibrierung.csv from compare_idd_matrad.py) maps the nominal energy in a parameter file to the
TOPAS energy and spread that reproduce the reference depth dose. Output names stay the same, in another folder.
Usage: python3 calibrated_cases.py --cal kalibrierung.csv --dst kalibriert "planF*_*.txt"
"""
import argparse, glob, os, re, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+"); ap.add_argument("--cal", required=True); ap.add_argument("--dst", required=True)
a = ap.parse_args()
tab = np.genfromtxt(a.cal, delimiter=",", names=True)
os.makedirs(a.dst, exist_ok=True)
E_RE = re.compile(r"^(\s*d:So/[^/]+/BeamEnergy\s*=\s*)([\d.eE+\-]+)(\s*MeV.*)$", re.M)
S_RE = re.compile(r"^(\s*u:So/[^/]+/BeamEnergySpread\s*=\s*)([\d.eE+\-]+)(.*)$", re.M)
n = 0
for f in sorted({x for p in a.files for x in glob.glob(p)}):
    t = open(f, errors="replace").read(); m = E_RE.search(t)
    if not m: continue
    e = float(m.group(2))
    if not (tab["energy_label_MeV"].min() - 0.5 <= e <= tab["energy_label_MeV"].max() + 0.5): continue
    en = float(np.interp(e, tab["energy_label_MeV"], tab["energy_topas_MeV"]))
    sp = float(np.interp(e, tab["energy_label_MeV"], tab["energy_spread_percent"]))
    t = E_RE.sub(lambda k: f"{k.group(1)}{en:.3f}{k.group(3)}", t, count=1)
    t = S_RE.sub(lambda k: f"{k.group(1)}{sp:.4f}{k.group(3)}", t, count=1) if S_RE.search(t) else t
    t = f"# Calibrated copy of {os.path.basename(f)}: nominal {e} MeV -> TOPAS {en:.3f} MeV, spread {sp:.3f} % (from {os.path.basename(a.cal)})\n" + t
    open(os.path.join(a.dst, os.path.basename(f)), "w").write(t); n += 1
    for stl in glob.glob(os.path.join(os.path.dirname(f), "*.stl")):
        if not os.path.exists(os.path.join(a.dst, os.path.basename(stl))): shutil.copy2(stl, a.dst)
print(f"{n} calibrated cases in {a.dst}")
