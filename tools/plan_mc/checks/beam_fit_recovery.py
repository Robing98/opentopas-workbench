"""Recovery test of the beam model fit (part A4): does fit.py return the parameters TOPAS curves were made with?

    python3 beam_fit_recovery.py write OUTDIR    # library (fit.py) + 2 depth doses and 4 spot runs with known values
    python3 beam_fit_recovery.py check OUTDIR    # after the runs

Energy: two depth doses at mean energies and spreads off the library grid are inverted on the library. Pass: mean
energy within 0.3 MeV (about 0.2 mm R80 at 100 MeV, more at 200 MeV is still below 0.5 mm), spread within 0.1 %.
Optics: a spot with known sigma, divergence and correlation at the source plane (115 cm) crosses 9 scoring planes from
+40 to -40 cm around the isocenter, once in vacuum and once in air. Pass: sigma within 5 %, divergence within 10 %,
correlation within 0.1, for vacuum without and for air with the air correction of fit.py.
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plan_mc import fit, topas_io  # noqa: E402

ENERGY_CASES = [(93.0, 0.15), (147.3, 0.55)]
OPTICS = {"sigma": 3.5, "div_mrad": 2.0, "corr": -0.4}       # x; y uses sigma 3.0, div 1.5, corr 0.3
OPTICS_Y = {"sigma": 3.0, "div_mrad": 1.5, "corr": 0.3}
OPTICS_E = [100.0, 200.0]
PLANES = [400 - 100 * i for i in range(9)]                     # mm from the isocenter, + towards the source
ZS, NB, BW = 1150.0, 160, 0.5


def optics_case(tag, e, world, histories=20000):
    planes = "".join(f"""s:Ge/P{i}/Type = "TsBox"
s:Ge/P{i}/Parent = "World"
s:Ge/P{i}/Material = "{world}"
d:Ge/P{i}/HLX = {NB * BW / 2:g} mm
d:Ge/P{i}/HLY = {NB * BW / 2:g} mm
d:Ge/P{i}/HLZ = 0.05 mm
d:Ge/P{i}/TransZ = {z:g} mm
i:Ge/P{i}/XBins = {NB}
i:Ge/P{i}/YBins = {NB}
s:Sc/F{i}/Quantity = "Fluence"
s:Sc/F{i}/Component = "P{i}"
s:Sc/F{i}/OutputType = "csv"
s:Sc/F{i}/OutputFile = "{tag}_plane{z:+g}"
s:Sc/F{i}/IfOutputFileAlreadyExists = "Overwrite"
b:Sc/F{i}/OutputToConsole = "False"
sv:Sc/F{i}/OnlyIncludeParticlesNamed = 1 "proton"
""" for i, z in enumerate(PLANES))
    return f"""# Optics recovery: {e:g} MeV, known emittance at the source plane ({ZS:g} mm upstream), world {world}.
i:Ts/NumberOfThreads = 0
i:Ts/Seed = 1
b:Ts/PauseBeforeQuit = "False"
s:Ge/World/Material = "{world}"
d:Ge/World/HLX = 0.3 m
d:Ge/World/HLY = 0.3 m
d:Ge/World/HLZ = 1.3 m
s:Ge/Src/Type = "Group"
s:Ge/Src/Parent = "World"
d:Ge/Src/TransZ = {ZS:g} mm
d:Ge/Src/RotX = 180 deg
{planes}s:So/Field/Type = "Emittance"
s:So/Field/Distribution = "BiGaussian"
s:So/Field/Component = "Src"
s:So/Field/EmittanceParticle = "proton"
d:So/Field/EmittanceEnergy = {e:g} MeV
u:So/Field/EmittanceEnergySpread = 0.2
d:So/Field/SigmaX = {OPTICS["sigma"]:g} mm
u:So/Field/SigmaXprime = {OPTICS["div_mrad"] / 1000:g}
u:So/Field/CorrelationX = {OPTICS["corr"]:g}
d:So/Field/SigmaY = {OPTICS_Y["sigma"]:g} mm
u:So/Field/SigmaYprime = {OPTICS_Y["div_mrad"] / 1000:g}
u:So/Field/CorrelationY = {OPTICS_Y["corr"]:g}
i:So/Field/NumberOfHistoriesInRun = {histories}
"""


def opt_tag(e, world): return f"opt_e{e:g}_{'vac' if world == 'Vacuum' else 'air'}"


def write(out, lib_histories=30000):
    os.makedirs(out, exist_ok=True)
    n = len(fit.library_write(os.path.join(out, "library"), lib_histories))
    for e, s in ENERGY_CASES:
        tag, text = fit.library_case(e, s, 100000)
        open(os.path.join(out, tag + ".txt"), "w").write(text.replace("Library depth dose", "Recovery depth dose"))
    for e in OPTICS_E:
        for w in ("Vacuum", "G4_AIR"):
            open(os.path.join(out, opt_tag(e, w) + ".txt"), "w").write(optics_case(opt_tag(e, w), e, w))
    print(f"{n} library cases in {out}/library, {len(ENERGY_CASES)} depth doses and {2 * len(OPTICS_E)} spot runs in {out}")


def plane_sigmas(out, tag):
    rows = []
    for z in PLANES:
        _, d = topas_io.read(os.path.join(out, f"{tag}_plane{z:+g}.csv"))
        f = d["Sum"][:, :, 0]
        x = (np.arange(NB) + 0.5) * BW - NB * BW / 2
        rows.append((z, fit.sigma_profile(x, f.sum(1))[0], fit.sigma_profile(x, f.sum(0))[0]))
    return rows


def check(out):
    ok_all, lines = True, []
    lib = fit.library_read(os.path.join(out, "library"))
    L = fit.Library(lib)
    for e, s in ENERGY_CASES:
        tag = fit.library_case(e, s, 1)[0]
        m = fit.metrics(*fit.load_curve(os.path.join(out, tag + ".csv")))
        r = L.invert(m["r80"], m["dfo"])
        ok = abs(r["mean"] - e) < 0.3 and abs(r["spread_pct"] - s) < 0.1
        ok_all &= ok
        lines.append(f"{'pass' if ok else 'FAIL'} energy {e:g} MeV / {s:g} %: fitted {r['mean']:.2f} MeV / {r['spread_pct']:.3f} %, R80 {m['r80']:.2f} mm")
    for e in OPTICS_E:
        vac, air = plane_sigmas(out, opt_tag(e, "Vacuum")), plane_sigmas(out, opt_tag(e, "G4_AIR"))
        for k, (ax, true) in enumerate((("x", OPTICS), ("y", OPTICS_Y))):
            s = [ZS - r[0] for r in vac]
            for label, sig, air2 in (("vacuum", [r[1 + k] for r in vac], None),
                                     ("air, uncorrected", [r[1 + k] for r in air], None),
                                     ("air, corrected", [r[1 + k] for r in air], [a[1 + k] ** 2 - v[1 + k] ** 2 for a, v in zip(air, vac)])):
                o = fit.fit_optics(s, sig, air2)
                ok = (abs(o["sigma"] / true["sigma"] - 1) < 0.05 and abs(o["div_mrad"] / true["div_mrad"] - 1) < 0.10
                      and abs(o["corr"] - true["corr"]) < 0.1)
                if label != "air, uncorrected": ok_all &= ok
                else: ok = None
                lines.append(f"{'info' if ok is None else 'pass' if ok else 'FAIL'} optics {e:g} MeV {ax} {label}: sigma {o['sigma']:.3f} "
                             f"(true {true['sigma']:g}) mm, div {o['div_mrad']:.3f} ({true['div_mrad']:g}) mrad, corr {o['corr']:.3f} ({true['corr']:g})")
    for l in lines: print(l)
    print("all pass" if ok_all else "FAIL")
    json.dump({"lines": lines, "pass": bool(ok_all)}, open(os.path.join(out, "recovery.json"), "w"), indent=1)
    return ok_all


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("write", "check"): sys.exit(__doc__)
    if sys.argv[1] == "write": write(sys.argv[2])
    else: sys.exit(0 if check(sys.argv[2]) else 1)
