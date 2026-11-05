"""CT range check in TOPAS: does a synthetic CT with the HLUT tuned Schneider file (hlut.py) give the ranges the HLUT
predicts?

    python3 ct_range.py write OUTDIR HLUT SCHNEIDER_TEMPLATE   # CTs, Schneider file, 4 TOPAS cases (~2 min each)
    python3 ct_range.py check OUTDIR HLUT                      # after the runs

One 120 MeV spot at gantry 0 (HFS) along +y through a 100 x 160 x 80 mm block with 0.5 mm rows along the beam:
water box (G4_WATER), CT of HU 0, CT of HU 0 with 20 mm of HU 1000 (bone), CT of HU 0 with 30 mm of HU -500 (lung).
With s(HU) the HLUT SPR, the water equivalent range is the same in all cases, so
R80(HU 0) = R80(water) / s(0), and R80(slab) = R80(HU 0) - t * (s(slab) - s(0)) / s(0).
Pass: each R80 within 0.5 mm of that (the SPR of the Schneider materials varies with energy by up to 0.3 % in bone,
see the hlut.py report). The dose of the HU 0 case is scored twice, as csv and as binary, to check the binary reader
(topas_io) bin order: pass when both arrays agree.
Queue the cases through the workbench; nothing here starts TOPAS.
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plan_mc import beam_model, fit, hlut, iec, plan as pl, synthetic, topas_io, topas_writer as tw  # noqa: E402

ENERGY, HISTORIES = 120.0, 20000
NX, NY, NZ = 50, 320, 40
SP = (2.0, 0.5, 2.0)
ORIGIN = (-(NX - 1) * SP[0] / 2, -(NY - 1) * SP[1] / 2, -(NZ - 1) * SP[2] / 2)
ENTRY_ROWS = 40                                  # slabs start 20 mm behind the entrance
CASES = {"water": None, "hu0": [], "bone": [(ENTRY_ROWS, ENTRY_ROWS + 40, 1000)],
         "lung": [(ENTRY_ROWS, ENTRY_ROWS + 60, -500)]}


def scorer_csv(text, name):
    return text.replace(f's:Sc/{name}/OutputType = "binary"', f's:Sc/{name}/OutputType = "csv"')


def write(out, hlut_path, template):
    out = os.path.abspath(out); os.makedirs(out, exist_ok=True)
    rep = hlut.run(hlut_path, template, os.path.join(out, "schneider.txt"))
    json.dump(rep, open(os.path.join(out, "schneider_report.json"), "w"), indent=1)
    m = beam_model.generic()
    for case, slabs in CASES.items():
        if slabs is None:
            half = (NX * SP[0] / 2, NY * SP[1] / 2, NZ * SP[2] / 2)
            text = tw.patient_block("box", "Patient", box={"center": (0.0, 0.0, 0.0), "half": half,
                                                            "bins": (1, NY, 1)}, let=False)
        else:
            ct = os.path.join(out, f"ct_{case}")
            synthetic.write_ct(ct, synthetic.slab_phantom((NZ, NY, NX), 0, slabs), ORIGIN, SP)
            text = tw.patient_block("ct", "Patient", schneider=os.path.join(out, "schneider.txt"), dicom_dir=ct,
                                    let=False)
        text = scorer_csv(text, "Dose")
        if case == "hu0":   # same quantity once more as binary: checks the bin order of topas_io.read
            text += "\n".join(l.replace("Sc/Dose/", "Sc/DoseBin/") for l in text.splitlines()
                              if l.startswith(("s:Sc/Dose/", "sv:Sc/Dose/", "b:Sc/Dose/"))).replace('"csv"', '"binary"')
            text += '\ns:Sc/DoseBin/OutputFile = "ct_hu0_dosebin"\n'
        pat = os.path.join(out, f"patient_{case}.txt")
        open(pat, "w").write(text)
        f = pl.Field(f"ct_{case}", 1, 0.0, 0.0, (0.0, 0.0, 0.0), [pl.Layer(ENERGY, [0.0], [0.0], [1.0])], "particles")
        t, _ = tw.field_file(pl.Plan("ctrange", "HFS", [f]), f, m, {}, pat, HISTORIES, out_name=f"ct_{case}")
        open(os.path.join(out, f"ct_{case}.txt"), "w").write(t)
    print(f"{len(CASES)} cases in {out}")


def depth(path, sign):
    """Integrated depth dose along the beam (DICOM y, direction `sign`): depth behind the entrance face, dose."""
    a = np.asarray(topas_io.read(path)[1]["Sum"], float)
    if a.shape[1] != NY: raise ValueError(f"{path}: expected {NY} bins along y (DICOM rows), got shape {a.shape}")
    d = a.sum(axis=(0, 2))          # topas_io arrays are [x, y, z]
    if sign < 0: d = d[::-1]
    return SP[1] * (np.arange(NY) + 0.5), d


def r80(y, d):
    return fit.distal(y, d, 0.8)


def check(out, hlut_path):
    out = os.path.abspath(out)
    ray = iec.spot_ray(0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), position="HFS")
    hu, s = hlut.read_hlut(hlut_path)
    spr = lambda v: float(np.interp(v, hu, s))
    res, ok = {}, True
    sign = float(np.sign(ray[1][1]))
    print(f"beam direction (DICOM) {np.round(ray[1], 3).tolist()}")
    if abs(ray[1][1]) < 0.999: raise ValueError("gantry 0 HFS should run along DICOM y")
    for case in CASES:
        res[case] = r80(*depth(os.path.join(out, f"ct_{case}_dose.csv"), sign))
    s0 = spr(0)
    exp = {"hu0": res["water"] / s0}
    for case, slabs in CASES.items():
        if case in ("water", "hu0"): continue
        a, b, v = slabs[0]
        exp[case] = exp["hu0"] - (b - a) * SP[1] * (spr(v) - s0) / s0
    for case in CASES:
        if case == "water":
            print(f"water R80 {res['water']:.2f} mm"); continue
        dev = res[case] - exp[case]
        good = abs(dev) <= 0.5; ok &= good
        print(f"{case:5} {'pass' if good else 'FAIL'} R80 {res[case]:.2f} mm, expected {exp[case]:.2f}, diff {dev:+.2f}")
    csv = topas_io.read(os.path.join(out, "ct_hu0_dose.csv"))[1]
    binp = os.path.join(out, "ct_hu0_dosebin.bin")
    if not os.path.exists(binp):
        print("binary FAIL: no ct_hu0_dosebin.bin"); ok = False
    else:
        b = topas_io.read(binp)[1]
        for r in csv:   # Sum and Standard_Deviation: two reports, so the bin/report order matters
            x, y = np.asarray(b[r], float), np.asarray(csv[r], float)
            same = x.shape == y.shape and np.allclose(x, y, rtol=1e-4, atol=1e-6 * y.max())
            ok &= same
            print(f"binary {'pass' if same else 'FAIL'} {r}: shape {x.shape}, max {x.max():.4g} vs csv {y.max():.4g}")
    print("all pass" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "write": write(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) >= 4 and sys.argv[1] == "check": sys.exit(0 if check(sys.argv[2], sys.argv[3]) else 1)
    else: sys.exit(__doc__)
