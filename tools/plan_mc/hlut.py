"""HLUT to TOPAS Schneider converter (part B1): tune DensityCorrection so that TOPAS sees the stopping power ratio
(SPR) of a given HLUT at every HU.

TOPAS builds one material per HU (TsImagingToMaterialSchneider.cc): density = (offset + factor * (factorOffset + HU))
* DensityCorrection[HU - MinImagingValue], composition from the material section of the HU. Without
`SchneiderMaterialMeanExcitationEnergy` Geant4 gives each material the mean excitation energy of Bragg additivity,
ln I = sum(n_i Z_i ln I_i) / sum(n_i Z_i) (G4IonisParamMat.cc, ComputeMeanParameters), with the element values of
data/g4_elements.json. This script computes the water equivalent ratio (WER: CSDA range in water over CSDA range in
the material, Bethe formula without shell or density correction, both nearly cancel in the ratio) of every such
material relative to the water of the beam model for initial energies 70 to 220 MeV, and sets
DensityCorrection = SPR_HLUT / mean over energies of the uncorrected WER. What stays is the energy dependence of the
WER, which no energy independent HLUT can follow: the report gives it per material section as a relative range error
(mm per 100 mm of that tissue). `--match spr` matches the mean stopping power ratio instead (the older behaviour).

    python3 hlut.py write HLUT TEMPLATE OUT.txt [--i-water 78] [--spr-scale 1.0] [--component Patient] [--match range|spr]
                                                                 [--report R.json]
    python3 hlut.py variants HLUT TEMPLATE OUTDIR [--i-water 78]   # SPR nominal, +2 %, -2 % and a report

HLUT: two columns HU and SPR (matRad .hlut format, '#' comments; commas allowed). TEMPLATE: a TOPAS Schneider
parameter file, for example examples/Patient/HUtoMaterialSchneider.txt of OpenTOPAS. Its own DensityCorrection is
replaced, all else is kept.

I_water must be the mean excitation energy of the water in which the beam model was fitted: G4_WATER has 78 eV. Use
75 only if that water was defined with 75 eV as well, else the range in the patient shifts against the beam model.
"""
import argparse, json, os, re, sys

import numpy as np
from scipy import constants as sc

HERE = os.path.dirname(os.path.abspath(__file__))
ELEMENTS = json.load(open(os.path.join(HERE, "data", "g4_elements.json")))["elements"]
TOPAS_ELEMENT = {"Hydrogen": "H", "Carbon": "C", "Nitrogen": "N", "Oxygen": "O", "Magnesium": "Mg",
                 "Phosphorus": "P", "Sulfur": "S", "Chlorine": "Cl", "Argon": "Ar", "Calcium": "Ca",
                 "Sodium": "Na", "Potassium": "K", "Titanium": "Ti"}
# G4_WATER: H2O by atom count, density 1.0 g/cm3 (G4NistMaterialBuilder)
WATER = {"H": 2 * ELEMENTS["H"]["A_g_per_mol"], "O": ELEMENTS["O"]["A_g_per_mol"]}
ENERGIES_MEV = np.linspace(70.0, 220.0, 16)
ME_C2_EV = sc.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
MP_C2_MEV = sc.physical_constants["proton mass energy equivalent in MeV"][0]


# ---------------------------------------------------------------- physics
def normalise(fractions):
    tot = sum(fractions.values())
    return {k: v / tot for k, v in fractions.items() if v > 0}


def electrons_per_mass(w):
    """sum w_i Z_i / A_i in mol/g: proportional to the electrons per gram."""
    return sum(f * ELEMENTS[e]["Z"] / ELEMENTS[e]["A_g_per_mol"] for e, f in w.items())


def bragg_i(w):
    """Mean excitation energy in eV as Geant4 computes it for a mixture without a chemical formula."""
    zw = {e: f * ELEMENTS[e]["Z"] / ELEMENTS[e]["A_g_per_mol"] for e, f in w.items()}
    return float(np.exp(sum(v * np.log(ELEMENTS[e]["I_eV"]) for e, v in zw.items()) / sum(zw.values())))


def stopping_number(i_ev, t_mev=ENERGIES_MEV):
    gamma = 1.0 + np.asarray(t_mev, float) / MP_C2_MEV
    b2g2 = gamma ** 2 - 1.0
    return np.log(2.0 * ME_C2_EV * b2g2 / i_ev) - b2g2 / gamma ** 2


def spr(density, w, i_water=78.0, t_mev=ENERGIES_MEV):
    """SPR relative to water (1 g/cm3, I_water) on the energy grid, Bethe without shell and density corrections."""
    w = normalise(w); ww = normalise(WATER)
    return (density * electrons_per_mass(w) / electrons_per_mass(ww)
            * stopping_number(bragg_i(w), t_mev) / stopping_number(i_water, t_mev))


# The CSDA integral runs from here: below 0.5 MeV both residual ranges are under 0.01 mm, so their ratio does not matter.
_GRID_MEV = np.geomspace(0.5, 250.0, 4000)


def csda(i_ev, epm, e0_mev):
    """CSDA range (in units proportional to g/cm2) for initial energies e0_mev, Bethe stopping power."""
    gamma = 1.0 + _GRID_MEV / MP_C2_MEV
    s = epm * stopping_number(i_ev, _GRID_MEV) / (1.0 - 1.0 / gamma ** 2)
    r = np.concatenate([[0.0], np.cumsum(0.5 * (1 / s[1:] + 1 / s[:-1]) * np.diff(_GRID_MEV))])
    return np.interp(e0_mev, _GRID_MEV, r)


def wer(density, w, i_water=78.0, e0_mev=ENERGIES_MEV):
    """Water equivalent ratio, CSDA range in water over CSDA range in the material, for each initial energy. This is
    what sets the range in the patient: the stopping power ratio grows towards low energies when the material has a
    lower I than water, and the range integral weighs those energies more than a mean over 70 to 220 MeV does
    (soft tissue of I 66 eV against G4_WATER at 120 MeV: 0.18 % more, 0.18 mm per 100 mm)."""
    w = normalise(w); ww = normalise(WATER)
    return density * csda(i_water, electrons_per_mass(ww), e0_mev) / csda(bragg_i(w), electrons_per_mass(w), e0_mev)


# ---------------------------------------------------------------- files
def read_hlut(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.split("#", 1)[0].replace(",", " ").split()
        if len(line) >= 2:
            rows.append((float(line[0]), float(line[1])))
    if len(rows) < 2: raise ValueError(f"{path}: fewer than two HU/SPR rows")
    hu, s = np.array(sorted(rows)).T
    if np.any(np.diff(hu) <= 0): raise ValueError(f"{path}: HU values repeat")
    return hu, s


PARAM = re.compile(r"^\s*(\w+):([\w/]+)\s*=\s*(.*)$")


def read_template(path):
    """Parameters of a Schneider file: name -> (type, list of value strings, unit or None), plus the raw lines."""
    lines = open(path, encoding="utf-8").read().splitlines()
    params = {}
    for line in lines:
        m = PARAM.match(line.split("#", 1)[0])
        if not m: continue
        typ, name, rest = m.groups()
        name = name.split("/")[-1]
        toks = re.findall(r'"[^"]*"|\S+', rest)
        unit = None
        if typ[-1] == "v":
            n = int(toks[0]); vals = toks[1:1 + n]
            if len(toks) > 1 + n: unit = toks[1 + n]
            if len(vals) != n: raise ValueError(f"{name}: {len(vals)} values, header says {n}")
        else:
            vals = toks
        params[name] = (typ, [v.strip('"') for v in vals], unit)
    return params, lines


class Schneider:
    def __init__(self, path):
        self.params, self.lines = read_template(path)
        p = lambda k, f=float: [f(v) for v in self.params[k][1]]
        self.hu_sections = p("SchneiderHounsfieldUnitSections", int)
        self.offset, self.factor = p("SchneiderDensityOffset"), p("SchneiderDensityFactor")
        self.factor_offset = p("SchneiderDensityFactorOffset")
        self.mat_sections = p("SchneiderHUToMaterialSections", int)
        names = self.params["SchneiderElements"][1]
        self.weights = []
        for k in range(1, len(self.mat_sections)):
            fr = p(f"SchneiderMaterialsWeight{k}")
            self.weights.append(normalise({TOPAS_ELEMENT[n]: f for n, f in zip(names, fr)}))
        self.min_hu = int(self.params["MinImagingValue"][1][0]) if "MinImagingValue" in self.params else -1000
        self.hu = np.arange(self.hu_sections[0], self.hu_sections[-1])
        if "SchneiderMaterialMeanExcitationEnergy" in self.params:
            raise ValueError("template sets SchneiderMaterialMeanExcitationEnergy; this script assumes Bragg additivity")

    def section(self, edges, hu):
        k = int(np.searchsorted(edges, hu, side="right")) - 1
        if not 0 <= k < len(edges) - 1: raise ValueError(f"HU {hu} outside the sections")
        return k

    def nominal_density(self, hu):
        k = self.section(self.hu_sections, hu)
        return self.offset[k] + self.factor[k] * (self.factor_offset[k] + hu)

    def material(self, hu):
        return self.section(self.mat_sections, hu)


def tune(hlut_hu, hlut_spr, sch, i_water=78.0, spr_scale=1.0, match="range"):
    """DensityCorrection per HU of the template and the per-HU residual (max relative error over energies).

    match "range": the water equivalent ratio (wer) averaged over initial energies 70 to 220 MeV equals the HLUT value,
    so ranges come out right. match "spr": the stopping power ratio averaged over the same energies equals it; this
    left soft tissue ranges 0.37 mm short per 100 mm in TOPAS (checks/ct_range, 2026-11-06)."""
    if match not in ("range", "spr"): raise ValueError(f"match must be range or spr, not {match}")
    ratio = wer if match == "range" else spr
    target = np.interp(sch.hu, hlut_hu, hlut_spr) * spr_scale
    corr = np.empty(len(sch.hu)); resid = np.empty(len(sch.hu)); i_mat = np.empty(len(sch.hu))
    cache = {}
    for j, h in enumerate(sch.hu):
        k = sch.material(int(h))
        if k not in cache:
            w = sch.weights[k]
            cache[k] = (ratio(1.0, w, i_water), bragg_i(w))
        unit_spr, i_mat[j] = cache[k]
        nom = sch.nominal_density(int(h)) * unit_spr
        corr[j] = target[j] / nom.mean()
        resid[j] = np.max(np.abs(corr[j] * nom / target[j] - 1.0))
    return {"hu": sch.hu, "target": target, "correction": corr, "residual": resid, "i_ev": i_mat}


def report(sch, res):
    rows = []
    for k in range(len(sch.mat_sections) - 1):
        lo, hi = sch.mat_sections[k], sch.mat_sections[k + 1]
        m = (res["hu"] >= lo) & (res["hu"] < hi)
        if not m.any(): continue
        r = res["residual"][m]
        rows.append({"hu_from": lo, "hu_to": hi - 1, "i_ev": round(float(res["i_ev"][m][0]), 2),
                     "spr_from": round(float(res["target"][m][0]), 5), "spr_to": round(float(res["target"][m][-1]), 5),
                     "correction_min": round(float(res["correction"][m].min()), 5),
                     "correction_max": round(float(res["correction"][m].max()), 5),
                     "range_error_mm_per_100mm": round(100.0 * float(r.max()), 3)})
    return rows


def write(sch, res, out, component="Patient", header=()):
    vals = " ".join(f"{c:.6g}" for c in res["correction"])
    dv = f"dv:Ge/{component}/DensityCorrection = {len(res['correction'])} {vals} g/cm3"
    body = []
    for line in sch.lines:
        m = PARAM.match(line.split("#", 1)[0])
        if m and m.group(2).split("/")[-1] == "DensityCorrection":
            body.append(dv); continue
        body.append(line.replace("Ge/Patient/", f"Ge/{component}/"))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for h in header: f.write(f"# {h}\n")
        f.write("\n".join(body) + "\n")


def run(hlut, template, out, i_water=78.0, spr_scale=1.0, component="Patient", match="range"):
    hu, s = read_hlut(hlut)
    sch = Schneider(template)
    if sch.min_hu != sch.hu_sections[0]:
        raise ValueError("MinImagingValue differs from the first HU section; DensityCorrection index would shift")
    res = tune(hu, s, sch, i_water, spr_scale, match)
    rows = report(sch, res)
    write(sch, res, out, component, header=[
        f"Schneider conversion tuned to the HLUT {os.path.basename(hlut)} by tools/plan_mc/hlut.py",
        f"I_water {i_water} eV, SPR scale {spr_scale}, matched to {match}, energies {ENERGIES_MEV[0]:g} to "
        f"{ENERGIES_MEV[-1]:g} MeV"])
    return {"hlut": os.path.basename(hlut), "template": os.path.basename(template), "i_water_ev": i_water,
            "spr_scale": spr_scale, "match": match, "energies_mev": [float(ENERGIES_MEV[0]), float(ENERGIES_MEV[-1])],
            "sections": rows}


def print_rows(rows):
    print(f"{'HU':>13} {'I/eV':>7} {'SPR':>15} {'corr':>15} {'mm/100mm':>9}")
    for r in rows:
        print(f"{r['hu_from']:>6}..{r['hu_to']:<6} {r['i_ev']:7.2f} {r['spr_from']:7.4f}..{r['spr_to']:<7.4f}"
              f"{r['correction_min']:7.4f}..{r['correction_max']:<7.4f}{r['range_error_mm_per_100mm']:9.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("write", "variants"):
        a = sub.add_parser(name)
        a.add_argument("hlut"); a.add_argument("template"); a.add_argument("out")
        a.add_argument("--i-water", type=float, default=78.0)
        a.add_argument("--component", default="Patient")
        a.add_argument("--match", choices=("range", "spr"), default="range")
        if name == "write":
            a.add_argument("--spr-scale", type=float, default=1.0)
            a.add_argument("--report")
    a = ap.parse_args()
    if a.cmd == "write":
        rep = run(a.hlut, a.template, a.out, a.i_water, a.spr_scale, a.component, a.match)
        if a.report: json.dump(rep, open(a.report, "w"), indent=1)
        print_rows(rep["sections"])
    else:
        os.makedirs(a.out, exist_ok=True)
        reps = {}
        for tag, scale in (("nominal", 1.0), ("spr_plus2", 1.02), ("spr_minus2", 0.98)):
            reps[tag] = run(a.hlut, a.template, os.path.join(a.out, f"schneider_{tag}.txt"), a.i_water, scale,
                            a.component, a.match)
        json.dump(reps, open(os.path.join(a.out, "hlut_report.json"), "w"), indent=1)
        print_rows(reps["nominal"]["sections"])
