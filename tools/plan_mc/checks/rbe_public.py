"""Public check of the variable RBE models (rbe.py) on runs of the public chain that scored ProtonLET.

    python3 rbe_public.py OUTDIR [--alpha-beta 2] [--fraction-gy 2] [--let-threshold 0.05]

OUTDIR is a folder of public_chain.py (chain.json, field*_dose.bin, field*_let.bin), e.g. the TG119 plan on its CT
(E4) or in water (E2, where each field is a spread out Bragg peak in water). Research only: variable RBE models carry
large uncertainties; clinically 1.1 applies.

The absolute dose of the chain is a hypothesis (w x 1e6 particles), and the RBE depends on the dose per fraction, so the
summed dose is scaled to a mean of FRACTION / 1.1 Gy where it is at least 50 % of its maximum (2 Gy(RBE) at 1.1).
Each field is also taken alone, as its own SOBP: its RBE from its own dose (scaled with the same factor) and LETd.
What the check expects from the literature (Paganetti 2014, doi:10.1088/0031-9155/59/22/R419; McNamara 2015): LETd and
RBE rise toward the distal edge of each field (here: the last 5 mm before the R80 of the field against 30 to 15 mm
before it, per ray), and the distal 80 % isodose of the RBE weighted dose lies about 1 to 3 mm deeper than with 1.1 (rbe.distal_shift
gives it at the absolute level of the 1.1 dose and for the shape alone).
It writes rbe_check.json into OUTDIR and prints a summary.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plan_mc import iec, rbe  # noqa: E402
from plan_mc.checks.public_chain import field_dose  # noqa: E402


WINDOWS = {"proximal": (-30.0, -15.0), "distal": (-5.0, 0.0)}   # mm before the R80 of the field on each ray


def trend(arrays, d, origin, spacing, u):
    """Dose weighted means of each array in the WINDOWS before the distal R80 of the field dose d, over all rays
    through its high dose region: {window: [mean of each array]}. The plan's own fields are not clean SOBPs (TG119
    spots spread over the target), so a region by dose level alone would pick noise; a window by depth does not."""
    sums = {w: np.zeros(len(arrays) + 1) for w in WINDOWS}
    for ts, (dp, *ps) in rbe.rays([d] + list(arrays), origin, spacing, u):
        r80 = rbe.distal_edge(ts, dp)
        if r80 is None: continue
        for w, (lo, hi) in WINDOWS.items():
            sel = (ts >= r80 + lo) & (ts < r80 + hi)
            sums[w] += [float(dp[sel].sum())] + [float((dp * p)[sel].sum()) for p in ps]
    return {w: (v[1:] / v[0]).tolist() if v[0] > 0 else None for w, v in sums.items()}


def check(out, ab=2.0, fraction_gy=2.0, let_threshold=0.05):
    out = os.path.abspath(out)
    c = json.load(open(os.path.join(out, "chain.json")))
    shape, origin, spacing = tuple(c["shape_zyx"]), c["origin"], c["spacing"]
    doses, lets = [], []
    for m in c["fields"]:
        doses.append(field_dose(out, m, shape)[0])
        let, _ = rbe.read_let(os.path.join(out, f"field{m['number']}_let.bin"), m["histories"])
        lets.append(let.transpose(2, 1, 0))
    total = sum(doses)
    scale = fraction_gy / rbe.CONSTANT / float(total[total >= 0.5 * total.max()].mean())
    doses = [d * scale for d in doses]; total = total * scale
    letd = rbe.combine_letd(doses, lets)
    valid = rbe.let_mask(total, let_threshold)
    hi = valid & (total >= 0.5 * total.max())
    res = {"alpha_beta_gy": ab, "fraction_gy_rbe_at_1_1": fraction_gy, "dose_scale": scale,
           "let_threshold": let_threshold, "letd_high_dose_q50_q95": np.percentile(letd[hi], [50, 95]).tolist(),
           "plan": {}, "fields": {}}
    for name in sorted(rbe.MODELS):
        r = rbe.rbe(name, total, letd, ab, valid)
        res["plan"][name] = {"rbe_high_dose_mean": float(r[hi].mean()),
                             "rbe_high_dose_q05_q50_q95": np.percentile(r[hi], [5, 50, 95]).tolist()}
    names = sorted(rbe.MODELS)
    for m, d, let in zip(c["fields"], doses, lets):
        _, u = iec.spot_ray(0.0, 0.0, m["gantry"], m["couch"], m["iso"], position="HFS")
        ok = rbe.let_mask(d, let_threshold)
        rs = [rbe.rbe(name, d, let, ab, ok) for name in names]      # the field alone, as its own SOBP
        t = trend([let] + rs, d, origin, spacing, u)
        f = {"direction": np.round(u, 4).tolist(), "windows_mm_before_r80": WINDOWS,
             "letd": {w: v[0] for w, v in t.items()}, "models": {}}
        for k, (name, r) in enumerate(zip(names, rs), 1):
            f["models"][name] = {"rbe": {w: v[k] for w, v in t.items()},
                                 "distal_shift": rbe.distal_shift(rbe.CONSTANT * d, r * d, origin, spacing, u)}
        res["fields"][str(m["number"])] = f
    json.dump(res, open(os.path.join(out, "rbe_check.json"), "w"), indent=1)
    print(f"{os.path.basename(out)}: alpha/beta {ab:g} Gy, {fraction_gy:g} Gy(RBE) per fraction at 1.1, "
          f"LETd above 50 %: median {res['letd_high_dose_q50_q95'][0]:.2f}, q95 {res['letd_high_dose_q50_q95'][1]:.2f} keV/um")
    for name, v in res["plan"].items(): print(f"  plan {name}: mean RBE above 50 % {v['rbe_high_dose_mean']:.3f}")
    for n, f in res["fields"].items():
        print(f"  field {n}: LETd 30 to 15 mm before R80 {f['letd']['proximal']:.2f}, last 5 mm {f['letd']['distal']:.2f} keV/um")
        for name, v in f["models"].items():
            s = v["distal_shift"]
            print(f"    {name}: RBE {v['rbe']['proximal']:.3f} -> {v['rbe']['distal']:.3f}; " + shift_text(v["distal_shift"]))
    return res


def shift_text(s):
    if not s or not s["rays"]["kept"]: return "no distal edge found"
    return "; ".join(f"{k} {s[k]['median_mm']:+.2f} mm (5 to 95 %: {s[k]['q05_mm']:+.2f} to {s[k]['q95_mm']:+.2f}, "
                     f"{s[k]['rays']} rays)" for k in ("absolute", "shape")) + f"; {s['rays']['dropped']} rays dropped"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out"); ap.add_argument("--alpha-beta", type=float, default=2.0)
    ap.add_argument("--fraction-gy", type=float, default=2.0); ap.add_argument("--let-threshold", type=float, default=0.05)
    a = ap.parse_args()
    check(a.out, a.alpha_beta, a.fraction_gy, a.let_threshold)
