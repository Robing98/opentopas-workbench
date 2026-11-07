"""Public chain E4 on a matRad plan (part E4 on public data): matRad plan > CT as DICOM > tuned Schneider file >
spot-by-spot TOPAS field files with the generic beam model > workbench queue > comparison with the matRad dose.

    python3 public_chain.py write OUTDIR PLAN.mat SCHNEIDER [--histories N]   # CT, patient file, one file per field
    python3 public_chain.py check OUTDIR [--workspace WS]                     # after the runs: report.json, summary

SCHNEIDER is a Schneider file that hlut.py tuned to the HLUT the matRad plan was computed with (for TG119 from
pyRadPlan: matRad_default.hlut). Dose is scored on the CT grid, so the TOPAS dose and resultGUI.physicalDose share it.
Dose of a field = TOPAS Sum * particles / histories (topas_writer: every history carries the same weight).
The generic beam model is not pyRadPlan's Generic machine (spot size, energy spread, no fitted mean energies), so
differences in the distal falloff and the penumbra are expected; the check reports them, it does not judge them.
Queue the files through the workbench (wbqueue.py); nothing here starts TOPAS.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plan_mc import beam_model, evaluate, matrad, synthetic, topas_io, topas_writer as tw  # noqa: E402


def write(out, mat, schneider, histories):
    out = os.path.abspath(out); os.makedirs(out, exist_ok=True)
    r = matrad.read(mat)
    ct = os.path.join(out, "ct")
    synthetic.write_ct(ct, r["hu"], r["origin"], r["spacing"])
    pat = os.path.join(out, "patient.txt")
    open(pat, "w").write(tw.patient_block("ct", "Patient", schneider=os.path.abspath(schneider), dicom_dir=ct))
    if r["dose"] is not None: np.save(os.path.join(out, "ref_dose.npy"), r["dose"])
    model = beam_model.generic()
    model.save(os.path.join(out, "beam_model.json"))
    metas, files = [], []
    for f in r["plan"].fields:
        # no physical shifter data for a matRad plan: dropping its range shifter would put every spot too deep
        if f.range_shifter: raise SystemExit(f"field {f.number} uses a range shifter, which this check cannot model")
        name = f"field{f.number}"
        text, meta = tw.field_file(r["plan"], f, model, {}, pat, histories, out_name=name)
        p = os.path.join(out, f"{name}.txt")
        open(p, "w").write(text)
        metas.append(meta); files.append(p)
    json.dump({"mat": os.path.abspath(mat), "schneider": os.path.abspath(schneider), "origin": list(r["origin"]),
               "spacing": list(r["spacing"]), "shape_zyx": list(r["hu"].shape), "fields": metas},
              open(os.path.join(out, "chain.json"), "w"), indent=1)
    print(f"{len(files)} fields, {histories} histories each, in {out}")
    for p in files: print(p)


def field_dose(out, meta, shape_zyx):
    """Dose [z, y, x] in Gy and its standard deviation for one field, scaled to the plan's particle number."""
    h, a = topas_io.read(os.path.join(out, f"field{meta['number']}_dose.bin"))
    k = meta["particles"] / meta["histories"]
    s = np.asarray(a["Sum"], float).transpose(2, 1, 0) * k            # topas_io gives [x, y, z]
    sd = topas_io.sum_sd(a["Standard_Deviation"], meta["histories"]).transpose(2, 1, 0) * k
    if s.shape != tuple(shape_zyx): raise ValueError(f"field {meta['number']}: TOPAS grid {s.shape}, CT {shape_zyx}")
    return s, sd


def runtime(ws, out, meta):
    """Wall time of the newest workbench run of a field [s], or None."""
    from plan_mc.checks.runtime_split import seconds
    return seconds(ws, os.path.join(out, f"field{meta['number']}.txt"))


def check(out, ws=None):
    out = os.path.abspath(out)
    c = json.load(open(os.path.join(out, "chain.json")))
    shape = tuple(c["shape_zyx"])
    total, var = np.zeros(shape), np.zeros(shape)
    rows = []
    for meta in c["fields"]:
        d, sd = field_dose(out, meta, shape)
        total += d; var += sd ** 2
        t = runtime(ws, out, meta) if ws else None
        rows.append({"field": meta["field"], "histories": meta["histories"], "particles": meta["particles"],
                     "max_gy": float(d.max()), "wall_s": t, "ms_per_history": 1000.0 * t / meta["histories"] if t else None})
    np.save(os.path.join(out, "topas_dose.npy"), total.astype(np.float32))
    m = matrad._load(c["mat"])
    cst = matrad.read_cst(m, shape)
    ref = np.load(os.path.join(out, "ref_dose.npy"))
    sd = np.sqrt(var)
    hi = total >= 0.5 * total.max()
    unc = float(np.sqrt(np.mean((sd[hi] / total[hi]) ** 2)))   # rms relative 1 sigma above 50 % of the maximum
    targets = [mk for _, t, mk in cst if t.upper() == "TARGET"]
    iso = np.mean(np.argwhere(targets[0]), axis=0) if targets else np.array(shape) / 2   # [z, y, x] index
    point = [c["origin"][k] + c["spacing"][k] * iso[2 - k] for k in range(3)]
    res = evaluate.compare(ref, total, c["origin"], c["spacing"], cst, point=point)
    # The absolute scale (w * 1e6 particles) is a hypothesis; the shape comparison removes it. The factor is the
    # mean ratio where the matRad dose is >= 50 % of its maximum, not in the target: in the water phantom (E2)
    # every spot stops before the target, which then has no dose at all.
    k = res.get("high_dose_mean_ratio_eval_over_ref") or 1.0
    shape_only = evaluate.compare(ref, total / k, c["origin"], c["spacing"], cst)
    report = {"fields": rows, "rel_uncertainty_rms_above_50pct": unc, "absolute": res,
              "normalised_to_high_dose_mean": {"factor": k, "gamma": shape_only["gamma"],
                                               "structures": shape_only["structures"]}}
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['field']}: {r['histories']} histories, max {r['max_gy']:.3g} Gy"
              + (f", {r['wall_s']:.0f} s, {r['ms_per_history']:.2f} ms per history" if r["wall_s"] else ""))
    print(f"rms relative uncertainty (1 sigma) above 50 % of the maximum: {100 * unc:.2f} %")
    t_ratio = res.get("target_mean_ratio_eval_over_ref")
    print(f"mean TOPAS / matRad where matRad >= 50 % of its maximum: {k:.4f}; in the target: "
          + (f"{t_ratio:.4f}" if t_ratio is not None else "no matRad dose there"))
    for lab, g in (("absolute", res["gamma"]), ("normalised", shape_only["gamma"])):
        print(lab + ": " + ", ".join(f"gamma {c_} {v['pass_pct']:.1f} %" for c_, v in g.items()))
    for name, v in shape_only["structures"].items():
        if v["ref"] and v["ref"]["mean"] > 0: print(f"{name}: D95 {v['ref']['D95']:.2f} vs {v['eval']['D95']:.2f}, "
                           f"D50 {v['ref']['D50']:.2f} vs {v['eval']['D50']:.2f}, mean {v['ref']['mean']:.2f} vs {v['eval']['mean']:.2f} (matRad vs TOPAS normalised)")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.add_argument("out"); w.add_argument("mat"); w.add_argument("schneider")
    w.add_argument("--histories", type=int, default=200000)
    k = sub.add_parser("check"); k.add_argument("out"); k.add_argument("--workspace")
    a = ap.parse_args()
    if a.cmd == "write": write(a.out, a.mat, a.schneider, a.histories)
    else: check(a.out, a.workspace)
