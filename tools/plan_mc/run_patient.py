"""Start script for the recalculation of a clinical plan in TOPAS, run locally by the person who holds the data.

Research use only: TOPAS is not approved for clinical use; the results never inform a treatment decision.

    python3 run_patient.py write    --workspace WS --name NAME --ct CTDIR --rtplan RP.dcm --hlut HLUT \
                                    --template HUtoMaterialSchneider.txt --beam-model MODEL.json \
                                    [--rtdose RD.dcm] [--histories 10000000] [--i-water 78] \
                                    [--rs-material Lexan --rs-thickness MM] [--ripple-filter t4 --rifi-distance MM]
    python3 run_patient.py queue    --workspace WS --name NAME
    python3 run_patient.py status   --workspace WS --name NAME
    python3 run_patient.py evaluate --workspace WS --name NAME

Everything goes to WS/patient/plan_mc/NAME/ and nowhere else: the Schneider file tuned to the HLUT (hlut.py), one
TOPAS file per field (spot by spot, topas_writer.py), the summed dose as RT Dose, and BERICHT.md with the comparison
against the TPS dose. The CT is read where it lies; nothing is copied out of the patient folder.

Range shifter: the RT Ion Plan states only the water equivalent thickness and the isocenter distance; the physical
thickness and the material of the machine's shifter come from --rs-material and --rs-thickness. A field with a range
shifter and without these values is refused.
Dose: per fraction as the plan's BeamMeterset is; an RT Dose with DoseSummationType PLAN is divided by the number of
fractions of the plan before the comparison. The beam model's particles_per_mu and dose_scale set the absolute dose
(abs_dose.py fits dose_scale to a reference measurement).
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan_mc import beam_model, evaluate, hlut, plan as pl, synthetic, topas_io, topas_writer as tw, wbqueue  # noqa: E402


def folder(ws, name):
    if not name or "/" in name or "\\" in name or name.startswith("."): raise SystemExit(f"bad run name {name!r}")
    return os.path.join(os.path.abspath(ws), "patient", "plan_mc", name)


def beamline(field, a):
    line = {}
    if field.range_shifter:
        if not (a.rs_material and a.rs_thickness):
            raise SystemExit(f"field {field.name} has a range shifter ({field.range_shifter.rs_id}, "
                             f"{field.range_shifter.wet:g} mm WET): give --rs-material and --rs-thickness")
        line["range_shifter"] = {"material": a.rs_material, "thickness_mm": a.rs_thickness,
                                 "iso_distance_mm": field.range_shifter.iso_distance}
    if a.ripple_filter:
        if a.rifi_distance is None: raise SystemExit("--ripple-filter needs --rifi-distance (isocenter distance, mm)")
        line["ripple_filter"] = {"variant": a.ripple_filter, "iso_distance_mm": a.rifi_distance}
    return line


def write(a):
    out = folder(a.workspace, a.name); os.makedirs(out, exist_ok=True)
    rep = hlut.run(a.hlut, a.template, os.path.join(out, "schneider.txt"), i_water=a.i_water)
    json.dump(rep, open(os.path.join(out, "schneider_report.json"), "w"), indent=1)
    plan = pl.read_rtplan(a.rtplan)
    model = beam_model.BeamModel.load(a.beam_model)
    model.save(os.path.join(out, "beam_model.json"))
    pat = os.path.join(out, "patient.txt")
    grid = "Patient/RTDoseGrid" if a.rtdose else "Patient"
    open(pat, "w").write(tw.patient_block("ct", grid, schneider=os.path.join(out, "schneider.txt"),
                                          dicom_dir=os.path.abspath(a.ct),
                                          rtdose=os.path.abspath(a.rtdose) if a.rtdose else None))
    metas = []
    for f in plan.fields:
        text, meta = tw.field_file(plan, f, model, beamline(f, a), pat, a.histories, out_name=f"field{f.number}")
        open(os.path.join(out, f"field{f.number}.txt"), "w").write(text)
        metas.append(meta)
    json.dump({"rtplan": os.path.abspath(a.rtplan), "rtdose": os.path.abspath(a.rtdose) if a.rtdose else None,
               "ct": os.path.abspath(a.ct), "fractions": plan.fractions, "histories": a.histories,
               "i_water": a.i_water, "fields": metas}, open(os.path.join(out, "run.json"), "w"), indent=1)
    print(f"{len(metas)} fields with {a.histories} histories each in {out}")


def _files(a):
    out = folder(a.workspace, a.name)
    run = json.load(open(os.path.join(out, "run.json")))
    return out, run, [os.path.join(out, f"field{m['number']}.txt") for m in run["fields"]]


def queue(a):
    _, _, files = _files(a)
    print(json.dumps(wbqueue.enqueue(a.workspace, files)))


def status(a):
    _, _, files = _files(a)
    for f, (st, rc) in wbqueue.status(a.workspace, files).items(): print(f"{st:9} rc={rc} {f}")


def summed_dose(out, run, model_scale):
    total = var = None
    for m in run["fields"]:
        _, arr = topas_io.read(os.path.join(out, f"field{m['number']}_dose.bin"))
        k = m["particles"] / m["histories"] * model_scale
        s = np.asarray(arr["Sum"], float).transpose(2, 1, 0) * k     # [x, y, z] -> [z, y, x]
        sd = topas_io.sum_sd(arr["Standard_Deviation"], m["histories"]).transpose(2, 1, 0) * k
        total = s if total is None else total + s
        var = sd ** 2 if var is None else var + sd ** 2
    return total, np.sqrt(var)


def evaluate_run(a):
    out, run, _ = _files(a)
    model = beam_model.BeamModel.load(os.path.join(out, "beam_model.json"))
    dose, sd = summed_dose(out, run, model.dose_scale)
    hi = dose >= 0.5 * dose.max()
    unc = float(np.sqrt(np.mean((sd[hi] / dose[hi]) ** 2)))
    lines = ["# TOPAS-Nachrechnung (nur Forschung, keine klinische Entscheidung)", "",
             f"Plan: {os.path.basename(run['rtplan'])}, {len(run['fields'])} Felder, {run['histories']} Historien je Feld.",
             f"Statistische Unsicherheit (1 sigma, rms ueber Voxel > 50 % Maximum): {100 * unc:.2f} %.",
             f"Maximale Dosis TOPAS je Fraktion: {dose.max():.3f} Gy.", ""]
    res = {"uncertainty_rms": unc, "topas_max_gy": float(dose.max())}
    if run["rtdose"]:
        import pydicom
        ref, origin, spacing = synthetic.read_rtdose(run["rtdose"])
        if str(getattr(pydicom.dcmread(run["rtdose"], stop_before_pixels=True), "DoseSummationType", "")).upper() == "PLAN":
            ref = ref / max(1, run["fractions"])
        if ref.shape != dose.shape: raise SystemExit(f"TPS dose grid {ref.shape} differs from the TOPAS grid {dose.shape}")
        synthetic.write_rtdose(os.path.join(out, "topas_dose.dcm"), dose, origin, spacing)
        cmp_ = evaluate.compare(ref, dose, origin, spacing)
        res["comparison"] = cmp_
        lines += ["| Kriterium | Bestehensrate |", "|---|---|"]
        lines += [f"| Gamma {k} global, Schwelle 10 % | {v['pass_pct']:.1f} % |" for k, v in cmp_["gamma"].items()]
        lines += ["", f"Maximum TPS {cmp_['ref_max']:.3f} Gy, TOPAS {cmp_['eval_max']:.3f} Gy je Fraktion."]
    else:
        np.save(os.path.join(out, "topas_dose.npy"), dose.astype(np.float32))
        lines.append("Keine TPS-Dosis angegeben: nur die TOPAS-Dosis (topas_dose.npy, Gitter des CT).")
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    open(os.path.join(out, "BERICHT.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("write", "queue", "status", "evaluate"))
    ap.add_argument("--workspace", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--ct"); ap.add_argument("--rtplan"); ap.add_argument("--rtdose"); ap.add_argument("--hlut")
    ap.add_argument("--template"); ap.add_argument("--beam-model")
    ap.add_argument("--histories", type=int, default=10_000_000); ap.add_argument("--i-water", type=float, default=78.0)
    ap.add_argument("--rs-material"); ap.add_argument("--rs-thickness", type=float)
    ap.add_argument("--ripple-filter"); ap.add_argument("--rifi-distance", type=float)
    a = ap.parse_args(argv)
    if a.cmd == "write":
        miss = [k for k in ("ct", "rtplan", "hlut", "template", "beam_model") if not getattr(a, k)]
        if miss: ap.error("write needs " + ", ".join("--" + k.replace("_", "-") for k in miss))
    {"write": write, "queue": queue, "status": status, "evaluate": evaluate_run}[a.cmd](a)


if __name__ == "__main__":
    main()
