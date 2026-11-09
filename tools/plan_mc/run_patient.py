"""Start script for the recalculation of a clinical plan in TOPAS, run locally by the person who holds the data.

Research use only: TOPAS is not approved for clinical use; the results never inform a treatment decision.

    eval "$(python3 run_patient.py find EXPORT [--hlut-dir DIR] [--workspace WS])"     # sets CT= RP= RD= HLUT=
    python3 run_patient.py write    --workspace WS --name NAME --ct "$CT" --rtplan "$RP" --hlut "$HLUT" \
                                    --template HUtoMaterialSchneider.txt --beam-model MODEL.json|generic \
                                    [--rtdose "$RD"] [--histories 10000000] [--i-water 78] [--spots-from-plan] \
                                    [--rs-material Lexan --rs-thickness MM [--rs-iso-distance MM]] \
                                    [--ripple-filter t4 --rifi-distance MM]
    python3 run_patient.py queue    --workspace WS --name NAME
    python3 run_patient.py status   --workspace WS --name NAME
    python3 run_patient.py evaluate --workspace WS --name NAME [--rbe 1.1] [--normalised-gamma]

Everything goes to WS/lokal/patient/plan_mc/NAME/ (WS/patient/plan_mc/NAME/ in a workspace without zones) and nowhere
else: the Schneider file tuned to the HLUT (hlut.py), one
TOPAS file per field (spot by spot, topas_writer.py), the summed dose as RT Dose, and BERICHT.md with the comparison
against the TPS dose. The CT is read where it lies; nothing is copied out of the patient folder. No message and no
report names a file of the patient data: file names of an export carry the patient's name.

find: reads the headers of every DICOM file below EXPORT and prints counts on stderr and shell assignments on stdout.
The plan, the dose and the CT are matched by their references (RT Dose to RT Ion Plan, frame of reference of the CT);
when the references do not fit, exactly one candidate of a kind is taken with a warning.

Range shifter: the RT Ion Plan states only the water equivalent thickness and the isocenter distance; the physical
thickness and the material of the machine's shifter come from --rs-material and --rs-thickness. A field with a range
shifter and without these values is refused. Without IsocenterToRangeShifterDistance in the plan, --rs-iso-distance
gives it; a distance of 0 would put the shifter into the isocenter and is refused.
Dose: per fraction as the plan's BeamMeterset is; an RT Dose with DoseSummationType PLAN is divided by the number of
fractions of the plan before the comparison. The beam model's particles_per_mu and dose_scale set the absolute dose
(abs_dose.py fits dose_scale to a reference measurement). A beam in NP (particles) needs no particles_per_mu.
RBE: TOPAS scores physical dose. A TPS dose of DoseType EFFECTIVE is compared with RBE x TOPAS, RBE 1.1 by default
(1.0 for PHYSICAL); --rbe sets it. The report states the RBE and the ratio of the physical doses.
"""
import argparse, glob, json, os, shlex, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan_mc import beam_model, dicom_compat, evaluate, hlut, plan as pl, synthetic, topas_io, topas_writer as tw, wbqueue  # noqa: E402,E501

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PUBLIC_HLUT = os.path.join("offen", "plan_mc", "hluts", "matRad_default.hlut")     # below a workspace
CT_CLASS, PLAN_CLASS, DOSE_CLASS = synthetic.CT_CLASS, "1.2.840.10008.5.1.4.1.1.481.8", synthetic.RTDOSE_CLASS


def warn(msg): print(f"warning: {msg}", file=sys.stderr)


def folder(ws, name):
    """Run folder of NAME. A workspace with data zones (offen/, robin/, lokal/) keeps patient runs in lokal/; an older
    workspace without them keeps the former place, so a run already written there is still found."""
    # ":" too: on Windows Python a drive-relative name such as "D:x" would leave the patient folder
    if not name or any(c in name for c in "/\\:") or name.startswith("."): raise SystemExit(f"bad run name {name!r}")
    ws = os.path.abspath(ws)
    base = os.path.join(ws, "lokal") if os.path.isdir(os.path.join(ws, "lokal")) else ws
    return os.path.join(base, "patient", "plan_mc", name)


def public_hlut(ws=None):
    """The public matRad default HLUT: in the given workspace, else in the workspace of this checkout (a git archive
    has none)."""
    for w in ([ws] if ws else []) + [os.path.join(ROOT, "workspace")]:
        p = os.path.join(w, PUBLIC_HLUT)
        if os.path.exists(p): return p
    return None


# ---------------------------------------------------------------- find
def scan(export):
    """Headers of the DICOM files below `export`: CT series, RT Ion Plans and RT Doses with their references."""
    import pydicom
    series, plans, doses, other_plans = {}, [], [], 0
    for d, _, names in os.walk(export):
        for n in sorted(names):
            p = os.path.join(d, n)
            try: ds = pydicom.dcmread(p, stop_before_pixels=True)
            except Exception: continue
            cls = str(getattr(ds, "SOPClassUID", ""))
            if cls == CT_CLASS:
                s = series.setdefault(str(ds.SeriesInstanceUID), {"dirs": set(), "slices": 0,
                                                                  "frame": str(getattr(ds, "FrameOfReferenceUID", ""))})
                s["dirs"].add(d); s["slices"] += 1
            elif cls == PLAN_CLASS:
                plans.append({"path": p, "uid": str(ds.SOPInstanceUID), "frame": str(getattr(ds, "FrameOfReferenceUID", ""))})
            elif cls.startswith("1.2.840.10008.5.1.4.1.1.481.") and str(getattr(ds, "Modality", "")) == "RTPLAN":
                other_plans += 1
            elif cls == DOSE_CLASS:
                refs = [str(r.ReferencedSOPInstanceUID) for r in getattr(ds, "ReferencedRTPlanSequence", [])]
                doses.append({"path": p, "refs": refs, "kind": str(getattr(ds, "DoseSummationType", "") or "?").upper(),
                              "frame": str(getattr(ds, "FrameOfReferenceUID", ""))})
    return series, plans, doses, other_plans


def find(a):
    series, plans, doses, other_plans = scan(a.export)
    err = lambda m: print(m, file=sys.stderr)
    err(f"CT series: {len(series)} (slices: {', '.join(str(s['slices']) for s in series.values()) or '-'})")
    err(f"RT Ion Plans: {len(plans)}" + (f" (and {other_plans} RT Plans of other kinds, not read)" if other_plans else ""))
    kinds = {}
    for d in doses: kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    err(f"RT Doses: {len(doses)}" + (" (" + ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())) + ")" if doses else ""))
    if not plans: raise SystemExit("no RT Ion Plan in the export")
    whole = [d for d in doses if d["kind"] in ("PLAN", "FRACTION")]
    if len(plans) == 1: plan = plans[0]
    else:
        hit = [p for p in plans if any(p["uid"] in d["refs"] for d in whole)]
        if len(hit) != 1:
            raise SystemExit(f"{len(plans)} RT Ion Plans and {len(hit)} of them referenced by a plan dose: "
                             "export one plan, or one plan with its dose")
        plan = hit[0]; warn(f"{len(plans)} RT Ion Plans: took the one the plan dose references")
    fits = [d for d in doses if plan["uid"] in d["refs"]]
    err(f"RT Dose to the plan: fits {len(fits)}, missing {sum(1 for d in doses if not d['refs'])}, "
        f"other plan {sum(1 for d in doses if d['refs'] and plan['uid'] not in d['refs'])}")
    rd = None
    for pool, note in ((fits, None), (doses, "the RT Dose does not reference this plan (another plan version?): "
                                             "check that it belongs to it")):
        cand = [d for d in pool if d["kind"] == "PLAN"] or [d for d in pool if d["kind"] == "FRACTION"]
        if len(cand) == 1:
            rd = cand[0]
            if note: warn(note)
            break
        if len(cand) > 1: raise SystemExit(f"{len(cand)} RT Doses of the whole plan fit: export only the one to compare")
    if rd is None: warn("no RT Dose of the whole plan (PLAN or FRACTION): write runs without comparison")
    same = [k for k, s in series.items() if s["frame"] and s["frame"] == plan["frame"]]
    if len(same) == 1: ct = same[0]
    elif len(same) > 1: raise SystemExit(f"{len(same)} CT series share the plan's frame of reference: export only the planning CT")
    elif len(series) == 1:
        ct = next(iter(series)); warn("the CT series has another frame of reference than the plan: check that it is the planning CT")
    else: raise SystemExit(f"{len(series)} CT series and none in the plan's frame of reference")
    dirs = series[ct]["dirs"]
    if len(dirs) != 1: raise SystemExit("the planning CT lies in several folders: TOPAS reads one folder")
    ct_dir = next(iter(dirs))
    if any(ct_dir in s["dirs"] for k, s in series.items() if k != ct):
        warn("the CT folder holds slices of another CT series too: TOPAS reads every slice in the folder")
    hl = None
    if a.hlut_dir:
        found = sorted(glob.glob(os.path.join(a.hlut_dir, "*.hlut")))
        if len(found) > 1: raise SystemExit(f"{len(found)} .hlut files in --hlut-dir: leave the one to use")
        hl = found[0] if found else None
    if hl is None:
        hl = public_hlut(a.workspace)
        if hl: warn("no HLUT of the clinic: the public matRad default HLUT stands in, for probes only")
        else: warn("no HLUT found: give --hlut-dir, --workspace with offen/plan_mc/hluts, or --hlut to write")
    for k, v in (("CT", ct_dir), ("RP", plan["path"]), ("RD", rd["path"] if rd else ""), ("HLUT", hl or "")):
        print(f"{k}={shlex.quote(v)}")


# ---------------------------------------------------------------- write
def rs_distance(field, a):
    """Isocenter to range shifter distance [mm] and where it comes from: the plan, else --rs-iso-distance."""
    d = field.range_shifter.iso_distance
    if d is not None and d > 0: return d, "plan"
    if d is not None: warn(f"field {field.name}: the plan gives the range shifter distance {d:g} mm, taken as missing")
    if a.rs_iso_distance is None:
        raise SystemExit(f"field {field.name}: the plan gives no IsocenterToRangeShifterDistance: give --rs-iso-distance")
    if a.rs_iso_distance <= 0: raise SystemExit("--rs-iso-distance must be above 0 (0 puts the shifter into the isocenter)")
    return a.rs_iso_distance, "--rs-iso-distance"


def beamline(field, a):
    line = {}
    if field.range_shifter:
        if not (a.rs_material and a.rs_thickness):
            raise SystemExit(f"field {field.name} has a range shifter ({field.range_shifter.rs_id}, "
                             f"{field.range_shifter.wet:g} mm WET): give --rs-material and --rs-thickness")
        d, src = rs_distance(field, a)
        print(f"field {field.name}: range shifter {field.range_shifter.wet:g} mm WET, "
              f"{a.rs_thickness:g} mm {a.rs_material}, {d:g} mm from the isocenter ({src})")
        line["range_shifter"] = {"material": a.rs_material, "thickness_mm": a.rs_thickness, "iso_distance_mm": d}
    if a.ripple_filter:
        if a.rifi_distance is None: raise SystemExit("--ripple-filter needs --rifi-distance (isocenter distance, mm)")
        line["ripple_filter"] = {"variant": a.ripple_filter, "iso_distance_mm": a.rifi_distance}
    return line


def one_range_shifter(plan):
    """--rs-thickness gives one physical slab: refuse a plan whose fields use different shifters."""
    kinds = {(f.range_shifter.rs_id, round(f.range_shifter.wet, 3)) for f in plan.fields if f.range_shifter}
    if len(kinds) > 1:
        raise SystemExit("fields use different range shifters (" + ", ".join(f"{i} {w:g} mm WET" for i, w in sorted(kinds))
                         + "); one --rs-thickness cannot stand for all: run the fields of each shifter as their own run")


def write(a):
    out = folder(a.workspace, a.name); os.makedirs(out, exist_ok=True)
    if os.path.basename(a.hlut).lower().startswith("matrad_default"):
        warn("the HLUT is the public matRad default, not the clinic's: for probes only")
    rep = hlut.run(a.hlut, a.template, os.path.join(out, "schneider.txt"), i_water=a.i_water)
    json.dump(rep, open(os.path.join(out, "schneider_report.json"), "w"), indent=1)
    plan = pl.read_rtplan(a.rtplan)
    plan.label = pl.safe_label(a.rtplan)          # the label goes into the TOPAS files and the report
    one_range_shifter(plan)
    if a.beam_model == "generic":
        model = beam_model.generic()
        warn("generic beam model (spot_dist placeholders, not the machine's data): for probes only")
    else: model = beam_model.BeamModel.load(a.beam_model)
    if a.spots_from_plan:
        try: model = beam_model.with_plan_spots(model, plan)
        except ValueError as e: raise SystemExit(f"--spots-from-plan: {e}")
        print(f"spot sizes from the plan for {len(model.rows)} energies, in beam_model.json of the run only")
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
               "ct": os.path.abspath(a.ct), "plan_label": plan.label, "fractions": plan.fractions,
               "histories": a.histories, "i_water": a.i_water, "spots_from_plan": bool(a.spots_from_plan),
               "fields": metas}, open(os.path.join(out, "run.json"), "w"), indent=1)
    units = sorted({m["unit"] for m in metas})
    print(f"{len(metas)} fields ({', '.join(units)}), {plan.fractions} fractions, {a.histories} histories each in {out}")


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


# ---------------------------------------------------------------- evaluate
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


def body_mask(run, dose, ref):
    """Voxels inside the patient, and how they were chosen. Air voxels at low statistics would rule the maximum and
    the uncertainty: TPS dose above 1 % of its maximum, else CT above -900 HU on the TOPAS grid, else the whole grid."""
    if ref is not None: return ref > 0.01 * ref.max(), "TPS-Dosis ueber 1 % ihres Maximums"
    try:
        hu, _, _ = synthetic.read_ct(run["ct"])
        if hu.shape == dose.shape: return hu > -900, "CT ueber -900 HU"
    except (OSError, ValueError):
        pass
    return np.ones(dose.shape, bool), "ganzes Gitter (CT nicht auf dem Dosisgitter)"


def kennzahlen(res, run):
    """The figures of an evaluation in the robin-only format of the hub (claude-hub docs/research-studies.md): a flat
    object of numbers and texts of at most 200 characters. No plan label, no file name and no path: the label can be
    the patient's name, and the paths name the export."""
    k = {"fields": len(run["fields"]), "fractions": run["fractions"], "histories_per_field": run["histories"],
         "uncertainty_rms_pct": 100 * res["uncertainty_rms"], "topas_max_gy": res["topas_max_gy"],
         "body_mask": res["body_mask"][:200]}
    if "comparison" in res:
        d = res["diagnostics"]
        k.update(rbe=res["rbe"], tps_dose_type=res["tps_dose_type"] or "fehlt", high_dose_mean_ratio=d["high_dose_mean_ratio"])
        for crit, v in res["comparison"]["gamma"].items(): k[f"gamma_{crit}_pass_pct"] = v["pass_pct"]
        for crit, v in res.get("gamma_normalised_to_tps_mean", {}).items(): k[f"gamma_{crit}_normalised_pass_pct"] = v["pass_pct"]
        for q, v in zip(("q05", "q50", "q95"), d["ratio_q05_q50_q95"]): k[f"ratio_{q}"] = v
        for ax, v in zip("xyz", d["centre_of_mass_shift_mm"] or ()): k[f"centre_of_mass_shift_{ax}_mm"] = v
        for ax, v in zip("xyz", d["rigid"]["shift_mm"]): k[f"rigid_shift_{ax}_mm"] = v
        k["rigid_rms_rel_best_pct"] = 100 * d["rigid"]["rms_rel_best"]
        k["rigid_rms_rel_unshifted_pct"] = 100 * d["rigid"]["rms_rel_unshifted"]
    return {n: v for n, v in k.items() if isinstance(v, str) or (v is not None and np.isfinite(v))}


def write_kennzahlen(ws, name, figures):
    """WS/robin/plan_mc/NAME/kennzahlen.json, only in a workspace with zones (robin/ exists): the hub shows it to Robin."""
    robin = os.path.join(os.path.abspath(ws), "robin")
    if not os.path.isdir(robin): return None
    d = os.path.join(robin, "plan_mc", name); os.makedirs(d, exist_ok=True)
    json.dump(figures, open(os.path.join(d, "kennzahlen.json"), "w"), indent=1)
    return os.path.join("robin", "plan_mc", name, "kennzahlen.json")


def evaluate_run(a):
    out, run, _ = _files(a)
    model = beam_model.BeamModel.load(os.path.join(out, "beam_model.json"))
    dose, sd = summed_dose(out, run, model.dose_scale)
    ref = dose_type = None
    if run["rtdose"]:
        import pydicom
        ref, origin, spacing = synthetic.read_rtdose(run["rtdose"])
        hdr = pydicom.dcmread(run["rtdose"], stop_before_pixels=True)
        kind = str(getattr(hdr, "DoseSummationType", "")).upper()
        dose_type = str(getattr(hdr, "DoseType", "") or "").upper()
        if kind not in ("PLAN", "FRACTION"):
            raise SystemExit(f"RT Dose with DoseSummationType {kind or 'missing'}: give the dose of the whole plan")
        if kind == "PLAN": ref = ref / max(1, run["fractions"])
        if ref.shape != dose.shape: raise SystemExit(f"TPS dose grid {ref.shape} differs from the TOPAS grid {dose.shape}")
    body, body_name = body_mask(run, dose, ref)
    dmax = float(dose[body].max())
    hi = body & (dose >= 0.5 * dmax)
    unc = float(np.sqrt(np.mean((sd[hi] / dose[hi]) ** 2)))
    label = run.get("plan_label") or "Plan"
    lines = ["# TOPAS-Nachrechnung (nur Forschung, keine klinische Entscheidung)", "",
             f"Plan: {label}, {len(run['fields'])} Felder, {run['fractions']} Fraktionen, {run['histories']} Historien je Feld"
             + (", Spotgroessen aus dem Plan." if run.get("spots_from_plan") else "."),
             f"Koerpermaske: {body_name}.",
             f"Statistische Unsicherheit (1 sigma, rms ueber Voxel der Maske > 50 % Maximum): {100 * unc:.2f} %.",
             f"Maximale physikalische Dosis TOPAS je Fraktion in der Maske: {dmax:.3f} Gy.", ""]
    res = {"uncertainty_rms": unc, "topas_max_gy": dmax, "body_mask": body_name}
    if ref is not None:
        rbe = a.rbe if a.rbe is not None else (1.1 if dose_type == "EFFECTIVE" else 1.0)
        if a.rbe is None and dose_type not in ("EFFECTIVE", "PHYSICAL"):
            warn(f"RT Dose with DoseType {dose_type or 'missing'}: compared as physical dose (RBE 1.0), --rbe sets it")
        synthetic.write_rtdose(os.path.join(out, "topas_dose.dcm"), dose, origin, spacing, summation="FRACTION")
        ev = rbe * dose
        cmp_ = evaluate.compare(ref, ev, origin, spacing)
        diag = evaluate.diagnostics(ref, ev, origin, spacing, body=body)
        res.update(rbe=rbe, tps_dose_type=dose_type or None, comparison=cmp_, diagnostics=diag)
        ratio = diag["high_dose_mean_ratio"]
        q, com, rig = diag["ratio_q05_q50_q95"], diag["centre_of_mass_shift_mm"], diag["rigid"]
        lines += [f"TPS-Dosis: DoseType {dose_type or 'fehlt'}; verglichen mit RBE {rbe:g} x TOPAS (physikalisch).", "",
                  "| Kriterium | Bestehensrate |", "|---|---|"]
        lines += [f"| Gamma {k} global, Schwelle 10 % | {v['pass_pct']:.1f} % |" for k, v in cmp_["gamma"].items()]
        if a.normalised_gamma:
            norm = evaluate.compare(ref, ev / ratio, origin, spacing)["gamma"]
            res["gamma_normalised_to_tps_mean"] = norm
            lines += [f"| Gamma {k}, TOPAS auf den TPS-Mittelwert (> 50 %) normiert | {v['pass_pct']:.1f} % |"
                      for k, v in norm.items()]
        lines += ["", f"Maximum TPS {cmp_['ref_max']:.3f} Gy, RBE x TOPAS {cmp_['eval_max']:.3f} Gy je Fraktion.", "",
                  "Diagnose im Hochdosisbereich (TPS > 50 % ihres Maximums):", "",
                  f"- Mittelwert physikalisch, TOPAS / (TPS / RBE): {ratio:.3f}; "
                  f"TOPAS / TPS wie gespeichert ({dose_type or 'DoseType fehlt'}): {ratio / rbe:.3f}.",
                  f"- Quantile 5/50/95 % des Verhaeltnisses je Voxel: {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}.",
                  "- Schwerpunkt TOPAS minus TPS (x/y/z, je Dosis ueber 10 % ihres Maximums, nur in der Koerpermaske): "
                  + (" / ".join(f"{v:+.1f}" for v in com) + " mm." if com else "nicht bestimmbar."),
                  f"- Beste starre Verschiebung (+-{rig['max_shift_voxels']} Voxel): "
                  + " / ".join(f"{v:+.1f}" for v in rig["shift_mm"])
                  + f" mm, RMS {100 * rig['rms_rel_best']:.1f} % (ohne Verschiebung {100 * rig['rms_rel_unshifted']:.1f} %)."]
    else:
        np.save(os.path.join(out, "topas_dose.npy"), dose.astype(np.float32))
        lines.append("Keine TPS-Dosis angegeben: nur die TOPAS-Dosis (topas_dose.npy, Gitter des CT).")
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    open(os.path.join(out, "BERICHT.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    rel = write_kennzahlen(a.workspace, a.name, kennzahlen(res, run))
    if rel: print(f"\nKennzahlen fuer den Hub (nur Robin): {rel}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("find", "write", "queue", "status", "evaluate"))
    ap.add_argument("export", nargs="?", help="find: folder of the DICOM export")
    ap.add_argument("--hlut-dir", help="find: folder with the clinic's .hlut")
    ap.add_argument("--workspace"); ap.add_argument("--name")
    ap.add_argument("--ct"); ap.add_argument("--rtplan"); ap.add_argument("--rtdose"); ap.add_argument("--hlut")
    ap.add_argument("--template"); ap.add_argument("--beam-model")
    ap.add_argument("--histories", type=int, default=10_000_000); ap.add_argument("--i-water", type=float, default=78.0)
    ap.add_argument("--rs-material"); ap.add_argument("--rs-thickness", type=float)
    ap.add_argument("--rs-iso-distance", type=float, help="isocenter to range shifter distance [mm] when the plan has none")
    ap.add_argument("--spots-from-plan", action="store_true", help="sigma per energy from the plan's ScanningSpotSize")
    ap.add_argument("--ripple-filter"); ap.add_argument("--rifi-distance", type=float)
    ap.add_argument("--rbe", type=float, help="evaluate: RBE for the TPS dose (default 1.1 for EFFECTIVE, else 1.0)")
    ap.add_argument("--normalised-gamma", action="store_true", help="evaluate: also gamma with TOPAS scaled to the TPS mean")
    a = ap.parse_args(argv)
    if a.cmd == "find":
        if not a.export: ap.error("find needs the export folder")
    elif not (a.workspace and a.name): ap.error(f"{a.cmd} needs --workspace and --name")
    if a.cmd == "write":
        miss = [k for k in ("ct", "rtplan", "hlut", "template", "beam_model") if not getattr(a, k)]
        if miss: ap.error("write needs " + ", ".join("--" + k.replace("_", "-") for k in miss))
    if a.cmd in ("find", "write", "evaluate"): dicom_compat.check()
    {"find": find, "write": write, "queue": queue, "status": status, "evaluate": evaluate_run}[a.cmd](a)


if __name__ == "__main__":
    main()
