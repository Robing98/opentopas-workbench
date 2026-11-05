"""TOPAS parameter files for a plan: one file per field, every spot as one sequential time step.

Geometry follows the PatientInIEC examples of TOPAS: World is the IEC fixed system, IEC_G (gantry) and IEC_S (patient
support) are groups at the isocenter, DICOM_to_IEC turns the DICOM patient system into IEC, and the patient sits at
-isocenter inside it. The beam line (source, range shifter, ripple filter) hangs in IEC_G along its +Z axis.

Rotation signs: TOPAS builds R = Rz*Ry*Rx from RotX/RotY/RotZ and hands it to G4PVPlacement, which turns the daughter
by the inverse. A component therefore maps local to parent coordinates with Rx(-RotX)*Ry(-RotY)*Rz(-RotZ). From that:
gantry angle g (source moves from +Z towards +X for HFS) needs RotY = -g, couch angle c (counterclockwise from above)
needs RotZ = -c, HFS needs DICOM_to_IEC RotX = +90 deg. tools/plan_mc/checks/alignment.py checks this in TOPAS.

Histories: each field gets `histories` primaries, spread over the spots in proportion to their particle numbers with
unbiased random rounding, so every simulated proton carries the same weight: dose = TOPAS dose * particles / histories.
"""
import importlib.util, json, math, os, random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_rf_spec = importlib.util.spec_from_file_location("ripple_filter", os.path.join(ROOT, "opentopas_workbench", "generators", "ripple_filter.py"))
ripple_filter = importlib.util.module_from_spec(_rf_spec); _rf_spec.loader.exec_module(ripple_filter)

# DICOM PatientPosition -> RotX, RotY, RotZ of DICOM_to_IEC (TOPAS convention, see the module text)
PATIENT_ROT = {"HFS": (90, 0, 0), "FFS": (90, 180, 0), "HFP": (90, 0, 180), "FFP": (-90, 0, 0)}


def g(v): return f"{v:.6g}"


def spot_sequence(field, model, histories, seed=1):
    """List of spots (energy, x_iso, y_iso, particles, histories) in delivery order, and the total particle number."""
    spots = []
    for l in field.layers:
        ppmu = model.at(l.energy)["particles_per_mu"] if field.unit in ("MU", "relative") else 1.0
        for x, y, w in zip(l.x, l.y, l.w):
            spots.append([l.energy, x, y, w * ppmu, 0])
    total = sum(s[3] for s in spots)
    if total <= 0: raise ValueError(f"field {field.name}: no particles")
    rng = random.Random(seed)
    for s in spots:
        exact = histories * s[3] / total
        s[4] = int(exact) + (1 if rng.random() < exact - int(exact) else 0)
    return spots, total


def tf(name, values, kind, unit=""):
    n = len(values)
    fmt = (lambda v: str(int(v))) if kind == "iv" else g
    return (f's:Tf/{name}/Function = "Step"\n'
            f"dv:Tf/{name}/Times = Tf/Spot/Times ms\n"
            f"{kind}:Tf/{name}/Values = {n} {' '.join(fmt(v) for v in values)}{(' ' + unit) if unit else ''}\n")


def beamline_block(line):
    """Range shifter and ripple filter in IEC_G. Distances are from the isocenter to the downstream face [mm].

    line = {"range_shifter": {"material", "thickness_mm", "iso_distance_mm", "half_size_mm"} or None,
            "ripple_filter": {"variant": "t3" | "t4" | ..., "iso_distance_mm", "period_mm"} or None}
    """
    out = []
    rs = line.get("range_shifter")
    if rs:
        t, d = float(rs["thickness_mm"]), float(rs["iso_distance_mm"])
        out.append(f'''
# Range shifter: {g(t)} mm {rs["material"]}, downstream face {g(d)} mm from the isocenter
s:Ge/RangeShifter/Type     = "TsBox"
s:Ge/RangeShifter/Parent   = "IEC_G"
s:Ge/RangeShifter/Material = "{rs["material"]}"
d:Ge/RangeShifter/HLX      = {g(rs.get("half_size_mm", 150.0))} mm
d:Ge/RangeShifter/HLY      = {g(rs.get("half_size_mm", 150.0))} mm
d:Ge/RangeShifter/HLZ      = {g(t / 2)} mm
d:Ge/RangeShifter/TransZ   = {g(d + t / 2)} mm
''')
    rf = line.get("ripple_filter")
    if rf:
        lam = float(rf.get("period_mm", 1.5))
        text = ripple_filter.variant_block(rf["variant"], lam)
        if "TsCAD" in text: raise ValueError("2D pin filters (STL) are not supported in the plan chain yet")
        d = float(rf["iso_distance_mm"])
        # the generator places its filter in World 80 cm upstream; here it hangs in IEC_G at the requested distance,
        # wide enough for a full field (periods over 200 mm)
        n_old = int(120 / lam); n_new = int(200 / lam)
        text = text.replace('"World"', '"IEC_G"').replace("= 12.0 cm", "= 20.0 cm")
        text = text.replace(f"d:Ge/RiFi1/TransZ   = 80.0 cm", f"d:Ge/RiFi1/TransZ   = {g(d)} mm")
        disp = " ".join(f"{(i - n_new // 2) * lam:.2f}" for i in range(n_new + 1))
        old = " ".join(f"{(i - n_old // 2) * lam:.2f}" for i in range(n_old + 1))
        text = text.replace(f"{n_old + 1} {old} mm", f"{n_new + 1} {disp} mm")
        if "RiFi2" in text: raise ValueError("crossed filters are not supported in the plan chain yet")
        out.append(f"\n# Ripple filter {rf['variant']}, period {g(lam)} mm, {g(d)} mm from the isocenter" + text)
    return "".join(out)


def field_file(plan, field, model, line, patient_include, histories, out_name=None, seed=1, threads=0):
    """Text of the TOPAS parameter file for one field. patient_include: absolute path of the patient/scorer file."""
    spots, total = spot_sequence(field, model, histories, seed)
    n = len(spots)
    out_name = out_name or f"field{field.number}"
    zs = model.source_distance
    rx, ry, rz = PATIENT_ROT.get(plan.patient_position.upper(), (None, None, None))
    if rx is None: raise ValueError(f"patient position {plan.patient_position} not supported")
    pars = [model.at(s[0]) for s in spots]
    px, py, ax, ay = [], [], [], []
    for s in spots:
        xs, ys, tx, ty = model.spot_at_source(s[1], s[2])
        # Nozzle has RotX = 180 deg: its local y is -y of IEC_G. The spot direction in nozzle coordinates must be
        # (tan tx, -tan ty, 1); with the TOPAS convention that needs RotX = -ty and RotY = -atan(tan tx * cos ty).
        px.append(xs); py.append(-ys)
        ax.append(-ty); ay.append(-math.atan(math.tan(tx) * math.cos(ty)))
    meta = {"field": field.name, "number": field.number, "gantry": field.gantry, "couch": field.couch, "iso": list(field.iso),
            "spots": n, "particles": total, "histories": sum(s[4] for s in spots), "unit": field.unit,
            "beam_model": model.name, "dose_scale": model.dose_scale}
    t = f"""# Field {field.name} (beam {field.number}) of plan {plan.label}: {n} spots, {len(field.layers)} layers.
# Gantry {g(field.gantry)} deg, couch {g(field.couch)} deg, isocenter {', '.join(g(v) for v in field.iso)} mm (DICOM), {plan.patient_position}.
# Beam model: {model.name}. Written by tools/plan_mc (spot by spot, research use only, not for clinical decisions).
# plan_mc: {json.dumps(meta)}
includeFile = {patient_include}

i:Ts/NumberOfThreads = {threads}
i:Ts/Seed = {seed}
b:Ts/PauseBeforeQuit = "False"
i:Ts/ShowHistoryCountAtInterval = {max(10000, histories // 20)}
s:Ge/World/Material = "G4_AIR"
d:Ge/World/HLX = 1.6 m
d:Ge/World/HLY = 1.6 m
d:Ge/World/HLZ = 1.6 m
b:Ge/World/Invisible = "True"

s:Ge/IEC_G/Type   = "Group"
s:Ge/IEC_G/Parent = "World"
d:Ge/IEC_G/RotY   = {g(-field.gantry)} deg
s:Ge/IEC_S/Type   = "Group"
s:Ge/IEC_S/Parent = "World"
d:Ge/IEC_S/RotZ   = {g(-field.couch)} deg
s:Ge/DICOM_to_IEC/Type   = "Group"
s:Ge/DICOM_to_IEC/Parent = "IEC_S"
d:Ge/DICOM_to_IEC/RotX   = {g(rx)} deg
d:Ge/DICOM_to_IEC/RotY   = {g(ry)} deg
d:Ge/DICOM_to_IEC/RotZ   = {g(rz)} deg
s:Ge/PatientIso/Type   = "Group"
s:Ge/PatientIso/Parent = "DICOM_to_IEC"
d:Ge/PatientIso/TransX = {g(-field.iso[0])} mm
d:Ge/PatientIso/TransY = {g(-field.iso[1])} mm
d:Ge/PatientIso/TransZ = {g(-field.iso[2])} mm

s:Ge/Nozzle/Type   = "Group"
s:Ge/Nozzle/Parent = "IEC_G"
d:Ge/Nozzle/TransZ = {g(zs)} mm
d:Ge/Nozzle/RotX   = 180. deg
s:Ge/Spot/Type   = "Group"
s:Ge/Spot/Parent = "Nozzle"
d:Ge/Spot/TransX = Tf/PosX/Value mm
d:Ge/Spot/TransY = Tf/PosY/Value mm
d:Ge/Spot/RotX   = Tf/AngX/Value rad
d:Ge/Spot/RotY   = Tf/AngY/Value rad
{beamline_block(line)}
d:Tf/TimelineStart = 0. ms
d:Tf/TimelineEnd = {n} ms
i:Tf/NumberOfSequentialTimes = {n}
dv:Tf/Spot/Times = {n} {' '.join(str(i + 1) for i in range(n))} ms
{tf("Energy", [p["mean"] for p in pars], "dv", "MeV")}{tf("Spread", [p["spread_pct"] for p in pars], "uv")}{tf("Histories", [s[4] for s in spots], "iv")}{tf("PosX", px, "dv", "mm")}{tf("PosY", py, "dv", "mm")}{tf("AngX", ax, "dv", "rad")}{tf("AngY", ay, "dv", "rad")}{tf("SigmaX", [p["sigma_x"] for p in pars], "dv", "mm")}{tf("SigmaY", [p["sigma_y"] for p in pars], "dv", "mm")}{tf("SigmaXprime", [p["div_x"] / 1000 for p in pars], "uv")}{tf("SigmaYprime", [p["div_y"] / 1000 for p in pars], "uv")}{tf("CorrX", [p["corr_x"] for p in pars], "uv")}{tf("CorrY", [p["corr_y"] for p in pars], "uv")}
s:So/Field/Type = "Emittance"
s:So/Field/Distribution = "BiGaussian"
s:So/Field/Component = "Spot"
s:So/Field/EmittanceParticle = "{model.particle}"
d:So/Field/EmittanceEnergy = Tf/Energy/Value MeV
u:So/Field/EmittanceEnergySpread = Tf/Spread/Value
i:So/Field/NumberOfHistoriesInRun = Tf/Histories/Value
d:So/Field/SigmaX = Tf/SigmaX/Value mm
d:So/Field/SigmaY = Tf/SigmaY/Value mm
u:So/Field/SigmaXprime = Tf/SigmaXprime/Value
u:So/Field/SigmaYprime = Tf/SigmaYprime/Value
u:So/Field/CorrelationX = Tf/CorrX/Value
u:So/Field/CorrelationY = Tf/CorrY/Value

# Scorers write {out_name}_*.bin; evaluate.py scales them with the particle number above.
s:Sc/Dose/OutputFile = "{out_name}_dose"
s:Sc/DoseMed/OutputFile = "{out_name}_dosemed"
s:Sc/LET/OutputFile = "{out_name}_let"
"""
    return t, meta


def patient_block(kind, scoring_component, schneider=None, dicom_dir=None, box=None, let=True, rtdose=None):
    """Patient and scorers, shared by the fields of one plan.

    kind "ct": TsDicomPatient from dicom_dir with the Schneider converter file `schneider`, placed so that DICOM
    coordinates hold inside PatientIso. kind "box": water box {"center": (x, y, z) DICOM mm, "half": (hx, hy, hz) mm,
    "bins": (nx, ny, nz)}. scoring_component: "Patient/RTDoseGrid" to score on a cloned RTDOSE grid, else "Patient".
    """
    if kind == "ct":
        geo = f'''includeFile = {schneider}
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/Parent = "PatientIso"
s:Ge/Patient/DicomDirectory = "{dicom_dir}"
# DicomOrigin is where the component centre sits in DICOM coordinates, so this places DICOM (0, 0, 0) at PatientIso.
# TOPAS overwrites these placeholders when it reads the CT; without them the Trans lines name an unknown parameter.
dc:Ge/Patient/DicomOriginX = 0.0 mm
dc:Ge/Patient/DicomOriginY = 0.0 mm
dc:Ge/Patient/DicomOriginZ = 0.0 mm
d:Ge/Patient/TransX = 1.0 * Ge/Patient/DicomOriginX mm
d:Ge/Patient/TransY = 1.0 * Ge/Patient/DicomOriginY mm
d:Ge/Patient/TransZ = 1.0 * Ge/Patient/DicomOriginZ mm
''' + (f's:Ge/Patient/CloneRTDoseGridFrom = "{rtdose}"\n' if rtdose else "")
    elif kind == "box":
        c, h, b = box["center"], box["half"], box["bins"]
        geo = f'''s:Ge/Patient/Type = "TsBox"
s:Ge/Patient/Parent = "PatientIso"
s:Ge/Patient/Material = "{box.get("material", "G4_WATER")}"
d:Ge/Patient/TransX = {g(c[0])} mm
d:Ge/Patient/TransY = {g(c[1])} mm
d:Ge/Patient/TransZ = {g(c[2])} mm
d:Ge/Patient/HLX = {g(h[0])} mm
d:Ge/Patient/HLY = {g(h[1])} mm
d:Ge/Patient/HLZ = {g(h[2])} mm
i:Ge/Patient/XBins = {b[0]}
i:Ge/Patient/YBins = {b[1]}
i:Ge/Patient/ZBins = {b[2]}
'''
    else:
        raise ValueError(kind)
    sc = ""
    for name, q in (("Dose", "DoseToWater"), ("DoseMed", "DoseToMedium")) + ((("LET", "ProtonLET"),) if let else ()):
        report = '2 "Sum" "Standard_Deviation"' if name != "LET" else '1 "Mean"'
        sc += f'''
s:Sc/{name}/Quantity = "{q}"
s:Sc/{name}/Component = "{scoring_component}"
s:Sc/{name}/OutputType = "binary"
s:Sc/{name}/IfOutputFileAlreadyExists = "Overwrite"
sv:Sc/{name}/Report = {report}
b:Sc/{name}/OutputToConsole = "False"
'''
    return geo + sc
