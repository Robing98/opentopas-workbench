"""Generator: horizontal HIT nozzle (vacuum window, beam monitors, ripple filter slide) in front of a water phantom.

Built from public values only. Order of the elements after Parodi et al. 2012 (PMB 57 3759, fig. 2a and 2b): vacuum
window, MWPC 1, IC 1, IC 2, IC 3, MWPC 2, air, ripple filter, isocentre. Water equivalent thickness per element:
window 360 um, each ionisation chamber (IC) 230 um, each multi-wire proportional chamber (MWPC) 160 um, 1.37 mm in all.
The end of the monitors (downstream face of MWPC 2) lies about 112 cm before the isocentre (Tessonnier et al. 2015,
Front. Oncol. 5:297). The positions of the other elements are not public: they are a field, the default is an assumption.

Models of the monitors:
  m1    every element a water layer of its water equivalent thickness (Parodi, method 1)
  m2    as m1, but each MWPC is 3 um tungsten on 130 um water (Parodi, method 2)
  wire  as m2, but the tungsten of each MWPC is single wires: two signal planes (radius 25 um, pitch 1 mm, crossed) and
        high voltage planes (radius 10 um, pitch 2 mm, 45 deg to the signal wires)
  none  no element, the same air and the same source: the reference for range shift and lateral spread

Outputs per case: IDD_<case>.csv (depth dose, 0.1 mm bins, water phantom at the isocentre) and LAT_<case>.csv (fluence of
the primaries along x, integrated over y, on an air plane just upstream of the phantom surface). Optionally a TOPAS phase
space at the end of the monitors, readable by a source of type PhaseSpace.
"""
import importlib.util, math, os, re


def _load(name):
    """A sibling module, loaded by path like the generators themselves (the loader has no package to import from)."""
    spec = importlib.util.spec_from_file_location("wbgen_" + name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


MON = _load("_monitors")
RF = _load("ripple_filter")

NAME = {"en": "HIT nozzle", "de": "HIT-Nozzle"}
DESCRIPTION = {
    "en": "Vacuum window, ionisation chambers and MWPCs of the horizontal HIT nozzle (public values), ripple filter at any distance, water phantom.",
    "de": "Vakuumfenster, Ionisationskammern und MWPCs der horizontalen HIT-Nozzle (öffentliche Werte), Ripple-Filter in beliebigem Abstand, Wasserphantom.",
}
ION = {"p": ("proton", 1, 1), "c": ("GenericIon(6,12)", 12, 6)}
ELEMENTS = ["Window", "MWPC1", "IC1", "IC2", "IC3", "MWPC2"]
WET_UM = {"Window": 360.0, "IC": 230.0, "MWPC": 160.0}      # water equivalent per element, Parodi 2012
M2_W_UM, M2_WATER_UM = 3.0, 130.0                          # Parodi method 2, per MWPC
SIGNAL_R, SIGNAL_PITCH = 0.025, 1.0                        # mm, tungsten signal wires
HV_R, HV_PITCH = 0.010, 2.0                                # mm, tungsten high voltage wires
WIRE_GAP = 2.0                                             # mm between the wire planes of one MWPC (assumption)
# HIT energy library (assumption, not taken from the two papers): the momentum spread falls from its value at the lowest
# energy to its value at the highest one exponentially in energy.
HIT_E = {"p": (48.0, 221.0), "c": (88.0, 430.0)}           # MeV/u
HIT_DPP = {"p": (0.30, 0.05), "c": (0.30, 0.10)}           # FWHM of dp/p in %, lowest and highest energy (Parodi 2012)
MODELS = {"none": {"en": "no nozzle", "de": "ohne Nozzle"},
          "m1": {"en": "water layers (Parodi method 1)", "de": "Wasserschichten (Parodi Methode 1)"},
          "m2": {"en": "MWPC as 3 um tungsten + water (Parodi method 2)", "de": "MWPC als 3 µm Wolfram + Wasser (Parodi Methode 2)"},
          "wire": {"en": "MWPC with single tungsten wires", "de": "MWPC mit einzelnen Wolframdrähten"}}
FIELDS = [
    {"key": "ions", "type": "multi", "label": {"en": "Ions", "de": "Ionen"}, "default": ["p"],
     "options": [["p", {"en": "Proton", "de": "Proton"}], ["c", {"en": "Carbon-12", "de": "Kohlenstoff-12"}]]},
    {"key": "energies", "type": "text", "label": {"en": "Energies [MeV/u], comma separated", "de": "Energien [MeV/u], durch Komma getrennt"},
     "default": "63, 132"},
    {"key": "models", "type": "multi", "label": {"en": "Monitor models", "de": "Monitormodelle"}, "default": ["m1"],
     "options": [[k, v] for k, v in MODELS.items()]},
    {"key": "positions", "type": "text", "default": "132, 128, 124, 120, 116, 112",
     "label": {"en": "Downstream face of window, MWPC 1, IC 1, IC 2, IC 3, MWPC 2 [cm before the isocentre]",
               "de": "Strahlabwärts gelegene Seite von Fenster, MWPC 1, IC 1, IC 2, IC 3, MWPC 2 [cm vor dem Isozentrum]"}},
    {"key": "variants", "type": "multi", "label": {"en": "Filter variants", "de": "Filtervarianten"}, "default": ["rifi0"],
     "options": [[k, v] for k, v in RF.VARIANTS.items()]},
    {"key": "filter_pos", "type": "text", "default": "108",
     "label": {"en": "Filter at [cm before the isocentre], comma separated", "de": "Filter bei [cm vor dem Isozentrum], durch Komma getrennt"}},
    {"key": "filter_mat", "type": "select", "label": {"en": "Filter material", "de": "Filtermaterial"}, "default": "PMMA_HIT",
     "options": [["PMMA_HIT", {"en": "PMMA 1.20 g/cm3, I = 74 eV (Parodi)", "de": "PMMA 1,20 g/cm³, I = 74 eV (Parodi)"}],
                 ["G4_PLEXIGLASS", "G4_PLEXIGLASS (1.19 g/cm3)"]]},
    {"key": "period", "type": "number", "label": {"en": "Period of 1D/2D filters [mm]", "de": "Periode der 1D/2D-Filter [mm]"}, "default": 1.5},
    {"key": "wire_half", "type": "number", "default": 40,
     "label": {"en": "Wire model: radius of the wired area [mm]", "de": "Drahtmodell: Radius der bespannten Fläche [mm]"}},
    {"key": "hv_planes", "type": "number", "default": 2,
     "label": {"en": "Wire model: high voltage planes per MWPC (mean tungsten 3.9 um with 0, 4.2 um with 2; method 2 has 3 um)",
               "de": "Drahtmodell: Hochspannungsebenen je MWPC (mittleres Wolfram 3,9 µm bei 0, 4,2 µm bei 2; Methode 2 hat 3 µm)"}},
    {"key": "histories", "type": "number", "label": {"en": "Histories", "de": "Historien"}, "default": 100000},
    {"key": "spread", "type": "text", "default": "hit",
     "label": {"en": "Energy spread [%, sigma], or \"hit\" for the HIT momentum spread", "de": "Energiebreite [%, Sigma], oder \"hit\" für die HIT-Impulsbreite"}},
    {"key": "sigma", "type": "number", "label": {"en": "Spot sigma at the window [mm]", "de": "Spot-Sigma am Fenster [mm]"}, "default": 3.0},
    {"key": "divergence", "type": "number", "label": {"en": "Angular spread [mrad]", "de": "Winkelbreite [mrad]"}, "default": 0.0},
    {"key": "phsp", "type": "select", "label": {"en": "Phase space at the end of the monitors", "de": "Phasenraum am Ende der Monitore"}, "default": "off",
     "options": [["off", {"en": "off", "de": "aus"}], ["write", {"en": "write", "de": "schreiben"}],
                 ["only", {"en": "write and stop the particles there", "de": "schreiben und Teilchen dort beenden"}]]},
    {"key": "prefix", "type": "text", "label": {"en": "File name prefix", "de": "Präfix Dateiname"}, "default": "noz"},
] + MON.FIELDS
WORLD_HXY, WORLD_HZ = 300.0, 1450.0                         # mm
DEPTH_BIN = 0.1                                             # mm
LAT_HALF, LAT_BIN = 50.0, 0.2                               # mm, lateral plane at the isocentre
RF_REF_CM = 80.0                                            # ripple_filter.py places its first plate here
TRANSZ_RE = re.compile(r"^(d:Ge/(?:RiFi\d+|Pins)/TransZ\s*=\s*)([-\d.]+)( cm)$", re.M)


def numbers(text, what):
    out = [float(t) for t in str(text or "").replace(";", ",").split(",") if t.strip()]
    if not out: raise ValueError(f"no {what} given")
    return out


def element_positions(text):
    pos = numbers(text, "element positions")
    if len(pos) != len(ELEMENTS): raise ValueError(f"give {len(ELEMENTS)} positions: {', '.join(ELEMENTS)}")
    # 1 cm between neighbours: an MWPC of the wire model is about 8 mm deep (4 wire planes 2 mm apart and its water)
    if any(a - b < 1.0 for a, b in zip(pos, pos[1:])): raise ValueError("element positions must fall from the window to MWPC 2, at least 1 cm apart")
    if pos[-1] <= 1 or pos[0] * 10 + 30 > WORLD_HZ: raise ValueError("element positions must lie between 1 and 142 cm")
    return pos


def momentum_fwhm(ion, e_u):
    """FWHM of dp/p in % at e_u MeV/u, falling exponentially from the lowest to the highest HIT energy."""
    (e0, e1), (f0, f1) = HIT_E[ion], HIT_DPP[ion]
    t = min(1.0, max(0.0, (e_u - e0) / (e1 - e0)))
    return f0 * (f1 / f0) ** t


def energy_sigma(ion, e_u):
    """Gaussian sigma of the kinetic energy in %, from the momentum FWHM: dE/E = (gamma + 1) / gamma * dp/p."""
    gamma = 1 + e_u / 931.494
    return momentum_fwhm(ion, e_u) / (2 * math.sqrt(2 * math.log(2))) * (gamma + 1) / gamma


def spread_for(v, ion, e_u):
    s = str(v.get("spread", "hit")).strip().lower()
    return energy_sigma(ion, e_u) if s in ("", "hit") else float(s.replace(",", "."))


def slab(name, mat, thick_um, face_cm, half_xy=60.0):
    """A layer whose downstream face lies at face_cm before the isocentre (the beam runs towards -z)."""
    hz = thick_um / 2000.0
    return f'''s:Ge/{name}/Type     = "TsBox"
s:Ge/{name}/Parent   = "World"
s:Ge/{name}/Material = "{mat}"
d:Ge/{name}/HLX      = {half_xy:g} mm
d:Ge/{name}/HLY      = {half_xy:g} mm
d:Ge/{name}/HLZ      = {hz:.6g} mm
d:Ge/{name}/TransZ   = {face_cm * 10 + hz:.6f} mm
'''


def wire_offsets(pitch, radius):
    n = int(math.floor(radius / pitch))
    return [i * pitch for i in range(-n, n + 1)]


def wire_plane(name, face_cm, angle, r_wire, pitch, radius):
    """Parallel tungsten wires inside a disc of the given radius, in a group turned by angle about the beam axis.

    The wires lie along y before the turn; the group turns them as one body, so offset and direction stay
    perpendicular whatever sense TOPAS gives a rotation."""
    z = face_cm * 10 + r_wire
    t = [f'''s:Ge/{name}/Type   = "Group"
s:Ge/{name}/Parent = "World"
d:Ge/{name}/RotZ   = {angle:g} deg
''']
    for i, d in enumerate(wire_offsets(pitch, radius)):
        hl = math.sqrt(radius ** 2 - d ** 2)
        if hl < 0.5: continue
        w = f"{name}W{i}"
        t.append(f'''s:Ge/{w}/Type     = "TsCylinder"
s:Ge/{w}/Parent   = "{name}"
s:Ge/{w}/Material = "G4_W"
d:Ge/{w}/RMax     = {r_wire:g} mm
d:Ge/{w}/HL       = {hl:.4f} mm
d:Ge/{w}/TransX   = {d:g} mm
d:Ge/{w}/TransZ   = {z:.4f} mm
d:Ge/{w}/RotX     = 90 deg
''')
    return "".join(t)


def mwpc_planes(hv):
    """(angle, radius, pitch) of the wire planes of one MWPC, downstream first: signal x, HV, signal y, HV."""
    if hv not in (0, 1, 2): raise ValueError("high voltage planes per MWPC: 0, 1 or 2")
    sig = [(0, SIGNAL_R, SIGNAL_PITCH), (90, SIGNAL_R, SIGNAL_PITCH)]
    hvs = [(45, HV_R, HV_PITCH), (-45, HV_R, HV_PITCH)][:hv]
    out = [sig[0]] + hvs[:1] + [sig[1]] + hvs[1:]
    return out


def mean_wire_um(hv):
    """Mean tungsten thickness of the wires of one MWPC [um]: pi r^2 / pitch per plane."""
    return sum(math.pi * r * r / p for _, r, p in mwpc_planes(hv)) * 1000


def nozzle_block(model, pos, wire_radius, hv):
    if model == "none": return ""
    t = [f"\n# Nozzle, model {model}. Downstream faces [cm before the isocentre]: "
         + ", ".join(f"{e} {p:g}" for e, p in zip(ELEMENTS, pos)) + "\n"]
    for e, p in zip(ELEMENTS, pos):
        kind = e.rstrip("0123456789")
        if kind != "MWPC" or model == "m1":
            t.append(slab(e, "G4_WATER", WET_UM[kind], p))
        elif model == "m2":
            t.append(slab(e + "W", "G4_W", M2_W_UM, p))
            t.append(slab(e, "G4_WATER", M2_WATER_UM, p + M2_W_UM / 1e4))
        elif model == "wire":
            planes = mwpc_planes(hv)
            for k, (ang, r, pitch) in enumerate(planes):
                t.append(wire_plane(f"{e}P{k}", p + k * WIRE_GAP / 10, ang, r, pitch, wire_radius))
            t.append(slab(e, "G4_WATER", M2_WATER_UM, p + len(planes) * WIRE_GAP / 10))
        else:
            raise ValueError(f"unknown model {model}")
    return "".join(t)


PMMA_HIT = '''
# PMMA of the ripple filter after Parodi 2012: 1.20 g/cm3 (covers a Kapton foil), I = 74 eV
sv:Ma/PMMA_HIT/Components = 3 "Hydrogen" "Carbon" "Oxygen"
uv:Ma/PMMA_HIT/Fractions = 3 0.080538 0.599848 0.319614
d:Ma/PMMA_HIT/Density = 1.20 g/cm3
d:Ma/PMMA_HIT/MeanExcitationEnergy = 74.0 eV
s:Ma/PMMA_HIT/DefaultColor = "lightblue"
'''


def filter_block(variant, at_cm, lam, mat):
    """The ripple_filter.py geometry, moved from its 80 cm to at_cm; stacked plates keep their distances."""
    text = RF.variant_block(variant, lam)
    if not text: return ""
    shift = at_cm - RF_REF_CM
    text = TRANSZ_RE.sub(lambda m: f"{m.group(1)}{float(m.group(2)) + shift:.3f}{m.group(3)}", text)
    if mat != "G4_PLEXIGLASS": text = text.replace('"G4_PLEXIGLASS"', f'"{mat}"') + PMMA_HIT
    return f"\n# Ripple filter {variant} at {at_cm:g} cm before the isocentre\n" + text


def angular(mrad):
    """TOPAS refuses an angular spread of 0 for a Gaussian, so a parallel beam needs the distribution "None"."""
    if mrad <= 0: return 's:So/Demo/BeamAngularDistribution  = "None"\n'
    return f'''s:So/Demo/BeamAngularDistribution  = "Gaussian"
d:So/Demo/BeamAngularSpreadX       = {mrad:g} mrad
d:So/Demo/BeamAngularSpreadY       = {mrad:g} mrad
'''


def case(v, ion, e_u, model, variant, at_cm, pos):
    part, a, _ = ION[ion]
    tag = f"_{variant}_d{at_cm:g}".replace(".", "p") if variant != "rifi0" else ""
    name = f"{v.get('prefix') or 'noz'}_{ion}{e_u:g}_{model}{tag}".replace(".", "p")
    depth = max(4.0, math.ceil((1.15 * RF.range_mm(ion, e_u) + 15) / 10.0))     # cm
    nb = int(round(depth * 10 / DEPTH_BIN))
    hist = int(v["histories"])
    spread = spread_for(v, ion, e_u)
    src_cm = pos[0] + 1.0
    vac_hz = (WORLD_HZ - 1 - pos[0] * 10 - 0.5) / 2                              # mm, from 0.5 mm before the window to 1 mm inside the world edge
    s = f'''# {part} {e_u:g} MeV/u ({a * e_u:g} MeV total), HIT nozzle model {model}, {RF.VARIANTS[variant]["en"]}{"" if variant == "rifi0" else f" at {at_cm:g} cm"}.
# Generated by OpenTOPAS Workbench (nozzle). Isocentre = phantom surface (z = 0), beam towards -z, vacuum upstream of the window.
i:Ts/NumberOfThreads            = 0
i:Ts/ShowHistoryCountAtInterval = {max(1000, hist // 50)}
b:Ts/PauseBeforeQuit            = "False"
b:Ge/CheckForOverlaps           = "False"
s:Ge/World/Material = "G4_AIR"
d:Ge/World/HLX = {WORLD_HXY:g} mm
d:Ge/World/HLY = {WORLD_HXY:g} mm
d:Ge/World/HLZ = {WORLD_HZ:g} mm
s:Ge/Vacuum/Type     = "TsBox"
s:Ge/Vacuum/Parent   = "World"
s:Ge/Vacuum/Material = "G4_Galactic"
d:Ge/Vacuum/HLX      = {WORLD_HXY - 1:g} mm
d:Ge/Vacuum/HLY      = {WORLD_HXY - 1:g} mm
d:Ge/Vacuum/HLZ      = {vac_hz:.3f} mm
d:Ge/Vacuum/TransZ   = {WORLD_HZ - 1 - vac_hz:.3f} mm
s:Ge/Phantom/Type     = "TsBox"
s:Ge/Phantom/Parent   = "World"
s:Ge/Phantom/Material = "G4_WATER"
d:Ge/Phantom/HLX      = 10.0 cm
d:Ge/Phantom/HLY      = 10.0 cm
d:Ge/Phantom/HLZ      = {depth / 2:g} cm
d:Ge/Phantom/TransZ   = -{depth / 2:g} cm
d:Ge/Phantom/RotX     = 180. deg
i:Ge/Phantom/ZBins    = {nb}
s:Ge/IsoPlane/Type     = "TsBox"
s:Ge/IsoPlane/Parent   = "World"
s:Ge/IsoPlane/Material = "G4_AIR"
d:Ge/IsoPlane/HLX      = {LAT_HALF:g} mm
d:Ge/IsoPlane/HLY      = {LAT_HALF:g} mm
d:Ge/IsoPlane/HLZ      = 0.05 mm
d:Ge/IsoPlane/TransZ   = 0.15 mm
i:Ge/IsoPlane/XBins    = {int(round(2 * LAT_HALF / LAT_BIN))}
d:Ge/BeamPosition/TransZ = {src_cm * 10:g} mm
s:So/Demo/BeamParticle             = "{part}"
d:So/Demo/BeamEnergy               = {a * e_u:g} MeV
u:So/Demo/BeamEnergySpread         = {spread:.5f}
s:So/Demo/BeamPositionDistribution = "Gaussian"
d:So/Demo/BeamPositionSpreadX      = {float(v["sigma"]):g} mm
d:So/Demo/BeamPositionSpreadY      = {float(v["sigma"]):g} mm
s:So/Demo/BeamPositionCutoffShape  = "Ellipse"
d:So/Demo/BeamPositionCutoffX      = {5 * float(v["sigma"]):g} mm
d:So/Demo/BeamPositionCutoffY      = {5 * float(v["sigma"]):g} mm
{angular(float(v.get("divergence") or 0))}i:So/Demo/NumberOfHistoriesInRun   = {hist}
s:Sc/IDD/Quantity   = "DoseToMedium"
s:Sc/IDD/Component  = "Phantom"
s:Sc/IDD/OutputType = "csv"
s:Sc/IDD/IfOutputFileAlreadyExists = "Overwrite"
s:Sc/IDD/OutputFile = "IDD_{name}"
s:Sc/LAT/Quantity   = "Fluence"
s:Sc/LAT/Component  = "IsoPlane"
s:Sc/LAT/OutputType = "csv"
s:Sc/LAT/IfOutputFileAlreadyExists = "Overwrite"
s:Sc/LAT/OutputFile = "LAT_{name}"
s:Sc/LAT/OnlyIncludeParticlesOfGeneration = "Primary"
'''
    s += nozzle_block(model, pos, float(v.get("wire_half") or 40), int(v.get("hv_planes", 2)))
    s += filter_block(variant, at_cm, float(v.get("period") or 1.5), v.get("filter_mat") or "PMMA_HIT")
    if v.get("phsp", "off") != "off":
        z = pos[-1] * 10 - 1.0                                   # 1 mm downstream of MWPC 2, air in air
        s += f'''
# Phase space 1 mm behind MWPC 2 (all particles leaving the -z face). A source of type PhaseSpace on a component at the
# same place starts from it: d:Ge/<comp>/TransZ = {z - 0.05:.2f} mm, s:So/<src>/PhaseSpaceFileName = "PHSP_{name}".
s:Ge/PhspPlane/Type     = "TsBox"
s:Ge/PhspPlane/Parent   = "World"
s:Ge/PhspPlane/Material = "G4_AIR"
d:Ge/PhspPlane/HLX      = {WORLD_HXY - 1:g} mm
d:Ge/PhspPlane/HLY      = {WORLD_HXY - 1:g} mm
d:Ge/PhspPlane/HLZ      = 0.05 mm
d:Ge/PhspPlane/TransZ   = {z - 0.05:.2f} mm
s:Sc/PHSP/Quantity   = "PhaseSpace"
s:Sc/PHSP/Surface    = "PhspPlane/ZMinusSurface"
s:Sc/PHSP/OutputType = "Binary"
s:Sc/PHSP/IfOutputFileAlreadyExists = "Overwrite"
s:Sc/PHSP/OutputFile = "PHSP_{name}"
b:Sc/PHSP/KillAfterPhaseSpace = "{"True" if v["phsp"] == "only" else "False"}"
'''
    auto = [pos[0] + 0.5, pos[-1] - 0.5] + ([at_cm + 1.5, at_cm - 3.5] if variant != "rifi0" else []) + [0.0]
    kept = []                                                   # an automatic plane within 1 mm of another one is dropped
    for p in auto:
        if all(abs(p - q) >= 0.1 for q in kept): kept.append(p)
    s += MON.block(v, name, kept, a * e_u, WORLD_HXY, WORLD_HZ)
    return name + ".txt", s


def generate(v):
    energies = numbers(v.get("energies"), "energies")
    pos = element_positions(v.get("positions") or FIELDS[3]["default"])
    at = numbers(v.get("filter_pos") or "108", "filter positions")
    for p in at:
        if not 5 <= p < pos[-1] - 1: raise ValueError(f"filter at {p:g} cm: must lie between 5 cm and 1 cm behind MWPC 2")
    lam = float(v.get("period") or 1.5)
    files = []
    for ion in v.get("ions") or []:
        for e in energies:
            for model in v.get("models") or []:
                if model not in MODELS: raise ValueError(f"unknown model {model}")
                for var in v.get("variants") or []:
                    for p in (at if var != "rifi0" else [None]):
                        files.append(case(v, ion, e, model, var, p, pos))
    for var in sorted(v.get("variants") or []):
        m = RF.PIN_RE.match(var)
        if m and not m.group(3):
            files.append((RF.stl_name(m.group(1), m.group(2)) + ".stl",
                          RF.pins_stl(float(m.group(2)), lam, profile="uniform" if m.group(1) == "pin" else "pyramid")))
    if not files: raise ValueError("select at least one ion, one model and one variant")
    return files
