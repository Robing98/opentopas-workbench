"""Generator: ripple filter (RiFi) depth-dose cases, 1D TsRidgeFilter and 2D pin filters (binary STL).

Example plugin for OpenTOPAS Workbench. Geometry follows Grevillot et al. (2D 2 mm triangle) and generic 1D triangles / 2D pins.
2D pins: square pins on a 0.1 mm base plate, either shaped for a uniform thickness distribution ("pin", same modulation as a
1D triangle) or straight pyramids ("pyr", thickness distribution falling linearly). Variants ending in "e" are 1D ridge filters
with the same thickness distribution as the 2D pins, to check that only the distribution matters for the depth dose.
Phantom surface at the isocentre, filters 80 cm upstream.
"""
import importlib.util, math, os, re, struct


def _monitor_helper():
    """The shared monitor option in _monitors.py, loaded by path like the generators themselves."""
    spec = importlib.util.spec_from_file_location("wbgen__monitors", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_monitors.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


MON = _monitor_helper()

NAME = {"en": "Ripple filter study", "de": "Ripple-Filter-Studie"}
DESCRIPTION = {
    "en": "Depth-dose cases with and without ripple filters (1D triangles, crossed pairs, 2D pins as STL).",
    "de": "Tiefendosis-Fälle mit und ohne Ripple-Filter (1D-Dreiecke, gekreuzte Paare, 2D-Pins als STL).",
}
ION = {"p": ("proton", 1, 1), "a": ("alpha", 4, 2), "c": ("GenericIon(6,12)", 12, 6)}
VARIANTS = {
    "rifi0": {"en": "no filter", "de": "ohne Filter"},
    "rifi1": {"en": "Grevillot triangle 2 mm", "de": "Grevillot-Dreieck 2 mm"},
    "rifi2": {"en": "2x Grevillot crossed", "de": "2x Grevillot gekreuzt"},
    "t3": {"en": "1D triangle 3 mm", "de": "1D-Dreieck 3 mm"},
    "t4": {"en": "1D triangle 4 mm", "de": "1D-Dreieck 4 mm"},
    "t6": {"en": "1D triangle 6 mm", "de": "1D-Dreieck 6 mm"},
    "pin3": {"en": "2D pins 3 mm", "de": "2D-Pins 3 mm"},
    "pin4": {"en": "2D pins 4 mm", "de": "2D-Pins 4 mm"},
    "pin6": {"en": "2D pins 6 mm", "de": "2D-Pins 6 mm"},
    "pyr3": {"en": "2D pyramids 3 mm", "de": "2D-Pyramiden 3 mm"},
    "pyr4": {"en": "2D pyramids 4 mm", "de": "2D-Pyramiden 4 mm"},
    "pyr6": {"en": "2D pyramids 6 mm", "de": "2D-Pyramiden 6 mm"},
    "pin4e": {"en": "1D equivalent of 2D pins 4 mm", "de": "1D-Äquivalent der 2D-Pins 4 mm"},
    "pyr4e": {"en": "1D equivalent of 2D pyramids 4 mm", "de": "1D-Äquivalent der 2D-Pyramiden 4 mm"},
}
PIN_BASE, PIN_TOP_HALF, PIN_RINGS = 0.1, 0.04, 8     # mm, mm, segments of the pin flank
PIN_RE = re.compile(r"^(pin|pyr)(\d+(?:\.\d+)?)(e?)$")
FIELDS = [
    {"key": "ions", "type": "multi", "label": {"en": "Ions", "de": "Ionen"}, "default": ["p"],
     "options": [["p", {"en": "Proton", "de": "Proton"}], ["a", "Helium-4"], ["c", {"en": "Carbon-12", "de": "Kohlenstoff-12"}]]},
    {"key": "energies", "type": "text", "label": {"en": "Energies [MeV/u], comma separated", "de": "Energien [MeV/u], durch Komma getrennt"},
     "default": "70, 100, 150"},
    {"key": "variants", "type": "multi", "label": {"en": "Filter variants", "de": "Filtervarianten"}, "default": ["rifi0", "t4"],
     "options": [[k, v] for k, v in VARIANTS.items()]},
    {"key": "histories", "type": "number", "label": {"en": "Histories", "de": "Historien"}, "default": 200000},
    {"key": "spread", "type": "number", "label": {"en": "Energy spread [%]", "de": "Energiebreite [%]"}, "default": 0.2},
    {"key": "period", "type": "number", "label": {"en": "Period of 1D/2D filters [mm]", "de": "Periode der 1D/2D-Filter [mm]"}, "default": 1.5},
    {"key": "prefix", "type": "text", "label": {"en": "File name prefix", "de": "Präfix Dateiname"}, "default": "mess"},
] + MON.FIELDS
# monitor planes for "before and after every component": after the source (85 cm), before the filters (80 cm,
# up to 6 mm thick), after them (the crossed one sits at 78.3 cm), and at the phantom surface. The same planes in every
# variant, so cases with and without a filter compare plane by plane.
AUTO_MONITORS = [84.5, 81.5, 77.0, 0.0]


def range_mm(ion, e_u):
    _, a, z = ION[ion]
    return 0.022 * e_u ** 1.77 * a / z ** 2


def ridge(name, xs, zs, width, n, tz, rotz=0):
    disp = " ".join(f"{(i - n // 2) * width:.2f}" for i in range(n + 1))
    return f'''
s:Ge/{name}/Type     = "TsRidgeFilter"
s:Ge/{name}/Parent   = "World"
s:Ge/{name}/Material = "G4_PLEXIGLASS"
d:Ge/{name}/TransZ   = {tz} cm
d:Ge/{name}/RotZ     = {rotz} deg
dv:Ge/{name}/XPoints = {len(xs)} {" ".join(f"{v:.3f}" for v in xs)} mm
dv:Ge/{name}/ZPoints = {len(zs)} {" ".join(f"{v:.3f}" for v in zs)} mm
d:Ge/{name}/Width    = {width} mm
d:Ge/{name}/Length   = 12.0 cm
dv:Ge/{name}/Displacement = {n + 1} {disp} mm
'''


def variant_block(v, lam):
    grev = ([0.0, 0.009, 0.46, 0.54, 0.991, 1.0], [0.4, 0.4, 2.0, 2.0, 0.4, 0.4])
    if v == "rifi0": return ""
    if v == "rifi1": return ridge("RiFi1", *grev, 1.0, 120, 80.0)
    if v == "rifi2": return ridge("RiFi1", *grev, 1.0, 120, 80.0) + ridge("RiFi2", *grev, 1.0, 120, 78.3, 90)
    if v.startswith("t"):
        d = float(v[1:]); n = int(120 / lam)
        return ridge("RiFi1", [0, 0.01, lam / 2 - 0.04, lam / 2 + 0.04, lam - 0.01, lam], [0.3, 0.3, d, d, 0.3, 0.3], lam, n, 80.0)
    m = PIN_RE.match(v)
    if m:
        kind, d, equiv = m.group(1), float(m.group(2)), m.group(3)
        profile = "uniform" if kind == "pin" else "pyramid"
        if equiv:                                      # 1D ridge with the thickness distribution of the 2D pin
            xs, zs = equivalent_ridge(d, lam, profile=profile)
            return ridge("RiFi1", xs, zs, lam, int(120 / lam), 80.0)
        return f'''
s:Ge/Pins/Type       = "TsCAD"
s:Ge/Pins/Parent     = "World"
s:Ge/Pins/Material   = "G4_PLEXIGLASS"
s:Ge/Pins/InputFile  = "{stl_name(kind, m.group(2))}"
s:Ge/Pins/FileFormat = "stl"
d:Ge/Pins/Units      = 1 mm
d:Ge/Pins/TransZ     = 80.0 cm
'''
    raise ValueError(v)


def stl_name(kind, d):
    return f"pins_{d}mm" if kind == "pin" else f"pyr_{d}mm"


def case(ion, e_u, v, hist, spread, lam, prefix):
    part, a, _ = ION[ion]
    hlz = max(4.0, math.ceil((1.15 * range_mm(ion, e_u) + 15) / 10.0))
    name = f"{prefix}_{ion}{int(round(e_u)):03d}_{v}"
    s = f'''# {part} {e_u:g} MeV/u ({a * e_u:g} MeV total), {VARIANTS[v]["en"]}. Generated by OpenTOPAS Workbench (ripple_filter).
# Phantom surface = isocentre (z = 0). Filter 80 cm upstream (second one 78.3 cm, crossed).
i:Ts/NumberOfThreads            = 0
i:Ts/ShowHistoryCountAtInterval = {max(1000, int(hist) // 50)}
b:Ge/CheckForOverlaps           = "False"
d:Ge/World/HLX = 0.3 m
d:Ge/World/HLY = 0.3 m
d:Ge/World/HLZ = 1.4 m
s:Ge/Phantom/Type     = "TsBox"
s:Ge/Phantom/Parent   = "World"
s:Ge/Phantom/Material = "G4_WATER"
d:Ge/Phantom/HLX      = 10.0 cm
d:Ge/Phantom/HLY      = 10.0 cm
d:Ge/Phantom/HLZ      = {hlz:g} cm
d:Ge/Phantom/TransZ   = -{hlz:g} cm
i:Ge/Phantom/ZBins    = {int(round(2 * hlz * 100))}
d:Ge/Phantom/RotX     = 180. deg
d:Ge/BeamPosition/TransZ = 85.0 cm
s:So/Demo/BeamParticle             = "{part}"
d:So/Demo/BeamEnergy               = {a * e_u:g} MeV
u:So/Demo/BeamEnergySpread         = {spread:g}
d:So/Demo/BeamPositionSpreadX      = 3.0 mm
d:So/Demo/BeamPositionSpreadY      = 3.0 mm
d:So/Demo/BeamAngularSpreadX       = 1.0 mrad
d:So/Demo/BeamAngularSpreadY       = 1.0 mrad
i:So/Demo/NumberOfHistoriesInRun   = {int(hist)}
s:Sc/IDD/Quantity   = "DoseToMedium"
s:Sc/IDD/Component  = "Phantom"
s:Sc/IDD/OutputType = "csv"
s:Sc/IDD/IfOutputFileAlreadyExists = "Overwrite"
s:Sc/IDD/OutputFile = "IDD_{name}"
'''
    if ion == "a":
        s += "\n# Helium-4 above 500/600 MeV total exceeds the default EM table range.\nd:Ph/Default/EMRangeMax = 1000. MeV\n"
        if v == "rifi0":
            s += '''
# Workaround for TOPAS 3.9 / Geant4 10.7 helium bug (harmless with OpenTOPAS): 1 um PMMA foil in the beam path.
s:Ge/Foil/Type     = "TsBox"
s:Ge/Foil/Parent   = "World"
s:Ge/Foil/Material = "G4_PLEXIGLASS"
d:Ge/Foil/HLX      = 6 cm
d:Ge/Foil/HLY      = 6 cm
d:Ge/Foil/HLZ      = 0.0005 mm
d:Ge/Foil/TransZ   = 79.0 cm
'''
    return name + ".txt", s + variant_block(v, lam)


# ---------------------------------------------------------------- 2D pins: rings, STL, 1D equivalent
def pin_rings(height, lam=1.5, base=PIN_BASE, top_half=PIN_TOP_HALF, k=PIN_RINGS, profile="uniform"):
    """Heights and half sides of the square pin cross-section, ring 0 = full cell on top of the base plate.

    uniform: area above height t falls linearly (half side ~ sqrt), so every thickness between base and height is
    equally frequent, like a 1D triangle. pyramid: straight flanks, area falls quadratically, thin parts dominate."""
    h = height - base
    zr = [base + h * r / k for r in range(k + 1)]
    if profile == "uniform": sr = [max((lam / 2) * math.sqrt(1 - r / k), top_half) for r in range(k + 1)]
    elif profile == "pyramid": sr = [max((lam / 2) * (1 - r / k), top_half) for r in range(k + 1)]
    else: raise ValueError(f"unknown pin profile {profile}")
    return zr, sr


def equivalent_ridge(height, lam=1.5, base=PIN_BASE, top_half=PIN_TOP_HALF, k=PIN_RINGS, profile="uniform", sub=6):
    """XPoints and ZPoints of a 1D ridge whose width above each height equals the area fraction of the 2D pin above it.

    Between two rings the pin side changes linearly, so its area fraction changes quadratically; each segment is split
    into `sub` pieces to follow that."""
    zr, sr = pin_rings(height, lam, base, top_half, k, profile)
    pts = []
    for r in range(k):
        for j in range(sub):
            f = j / sub
            s = sr[r] + (sr[r + 1] - sr[r]) * f
            pts.append(((lam / 2) * (1 - (s / (lam / 2)) ** 2), zr[r] + (zr[r + 1] - zr[r]) * f))
    pts.append(((lam / 2) * (1 - (sr[k] / (lam / 2)) ** 2), zr[k]))
    left = []
    for x, z in pts:                                   # drop points that collapse onto the previous x (flat steps)
        if not left or x > left[-1][0] + 1e-6: left.append((x, z))
    right = [(lam - x, z) for x, z in reversed(left)]
    xs = [p[0] for p in left + right]; zs = [p[1] for p in left + right]
    return xs, zs


def pins_stl(height, lam=1.5, base=PIN_BASE, half=22.5, top_half=PIN_TOP_HALF, k=PIN_RINGS, profile="uniform"):
    h = height - base
    n = int(round(2 * half / lam)); half = n * lam / 2
    zr, sr = pin_rings(height, lam, base, top_half, k, profile)
    zc = (base + h) / 2
    tris = []
    def quad(a, b, c, d): tris.append((a, b, c)); tris.append((a, c, d))
    for i in range(n):
        for j in range(n):
            cx, cy = -half + (i + .5) * lam, -half + (j + .5) * lam
            P = lambda r, sx, sy: (cx + sx * sr[r], cy + sy * sr[r], zr[r])
            for r in range(k):
                quad(P(r, -1, -1), P(r, 1, -1), P(r + 1, 1, -1), P(r + 1, -1, -1))
                quad(P(r, 1, -1), P(r, 1, 1), P(r + 1, 1, 1), P(r + 1, 1, -1))
                quad(P(r, 1, 1), P(r, -1, 1), P(r + 1, -1, 1), P(r + 1, 1, 1))
                quad(P(r, -1, 1), P(r, -1, -1), P(r + 1, -1, -1), P(r + 1, -1, 1))
            quad(P(k, -1, -1), P(k, 1, -1), P(k, 1, 1), P(k, -1, 1))
            x0, x1, y0, y1 = cx - lam / 2, cx + lam / 2, cy - lam / 2, cy + lam / 2
            quad((x0, y0, 0), (x0, y1, 0), (x1, y1, 0), (x1, y0, 0))
            if j == 0:     quad((x0, y0, 0), (x1, y0, 0), (x1, y0, base), (x0, y0, base))
            if j == n - 1: quad((x1, y1, 0), (x0, y1, 0), (x0, y1, base), (x1, y1, base))
            if i == 0:     quad((x0, y1, 0), (x0, y0, 0), (x0, y0, base), (x0, y1, base))
            if i == n - 1: quad((x1, y0, 0), (x1, y1, 0), (x1, y1, base), (x1, y0, base))
    out = bytearray(f"2D {profile} pin ripple filter, OpenTOPAS Workbench".encode().ljust(80, b" ")) + struct.pack("<I", len(tris))
    for t in tris:
        t = [(x, y, z - zc) for x, y, z in t]
        u = [t[1][q] - t[0][q] for q in range(3)]; w = [t[2][q] - t[0][q] for q in range(3)]
        nr = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
        ln = math.sqrt(sum(c * c for c in nr)) or 1
        out += struct.pack("<12fH", *(c / ln for c in nr), *t[0], *t[1], *t[2], 0)
    return bytes(out)


def generate(v):
    energies = [float(x) for x in str(v["energies"]).replace(";", ",").split(",") if x.strip()]
    if not energies: raise ValueError("no energies given")
    lam = float(v.get("period") or 1.5)
    files = []
    for ion in v.get("ions") or []:
        for e in energies:
            for var in v.get("variants") or []:
                name, text = case(ion, e, var, v["histories"], float(v["spread"]), lam, v.get("prefix") or "mess")
                files.append((name, text + MON.block(v, name[:-4], AUTO_MONITORS, ION[ion][1] * e, 300, 1400)))
    for var in sorted(v.get("variants") or []):
        m = PIN_RE.match(var)
        if m and not m.group(3):
            files.append((stl_name(m.group(1), m.group(2)) + ".stl",
                          pins_stl(float(m.group(2)), lam, profile="uniform" if m.group(1) == "pin" else "pyramid")))
    if not files: raise ValueError("select at least one ion and one variant")
    return files
