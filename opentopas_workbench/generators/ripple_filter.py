"""Generator: ripple filter (RiFi) depth-dose cases, 1D TsRidgeFilter and 2D pin filters (binary STL).

Example plugin for OpenTOPAS Workbench. Geometry follows Grevillot et al. (2D 2 mm triangle) and generic 1D triangles / 2D pins.
2D pins: square pins on a 0.1 mm base plate, either shaped for a uniform thickness distribution ("pin", same modulation as a
1D triangle) or straight pyramids ("pyr", thickness distribution falling linearly). Variants ending in "e" are 1D ridge filters
with the same thickness distribution as the 2D pins, to check that only the distribution matters for the depth dose.
Binary filters: plates with two thicknesses, half of each period thick ("bin"). A stack of binary plates whose steps halve
from plate to plate ("stk", 2 plates crossed = 4 equally frequent thicknesses, "x3" = 3 plates = 8) samples the same
uniform kernel as a triangle in coarser steps. "pair" is two equal triangles of half the height crossed, "pair...v" two of
the full height (Weber and Kraft's double filter, as rifi2).
Staircases ("stp"): one piece with four equally wide steps of a quarter of the height each, the lowest step open (no
base plate), as a laminate of three plates covering 75, 50 and 25 % of each period. Same 4-step kernel as "stk" without
the second base plate. "stp...g" has a coarse period of 6 mm (a test plate cut by hand or a simple laser).
Gap and tilt: "stp4s10" builds the staircase as the laminate itself, three separate 1 mm plates (75, 50, 25 % cover,
aligned) with 10 mm of air between them; "xb4s10" is two crossed binary plates without base plate (2 and 1 mm), 10 mm
apart. A suffix "n15" tilts the filter group by 15 mrad about the y axis (perpendicular to the stripes of the first
plate), so a plate further downstream slides sideways by tilt x distance against the first one. The laminate's plates
share their right edge: at +15 mrad the smaller plates slide back over the larger ones and the kernel stays the same,
at -15 mrad ("n-15") they overhang the open quarter.
Phantom surface at the isocentre, filters 80 cm upstream.
"""
import importlib.util, math, os, re, struct


def _helper(name):
    """A shared module next to this file (_monitors.py, _rifi.py), loaded by path like the generators themselves."""
    spec = importlib.util.spec_from_file_location("wbgen_" + name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


MON = _helper("_monitors")
R = _helper("_rifi")

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
    "bin4": {"en": "binary plate, step 2 mm", "de": "Binärplatte, Stufe 2 mm"},
    "stk4": {"en": "2 binary plates crossed, steps 2 + 1 mm", "de": "2 Binärplatten gekreuzt, Stufen 2 + 1 mm"},
    "stk4x3": {"en": "3 binary plates, steps 2 + 1 + 0.5 mm", "de": "3 Binärplatten, Stufen 2 + 1 + 0,5 mm"},
    "pair4": {"en": "2 triangles 2 mm crossed", "de": "2 Dreiecke 2 mm gekreuzt"},
    "pair4v": {"en": "2 triangles 4 mm crossed", "de": "2 Dreiecke 4 mm gekreuzt"},
    "stp4": {"en": "staircase 4 steps of 1 mm, open lowest step", "de": "Treppe 4 Stufen je 1 mm, unterste offen"},
    "stp3": {"en": "staircase 4 steps of 0.75 mm, open lowest step", "de": "Treppe 4 Stufen je 0,75 mm, unterste offen"},
    "stp4g": {"en": "staircase 4 steps of 1 mm, coarse period 6 mm", "de": "Treppe 4 Stufen je 1 mm, grobe Periode 6 mm"},
    "stp4n15": {"en": "staircase 4 steps of 1 mm, tilted 15 mrad", "de": "Treppe 4 Stufen je 1 mm, 15 mrad geneigt"},
    "stp4s10": {"en": "staircase as 3 plates of 1 mm, 10 mm air between", "de": "Treppe aus 3 Platten je 1 mm, 10 mm Luft dazwischen"},
    "stp4s10n15": {"en": "staircase as 3 plates, 10 mm air, tilted 15 mrad", "de": "Treppe aus 3 Platten, 10 mm Luft, 15 mrad geneigt"},
    "stp4s10n-15": {"en": "staircase as 3 plates, 10 mm air, tilted -15 mrad (upper plates overhang)", "de": "Treppe aus 3 Platten, 10 mm Luft, -15 mrad geneigt (obere Platten stehen über)"},
    "xb4s10": {"en": "2 binary plates crossed without base, 2 + 1 mm, 10 mm air", "de": "2 Binärplatten gekreuzt ohne Grundplatte, 2 + 1 mm, 10 mm Luft"},
    "xb4s10n15": {"en": "2 binary plates crossed without base, 10 mm air, tilted 15 mrad", "de": "2 Binärplatten gekreuzt ohne Grundplatte, 10 mm Luft, 15 mrad geneigt"},
}
# Definitions saved with the RiFi designer of the web UI (workspace/.workbench/rifi), selectable as lib-NAME.
LIBRARY = R.library()
VARIANTS.update({"lib-" + k: {"en": f"{d['title'] or k} (library)", "de": f"{d['title'] or k} (Bibliothek)"} for k, d in LIBRARY.items()})
PIN_BASE, PIN_TOP_HALF, PIN_RINGS = 0.1, 0.04, 8     # mm, mm, segments of the pin flank
PIN_RE = re.compile(r"^(pin|pyr)(\d+(?:\.\d+)?)(e?)$")
SHAPE_RE = re.compile(r"^(bin|stk|pair)(\d+(?:\.\d+)?)(x3|v)?$")
STAIR_RE = re.compile(r"^stp(\d+(?:\.\d+)?)(g?)(?:s(\d+(?:\.\d+)?))?(?:n(-?\d+(?:\.\d+)?))?$")
XB_RE = re.compile(r"^xb(\d+(?:\.\d+)?)s(\d+(?:\.\d+)?)(?:n(-?\d+(?:\.\d+)?))?$")
PIVOT = 80.0        # cm, the tilted filter group turns about the first plate
COARSE = 6.0        # mm, period of the coarse staircase "stp...g"
BASE = 0.3          # mm, base plate of the 1D triangles and binary plates
EDGE = 0.001        # mm, flank of a binary step: TsRidgeFilter needs XPoints that rise strictly
# Plates behind the first one: (TransZ cm, RotZ deg). The third plate turns by 45 deg, not 90: at 0 deg it would line
# up with the first (same period) and the two plates would add to 2 thicknesses instead of 4. The third plate lies
# behind the 77.0 cm "after the filters" monitor plane.
STACK = [(80.0, 0), (78.3, 90), (76.6, 45)]
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


def ridge(name, xs, zs, width, n, tz, rotz=0, pitch=None, shift=0.0, tilt=0.0, material="G4_PLEXIGLASS"):
    """A TsRidgeFilter of n + 1 ridges. With pitch > width the ridges stand apart and the gaps hold no material.

    shift (mm) moves the plate in x; TOPAS centres a ridge of width w on its displacement, so plates of different
    widths line up at their right edges with shift = (w_max - w) / 2. It goes into TransX, not into the displacements:
    TOPAS sizes the envelope box symmetric about 0 from the displacements, so shifted ridges would stick out of it. tilt (mrad) turns the plate about y as part of a
    group that pivots at PIVOT: the plate turns and slides sideways by its distance from the pivot times the tilt. The
    slide is what matters; turning a 1 mm plate by 15 mrad changes its path length by 1e-4."""
    pitch = pitch or width
    disp = " ".join(f"{(i - n // 2) * pitch:.3f}" for i in range(n + 1))
    tx = shift + (tz - PIVOT) * 10 * tilt / 1000
    turn = (f"d:Ge/{name}/RotY     = {tilt:g} mrad\n" if tilt else "") + \
           (f"d:Ge/{name}/TransX   = {tx:.4f} mm\n" if tilt or shift else "")
    return f'''
s:Ge/{name}/Type     = "TsRidgeFilter"
s:Ge/{name}/Parent   = "World"
s:Ge/{name}/Material = "{material}"
d:Ge/{name}/TransZ   = {tz} cm
d:Ge/{name}/RotZ     = {rotz} deg
{turn}dv:Ge/{name}/XPoints = {len(xs)} {" ".join(f"{v:.3f}" for v in xs)} mm
dv:Ge/{name}/ZPoints = {len(zs)} {" ".join(f"{v:.3f}" for v in zs)} mm
d:Ge/{name}/Width    = {width} mm
d:Ge/{name}/Length   = 12.0 cm
dv:Ge/{name}/Displacement = {n + 1} {disp} mm
'''


def triangle(d, lam):
    return [0, 0.01, lam / 2 - 0.04, lam / 2 + 0.04, lam - 0.01, lam], [BASE, BASE, d, d, BASE, BASE]


def binary(step, lam):
    """One period: thin (BASE) on the first half, BASE + step on the second, flanks EDGE wide.

    The points sit on the 0.001 mm grid that ridge() writes, so an odd third decimal of the period cannot round two
    of them onto the same value."""
    half, end = round(lam / 2, 3), round(lam, 3)
    return [0, half - EDGE, half, end - EDGE, end], [BASE, BASE, BASE + step, BASE + step, BASE]


def staircase(height, lam):
    """One ridge of a staircase with four equal steps, the lowest one open: (XPoints, ZPoints, ridge width).

    The ridge covers the upper three steps (3/4 of the period); the open quarter is the gap to the next ridge, so
    there is no base plate. TOPAS closes the profile at 0 on both ends, and both ends here are vertical."""
    q, step = round(lam / 4, 3), height / 4
    xs = [0, q - EDGE, q, 2 * q - EDGE, 2 * q, 3 * q]
    zs = [step, step, 2 * step, 2 * step, 3 * step, 3 * step]
    return xs, zs, round(3 * q, 3)


def laminate(height, lam, gap):
    """The staircase as three flat plates of height/4 each, covering 3, 2 and 1 quarters of a period, gap mm apart.

    Plate k spans quarters k..2 of the period, so the stack reads 1, 2, 3 steps over quarters 0, 1, 2, as staircase().
    Returns (XPoints, ZPoints, width, shift, TransZ cm) per plate, the first (widest) plate at 80 cm."""
    q, step = round(lam / 4, 3), height / 4
    plates = []
    for k in range(3):
        w = round((3 - k) * q, 3)
        plates.append(([0, w], [step, step], w, round((3 * q - w) / 2, 4), round(PIVOT - k * (step + gap) / 10, 4)))
    return plates


def shape_plates(v, lam):
    """Profiles (XPoints, ZPoints) of the plates of a bin/stk/pair variant, first plate first."""
    m = SHAPE_RE.match(v)
    if not m or (m.group(3) == "x3" and m.group(1) != "stk") or (m.group(3) == "v" and m.group(1) != "pair"):
        raise ValueError(v)
    kind, d = m.group(1), float(m.group(2))
    if kind == "bin": return [binary(d / 2, lam)]
    if kind == "stk": return [binary(d / 2 ** (i + 1), lam) for i in range(3 if m.group(3) else 2)]
    return [triangle(d if m.group(3) else d / 2, lam)] * 2


# ---------------------------------------------------------------- filters as rifi/v1 definitions (_rifi.py)
TEMPLATES = ["t3", "t4", "t6", "rifi1", "rifi2", "bin4", "stk4", "stk4x3", "stp4", "stp3", "pair4", "pair4v"]
LIB = "lib-"          # variant prefix of a definition saved in the workspace library (RiFi designer)
GREVILLOT = [[0.0, 0.0], [0.009, 0.0], [0.46, 1.6], [0.54, 1.6], [0.991, 0.0], [1.0, 0.0]]   # on a 0.4 mm base


def _tri_layer(d, lam, **kw):
    xs, zs = triangle(d, lam)
    return dict(period=lam, base=BASE, profile=[[x, z - BASE] for x, z in zip(xs, zs)], **kw)


def _bin_layer(step, lam, **kw):
    return dict(period=lam, base=BASE, steps=[[0.5, 0.0], [0.5, step]], **kw)


def _stacked(layers):
    """Plates 17 mm apart centre to centre and turned as STACK, written as the air gaps of the definition."""
    out = [dict(layers[0], rotz=STACK[0][1])]
    for L, (tz, rot), (tz0, _) in zip(layers[1:], STACK[1:], STACK):
        h0, h = R.layer_height(out[-1]), R.layer_height(L)
        out.append(dict(L, rotz=rot, gap=round((tz0 - tz) * 10 - h0 / 2 - h / 2, 6)))
    return out


def template(v, lam):
    """The definition of a built-in variant for the period lam, or None for the variants outside rifi/v1 (pins, tilt,
    gaps): t<d>, rifi1, rifi2, bin<d>, stk<d>[x3], pair<d>[v], stp<d> (without g, s or n)."""
    title = VARIANTS.get(v, {}).get("en", v)
    d = {"schema": R.SCHEMA, "name": v, "title": title, "material": "G4_PLEXIGLASS", "spr": R.SPR_PMMA}
    if v in ("rifi1", "rifi2"):
        L = dict(period=1.0, base=0.4, profile=GREVILLOT, count=120)
        d["layers"] = [L] if v == "rifi1" else _stacked([L, L])
        return d
    if re.match(r"^t\d+(?:\.\d+)?$", v):
        d["layers"] = [_tri_layer(float(v[1:]), lam)]
        return d
    m = SHAPE_RE.match(v)
    if m and not ((m.group(3) == "x3" and m.group(1) != "stk") or (m.group(3) == "v" and m.group(1) != "pair")):
        kind, h = m.group(1), float(m.group(2))
        if kind == "bin": d["layers"] = [_bin_layer(h / 2, lam)]
        elif kind == "stk": d["layers"] = _stacked([_bin_layer(h / 2 ** (i + 1), lam) for i in range(3 if m.group(3) else 2)])
        else: d["layers"] = _stacked([_tri_layer(h if m.group(3) else h / 2, lam)] * 2)
        return d
    m = STAIR_RE.match(v)
    if m and not (m.group(2) or m.group(3) or m.group(4)):
        s = float(m.group(1)) / 4
        d["layers"] = [dict(period=4 * round(lam / 4, 3), base=0.0, steps=[[0.25, s], [0.25, 2 * s], [0.25, 3 * s], [0.25, 0.0]],
                            count=int(120 / lam))]
        return d
    return None


def definition_block(d):
    """TOPAS geometry of a rifi/v1 definition: one TsRidgeFilter per layer, the first at 80 cm."""
    rot = lambda r: int(r) if float(r).is_integer() else r
    return "".join(ridge(r["name"], r["x"], r["z"], r["width"], r["n"], r["tz"], rot(r["rotz"]), pitch=r["pitch"],
                         shift=r["shift"], material=r["material"]) for r in R.ridges(d))


def definition(v, lam):
    """The rifi/v1 definition behind variant v (built in, or lib-NAME from the workspace library), else None."""
    if v.startswith(LIB):
        d = LIBRARY.get(v[len(LIB):])
        if d is None: raise ValueError(f"filter {v[len(LIB):]} is not in the library")
        return d
    return template(v, lam)


def variant_block(v, lam):
    if v == "rifi0": return ""
    d = definition(v, lam)
    if d is not None: return definition_block(R.validate(d))
    m = STAIR_RE.match(v)
    if m:
        p = COARSE if m.group(2) else lam
        tilt = float(m.group(4) or 0)
        if m.group(3) is not None:
            return "".join(ridge(f"RiFi{i + 1}", xs, zs, w, int(120 / p), tz, pitch=4 * round(p / 4, 3), shift=sh, tilt=tilt)
                           for i, (xs, zs, w, sh, tz) in enumerate(laminate(float(m.group(1)), p, float(m.group(3)))))
        xs, zs, w = staircase(float(m.group(1)), p)
        return ridge("RiFi1", xs, zs, w, int(120 / p), 80.0, pitch=4 * round(p / 4, 3), tilt=tilt)
    m = XB_RE.match(v)
    if m:
        d, gap, tilt = float(m.group(1)), float(m.group(2)), float(m.group(3) or 0)
        half, s1, s2 = round(lam / 2, 3), d / 2, d / 4    # half of each period carries the step, the other half is open
        tz2 = round(PIVOT - (s1 / 2 + gap + s2 / 2) / 10, 4)
        return (ridge("RiFi1", [0, half], [s1, s1], half, int(120 / lam), PIVOT, 0, pitch=lam, tilt=tilt)
                + ridge("RiFi2", [0, half], [s2, s2], half, int(120 / lam), tz2, 90, pitch=lam, tilt=tilt))
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


def frame(ion, e_u, v, prefix):
    """Case name and phantom half depth (cm), shared by the TOPAS file and the case model."""
    hlz = max(4.0, math.ceil((1.15 * range_mm(ion, e_u) + 15) / 10.0))
    return f"{prefix}_{ion}{int(round(e_u)):03d}_{v}", hlz


def case(ion, e_u, v, hist, spread, lam, prefix):
    part, a, _ = ION[ion]
    name, hlz = frame(ion, e_u, v, prefix)
    s = f'''# {part} {e_u:g} MeV/u ({a * e_u:g} MeV total), {VARIANTS[v]["en"]}. Generated by OpenTOPAS Workbench (ripple_filter).
# Phantom surface = isocentre (z = 0). First filter plate 80 cm upstream, further plates as in the block at the end.
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


# ---------------------------------------------------------------- the case as a model, for the FLUKA writer (_fluka.py)
FK = _helper("_fluka")
TOPAS_PARTICLE = {"p": "proton", "a": "alpha", "c": "GenericIon(6,12)"}


def case_model(ion, e_u, v, hist, spread, lam, prefix):
    """World, phantom, beam, filter layers and scorer of case() as data, in the TOPAS frame (cm, mm where named).

    The filters are the same _rifi.ridges records that definition_block writes as TsRidgeFilter; variants outside
    rifi/v1 (STL pins, tilted or spaced plates, the 1D pin equivalents) raise _fluka.Unsupported."""
    if ion not in TOPAS_PARTICLE: raise FK.Unsupported(f"ion {ion}: the FLUKA writer takes protons, helium-4 and carbon-12")
    _, a, _ = ION[ion]
    name, hlz = frame(ion, e_u, v, prefix)
    filters = []
    if v != "rifi0":
        d = definition(v, lam)
        if d is None: raise FK.Unsupported(f"variant {v} is not a rifi/v1 definition (pins, tilt, gaps)")
        filters = R.ridges(R.validate(d))
    return {"title": f"{name}: {ION[ion][0]} {e_u:g} MeV/u, {VARIANTS[v]['en']}",
            "world": {"hl_cm": [30.0, 30.0, 140.0], "material": "G4_AIR"},
            "phantom": {"hl_cm": [10.0, 10.0, float(hlz)], "tz_cm": -float(hlz), "material": "G4_WATER",
                        "zbins": int(round(2 * hlz * 100))},
            "beam": {"particle": TOPAS_PARTICLE[ion], "energy_mev": a * e_u, "spread_pct": float(spread),
                     "sigma_mm": [3.0, 3.0], "div_mrad": [1.0, 1.0], "z_cm": 85.0, "histories": int(hist)},
            "filters": filters,
            "score": {"name": f"IDD_{name}", "short": "IDD"}}


def generate_fluka(v):
    """FLUKA inputs (name.inp) of the cases that generate() writes for TOPAS; cases without a 1:1 translation are
    listed in a note file instead of being left out silently."""
    energies = [float(x) for x in str(v["energies"]).replace(";", ",").split(",") if x.strip()]
    if not energies: raise ValueError("no energies given")
    lam = float(v.get("period") or 1.5)
    files, skipped = [], []
    for ion in v.get("ions") or []:
        for e in energies:
            for var in v.get("variants") or []:
                try: text = FK.write(case_model(ion, e, var, v["histories"], float(v["spread"]), lam, v.get("prefix") or "mess"))
                except FK.Unsupported as x: skipped.append(f"{ion} {e:g} {var}: {x}"); continue
                files.append((frame(ion, e, var, v.get("prefix") or "mess")[0] + ".inp", text))
    if skipped: files.append(("FLUKA_not_written.txt", "Cases without a FLUKA input:\n" + "\n".join(skipped) + "\n"))
    if not files: raise ValueError("select at least one ion and one variant")
    return files


HEAD_RE = re.compile(r"^# (\S+) ([0-9.eE+-]+) MeV/u \(.*\. Generated by OpenTOPAS Workbench \(ripple_filter\)\.$")
MON_NAME_RE = re.compile(r"^(Ge|Sc)/Mon[^/]*/")


def _value(text, name):
    m = re.search(r"^\S+:" + re.escape(name) + r"\s*=\s*(\S+)", text, re.M)
    if m is None: raise ValueError(f"{name} is missing")
    return m.group(1).strip('"')


def fluka_from_topas(text):
    """The FLUKA input of a TOPAS file that generate() wrote, or ValueError with the reason why not.

    The TOPAS file carries no record of the form it came from; that would change every file generate() writes. The
    case is read back from the file's own lines instead, and accepted only if case() writes exactly this text again:
    an edited file, or one from an older version of this generator, is refused rather than translated half right.
    After the case only the beam profile monitors of _monitors.block may follow; they lie in a parallel world, do not
    touch the physics, and FLUKA leaves them out. The filter period is not in the file as such: the numbers of the
    file are tried as candidates, the same test decides."""
    lines = text.splitlines()
    m = HEAD_RE.match(lines[0]) if lines else None
    if m is None: raise ValueError("the first line is not the header that the ripple filter generator writes")
    ion = next((k for k, x in ION.items() if x[0] == m.group(1)), None)
    if ion is None: raise ValueError(f"unknown particle {m.group(1)}")
    e_u = float(m.group(2))
    hist, spread = int(_value(text, "So/Demo/NumberOfHistoriesInRun")), float(_value(text, "So/Demo/BeamEnergySpread"))
    name = _value(text, "Sc/IDD/OutputFile")[len("IDD_"):]
    # The progress interval follows the histories when case() writes it; a file whose histories were changed by hand
    # keeps the old one, which only sets how often TOPAS prints its progress.
    text = re.sub(r"^(i:Ts/ShowHistoryCountAtInterval\s*=\s*)\d+", lambda x: x.group(1) + str(max(1000, hist // 50)),
                  text, count=1, flags=re.M)
    tag = f"_{ion}{int(round(e_u)):03d}_"
    lams = [1.5] + sorted({float(x) for x in re.findall(r"(?<![\w.])\d+\.?\d*(?![\w.])", text) if 0.1 <= float(x) <= 20} - {1.5})
    at = name.find(tag)
    while at > 0:
        prefix, var = name[:at], name[at + len(tag):]
        if var in VARIANTS:
            for lam in (lams if var != "rifi0" else [1.5]):
                try: case_text = case(ion, e_u, var, hist, spread, lam, prefix)[1]
                except (ValueError, KeyError, ZeroDivisionError): break   # this variant cannot be written at all
                if text.startswith(case_text):
                    rest = text[len(case_text):]
                    for n, line in enumerate(rest.splitlines(), text[:len(case_text)].count("\n") + 1):
                        p = re.match(r"^\s*\w+:(\S+)\s*=", line)
                        if line.strip() and not line.lstrip().startswith("#") and not (p and MON_NAME_RE.match(p.group(1))):
                            raise ValueError(f"line {n} was added after the generated case: {line.strip()[:80]}")
                    return FK.write(case_model(ion, e_u, var, hist, spread, lam, prefix))
        at = name.find(tag, at + 1)
    raise ValueError("the file differs from what the ripple filter generator writes for this case (edited, or from an older version)")


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
