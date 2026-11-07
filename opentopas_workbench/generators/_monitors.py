"""Beam profile monitors for generators that build a beam line: thin scoring planes in a parallel world.

Not a generator itself (the loader skips files that start with "_"). A generator adds FIELDS to its own and calls
block() for each case. The planes sit in their own parallel world, so they score fluence without changing the physics.
Outputs, read by the Beam profile view:
    mon_<position in cm>_<case>_<p|e|o|all>.bin    2D fluence map (p protons, e electrons, o everything else)
    mon_<position in cm>s_<case>_<spec|espec>.csv  energy spectrum on a wide unbinned plane 0.3 mm upstream
Positions are in cm before the isocenter (z = 0), "p" stands for the decimal point (19.52 cm -> 19p52).
"""
import math

FIELDS = [
    {"key": "mon_pos", "type": "text", "default": "",
     "label": {"en": "Beam profile monitors at [cm before the isocenter], comma separated",
               "de": "Strahlprofil-Monitore bei [cm vor dem Isozentrum], durch Komma getrennt"}},
    {"key": "mon_auto", "type": "bool", "default": False,
     "label": {"en": "Monitors before and after every component", "de": "Monitore vor und hinter jeder Komponente"}},
    {"key": "mon_bin", "type": "number", "default": 0.5,
     "label": {"en": "Monitor bin width [mm]", "de": "Monitor-Binbreite [mm]"}},
    {"key": "mon_half", "type": "number", "default": 40,
     "label": {"en": "Monitor half width [mm]", "de": "Monitor-Halbbreite [mm]"}},
    {"key": "mon_split", "type": "bool", "default": True,
     "label": {"en": "Monitors: protons, electrons and other particles separately",
               "de": "Monitore: Protonen, Elektronen und andere Teilchen getrennt"}},
    {"key": "mon_spec", "type": "bool", "default": True,
     "label": {"en": "Monitors: energy spectrum", "de": "Monitore: Energiespektrum"}},
]
THICK = 0.1          # mm, plane thickness
SPEC_GAP = 0.3       # mm, the spectrum plane sits this far upstream of the map plane
MAX_EBINS = 1000


def positions(text):
    out = []
    for tok in str(text or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok: continue
        p = float(tok)
        if p < 0: raise ValueError(f"monitor position must be 0 or more cm before the isocenter: {tok}")
        out.append(p)
    return out


def tag(p):
    """19.52 -> 19p52, 5.0 -> 5, 0.5 -> 0p5: the position in output and component names."""
    return f"{p:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def wanted(v, auto):
    pos = positions(v.get("mon_pos")) + (list(auto) if v.get("mon_auto") else [])
    pos = sorted({round(p, 2) for p in pos}, reverse=True)
    for a, b in zip(pos, pos[1:]):
        if a - b < 0.1: raise ValueError(f"monitor planes closer than 1 mm: {a:g} and {b:g} cm")
    return pos


def block(v, case, auto, e_max, world_half_xy, world_half_z):
    """Parameter text for the monitors of one case, or "" when none are wanted.

    auto: positions [cm] before and after the components of this beam line; e_max: largest kinetic energy [MeV] for the
    spectrum range; world_half_xy, world_half_z: half sizes of the world [mm], the planes must fit into it."""
    pos = wanted(v, auto)
    if not pos: return ""
    bin_mm = float(v.get("mon_bin") or 0.5)
    if bin_mm <= 0: raise ValueError("monitor bin width must be positive")
    half = min(float(v.get("mon_half") or 40), world_half_xy - 1)
    nb = max(1, int(round(2 * half / bin_mm)))
    half = nb * bin_mm / 2
    split, spec = v.get("mon_split", True), v.get("mon_spec", True)
    kinds = [("p", '1 "proton"', "Named"), ("e", '1 "e-"', "Named"), ("o", '2 "proton" "e-"', "NotNamed")] if split else [("all", None, None)]
    ew = max(0.25, math.ceil(e_max * 1.02 / MAX_EBINS * 4) / 4)
    en = int(math.ceil(e_max * 1.02 / ew))
    spectra = []
    if spec:
        spectra = [("spec", '1 "proton"', en, ew), ("espec", '1 "e-"', 50, 0.02)] if split else [("spec", None, en, ew)]
    t = [f"""
# Beam profile monitors at {", ".join(f"{p:g}" for p in pos)} cm before the isocenter, in a parallel world (no effect on
# the physics). Read them in the Beam profile view.
s:Ge/Monitors/Type       = "Group"
s:Ge/Monitors/Parent     = "World"
b:Ge/Monitors/IsParallel = "True"
"""]
    for p in pos:
        z = p * 10 + THICK / 2              # downstream face of the plane at the position
        if z + SPEC_GAP + THICK > world_half_z: raise ValueError(f"monitor at {p:g} cm lies outside the world")
        n, out = "Mon" + tag(p), f"mon_{tag(p)}"
        t.append(f"""s:Ge/{n}/Type       = "TsBox"
s:Ge/{n}/Parent     = "Monitors"
b:Ge/{n}/IsParallel = "True"
d:Ge/{n}/HLX        = {half:g} mm
d:Ge/{n}/HLY        = {half:g} mm
d:Ge/{n}/HLZ        = {THICK / 2:g} mm
d:Ge/{n}/TransZ     = {z:.2f} mm
i:Ge/{n}/XBins      = {nb}
i:Ge/{n}/YBins      = {nb}
""")
        for k, names, how in kinds:
            sc = f"{n}{k.upper()}"
            t.append(f"""s:Sc/{sc}/Quantity   = "Fluence"
s:Sc/{sc}/Component  = "{n}"
s:Sc/{sc}/OutputType = "binary"
s:Sc/{sc}/IfOutputFileAlreadyExists = "Overwrite"
s:Sc/{sc}/OutputFile = "{out}_{case}_{k}"
""" + (f"sv:Sc/{sc}/OnlyIncludeParticles{how} = {names}\n" if names else ""))
        if spectra:
            t.append(f"""s:Ge/{n}s/Type       = "TsBox"
s:Ge/{n}s/Parent     = "Monitors"
b:Ge/{n}s/IsParallel = "True"
d:Ge/{n}s/HLX        = {world_half_xy - 1:g} mm
d:Ge/{n}s/HLY        = {world_half_xy - 1:g} mm
d:Ge/{n}s/HLZ        = {THICK / 2:g} mm
d:Ge/{n}s/TransZ     = {z + SPEC_GAP:.2f} mm
""")
        for k, names, nbins, w in spectra:
            sc = f"{n}{k.capitalize()}"
            t.append(f"""s:Sc/{sc}/Quantity   = "Fluence"
s:Sc/{sc}/Component  = "{n}s"
s:Sc/{sc}/OutputType = "csv"
s:Sc/{sc}/IfOutputFileAlreadyExists = "Overwrite"
s:Sc/{sc}/OutputFile = "{out}s_{case}_{k}"
i:Sc/{sc}/EBins      = {nbins}
d:Sc/{sc}/EBinMin    = 0 MeV
d:Sc/{sc}/EBinMax    = {nbins * w:g} MeV
""" + (f"sv:Sc/{sc}/OnlyIncludeParticlesNamed = {names}\n" if names else ""))
    return "".join(t)

