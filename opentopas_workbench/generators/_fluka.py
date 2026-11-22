"""FLUKA input from a case model, the second backend next to the TOPAS parameter files.

A case model is the description that a generator also writes its TOPAS file from (ripple_filter.case_model,
pencil_beam.case_model): world, phantom, beam, filter layers (the TsRidgeFilter records of _rifi.ridges) and the depth
dose scorer, all in the TOPAS frame. The card syntax follows the public FLUKA manual (version 4-5.2,
https://flukafiles.web.cern.ch/manual/); nothing here comes from a FLUKA installation, and no user routine is needed.

Frame: FLUKA describes beam width and divergence for a beam along +z, and TOPAS shoots along -z. The FLUKA frame is
the TOPAS frame turned by 180 deg about y (x -> -x, z -> -z): a proper rotation, so no geometry is mirrored, the beam
runs along +z, and the phantom entrance sits at the low z end, so depth bin 0 is the entrance in both codes.

Filters: each layer is cut along its profile coordinate into slabs between two break points, where the thickness is
linear. A slab is the intersection of two planes across the profile, the plate's bottom, its top line, and the two
ends along the ridge; the air above the line is a zone of its own, so the layer's envelope is tiled exactly. LATTICE
would copy one period instead, but needs one ROT-DEFI transformation per cell (81 cells per layer at 1.5 mm, up to
three layers), more than the manual's index range of ROT-DEFI allows.
"""
import json, math

GEV = 1e-3          # MeV to GeV
FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))   # Gaussian FWHM / sigma; FLUKA's BEAM card takes FWHM
MASS_GEV = {"proton": 0.93827209, "alpha": 3.72737941}
BEAM_NAME = {"proton": "PROTON", "alpha": "4-HELIUM"}   # 4-HELIUM: WHAT(1) is the energy of the whole ion
# Geant4 NIST materials of the TOPAS cases: density (g/cm3), mean excitation energy I (eV), composition. Atom counts
# (positive) or mass fractions (negative), as COMPOUND reads them. Elements are FLUKA's pre-defined ones.
MATERIALS = {
    "G4_WATER": ("WATERG4", 1.0, 78.0, [(2.0, "HYDROGEN"), (1.0, "OXYGEN")]),
    "G4_PLEXIGLASS": ("PMMAG4", 1.19, 74.0, [(5.0, "CARBON"), (8.0, "HYDROGEN"), (2.0, "OXYGEN")]),
    "G4_AIR": ("AIRG4", 0.00120479, 85.7, [(-0.000124, "CARBON"), (-0.755267, "NITROGEN"), (-0.231781, "OXYGEN"),
                                            (-0.012827, "ARGON")]),
}
# Thresholds. TOPAS keeps Geant4's production cut of 0.05 mm range for every particle (about 60 keV for electrons in
# water, a few keV for photons). FLUKA cuts by energy: DEFAULTS HADROTHErapy sets 100 keV for hadron transport and
# delta rays; electrons and positrons are produced and transported down to 100 keV, photons to 10 keV.
E_CUT_GEV, G_CUT_GEV, DELTA_GEV = 1e-4, 1e-5, 1e-4
UNIT = 21           # USRBIN output unit, binary (negative on the card), merged by usbsuw and printed by usbrea
CYCLES = 5          # rfluka -M: independent runs, so usbrea gives a statistical error per bin


CASE_TAG = "* workbench-case "   # a comment card: FLUKA skips it, the converter reads the scorer layout from it


class Unsupported(ValueError):
    """A case that the FLUKA writer cannot write 1:1 (tilted plates, STL pins, ions beyond helium)."""


def num(v):
    """A number for a fixed-format WHAT field: at most 10 characters, always with a decimal point."""
    if isinstance(v, str): return v
    v = float(v)
    if v == 0: return "0.0"
    for p in range(9, 0, -1):
        s = f"{v:.{p}g}"
        if "." not in s and "e" not in s: s += "."
        if "e" in s:
            m, e = s.split("e"); s = (m if "." in m else m + ".") + "E" + str(int(e))
        if len(s) <= 10: return s
    raise ValueError(f"number {v} does not fit a WHAT field")


def card(name, *what, sdum=""):
    """One card in FLUKA's fixed format A8,2X,6E10.0,A8; a blank WHAT is None."""
    w = list(what) + [None] * (6 - len(what))
    return f"{name:<8}  " + "".join(f"{'' if x is None else num(x):>10}" for x in w) + f"{sdum:<8}".rstrip() + "\n"


def g(v):
    """A number in the free-format geometry: full precision, no exponent for ordinary values."""
    v = round(float(v), 9) + 0.0
    return f"{v:.9g}" if abs(v) >= 1e-4 or v == 0 else f"{v:.6e}"


def to_fluka(p):
    """A point or direction (x, y, z) of the TOPAS frame in the FLUKA frame (180 deg about y)."""
    return (-p[0], p[1], -p[2])


def material(key):
    if key not in MATERIALS: raise Unsupported(f"material {key} has no FLUKA definition here")
    return MATERIALS[key][0]


# ------------------------------------------------------------------ filter layers as planes and zones
def thickness_segments(r):
    """[(u0, u1, t0, t1)] in mm along the layer's profile axis, t linear between, t = 0 in the gaps between ridges.

    A ridge of width w stands centred on its displacement (k - n // 2) * pitch, moved by shift; TOPAS closes the
    profile at 0 on both ends, so the ridge's first and last points rise vertically from the bottom."""
    xs, zs, w = [float(v) for v in r["x"]], [float(v) for v in r["z"]], float(r["width"])
    if xs[0] > 0: xs, zs = [0.0] + xs, [0.0] + zs
    if xs[-1] < w: xs, zs = xs + [w], zs + [0.0]
    segs = []
    for k in range(r["n"] + 1):
        o = (k - r["n"] // 2) * r["pitch"] - w / 2 + r["shift"]
        if segs and o > segs[-1][1] + 1e-9: segs.append((segs[-1][1], o, 0.0, 0.0))     # gap between ridges
        for j in range(len(xs) - 1):
            if xs[j + 1] - xs[j] > 1e-9: segs.append((o + xs[j], o + xs[j + 1], zs[j], zs[j + 1]))
    return segs


def layer_geometry(i, r):
    """Bodies, the zones of the plate, the zones of the air in its envelope, and the envelope, for layer i (1-based).

    The layer's own frame: u across the ridges, v along them, h up from the plate's bottom (towards the source).
    Its bottom sits at TransZ minus half the envelope height (the envelope is the full profile height)."""
    th = math.radians(float(r["rotz"]))
    eu = (math.cos(th), math.sin(th), 0.0)                 # RotZ turns the plate about the beam axis
    ev = (-math.sin(th), math.cos(th), 0.0)
    h_env, z0 = float(r["height"]), float(r["tz"]) * 10 - float(r["height"]) / 2   # mm, TOPAS frame
    half_v = 60.0                                          # mm, Length = 12 cm along the ridges
    def point(u, v, h): return tuple((u * eu[q] + v * ev[q]) / 10 + (0, 0, (z0 + h) / 10)[q] for q in range(3))
    def normal(nu, nv, nh): return tuple(nu * eu[q] + nv * ev[q] + (0, 0, nh)[q] for q in range(3))
    bodies, names = [], {}
    def plane(key, n, pt):
        if key in names: return names[key]
        name = f"L{i}{key[0]}{len(bodies):05d}"
        n, pt = to_fluka(n), to_fluka(pt)
        ln = math.sqrt(sum(c * c for c in n))
        bodies.append(f"PLA {name} " + " ".join(g(c / ln) for c in n) + " " + " ".join(g(c) for c in pt))
        names[key] = name
        return name
    U = lambda u: plane(("U", round(u, 6)), normal(1, 0, 0), point(u, 0, 0))          # inside: profile coordinate < u
    bottom = plane(("B",), normal(0, 0, -1), point(0, 0, 0))                          # inside: above the bottom
    top = plane(("T",), normal(0, 0, 1), point(0, 0, h_env))                          # inside: below the envelope top
    vp = plane(("P",), normal(0, 1, 0), point(0, half_v, 0))
    vm = plane(("M",), normal(0, -1, 0), point(0, -half_v, 0))
    box = f"+{bottom} +{top} +{vp} +{vm}"
    solid, air = [], []
    segs = thickness_segments(r)
    for u0, u1, t0, t1 in segs:
        a, b = U(u0), U(u1)
        if t0 <= 1e-9 and t1 <= 1e-9:
            air.append(f"+{b} -{a} {box}"); continue
        if abs(t1 - t0) < 1e-9: line = plane(("H", round(t0, 6)), normal(0, 0, 1), point(0, 0, t0))
        else: line = plane(("S", u0), normal(-(t1 - t0), 0, u1 - u0), point(u0, 0, t0))   # normal points up and out
        solid.append(f"+{b} -{a} +{line} {box}")
        if t0 < h_env - 1e-9 or t1 < h_env - 1e-9: air.append(f"+{b} -{a} -{line} {box}")
    envelope = f"+{U(segs[-1][1])} -{U(segs[0][0])} {box}"
    return bodies, solid, air, envelope


def region(name, zones, width=120):
    """A region of one or more zones. Continuation lines start with blanks; the manual allows 132 characters a line."""
    lines, cur = [], f"{name:<8} 5"
    for k, z in enumerate(zones):
        for tok in ((["|"] if len(zones) > 1 else []) + z.split()):
            if len(cur) + 1 + len(tok) > width: lines.append(cur); cur = " " * 10
            cur += " " + tok
    return "\n".join(lines + [cur]) + "\n"


# ------------------------------------------------------------------ the whole input
def write(m):
    """The FLUKA input (text) of a case model. Raises Unsupported for what has no 1:1 translation."""
    b, ph, w = m["beam"], m["phantom"], m["world"]
    part = b["particle"]
    if part not in BEAM_NAME: raise Unsupported(f"beam particle {part}: only protons and helium-4 are written for FLUKA")
    if any(f.get("tilt") for f in m["filters"]): raise Unsupported("tilted filter plates have no FLUKA writer yet")
    t_gev = b["energy_mev"] * GEV
    mass = MASS_GEV[part]
    p_gev = math.sqrt(t_gev ** 2 + 2 * t_gev * mass)
    dp_fwhm = FWHM * (t_gev + mass) / p_gev * t_gev * b["spread_pct"] / 100     # dp = (T + m) / p dT
    per_cycle = primaries(m) // CYCLES
    mats = [w["material"], ph["material"]] + [f["material"] for f in m["filters"]]
    used = list(dict.fromkeys(mats))
    out = ["TITLE\n", m["title"][:80] + "\n",
           "* Generated by OpenTOPAS Workbench from the same case model as the TOPAS file. Frame: the TOPAS frame\n",
           "* turned by 180 deg about y, so the beam runs along +z and the phantom entrance is at z = 0.\n",
           CASE_TAG + json.dumps(case_record(m), separators=(",", ":")) + "\n",
           card("GLOBAL", 5000, None, None, None, 1.0),
           card("DEFAULTS", sdum="HADROTHE"),
           card("BEAM", -t_gev, -dp_fwhm, -FWHM * b["div_mrad"][0], -FWHM * b["sigma_mm"][0] / 10,
                -FWHM * b["sigma_mm"][1] / 10, None, sdum=BEAM_NAME[part]),
           card("BEAMPOS", *to_fluka((0.0, 0.0, b["z_cm"]))),
           card("GEOBEGIN", sdum="COMBNAME"),
           "    0    0          " + m["title"][:60] + "\n"]
    hx, hy, hz = w["hl_cm"]
    pz0, pz1 = sorted(to_fluka((0, 0, ph["tz_cm"] + s * ph["hl_cm"][2]))[2] for s in (-1, 1))
    geo = [f"RPP BLKBODY {g(-2 * hx)} {g(2 * hx)} {g(-2 * hy)} {g(2 * hy)} {g(-2 * hz)} {g(2 * hz)}",
           f"RPP WORLD {g(-hx)} {g(hx)} {g(-hy)} {g(hy)} {g(-hz)} {g(hz)}",
           f"RPP PHANT {g(-ph['hl_cm'][0])} {g(ph['hl_cm'][0])} {g(-ph['hl_cm'][1])} {g(ph['hl_cm'][1])} {g(pz0)} {g(pz1)}"]
    regs, assign, envelopes = [], [], []
    for i, f in enumerate(m["filters"], 1):
        bodies, solid, air, env = layer_geometry(i, f)
        geo += bodies
        regs.append(region(f"RIFI{i}", solid))
        if air: regs.append(region(f"RIAIR{i}", air))
        envelopes.append(env)
        assign.append((material(f["material"]), f"RIFI{i}"))
        if air: assign.append((material(w["material"]), f"RIAIR{i}"))
    regs = [region("BLKHOLE", ["+BLKBODY -WORLD"]),
            region("AIR", ["+WORLD -PHANT" + "".join(f" -( {e} )" for e in envelopes)]),
            region("PHANTOM", ["+PHANT"])] + regs
    assign = [("BLCKHOLE", "BLKHOLE"), (material(w["material"]), "AIR"), (material(ph["material"]), "PHANTOM")] + assign
    out += [l + "\n" for l in geo] + ["END\n"] + regs + ["END\n", card("GEOEND")]
    for key in used:
        name, rho, ival, comp = MATERIALS[key]
        out.append(card("MATERIAL", None, None, rho, None, None, None, sdum=name))
        for k in range(0, len(comp), 3):
            out.append(card("COMPOUND", *[x for c in comp[k:k + 3] for x in c], sdum=name))
        out.append(card("MAT-PROP", None, None, ival, name, None, None))
    for mat, reg in assign: out.append(card("ASSIGNMA", mat, reg))
    for key in used:
        name = MATERIALS[key][0]
        out.append(card("EMFCUT", -E_CUT_GEV, G_CUT_GEV, 1.0, name, None, None, sdum="PROD-CUT"))
        out.append(card("DELTARAY", DELTA_GEV, None, None, name, None, None))
    for _, reg in assign[1:]: out.append(card("EMFCUT", -E_CUT_GEV, G_CUT_GEV, None, reg, None, None))
    s = m["score"]
    out += [card("USRBIN", 10.0, "DOSE", -UNIT, ph["hl_cm"][0], ph["hl_cm"][1], pz1, sdum=s["short"]),
            card("USRBIN", -ph["hl_cm"][0], -ph["hl_cm"][1], pz0, 1.0, 1.0, float(ph["zbins"]), sdum="&"),
            card("RANDOMIZ", 1.0, float(b.get("seed", 54217137))),
            card("START", float(per_cycle)),
            card("STOP")]
    return "".join(out)


# ------------------------------------------------------------------ results: usbrea output to the workbench CSV
def primaries(m):
    """Primaries that FLUKA runs: CYCLES cycles of START each, so the requested count is rounded up."""
    return CYCLES * max(1, math.ceil(int(m["beam"]["histories"]) / CYCLES))


def case_record(m):
    """What the converter needs of the case model, written into the input so a moved .inp still converts."""
    return {"name": m["score"]["name"], "primaries": primaries(m), "hl_cm": m["phantom"]["hl_cm"],
            "zbins": int(m["phantom"]["zbins"])}


def read_case(text):
    """The case record of an input written here, as the part of a case model that to_csv reads."""
    line = next((l for l in text.splitlines() if l.startswith(CASE_TAG)), None)
    if line is None: raise ValueError("not an input of the workbench's FLUKA writer: no case record")
    rec = json.loads(line[len(CASE_TAG):])
    title = text.splitlines()[1].strip() if text.startswith("TITLE") else rec["name"]
    return {"title": title, "score": {"name": rec["name"]}, "beam": {"histories": rec["primaries"]},
            "phantom": {"hl_cm": rec["hl_cm"], "zbins": rec["zbins"]}}


def _floats(line):
    try: return [float(t) for t in line.replace(",", " ").split()]
    except ValueError: return None


def read_usbrea(text, nbins):
    """(values, relative errors in %) of the first USRBIN detector in the ASCII file that usbrea writes.

    The file holds header lines in words, a line "Data follow in a matrix ...", the bin values, a line "Percentage
    errors follow in a matrix ...", the errors (absent when usbrea read a single cycle). Numbers count only after one
    of these two lines: the file opens with a Fortran carriage-control "1" on a line of its own, which once was read
    as the value of the first bin and moved every bin one deeper (task 3357)."""
    sec, cur = {}, None
    for line in text.splitlines():
        low = line.lower()
        if "follow in a matrix" in low:
            key = "errs" if "error" in low else "vals"
            if key in sec: break          # a second detector: only the first is read
            cur = sec[key] = []
            continue
        if "binning n." in low and sec: break
        f = _floats(line)
        if f and cur is not None: cur += f
    vals, errs = sec.get("vals", []), sec.get("errs")
    if len(vals) != nbins:
        raise ValueError(f"usbrea output holds {len(vals)} values after its data header, the detector has {nbins} bins")
    return vals, (errs if errs is not None and len(errs) == nbins else None)


GY_PER_GEV_G = 1.602176634e-10 / 1e-3     # J per GeV over kg per g


def to_csv(vals, errs, m):
    """The depth dose as a TOPAS-style CSV (DoseToMedium, Gy summed over all histories), so the workbench's
    results, comparisons and the MCP tools read it unchanged. FLUKA gives GeV/g per primary; the sum is over the
    primaries that ran (primaries(m)), as TOPAS sums over its histories."""
    ph, n = m["phantom"], primaries(m)
    nb = int(ph["zbins"])
    width = 2 * ph["hl_cm"][2] / nb
    head = [f"# FLUKA (OpenTOPAS Workbench converter from usbrea), case {m['title']}",
            f"# Results for scorer: {m['score']['name']}",
            f"# X in 1 bin of {g(2 * ph['hl_cm'][0])} cm", f"# Y in 1 bin of {g(2 * ph['hl_cm'][1])} cm",
            f"# Z in {nb} bins of {g(width)} cm",
            "# DoseToMedium ( Gy ) : Sum" + ("   Standard_Deviation" if errs else "")]
    rows = []
    for k, v in enumerate(vals):
        d = v * GY_PER_GEV_G * n
        rows.append(f"0, 0, {k}, {d:.6g}" + (f", {d * errs[k] / 100:.6g}" if errs else ""))
    return "\n".join(head + rows) + "\n"
