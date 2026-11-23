"""A plain TOPAS parameter file to the case model of _fluka.py, for files that no generator of the workbench wrote.

Translated: the world, boxes, cylinders along z, TsRidgeFilter plates, one Gaussian beam from BeamPosition, and one
depth dose grid (DoseToMedium on a box with ZBins). Components in a parallel world (the beam monitors) and settings
that only change the output or the display are left out and named in the input. Every other line is refused with
its line number: a case that is translated half right would compare two different set-ups. One direction only; a
FLUKA input is never turned back into TOPAS.

Defaults follow what the generators of the workbench rely on: the world is G4_AIR unless set, the source Demo is a
beam from the component BeamPosition with Gaussian position and angle, and BeamPosition sits at the world's top
(TransZ = Ge/World/HLZ) shooting along -z.
"""
import importlib.util, math, os, re

_spec = importlib.util.spec_from_file_location("wb_fluka_writer", os.path.join(os.path.dirname(__file__), "_fluka.py"))
FK = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FK)

LINE_RE = re.compile(r"^(\w+):(\S+?)\s*=\s*(.*)$")
TO_CM = {"m": 100.0, "cm": 1.0, "mm": 0.1, "um": 1e-4, "nm": 1e-7}
TO_DEG = {"deg": 1.0, "rad": 180.0 / math.pi, "mrad": 0.18 / math.pi}
TO_MEV = {"ev": 1e-6, "kev": 1e-3, "mev": 1.0, "gev": 1e3, "tev": 1e6}
TO_MM = {"m": 1000.0, "cm": 10.0, "mm": 1.0, "um": 1e-3, "nm": 1e-6}   # the ridge profiles, so 0.01 mm stays 0.01
UNITS = {"length": TO_CM, "mm": TO_MM, "angle": TO_DEG, "energy": TO_MEV}
PARTICLES = {"proton": "proton", "alpha": "alpha", "genericion(6,12)": "GenericIon(6,12)",
             "genericion(6,12,6)": "GenericIon(6,12)"}
# Settings without an effect on the physics of a run: threads, progress, display, overlap checks, file handling.
HARMLESS = {"ts/numberofthreads", "ts/showhistorycountatinterval", "ts/showhistorycountonsingleline",
            "ts/pausebeforequit", "ts/useqt", "ts/pausebeforesequence", "ts/maxinterruptedhistories",
            "ge/checkforoverlaps", "ge/quitifoverlapdetected", "ge/checkinsideenvelopesforoverlaps",
            "ph/default/emrangemax"}   # the upper end of Geant4's EM tables, nothing that FLUKA has
SHOW_KEYS = {"color", "drawingstyle", "invisible", "vissegspercircle"}
COMMON_KEYS = {"type", "parent", "material", "transx", "transy", "transz", "rotx", "roty", "rotz", "isparallel"}
TYPE_KEYS = {"tsbox": {"hlx", "hly", "hlz", "xbins", "ybins", "zbins"},
             "tscylinder": {"rmin", "rmax", "hl", "sphi", "dphi"},
             "tsridgefilter": {"xpoints", "zpoints", "width", "length", "displacement"}}
BEAM_KEYS = {"type", "component", "beamparticle", "beamenergy", "beamenergyspread", "beampositiondistribution",
             "beampositionspreadx", "beampositionspready", "beampositioncutoffshape", "beampositioncutoffx",
             "beampositioncutoffy", "beamangulardistribution", "beamangularspreadx", "beamangularspready",
             "beamangularcutoffx", "beamangularcutoffy", "numberofhistoriesinrun"}
SCORER_KEYS = {"quantity", "component", "outputtype", "outputfile", "ifoutputfilealreadyexists"}
CUTOFF_SIGMAS = 3.0     # an elliptic cut at 3 sigma loses about 1 % of the beam; FLUKA's BEAM card has no cut
MAX_REASONS = 8


class Refused(ValueError):
    """The lines without a FLUKA translation, each as (line number, reason)."""

    def __init__(self, reasons):
        self.reasons = sorted(reasons)
        shown = [f"line {n}: {r}" if n else r for n, r in self.reasons[:MAX_REASONS]]
        more = len(self.reasons) - MAX_REASONS
        super().__init__("no FLUKA translation for this TOPAS file: " + "; ".join(shown)
                         + (f"; and {more} more" if more > 0 else ""))


class Param:
    def __init__(self, n, kind, name, raw):
        self.n, self.kind, self.name, self.raw = n, kind, name, raw


def read_params(text):
    """{lower-case name: Param}, and the reasons of lines that are not parameters this reader takes."""
    params, bad = {}, []
    for n, line in enumerate(text.splitlines(), 1):
        s = line.split("#", 1)[0].strip()
        if not s: continue
        if s.lower().startswith("includefile"):
            bad.append((n, "includeFile: the converter reads one file, the included parameters stay unknown")); continue
        m = LINE_RE.match(s)
        if m is None: bad.append((n, f"not a TOPAS parameter: {s[:60]}")); continue
        kind, name, raw = m.group(1).lower(), m.group(2), m.group(3).strip()
        key = name.lower()
        if key in params:
            bad.append((n, f"{name} is set again (first in line {params[key].n})")); continue
        params[key] = Param(n, kind, name, raw)
    return params, bad


def scalar(p, unit=None):
    """The value of a parameter: a str, bool, int, or a float in cm, deg or MeV."""
    raw = p.raw
    if p.kind == "s": return raw.strip('"')
    if p.kind == "b": return raw.strip('"').lower() in ("true", "t", "1")
    toks = raw.split()
    try:
        if p.kind in ("i", "u") and len(toks) == 1:
            return int(toks[0]) if p.kind == "i" else float(toks[0])
        if p.kind == "d" and len(toks) == 2 and unit:
            u = UNITS[unit].get(toks[1].lower())
            if u is None: raise ValueError(f"unit {toks[1]} is not a unit of {unit}")
            return float(toks[0]) * u
    except ValueError as e:
        if "unit" in str(e): raise
    raise ValueError("expressions and references to other parameters are not read")


def vector(p, unit):
    toks = p.raw.split()
    if p.kind != "dv" or len(toks) < 3: raise ValueError("expected a dv vector: count, values, unit")
    count, vals, u = int(toks[0]), [float(t) for t in toks[1:-1]], UNITS[unit].get(toks[-1].lower())
    if u is None: raise ValueError(f"unit {toks[-1]} is not a unit of {unit}")
    if count != len(vals): raise ValueError(f"the count {count} does not match the {len(vals)} values")
    return [v * u for v in vals]


class Reader:
    """Values by name with the reasons of the lines that fail, so that one pass names every problem of a file."""

    def __init__(self, params):
        self.p, self.bad, self.used = params, [], set()

    def has(self, name): return name.lower() in self.p

    def line(self, name): return self.p[name.lower()].n if self.has(name) else 0

    def get(self, name, unit=None, default=None, vec=False):
        p = self.p.get(name.lower())
        if p is None: return default
        self.used.add(name.lower())
        try: return vector(p, unit) if vec else scalar(p, unit)
        except (ValueError, IndexError) as e:
            self.bad.append((p.n, f"{p.name}: {e}")); return default

    def need(self, name, unit=None, vec=False):
        v = self.get(name, unit, vec=vec)
        if v is None and not self.has(name): self.bad.append((0, f"{name} is missing"))
        return v

    def refuse(self, name, why): self.bad.append((self.line(name), f"{self.p[name.lower()].name if self.has(name) else name}: {why}"))


def _zero(v): return v is None or abs(v) < 1e-9


def _half_turn(v): return _zero(v) or abs(abs(v) - 180.0) < 1e-6


def convert(text, title):
    """The FLUKA input of a plain TOPAS file, or Refused with every line that has no translation."""
    params, bad = read_params(text)
    r = Reader(params)
    r.bad += bad
    groups = {}
    for key, p in params.items():
        parts = key.split("/")
        if key in HARMLESS or parts[0] == "gr": r.used.add(key); continue
        if len(parts) == 3 and parts[0] in ("ge", "so", "sc"): groups.setdefault((parts[0], parts[1]), {})[parts[2]] = p
        elif key == "ts/seed": pass
        else: r.bad.append((p.n, f"{p.name}: not translated to FLUKA"))
    comps = {k[1]: v for k, v in groups.items() if k[0] == "ge" and k[1] not in ("world", "beamposition")}
    names = {k: next(iter(v.values())).name.split("/")[1] for k, v in comps.items()}

    def parallel(c, seen=()):
        """True for a component in a parallel world: it scores, but holds no matter that the beam meets."""
        if c in seen or c not in comps: return False
        if r.get(f"Ge/{c}/IsParallel") is True: return True
        par = r.get(f"Ge/{c}/Parent")
        return isinstance(par, str) and par.lower() != "world" and parallel(par.lower(), seen + (c,))
    left_out = sorted(c for c in comps if parallel(c))
    for c in left_out:
        for k in comps[c]: r.used.add(f"ge/{c}/{k}")

    # world
    w_hl = [r.need(f"Ge/World/HL{a}", "length") for a in "XYZ"]
    w_mat = r.get("Ge/World/Material", default="G4_AIR")
    for k, p in groups.get(("ge", "world"), {}).items():
        if k not in {"hlx", "hly", "hlz", "material"} | SHOW_KEYS: r.bad.append((p.n, f"{p.name}: not translated to FLUKA"))
    if w_mat not in FK.MATERIALS: r.refuse("Ge/World/Material", f"material {w_mat} has no FLUKA definition here")

    # beam position: TOPAS shoots along its local +z, which RotX = 180 deg turns into -z
    for k, p in groups.get(("ge", "beamposition"), {}).items():
        if k not in {"transz", "transx", "transy", "rotx", "roty", "rotz", "parent"}:
            r.bad.append((p.n, f"{p.name}: not translated to FLUKA"))
    if r.get("Ge/BeamPosition/Parent", default="World").lower() != "world": r.refuse("Ge/BeamPosition/Parent", "only World")
    for a in "XY":
        if not _zero(r.get(f"Ge/BeamPosition/Trans{a}", "length")): r.refuse(f"Ge/BeamPosition/Trans{a}", "the beam must be on the axis")
    if abs(r.get("Ge/BeamPosition/RotX", "angle", default=180.0) - 180.0) > 1e-6:
        r.refuse("Ge/BeamPosition/RotX", "the beam must point along -z (RotX = 180 deg)")
    for a in "YZ":
        if not _zero(r.get(f"Ge/BeamPosition/Rot{a}", "angle")): r.refuse(f"Ge/BeamPosition/Rot{a}", "the beam must point along -z")
    beam_z = r.get("Ge/BeamPosition/TransZ", "length", default=w_hl[2])

    # source
    sources = sorted(k[1] for k in groups if k[0] == "so")
    beam = None
    if len(sources) != 1:
        r.bad.append((min((p.n for k, v in groups.items() if k[0] == "so" for p in v.values()), default=0),
                      f"exactly one source is translated, the file has {len(sources)}"))
    else:
        s = sources[0]; S = f"So/{s}/"
        for k, p in groups[("so", s)].items():
            if k not in BEAM_KEYS: r.bad.append((p.n, f"{p.name}: not translated to FLUKA"))
        default_type = "Beam" if s == "demo" else None
        typ = r.get(S + "Type", default=default_type)
        if typ is None: r.bad.append((0, f"{S}Type is missing"))
        elif typ.lower() != "beam": r.refuse(S + "Type", f"source type {typ}: only Beam is translated")
        if r.get(S + "Component", default="BeamPosition").lower() != "beamposition":
            r.refuse(S + "Component", "the beam must start at BeamPosition")
        for d in ("BeamPositionDistribution", "BeamAngularDistribution"):
            if r.get(S + d, default="Gaussian").lower() != "gaussian": r.refuse(S + d, "only a Gaussian beam is translated")
        part = r.need(S + "BeamParticle")
        if part is not None and part.lower() not in PARTICLES:
            r.refuse(S + "BeamParticle", f"{part}: only proton, alpha and GenericIon(6,12) are translated")
        sig = [r.need(S + f"BeamPositionSpread{a}", "length") for a in "XY"]
        div = [r.need(S + f"BeamAngularSpread{a}", "angle") for a in "XY"]
        if None not in div and abs(div[0] - div[1]) > 1e-9:
            r.refuse(S + "BeamAngularSpreadY", "FLUKA's BEAM card takes one divergence for x and y")
        for a, i in (("X", 0), ("Y", 1)):
            cut = r.get(S + f"BeamPositionCutoff{a}", "length")
            if cut is not None and sig[i] is not None and cut < CUTOFF_SIGMAS * sig[i]:
                r.refuse(S + f"BeamPositionCutoff{a}", f"a cut below {CUTOFF_SIGMAS:g} sigma changes the beam; FLUKA has no cut")
            cut = r.get(S + f"BeamAngularCutoff{a}", "angle")
            if cut is not None and div[i] is not None and cut < CUTOFF_SIGMAS * div[i]:
                r.refuse(S + f"BeamAngularCutoff{a}", f"a cut below {CUTOFF_SIGMAS:g} sigma changes the beam; FLUKA has no cut")
        r.get(S + "BeamPositionCutoffShape")
        energy, hist = r.need(S + "BeamEnergy", "energy"), r.need(S + "NumberOfHistoriesInRun")
        if hist is not None and hist < 1: r.refuse(S + "NumberOfHistoriesInRun", "no histories")
        if energy is not None and energy <= 0: r.refuse(S + "BeamEnergy", "the energy must be above 0"); energy = None
        # TOPAS gives the Demo source a spread of its own when the line is missing; the converter does not guess it
        spread = r.need(S + "BeamEnergySpread")
        if spread is not None and spread < 0: r.refuse(S + "BeamEnergySpread", "the spread must be 0 or more"); spread = None
        if part is not None and energy is not None and part.lower() in PARTICLES:
            ion = FK.HEAVY_ION.get(PARTICLES[part.lower()])
            if ion and energy / ion[1] > FK.BME_MAX_MEV_U:
                r.refuse(S + "BeamEnergy", f"above {FK.BME_MAX_MEV_U:g} MeV/u FLUKA needs DPMJET and RQMD linked in"); energy = None
        if any(v is not None and v < 0 for v in sig + div): r.refuse(S + "BeamPositionSpreadX", "spreads must be 0 or more"); sig = [None]
        if None not in (part, energy, hist, spread) and None not in sig and None not in div and part.lower() in PARTICLES:
            beam = {"particle": PARTICLES[part.lower()], "energy_mev": energy, "spread_pct": float(spread),
                    "sigma_mm": [10 * v for v in sig], "div_mrad": [v * math.pi / 0.18 for v in div],
                    "z_cm": beam_z, "histories": int(hist)}
            seed = r.get("Ts/Seed")
            if seed is not None:
                if isinstance(seed, int) and 0 < seed < FK.SEED_LIMIT: beam["seed"] = seed
                else: r.refuse("Ts/Seed", f"FLUKA takes a seed from 1 to {FK.SEED_LIMIT - 1}")

    # scorers: one depth dose grid; scorers of the parallel world go with it
    scorer = None
    for s in sorted(k[1] for k in groups if k[0] == "sc"):
        sp, S = groups[("sc", s)], f"Sc/{s}/"
        comp = r.get(S + "Component")
        if isinstance(comp, str) and comp.lower() in left_out:
            for k in sp: r.used.add(f"sc/{s}/{k}")
            continue
        for k, p in sp.items():
            if k not in SCORER_KEYS: r.bad.append((p.n, f"{p.name}: not translated to FLUKA"))
        q = r.get(S + "Quantity")
        if q is None or q.lower() != "dosetomedium":
            r.bad.append((r.line(S + "Quantity"), f"{S}Quantity {q}: only DoseToMedium is translated")); continue
        if (r.get(S + "OutputType", default="csv") or "").lower() != "csv":
            r.refuse(S + "OutputType", "the FLUKA result is written as csv, so the TOPAS scorer must be csv too")
        if scorer is not None:
            r.bad.append((r.line(S + "Quantity"), "a second dose scorer: only one depth dose grid is translated")); continue
        own = next(iter(sp.values())).name.split("/")[1]
        scorer = (own, comp, r.get(S + "OutputFile", default=own))
        r.get(S + "IfOutputFileAlreadyExists")
    if scorer is None: r.bad.append((0, "no dose scorer: a DoseToMedium grid in World is needed"))
    elif not isinstance(scorer[1], str): r.bad.append((0, f"Sc/{scorer[0]}/Component is missing"))

    # components
    phantom, solids, filters, boxes = None, [], [], []
    for c in sorted((c for c in comps if c not in left_out), key=lambda c: min(p.n for p in comps[c].values())):
        G, cp = f"Ge/{names[c]}/", comps[c]
        typ = (r.get(G + "Type") or "")
        t = typ.lower()
        if t not in TYPE_KEYS:
            r.bad.append((r.line(G + "Type") or min(p.n for p in cp.values()),
                          f"{G}Type {typ or 'missing'}: only TsBox, TsCylinder and TsRidgeFilter are translated")); continue
        for k, p in cp.items():
            if k not in COMMON_KEYS | TYPE_KEYS[t] | SHOW_KEYS: r.bad.append((p.n, f"{p.name}: not translated to FLUKA"))
        if (r.get(G + "Parent") or "").lower() != "world": r.refuse(G + "Parent", "only components placed in World are translated")
        mat = r.need(G + "Material")
        if mat is not None and mat not in FK.MATERIALS: r.refuse(G + "Material", f"material {mat} has no FLUKA definition here")
        tr = [r.get(G + f"Trans{a}", "length", default=0.0) for a in "XYZ"]
        rot = [r.get(G + f"Rot{a}", "angle", default=0.0) for a in "XYZ"]
        if t == "tsbox":
            hl = [r.need(G + f"HL{a}", "length") for a in "XYZ"]
            bins = [r.get(G + f"{a}Bins", default=1) for a in "XYZ"]
            scored = scorer is not None and isinstance(scorer[1], str) and scorer[1].lower() == c
            if None in hl or None in tr: continue
            if min(hl) <= 0: r.refuse(G + "HL" + "XYZ"[hl.index(min(hl))], "half lengths must be above 0"); continue
            if scored:
                if bins[0] != 1 or bins[1] != 1: r.refuse(G + ("XBins" if bins[0] != 1 else "YBins"), "only a depth grid (XBins = YBins = 1) is translated")
                if not isinstance(bins[2], int) or bins[2] < 1: r.refuse(G + "ZBins", "a whole number of depth bins is needed")
                if abs(rot[0] - 180.0) > 1e-6 or not _zero(rot[1]) or not _zero(rot[2]):
                    r.refuse(G + "RotX", "the dose grid must be turned by RotX = 180 deg, so that bin 0 is the entrance")
                for a, v in zip("XY", tr[:2]):
                    if not _zero(v): r.refuse(G + f"Trans{a}", "the dose grid must be on the beam axis")
                phantom = {"hl_cm": hl, "tz_cm": tr[2], "material": mat, "zbins": bins[2]}
            else:
                for a, b in zip("XYZ", bins):
                    if b != 1: r.refuse(G + f"{a}Bins", "bins on a box that no dose scorer reads")
                for a, v in zip("XYZ", rot):
                    if not _half_turn(v): r.refuse(G + f"Rot{a}", "a box is translated turned by 0 or 180 deg only")
                solids.append({"kind": "box", "hl_cm": hl, "t_cm": tr, "material": mat, "line": r.line(G + "Type")})
            boxes.append((G, [(tr[q] - hl[q], tr[q] + hl[q]) for q in range(3)]))
        elif t == "tscylinder":
            rmin, rmax, hl = r.get(G + "RMin", "length", default=0.0), r.need(G + "RMax", "length"), r.need(G + "HL", "length")
            if not _zero(r.get(G + "SPhi", "angle")) or abs(r.get(G + "DPhi", "angle", default=360.0) - 360.0) > 1e-6:
                r.refuse(G + "DPhi", "only full cylinders are translated")
            for a, v in zip("XY", rot):
                if not _half_turn(v): r.refuse(G + f"Rot{a}", "a cylinder is translated along z only")
            if None in (rmin, rmax, hl): continue
            if not 0 <= rmin < rmax or hl <= 0: r.refuse(G + "RMax", "0 <= RMin < RMax and HL above 0 are needed"); continue
            solids.append({"kind": "cyl", "rmin_cm": rmin, "rmax_cm": rmax, "hl_cm": hl, "t_cm": tr, "material": mat,
                           "line": r.line(G + "Type")})
            boxes.append((G, [(tr[0] - rmax, tr[0] + rmax), (tr[1] - rmax, tr[1] + rmax), (tr[2] - hl, tr[2] + hl)]))
        else:
            f = ridge(r, G, tr, rot, mat)
            if f is not None:
                filters.append(f)
                boxes.append((G, ridge_box(f)))
    if scorer is not None and phantom is None and isinstance(scorer[1], str):
        r.bad.append((r.line(f"Sc/{scorer[0]}/Component"), f"the dose scorer must sit on a TsBox in World, not {scorer[1]}"))

    # overlaps: TOPAS would stop on them too, and FLUKA needs every point in exactly one region
    if None not in w_hl:
        for G, bb in boxes:
            if any(lo < -h - 1e-9 or hi > h + 1e-9 for (lo, hi), h in zip(bb, w_hl)):
                r.bad.append((r.line(G + "Type"), f"{G[3:-1]} reaches out of the world"))
        if not _zero(beam_z) and abs(beam_z) > w_hl[2]: r.refuse("Ge/BeamPosition/TransZ", "the beam starts outside the world")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (Ga, a), (Gb, b) = boxes[i], boxes[j]
            if all(min(a[q][1], b[q][1]) - max(a[q][0], b[q][0]) > 1e-9 for q in range(3)):
                r.bad.append((r.line(Gb + "Type"), f"{Gb[3:-1]} overlaps {Ga[3:-1]}"))
    if r.bad: raise Refused(r.bad)
    notes = [f"Converted from the TOPAS file {title} by the workbench's converter for plain TOPAS files."]
    if not r.has("Ge/World/Material"): notes.append("Ge/World/Material is not set: G4_AIR, as the workbench's generators assume.")
    if left_out: notes.append("Left out, in a parallel world without matter: " + ", ".join(names[c] for c in left_out) + ".")
    for s in solids: s.pop("line")
    name = scorer[2]
    return FK.write({"title": f"{title}: {beam['particle']} {beam['energy_mev']:g} MeV", "notes": notes,
                     "world": {"hl_cm": w_hl, "material": w_mat}, "phantom": phantom, "beam": beam, "filters": filters,
                     "solids": solids, "score": {"name": name, "short": re.sub(r"\W", "", scorer[0])[:8] or "DOSE"}})


def ridge(r, G, tr, rot, mat):
    """The _rifi.ridges record of a TsRidgeFilter, or None after its problems went to the reader.

    TOPAS stands ridge k centred on Displacement k along the profile axis; the writer takes an even spacing about
    the plate's middle ridge plus a shift, so the displacements must be evenly spaced. TransX and TransY move the
    plate in the world, which after RotZ is the profile axis and the ridge axis."""
    xs, zs = r.need(G + "XPoints", "mm", vec=True), r.need(G + "ZPoints", "mm", vec=True)
    width, disp = r.need(G + "Width", "mm"), r.need(G + "Displacement", "mm", vec=True)
    length = r.get(G + "Length", "mm", default=None)
    tx, ty = (r.get(G + f"Trans{a}", "mm", default=0.0) for a in "XY")
    if length is None: r.bad.append((r.line(G + "Type"), f"{G}Length is missing")); return None
    if length <= 0: r.refuse(G + "Length", "must be above 0"); return None
    for a, v in zip("XY", rot):
        if not _zero(v): r.refuse(G + f"Rot{a}", "tilted plates have no FLUKA writer yet"); return None
    if None in (xs, zs, width, disp): return None
    if len(xs) != len(zs) or len(xs) < 2: r.refuse(G + "ZPoints", "XPoints and ZPoints need the same count, at least 2"); return None
    if any(b <= a for a, b in zip(xs, xs[1:])) or xs[0] < 0 or xs[-1] > width + 1e-9:
        r.refuse(G + "XPoints", "XPoints must rise strictly from 0 up to Width"); return None
    if min(zs) < 0 or max(zs) <= 0: r.refuse(G + "ZPoints", "ZPoints must be thicknesses of 0 or more"); return None
    n = len(disp) - 1
    pitch = disp[1] - disp[0] if n else width
    if n and (pitch < width - 1e-9 or any(abs(b - a - pitch) > 1e-7 for a, b in zip(disp, disp[1:]))):
        r.refuse(G + "Displacement", "the ridges must be evenly spaced by at least Width"); return None
    # TOPAS sizes the plate's envelope symmetric about 0 from the displacements (ripple_filter.ridge), so an offset
    # set there would cut ridges off in TOPAS: it is taken only as TransX, as the generators write it.
    if abs(disp[0] + (n // 2) * pitch) > 1e-7:
        r.refuse(G + "Displacement", "the displacements must be (k - n/2) times the spacing; move the plate with TransX"); return None
    # The sign of a move of a turned plate in TOPAS was never checked against a TOPAS run, so it is refused.
    if not _zero(rot[2] % 360) and not (_zero(tx) and _zero(ty)):
        r.refuse(G + "TransX", "a turned plate moved sideways is not translated"); return None
    th = math.radians(rot[2])
    along = tx * math.cos(th) + ty * math.sin(th)        # mm, the plate's move along its profile axis
    across = -tx * math.sin(th) + ty * math.cos(th)
    if not _zero(across): r.refuse(G + "TransY", "a move along the ridges is not translated"); return None
    shift = disp[0] + (n // 2) * pitch + along           # as thickness_segments places the ridges
    return {"x": xs, "z": zs, "width": width, "pitch": pitch, "n": n, "tz": tr[2], "rotz": rot[2],
            "shift": shift + 0.0, "material": mat, "height": max(zs), "length_mm": length}


def ridge_box(f):
    """The plate's envelope as a box in the TOPAS frame (cm), for the overlap check."""
    segs = FK.thickness_segments(f)
    th = math.radians(f["rotz"])
    corners = [(u * math.cos(th) - v * math.sin(th), u * math.sin(th) + v * math.cos(th))
               for u in (segs[0][0], segs[-1][1]) for v in (-f["length_mm"] / 2, f["length_mm"] / 2)]
    zc, h = f["tz"], f["height"] / 20
    return [(min(c[0] for c in corners) / 10, max(c[0] for c in corners) / 10),
            (min(c[1] for c in corners) / 10, max(c[1] for c in corners) / 10), (zc - h, zc + h)]
