"""Ripple filter definitions (rifi/v1): layers of a periodic filter, their thickness kernel, checks, and SVG masks.

Shared by the ripple_filter and nozzle generators, the RiFi designer of the web UI (server.py) and
tools/rifi_kernel_fit.py. Pure Python (no numpy): the workbench server runs on the standard library alone.

A definition is JSON:
    {"schema": "rifi/v1", "name": "t4", "title": "...", "material": "G4_PLEXIGLASS", "spr": 1.16, "layers": [LAYER, ...]}
A layer is one TsRidgeFilter, a strip profile repeated with its period:
    period   mm, spacing of the ridges
    profile  [[x, z], ...] over one ridge, x rising from 0 to the ridge width (mm), z the height above the base plate
    steps    instead of a profile: [[fraction, height], ...], equally high stripes covering a fraction of the period each
             (a staircase or a binary plate); zero-height steps at the end with no base plate are open
    width    mm, ridge width when it is narrower than the period (profile only; the rest of the period is open)
    base     mm, base plate under the profile (0 = none)
    rotz     deg, turn about the beam axis (0 or 90; 45 for a third plate)
    shift    mm, sideways offset of the ridges
    gap      mm of air between this layer and the one before it (ignored for the first layer)
    count    number of ridges (default: as many as fit into 120 mm)
Layers with the same orientation add up point by point; crossed layers are independent (their kernels convolve).
"""
import bisect, json, math, os, re

SCHEMA = "rifi/v1"
EDGE = 0.001          # mm, flank of a step: TsRidgeFilter needs XPoints that rise strictly
EXTENT = 120.0        # mm, filter size: number of ridges and length of the strips
FIRST_CM = 80.0       # cm, TransZ of the first layer (ripple_filter.py places its filters there)
SPR_PMMA = 1.16       # PMMA to water, as in the shape comparison (task 3176)
PLATES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)   # mm, standard PMMA plate thicknesses
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")


# ------------------------------------------------------------------ definition and validation
def _num(v, what, lo=None, hi=None):
    try: f = float(v)
    except (TypeError, ValueError): raise ValueError(f"{what}: not a number ({v!r})")
    if not math.isfinite(f) or (lo is not None and f < lo) or (hi is not None and f > hi):
        raise ValueError(f"{what}: {f:g} out of range")
    return f


def validate(d):
    """A cleaned copy of definition d, or ValueError naming the first fault."""
    if not isinstance(d, dict): raise ValueError("definition must be an object")
    if d.get("schema", SCHEMA) != SCHEMA: raise ValueError(f"schema must be {SCHEMA}")
    name = str(d.get("name", ""))
    if not NAME_RE.match(name): raise ValueError("name: letters, digits, _ . - only, at most 40 characters")
    layers = d.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= 6: raise ValueError("layers: 1 to 6 layers")
    out = {"schema": SCHEMA, "name": name, "title": " ".join(str(d.get("title", "")).split())[:120],
           "material": str(d.get("material") or "G4_PLEXIGLASS"), "spr": _num(d.get("spr", SPR_PMMA), "spr", 0.1, 5.0),
           "layers": []}
    if not re.match(r"^[A-Za-z0-9_]+$", out["material"]): raise ValueError("material: a TOPAS material name")
    for i, L in enumerate(layers):
        w = f"layer {i + 1}"
        if not isinstance(L, dict): raise ValueError(f"{w}: must be an object")
        c = {"period": _num(L.get("period"), f"{w} period", 0.05, 100.0), "base": _num(L.get("base", 0), f"{w} base", 0, 50),
             "rotz": _num(L.get("rotz", 0), f"{w} rotz", -360, 360), "shift": _num(L.get("shift", 0), f"{w} shift", -100, 100),
             "gap": _num(L.get("gap", 0), f"{w} gap", 0, 1000)}
        if L.get("count") is not None: c["count"] = int(_num(L["count"], f"{w} count", 1, 10000))
        if L.get("steps") is not None:
            st = [[_num(f, f"{w} step fraction", 0, 1), _num(h, f"{w} step height", 0, 50)] for f, h in L["steps"]]
            if len(st) < 2: raise ValueError(f"{w}: at least two steps")
            if abs(sum(f for f, _ in st) - 1) > 1e-6: raise ValueError(f"{w}: step fractions must add up to 1")
            if any(f <= 0 for f, _ in st): raise ValueError(f"{w}: every step needs a fraction above 0")
            c["steps"] = st
        else:
            pr = L.get("profile")
            if not isinstance(pr, list) or len(pr) < 2: raise ValueError(f"{w}: profile needs at least two points")
            pts = [[_num(x, f"{w} profile x", 0, 100), _num(z, f"{w} profile z", 0, 50)] for x, z in pr]
            if pts[0][0] != 0: raise ValueError(f"{w}: profile must start at x = 0")
            if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])): raise ValueError(f"{w}: profile x must rise strictly")
            width = _num(L.get("width", pts[-1][0]), f"{w} width", 0.01, 100)
            if abs(width - pts[-1][0]) > 1e-9: raise ValueError(f"{w}: profile must end at the ridge width")
            if width > c["period"] + 1e-9: raise ValueError(f"{w}: ridge width above the period")
            c["profile"] = pts
            if L.get("width") is not None: c["width"] = width
        out["layers"].append(c)
    return out


def layer_points(L):
    """XPoints, ZPoints (mm, base plate included) and ridge width of a layer for TsRidgeFilter."""
    p, b = L["period"], L["base"]
    if "profile" in L:
        xs = [x for x, _ in L["profile"]]; zs = [b + z for _, z in L["profile"]]
        return xs, zs, L.get("width", p)
    st = list(L["steps"])
    open_end = 0
    while b == 0 and st and st[-1][1] == 0: open_end += st.pop()[0]
    if not st: raise ValueError("a layer of steps needs material")
    bounds, acc = [], 0.0
    for f, _ in st: acc += f; bounds.append(round(acc * p, 3))
    xs, zs = [0], [b + st[0][1]]
    for k in range(1, len(st)):
        c = bounds[k - 1]; xs += [c - EDGE, c]; zs += [b + st[k - 1][1], b + st[k][1]]
    if open_end:                                       # vertical end, the open steps are the gap to the next ridge
        xs.append(bounds[-1]); zs.append(b + st[-1][1])
        return xs, zs, bounds[-1]
    xs += [bounds[-1] - EDGE, bounds[-1]]; zs += [b + st[-1][1], b + st[0][1]]
    return xs, zs, p                                    # closes back to the first step, so the next ridge joins on


def layer_height(L):
    return max(layer_points(L)[1])


def ridges(d, first_cm=FIRST_CM):
    """The TsRidgeFilter of each layer: name, profile, width, pitch, count, TransZ (cm), RotZ, shift, left edge x0.

    TransZ is the centre of the layer's envelope (its full height); a gap is air between two envelopes."""
    out, tz, prev_h = [], first_cm, None
    for i, L in enumerate(d["layers"]):
        xs, zs, width = layer_points(L)
        h = max(zs)
        if prev_h is not None: tz = round(tz - (prev_h / 2 + L["gap"] + h / 2) / 10, 4)
        n = L.get("count") or int(EXTENT / L["period"])
        x0 = -(n // 2) * L["period"] - width / 2 + L["shift"]   # as rifi_kernel_fit.left_edge reads it back
        out.append({"name": f"RiFi{i + 1}", "x": xs, "z": zs, "width": width, "pitch": L["period"], "n": n, "tz": tz,
                    "rotz": L["rotz"], "shift": L["shift"], "x0": x0, "material": d["material"], "height": h})
        prev_h = h
    return out


# ------------------------------------------------------------------ thickness kernel
def interp(q, x, z):
    """Linear interpolation of the profile (x rising, equal x allowed: the later point wins, as numpy.interp)."""
    i = bisect.bisect_right(x, q) - 1
    if i < 0: return z[0]
    if i >= len(x) - 1: return z[-1]
    x0, x1 = x[i], x[i + 1]
    return z[i] if x1 <= x0 else z[i] + (z[i + 1] - z[i]) * (q - x0) / (x1 - x0)


def ridge_samples(x, z, width=None, n=200001, pitch=None, offset=0.0):
    """Thickness along one period of a TsRidgeFilter profile (TOPAS closes the profile at 0 on both ends).

    With a ridge spacing (pitch) wider than the ridge, the gap between two ridges holds no material. offset (mm) is
    where the ridge starts on the common grid, so plates of one period sampled with their own offsets add point by
    point (a laminate, or plates slid against each other by a tilt)."""
    x = [float(v) for v in x]; z = [float(v) for v in z]
    width = width or x[-1]
    if x[0] > 0: x = [0.0] + x; z = [0.0] + z
    if x[-1] < width: x = x + [width]; z = z + [0.0]
    period = max(pitch or width, width)
    if period > width * (1 + 1e-9): x += [math.nextafter(width, math.inf), period]; z += [0.0, 0.0]
    step = period / n
    return [interp(((k * step + step / 2) - offset) % period, x, z) for k in range(n)]


def group_ridges(rs):
    """The combination rule of the thickness kernel: [(rotz, [ridges], aligned)], one entry per independent part.

    Ridges with the same orientation (RotZ mod 180) and the same period are aligned: their thicknesses add point by
    point at their positions. Crossed ridges, or ridges of different periods, count as independent."""
    groups = {}
    for r in rs: groups.setdefault(round(float(r["rotz"]) % 180), []).append(r)
    per = lambda r: max(r.get("pitch") or 0, r["width"] or r["x"][-1])
    out = []
    for rot, g in sorted(groups.items()):
        if all(abs(per(r) - per(g[0])) < 1e-9 for r in g): out.append((rot, g, True))
        else: out.extend((rot, [r], False) for r in g)
    return out


def part_samples(group, spr, n):
    """Water equivalent thickness along one period of an aligned group (or one ridge)."""
    rs = group
    x0 = rs[0].get("x0") or 0.0
    tot = None
    for r in rs:
        s = ridge_samples(r["x"], r["z"], r["width"], n=n, pitch=r.get("pitch"), offset=(r.get("x0") or 0.0) - x0)
        f = spr(r["material"])
        tot = [v * f for v in s] if tot is None else [a + v * f for a, v in zip(tot, s)]
    return tot


def histogram(samples, step):
    lo = math.floor(min(samples) / step) * step
    nb = int((max(samples) - lo) / step) + 2
    w = [0] * nb
    for s in samples: w[min(int((s - lo) / step), nb - 1)] += 1
    while w and w[-1] == 0: w.pop()
    tot = float(len(samples))
    return [lo + (i + 0.5) * step for i in range(len(w))], [c / tot for c in w]


def convolve(a, b, step):
    ta, wa = a; tb, wb = b
    w = [0.0] * (len(wa) + len(wb) - 1)
    for i, x in enumerate(wa):
        if x:
            for j, y in enumerate(wb): w[i + j] += x * y
    return [ta[0] + tb[0] - step / 2 + (k + 0.5) * step for k in range(len(w))], w


def moments(t, w):
    m = sum(a * b for a, b in zip(t, w)); return m, math.sqrt(max(sum(b * (a - m) ** 2 for a, b in zip(t, w)), 0.0))


def kernel(d, step=0.02, n=6000):
    """Thickness kernel of a definition in mm water: grid, weights (sum 1), mean, sigma, raw samples per part."""
    spr = lambda mat: 1.0 if mat.upper() in ("G4_WATER", "WATER") else d["spr"]
    k, parts = None, []
    for _, g, _aligned in group_ridges(ridges(d)):
        s = part_samples(g, spr, n); parts.append(s)
        h = histogram(s, step)
        k = h if k is None else convolve(k, h, step)
    t, w = k
    m, sg = moments(t, w)
    return {"t": t, "w": w, "mean": m, "sigma": sg, "parts": parts}


def design_spacing(t, w, tol=1e-3):
    """Layer spacing (mm water) the kernel is made for: its full width, plus one level spacing for discrete levels.

    N equally spaced levels s apart fill a gap of N s (a 4-step staircase of 1 mm: 4 mm PMMA); a continuous kernel
    fills its own width."""
    step = t[1] - t[0] if len(t) > 1 else 1.0
    runs = []                                           # runs of occupied bins = discrete levels
    for x, y in zip(t, w):
        if y <= tol: continue
        if runs and x - runs[-1][-1][0] < 1.5 * step: runs[-1].append((x, y))
        else: runs.append([(x, y)])
    if not runs: return 0.0
    if 2 <= len(runs) <= 16 and all(len(r) <= 3 for r in runs):
        c = [sum(x * y for x, y in r) / sum(y for _, y in r) for r in runs]
        return (c[-1] - c[0]) * len(c) / (len(c) - 1)
    return runs[-1][-1][0] - runs[0][0][0] + step


# ------------------------------------------------------------------ ripple and lateral width from reference curves
REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rifi_ref.json")
LAYERS = 6            # SOBP layers, as auswertung.py of the shape comparison


def reference():
    with open(REF_FILE, encoding="utf-8") as fh: return json.load(fh)


def _lin(z, y):
    n = len(z); mz = sum(z) / n; my = sum(y) / n
    sxx = sum((a - mz) ** 2 for a in z) or 1.0
    b = sum((a - mz) * (c - my) for a, c in zip(z, y)) / sxx
    return b, my - b * mz


def _shift(z0, dz, d, q):
    """Curve value at depth q, zero beyond its end and its first value before it."""
    f = (q - z0) / dz
    if f <= 0: return d[0]
    i = int(f)
    if i >= len(d) - 1: return 0.0 if f > len(d) - 1 else d[-1]
    return d[i] + (d[i + 1] - d[i]) * (f - i)


def sobp_ripple(z0, dz, d, delta):
    """Peak-to-peak (%) of LAYERS copies shifted by delta, over two whole periods in the middle of the plateau,
    after a straight line is taken off (as sobp() in workspace/offen/rifi_formvergleich/auswertung.py)."""
    zs = [z0 + i * dz for i in range(len(d))]
    peak = zs[max(range(len(d)), key=d.__getitem__)]
    lo, hi = peak - (LAYERS - 2.5) * delta, peak - 1.5 * delta
    idx = [i for i, z in enumerate(zs) if lo <= z <= hi]
    if len(idx) < 3: return None
    s = [sum(_shift(z0, dz, d, zs[i] + k * delta) for k in range(LAYERS)) for i in idx]
    zz = [zs[i] for i in idx]
    b, a = _lin(zz, s)
    rest = [v - (a + b * z) for v, z in zip(s, zz)]
    return 100 * (max(rest) - min(rest)) / (sum(s) / len(s))


def filtered(curve, t, w, step=0.05):
    """The curve without filter, convolved with the kernel (rebinned to step for speed): IDD(z) = sum K(t) IDD0(z + t)."""
    tb = {}
    for a, b in zip(t, w):
        if b > 0: k = round(a / step); tb[k] = tb.get(k, 0.0) + b
    z0, dz, d = curve["z0"], curve["dz"], curve["d"]
    return [sum(b * _shift(z0, dz, d, z0 + i * dz + k * step) for k, b in tb.items()) for i in range(len(d))]


def ripple_table(t, w, deltas):
    """Ripple (%) per reference energy and spacing delta (mm water), with filter and without."""
    ref = reference()
    out = {}
    for e, c in ref["curves"].items():
        f = filtered(c, t, w)
        out[e] = {"filter": [sobp_ripple(c["z0"], c["dz"], f, dl) for dl in deltas],
                  "none": [sobp_ripple(c["z0"], c["dz"], c["d"], dl) for dl in deltas]}
    return out


def lateral(mean_wet):
    """Estimated extra lateral sigma (mm) at the isocentre per energy: c(E) * sqrt(mean thickness in mm water)."""
    ref = reference()
    return {e: round(c * math.sqrt(max(mean_wet, 0.0)), 2) for e, c in ref["lateral_coef"].items()}


# ------------------------------------------------------------------ manufacturing hints
def hints(d):
    """Notes for building the filter: [(level, text_en, text_de)], level 'ok', 'info' or 'warn'."""
    out = []
    rs = ridges(d)
    for i, (L, r) in enumerate(zip(d["layers"], rs)):
        tag = f"Layer {i + 1}", f"Lage {i + 1}"
        xs, zs = r["x"], r["z"]
        flats = [b[0] - a[0] for a, b in zip(list(zip(xs, zs)), list(zip(xs[1:], zs[1:]))) if abs(b[1] - a[1]) < 1e-9 and b[0] - a[0] > 1.5 * EDGE]
        gapw = L["period"] - r["width"]
        feat = min(flats + ([gapw] if gapw > 1e-6 else [])) if flats or gapw > 1e-6 else None
        if feat is not None:
            lvl = "warn" if feat < 0.2 else "info"
            out.append((lvl, f"{tag[0]}: smallest feature {feat:.3f} mm", f"{tag[1]}: kleinste Strukturbreite {feat:.3f} mm".replace(".", ",")))
        if "steps" in L:
            hs = sorted({h for _, h in L["steps"]})
            rises = [round(b - a, 4) for a, b in zip(hs, hs[1:])]
            odd = sorted({x for x in rises + ([L["base"]] if L["base"] else []) if not any(abs(x - p) < 1e-6 for p in PLATES)})
            if odd:
                v = ", ".join(f"{x:g}" for x in odd)
                out.append(("warn", f"{tag[0]}: steps of {v} mm match no standard plate ({', '.join(f'{p:g}' for p in PLATES)} mm)",
                            f"{tag[1]}: Stufen von {v} mm passen zu keiner Standardplatte ({', '.join(f'{p:g}' for p in PLATES)} mm)".replace(".", ",")))
            else:
                out.append(("ok", f"{tag[0]}: all steps match standard plates", f"{tag[1]}: alle Stufen passen zu Standardplatten"))
        else:
            out.append(("info", f"{tag[0]}: continuous profile, milled or printed", f"{tag[1]}: stetiges Profil, gefräst oder gedruckt"))
        out.append(("info", f"{tag[0]}: base plate {L['base']:g} mm" if L["base"] else f"{tag[0]}: no base plate",
                    (f"{tag[1]}: Grundplatte {L['base']:g} mm" if L["base"] else f"{tag[1]}: keine Grundplatte").replace(".", ",")))
    for _, g, aligned in group_ridges(rs):
        if not aligned or len(g) < 2: continue
        idx = sorted(rs.index(r) for r in g)
        # a layer of another orientation between two aligned ones also keeps them apart
        if any(b != a + 1 or d["layers"][b]["gap"] > 0 for a, b in zip(idx, idx[1:])):
            out.append(("warn", "Aligned layers " + ", ".join(str(k + 1) for k in idx) + " have air between them: laminated layers must touch (task 3176), only crossed ones tolerate a gap",
                        "Gleich ausgerichtete Lagen " + ", ".join(str(k + 1) for k in idx) + " haben Luft dazwischen: laminierte Lagen müssen anliegen (Task 3176), nur gekreuzte vertragen Spalt"))
    return [{"level": a, "en": b, "de": c} for a, b, c in out]


# ------------------------------------------------------------------ SVG mask for the laser lab
def svg_mask(d, i, margin=10.0, hole=3.0):
    """Top view of layer i in millimetres (1 user unit = 1 mm): stripes shaded by height, cut lines at every step,
    and four registration holes at the same place on every layer, so pins through them line the layers up."""
    L = d["layers"][i]; r = ridges(d)[i]
    size = EXTENT + 2 * margin
    xs, zs = r["x"], r["z"]
    hmax = max(zs) or 1.0
    # flat runs of the profile, as (start, end, height) within one ridge
    runs = []
    for a, b, za, zb in zip(xs, xs[1:], zs, zs[1:]):
        if b - a <= 1.5 * EDGE: continue
        runs.append((a, b, (za + zb) / 2))
    rects = []
    for k in range(r["n"] + 1):
        left = r["x0"] + k * r["pitch"]
        for a, b, h in runs:
            x1, x2 = max(left + a, -EXTENT / 2), min(left + b, EXTENT / 2)
            if x2 <= x1: continue
            g = int(235 - 175 * h / hmax)
            rects.append(f'<rect x="{x1 + EXTENT / 2 + margin:.3f}" y="{margin:.3f}" width="{x2 - x1:.3f}" height="{EXTENT:.3f}" '
                         f'fill="rgb({g},{g},{g})" stroke="#000" stroke-width="0.05"/>')
    c = size / 2
    rot = f' transform="rotate({-L["rotz"]:g} {c:.3f} {c:.3f})"' if L["rotz"] else ""
    holes = "".join(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{hole / 2:g}" fill="none" stroke="#d00" stroke-width="0.1"/>'
                    for x in (margin / 2, size - margin / 2) for y in (margin / 2, size - margin / 2))
    label = f'{d["name"]} layer {i + 1}: period {L["period"]:g} mm, height {hmax:g} mm, rotz {L["rotz"]:g} deg, base {L["base"]:g} mm'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size:g}mm" height="{size:g}mm" viewBox="0 0 {size:g} {size:g}">\n'
            f'<title>{label}</title>\n<rect x="0" y="0" width="{size:g}" height="{size:g}" fill="none" stroke="#000" stroke-width="0.1"/>\n'
            f'<g{rot}>\n' + "\n".join(rects) + f'\n</g>\n{holes}\n'
            f'<text x="{margin:.1f}" y="{size - 1.5:.1f}" font-size="2.5" font-family="sans-serif">{label}</text>\n</svg>\n')


# ------------------------------------------------------------------ library in the workspace
def library_dir(wb=None):
    wb = wb or os.environ.get("OPENTOPAS_WB_DIR")
    return os.path.join(wb, "rifi") if wb else None


def library(wb=None):
    """Saved definitions by name; a broken file is skipped."""
    p = library_dir(wb); out = {}
    if not p or not os.path.isdir(p): return out
    for f in sorted(os.listdir(p)):
        if not f.endswith(".json"): continue
        try:
            with open(os.path.join(p, f), encoding="utf-8") as fh: d = validate(json.load(fh))
            out[d["name"]] = d
        except (OSError, ValueError, TypeError, KeyError): continue
    return out


def save(d, wb=None):
    d = validate(d); p = library_dir(wb)
    if not p: raise ValueError("no workspace")
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, d["name"] + ".json"), "w", encoding="utf-8") as fh: json.dump(d, fh, indent=1, ensure_ascii=False)
    return d


def remove(name, wb=None):
    p = library_dir(wb)
    if not p or not NAME_RE.match(name): raise ValueError("bad name")
    f = os.path.join(p, name + ".json")
    if os.path.exists(f): os.remove(f)
