"""The physics of a case in both codes, side by side: the TOPAS physics list and production cuts against the FLUKA
DEFAULTS set and its energy thresholds. Pure text parsing; the server reads the files.

TOPAS cuts by range (a length that Geant4 turns into an energy per material), FLUKA by energy, so a row puts the two
next to each other without converting: the conversion depends on the material and would claim more than the file says.
"""
import re

# The defaults of the OpenTOPAS parameter defaults page (opentopas.readthedocs.io, parameters/defaults.html, read
# 2026-11-23). It names no default production cut, so an unset cut is shown as unset, never as a number.
TOPAS_DEFAULTS = {
    "listname": "Default",
    "type": "Geant4_Modular",
    "modules": ["g4em-standard_opt4", "g4h-phy_QGSP_BIC_HP", "g4decay", "g4ion-binarycascade", "g4h-elastic_HP",
                "g4stopping"],
    "emrangemin": "100 eV",
    "emrangemax": "600 MeV",
}
CUT_KEYS = {"electron": "cutforelectron", "positron": "cutforpositron", "gamma": "cutforgamma",
            "proton": "cutforproton"}
ROWS = ["list", "modules", "e_prod", "g_prod", "delta", "e_trans", "g_trans", "had_trans", "emrange"]

_PARAM = re.compile(r"^\s*(\w+):\s*Ph/(\S+?)\s*=\s*(.+?)\s*$", re.I)


def _strip(line):
    """A TOPAS line without its comment; a # inside quotes stays."""
    out, quoted = [], False
    for ch in line:
        if ch == '"': quoted = not quoted
        if ch == "#" and not quoted: break
        out.append(ch)
    return "".join(out)


def _unquote(v):
    return " ".join(re.findall(r'"([^"]*)"', v)) or v.strip()


def topas_physics(text):
    """The Ph/ parameters of a TOPAS file: {name: (value, line)}, names lower case as TOPAS compares them, plus whether
    the file includes others (whose Ph/ parameters stay unknown)."""
    params, includes = {}, False
    for n, raw in enumerate(text.splitlines(), 1):
        line = _strip(raw)
        if line.strip().lower().startswith("includefile"): includes = True; continue
        m = _PARAM.match(line)
        if m: params[m.group(2).lower()] = (m.group(3).strip(), n)
    return params, includes


def topas_column(text):
    """Row key to {"v": text, "src": "file" | "default" | "none", "line": n}."""
    params, includes = topas_physics(text)
    lst = _unquote(params["listname"][0]) if "listname" in params else TOPAS_DEFAULTS["listname"]

    def get(name):
        p = params.get(f"{lst.lower()}/{name}")
        return (p[0], p[1]) if p else (None, None)

    def cell(v, line, default=None):
        if v is not None: return {"v": v, "src": "file", "line": line}
        if default is not None: return {"v": default, "src": "default"}
        return {"v": "not set", "src": "none"}

    typ, tl = get("type")
    lv = f"{lst} ({_unquote(typ) if typ else TOPAS_DEFAULTS['type']})"
    col = {"list": {"v": lv, "src": "file" if ("listname" in params or typ) else "default",
                    **({"line": params["listname"][1]} if "listname" in params else {})}}
    mods, ml = get("modules")
    if mods is not None:
        col["modules"] = cell(", ".join(re.findall(r'"([^"]*)"', mods)) or mods, ml)
    else:
        col["modules"] = cell(None, None, ", ".join(TOPAS_DEFAULTS["modules"]) if lst.lower() == "default" else None)
    all_cut, al = get("cutforallparticles")

    def cut(particle):
        v, line = get(CUT_KEYS[particle])
        if v is not None: return cell(_unit(v) + " range", line)
        if all_cut is not None: return cell(_unit(all_cut) + " range", al)
        return cell(None, None)

    col["e_prod"], col["g_prod"] = cut("electron"), cut("gamma")
    # Geant4 produces delta rays above the electron production cut: the same number, stated as such
    col["delta"] = dict(col["e_prod"], v=col["e_prod"]["v"] + " (electron cut)") if col["e_prod"]["src"] != "none" \
        else {"v": "electron cut, not set", "src": "none"}
    for k in ("e_trans", "g_trans", "had_trans"): col[k] = {"v": "no parameter", "src": "none"}
    lo, ll = get("emrangemin")
    hi, hl = get("emrangemax")
    col["emrange"] = {"v": f"{_unit(lo) if lo else TOPAS_DEFAULTS['emrangemin']} to "
                           f"{_unit(hi) if hi else TOPAS_DEFAULTS['emrangemax']}",
                      "src": "file" if (lo or hi) else "default", **({"line": hl or ll} if (lo or hi) else {})}
    return col, includes


def _unit(v):
    """ "0.05 mm" and "1000. MeV" as written, with the number tidied."""
    parts = v.split()
    try: return f"{float(parts[0]):g} {' '.join(parts[1:])}".strip()
    except (ValueError, IndexError): return v


# ------------------------------------------------------------------ FLUKA
def cards(text):
    """The cards of a fixed-format FLUKA input: (name, [WHAT(1)..WHAT(6) as float or None], SDUM, line). Comment lines
    (* or !) and the geometry between GEOBEGIN and GEOEND are skipped. A FREE card stops the reading: free format is
    not read, and the caller says so."""
    out, geo = [], False
    for n, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        if not raw.strip() or raw[0] in "*!": continue
        name = raw[:8].strip().upper()
        if geo:
            if name == "GEOEND": geo = False
            else: continue
        if name == "GEOBEGIN": geo = True; continue
        if name == "FREE": out.append(("FREE", [], "", n)); break
        what = []
        for k in range(6):
            f = raw[10 + 10 * k:20 + 10 * k].strip()
            # a Fortran D exponent reads like E; 0.0 means "ignored" for every card read here, so it counts as blank
            try: v = float(f.upper().replace("D", "E")) if f else None
            except ValueError: v = None
            what.append(v if v else None)
        out.append((name, what, raw[70:78].strip(), n))
    return out


def _energy(gev, kinetic_if_negative=True):
    """A FLUKA threshold in GeV as keV or MeV. EMFCUT counts a negative value as kinetic energy and a positive one as
    total energy, rest mass included; the text says which."""
    if gev is None: return None
    kind = ""
    if kinetic_if_negative: kind = " kinetic" if gev < 0 else " total"
    e = abs(gev) * 1e6   # keV
    return (f"{e / 1000:g} MeV" if e >= 1000 else f"{e:g} keV") + kind


def fluka_column(text):
    """Row key to {"v", "src", "line"}, and whether the input is in free format (then nothing below FREE is read)."""
    col = {k: {"v": "not set", "src": "none"} for k in ROWS}
    col["modules"] = {"v": "no modules: models chosen by FLUKA", "src": "none"}
    col["emrange"] = {"v": "no parameter", "src": "none"}
    free = False
    for name, w, sdum, n in cards(text):
        if name == "FREE": free = True; break
        if name == "DEFAULTS":
            col["list"] = {"v": f"DEFAULTS {sdum or 'NEW-DEFA'}", "src": "file", "line": n}
        elif name == "EMFCUT" and sdum.upper().startswith("PROD-CUT"):
            if w[0] is not None: col["e_prod"] = {"v": _energy(w[0]), "src": "file", "line": n}
            if w[1] is not None: col["g_prod"] = {"v": _energy(w[1], False), "src": "file", "line": n}
        elif name == "EMFCUT" and not sdum:
            # the first transport card wins the row; later ones for other regions repeat it in the writer's inputs
            if w[0] is not None and col["e_trans"]["src"] == "none":
                col["e_trans"] = {"v": _energy(w[0]), "src": "file", "line": n}
            if w[1] is not None and col["g_trans"]["src"] == "none":
                col["g_trans"] = {"v": _energy(w[1], False), "src": "file", "line": n}
        elif name == "DELTARAY" and w[0] is not None and col["delta"]["src"] == "none":
            col["delta"] = {"v": _energy(w[0], False), "src": "file", "line": n}
        elif name == "PART-THR" and w[0] is not None and col["had_trans"]["src"] == "none":
            # PART-THR reads a negative WHAT(1) as kinetic energy and a positive one as momentum (GeV/c)
            v = _energy(w[0], False) + " kinetic" if w[0] < 0 else f"{w[0] * 1e3:g} MeV/c momentum"
            col["had_trans"] = {"v": v, "src": "file", "line": n}
    if col["list"]["src"] == "none":
        col["list"] = {"v": "no DEFAULTS card (NEW-DEFAults)", "src": "default"}
    if col["had_trans"]["src"] == "none":
        by = "NEW-DEFAults" if col["list"]["src"] == "default" else col["list"]["v"]
        col["had_trans"] = {"v": f"set by {by}", "src": "default"}
    return col, free


def compare(topas_text=None, fluka_text=None):
    """The rows of both codes. A code without a text shows its defaults (TOPAS) or nothing (FLUKA)."""
    t, includes = topas_column(topas_text if topas_text is not None else "")
    f, free = fluka_column(fluka_text) if fluka_text is not None else ({}, False)
    rows = [{"key": k, "topas": t.get(k), "fluka": f.get(k)} for k in ROWS]
    return {"rows": rows, "includes": includes, "free": free}
