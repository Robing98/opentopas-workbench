"""The case file fall.json, schema fall/v1 (docs/design/fall-datei.md, section 2). Standard library only, so that the
workbench server, tools/fall_check.py and the other tools can all use it without installing anything.

A path in fall.json is relative to the case folder, with "/", never absolute and never with "..". A path that
starts with "@NAME/" lies under a place named in orte.json next to fall.json: the folders of the MRI viewer and the
plan_mc runs lie outside the case folder on the user's computer (OneDrive, the private zones), and their absolute
paths must not enter fall.json, which moves between computers. orte.json is the only file of a case that holds such
paths; it stays on the computer.
"""
import hashlib
import json
import os
import re
import time

SCHEMA = "fall/v1"
NAME = "fall.json"
ORTE = "orte.json"
MODALITIES = ("CT", "MR", "PT", "SM")
SECTIONS = ("bezug", "plan", "strukturen", "rifi", "maschine", "lieferung", "viewer", "einstellungen")
RESULT_KINDS = ("workbench", "simulator", "viewer")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


class FallError(ValueError):
    """A case file that cannot be read or does not fit fall/v1; the message names the field."""


# ------------------------------------------------------------------ paths
def path_problem(p):
    """Why `p` is no valid path of a case file, or None."""
    if not isinstance(p, str) or not p: return "leer"
    if "\\" in p: return "Backslash, erlaubt ist nur /"
    if p.startswith("/") or re.match(r"^[A-Za-z]:", p): return "absolut"
    parts = p.split("/")
    if ".." in parts: return "enthält .."
    if parts[0].startswith("@") and (len(parts[0]) < 2 or not _ID.match(parts[0][1:])): return "Ortname ungültig"
    return None


def resolve(folder, p, orte=None):
    """The local path of a case path. Raises FallError for a path that does not fit or an unknown place."""
    why = path_problem(p)
    if why: raise FallError(f"Pfad {p!r}: {why}")
    head, _, rest = p.partition("/")
    if head.startswith("@"):
        orte = load_orte(folder) if orte is None else orte
        base = orte.get(head[1:])
        if not base: raise FallError(f"Pfad {p!r}: Ort {head[1:]!r} fehlt in {ORTE}")
        return os.path.normpath(os.path.join(base, rest)) if rest else os.path.normpath(base)
    return os.path.normpath(os.path.join(folder, p))


def to_case_path(folder, local, orte):
    """A local path as a case path: relative when it lies in the case folder, else under the place in `orte` that
    holds it (the deepest one), else a new place named after its last folder, added to `orte`."""
    local = os.path.normpath(os.path.abspath(local))
    folder = os.path.normpath(os.path.abspath(folder))
    if local == folder or local.startswith(folder + os.sep):
        return os.path.relpath(local, folder).replace(os.sep, "/") if local != folder else "."
    best = None
    for name, base in orte.items():
        b = os.path.normpath(base)
        if local == b or local.startswith(b + os.sep):
            if best is None or len(b) > len(os.path.normpath(orte[best])): best = name
    if best is None:
        parent = os.path.dirname(local) if os.path.isfile(local) else local
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", os.path.basename(parent)).strip("-._").lower() or "ort"
        best, n = stem[:40], 2
        while best in orte and os.path.normpath(orte[best]) != parent: best = f"{stem[:37]}-{n}"; n += 1
        orte[best] = parent
    rel = os.path.relpath(local, os.path.normpath(orte[best])).replace(os.sep, "/")
    return f"@{best}" if rel == "." else f"@{best}/{rel}"


def load_orte(folder):
    try:
        with open(os.path.join(folder, ORTE), encoding="utf-8") as fh: d = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise FallError(f"{ORTE}: {e}")
    if not isinstance(d, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in d.items()):
        raise FallError(f"{ORTE}: erwartet {{Name: Ordner}}")
    return d


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""): h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ schema
def validate(d):
    """Problems of a parsed fall.json against fall/v1 as a list of "field: why"; empty when it fits. New fields are
    allowed (the design keeps v1 open for them); a known field with the wrong type is a problem."""
    bad = []
    add = lambda where, why: bad.append(f"{where}: {why}")
    if not isinstance(d, dict): return ["Wurzel: kein Objekt"]
    if d.get("schema") != SCHEMA: add("schema", f"erwartet {SCHEMA!r}, steht {d.get('schema')!r}")
    if not isinstance(d.get("rev"), int) or isinstance(d.get("rev"), bool) or d["rev"] < 0: add("rev", "ganze Zahl ab 0")
    if not isinstance(d.get("fall_id"), str) or not _ID.match(d.get("fall_id", "")): add("fall_id", "Buchstaben, Ziffern, . _ -")
    for s in SECTIONS:
        if s in d and d[s] is not None and not isinstance(d[s], dict): add(s, "Objekt oder null")

    def path_field(where, v, nullable=True):
        if v is None and nullable: return
        why = path_problem(v)
        if why: add(where, why)

    def sha_field(where, v):
        if v is not None and not (isinstance(v, str) and _SHA.match(v)): add(where, "SHA-256 (64 Hexzeichen) oder null")

    plan = d.get("plan") or {}
    if plan:
        path_field("plan.datei", plan.get("datei"), nullable=False); sha_field("plan.sha256", plan.get("sha256"))
        fx = plan.get("fraktionen")
        if fx is not None and (not isinstance(fx, int) or isinstance(fx, bool) or fx < 1): add("plan.fraktionen", "ganze Zahl ab 1")
        rx = plan.get("verschreibung_gy_rbe")
        if rx is not None and (not isinstance(rx, (int, float)) or isinstance(rx, bool) or rx <= 0):
            add("plan.verschreibung_gy_rbe", "Zahl über 0")
    st = d.get("strukturen") or {}
    if st:
        path_field("strukturen.datei", st.get("datei"), nullable=False); sha_field("strukturen.sha256", st.get("sha256"))
    for s, keys in (("viewer", ("seite", "daten", "skripte", "cache", "bilder")), ("einstellungen", ("ausgabe", "kennzahlen")),
                    ("rifi", ("definition",)), ("maschine", ("strahlmodell", "hlut"))):
        sec = d.get(s) or {}
        for k in keys:
            if k in sec: path_field(f"{s}.{k}", sec[k])
    bez = d.get("bezug") or {}
    names = set()
    bilder = d.get("bilder", [])
    if not isinstance(bilder, list): add("bilder", "Liste"); bilder = []
    for i, b in enumerate(bilder):
        w = f"bilder[{i}]"
        if not isinstance(b, dict): add(w, "Objekt"); continue
        if not isinstance(b.get("serie"), str) or not _ID.match(b.get("serie", "")): add(f"{w}.serie", "Name aus Buchstaben, Ziffern, . _ -")
        elif b["serie"] in names: add(f"{w}.serie", f"doppelt: {b['serie']}")
        else: names.add(b["serie"])
        if b.get("modalitaet") not in MODALITIES: add(f"{w}.modalitaet", f"eine von {', '.join(MODALITIES)}")
        if b.get("datum") is not None and not (isinstance(b["datum"], str) and _DATE.match(b["datum"])): add(f"{w}.datum", "JJJJ-MM-TT")
        path_field(f"{w}.ordner", b.get("ordner"), nullable=False)
        path_field(f"{w}.nach_bezug", b.get("nach_bezug"))
        g = b.get("guete")
        if g is not None:
            if not isinstance(g, dict) or not isinstance(g.get("ok"), bool): add(f"{w}.guete", "Objekt mit ok: true/false")
            elif g.get("seitenpruefung") is not None:
                sp = g["seitenpruefung"]
                if not isinstance(sp, dict) or not isinstance(sp.get("ok"), bool): add(f"{w}.guete.seitenpruefung", "Objekt mit ok")
        if b.get("modalitaet") == "PT" and b.get("pet") is not None:
            pet = b["pet"]
            if not isinstance(pet, dict): add(f"{w}.pet", "Objekt")
            elif isinstance(pet.get("referenz"), dict): path_field(f"{w}.pet.referenz.maske", pet["referenz"].get("maske"))
    if bez:
        if bez.get("serie") is not None and bez["serie"] not in names: add("bezug.serie", f"keine Serie {bez['serie']!r} in bilder")
    zl = d.get("zeitleiste", [])
    if not isinstance(zl, list): add("zeitleiste", "Liste"); zl = []
    for i, z in enumerate(zl):
        w = f"zeitleiste[{i}]"
        if not isinstance(z, dict): add(w, "Objekt"); continue
        if not (isinstance(z.get("datum"), str) and _DATE.match(z["datum"])): add(f"{w}.datum", "JJJJ-MM-TT")
        if not isinstance(z.get("art"), str) or not z["art"]: add(f"{w}.art", "Text")
        if "ausgangszustand" in z and not isinstance(z["ausgangszustand"], bool): add(f"{w}.ausgangszustand", "true/false")
        path_field(f"{w}.befund", z.get("befund"))
    erg = d.get("ergebnisse", {})
    if not isinstance(erg, dict): add("ergebnisse", "Objekt"); erg = {}
    for kind in RESULT_KINDS:
        lst = erg.get(kind, [])
        if not isinstance(lst, list): add(f"ergebnisse.{kind}", "Liste"); continue
        ids = set()
        for i, r in enumerate(lst):
            w = f"ergebnisse.{kind}[{i}]"
            if not isinstance(r, dict): add(w, "Objekt"); continue
            if not isinstance(r.get("id"), str) or not _ID.match(r.get("id", "")): add(f"{w}.id", "Name")
            elif r["id"] in ids: add(f"{w}.id", f"doppelt: {r['id']}")
            else: ids.add(r["id"])
            path_field(f"{w}.ordner", r.get("ordner")); path_field(f"{w}.datei", r.get("datei"))
            for sec in ("eingaben", "dateien"):
                m = r.get(sec, {})
                if not isinstance(m, dict): add(f"{w}.{sec}", "Objekt"); continue
                for k, v in m.items(): sha_field(f"{w}.{sec}.{k}", v)
            for k in r.get("dateien", {}) if isinstance(r.get("dateien"), dict) else ():
                path_field(f"{w}.dateien.{k}", k)
    return bad


# ------------------------------------------------------------------ read and write
def load(folder):
    """fall.json of a case folder, parsed and checked against the schema. Raises FallError, naming each problem."""
    p = os.path.join(folder, NAME)
    try:
        with open(p, encoding="utf-8") as fh: d = json.load(fh)
    except FileNotFoundError:
        raise FallError(f"{p}: keine {NAME}")
    except (OSError, ValueError) as e:
        raise FallError(f"{p}: {e}")
    bad = validate(d)
    if bad: raise FallError("; ".join(bad))
    return d


def new(fall_id):
    return {"schema": SCHEMA, "rev": 0, "fall_id": fall_id, "bilder": [], "zeitleiste": [],
            "ergebnisse": {k: [] for k in RESULT_KINDS}}


def _write_json(path, d):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1); fh.write("\n")
    os.replace(tmp, path)


def save_orte(folder, orte):
    _write_json(os.path.join(folder, ORTE), dict(sorted(orte.items())))


def update(folder, change, tries=5, create=None):
    """Reads fall.json, applies change(d) (which edits d in place), raises rev and writes it back through a temporary
    file. When another tool wrote in between (rev moved), it reads again and repeats, so no tool overwrites another's
    section. `create` is a fall_id for a case folder without fall.json."""
    p = os.path.join(folder, NAME)
    for _ in range(tries):
        if create and not os.path.exists(p): d = new(create)
        else: d = load(folder)
        rev = d["rev"]
        change(d)
        d["rev"] = rev + 1
        bad = validate(d)
        if bad: raise FallError("; ".join(bad))
        try:
            with open(p, encoding="utf-8") as fh: now = json.load(fh).get("rev")
        except FileNotFoundError:
            now = None if create else rev
        if now not in (rev, None) or (now is None and not create): continue
        _write_json(p, d)
        return d
    raise FallError(f"{p}: rev ändert sich ständig, nicht geschrieben")


# ------------------------------------------------------------------ check
def check(folder, hashes=True):
    """The report of fall_check.py: {"fehler": [...], "warnungen": [...], "ergebnisse": [{art, id, zustand, grund}]}.
    zustand is aktuell, veraltet (an input changed since the result was made) or kaputt (a file it names is gone or
    changed). `hashes=False` skips reading files, for a quick look from the web UI."""
    rep = {"fehler": [], "warnungen": [], "ergebnisse": [], "lesbar": True}
    try:
        d = load(folder); orte = load_orte(folder)
    except FallError as e:
        rep["fehler"].append(str(e)); rep["lesbar"] = False; return rep

    def local(p, where):
        try: return resolve(folder, p, orte)
        except FallError as e: rep["fehler"].append(f"{where}: {e}"); return None

    current = {}
    for sec in ("plan", "strukturen"):
        s = d.get(sec) or {}
        if not s: continue
        lp = local(s["datei"], f"{sec}.datei")
        if lp is None: continue
        if not os.path.isfile(lp): rep["fehler"].append(f"{sec}.datei: fehlt ({s['datei']})"); continue
        if hashes:
            h = sha256(lp); current[sec] = h
            if s.get("sha256") and s["sha256"] != h:
                rep["warnungen"].append(f"{sec}.datei: Hash weicht von {sec}.sha256 ab, die Datei wurde geändert")
    for sec, keys in (("rifi", ("definition",)), ("maschine", ("strahlmodell", "hlut"))):
        s = d.get(sec) or {}
        for k in keys:
            if s.get(k):
                lp = local(s[k], f"{sec}.{k}")
                if lp and not os.path.isfile(lp): rep["fehler"].append(f"{sec}.{k}: fehlt ({s[k]})")
                elif lp and hashes and k == keys[0]: current[sec] = sha256(lp)
    for sec, keys in (("viewer", ("seite", "daten", "skripte", "cache", "bilder")), ("einstellungen", ("ausgabe", "kennzahlen"))):
        s = d.get(sec) or {}
        for k in keys:
            if s.get(k) and not (sec == "einstellungen"):   # outputs need not exist yet
                lp = local(s[k], f"{sec}.{k}")
                if lp and not os.path.exists(lp): rep["fehler"].append(f"{sec}.{k}: fehlt ({s[k]})")
            elif s.get(k): local(s[k], f"{sec}.{k}")
    bezug = (d.get("bezug") or {}).get("serie")
    for b in d.get("bilder", []):
        lp = local(b["ordner"], f"bilder {b['serie']}.ordner")
        if lp and not os.path.isdir(lp): rep["fehler"].append(f"bilder {b['serie']}.ordner: fehlt ({b['ordner']})")
        if b["serie"] != bezug and b["modalitaet"] != "SM" and not b.get("nach_bezug"):
            rep["warnungen"].append(f"bilder {b['serie']}: noch keine Transformation zum Bezug (wird beim Lauf berechnet)")
        elif b.get("nach_bezug"):
            tp = local(b["nach_bezug"], f"bilder {b['serie']}.nach_bezug")
            if tp and not os.path.isfile(tp): rep["fehler"].append(f"bilder {b['serie']}.nach_bezug: fehlt ({b['nach_bezug']})")
        g = b.get("guete") or {}
        if g and g.get("ok") is False: rep["warnungen"].append(f"bilder {b['serie']}: Registrierung nicht ok")
        sp = g.get("seitenpruefung") or {}
        if sp and sp.get("ok") is False: rep["warnungen"].append(f"bilder {b['serie']}: Seitenprüfung nicht bestanden")
    for z in d.get("zeitleiste", []):
        if z.get("befund"):
            lp = local(z["befund"], f"zeitleiste {z['datum']}.befund")
            if lp and not os.path.isfile(lp): rep["fehler"].append(f"zeitleiste {z['datum']}.befund: fehlt")
    for kind in RESULT_KINDS:
        for r in d["ergebnisse"].get(kind, []) if isinstance(d.get("ergebnisse"), dict) else []:
            state, why = "aktuell", []
            base = local(r["ordner"], f"ergebnisse.{kind} {r['id']}.ordner") if r.get("ordner") else folder
            if r.get("ordner") and base and not os.path.isdir(base): state, why = "kaputt", [f"Ordner fehlt: {r['ordner']}"]
            if r.get("datei"):
                fp = local(r["datei"], f"ergebnisse.{kind} {r['id']}.datei")
                if fp and not os.path.isfile(fp): state = "kaputt"; why.append(f"Datei fehlt: {r['datei']}")
            for name, h in (r.get("dateien") or {}).items():
                if base is None or state == "kaputt" and not os.path.isdir(base): break
                fp = os.path.join(base, name)
                if not os.path.isfile(fp): state = "kaputt"; why.append(f"fehlt: {name}")
                elif hashes and h and sha256(fp) != h: state = "kaputt"; why.append(f"geändert: {name}")
            if state == "aktuell" and hashes:
                for k, h in (r.get("eingaben") or {}).items():
                    if k in current and h and current[k] != h: state = "veraltet"; why.append(f"Eingabe {k} geändert")
            rep["ergebnisse"].append({"art": kind, "id": r["id"], "zustand": state, "grund": "; ".join(why)})
    plan = d.get("plan") or {}
    parts = [s.lower() for s in os.path.realpath(folder).split(os.sep) if s]
    if plan and not plan.get("oeffentlich") and any(a == "workspace" and b == "offen" for a, b in zip(parts, parts[1:])):
        rep["warnungen"].append("Fall liegt in workspace/offen, aber plan.oeffentlich ist nicht true: ein klinischer Plan gehört in die Zone lokal")
    return rep


def ok(rep):
    """rc 0 only without errors and without a broken result."""
    return not rep["fehler"] and not any(r["zustand"] == "kaputt" for r in rep["ergebnisse"])


def summary(d):
    """The few facts the web UI lists per case, nothing read from the images."""
    plan = d.get("plan") or {}
    laeufe = [r["id"] for r in d.get("ergebnisse", {}).get("workbench", [])]
    return {"fall_id": d["fall_id"], "rev": d["rev"], "fraktionen": plan.get("fraktionen"),
            "verschreibung_gy_rbe": plan.get("verschreibung_gy_rbe"), "laeufe": laeufe,
            "pet": [b["serie"] for b in d.get("bilder", []) if b.get("modalitaet") == "PT"],
            "geaendert": d.get("geaendert")}


def stamp(d):
    d["geaendert"] = time.strftime("%Y-%m-%dT%H:%M:%S")
