#!/usr/bin/env python3
"""Moves the thesis workspace into three zones by top folder: offen, robin, lokal.

offen  public and synthetic data and the TOPAS results from them.
robin  only kennzahlen.json files from Robin's own local scripts (starts empty).
lokal  patient data and all data of MIT, with every number derived from it.

The zone of a file follows from its path and name only; the script never opens a file. A folder that goes to lokal as a
whole is moved in one piece and only counted, its contents are never listed. Anything that matches no offen rule but
whose name hints at MIT base data, machine data, HLUTs or measurements goes to lokal and is marked as unsure.

Dry run by default:  python tools/zonen_umzug.py WORKSPACE
Move:                python tools/zonen_umzug.py WORKSPACE --apply
The move writes WORKSPACE/offen/ZONEN_UMZUG_<date>.txt: offen moves by path, lokal moves only as counts.
Standard library only, Python 3.9 and later.
"""
import argparse, datetime, fnmatch, json, os, re, shutil, sys

ZONES = ("offen", "robin", "lokal")
# Never moved: the workbench keeps its state at the workspace root.
KEEP = {".workbench"}
# Top folders that go to lokal as a whole.
# _trash is where the workbench puts deleted files, so it may hold anything.
LOKAL_WHOLE_TOP = {"patient", "_to_delete", "_trash"}
# Folders anywhere below the workspace that go to lokal as a whole (relative path or bare folder name).
LOKAL_WHOLE_DIRS = {"plan_peaks/kalibriert"}
LOKAL_DIR_NAMES = {"vergleich_topas"}
# Single files that go to lokal (glob on the path relative to the workspace).
LOKAL_FILES = ("plan_peaks/protons_MIT_*.mat", "ba_rifi/messenergien/IDD_mess_*")
# Name hints for MIT base data, machine data, HLUTs and measurements; code and reports are exempt, they hold no data.
HINT = re.compile(r"(^|[^a-z])(mit|hlut|machine|maschine|basedata|basisdaten)([^a-z]|$)|kalib|_mess_|^mess_", re.I)
CODE_EXT = {".py", ".m", ".bat", ".cmd", ".sh", ".ps1", ".md", ".pyc"}
# Reasons for unsure decisions: the top folder already marks the whole tree as lokal, but Robin did not name it.
UNSURE_WHOLE_TOP = {"_to_delete", "_trash"}


def low(s): return s.casefold()   # Windows and /mnt/c ignore case, so the rules do too


def _hinted(name):
    # Output of compare_idd_matrad.py compares with the matRad machine of MIT, so even its report counts as MIT-derived.
    if low(name).startswith("vergleich_topas") and low(name) not in LOKAL_DIR_NAMES: return True
    return os.path.splitext(name)[1].lower() not in CODE_EXT and bool(HINT.search(name))


def _count(path):
    """Number of files and bytes below path, from metadata only."""
    if os.path.isfile(path): return 1, os.path.getsize(path)
    n = size = 0
    for d, _dirs, files in os.walk(path):
        for f in files:
            try: size += os.path.getsize(os.path.join(d, f))
            except OSError: pass
            n += 1
    return n, size


def plan(ws):
    """Planned moves as dicts: src, dst (relative to ws), zone, files, bytes, whole (lokal folder moved unseen), unsure."""
    moves = []

    def add(rel, zone, whole=False, unsure=False):
        n, size = _count(os.path.join(ws, rel))
        moves.append({"src": rel, "dst": zone + "/" + rel, "zone": zone, "files": n, "bytes": size,
                      "whole": whole, "unsure": unsure})

    def walk(rel):
        """Splits a folder that holds lokal parts; returns True if everything below rel is offen."""
        full = os.path.join(ws, rel)
        entries = sorted(os.listdir(full))
        decisions = []
        for e in entries:
            r = rel + "/" + e
            p = os.path.join(full, e)
            if os.path.isdir(p) and not os.path.islink(p):
                if low(r) in LOKAL_WHOLE_DIRS or low(e) in LOKAL_DIR_NAMES: decisions.append((r, "lokal-whole"))
                elif _hinted(e): decisions.append((r, "lokal-unsure"))
                elif walk_needed(r): decisions.append((r, "split"))
                else: decisions.append((r, "offen"))
            else:
                if any(fnmatch.fnmatchcase(low(r), low(g)) for g in LOKAL_FILES): decisions.append((r, "lokal"))
                elif _hinted(e): decisions.append((r, "lokal-unsure"))
                else: decisions.append((r, "offen"))
        return decisions

    def walk_needed(rel):
        """True if some path below rel goes to lokal (checked by names, folders that go whole are not entered)."""
        for d, dirs, files in os.walk(os.path.join(ws, rel)):
            drel = os.path.relpath(d, ws).replace(os.sep, "/")
            for x in dirs:
                r = drel + "/" + x
                if low(r) in LOKAL_WHOLE_DIRS or low(x) in LOKAL_DIR_NAMES or _hinted(x): return True
            for f in files:
                r = drel + "/" + f
                if any(fnmatch.fnmatchcase(low(r), low(g)) for g in LOKAL_FILES) or _hinted(f): return True
        return False

    def emit(rel):
        for r, how in walk(rel):
            if how == "offen": add(r, "offen")
            elif how == "lokal": add(r, "lokal")
            elif how == "lokal-whole": add(r, "lokal", whole=True)
            elif how == "lokal-unsure": add(r, "lokal", whole=os.path.isdir(os.path.join(ws, r)), unsure=True)
            else: emit(r)

    for e in sorted(os.listdir(ws)):
        p = os.path.join(ws, e)
        if low(e) in ZONES and e not in ZONES: raise SystemExit(f"Ordner {e} gleicht einer Zone bis auf die Schreibung, bitte erst umbenennen.")
        if e in ZONES or low(e) in KEEP: continue
        if low(e) in LOKAL_WHOLE_TOP: add(e, "lokal", whole=True, unsure=low(e) in UNSURE_WHOLE_TOP)
        elif os.path.isdir(p) and not os.path.islink(p):
            if _hinted(e): add(e, "lokal", whole=True, unsure=True)
            elif walk_needed(e): emit(e)
            else: add(e, "offen")
        elif _hinted(e): add(e, "lokal", unsure=True)
        else: add(e, "offen")
    return moves


def mb(n): return f"{n / 1e6:.1f} MB"


def show(moves, out=sys.stdout):
    for m in moves:
        flag = "  (unsicher, bitte pruefen)" if m["unsure"] else ""
        whole = "  (ganzer Ordner, Inhalt nicht gelistet)" if m["whole"] else ""
        out.write(f"{m['zone']:5}  {m['src']}  ->  {m['dst']}  {m['files']} Dateien, {mb(m['bytes'])}{whole}{flag}\n")
    for z in ZONES:
        sel = [m for m in moves if m["zone"] == z]
        if sel: out.write(f"Summe {z}: {len(sel)} Verschiebungen, {sum(m['files'] for m in sel)} Dateien, "
                          f"{mb(sum(m['bytes'] for m in sel))}\n")
    unsure = [m for m in moves if m["unsure"]]
    if unsure: out.write(f"Unsicher und darum nach lokal: {len(unsure)}, oben markiert.\n")


def check(ws, moves):
    """Refuses a move whose target exists already, so nothing is overwritten."""
    clash = [m["dst"] for m in moves if os.path.lexists(os.path.join(ws, m["dst"]))]
    if clash: raise SystemExit("Ziel existiert schon, nichts verschoben: " + ", ".join(clash[:5]))


def apply(ws, moves, today=None):
    check(ws, moves)
    if os.path.exists(os.path.join(ws, ".workbench", "server.json")):
        raise SystemExit("Eine Workbench laeuft fuer diesen Ordner (.workbench/server.json). Erst beenden, dann verschieben.")
    for z in ZONES: os.makedirs(os.path.join(ws, z), exist_ok=True)
    done = []
    try:
        for m in moves:
            dst = os.path.join(ws, m["dst"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(os.path.join(ws, m["src"]), dst)
            done.append(m)
    finally:
        # Written also after a failed move, so the record names every move that happened.
        path = report(ws, done, today, partial=len(done) < len(moves))
    # Folders emptied by a split stay behind; remove them only when nothing is left in them.
    for e in sorted({m["src"].split("/")[0] for m in moves}):
        p = os.path.join(ws, e)
        if os.path.isdir(p):
            for d, _dirs, _files in sorted(os.walk(p, topdown=False), key=lambda t: -len(t[0])):
                try: os.rmdir(d)
                except OSError: pass
    migrate_workbench(ws, moves)
    return path


def _moved(rel, moves):
    """New path of a workspace path after the moves, and its zone, or (rel, None) when it did not move."""
    for m in moves:
        if rel == m["src"] or rel.startswith(m["src"] + "/"): return m["dst"] + rel[len(m["src"]):], m["zone"]
    return rel, None


def migrate_workbench(ws, moves):
    """Job records name the old paths. A job of a file that went to lokal takes its record and log along to
    lokal/.workbench, the rest get their new paths. The result cache is keyed by old paths and is dropped (the
    workbench rebuilds it)."""
    wb = os.path.join(ws, ".workbench")
    try:
        with open(os.path.join(wb, "jobs.json"), encoding="utf-8") as fh: jobs = json.load(fh)
    except (OSError, ValueError): jobs = None
    if jobs is not None:
        lwb = os.path.join(ws, "lokal", ".workbench")
        try:
            with open(os.path.join(lwb, "jobs.json"), encoding="utf-8") as fh: lok = json.load(fh)
        except (OSError, ValueError): lok = []
        shared = []
        for j in jobs:
            j["file"], zone = _moved(j.get("file", ""), moves)
            j["outputs"] = [_moved(o, moves)[0] for o in j.get("outputs", [])]
            if zone != "lokal": shared.append(j); continue
            if j.get("log", "").startswith(".workbench/"):
                src = os.path.join(ws, j["log"]); new = "lokal/" + j["log"]
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(os.path.join(ws, new)), exist_ok=True)
                    shutil.move(src, os.path.join(ws, new))
                j["log"] = new
            lok.append(j)
        if lok:
            os.makedirs(lwb, exist_ok=True)
            with open(os.path.join(lwb, "jobs.json"), "w", encoding="utf-8") as fh: json.dump(sorted(lok, key=lambda j: j["id"]), fh, indent=0)
        with open(os.path.join(wb, "jobs.json"), "w", encoding="utf-8") as fh: json.dump(shared, fh, indent=0)
    try: os.remove(os.path.join(wb, "index.json"))
    except OSError: pass


def report(ws, moves, today=None, partial=False):
    today = today or datetime.date.today().isoformat()
    path = os.path.join(ws, "offen", f"ZONEN_UMZUG_{today}.txt")
    lok = [m for m in moves if m["zone"] == "lokal"]
    # Appended, so a second run on the same day keeps the record of the first.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"Umzug in die Datenzonen am {today}, {datetime.datetime.now():%H:%M}" + (", ABGEBROCHEN" if partial else "") + "\n\noffen:\n")
        for m in moves:
            if m["zone"] == "offen": fh.write(f"  {m['src']} -> {m['dst']}  {m['files']} Dateien, {mb(m['bytes'])}\n")
        fh.write(f"\nlokal: {len(lok)} Verschiebungen, {sum(m['files'] for m in lok)} Dateien, "
                 f"{mb(sum(m['bytes'] for m in lok))}, davon {sum(m['unsure'] for m in lok)} unsicher\n\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Workspace in die Zonen offen, robin, lokal verschieben (Probelauf ohne --apply).")
    ap.add_argument("workspace"); ap.add_argument("--apply", action="store_true", help="wirklich verschieben")
    a = ap.parse_args(argv)
    ws = os.path.abspath(a.workspace)
    if not os.path.isdir(ws): raise SystemExit(f"Kein Ordner: {ws}")
    moves = plan(ws)
    show(moves)
    if not moves: print("Nichts zu verschieben."); return 0
    if not a.apply:
        check(ws, moves)
        print("Probelauf, nichts verschoben. Mit --apply verschieben."); return 0
    print("Bericht:", apply(ws, moves)); return 0


if __name__ == "__main__":
    sys.exit(main())
