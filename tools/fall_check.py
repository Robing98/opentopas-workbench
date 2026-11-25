#!/usr/bin/env python3
"""Checks a case folder against fall/v1: fall_check.py FALLORDNER [--schnell] [--json].

Reports the schema, every path (relative, inside the case folder or under a place of orte.json), every named file and
its hash, and per result aktuell, veraltet or kaputt. Ends with rc 0 only when nothing is broken; rc 1 when a file is
missing or a result is broken, rc 2 when fall.json itself does not fit the schema. Standard library only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opentopas_workbench import fall  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ordner", help="Fallordner mit fall.json")
    ap.add_argument("--schnell", action="store_true", help="ohne Hashes: nur Schema, Pfade und vorhandene Dateien")
    ap.add_argument("--json", action="store_true", help="Bericht als JSON")
    a = ap.parse_args(argv)
    rep = fall.check(a.ordner, hashes=not a.schnell)
    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    else:
        for e in rep["fehler"]: print(f"FEHLER   {e}")
        for w in rep["warnungen"]: print(f"WARNUNG  {w}")
        for r in rep["ergebnisse"]:
            print(f"{r['zustand'].upper():8} {r['art']}/{r['id']}" + (f"  ({r['grund']})" if r["grund"] else ""))
        print("OK" if fall.ok(rep) else "KAPUTT")
    if fall.ok(rep): return 0
    return 1 if rep["lesbar"] else 2


if __name__ == "__main__":
    sys.exit(main())
