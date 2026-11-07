"""Runtime of a spot-by-spot field split into start-up, cost per spot and cost per history (part D).

Each spot is one TOPAS run (time feature step), so a field costs  t = start + per_spot * spots + per_history * N.
Three short runs of the first layers of a matRad field fix the three numbers:
S spots with N histories, S spots with k*N histories, and about 3 S spots with N histories.

    python3 runtime_split.py write OUTDIR CHAINDIR [--spots S] [--histories N] [--factor k]
    python3 runtime_split.py check OUTDIR --workspace WS

CHAINDIR is a directory written by public_chain.py write (its patient.txt, CT and chain.json are reused).
Wall times come from the workbench record (.workbench/jobs.json, field "seconds"). Queue the files with wbqueue.py
while no other TOPAS run is active, else the times mean nothing.
"""
import argparse, dataclasses, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from plan_mc import beam_model, matrad, topas_writer as tw  # noqa: E402


def first_layers(field, spots):
    """The field cut to its first layers holding at least `spots` spots."""
    layers, n = [], 0
    for l in field.layers:
        if n >= spots: break
        layers.append(l); n += len(l.x)
    return dataclasses.replace(field, layers=layers), n


def write(out, chain, spots, histories, factor):
    out, chain = os.path.abspath(out), os.path.abspath(chain)
    os.makedirs(out, exist_ok=True)
    c = json.load(open(os.path.join(chain, "chain.json")))
    r = matrad.read(c["mat"])
    model = beam_model.generic()
    pat = os.path.join(chain, "patient.txt")
    runs = []
    for name, s, h in (("a", spots, histories), ("b", spots, histories * factor), ("c", 3 * spots, histories)):
        f, n = first_layers(r["plan"].fields[0], s)
        text, meta = tw.field_file(r["plan"], f, model, {}, pat, h, out_name=f"run_{name}")
        open(os.path.join(out, f"run_{name}.txt"), "w").write(text)
        runs.append({"name": name, "spots": n, "histories": meta["histories"]})
    json.dump({"chain": chain, "runs": runs}, open(os.path.join(out, "runs.json"), "w"), indent=1)
    for x in runs: print(f"run_{x['name']}.txt: {x['spots']} spots, {x['histories']} histories")


def seconds(ws, path):
    """Wall time [s] of the newest finished workbench run of `path`, or None. Skips patient runs."""
    rel = os.path.relpath(path, ws).replace(os.sep, "/")
    if rel.startswith("..") or rel.startswith("patient/"): return None
    try:
        jobs = json.load(open(os.path.join(ws, ".workbench", "jobs.json")))
    except (OSError, ValueError):
        return None
    best = None
    for j in jobs.get("jobs", jobs) if isinstance(jobs, dict) else jobs:
        if j.get("file") == rel and j.get("status") == "done" and isinstance(j.get("seconds"), (int, float)):
            best = float(j["seconds"])
    return best


def split(runs):
    """(start, per_spot, per_history) in s from runs [{"spots", "histories", "seconds"}] by least squares."""
    a = np.array([[1.0, r["spots"], r["histories"]] for r in runs])
    t = np.array([r["seconds"] for r in runs])
    sol, *_ = np.linalg.lstsq(a, t, rcond=None)
    return tuple(float(v) for v in sol)


def check(out, ws):
    out = os.path.abspath(out)
    d = json.load(open(os.path.join(out, "runs.json")))
    for r in d["runs"]:
        r["seconds"] = seconds(ws, os.path.join(out, f"run_{r['name']}.txt"))
        if r["seconds"] is None: raise SystemExit(f"run_{r['name']}.txt has no finished workbench run")
    start, per_spot, per_hist = split(d["runs"])
    d["split"] = {"start_s": start, "per_spot_s": per_spot, "per_history_ms": 1000 * per_hist}
    json.dump(d, open(os.path.join(out, "runs.json"), "w"), indent=1)
    for r in d["runs"]: print(f"run_{r['name']}: {r['spots']} spots, {r['histories']} histories, {r['seconds']:.0f} s")
    print(f"start {start:.1f} s, per spot {per_spot:.3f} s, per history {1000 * per_hist:.4f} ms (wall, all threads)")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.add_argument("out"); w.add_argument("chain")
    w.add_argument("--spots", type=int, default=100); w.add_argument("--histories", type=int, default=20000)
    w.add_argument("--factor", type=int, default=20)
    k = sub.add_parser("check"); k.add_argument("out"); k.add_argument("--workspace", required=True)
    a = ap.parse_args()
    if a.cmd == "write": write(a.out, a.chain, a.spots, a.histories, a.factor)
    else: check(a.out, a.workspace)
