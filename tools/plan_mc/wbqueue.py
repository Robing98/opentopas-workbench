"""Queue TOPAS runs through a running workbench (HTTP API), never by starting TOPAS directly.

    python3 wbqueue.py enqueue WORKSPACE FILE...    # parameter files, absolute or relative to WORKSPACE
    python3 wbqueue.py status WORKSPACE FILE...     # status of the newest job per file

The workbench writes WORKSPACE/.workbench/server.json (url, token) while it runs.
"""
import json, os, sys, urllib.request

BINARY = "OpenTOPAS (topas4)"


def _server(ws):
    p = os.path.join(ws, ".workbench", "server.json")
    if not os.path.exists(p): raise SystemExit(f"no running workbench: {p} is missing")
    return json.load(open(p))


def _call(ws, path, body=None):
    s = _server(ws)
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(s["url"].rstrip("/") + path, data=data, headers={"X-Token": s["token"],
                                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r: return json.load(r)


def rel(ws, f):
    a = os.path.abspath(f if os.path.isabs(f) else os.path.join(ws, f))
    r = os.path.relpath(a, os.path.abspath(ws))
    if r.startswith(".."): raise ValueError(f"{f} is not below the workspace {ws}")
    return r.replace(os.sep, "/")


def enqueue(ws, files, binary=BINARY):
    return _call(ws, "/api/enqueue", {"files": [rel(ws, f) for f in files], "binary": binary})


def status(ws, files):
    jobs = _call(ws, "/api/jobs?hist=5000")["jobs"]
    want = {rel(ws, f) for f in files}
    last = {}
    for j in jobs:
        if j.get("file") in want: last[j["file"]] = j
    return {f: (last[f]["status"], last[f].get("rc")) if f in last else ("unknown", None) for f in sorted(want)}


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] not in ("enqueue", "status"): sys.exit(__doc__)
    ws, files = sys.argv[2], sys.argv[3:]
    if sys.argv[1] == "enqueue": print(json.dumps(enqueue(ws, files)))
    else:
        for f, (st, rc) in status(ws, files).items(): print(f"{st:9} rc={rc} {f}")
