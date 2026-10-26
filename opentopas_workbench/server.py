#!/usr/bin/env python3
"""OpenTOPAS Workbench: a small local web UI for OpenTOPAS and TOPAS.

Manage parameter files, run them in a queue, look at scorer output, and generate parameter sets from plugins.
Pure Python standard library. Runs on Linux, macOS and WSL.

    opentopas-workbench [--workspace DIR] [--port 8765] [--no-browser]
    python3 -m opentopas_workbench [...]

Security: listens on 127.0.0.1 only, checks the Host header and requires a per-session token for every API call,
and only touches files inside the workspace.
"""
import argparse, csv, importlib.util, json, os, re, secrets, shutil, signal, subprocess, sys, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import __version__ as VERSION
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_EXT = {".txt", ".py", ".sh", ".bat", ".json", ".csv", ".md", ".log", ".in", ".mac", ".xml", ".header", ".binheader"}
TRASH = "_trash"
TOKEN = secrets.token_urlsafe(24)
STATE = {"root": None, "cfg": {}, "lock": threading.Lock(), "jobs": [], "proc": None, "stop": False, "next_id": 1}

# ------------------------------------------------------------------ config
def wb_dir(): return os.path.join(STATE["root"], ".workbench")


def detect_binaries():
    home = os.path.expanduser("~")
    found = []
    if os.path.exists(os.path.join(home, ".local/bin/topas4")):
        found.append({"name": "OpenTOPAS (topas4)", "cmd": os.path.join(home, ".local/bin/topas4"), "env": {}, "subdir": ""})
    ot = os.path.join(home, "Applications/TOPAS/OpenTOPAS-install/bin/topas")
    if os.path.exists(ot):
        found.append({"name": "OpenTOPAS (Applications)", "cmd": ot, "env": {"TOPAS_G4_DATA_DIR": os.path.join(home, "Applications/GEANT4/G4DATA")}, "subdir": ""})
    t39 = os.path.join(home, "topas/topas/bin/topas")
    if os.path.exists(t39):
        found.append({"name": "TOPAS 3.9", "cmd": t39, "env": {"TOPAS_G4_DATA_DIR": os.path.join(home, "topas/G4Data")}, "subdir": ""})
    p = shutil.which("topas")
    if p and p not in [b["cmd"] for b in found]:
        found.append({"name": "topas (PATH)", "cmd": p, "env": {}, "subdir": ""})
    return found


def load_cfg():
    path = os.path.join(wb_dir(), "config.json")
    cfg = {}
    try:
        with open(path) as fh: cfg = json.load(fh)
    except (OSError, ValueError): pass
    if not cfg.get("binaries"): cfg["binaries"] = detect_binaries()
    cfg.setdefault("lang", "en")
    STATE["cfg"] = cfg


def save_cfg():
    os.makedirs(wb_dir(), exist_ok=True)
    with open(os.path.join(wb_dir(), "config.json"), "w") as fh: json.dump(STATE["cfg"], fh, indent=1)


def load_jobs():
    try:
        with open(os.path.join(wb_dir(), "jobs.json")) as fh: STATE["jobs"] = json.load(fh)
    except (OSError, ValueError): STATE["jobs"] = []
    for j in STATE["jobs"]:
        if j["status"] in ("running", "queued"): j["status"] = "aborted"
    STATE["next_id"] = max([j["id"] for j in STATE["jobs"]] + [0]) + 1


def save_jobs():
    os.makedirs(wb_dir(), exist_ok=True)
    with open(os.path.join(wb_dir(), "jobs.json"), "w") as fh: json.dump(STATE["jobs"][-500:], fh, indent=0)

# ------------------------------------------------------------------ paths
def resolve(rel):
    root = os.path.realpath(STATE["root"])
    p = os.path.realpath(os.path.join(root, rel or ""))
    if p != root and not p.startswith(root + os.sep): raise ValueError("path outside workspace")
    return p


def relp(p):
    r = os.path.relpath(p, os.path.realpath(STATE["root"])).replace(os.sep, "/")
    return "" if r == "." else r


def natkey(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

# ------------------------------------------------------------------ parameter files
PARAM_RE = re.compile(r"^\s*(?:(d|u|i|b|s|dv|uv|iv|bv|sv):)?([A-Za-z0-9_/\.\-]+)\s*=\s*(.*?)\s*$")


def is_param_file(path):
    if not path.endswith(".txt"): return False
    try:
        with open(path, errors="replace") as fh: head = fh.read(6000)
    except OSError: return False
    return bool(re.search(r"^\s*(d|u|i|b|s|dv|sv|iv|uv|bv):(Ge|So|Sc|Ph|Ts|Gr|Ma|El|Tf|Vr)/", head, re.M)) or "includeFile" in head


def lint(text):
    """Light syntax check of a TOPAS parameter file. Returns list of (line, severity, message-key, detail)."""
    issues = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line: continue
        if line.startswith("includeFile"):
            continue
        m = PARAM_RE.match(line)
        if not m:
            issues.append([n, "error", "lint_syntax", raw.strip()[:60]]); continue
        typ, name, val = m.groups()
        if typ is None:
            issues.append([n, "error", "lint_notype", name]); continue
        if typ == "b" and val.strip('"').lower() not in ("true", "false", "t", "f", "1", "0") and "/" not in val:
            issues.append([n, "warn", "lint_bool", val])
        if typ == "s" and not (val.startswith('"') and val.endswith('"')) and "/" not in val:
            issues.append([n, "warn", "lint_string", val])
        if typ == "d" and val and not re.search(r"[A-Za-z]", val):
            issues.append([n, "warn", "lint_unit", val])
        if typ in ("i",) and not re.fullmatch(r"-?\d+", val) and "/" not in val:
            issues.append([n, "warn", "lint_int", val])
        if typ.endswith("v"):
            toks = val.replace('"', ' " ').split() if typ == "sv" else val.split()
            if toks and toks[0].isdigit():
                cnt = int(toks[0])
                rest = toks[1:]
                if typ in ("dv",) and rest and re.fullmatch(r"[A-Za-z/%]+\w*", rest[-1]): rest = rest[:-1]
                if typ == "sv":
                    rest = re.findall(r'"[^"]*"', val)
                if len(rest) != cnt:
                    issues.append([n, "warn", "lint_count", f"{cnt} / {len(rest)}"])
    return issues


def scorer_outputs(text):
    """Base names of scorer output files declared in a parameter file."""
    return re.findall(r'^\s*s:Sc/[^/]+/OutputFile\s*=\s*"([^"]+)"', text, re.M)

# ------------------------------------------------------------------ job queue
def job_worker():
    while True:
        time.sleep(0.5)
        with STATE["lock"]:
            if STATE["proc"] is not None: continue
            job = next((j for j in STATE["jobs"] if j["status"] == "queued"), None)
            if job is None: continue
            job["status"] = "running"; job["start"] = time.strftime("%Y-%m-%d %H:%M:%S"); save_jobs()
        run_job(job)


def run_job(job):
    binary = next((b for b in STATE["cfg"]["binaries"] if b["name"] == job["binary"]), None)
    logdir = os.path.join(wb_dir(), "logs"); os.makedirs(logdir, exist_ok=True)
    job["log"] = relp(os.path.join(logdir, f"job{job['id']:05d}.log"))
    try:
        param = resolve(job["file"])
        out = resolve(os.path.join(os.path.dirname(job["file"]), job.get("subdir") or ""))
        os.makedirs(out, exist_ok=True)
        if binary is None: raise RuntimeError(f"binary '{job['binary']}' not configured")
        env = dict(os.environ, **{k: os.path.expanduser(v) for k, v in binary.get("env", {}).items()})
        # copy companion geometry files (STL/PLY) so relative InputFile paths work in a subfolder
        if out != os.path.dirname(param):
            for f in os.listdir(os.path.dirname(param)):
                if f.lower().endswith((".stl", ".ply")) and not os.path.exists(os.path.join(out, f)):
                    shutil.copy2(os.path.join(os.path.dirname(param), f), out)
        def snapshot():
            return {f: os.path.getmtime(os.path.join(out, f)) for f in os.listdir(out) if os.path.isfile(os.path.join(out, f))}
        before = snapshot()
        t0 = time.time()
        with open(resolve(job["log"]), "w") as log:
            log.write(f"# {job['start']}  {binary['cmd']} {param}\n# cwd: {out}\n"); log.flush()
            proc = subprocess.Popen([os.path.expanduser(binary["cmd"]), param], cwd=out, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            with STATE["lock"]: STATE["proc"] = proc
            rc = proc.wait()
        job["seconds"] = round(time.time() - t0)
        job["outputs"] = sorted(relp(os.path.join(out, f)) for f, mt in snapshot().items()
                                if before.get(f) != mt and not f.lower().endswith((".stl", ".ply")))
        job["status"] = "stopped" if STATE["stop"] else ("done" if rc == 0 else "failed")
        job["rc"] = rc
    except Exception as e:  # report everything in the job record
        job["status"] = "failed"; job["error"] = str(e)
    finally:
        with STATE["lock"]:
            STATE["proc"] = None; STATE["stop"] = False; job["end"] = time.strftime("%Y-%m-%d %H:%M:%S"); save_jobs()


def enqueue(files, binary, subdir, skip_existing):
    added, skipped = [], []
    for f in files:
        p = resolve(f)
        if not os.path.isfile(p): raise ValueError(f"not a file: {f}")
        if skip_existing:
            with open(p, errors="replace") as fh: outs = scorer_outputs(fh.read())
            od = resolve(os.path.join(os.path.dirname(f), subdir or ""))
            if outs and os.path.isdir(od) and all(any(x.startswith(o + ".") for x in os.listdir(od)) for o in outs):
                skipped.append(f); continue
        with STATE["lock"]:
            STATE["jobs"].append({"id": STATE["next_id"], "file": f, "binary": binary, "subdir": subdir, "status": "queued",
                                  "queued": time.strftime("%Y-%m-%d %H:%M:%S")})
            STATE["next_id"] += 1
        added.append(f)
    with STATE["lock"]: save_jobs()
    return added, skipped


def stop_current():
    with STATE["lock"]:
        p = STATE["proc"]
        if p is not None and p.poll() is None:
            STATE["stop"] = True
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)

# ------------------------------------------------------------------ TOPAS CSV reader
def read_topas_csv(path, stat=0, collapse=None, slice_index=None):
    dims, quantity, stats = [], "", []
    with open(path, errors="replace") as fh:
        lines = fh.readlines()
    for l in lines:
        if not l.startswith("#"): break
        m = re.match(r"#\s*(\w+)\s+in\s+(\d+)\s+bins?\s+of\s+([\d.eE+\-]+)\s*(\w+)", l)
        if m: dims.append({"name": m.group(1), "n": int(m.group(2)), "width": float(m.group(3)), "unit": m.group(4)})
        m = re.match(r"#\s*(.+?)\s*:\s*(.+)$", l)
        if m and "bins" not in l and ("(" in m.group(1) or "Sum" in m.group(2) or "Mean" in m.group(2)):
            quantity, stats = m.group(1).strip(), m.group(2).split()
    if len(dims) != 3: raise ValueError("no TOPAS 3D binning header found")
    shape = [d["n"] for d in dims]
    import array
    nstat = max(len(stats), 1)
    data = [array.array("d", bytes(8 * shape[0] * shape[1] * shape[2])) for _ in range(nstat)]
    for l in lines:
        if l.startswith("#") or not l.strip(): continue
        parts = l.replace(",", " ").split()
        i, j, k = int(parts[0]), int(parts[1]), int(parts[2])
        idx = (i * shape[1] + j) * shape[2] + k
        for s in range(min(nstat, len(parts) - 3)): data[s][idx] = float(parts[3 + s])
    vals = data[min(stat, nstat - 1)]
    varying = [a for a in range(3) if shape[a] > 1]
    def at(i, j, k): return vals[(i * shape[1] + j) * shape[2] + k]
    res = {"dims": dims, "quantity": quantity, "stats": stats or ["value"], "stat": stat}
    if len(varying) <= 1:
        a = varying[0] if varying else 2
        res["mode"] = "1d"; res["axis"] = a
        res["x"] = [(n + 0.5) * dims[a]["width"] for n in range(shape[a])]
        res["y"] = [at(*[n if q == a else 0 for q in range(3)]) for n in range(shape[a])]
        return res
    if len(varying) == 3:
        c = collapse if collapse in (0, 1, 2) else 2
        varying = [a for a in range(3) if a != c]
    else:
        c = [a for a in range(3) if a not in varying][0]
    a, b = varying
    grid = []
    for u in range(shape[a]):
        row = []
        for v in range(shape[b]):
            if shape[c] == 1: row.append(at(*[{a: u, b: v}.get(q, 0) for q in range(3)]))
            elif slice_index is not None:
                row.append(at(*[{a: u, b: v, c: min(slice_index, shape[c] - 1)}[q] for q in range(3)]))
            else:
                row.append(sum(at(*[{a: u, b: v, c: w}[q] for q in range(3)]) for w in range(shape[c])))
        grid.append(row)
    res.update(mode="2d", axes=[a, b], collapsed=c, grid=grid)
    return res

# ------------------------------------------------------------------ generator plugins
def load_generators():
    gens = {}
    for d in (os.path.join(TOOL_DIR, "generators"), os.path.join(wb_dir(), "generators")):
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py") or f.startswith("_"): continue
            try:
                spec = importlib.util.spec_from_file_location("wbgen_" + f[:-3], os.path.join(d, f))
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                gens[f[:-3]] = mod
            except Exception as e:
                print(f"generator {f} not loaded: {e}", file=sys.stderr)
    return gens


def gen_schema():
    return [{"id": k, "name": getattr(m, "NAME", {"en": k}), "description": getattr(m, "DESCRIPTION", {}),
             "fields": getattr(m, "FIELDS", [])} for k, m in load_generators().items()]

# ------------------------------------------------------------------ file manager
def fm_ls(rel):
    p = resolve(rel)
    items = []
    for f in os.listdir(p):
        if f == ".workbench": continue
        fp = os.path.join(p, f)
        try: st = os.stat(fp)
        except OSError: continue
        items.append({"name": f, "dir": os.path.isdir(fp), "size": st.st_size, "mtime": st.st_mtime})
    items.sort(key=lambda i: (not i["dir"], natkey(i["name"])))
    return {"path": relp(p), "items": items}


def fm_target(src, dst):
    s = resolve(src)
    d = resolve(dst if "/" in dst else os.path.join(os.path.dirname(relp(s)), dst))
    if os.path.isdir(d): d = os.path.join(d, os.path.basename(s))
    if os.path.exists(d): raise ValueError("target exists")
    return s, d


def list_params():
    out = []
    root = os.path.realpath(STATE["root"])
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if not d.startswith(".") and d != TRASH)
        for f in sorted(fns, key=natkey):
            fp = os.path.join(dp, f)
            if is_param_file(fp): out.append(relp(fp))
        if len(out) > 5000: break
    last = {}
    for j in STATE["jobs"]:
        last[(j["file"], j["binary"])] = {"status": j["status"], "end": j.get("end", "")}
    return {"files": out, "last": {f"{k[0]}|{k[1]}": v for k, v in last.items()}}

# ------------------------------------------------------------------ HTTP
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
        self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(data)

    def guard(self, need_token=True):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("localhost", "127.0.0.1"): self.send(403, {"error": "bad host"}); return False
        if need_token and self.headers.get("X-Token") != TOKEN and parse_qs(urlparse(self.path).query).get("token", [""])[0] != TOKEN:
            self.send(403, {"error": "bad token"}); return False
        return True

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            if not self.guard(False): return
            page = PAGE.replace("%TOKEN%", TOKEN).replace("%VERSION%", VERSION).replace("%LANG%", STATE["cfg"].get("lang", "en"))
            return self.send(200, page, "text/html")
        if not self.guard(): return
        try:
            if u.path == "/api/info":
                return self.send(200, {"root": STATE["root"], "cfg": STATE["cfg"], "version": VERSION, "detected": detect_binaries()})
            if u.path == "/api/ls": return self.send(200, fm_ls(q.get("p", "")))
            if u.path == "/api/read":
                p = resolve(q.get("p", ""))
                if os.path.splitext(p)[1].lower() not in TEXT_EXT: raise ValueError("not a text file")
                if os.path.getsize(p) > 3_000_000: raise ValueError("file too large for the editor (> 3 MB)")
                with open(p, errors="replace") as fh: txt = fh.read()
                return self.send(200, {"text": txt, "lint": lint(txt) if p.endswith(".txt") else []})
            if u.path == "/api/params": return self.send(200, list_params())
            if u.path == "/api/jobs":
                with STATE["lock"]: jobs = list(STATE["jobs"][-200:]); running = STATE["proc"] is not None
                return self.send(200, {"jobs": jobs, "running": running})
            if u.path == "/api/log":
                p = resolve(q.get("p", ""))
                with open(p, errors="replace") as fh: fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 12000)); return self.send(200, {"text": fh.read()})
            if u.path == "/api/csv":
                col = q.get("collapse"); sl = q.get("slice")
                return self.send(200, read_topas_csv(resolve(q.get("p", "")), int(q.get("stat", 0)),
                                                     int(col) if col not in (None, "") else None, int(sl) if sl not in (None, "") else None))
            if u.path == "/api/generators": return self.send(200, gen_schema())
            if u.path == "/file":
                p = resolve(q.get("p", ""))
                ext = os.path.splitext(p)[1].lower()
                types = {".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".csv": "text/plain", ".txt": "text/plain"}
                if ext not in types: raise ValueError("type not served")
                with open(p, "rb") as fh: return self.send(200, fh.read(), types[ext])
            self.send(404, {"error": "not found"})
        except (OSError, ValueError, RuntimeError) as e:
            self.send(400, {"error": str(e)})

    def do_POST(self):
        if not self.guard(): return
        u = urlparse(self.path)
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            if u.path == "/api/write":
                p = resolve(b["p"])
                if os.path.splitext(p)[1].lower() not in TEXT_EXT: raise ValueError("text files only")
                with open(p, "w", newline="") as fh: fh.write(b["text"])
                return self.send(200, {"ok": True, "lint": lint(b["text"]) if p.endswith(".txt") else []})
            if u.path == "/api/lint": return self.send(200, {"lint": lint(b.get("text", ""))})
            if u.path == "/api/mkdir":
                d = resolve(b["p"])
                if os.path.exists(d): raise ValueError("exists")
                os.makedirs(d); return self.send(200, {"ok": True})
            if u.path == "/api/newfile":
                p = resolve(b["p"])
                if os.path.exists(p): raise ValueError("exists")
                if os.path.splitext(p)[1].lower() not in TEXT_EXT: raise ValueError("text files only")
                open(p, "w").close(); return self.send(200, {"ok": True})
            if u.path == "/api/move":
                s, d = fm_target(b["src"], b["dst"]); os.rename(s, d); return self.send(200, {"ok": True})
            if u.path == "/api/copy":
                s, d = fm_target(b["src"], b["dst"]); (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, d)
                return self.send(200, {"ok": True})
            if u.path == "/api/trash":
                p = resolve(b["p"])
                if p == os.path.realpath(STATE["root"]) or os.path.basename(p) == TRASH: raise ValueError("not allowed")
                t = resolve(TRASH); os.makedirs(t, exist_ok=True)
                os.rename(p, os.path.join(t, time.strftime("%Y%m%d-%H%M%S_") + os.path.basename(p)))
                return self.send(200, {"ok": True})
            if u.path == "/api/enqueue":
                a, s = enqueue(b["files"], b["binary"], b.get("subdir", ""), b.get("skip", False))
                return self.send(200, {"added": a, "skipped": s})
            if u.path == "/api/stop": stop_current(); return self.send(200, {"ok": True})
            if u.path == "/api/dequeue":
                with STATE["lock"]:
                    for j in STATE["jobs"]:
                        if j["id"] in b.get("ids", []) and j["status"] == "queued": j["status"] = "removed"
                    save_jobs()
                return self.send(200, {"ok": True})
            if u.path == "/api/clearhistory":
                with STATE["lock"]:
                    STATE["jobs"] = [j for j in STATE["jobs"] if j["status"] in ("queued", "running")]; save_jobs()
                return self.send(200, {"ok": True})
            if u.path == "/api/config":
                cfg = STATE["cfg"]
                if "lang" in b: cfg["lang"] = b["lang"] if b["lang"] in ("de", "en") else "en"
                if "binaries" in b:
                    bins = []
                    for x in b["binaries"]:
                        if x.get("name") and x.get("cmd"):
                            bins.append({"name": str(x["name"])[:60], "cmd": str(x["cmd"]), "env": {str(k): str(v) for k, v in (x.get("env") or {}).items()},
                                         "subdir": str(x.get("subdir", ""))})
                    cfg["binaries"] = bins
                save_cfg(); return self.send(200, {"ok": True, "cfg": cfg})
            if u.path == "/api/generate":
                gens = load_generators()
                mod = gens.get(b["id"])
                if mod is None: raise ValueError("unknown generator")
                files = mod.generate(b.get("values", {}))
                folder = b.get("folder", "")
                if b.get("preview"):
                    first = next((t for _, t in files if isinstance(t, str)), "")
                    return self.send(200, {"names": [n for n, _ in files], "example": first})
                written, skipped = [], []
                for name, content in files:
                    p = resolve(os.path.join(folder, name))
                    if os.path.exists(p) and not b.get("overwrite"): skipped.append(name); continue
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "wb" if isinstance(content, bytes) else "w") as fh: fh.write(content)
                    written.append(name)
                return self.send(200, {"written": written, "skipped": skipped})
            self.send(404, {"error": "not found"})
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as e:
            self.send(400, {"error": str(e)})


PAGE = open(os.path.join(TOOL_DIR, "ui.html"), encoding="utf-8").read() if os.path.exists(os.path.join(TOOL_DIR, "ui.html")) else "<p>ui.html missing</p>"


def main(argv=None):
    ap = argparse.ArgumentParser(description="OpenTOPAS Workbench")
    ap.add_argument("--workspace", default=os.getcwd(), help="folder with your parameter files (default: current folder)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    STATE["root"] = os.path.realpath(a.workspace)
    if not os.path.isdir(STATE["root"]): sys.exit(f"workspace not found: {STATE['root']}")
    load_cfg(); load_jobs(); save_cfg()
    threading.Thread(target=job_worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://localhost:{a.port}/"
    print(f"OpenTOPAS Workbench {VERSION}\n  workspace: {STATE['root']}\n  open:      {url}\n  stop:      Ctrl+C")
    if not a.no_browser:
        try: webbrowser.open(url)
        except Exception: pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_current()


if __name__ == "__main__":
    main()
