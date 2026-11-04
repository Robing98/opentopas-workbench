"""MCP server for the OpenTOPAS Workbench: lets an MCP client (Claude Code, Claude Desktop, ...) drive a running workbench.

    opentopas-workbench mcp [--workspace DIR] [--url URL --token TOKEN] [--read-only] [--no-start]

A thin adapter over the HTTP API: JSON-RPC 2.0 over stdin and stdout, one message per line (MCP stdio transport).
It speaks both protocol eras: the "initialize" handshake of the revisions up to 2025-11-25, and the per-request
"_meta" with "server/discover" of 2026-07-28. Standard library only.

Finding the workbench: every workbench writes its address and token to <workspace>/.workbench/server.json. With
--workspace (default: the nearest folder upward with a .workbench folder, else the current folder) the adapter reads
that file. When no workbench runs for the workspace,
it starts one in the background (no browser, output in .workbench/server.log), because queued runs have to outlive
the MCP client; --no-start reports an error instead. With --url the token comes from --token or the environment
variable OPENTOPAS_WORKBENCH_TOKEN.

Paths are relative to the workspace; the HTTP API refuses anything outside it. There is no shell access.
"""
import argparse, fnmatch, json, math, os, subprocess, sys, time, urllib.error, urllib.parse, urllib.request

from . import __version__ as VERSION

# cwd of a started workbench: "python -m" puts the cwd first on sys.path, so a folder opentopas_workbench in the
# workspace would be imported instead of this package
PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODERN_VERSIONS = ["2026-07-28"]                                           # per-request _meta, server/discover
LEGACY_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]   # initialize handshake
ALL_VERSIONS = MODERN_VERSIONS + LEGACY_VERSIONS
META_VERSION = "io.modelcontextprotocol/protocolVersion"
START_TIMEOUT = 30.0
HTTP_TIMEOUT = 120.0   # the first index of a large workspace can take a while

INSTRUCTIONS = (
    "Drives an OpenTOPAS Workbench: TOPAS parameter files (cases), a run queue, and scorer results. "
    "Paths are relative to the workspace. Typical flow: list_generators, generate, enqueue, queue_status, job_log, "
    "then list_results and get_curve. Depth metrics (peak, r80, bpw80) are in mm.")

LINT_TEXT = {
    "lint_include": "included file not found", "lint_syntax": "not a parameter line",
    "lint_notype": "missing type prefix (d:, u:, i:, b:, s:, dv:, ...)", "lint_bool": "boolean value expected",
    "lint_string": "string value should be quoted", "lint_unit": "double value without a unit",
    "lint_int": "integer value expected", "lint_count": "vector length does not match the count (count / values)",
}


class ApiError(Exception):
    """An error to report to the model as a tool error."""


# ------------------------------------------------------------------ connection to the workbench
def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, TypeError, ValueError):
        return True   # exists but belongs to someone else, or unknown: let the HTTP probe decide
    return True


def read_server_file(workspace):
    """(url, token, pid) from <workspace>/.workbench/server.json, or None when it is missing or its process is gone."""
    try:
        with open(os.path.join(workspace, ".workbench", "server.json")) as fh: s = json.load(fh)
        url, token, pid = s["url"].replace("//localhost:", "//127.0.0.1:"), s["token"], s.get("pid")   # the server binds 127.0.0.1 only
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return None
    if pid is not None and not pid_alive(pid): return None   # left behind by a killed workbench
    return url, token, pid


def workbench_without_file(workspace):
    """Pid of a running workbench server for this workspace (Linux and WSL, via /proc), else None. Subcommands
    (mcp, enqueue, jobs) do not count. Only called when server.json gave no live workbench."""
    from .server import PROC_DIR
    if not os.path.isdir(PROC_DIR): return None
    for d in os.listdir(PROC_DIR):
        if not d.isdigit() or int(d) == os.getpid(): continue
        try:
            with open(os.path.join(PROC_DIR, d, "cmdline"), "rb") as fh:
                argv = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
            cwd = os.readlink(os.path.join(PROC_DIR, d, "cwd"))
        except OSError:
            continue   # gone meanwhile, or another user's process
        if "opentopas_workbench" in argv:
            rest = argv[argv.index("opentopas_workbench") + 1:]
        elif argv and os.path.basename(argv[0]) == "opentopas-workbench":
            rest = argv[1:]
        elif len(argv) > 1 and os.path.basename(argv[1]) == "opentopas-workbench":   # python3 .../bin/opentopas-workbench
            rest = argv[2:]
        else:
            continue
        if rest[:1] in (["mcp"], ["enqueue"], ["jobs"]): continue
        ws = cwd
        for i, a in enumerate(rest):
            if a == "--workspace" and i + 1 < len(rest): ws = os.path.join(cwd, rest[i + 1])
            elif a.startswith("--workspace="): ws = os.path.join(cwd, a.split("=", 1)[1])
        if os.path.realpath(ws) == workspace: return int(d)
    return None


class Workbench:
    def __init__(self, workspace=None, url=None, token=None, start=True):
        self.workspace = os.path.realpath(workspace) if workspace else None
        self.fixed_url, self.fixed_token, self.start = url, token, start
        self.url = self.token = None
        self.started = None   # pid of a workbench this adapter started

    def _request(self, url, token, path, query=None, body=None):
        q = "?" + urllib.parse.urlencode({k: v for k, v in (query or {}).items() if v is not None}) if query else ""
        req = urllib.request.Request(url.rstrip("/") + path + q, data=None if body is None else json.dumps(body).encode())
        req.add_header("X-Token", token); req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try: msg = json.loads(e.read()).get("error") or str(e)
            except ValueError: msg = str(e)
            raise ApiError(msg)

    def _probe(self, url, token):
        """Info of the workbench at url, or None when nothing answers there with this token."""
        try: return self._request(url, token, "/api/info")
        except (ApiError, OSError, ValueError): return None

    def _from_file(self):
        found = read_server_file(self.workspace)
        if not found: return None
        info = self._probe(found[0], found[1])
        if info and os.path.realpath(info.get("root", "")) == self.workspace: return found
        return None

    def _refuse_second(self):
        other = workbench_without_file(self.workspace)
        if other:
            # a second workbench would run a second queue on the same files; a version before 0.6.0 writes no server.json
            raise ApiError(f"a workbench (pid {other}) already runs for {self.workspace} but wrote no .workbench/server.json, "
                           f"probably an older version. Restart it with this version, or pass --url and --token.")

    def _spawn(self):
        wb = os.path.join(self.workspace, ".workbench"); os.makedirs(wb, exist_ok=True)
        log = os.path.join(wb, "server.log")
        with open(log, "w") as fh:
            proc = subprocess.Popen([sys.executable, "-m", "opentopas_workbench", "--workspace", self.workspace, "--no-browser"],
                                    stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, cwd=PACKAGE_PARENT,
                                    start_new_session=True)   # keeps running, with its queue, after the MCP client exits
        t_end = time.time() + START_TIMEOUT
        while time.time() < t_end:
            found = self._from_file()
            if found and found[2] == proc.pid:
                self.started = proc.pid
                return found
            if proc.poll() is not None: break
            time.sleep(0.2)
        try:
            with open(log, errors="replace") as fh: tail = fh.read()[-600:]
        except OSError:
            tail = ""
        if proc.poll() is None: proc.terminate()
        raise ApiError(f"could not start the workbench for {self.workspace}. {tail.strip()}")

    def connect(self):
        if self.url: return
        if self.fixed_url:
            if not self.fixed_token:
                raise ApiError("--url needs --token or OPENTOPAS_WORKBENCH_TOKEN (the token is in .workbench/server.json of the workspace)")
            if self._probe(self.fixed_url, self.fixed_token) is None:
                raise ApiError(f"no workbench answers at {self.fixed_url} with this token")
            self.url, self.token = self.fixed_url, self.fixed_token
            return
        if not os.path.isdir(self.workspace): raise ApiError(f"workspace not found: {self.workspace}")
        found = self._from_file()
        if not found:
            self._refuse_second()
            if not self.start:
                raise ApiError(f"no workbench is running for {self.workspace}. Start it with "
                               f"'opentopas-workbench --workspace {self.workspace}'.")
            found = self._spawn()
        self.url, self.token = found[0], found[1]

    def call(self, path, query=None, body=None):
        self.connect()
        try:
            return self._request(self.url, self.token, path, query, body)
        except (OSError, ValueError) as e:   # the workbench went away; the next call looks for (or starts) a new one
            self.url = self.token = None
            raise ApiError(f"lost the connection to the workbench: {e}")

    def get(self, path, **query): return self.call(path, query=query)

    def post(self, path, body): return self.call(path, body=body)


# ------------------------------------------------------------------ helpers
def matches(text, pattern):
    """Glob when the pattern has wildcards, else a case-insensitive substring."""
    if not pattern: return True
    if any(c in pattern for c in "*?["): return fnmatch.fnmatch(text.lower(), pattern.lower())
    return pattern.lower() in text.lower()


def lint_items(issues):
    return [{"line": n, "severity": sev, "message": LINT_TEXT.get(key, key), "detail": detail}
            for n, sev, key, detail in issues]


def downsample(n, m, keep=None):
    """At most m indices out of range(n), evenly spaced, with the first, the last, and keep."""
    if n <= m: return list(range(n))
    idx = {round(i * (n - 1) / (m - 1)) for i in range(m)}
    if keep is not None: idx.add(keep)
    return sorted(idx)


def en(v):
    return v.get("en", next(iter(v.values()), "")) if isinstance(v, dict) else v


def find_job(wb, job_id):
    jobs = wb.get("/api/jobs", hist=5000)["jobs"]
    if job_id is None:   # the running job, else the last one that ran
        job = next((j for j in jobs if j["status"] == "running"), None)
        job = job or next((j for j in reversed(jobs) if j.get("log")), None)
        if job is None: raise ApiError("no job has run yet")
        return job
    job = next((j for j in jobs if j["id"] == job_id), None)
    if job is None: raise ApiError(f"no job with id {job_id}")
    return job


def job_brief(j, full=False):
    keys = ("id", "file", "binary", "subdir", "status", "queued", "start", "end", "seconds", "rc", "error")
    out = {k: j[k] for k in keys if j.get(k) not in (None, "")}
    if full:
        if j.get("hints"): out["hints"] = j["hints"]
        if j.get("outputs"): out["outputs"] = j["outputs"][:20]
    return out


# ------------------------------------------------------------------ tools
def t_info(wb, a):
    r = wb.get("/api/info")
    out = {"workspace": r["root"], "version": r["version"],
           "programs": [{"name": b["name"], "cmd": b["cmd"], "subdir": b.get("subdir", "")} for b in r["cfg"].get("binaries", [])],
           "folders": r.get("folders", [])[:100]}
    if wb.started: out["note"] = f"this MCP server started the workbench (pid {wb.started}), output in .workbench/server.log"
    return out


def t_list_files(wb, a):
    return wb.get("/api/ls", p=a.get("path", ""))


def t_list_cases(wb, a):
    r = wb.get("/api/params", scope=a.get("folder", ""))
    last = {}
    for key, v in r["last"].items():
        f, prog = key.rsplit("|", 1)
        last.setdefault(f, {})[prog] = v["status"]
    status, prog = a.get("status", "any"), a.get("program")
    out = []
    for f in r["files"]:
        if not matches(f, a.get("filter", "")): continue
        runs = {p: s for p, s in last.get(f, {}).items() if not prog or p == prog}
        res = r["results"].get(f, [])
        if status == "not_run" and any(s != "removed" for s in runs.values()): continue
        if status == "has_results" and not res: continue
        if status == "no_results" and res: continue
        if status not in ("any", "not_run", "has_results", "no_results") and status not in runs.values(): continue
        item = {"file": f}
        if runs: item["last_run"] = runs
        if res: item["results"] = res
        out.append(item)
    limit = a.get("limit", 200)
    return {"total": len(out), "cases": out[:limit], "truncated": len(out) > limit}


def inside_wb_dir(path):
    """True for paths in the workbench's own folder: settings (program commands), generator plugins (Python), the token."""
    parts = [x for x in os.path.normpath(str(path).replace("\\", "/")).split(os.sep) if x not in ("", ".")]
    return bool(parts) and parts[0] == ".workbench"


def t_read_file(wb, a):
    if os.path.normpath(str(a["path"])).lstrip(os.sep) == os.path.join(".workbench", "server.json"):
        raise ApiError("the server file holds the API token and is not readable through MCP")
    r = wb.get("/api/read", p=a["path"])
    return {"path": a["path"], "text": r["text"], "lint": lint_items(r["lint"]),
            "includes": [{"line": n, "name": name, "path": p} for n, name, p in r["includes"]]}


def t_lint(wb, a):
    body = {"text": a["text"]}
    if a.get("path"): body["p"] = a["path"]
    r = wb.post("/api/lint", body)
    return {"lint": lint_items(r["lint"]), "includes": [{"line": n, "name": name, "path": p} for n, name, p in r["includes"]]}


def t_list_generators(wb, a):
    out = []
    for g in wb.get("/api/generators"):
        fields = []
        for f in g["fields"]:
            d = {"key": f["key"], "type": f.get("type", "text"), "label": en(f.get("label", "")), "default": f.get("default")}
            if f.get("options"): d["options"] = [o[0] for o in f["options"]]
            fields.append(d)
        out.append({"id": g["id"], "name": en(g["name"]), "description": en(g["description"]), "fields": fields})
    return {"generators": out}


def t_list_results(wb, a):
    r = wb.get("/api/index", scope=a.get("folder", ""))
    keys = ("path", "param", "particle", "energy", "quantity", "mode", "peak", "r80", "bpw80", "dfo", "tail", "tags")
    out = []
    for it in r["items"]:
        if it.get("kind") not in ("data", "measured") and not a.get("include_images"): continue
        if not matches(it["path"], a.get("filter", "")): continue
        if a.get("particle") and (it.get("particle") or "").lower() != a["particle"].lower(): continue
        item = {k: it[k] for k in keys if it.get(k) not in (None, "", [])}
        item["kind"] = it.get("kind")
        out.append(item)
    limit = a.get("limit", 200)
    return {"total": len(out), "results": out[:limit], "truncated": len(out) > limit, "stale": r.get("stale", False)}


def t_get_curve(wb, a):
    q = {"p": a["path"], "stat": a.get("stat", 0), "collapse": a.get("collapse"), "slice": a.get("slice"), "scan": a.get("scan", 0)}
    r = wb.get("/api/csv", **q)
    out = {"path": a["path"], "mode": r["mode"], "quantity": r.get("quantity", ""), "stats": r.get("stats", []),
           "dims": r.get("dims", [])}
    if r.get("beam"): out["beam"] = r["beam"]
    if r["mode"] == "1d":
        x, y = r["x"], r["y"]
        peak = max(range(len(y)), key=y.__getitem__) if y else None
        idx = downsample(len(x), max(2, a.get("max_points", 200)), peak)
        dim = r["dims"][r["axis"]] if r.get("dims") else {}
        out.update(x_unit=dim.get("unit", ""), x_name=dim.get("name", ""), points=len(x),
                   x=[x[i] for i in idx], y=[y[i] for i in idx])
    else:
        grid = r["grid"]
        side = max(2, min(a.get("max_points", 40), 200))
        rows = downsample(len(grid), side)
        cols = downsample(len(grid[0]) if grid else 0, side)
        out.update(axes=r["axes"], collapsed=r.get("collapsed"), shape=[len(grid), len(grid[0]) if grid else 0],
                   rows=rows, cols=cols, grid=[[grid[i][j] for j in cols] for i in rows])
    return out


def t_queue_status(wb, a):
    n = a.get("history", 5)
    r = wb.get("/api/jobs", hist=n)
    jobs = r["jobs"]
    running = []
    for j in jobs:
        if j["status"] != "running": continue
        b = job_brief(j)
        for k in ("total", "done", "progress", "eta"):
            if j.get(k) is not None: b[k] = round(j[k], 3) if k == "progress" else j[k]
        lv = j.get("live") or {}
        b["live"] = {k: lv[k] for k in ("phase", "rate", "cores", "rss_mb", "last") if lv.get(k) is not None}
        running.append(b)
    waiting = [job_brief(j) for j in jobs if j["status"] == "queued"]
    finished = [job_brief(j, full=True) for j in jobs if j["status"] not in ("queued", "running")]
    return {"running": running, "waiting_count": len(waiting), "waiting": waiting[:20],
            "finished": list(reversed(finished))[:n], "external": r.get("external", []), "counts": r["counts"],
            "queue_eta": r.get("queue_eta"),
            "keep_awake": r.get("awake")}


def t_job_log(wb, a):
    job = find_job(wb, a.get("job_id"))
    if not job.get("log"): raise ApiError(f"job {job['id']} has no log yet (status {job['status']})")
    text = wb.get("/api/log", p=job["log"])["text"]
    lines = text.splitlines()[-max(1, min(a.get("lines", 60), 400)):]
    return {"job": job_brief(job, full=True), "log": job["log"], "tail": "\n".join(lines)}


def t_case_outputs(wb, a):
    return wb.post("/api/outputs", {"files": a["files"]})


def t_write_file(wb, a):
    # the settings name the programs that runs execute, and generators are Python: writing there would run any code
    if inside_wb_dir(a["path"]): raise ApiError("files in .workbench (settings, generator plugins) cannot be written through MCP")
    r = wb.post("/api/write", {"p": a["path"], "text": a["text"]})
    return {"ok": True, "path": a["path"], "lint": lint_items(r["lint"])}


def t_generate(wb, a):
    gens = {g["id"]: g for g in wb.get("/api/generators")}
    g = gens.get(a["generator"])
    if g is None: raise ApiError(f"unknown generator '{a['generator']}'; available: {', '.join(sorted(gens))}")
    values = {f["key"]: f.get("default") for f in g["fields"]}   # the generators read every field
    values.update(a.get("values") or {})
    unknown = sorted(set(a.get("values") or {}) - {f["key"] for f in g["fields"]})
    if unknown: raise ApiError(f"unknown fields for {a['generator']}: {', '.join(unknown)}")
    if inside_wb_dir(a.get("folder", "")): raise ApiError("cannot generate into .workbench")
    r = wb.post("/api/generate", {"id": a["generator"], "values": values, "folder": a.get("folder", ""),
                                  "overwrite": bool(a.get("overwrite")), "preview": bool(a.get("preview"))})
    folder = a.get("folder", "").strip("/")
    if a.get("preview"): return {"would_write": r["names"], "example": r["example"]}
    join = lambda n: f"{folder}/{n}" if folder else n
    return {"written": [join(n) for n in r["written"]], "skipped_existing": [join(n) for n in r["skipped"]]}


def t_enqueue(wb, a):
    prog = a.get("program")
    if not prog:
        progs = [b["name"] for b in wb.get("/api/info")["cfg"].get("binaries", [])]
        if len(progs) != 1: raise ApiError(f"name the program; configured: {', '.join(progs) or 'none (see Settings)'}")
        prog = progs[0]
    r = wb.post("/api/enqueue", {"files": a["files"], "binary": prog, "subdir": a.get("subfolder", ""),
                                 "skip": bool(a.get("skip_existing"))})
    return {"program": prog, "added": r["added"], "skipped_with_results": r["skipped"]}


def t_dequeue(wb, a):
    if not a.get("all") and not a.get("ids"): raise ApiError("give ids, or all: true")
    r = wb.post("/api/dequeue", {"ids": a.get("ids", []), "all": bool(a.get("all"))})
    return {"removed": r["removed"]}


def t_stop(wb, a):
    wb.post("/api/stop", {})
    return {"ok": True, "note": "the running job was sent SIGTERM; waiting jobs start next (dequeue them first to halt the queue)"}


S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}


def obj(props=None, required=()):
    o = {"type": "object", "properties": props or {}, "additionalProperties": False}
    if required: o["required"] = list(required)
    return o


def d(schema, text): return dict(schema, description=text)


FOLDER = d(S, "limit to this workspace folder (default: whole workspace)")
FILTER = d(S, "substring of the path, or a glob such as 'cases/*_p1*.txt'")
LIMIT = d(I, "maximum number of entries (default 200)")

# name: (function, writes, description, input schema)
TOOLS = {
    "info": (t_info, False, "Workspace, version, configured TOPAS programs (names for enqueue), and folders.", obj()),
    "list_files": (t_list_files, False, "List one folder of the workspace (name, dir, size, mtime).",
                   obj({"path": d(S, "folder relative to the workspace, '' for the top")})),
    "list_cases": (t_list_cases, False,
                   "List TOPAS parameter files (cases) with the last run status per program and the result files found.",
                   obj({"filter": FILTER, "folder": FOLDER, "limit": LIMIT,
                        "status": d({"type": "string", "enum": ["any", "not_run", "has_results", "no_results", "done", "failed",
                                                                 "stopped", "aborted", "queued", "running"]},
                                    "run status filter (default any)"),
                        "program": d(S, "only runs of this program")})),
    "read_file": (t_read_file, False, "Read a text file of the workspace; parameter files come with lint issues and includes.",
                  obj({"path": d(S, "file relative to the workspace")}, ["path"])),
    "lint": (t_lint, False, "Check TOPAS parameter text without saving it.",
             obj({"text": S, "path": d(S, "where the text would live, to resolve includeFile")}, ["text"])),
    "list_generators": (t_list_generators, False, "Generators that create parameter files, with their fields and defaults.", obj()),
    "list_results": (t_list_results, False,
                     "Result files with the producing case, particle, energy (MeV/u), quantity, and depth-dose metrics "
                     "(peak, r80, bpw80, dfo = distal fall-off D80-D20, in mm; tail = relative dose behind the peak).",
                     obj({"filter": FILTER, "folder": FOLDER, "limit": LIMIT, "particle": d(S, "e.g. p, He-4, C-12"),
                          "include_images": d(B, "also list PNG/SVG files")})),
    "get_curve": (t_get_curve, False,
                  "Data of one result file. 1D: x (in x_unit) and y, downsampled to max_points with the peak kept. "
                  "2D or 3D: a grid downsampled to at most max_points rows and columns (3D is summed over 'collapse').",
                  obj({"path": d(S, "result file (.csv, .mcc, .asc, two-column text)"),
                       "max_points": d(I, "1D: points (default 200); 2D: rows and columns (default 40)"),
                       "stat": d(I, "statistic column of a TOPAS file, 0 = first"),
                       "collapse": d(I, "3D: axis to sum over, 0 = x, 1 = y, 2 = z"),
                       "slice": d(I, "3D: take this bin of the collapsed axis instead of the sum"),
                       "scan": d(I, "measured file with several scans: index")}, ["path"])),
    "queue_status": (t_queue_status, False, "Running job with progress and ETA, waiting jobs, recent finished jobs, queue ETA, and TOPAS runs started "
                     "outside the queue in this workspace (external, shown only).",
                     obj({"history": d(I, "number of finished jobs to show (default 5)")})),
    "job_log": (t_job_log, False, "Tail of a job's TOPAS log, with error hints. Without job_id: the running or last job.",
                obj({"job_id": I, "lines": d(I, "lines from the end (default 60, max 400)")})),
    "case_outputs": (t_case_outputs, False, "Result files written by the given cases (next to them and in result subfolders).",
                     obj({"files": {"type": "array", "items": S}}, ["files"])),
    "write_file": (t_write_file, True, "Create or overwrite a text file in the workspace. Returns lint issues for parameter files.",
                   obj({"path": S, "text": S}, ["path", "text"])),
    "generate": (t_generate, True,
                 "Create parameter files with a generator (see list_generators). Missing fields take their defaults. "
                 "Existing files are skipped unless overwrite is true; preview returns the names and one example without writing.",
                 obj({"generator": d(S, "generator id"), "values": {"type": "object", "description": "field key to value"},
                      "folder": d(S, "target folder in the workspace"), "overwrite": B, "preview": B}, ["generator"])),
    "enqueue": (t_enqueue, True, "Queue cases for a run. Runs start one after another in the background.",
                obj({"files": {"type": "array", "items": S, "description": "parameter files"},
                     "program": d(S, "program name from info (optional when only one is configured)"),
                     "subfolder": d(S, "write results into this subfolder next to each case"),
                     "skip_existing": d(B, "skip cases whose outputs already exist")}, ["files"])),
    "dequeue": (t_dequeue, True, "Remove waiting jobs from the queue (not the running one).",
                obj({"ids": {"type": "array", "items": I}, "all": d(B, "remove every waiting job")})),
    "stop": (t_stop, True, "Stop the running job. Waiting jobs stay in the queue.", obj()),
}


def tool_list(read_only):
    out = []
    for name, (_, writes, desc, schema) in TOOLS.items():
        if writes and read_only: continue
        ann = {"readOnlyHint": not writes, "openWorldHint": False}
        if writes: ann.update(destructiveHint=name in ("write_file", "dequeue", "stop"), idempotentHint=name in ("write_file", "stop"))
        out.append({"name": name, "description": desc, "inputSchema": schema, "annotations": ann})
    return out


# ------------------------------------------------------------------ JSON-RPC
class RpcError(Exception):
    def __init__(self, code, message, data=None):
        super().__init__(message); self.code, self.message, self.data = code, message, data


class Server:
    def __init__(self, wb, read_only=False):
        self.wb, self.read_only = wb, read_only

    def server_info(self):
        return {"name": "opentopas-workbench", "title": "OpenTOPAS Workbench", "version": VERSION}

    def call_tool(self, params):
        name, args = params.get("name"), params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(args, dict): raise RpcError(-32602, "tools/call needs a name and an arguments object")
        if name not in TOOLS: raise RpcError(-32602, f"Unknown tool: {name}")
        fn, writes, _, schema = TOOLS[name]
        if writes and self.read_only:
            return self.tool_error(f"{name} changes the workspace or the queue; this MCP server runs with --read-only")
        missing = [k for k in schema.get("required", []) if k not in args]
        if missing: return self.tool_error(f"missing argument: {', '.join(missing)}")
        try:
            res = fn(self.wb, args)
        except ApiError as e:
            return self.tool_error(str(e))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            return self.tool_error(f"bad arguments or answer: {e!r}")
        return {"content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}], "structuredContent": res,
                "isError": False}

    @staticmethod
    def tool_error(msg):
        return {"content": [{"type": "text", "text": msg}], "isError": True}

    def handle(self, msg):
        """The response to one message, or None for a notification."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return {"jsonrpc": "2.0", "id": msg.get("id") if isinstance(msg, dict) else None,
                    "error": {"code": -32600, "message": "Invalid Request"}}
        if "id" not in msg: return None   # notifications: initialized, cancelled; nothing to answer
        params = msg.get("params") or {}
        try:
            if not isinstance(params, dict): raise RpcError(-32602, "params must be an object")
            result = self.dispatch(msg["method"], params)
            return {"jsonrpc": "2.0", "id": msg["id"], "result": result}
        except RpcError as e:
            err = {"code": e.code, "message": e.message}
            if e.data is not None: err["data"] = e.data
            return {"jsonrpc": "2.0", "id": msg["id"], "error": err}
        except Exception as e:   # one bad message must not end the server for the client
            return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32603, "message": f"Internal error: {e!r}"}}

    def dispatch(self, method, params):
        if method == "initialize":   # legacy handshake: answer the requested version if known, else the newest legacy one
            want = params.get("protocolVersion")
            return {"protocolVersion": want if want in LEGACY_VERSIONS else LEGACY_VERSIONS[0],
                    "capabilities": {"tools": {}}, "serverInfo": self.server_info(), "instructions": INSTRUCTIONS}
        meta = params.get("_meta") or {}
        modern = META_VERSION in meta
        if modern and meta[META_VERSION] not in ALL_VERSIONS:
            raise RpcError(-32022, "Unsupported protocol version", {"supported": ALL_VERSIONS, "requested": meta[META_VERSION]})
        if method == "server/discover":
            result = {"supportedVersions": ALL_VERSIONS, "capabilities": {"tools": {}},
                      "_meta": {"io.modelcontextprotocol/serverInfo": self.server_info()}, "instructions": INSTRUCTIONS}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": tool_list(self.read_only)}
        elif method == "tools/call":
            result = self.call_tool(params)
        else:
            raise RpcError(-32601, f"Method not found: {method}")
        if modern or method == "server/discover": result["resultType"] = "complete"
        return result

    def serve(self, inp, out):
        for raw in inp:
            line = raw.strip()
            if not line: continue
            try:
                msg = json.loads(line)
            except ValueError:
                resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            else:
                resp = self.handle(msg)
            if resp is not None:
                out.write(json.dumps(resp, ensure_ascii=False) + "\n"); out.flush()


def default_workspace():
    """The nearest folder upward from the current one that holds a .workbench folder, else the current folder."""
    d = os.path.realpath(os.getcwd())
    home = os.path.realpath(os.path.expanduser("~"))
    while True:
        # the home folder only when it is the current one: before 0.6.0 the default workspace was the current folder,
        # so a ~/.workbench from one start in ~ would otherwise make every folder below it that workspace
        if os.path.isdir(os.path.join(d, ".workbench")) and (d != home or d == os.path.realpath(os.getcwd())): return d
        up = os.path.dirname(d)
        if up == d: return os.getcwd()
        d = up


def connection_args(ap):
    ap.add_argument("--workspace", help="workspace of the workbench (default: the nearest folder upward with a .workbench folder, "
                                        "else the current folder)")
    ap.add_argument("--url", help="address of a running workbench instead of finding it through the workspace")
    ap.add_argument("--token", default=os.environ.get("OPENTOPAS_WORKBENCH_TOKEN"), help="API token for --url")
    ap.add_argument("--no-start", action="store_true", help="report an error instead of starting a workbench")


def connect(a):
    return Workbench(a.workspace or default_workspace(), a.url, a.token, start=not a.no_start)


def cli_enqueue(argv):
    """opentopas-workbench enqueue FILES...: queue parameter files of the workspace in the running workbench."""
    ap = argparse.ArgumentParser(prog="opentopas-workbench enqueue", description="Queue TOPAS parameter files in the workbench.")
    ap.add_argument("files", nargs="+", help="parameter files (relative to the current folder or absolute)")
    ap.add_argument("--binary", help="program name as in Settings (default: the only configured one)")
    ap.add_argument("--subdir", default="", help="write results into this subfolder next to each file")
    ap.add_argument("--skip-existing", action="store_true", help="skip files whose outputs already exist")
    connection_args(ap)
    a = ap.parse_args(argv)
    wb = connect(a)
    try: root = os.path.realpath(wb.get("/api/info")["root"])
    except ApiError as e: sys.exit(f"error: {e}")
    rel = []
    for f in a.files:
        p = os.path.realpath(f)
        if not (p == root or p.startswith(root + os.sep)): sys.exit(f"not inside the workspace {root}: {f}")
        rel.append(os.path.relpath(p, root).replace(os.sep, "/"))
    try:
        r = t_enqueue(wb, {"files": rel, "program": a.binary, "subfolder": a.subdir, "skip_existing": a.skip_existing})
    except ApiError as e:
        sys.exit(f"error: {e}")
    for f in r["added"]: print(f"queued   {f}  ({r['program']})")
    for f in r["skipped_with_results"]: print(f"skipped  {f}  (results exist)")


def cli_jobs(argv):
    """opentopas-workbench jobs: the queue of the running workbench."""
    ap = argparse.ArgumentParser(prog="opentopas-workbench jobs", description="Show the run queue of the workbench.")
    ap.add_argument("--history", type=int, default=5, help="finished jobs to show (default 5)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    connection_args(ap)
    a = ap.parse_args(argv)
    a.no_start = True   # showing the queue must not start a workbench
    wb = connect(a)
    try:
        r = t_queue_status(wb, {"history": a.history})
    except ApiError as e:
        sys.exit(f"error: {e}")
    if a.json:
        print(json.dumps(r, indent=1)); return
    for j in r["running"]:
        pr = f" {j['progress'] * 100:.0f} %" if j.get("progress") is not None else ""
        eta = f", ETA {j['eta'] // 60} min" if j.get("eta") is not None else ""
        print(f"running  #{j['id']} {j['file']} ({j['binary']}){pr}{eta}")
    ext = r["external"]
    for x in ext:
        print(f"external pid {x['pid']} {x['file'] or x['cwd']} ({x['program']}), {x.get('elapsed', 0) // 60} min")
    for j in r["waiting"]: print(f"waiting  #{j['id']} {j['file']} ({j['binary']})")
    if r["waiting_count"] > len(r["waiting"]): print(f"         ... {r['waiting_count'] - len(r['waiting'])} more waiting")
    for j in r["finished"]: print(f"{j['status']:<8} #{j['id']} {j['file']} ({j['binary']}) {j.get('end', '')}")
    if r.get("queue_eta"): print(f"queue ends about {r['queue_eta']['finish']}")
    if not (r["running"] or ext or r["waiting"] or r["finished"]): print("the queue is empty")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="opentopas-workbench mcp",
                                 description="MCP server (stdio) for a running OpenTOPAS Workbench. Starts one when none runs.")
    connection_args(ap)
    ap.add_argument("--read-only", action="store_true", help="offer only the tools that read")
    a = ap.parse_args(argv)
    wb = connect(a)
    # stdout carries only protocol messages; anything else that gets printed goes to stderr
    out = sys.stdout; sys.stdout = sys.stderr
    inp = open(sys.stdin.fileno(), encoding="utf-8", errors="replace", closefd=False)
    out = open(out.fileno(), "w", encoding="utf-8", closefd=False)
    try:
        Server(wb, a.read_only).serve(inp, out)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
