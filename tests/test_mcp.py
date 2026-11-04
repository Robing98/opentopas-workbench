"""The MCP adapter (opentopas_workbench/mcp.py) driven over pipes against a workbench served from this test process."""
import json, os, signal, stat, subprocess, sys, threading, time, urllib.request
from http.server import ThreadingHTTPServer

import pytest

from opentopas_workbench import mcp, server as wb

PARAM = 's:So/B/BeamParticle = "proton"\nd:So/B/BeamEnergy = 100 MeV\ns:Sc/D/OutputFile = "IDD"\n'
INIT = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}
MODERN_META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28", "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "0"},
               "io.modelcontextprotocol/clientCapabilities": {}}
WRITE_TOOLS = {"write_file", "generate", "enqueue", "dequeue", "stop"}


@pytest.fixture
def ws(tmp_path):
    wb.STATE["root"] = str(tmp_path)
    wb.STATE["cfg"] = {"binaries": [], "lang": "en"}
    wb.STATE["jobs"] = []; wb.STATE["proc"] = None; wb.STATE["next_id"] = 1
    wb.invalidate_cache()
    wb._EXTERNAL["t"] = 0.0
    return tmp_path


@pytest.fixture
def srv(ws):
    s = ThreadingHTTPServer(("127.0.0.1", 0), wb.H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://localhost:{s.server_address[1]}/"
    wb.write_server_file(url)   # what main() does, so the adapter finds this server through the workspace
    yield url
    s.shutdown()


class Client:
    """An MCP client over the pipes of `opentopas-workbench mcp`."""
    def __init__(self, ws, *extra):
        self.p = subprocess.Popen([sys.executable, "-m", "opentopas_workbench", "mcp", "--workspace", str(ws), *extra],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.n = 0

    def send(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None: msg["params"] = params
        if not notify: self.n += 1; msg["id"] = self.n
        self.p.stdin.write(json.dumps(msg) + "\n"); self.p.stdin.flush()
        if notify: return None
        resp = json.loads(self.p.stdout.readline())
        assert resp["id"] == self.n
        return resp

    def init(self):
        r = self.send("initialize", INIT)["result"]
        self.send("notifications/initialized", notify=True)
        return r

    def tool(self, name, **args):
        r = self.send("tools/call", {"name": name, "arguments": args})["result"]
        return r, (r.get("structuredContent") if not r["isError"] else r["content"][0]["text"])

    def close(self):
        self.p.stdin.close()
        assert self.p.wait(timeout=10) == 0


@pytest.fixture
def client(srv, ws):
    c = Client(ws, "--no-start"); c.init()
    yield c
    c.close()


# ------------------------------------------------------------------ protocol
def test_legacy_handshake_negotiates_the_version(srv, ws):
    c = Client(ws, "--no-start")
    r = c.send("initialize", INIT)["result"]
    assert r["protocolVersion"] == "2025-06-18" and "tools" in r["capabilities"] and r["serverInfo"]["name"] == "opentopas-workbench"
    assert c.send("initialize", dict(INIT, protocolVersion="1999-01-01"))["result"]["protocolVersion"] == mcp.LEGACY_VERSIONS[0]
    c.send("notifications/initialized", notify=True)   # no answer; the next line read belongs to the ping
    assert c.send("ping")["result"] == {}
    assert c.send("no/such/method")["error"]["code"] == -32601
    c.close()


def test_modern_discover_and_unsupported_version(srv, ws):
    c = Client(ws, "--no-start")
    r = c.send("server/discover", {"_meta": MODERN_META})["result"]
    assert r["resultType"] == "complete" and "2026-07-28" in r["supportedVersions"] and "tools" in r["capabilities"]
    assert r["_meta"]["io.modelcontextprotocol/serverInfo"]["version"] == wb.VERSION
    r = c.send("tools/list", {"_meta": MODERN_META})["result"]
    assert r["resultType"] == "complete" and r["tools"]
    e = c.send("tools/list", {"_meta": dict(MODERN_META, **{"io.modelcontextprotocol/protocolVersion": "1900-01-01"})})["error"]
    assert e["code"] == -32022 and e["data"]["requested"] == "1900-01-01" and "2026-07-28" in e["data"]["supported"]
    c.close()


def test_tools_list_has_schemas_and_hints(client):
    tools = {t["name"]: t for t in client.send("tools/list")["result"]["tools"]}
    assert WRITE_TOOLS <= set(tools) and {"info", "list_cases", "read_file", "list_results", "get_curve", "queue_status", "job_log"} <= set(tools)
    for t in tools.values():
        assert t["inputSchema"]["type"] == "object" and t["description"]
        assert t["annotations"]["readOnlyHint"] == (t["name"] not in WRITE_TOOLS)


def test_unknown_tool_is_a_protocol_error(client):
    assert client.send("tools/call", {"name": "rm_rf", "arguments": {}})["error"]["code"] == -32602


# ------------------------------------------------------------------ read tools
def test_read_file_with_lint(client, ws):
    (ws / "a.txt").write_text(PARAM + "d:Ge/A/HLX = 1\n")
    r, out = client.tool("read_file", path="a.txt")
    assert not r["isError"] and out["text"].startswith("s:So/B")
    assert out["lint"] == [{"line": 4, "severity": "warn", "message": "double value without a unit", "detail": "1"}]
    assert json.loads(r["content"][0]["text"]) == out   # the same data as text for clients without structuredContent


def test_api_error_comes_back_as_tool_error(client, ws):
    r, msg = client.tool("read_file", path="../outside.txt")
    assert r["isError"] and "workspace" in msg.lower()
    r, msg = client.tool("read_file")
    assert r["isError"] and "missing argument: path" in msg


def write_idd(path, n=1000):
    """A Bragg-like depth dose in n bins of 0.01 cm with its peak at bin 700."""
    rows = [[0, 0, k, (1 + k / 1000) if k < 700 else max(0.0, 1.7 - (k - 700) * 0.1)] for k in range(n)]
    head = ["# TOPAS Version: test", "# Results for scorer: D", "# X in 1 bin of 10 cm", "# Y in 1 bin of 10 cm",
            f"# Z in {n} bins of 0.01 cm", "# DoseToMedium ( Gy ) : Sum"]
    path.write_text("\n".join(head + [", ".join(map(str, r)) for r in rows]) + "\n")


def test_results_and_downsampled_curve(client, ws):
    (ws / "c").mkdir(); (ws / "c" / "a.txt").write_text(PARAM); write_idd(ws / "c" / "IDD.csv")
    _, out = client.tool("list_results")
    assert out["total"] == 1
    it = out["results"][0]
    assert it["path"] == "c/IDD.csv" and it["param"] == "c/a.txt" and it["particle"] == "p" and it["energy"] == 100
    assert it["peak"] == pytest.approx(70.05) and it["r80"] == pytest.approx(70.39)   # bin centers: (700 + 0.5) x 0.1 mm
    _, cur = client.tool("get_curve", path="c/IDD.csv", max_points=50)
    assert cur["points"] == 1000 and 50 <= len(cur["x"]) <= 51 and cur["x_unit"] == "cm"
    assert max(cur["y"]) == pytest.approx(1.7) and cur["beam"]["param"] == "c/a.txt"


def test_list_cases_status_filter(client, ws):
    for n in ("a", "b"): (ws / f"{n}.txt").write_text(PARAM)
    wb.STATE["jobs"] = [{"id": 1, "file": "a.txt", "binary": "t", "status": "failed"}]
    _, out = client.tool("list_cases")
    assert [c["file"] for c in out["cases"]] == ["a.txt", "b.txt"] and out["cases"][0]["last_run"] == {"t": "failed"}
    assert [c["file"] for c in client.tool("list_cases", status="failed")[1]["cases"]] == ["a.txt"]
    assert [c["file"] for c in client.tool("list_cases", status="not_run")[1]["cases"]] == ["b.txt"]
    assert [c["file"] for c in client.tool("list_cases", filter="b*")[1]["cases"]] == ["b.txt"]


def test_job_log_tail(client, ws):
    (ws / ".workbench" / "logs").mkdir(parents=True)
    (ws / ".workbench" / "logs" / "job00001.log").write_text("".join(f"line {i}\n" for i in range(100)))
    wb.STATE["jobs"] = [{"id": 1, "file": "a.txt", "binary": "t", "status": "done", "log": ".workbench/logs/job00001.log"}]
    _, out = client.tool("job_log", job_id=1, lines=3)
    assert out["tail"] == "line 97\nline 98\nline 99" and out["job"]["status"] == "done"
    assert client.tool("job_log", job_id=7)[0]["isError"]


# ------------------------------------------------------------------ write tools
def test_write_generate_enqueue_dequeue(client, ws):
    r, out = client.tool("write_file", path="new.txt", text=PARAM)
    assert not r["isError"] and (ws / "new.txt").read_text() == PARAM
    _, out = client.tool("generate", generator="pencil_beam", values={"energies": "100"}, folder="cases")
    assert out["written"] == ["cases/pb_p100.txt"] and (ws / "cases" / "pb_p100.txt").exists()
    assert client.tool("generate", generator="pencil_beam", values={"energies": "100"}, folder="cases")[1]["skipped_existing"] == ["cases/pb_p100.txt"]
    r, msg = client.tool("generate", generator="pencil_beam", values={"nonsense": 1})
    assert r["isError"] and "nonsense" in msg
    r, msg = client.tool("enqueue", files=["cases/pb_p100.txt"])
    assert r["isError"] and "name the program" in msg   # no program configured in this workspace
    _, out = client.tool("enqueue", files=["cases/pb_p100.txt", "new.txt"], program="fake")
    assert out["added"] == ["cases/pb_p100.txt", "new.txt"]
    _, q = client.tool("queue_status")
    assert q["waiting_count"] == 2 and q["counts"] == {"queued": 2}
    assert client.tool("dequeue", all=True)[1]["removed"] == 2


def test_read_only_hides_and_refuses_writes(srv, ws):
    c = Client(ws, "--no-start", "--read-only"); c.init()
    names = {t["name"] for t in c.send("tools/list")["result"]["tools"]}
    assert not names & WRITE_TOOLS and "read_file" in names
    r, msg = c.tool("write_file", path="x.txt", text="x")
    assert r["isError"] and "--read-only" in msg and not (ws / "x.txt").exists()
    c.close()


# ------------------------------------------------------------------ finding and starting the workbench
def test_no_start_reports_how_to_start(ws):
    c = Client(ws, "--no-start"); c.init()
    r, msg = c.tool("info")
    assert r["isError"] and "no workbench is running" in msg and "opentopas-workbench --workspace" in msg
    c.close()


def test_stale_server_file_is_ignored(ws):
    p = subprocess.Popen([sys.executable, "-c", "pass"]); p.wait()   # a pid that no longer exists
    (ws / ".workbench").mkdir()
    (ws / ".workbench" / "server.json").write_text(json.dumps({"url": "http://localhost:1/", "token": "x", "pid": p.pid}))
    assert mcp.read_server_file(str(ws)) is None


def wait_for(cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        v = cond()
        if v: return v
        time.sleep(0.1)
    raise AssertionError("timed out")


def test_server_file_written_and_removed(tmp_path):
    p = subprocess.Popen([sys.executable, "-m", "opentopas_workbench", "--workspace", str(tmp_path), "--no-browser", "--port", "0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        f = tmp_path / ".workbench" / "server.json"
        s = wait_for(lambda: f.exists() and json.loads(f.read_text()))
        assert s["pid"] == p.pid and stat.S_IMODE(f.stat().st_mode) == 0o600
        req = urllib.request.Request(s["url"].replace("localhost", "127.0.0.1") + "api/info", headers={"X-Token": s["token"]})
        with urllib.request.urlopen(req) as r: assert os.path.realpath(json.loads(r.read())["root"]) == os.path.realpath(tmp_path)
        p.send_signal(signal.SIGTERM); p.wait(timeout=10)
        assert not f.exists()
    finally:
        if p.poll() is None: p.kill()


def test_adapter_starts_a_workbench_when_none_runs(tmp_path):
    c = Client(tmp_path); c.init()
    pid = None
    try:
        r, out = c.tool("info")
        assert not r["isError"] and os.path.realpath(out["workspace"]) == os.path.realpath(tmp_path)
        pid = json.loads((tmp_path / ".workbench" / "server.json").read_text())["pid"]
        assert f"pid {pid}" in out["note"]
        c.close()
        os.kill(pid, 0)   # still running after the MCP client went away: queued runs must outlive it
    finally:
        if pid: os.kill(pid, signal.SIGTERM)
    wait_for(lambda: not (tmp_path / ".workbench" / "server.json").exists(), 10)


# ------------------------------------------------------------------ CLI
def cli(ws, *args):
    return subprocess.run([sys.executable, "-m", "opentopas_workbench", *args, "--no-start"], cwd=ws / "cases",
                          capture_output=True, text=True, timeout=60)


def test_cli_enqueue_and_jobs(srv, ws):
    (ws / "cases").mkdir(); (ws / "cases" / "a.txt").write_text(PARAM)
    r = cli(ws, "enqueue", "a.txt", "--binary", "fake")   # relative to the current folder; the workspace is found upward
    assert r.returncode == 0 and "queued   cases/a.txt  (fake)" in r.stdout
    assert [j["file"] for j in wb.STATE["jobs"]] == ["cases/a.txt"]
    r = cli(ws, "jobs")
    assert r.returncode == 0 and "waiting  #1 cases/a.txt (fake)" in r.stdout
    r = cli(ws, "enqueue", str(ws.parent / "elsewhere.txt"), "--binary", "fake")
    assert r.returncode != 0 and "not inside the workspace" in r.stderr


# ------------------------------------------------------------------ runs outside the queue
def fake_proc(root, pid, argv, cwd, ppid=1, pgid=None, start=500):
    d = root / str(pid); d.mkdir()
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    fields = ["S", ppid, pgid or pid] + [0] * 16 + [start]   # field 22 of /proc/PID/stat is the start time in clock ticks
    (d / "stat").write_text(f"{pid} ({os.path.basename(argv[0])}) " + " ".join(map(str, fields)) + "\n")
    os.symlink(cwd, d / "cwd")


def test_external_runs_listed_read_only(srv, ws, tmp_path_factory, monkeypatch):
    proc = tmp_path_factory.mktemp("proc"); other = tmp_path_factory.mktemp("other")
    (ws / "cases").mkdir(); (ws / "cases" / "a.txt").write_text(PARAM)
    (proc / "stat").write_text(f"cpu 1 2 3\nbtime {int(time.time()) - 1000}\n")
    fake_proc(proc, 100, ["/bin/bash", "/opt/topas/bin/topas", "a.txt"], ws / "cases")      # wrapper script
    fake_proc(proc, 101, ["/opt/topas/bin/topas", "a.txt"], ws / "cases", ppid=100)         # the binary it starts
    fake_proc(proc, 200, ["topas", "b.txt"], other)                                         # another folder
    fake_proc(proc, 300, ["python3", "a.txt"], ws)                                          # not TOPAS
    fake_proc(proc, 400, ["topas4", "cases/a.txt"], ws, pgid=400)                           # a job of the queue
    fake_proc(proc, 500, ["topas4", str(ws / "cases" / "a.txt")], other)                    # absolute path into the workspace
    monkeypatch.setattr(wb, "PROC_DIR", str(proc))
    wb.STATE["cfg"]["binaries"] = [{"name": "TOPAS 3.9", "cmd": "/opt/topas/bin/topas"}]
    wb.STATE["proc"] = type("P", (), {"pid": 400})()
    runs = wb.external_runs(max_age=0)
    assert [(r["pid"], r["program"], r["file"], r["cwd"]) for r in runs] == [
        (100, "TOPAS 3.9", "cases/a.txt", "cases"), (500, "topas4", "cases/a.txt", str(other))]
    assert runs[0]["elapsed"] == pytest.approx(1000 - 500 / os.sysconf("SC_CLK_TCK"), abs=5)
    # end to end: the HTTP answer and the MCP tool carry them
    c = Client(ws, "--no-start"); c.init()
    _, q = c.tool("queue_status")
    assert [x["pid"] for x in q["external"]] == [100, 500] and q["running"] == []
    c.close()
    wb.STATE["proc"] = None


def test_external_runs_empty_without_proc(ws, monkeypatch):
    monkeypatch.setattr(wb, "PROC_DIR", str(ws / "no-proc"))
    assert wb.external_runs(max_age=0) == []
