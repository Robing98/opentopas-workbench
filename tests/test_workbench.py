import importlib.util, json, os, struct, threading, urllib.request, urllib.error
from http.server import ThreadingHTTPServer

import pytest

from opentopas_workbench import server as wb

GEN_DIR = os.path.join(os.path.dirname(wb.__file__), "generators")


def load_gen(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(GEN_DIR, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def defaults(mod):
    return {f["key"]: f.get("default") for f in mod.FIELDS}


@pytest.fixture
def ws(tmp_path):
    wb.STATE["root"] = str(tmp_path)
    wb.STATE["cfg"] = {"binaries": [], "lang": "en"}
    wb.STATE["jobs"] = []
    return tmp_path


def write_csv(path, dims, rows):
    head = ["# TOPAS Version: test", "# Results for scorer: Dose"]
    head += [f"# {n} in {k} {'bin ' if k == 1 else 'bins'} of {w} cm" for n, k, w in dims]
    head.append("# DoseToMedium ( Gy ) : Sum   Mean")
    path.write_text("\n".join(head + [", ".join(map(str, r)) for r in rows]) + "\n")


# ------------------------------------------------------------------ lint
def test_lint_clean():
    txt = 's:Ge/A/Type = "TsBox"\nd:Ge/A/HLX = 1 cm\ni:Ge/A/ZBins = 10\nb:Ge/CheckForOverlaps = "False"\ndv:X/Y = 2 1 2 mm\n'
    assert wb.lint(txt) == []


@pytest.mark.parametrize("line,key", [
    ("Ge/A/HLX = 1 cm", "lint_notype"),
    ("s:Ge/A/Type = TsBox", "lint_string"),
    ("d:Ge/A/HLX = 1", "lint_unit"),
    ("i:Ge/A/ZBins = 1.5", "lint_int"),
    ("dv:X/Y = 3 1 2 mm", "lint_count"),
    ("this is not a parameter", "lint_syntax"),
])
def test_lint_finds(line, key):
    assert [i[2] for i in wb.lint(line)] == [key]


def test_scorer_outputs():
    assert wb.scorer_outputs('s:Sc/A/OutputFile = "a"\n s:Sc/B/OutputFile = "b"') == ["a", "b"]


# ------------------------------------------------------------------ CSV reader
def test_csv_1d(tmp_path):
    p = tmp_path / "d.csv"
    write_csv(p, [("X", 1, 10), ("Y", 1, 10), ("Z", 4, 0.5)], [[0, 0, k, k * 1.0, 0.1] for k in range(4)])
    r = wb.read_topas_csv(str(p))
    assert r["mode"] == "1d" and r["axis"] == 2 and r["stats"] == ["Sum", "Mean"]
    assert r["x"] == [0.25, 0.75, 1.25, 1.75] and r["y"] == [0, 1, 2, 3]
    assert wb.read_topas_csv(str(p), stat=1)["y"] == [0.1] * 4


def test_csv_2d_and_3d(tmp_path):
    p = tmp_path / "m.csv"
    write_csv(p, [("X", 2, 1), ("Y", 1, 10), ("Z", 3, 1)], [[i, 0, k, i * 10 + k] for i in range(2) for k in range(3)])
    r = wb.read_topas_csv(str(p))
    assert r["mode"] == "2d" and r["axes"] == [0, 2] and r["grid"] == [[0, 1, 2], [10, 11, 12]]
    q = tmp_path / "v.csv"
    write_csv(q, [("X", 2, 1), ("Y", 2, 1), ("Z", 2, 1)], [[i, j, k, 1] for i in range(2) for j in range(2) for k in range(2)])
    r = wb.read_topas_csv(str(q), collapse=1)
    assert r["axes"] == [0, 2] and r["grid"] == [[2, 2], [2, 2]]


def test_csv_without_header(tmp_path):
    p = tmp_path / "x.csv"; p.write_text("1,2,3\n")
    with pytest.raises(ValueError):
        wb.read_topas_csv(str(p))


# ------------------------------------------------------------------ paths
def test_resolve_stays_in_workspace(ws):
    assert wb.resolve("a/b") == os.path.join(os.path.realpath(ws), "a", "b")
    for bad in ("..", "../x", "/etc/passwd", "a/../../x"):
        with pytest.raises(ValueError):
            wb.resolve(bad)


def test_resolve_blocks_symlink_escape(ws, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    os.symlink(outside, ws / "link")
    with pytest.raises(ValueError):
        wb.resolve("link/file.txt")


# ------------------------------------------------------------------ generators
@pytest.mark.parametrize("name", ["pencil_beam", "ripple_filter"])
def test_generators_produce_clean_files(name):
    mod = load_gen(name)
    assert {"NAME", "DESCRIPTION", "FIELDS", "generate"} <= set(dir(mod))
    files = mod.generate(defaults(mod))
    texts = [(n, t) for n, t in files if isinstance(t, str)]
    assert texts
    for n, t in texts:
        assert n.endswith(".txt")
        assert [i for i in wb.lint(t) if i[1] == "error"] == [], n
        assert wb.scorer_outputs(t)


def test_alpha_gets_em_range():
    mod = load_gen("pencil_beam")
    v = defaults(mod); v.update(particles=["a", "c"], energies="200")
    out = dict(mod.generate(v))
    assert "EMRangeMax" in out["pb_a200.txt"] and "EMRangeMax" not in out["pb_c200.txt"]


def test_pin_stl_is_watertight():
    stl = load_gen("ripple_filter").pins_stl(4.0, half=3.0)
    n = struct.unpack("<I", stl[80:84])[0]
    assert len(stl) == 84 + 50 * n
    key = lambda p: tuple(round(c, 4) for c in p)
    edges = {}
    for k in range(n):
        v = struct.unpack("<9f", stl[84 + 50 * k + 12: 84 + 50 * k + 48])
        tri = [key(v[0:3]), key(v[3:6]), key(v[6:9])]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[(tri[a], tri[b])] = edges.get((tri[a], tri[b]), 0) + 1
    assert all(c == 1 and edges.get((b, a)) == 1 for (a, b), c in edges.items())


# ------------------------------------------------------------------ HTTP
@pytest.fixture
def srv(ws):
    s = ThreadingHTTPServer(("127.0.0.1", 0), wb.H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


def call(url, body=None, token=wb.TOKEN, host=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None)
    if token: req.add_header("X-Token", token)
    if host: req.add_header("Host", host)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_requires_token_and_local_host(srv):
    assert call(srv + "/api/ls", token=None)[0] == 403
    assert call(srv + "/api/ls", token="wrong")[0] == 403
    assert call(srv + "/api/ls", host="evil.example")[0] == 403
    assert call(srv + "/api/ls")[0] == 200


def test_http_generate_list_and_trash(srv, ws):
    v = defaults(load_gen("pencil_beam")); v["energies"] = "100"
    code, r = call(srv + "/api/generate", {"id": "pencil_beam", "values": v, "folder": "cases"})
    assert code == 200 and r["written"] == ["pb_p100.txt"]
    code, r = call(srv + "/api/generate", {"id": "pencil_beam", "values": v, "folder": "cases"})
    assert r["skipped"] == ["pb_p100.txt"]
    assert call(srv + "/api/params")[1]["files"] == ["cases/pb_p100.txt"]
    assert call(srv + "/api/trash", {"p": "cases/pb_p100.txt"})[0] == 200
    assert not (ws / "cases" / "pb_p100.txt").exists() and len(os.listdir(ws / "_trash")) == 1
    assert call(srv + "/api/write", {"p": "../evil.txt", "text": "x"})[0] == 400


# ------------------------------------------------------------------ 0.4 features
def test_includes_and_missing_include_lint(ws):
    (ws / "a").mkdir(); (ws / "common.txt").write_text("i:Ts/NumberOfThreads = 0\n")
    txt = "includeFile = ../common.txt nothere.txt\n"
    inc = wb.includes(txt, str(ws / "a"))
    assert inc == [[1, "../common.txt", "common.txt"], [1, "nothere.txt", None]]
    assert [i[2] for i in wb.lint(txt, str(ws / "a"))] == ["lint_include"]
    assert wb.lint(txt) == []  # without a base folder includes are not checked


def test_existing_results_finds_subfolders(ws):
    (ws / "c").mkdir(); (ws / "c" / "opentopas").mkdir(); (ws / "c" / "old").mkdir()
    txt = 's:Sc/A/OutputFile = "out_a"\ns:Sc/B/OutputFile = "out_b"\ni:So/Demo/NumberOfHistoriesInRun = 5\ns:Ge/X/Type = "TsBox"\n'
    (ws / "c" / "case.txt").write_text(txt)
    for n in ("out_a.csv", "out_b.csv"): (ws / "c" / "opentopas" / n).write_text("x")
    (ws / "c" / "old" / "out_a.csv").write_text("x")  # incomplete
    (ws / "c" / "out_a.txt").write_text("x")  # not an output type
    assert wb.list_params()["results"] == {"c/case.txt": ["opentopas"]}


def test_progress_and_hints(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("G4WT1 > Begin processing for Run: 0, History: 1000\nG4WT0 > Begin processing for Run: 0, History: 3000\n"
                   "This is not necessarily an error.\nTopas is exiting due to a serious error.\n")
    assert wb.log_progress(str(log)) == 3000
    assert wb.log_hints(str(log)) == ["Topas is exiting due to a serious error."]
    assert wb.total_histories("i:So/A/NumberOfHistoriesInRun = 10\ni:So/B/NumberOfHistoriesInRun = 5\n") == 15


def test_read_measured_formats(tmp_path):
    mcc = tmp_path / "m.mcc"
    mcc.write_text("BEGIN_SCAN_DATA\nBEGIN_SCAN 1\nSCAN_CURVETYPE=PDD\nBEGIN_DATA\n0.0 50\n1.0 60\n2.0 100\nEND_DATA\nEND_SCAN 1\n"
                   "BEGIN_SCAN 2\nSCAN_CURVETYPE=CROSSPLANE_PROFILE\nBEGIN_DATA\n-1 5\n1 5\nEND_DATA\nEND_SCAN 2\nEND_SCAN_DATA\n")
    r = wb.read_measured(str(mcc))
    assert r["scans"] == 2 and r["x"] == [0.0, 1.0, 2.0] and r["y"] == [50, 60, 100] and r["dims"][0]["name"] == "Depth"
    assert wb.read_measured(str(mcc), scan=1)["dims"][0]["name"] == "Position"
    asc = tmp_path / "i.asc"
    asc.write_text("$STOM\n%PTS 3\n= 0.0 0.0 0.0 10\n= 0.0 0.0 5.0 20\n= 0.0 0.0 10.0 15\n$ENOM\n")
    r = wb.read_measured(str(asc))
    assert r["x"] == [0.0, 5.0, 10.0] and r["y"] == [10, 20, 15]
    col = tmp_path / "c.dat"
    col.write_text("# unit: cm\ndepth;dose\n0,1;2\n0,2;3\n1.5\t4\n")
    r = wb.read_measured(str(col))
    assert r["x"] == [1.0, 2.0, 15.0] and r["y"] == [2.0, 3.0, 4.0]  # decimal commas in ; or tab separated files
    p = tmp_path / "topas.csv"
    write_csv(p, [("X", 1, 1), ("Y", 1, 1), ("Z", 2, 1)], [[0, 0, 0, 1], [0, 0, 1, 2]])
    assert not wb.read_any(str(p)).get("measured") and wb.read_any(str(col))["measured"]


def test_workspace_names(ws):
    (ws / "a.txt").write_text('s:Ge/Box/Type = "TsBox"\nd:Ge/Box/HLX = 1 cm\n')
    assert wb.workspace_names() == ["d:Ge/Box/HLX", "s:Ge/Box/Type"]


def test_beam_info():
    assert wb.beam_info('s:So/D/BeamParticle = "proton"\nd:So/D/BeamEnergy = 150 MeV\n') == ("p", 150.0)
    assert wb.beam_info('s:So/D/BeamParticle = "GenericIon(6,12)"\nd:So/D/BeamEnergy = 4800 MeV\n') == ("C-12", 400.0)
    assert wb.beam_info('s:So/D/BeamParticle = "alpha"\nd:So/D/BeamEnergy = 0.6 GeV\n') == ("He-4", 150.0)


def test_build_index_links_outputs_to_parameter_file(ws):
    (ws / "c" / "out").mkdir(parents=True)
    (ws / "c" / "case_rifi0.txt").write_text('s:So/D/BeamParticle = "proton"\nd:So/D/BeamEnergy = 100 MeV\ns:Sc/A/OutputFile = "IDD_case"\n')
    rows = [[0, 0, k, v] for k, v in enumerate([1, 1, 1.2, 1.5, 3, 4, 2, 0.2, 0, 0])]
    write_csv(ws / "c" / "out" / "IDD_case.csv", [("X", 1, 1), ("Y", 1, 1), ("Z", 10, 0.1)], rows)
    (ws / "c" / "out" / "plot.png").write_bytes(b"\x89PNG")
    items = {i["name"]: i for i in wb.build_index()}
    d = items["IDD_case.csv"]
    assert d["particle"] == "p" and d["energy"] == 100.0 and d["param"] == "c/case_rifi0.txt" and d["kind"] == "data"
    assert abs(d["r80"] - 5.9) < 1e-6 and d["peak"] == 5.5 and items["plot.png"]["kind"] == "image"
    assert (ws / ".workbench" / "index.json").exists()


def test_keep_awake_linux(ws, tmp_path, monkeypatch):
    fake = tmp_path / "bin"; fake.mkdir()
    (fake / "systemd-inhibit").write_text("#!/bin/sh\nexec sleep 30\n"); os.chmod(fake / "systemd-inhibit", 0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    monkeypatch.setattr(wb.KeepAwake, "platform", staticmethod(lambda: "linux"))
    k = wb.KeepAwake(); k.start()
    assert k.active() and k.status()["method"] == "systemd-inhibit"
    k.stop()
    assert not k.active()


def test_keep_awake_wsl_command(ws, monkeypatch):
    calls = []
    class P:
        def __init__(self, args, **kw): calls.append(args)
        def poll(self): return None
        def terminate(self): pass
    monkeypatch.setattr(wb.KeepAwake, "platform", staticmethod(lambda: "wsl"))
    monkeypatch.setattr(wb.subprocess, "Popen", P)
    monkeypatch.setattr(wb.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "C:\\\\ws\\\\.workbench\\\\keepawake.flag"})())
    k = wb.KeepAwake(); k.start()
    import base64
    script = base64.b64decode(calls[0][-1]).decode("utf-16-le")
    assert calls[0][0] == "powershell.exe" and "SetThreadExecutionState(0x80000001)" in script and "keepawake.flag" in script
    assert (ws / ".workbench" / "keepawake.flag").exists()
    k.stop()
    assert not (ws / ".workbench" / "keepawake.flag").exists()


def test_make_variants(ws):
    (ws / "c").mkdir()
    (ws / "c" / "base.txt").write_text('d:So/D/BeamEnergy = 100 MeV # energy\ns:Ge/P/Material = "G4_WATER"\n'
                                       's:Sc/A/OutputFile = "dose"\ni:So/D/NumberOfHistoriesInRun = 10\n')
    out = dict(wb.make_variants("c/base.txt", "d:So/D/BeamEnergy", ["70", "150.5 MeV"]))
    assert list(out) == ["c/base_70.txt", "c/base_150p5MeV.txt"]
    assert "d:So/D/BeamEnergy = 70 MeV # energy" in out["c/base_70.txt"] and 'OutputFile = "dose_70"' in out["c/base_70.txt"]
    assert "= 150.5 MeV" in out["c/base_150p5MeV.txt"]
    m = dict(wb.make_variants("c/base.txt", "Ge/P/Material", ["G4_PLEXIGLASS"], "{name}-{value}"))
    assert 's:Ge/P/Material = "G4_PLEXIGLASS"' in m["c/base-G4PLEXIGLASS.txt"]
    with pytest.raises(ValueError):
        wb.make_variants("c/base.txt", "So/D/Nope", ["1"])


def test_http_views_zip_variants(srv, ws):
    assert call(srv + "/api/views")[1] == {"views": []}
    assert call(srv + "/api/views", {"views": [{"name": "a", "pick": ["x.csv"]}]})[0] == 200
    assert call(srv + "/api/views")[1]["views"][0]["name"] == "a"
    (ws / "f.txt").write_text('d:So/D/BeamEnergy = 1 MeV\ns:Sc/A/OutputFile = "o"\n')
    code, r = call(srv + "/api/variants", {"src": "f.txt", "param": "So/D/BeamEnergy", "values": ["2", "3"]})
    assert code == 200 and r["written"] == ["f_2.txt", "f_3.txt"]
    import zipfile, io
    req = urllib.request.Request(srv + "/api/zip", data=json.dumps({"paths": ["f_2.txt", "f_3.txt"]}).encode(),
                                 headers={"X-Token": wb.TOKEN, "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert sorted(zipfile.ZipFile(io.BytesIO(resp.read())).namelist()) == ["f_2.txt", "f_3.txt"]


def test_queue_estimate(ws):
    for name, e in (("a.txt", 100), ("b.txt", 200), ("c.txt", 100)):
        (ws / name).write_text(f"d:So/D/BeamEnergy = {e} MeV\ni:So/D/NumberOfHistoriesInRun = 1000\n")
    jobs = [{"file": "a.txt", "binary": "t", "status": "done", "seconds": 100, "total": 1000, "energy": 100.0},
            {"file": "b.txt", "binary": "t", "status": "queued"},
            {"file": "c.txt", "binary": "t", "status": "queued"}]
    est = wb.queue_estimate(jobs)
    assert est["seconds"] == 300 and est["unknown"] == 0   # 200 MeV takes twice as long as 100 MeV
    assert wb.queue_estimate([{"file": "b.txt", "binary": "t", "status": "queued"}]) is None


def test_scan_include_exclude(ws):
    for d in ("a", "b", "b/old", "tools"):
        (ws / d).mkdir(parents=True, exist_ok=True)
    for d in ("a", "b", "b/old", "tools", ""):
        (ws / d / "case.txt").write_text('s:Ge/X/Type = "TsBox"\n')
    wb.invalidate_cache()
    assert len(wb.list_param_files()) == 5
    wb.STATE["cfg"]["exclude"] = ["tools", "*/old"]; wb.invalidate_cache()
    assert sorted(wb.list_param_files()) == ["a/case.txt", "b/case.txt", "case.txt"]
    wb.STATE["cfg"]["include"] = ["b"]; wb.invalidate_cache()
    assert wb.list_param_files() == ["b/case.txt"]


def test_scope_per_tab(ws):
    for d in ("a", "b/sub"):
        (ws / d).mkdir(parents=True, exist_ok=True)
        (ws / d / "case.txt").write_text('s:Ge/X/Type = "TsBox"\n')
    (ws / "top.txt").write_text('s:Ge/X/Type = "TsBox"\n')
    wb.invalidate_cache()
    assert wb.list_param_files("b") == ["b/sub/case.txt"]
    assert len(wb.list_param_files("")) == 3
    assert wb.list_params("a")["files"] == ["a/case.txt"]
    assert wb.build_index("a") == [] or all(i["path"].startswith("a/") for i in wb.build_index("a"))
    with pytest.raises(ValueError):
        wb.scope_dir("missing")
    with pytest.raises(Exception):
        wb.scope_dir("../x")
