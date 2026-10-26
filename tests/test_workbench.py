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
