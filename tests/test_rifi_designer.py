"""Tests for the rifi/v1 filter definitions (generators/_rifi.py): templates, kernel, TOPAS output, SVG, library."""
import hashlib, importlib.util, json, math, os, re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "opentopas_workbench", "generators")


def load(name, path=None):
    spec = importlib.util.spec_from_file_location("t_" + name, path or os.path.join(GEN, name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


R = load("_rifi")
GOLDEN = json.load(open(os.path.join(ROOT, "tests", "data", "rifi_golden.json")))
sha = lambda t: hashlib.sha256(t if isinstance(t, bytes) else t.encode()).hexdigest()


# ------------------------------------------------------------------ reproducibility of the old variants
def test_variant_blocks_unchanged():
    """Every variant writes the same geometry as before the definitions, for several periods (old studies rerun)."""
    rf = load("ripple_filter")
    bad = [k for k, h in GOLDEN["block"].items() if sha(rf.variant_block(k.split("|")[0], float(k.split("|")[1]))) != h]
    assert not bad
    assert len(GOLDEN["block"]) == 140


def test_cases_and_nozzle_unchanged():
    rf, nz = load("ripple_filter"), load("nozzle")
    for v, h in GOLDEN["case"].items():
        n, t = rf.case("p", 100, v, 1000, 0.2, 1.5, "mess")
        assert sha(n + t) == h, v
    vals = {f["key"]: f.get("default") for f in nz.FIELDS}
    vals["variants"] = [v for v in rf.VARIANTS if not v.startswith("lib-")]
    files = dict(nz.generate(vals))
    assert set(files) == set(GOLDEN["nozzle"])
    assert all(sha(files[n]) == h for n, h in GOLDEN["nozzle"].items())


def test_templates_go_through_definitions():
    rf = load("ripple_filter")
    for v in rf.TEMPLATES:
        d = rf.template(v, 1.5)
        assert d is not None and R.validate(d)["name"] == v
    assert rf.template("pin4", 1.5) is None and rf.template("stp4s10", 1.5) is None and rf.template("stp4g", 1.5) is None


# ------------------------------------------------------------------ definitions
def plate(step, **kw):
    return dict(dict(period=1.5, base=0.3, steps=[[0.5, 0.0], [0.5, step]]), **kw)


def defn(*layers, name="x"):
    return R.validate({"schema": "rifi/v1", "name": name, "layers": list(layers)})


@pytest.mark.parametrize("bad, msg", [
    ({"name": "a b", "layers": [plate(1)]}, "name"),
    ({"name": "a", "layers": []}, "layers"),
    ({"name": "a", "layers": [dict(period=1.5, steps=[[0.5, 0], [0.4, 1]])]}, "add up to 1"),
    ({"name": "a", "layers": [dict(period=1.5, profile=[[0, 0], [1, 1], [0.9, 0]])]}, "rise strictly"),
    ({"name": "a", "layers": [dict(period=1.5, profile=[[0, 0], [2, 1]])]}, "above the period"),
    ({"name": "a", "layers": [dict(period=-1, profile=[[0, 0], [1, 1]])]}, "period"),
    ({"name": "a", "schema": "rifi/v9", "layers": [plate(1)]}, "schema"),
])
def test_validate_refuses(bad, msg):
    with pytest.raises(ValueError, match=msg):
        R.validate(bad)


def test_steps_give_the_old_profiles():
    rf = load("ripple_filter")
    for lam in (1.5, 1.3, 2.0):
        xs, zs, w = R.layer_points(R.validate({"name": "b", "layers": [plate(1.0, period=lam)]})["layers"][0])
        assert (xs, zs) == tuple(map(list, rf.binary(1.0, lam))) and w == lam
        p = 4 * round(lam / 4, 3)
        L = R.validate({"name": "s", "layers": [dict(period=p, steps=[[0.25, 1], [0.25, 2], [0.25, 3], [0.25, 0]])]})["layers"][0]
        sx, sz, sw = rf.staircase(4.0, lam)
        xs, zs, w = R.layer_points(L)
        assert [round(v, 6) for v in xs] == [round(v, 6) for v in sx] and zs == sz and w == sw


def test_kernel_aligned_adds_crossed_convolves():
    a = R.kernel(defn(plate(2.0), plate(1.0, gap=0)))                    # aligned: same half thick on both, 2 levels
    levels = [round(t, 1) for t, w in zip(a["t"], a["w"]) if w > 0.01]
    assert len(levels) == 2 and a["mean"] == pytest.approx(1.16 * (0.6 + 1.5), abs=0.01)
    c = R.kernel(defn(plate(2.0), plate(1.0, rotz=90, gap=15)))         # crossed: 4 equally frequent levels
    w = [x for x in c["w"] if x > 0.01]
    assert len(w) == 4 and all(abs(x - 0.25) < 0.01 for x in w)
    assert c["mean"] == pytest.approx(1.16 * 2.1, abs=0.01)
    assert c["sigma"] == pytest.approx(1.16 * math.sqrt(1.0 + 0.25), abs=0.01)
    assert R.design_spacing(c["t"], c["w"]) == pytest.approx(1.16 * 4, abs=0.02)


def test_kernel_matches_the_shape_comparison():
    """Means and sigmas of task 3176 (kernfit, TOPAS parameter files) for the templates at 1.5 mm."""
    rf = load("ripple_filter")
    study = {"t4": (2.58, 1.316), "bin4": (1.508, 1.159), "stk4": (2.44, 1.296), "stk4x3": (3.08, 1.328), "pair4": (2.746, 0.854)}
    for v, (m, s) in study.items():
        k = R.kernel(R.validate(rf.template(v, 1.5)))
        assert k["mean"] == pytest.approx(m, abs=0.01) and k["sigma"] == pytest.approx(s, abs=0.01), v


def test_ripple_matches_the_shape_comparison_model():
    """ripple_<delta>_model of auswertung.csv (task 3176), protons: the designer's model on the embedded curves."""
    rf = load("ripple_filter")
    want = {("t4", "110"): (1.74, 2.59, 1.26), ("bin4", "63"): (31.44, 22.44, 20.13), ("stk4", "63"): (10.68, 10.47, 2.37)}
    for (v, e), vals in want.items():
        k = R.kernel(R.validate(rf.template(v, 1.5)))
        got = R.ripple_table(k["t"], k["w"], [3.0, 4.0, 4.64])[e]["filter"]
        assert got == pytest.approx(vals, abs=1.0), (v, e, got)


def test_lateral_scales_with_root_of_mean():
    a, b = R.lateral(1.0), R.lateral(4.0)
    assert set(a) == {"63", "80", "94", "110"}
    assert all(b[e] == pytest.approx(2 * a[e], abs=0.02) for e in a)
    assert a["63"] > a["110"]


def test_hints():
    h = R.hints(defn(plate(2.0), plate(1.0, gap=5)))
    text = " ".join(x["en"] for x in h)
    assert "have air between them" in text and "base plate 0.3 mm matches no standard plate" in text
    assert "all steps match standard plates" in text
    h = R.hints(defn(plate(0.6, base=0), plate(1.0, rotz=90, gap=5, base=0)))
    text = " ".join(x["en"] for x in h)
    assert "air between" not in text and "steps of 0.6 mm match no standard plate" in text and "no base plate" in text


# ------------------------------------------------------------------ TOPAS output, SVG, library
def test_topas_block_of_a_definition():
    rf = load("ripple_filter")
    d = defn(dict(period=2.0, base=0.5, profile=[[0, 0], [1.0, 2.0], [2.0, 0]], shift=0.25),
             plate(1.0, period=2.0, rotz=90, gap=10, base=0))
    text = rf.definition_block(d)
    assert text.count('"TsRidgeFilter"') == 2
    assert "dv:Ge/RiFi1/ZPoints = 3 0.500 2.500 0.500 mm" in text and "d:Ge/RiFi1/TransX   = 0.2500 mm" in text
    tz2 = float(re.search(r"d:Ge/RiFi2/TransZ   = ([\d.]+) cm", text).group(1))
    assert tz2 == pytest.approx(80.0 - (2.5 / 2 + 10 + 1.0 / 2) / 10)
    assert "d:Ge/RiFi2/RotZ     = 90 deg" in text and "dv:Ge/RiFi2/Displacement = 61 " in text


def test_svg_mask():
    import xml.etree.ElementTree as ET
    d = defn(plate(2.0), plate(1.0, rotz=90, gap=15))
    root = ET.fromstring(R.svg_mask(d, 1))
    ns = "{http://www.w3.org/2000/svg}"
    assert root.get("width") == "140mm" and root.get("viewBox") == "0 0 140 140"
    assert len(root.findall(f"{ns}circle")) == 4
    g = root.find(f"{ns}g")
    assert "rotate(-90" in g.get("transform")
    xs = sorted(float(r.get("x")) for r in g.findall(f"{ns}rect"))
    assert len(xs) >= 2 * 80 and 10.0 <= xs[0] and xs[-1] <= 130.0


def test_library_variant_in_both_generators(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENTOPAS_WB_DIR", str(tmp_path))
    d = {"schema": "rifi/v1", "name": "mine", "title": "my stack", "layers": [plate(2.0), plate(1.0, rotz=90, gap=15)]}
    R.save(d)
    assert json.load(open(tmp_path / "rifi" / "mine.json"))["name"] == "mine"
    rf, nz = load("ripple_filter"), load("nozzle")
    assert "lib-mine" in rf.VARIANTS and "lib-mine" in [o[0] for f in nz.FIELDS if f["key"] == "variants" for o in f["options"]]
    files = dict(rf.generate({"ions": ["p"], "energies": "100", "variants": ["lib-mine"], "histories": 1000, "spread": 0.2}))
    text = files["mess_p100_lib-mine.txt"]
    assert text.count('"TsRidgeFilter"') == 2 and "my stack (library)" in text
    vals = {f["key"]: f.get("default") for f in nz.FIELDS}
    vals["variants"] = ["lib-mine"]
    names = [n for n, _ in nz.generate(vals)]
    assert names and all("lib-mine" in n for n in names)
    R.remove("mine")
    assert "lib-mine" not in load("ripple_filter").VARIANTS


def test_server_rifi_api(tmp_path):
    import sys
    sys.path.insert(0, ROOT)
    from opentopas_workbench import server
    server.STATE["root"] = str(tmp_path)
    lst = server.rifi_api("list", {})
    assert [d["name"] for d in lst["templates"]][:3] == ["t3", "t4", "t6"] and lst["library"] == []
    d = next(x for x in lst["templates"] if x["name"] == "stk4")
    r = server.rifi_api("eval", {"def": d})
    assert r["mean"] == pytest.approx(2.44, abs=0.01) and r["design"] == pytest.approx(4.64, abs=0.05)
    assert set(r["ripple"]) == {"63", "80", "94", "110"} and len(r["ripple"]["63"]["filter"]) == len(r["deltas"])
    assert abs(sum(w for _, w in r["kernel"]) - 1) < 1e-3 and "RiFi2" in r["topas"]
    saved = server.rifi_api("save", {"def": dict(d, name="kopie")})
    assert saved["variant"] == "lib-kopie" and [x["name"] for x in server.rifi_api("list", {})["library"]] == ["kopie"]
    assert [f["name"] for f in server.rifi_api("svg", {"def": d})["files"]] == ["stk4_lage1.svg", "stk4_lage2.svg"]
    server.rifi_api("delete", {"name": "kopie"})
    assert server.rifi_api("list", {})["library"] == []
    with pytest.raises(ValueError):
        server.rifi_api("eval", {"def": {"name": "x", "layers": []}})
