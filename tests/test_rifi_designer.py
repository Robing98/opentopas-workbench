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
