"""Physics in the run detail: the TOPAS physics list and cuts against the FLUKA defaults and thresholds of the same
case, read from the inputs that the generator writes, and through the server's /api/physics."""
import importlib.util, json, os, urllib.error, urllib.request

import pytest

from opentopas_workbench import physics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "opentopas_workbench", "generators")


def load(name):
    spec = importlib.util.spec_from_file_location("tp_" + name, os.path.join(GEN, name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


RF = load("ripple_filter")
CTRL = {"ions": ["p", "a"], "energies": "63, 250", "variants": ["rifi0"], "histories": 1000, "spread": 0.2,
        "period": 1.5, "prefix": "ph"}


def rows(result):
    return {r["key"]: r for r in result["rows"]}


def test_generator_case_shows_the_topas_defaults_and_the_writers_fluka_thresholds():
    txt, inp = dict(RF.generate(CTRL)), dict(RF.generate_fluka(CTRL))
    r = rows(physics.compare(txt["ph_p063_rifi0.txt"], inp["ph_p063_rifi0.inp"]))
    assert r["list"]["topas"] == {"v": "Default (Geant4_Modular)", "src": "default"}
    assert r["modules"]["topas"]["v"].split(", ") == physics.TOPAS_DEFAULTS["modules"]
    assert r["e_prod"]["topas"] == {"v": "not set", "src": "none"}   # no default cut is documented: never a number
    assert r["emrange"]["topas"] == {"v": "100 eV to 600 MeV", "src": "default"}
    f = {k: v["fluka"] for k, v in r.items()}
    assert f["list"]["v"] == "DEFAULTS HADROTHE" and f["list"]["src"] == "file"
    # the writer's constants: electrons 100 keV kinetic, photons 10 keV, delta rays 100 keV
    assert (f["e_prod"]["v"], f["g_prod"]["v"], f["delta"]["v"]) == ("100 keV kinetic", "10 keV", "100 keV")
    assert (f["e_trans"]["v"], f["g_trans"]["v"]) == ("100 keV kinetic", "10 keV")
    assert f["had_trans"] == {"v": "set by DEFAULTS HADROTHE", "src": "default"}
    lines = inp["ph_p063_rifi0.inp"].splitlines()
    assert lines[f["delta"]["line"] - 1].startswith("DELTARAY")
    assert lines[f["e_prod"]["line"] - 1].startswith("EMFCUT") and "PROD-CUT" in lines[f["e_prod"]["line"] - 1]


def test_an_em_range_set_in_the_file_wins_over_the_default():
    txt = dict(RF.generate(CTRL))["ph_a250_rifi0.txt"]   # helium above 600 MeV total sets EMRangeMax
    assert "Ph/Default/EMRangeMax" in txt
    e = rows(physics.compare(txt, None))["emrange"]["topas"]
    assert e["v"] == "100 eV to 1000 MeV" and e["src"] == "file"
    assert "EMRangeMax" in txt.splitlines()[e["line"] - 1]


def test_cuts_list_and_modules_set_by_hand():
    text = """s:Ph/ListName = "Mine"   # my list
s:Ph/Mine/Type = "Geant4_Modular"
sv:Ph/Mine/Modules = 2 "g4em-standard_opt3" "g4h-phy_QGSP_BIC"
d:Ph/Mine/CutForAllParticles = 0.10 mm
d:Ph/Mine/CutForGamma = 1 mm
d:Ph/Default/CutForElectron = 9 mm # another list: not the one in use
"""
    r = rows(physics.compare(text, None))
    t = {k: v["topas"] for k, v in r.items()}
    assert t["list"]["v"] == "Mine (Geant4_Modular)" and t["list"]["line"] == 1
    assert t["modules"] == {"v": "g4em-standard_opt3, g4h-phy_QGSP_BIC", "src": "file", "line": 3}
    assert t["e_prod"] == {"v": "0.1 mm range", "src": "file", "line": 4}
    assert t["g_prod"] == {"v": "1 mm range", "src": "file", "line": 5}
    assert t["delta"]["v"] == "0.1 mm range (electron cut)"
    assert all(v["fluka"] is None for v in r.values())


def test_another_list_has_no_default_modules():
    r = rows(physics.compare('s:Ph/ListName = "Other"\n', None))
    assert r["modules"]["topas"] == {"v": "not set", "src": "none"}


def test_an_included_file_is_named():
    assert physics.compare("includeFile = base.txt\n", None)["includes"] is True
    assert physics.compare("d:Ge/X/HLX = 1 cm\n", None)["includes"] is False


def card(name, *what, sdum=""):
    w = list(what) + [None] * (6 - len(what))
    return f"{name:<8}  " + "".join(f"{'' if x is None else x:>10}" for x in w) + sdum


def test_fluka_cards_read_by_column_and_geometry_skipped():
    text = "\n".join([
        "TITLE", "a case", "* EMFCUT        -1.0      1.0      comment, not a card",
        card("DEFAULTS", sdum="PRECISIO"),
        card("GEOBEGIN", sdum="COMBNAME"), "EMFCUT  geometry text that only looks like a card", card("GEOEND"),
        card("EMFCUT", "0.0006", "2.E-5", "1.0", "WATER", sdum="PROD-CUT"),
        card("PART-THR", "-0.01", "1.0", "6.0"),
        card("DELTARAY", "0.001", None, None, "WATER"),
        card("EMFCUT", "0.0007", "3.E-5", None, "REG1"),
        card("EMFCUT", "-0.5", "9.E-5", None, "REG2"),
    ])
    f = {k: v["fluka"] for k, v in rows(physics.compare(None, text)).items()}
    assert f["list"] == {"v": "DEFAULTS PRECISIO", "src": "file", "line": 4}
    assert f["e_prod"]["v"] == "600 keV total" and f["g_prod"]["v"] == "20 keV"
    assert f["had_trans"]["v"] == "10 MeV kinetic" and f["delta"]["v"] == "1 MeV"
    assert f["e_trans"] == {"v": "700 keV total", "src": "file", "line": 11}   # the first transport card
    assert f["g_trans"]["v"] == "30 keV"


def test_a_positive_part_thr_is_a_momentum():
    f = rows(physics.compare(None, card("PART-THR", "0.2", "1.0")))["had_trans"]["fluka"]
    assert f["v"] == "200 MeV/c momentum"


def test_free_format_is_not_read_and_said_so():
    text = card("DEFAULTS", sdum="HADROTHE") + "\nFREE\nEMFCUT, -1.E-4, 1.E-5\n"
    out = physics.compare(None, text)
    assert out["free"] is True and rows(out)["e_prod"]["fluka"]["src"] == "none"


def test_without_defaults_card_fluka_uses_its_new_defaults():
    f = rows(physics.compare(None, card("BEAM", "-0.1")))["list"]["fluka"]
    assert f["src"] == "default" and "NEW-DEFA" in f["v"]


# ------------------------------------------------------------------ the server
@pytest.fixture
def ws(tmp_path, monkeypatch):
    from opentopas_workbench import server as wb
    root = tmp_path / "ws"; root.mkdir()
    monkeypatch.setitem(wb.STATE, "root", str(root))
    return wb, root


def test_server_pairs_a_topas_file_with_the_input_beside_it(ws):
    wb, root = ws
    txt, inp = dict(RF.generate(CTRL)), dict(RF.generate_fluka(CTRL))
    (root / "c").mkdir()
    (root / "c" / "ph_p063_rifi0.txt").write_text(txt["ph_p063_rifi0.txt"])
    out = wb.physics_of("c/ph_p063_rifi0.txt")
    assert out["topas_from"] == "c/ph_p063_rifi0.txt" and out["fluka_from"] == "written for this file"
    assert not (root / "c" / "ph_p063_rifi0.inp").exists()    # shown, never written
    assert rows(out)["list"]["fluka"]["v"] == "DEFAULTS HADROTHE"
    edited = "\n".join(l.replace("HADROTHE", "PRECISIO") if l.startswith("DEFAULTS") else l
                       for l in inp["ph_p063_rifi0.inp"].split("\n"))
    assert edited != inp["ph_p063_rifi0.inp"]
    (root / "c" / "ph_p063_rifi0.inp").write_text(edited)
    for f in ("c/ph_p063_rifi0.txt", "c/ph_p063_rifi0.inp"):
        out = wb.physics_of(f)
        assert (out["topas_from"], out["fluka_from"]) == ("c/ph_p063_rifi0.txt", "c/ph_p063_rifi0.inp"), f
        assert rows(out)["list"]["fluka"]["v"] == "DEFAULTS PRECISIO", f


def test_server_names_why_a_file_has_no_fluka_side(ws):
    wb, root = ws
    (root / "odd.txt").write_text('s:Ge/World/Type = "TsSphere"\n')
    out = wb.physics_of("odd.txt")
    assert out["fluka_from"] is None and "1" in out["fluka_why"]
    assert all(r["fluka"] is None for r in out["rows"])
    (root / "lone.inp").write_text(card("DEFAULTS", sdum="HADROTHE") + "\n")
    out = wb.physics_of("lone.inp")
    assert out["topas_from"] is None and rows(out)["list"]["topas"]["src"] == "default"
    with pytest.raises(ValueError): wb.physics_of("../outside.txt")
