"""Tests for the HLUT to Schneider tuning (tools/plan_mc/hlut.py)."""
import os, sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from plan_mc import hlut  # noqa: E402

WATER_W = {"H": 0.111898, "O": 0.888102}

# Three density sections and three material sections over HU -1000..1999, as in the TOPAS Schneider format.
TEMPLATE = """# test template
s:Ge/Patient/ImagingToMaterialConverter = "Schneider"
dv:Ge/Patient/DensityCorrection = 3000 {ones} g/cm3
iv:Ge/Patient/SchneiderHounsfieldUnitSections = 4 -1000 -100 100 2000
uv:Ge/Patient/SchneiderDensityOffset = 3 0.00121 1.0 1.05
uv:Ge/Patient/SchneiderDensityFactor = 3 0.0010 0.001 0.0006
uv:Ge/Patient/SchneiderDensityFactorOffset = 3 1000. 0. 0.
sv:Ge/Patient/SchneiderElements = 4 "Hydrogen" "Carbon" "Oxygen" "Calcium"
iv:Ge/Patient/SchneiderHUToMaterialSections = 4 -1000 -500 200 2000
uv:Ge/Patient/SchneiderMaterialsWeight1 = 4 0.0 0.0 1.0 0.0
uv:Ge/Patient/SchneiderMaterialsWeight2 = 4 0.11 0.10 0.79 0.0
uv:Ge/Patient/SchneiderMaterialsWeight3 = 4 0.06 0.30 0.44 0.20
iv:Gr/Color/PatientTissue1 = 3 63 63 63
"""


@pytest.fixture
def files(tmp_path):
    t = tmp_path / "schneider.txt"
    t.write_text(TEMPLATE.format(ones=" ".join(["1"] * 3000)))
    h = tmp_path / "test.hlut"
    h.write_text("# HU SPR\n1500, 1.85\n-1000 0.001\n0 1.0\n# comment\n300 1.15\n")
    return str(h), str(t)


def test_water_against_its_own_bragg_i_has_spr_one_at_every_energy():
    i_w = hlut.bragg_i(WATER_W)
    assert hlut.spr(1.0, WATER_W, i_water=i_w) == pytest.approx(np.ones(len(hlut.ENERGIES_MEV)), abs=1e-4)


def test_water_bragg_i_is_below_g4_water():
    # Geant4 gives a mixture of H and O the Bragg value from element I (H 19.2, O 95 eV), not the 78 eV of G4_WATER.
    assert 68.0 < hlut.bragg_i(WATER_W) < 70.5


def test_lower_i_raises_spr_less_at_high_energy():
    s = hlut.spr(1.0, WATER_W, i_water=78.0)
    assert np.all(s > 1.0) and np.all(np.diff(s) < 0)


def test_read_hlut_sorts_and_skips_comments(files):
    hu, s = hlut.read_hlut(files[0])
    assert list(hu) == [-1000, 0, 300, 1500] and list(s) == [0.001, 1.0, 1.15, 1.85]


def reread_mean_spr(path, i_water, quantity=None):
    """SPR (or another ratio such as hlut.wer) per HU as TOPAS would build it from the written file, computed
    independently of hlut.tune."""
    quantity = quantity or hlut.spr
    params, _ = hlut.read_template(path)
    corr = np.array([float(v) for v in params["DensityCorrection"][1]])
    sec = [int(v) for v in params["SchneiderHounsfieldUnitSections"][1]]
    off, fac, fo = ([float(v) for v in params[k][1]] for k in
                    ("SchneiderDensityOffset", "SchneiderDensityFactor", "SchneiderDensityFactorOffset"))
    msec = [int(v) for v in params["SchneiderHUToMaterialSections"][1]]
    names = params["SchneiderElements"][1]
    out = {}
    for hu in (-1000, -700, -101, -100, 0, 99, 100, 199, 200, 1000, 1999):
        k = max(i for i in range(len(sec) - 1) if sec[i] <= hu)
        dens = (off[k] + fac[k] * (fo[k] + hu)) * corr[hu - sec[0]]
        m = max(i for i in range(len(msec) - 1) if msec[i] <= hu)
        w = {hlut.TOPAS_ELEMENT[n]: float(f) for n, f in zip(names, params[f"SchneiderMaterialsWeight{m + 1}"][1])}
        out[hu] = quantity(dens, w, i_water).mean()
    return out


@pytest.mark.parametrize("i_water,scale", [(78.0, 1.0), (75.0, 1.0), (78.0, 1.02), (78.0, 0.98)])
def test_written_file_gives_the_hlut_spr_at_every_hu(files, tmp_path, i_water, scale):
    out = tmp_path / "out.txt"
    rep = hlut.run(files[0], files[1], str(out), i_water=i_water, spr_scale=scale, match="spr")
    hu, s = hlut.read_hlut(files[0])
    for h, got in reread_mean_spr(str(out), i_water).items():
        assert got == pytest.approx(scale * np.interp(h, hu, s), rel=1e-5), h
    assert [r["hu_from"] for r in rep["sections"]] == [-1000, -500, 200]


def test_range_match_gives_the_hlut_value_as_mean_water_equivalent_ratio(files, tmp_path):
    out = tmp_path / "out.txt"
    hlut.run(files[0], files[1], str(out))
    hu, s = hlut.read_hlut(files[0])
    for h, got in reread_mean_spr(str(out), 78.0, hlut.wer).items():
        assert got == pytest.approx(np.interp(h, hu, s), rel=1e-5), h


def numeric_csda(i_ev, w, e0):
    """Range by a plain Riemann sum of 1/S from 0.5 MeV, written apart from hlut.csda."""
    e = np.linspace(0.5, e0, 200001)
    gamma = 1 + e / hlut.MP_C2_MEV
    s = hlut.electrons_per_mass(hlut.normalise(w)) * hlut.stopping_number(i_ev, e) / (1 - 1 / gamma ** 2)
    return float(np.sum(1 / s[:-1]) * (e[1] - e[0]))


def test_wer_of_soft_tissue_exceeds_its_mean_spr_and_matches_the_range_integral():
    # Lower I than water: SPR rises towards low energy, so the range ratio lies above the 70 to 220 MeV SPR mean.
    tissue = {"H": 0.10, "C": 0.20, "N": 0.04, "O": 0.66}
    w120 = float(hlut.wer(1.0, tissue, 78.0, np.array([120.0]))[0])
    want = numeric_csda(78.0, WATER_W, 120.0) / numeric_csda(hlut.bragg_i(tissue), tissue, 120.0)
    assert w120 == pytest.approx(want, rel=2e-4)
    assert 0.0005 < w120 / hlut.spr(1.0, tissue, 78.0).mean() - 1 < 0.004


def test_match_must_be_known(files, tmp_path):
    with pytest.raises(ValueError):
        hlut.run(files[0], files[1], str(tmp_path / "o.txt"), match="dose")


def test_residual_is_small_for_soft_tissue_and_larger_for_bone(files, tmp_path):
    rep = hlut.run(files[0], files[1], str(tmp_path / "o.txt"))
    soft, bone = rep["sections"][1], rep["sections"][2]
    assert 0.0 < soft["range_error_mm_per_100mm"] < 0.5
    assert bone["i_ev"] > soft["i_ev"]


def test_component_name_and_length(files, tmp_path):
    out = tmp_path / "o.txt"
    hlut.run(files[0], files[1], str(out), component="CT")
    text = out.read_text()
    assert "Ge/Patient/" not in text and "s:Ge/CT/ImagingToMaterialConverter" in text
    params, _ = hlut.read_template(str(out))
    assert len(params["DensityCorrection"][1]) == 3000 and params["DensityCorrection"][2] == "g/cm3"
    assert params["SchneiderElements"][1] == ["Hydrogen", "Carbon", "Oxygen", "Calcium"]
