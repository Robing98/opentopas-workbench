"""Tests for the beam model fit (tools/plan_mc/fit.py) and the TOPAS output reader (topas_io.py)."""
import os, sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from plan_mc import fit, topas_io  # noqa: E402


def sigma_at(s, sigma0, theta, rho): return np.sqrt(sigma0 ** 2 + 2 * rho * sigma0 * theta * s + theta ** 2 * s ** 2)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.6])
def test_optics_fit_recovers_known_emittance(rho):
    s = 1150.0 - np.array([400.0, 200.0, 0.0, -200.0, -400.0])
    o = fit.fit_optics(s, sigma_at(s, 3.2, 1.8e-3, rho))
    assert o["sigma"] == pytest.approx(3.2, rel=1e-6) and o["div_mrad"] == pytest.approx(1.8, rel=1e-6)
    assert o["corr"] == pytest.approx(rho, abs=1e-6)


def test_optics_fit_subtracts_air():
    s = 1150.0 - np.array([400.0, 0.0, -400.0])
    air = 0.002 * s                                     # any extra spread per plane
    o = fit.fit_optics(s, np.sqrt(sigma_at(s, 3.0, 1e-3, 0.2) ** 2 + air), air)
    assert o["sigma"] == pytest.approx(3.0, rel=1e-6) and o["corr"] == pytest.approx(0.2, abs=1e-6)


def test_optics_fit_needs_three_planes():
    with pytest.raises(ValueError): fit.fit_optics([700.0, 900.0], [3.0, 3.2])


def toy_curve(r, w, n=4000, bin_mm=0.1):
    """Bragg-like curve: rising plateau, peak at r, Gaussian distal edge of width w (shape only, no physics claim)."""
    z = (np.arange(n) + 0.5) * bin_mm
    d = np.where(z < r, 0.3 + 0.7 * np.exp(-(r - z) / 8.0), np.exp(-0.5 * ((z - r) / w) ** 2))
    return z, d


def test_metrics_on_known_edge():
    z, d = toy_curve(100.0, 2.0)
    m = fit.metrics(z, d)
    # Gaussian edge: D80 at r + w*sqrt(2 ln 1.25), D20 at r + w*sqrt(2 ln 5)
    assert m["r80"] == pytest.approx(100.0 + 2.0 * np.sqrt(2 * np.log(1.25)), abs=0.02)
    assert m["dfo"] == pytest.approx(2.0 * (np.sqrt(2 * np.log(5)) - np.sqrt(2 * np.log(1.25))), abs=0.03)


def test_orient_puts_the_surface_first():
    z, d = toy_curve(100.0, 2.0)
    assert fit.metrics(*fit.orient(z, d[::-1]))["r80"] == pytest.approx(fit.metrics(z, d)["r80"], abs=1e-9)


def toy_library():
    """R80 and falloff from a smooth toy relation, so the inversion is tested on its own."""
    rows = []
    for e in fit.LIB_ENERGIES:
        for s in fit.LIB_SPREADS:
            r80 = 0.0022 * e ** 1.77 * 10
            rows.append({"mean": e, "spread_pct": s, "r80": r80, "dfo": np.hypot(0.012 * r80, 0.05 * r80 * s / 1.0 + 1e-3)})
    return {"histories": 0, "rows": rows}


@pytest.mark.parametrize("e,s", [(93.0, 0.15), (147.3, 0.55), (211.0, 1.0)])
def test_library_inversion(e, s):
    L = fit.Library(toy_library())
    r80, dfo = L.predict(e, s)
    r = L.invert(r80, dfo)
    assert r["mean"] == pytest.approx(e, abs=0.05) and r["spread_pct"] == pytest.approx(s, abs=0.02)


def test_library_grid_must_be_complete():
    lib = toy_library(); lib["rows"].pop(3)
    with pytest.raises(ValueError): fit.Library(lib)


HEADER = """# TOPAS Version: 4.3
# Parameter File: x.txt
# Results for scorer: Dose
# Scored in component: Box
# X in 2 bins of 0.5 cm
# Y in 1 bin of 1 cm
# Z in 3 bins of 1 mm
# DoseToWater ( Gy ) : Sum   Standard_Deviation
"""


def test_topas_csv_and_binary(tmp_path):
    vals = {(i, 0, k): (10 * i + k, 0.1 * (10 * i + k)) for i in range(2) for k in range(3)}
    csv = tmp_path / "d.csv"
    csv.write_text(HEADER + "".join(f"{i}, {j}, {k}, {a}, {b}\n" for (i, j, k), (a, b) in sorted(vals.items())))
    h, d = topas_io.read(str(csv))
    assert h["bins"]["X"][:2] == (2, 5.0) and h["bins"]["Z"][:2] == (3, 1.0) and h["reports"] == ["Sum", "Standard_Deviation"]
    assert d["Sum"].shape == (2, 1, 3) and d["Sum"][1, 0, 2] == 12 and d["Standard_Deviation"][1, 0, 2] == pytest.approx(1.2)
    (tmp_path / "d.binheader").write_text(HEADER)
    np.array([v for _, v in sorted(vals.items())], "<f8").ravel().tofile(str(tmp_path / "d.bin"))
    hb, db = topas_io.read(str(tmp_path / "d"))
    assert np.array_equal(db["Sum"], d["Sum"]) and np.allclose(db["Standard_Deviation"], d["Standard_Deviation"])
    z, c = topas_io.depth_curve(str(csv))
    assert list(z) == [0.5, 1.5, 2.5] and list(c) == [10, 12, 14]


def test_library_case_parses_as_topas_lines():
    tag, text = fit.library_case(147.3, 0.55, 1000)
    assert tag == "lib_e147p3_s0p55"
    assert "d:So/Beam/BeamEnergy = 147.3 MeV" in text and "u:So/Beam/BeamEnergySpread = 0.55" in text
    assert all(l.startswith("#") or ":" in l.split("=")[0] for l in text.splitlines() if l.strip())
