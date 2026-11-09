"""Variable proton RBE models (plan_mc.rbe) and the RT Structure Set masks (plan_mc.rtstruct), task 2351.

Research only: variable RBE models carry large uncertainties; clinically 1.1 applies. Synthetic data only.
"""
import os, sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("pydicom")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from plan_mc import rbe, rtstruct  # noqa: E402


# ------------------------------------------------------------------ the models against published values
def test_mcnamara_rbemax_matches_the_values_of_the_paper():
    # McNamara 2015 sec. 3.2: RBEmax 1.08 for (a/b) 10 Gy and 1.05 for 15 Gy at LETd 2.5 keV/um
    assert float(rbe.mcnamara(2.5, 10.0)[0]) == pytest.approx(1.08, abs=0.005)
    assert float(rbe.mcnamara(2.5, 15.0)[0]) == pytest.approx(1.05, abs=0.005)


def test_wedenberg_slope_matches_table_two():
    # Wedenberg 2013 Table II: k = q / (a/b) per keV/um; RBEmin is 1 (beta does not depend on LET)
    for ab, k in ((2.7, 0.16), (3.1, 0.14), (7.7, 0.057), (15.0, 0.029), (18.0, 0.024)):
        mx0, _ = rbe.wedenberg(0.0, ab)
        mx1, mn = rbe.wedenberg(1.0, ab)
        assert float(mx1 - mx0) == pytest.approx(k, abs=0.002)
        assert float(mn) == 1.0


def test_carabe_on_its_own_cell_line():
    # at the alpha/beta of the V79 cells (2.686 Gy) the scaling drops out: RBEmax = 0.843 + 0.154 LETd
    mx, mn = rbe.carabe(np.array([1.0, 10.0]), 2.686)
    assert mx == pytest.approx([0.997, 2.383]) and mn == pytest.approx([1.096, 1.15])


@pytest.mark.parametrize("model", sorted(rbe.MODELS))
def test_rbe_rises_with_let_and_falls_with_alpha_beta(model):
    let = np.array([1.0, 3.0, 8.0])
    r2 = rbe.rbe(model, np.full(3, 2.0), let, 2.0)
    r10 = rbe.rbe(model, np.full(3, 2.0), let, 10.0)
    assert np.all(np.diff(r2) > 0) and np.all(r2 > r10)


# ------------------------------------------------------------------ linear quadratic RBE
def lq_rbe_by_root(d, ab, mx, mn):
    """RBE from equal effect: (mx a) D + (mn^2 b) D^2 = a Dx + b Dx^2, with b = 1 and a = ab, solved for Dx."""
    a, b = ab, 1.0
    effect = mx * a * d + mn * mn * b * d * d
    dx = max(np.roots([b, a, -effect]))
    return dx / d


@pytest.mark.parametrize("d,ab,mx,mn", [(2.0, 2.0, 1.3, 1.05), (0.5, 10.0, 1.1, 1.0), (8.0, 3.0, 1.6, 1.2)])
def test_rbe_lq_gives_the_dose_of_equal_effect(d, ab, mx, mn):
    assert float(rbe.rbe_lq(d, ab, mx, mn)) == pytest.approx(lq_rbe_by_root(d, ab, mx, mn), rel=1e-9)


def test_rbe_lq_limits():
    assert rbe.rbe_lq(np.array([0.0, 1e-6]), 2.0, 1.4, 1.1) == pytest.approx([1.4, 1.4], abs=1e-5)
    assert float(rbe.rbe_lq(1e6, 2.0, 1.4, 1.1)) == pytest.approx(1.1, abs=1e-4)
    assert rbe.rbe_lq(np.array([0.1, 2.0, 50.0]), 3.0, 1.0, 1.0) == pytest.approx([1.0, 1.0, 1.0])


def test_rbe_keeps_1_1_outside_the_trusted_voxels():
    d, let = np.array([2.0, 0.01]), np.array([3.0, 60.0])
    r = rbe.rbe("mcnamara", d, let, 2.0, valid=np.array([True, False]))
    assert r[1] == 1.1 and r[0] != 1.1
    assert rbe.rbe("1.1", d, let, 2.0) == pytest.approx([1.1, 1.1])


def test_rbe_takes_alpha_beta_per_voxel():
    d, let = np.full(2, 2.0), np.full(2, 4.0)
    r = rbe.rbe("wedenberg", d, let, np.array([2.0, 10.0]))
    assert r[0] == pytest.approx(float(rbe.rbe("wedenberg", 2.0, 4.0, 2.0)))
    assert r[1] == pytest.approx(float(rbe.rbe("wedenberg", 2.0, 4.0, 10.0)))


# ------------------------------------------------------------------ LET of TOPAS
LET_HEADER = """# TOPAS Version: 4.0
# Parameter File: field1.txt
# Results for scorer: Field1LET
# Scored in component: Patient
# X in {nx} bins of 0.2 cm
# Y in {ny} bins of 0.2 cm
# Z in {nz} bins of 0.2 cm
# ProtonLET ( MeV/mm/(g/cm3) ) : {report}
"""


def write_let(path, xyz, report):
    nx, ny, nz = xyz.shape
    with open(path + "header", "w") as fh: fh.write(LET_HEADER.format(nx=nx, ny=ny, nz=nz, report=report))
    xyz.ravel(order="F").astype("<f8").tofile(path)


def test_read_let_scales_mean_by_the_histories(tmp_path):
    xyz = np.arange(24, dtype=float).reshape(2, 3, 4) / 1000.0
    p = str(tmp_path / "field1_let.bin")
    write_let(p, xyz, "Mean")
    let, how = rbe.read_let(p, 2000)
    assert let == pytest.approx(xyz * 2000) and "2000" in how
    write_let(p, xyz, "Sum")
    let, how = rbe.read_let(p, 2000)
    assert let == pytest.approx(xyz) and how == "Sum"


def test_combine_letd_weights_by_dose():
    d1, d2 = np.array([1.0, 0.0, 0.0]), np.array([3.0, 2.0, 0.0])
    l1, l2 = np.array([2.0, 9.0, 9.0]), np.array([6.0, 4.0, 9.0])
    assert rbe.combine_letd([d1, d2], [l1, l2]) == pytest.approx([5.0, 4.0, 0.0])


def test_let_mask_threshold():
    d = np.array([0.0, 0.04, 0.05, 1.0])
    assert rbe.let_mask(d).tolist() == [False, False, True, True]
    assert not rbe.let_mask(np.zeros(3)).any()


# ------------------------------------------------------------------ distal shift along a beam
def falloff(edge, n=60, h=1.0):
    """Dose [z, y, x] flat to depth `edge` mm along +z, then a linear falloff over 10 mm; 3 x 3 rays."""
    z = np.arange(n) * h
    prof = np.clip((edge + 5.0 - z) / 10.0 + 0.5, 0.0, 1.0)
    return np.broadcast_to(prof[:, None, None], (n, 3, 3)).copy()


@pytest.mark.parametrize("shift", [0.0, 1.5, 2.7])
def test_distal_shift_finds_a_known_shift(shift):
    ref, ev = falloff(30.0), falloff(30.0 + shift)
    s = rbe.distal_shift(ref, ev, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0))
    assert s["median_mm"] == pytest.approx(shift, abs=0.05) and s["rays"] >= 1


def test_distal_shift_looks_along_the_beam_direction():
    ref, ev = falloff(30.0), falloff(32.0)
    back = rbe.distal_shift(ref[::-1], ev[::-1], (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, -1.0))
    assert back["median_mm"] == pytest.approx(2.0, abs=0.05)
    assert rbe.distal_shift(np.zeros((4, 3, 3)), ev[:4], (0, 0, 0), (1, 1, 1), (0, 0, 1)) is None


# ------------------------------------------------------------------ RT Structure Set
def test_rtstruct_write_read_round_trip(tmp_path):
    shape, origin, spacing = (6, 8, 10), (-9.0, -7.0, -5.0), (2.0, 2.0, 2.0)
    ptv = np.zeros(shape, bool); ptv[2:4, 3:5, 4:7] = True
    lung = np.zeros(shape, bool); lung[1:5, 0:2, 0:3] = True
    p = rtstruct.write(str(tmp_path / "rs.dcm"), [("PTV", "PTV", ptv), ("Lunge li", "ORGAN", lung)], origin, spacing)
    got = {n: (k, m) for n, k, m in rtstruct.read(p, shape, origin, spacing)}
    assert got["PTV"][0] == "TARGET" and got["Lunge li"][0] == "OAR"
    assert np.array_equal(got["PTV"][1], ptv) and np.array_equal(got["Lunge li"][1], lung)


def test_rtstruct_leaves_out_structures_off_the_grid_and_keeps_holes(tmp_path):
    shape, origin, spacing = (3, 5, 5), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    far = np.zeros((40, 5, 5), bool); far[30, 1:3, 1:3] = True               # z = 30 mm, outside the 3 slices
    p = rtstruct.write(str(tmp_path / "rs.dcm"), [("GTV weit", "", far)], origin, spacing)
    assert rtstruct.read(p, shape, origin, spacing) == []
    sq = lambda a, b: [[a, a], [b, a], [b, b], [a, b]]
    px, py = np.meshgrid(np.arange(5.0), np.arange(5.0))
    ring = rtstruct.inside(px, py, sq(-0.5, 4.5)) ^ rtstruct.inside(px, py, sq(1.5, 2.5))
    assert ring.sum() == 24 and not ring[2, 2]
