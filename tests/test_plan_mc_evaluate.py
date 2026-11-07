"""Tests for the dose comparison (tools/plan_mc/evaluate.py)."""
import os, sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from plan_mc import evaluate  # noqa: E402

SP = (1.0, 1.0, 1.0)


def blob(shift_x=0.0, scale=1.0):
    z, y, x = np.mgrid[0:20, 0:20, 0:30].astype(float)
    return scale * np.exp(-((x - 15 - shift_x) ** 2 / 40 + (y - 10) ** 2 / 20 + (z - 10) ** 2 / 20))


def test_identical_doses_pass_everywhere_with_gamma_zero():
    g, rate = evaluate.gamma(blob(), blob(), SP)
    assert rate == 100.0 and np.nanmax(g) == pytest.approx(0.0, abs=1e-9)


def test_a_1_mm_shift_passes_2_mm_and_fails_0_5_mm_dta():
    # step 0.5 mm puts the true 1 mm offset on the search grid
    assert evaluate.gamma(blob(), blob(1.0), SP, dta=2.0, dd=0.1, step=0.5)[1] > 99.0
    assert evaluate.gamma(blob(), blob(1.0), SP, dta=0.5, dd=0.1, step=0.1)[1] < 80.0


def test_a_5_percent_scale_fails_2_percent_where_the_dose_is_flat():
    # in a flat field no offset helps: gamma is 5 / 2 everywhere
    ref = np.ones((10, 10, 10))
    g, rate = evaluate.gamma(ref, 1.05 * ref, SP, dta=2.0, dd=2.0)
    assert rate == 0.0 and np.nanmin(g) == pytest.approx(2.5)


def test_dvh_figures_of_a_known_distribution():
    d = np.arange(1, 101, dtype=float)           # 100 voxels, 1 to 100 Gy
    f = evaluate.dvh_figures(d, np.ones(100, bool), prescription=50.0)
    assert (f["D98"], f["D50"], f["D2"], f["mean"]) == (3.0, 51.0, 99.0, 50.5)
    assert f["V95_pct"] == pytest.approx(53.0) and f["V107_pct"] == pytest.approx(47.0)
