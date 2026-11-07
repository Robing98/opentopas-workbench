"""Tests for the uncertainty of a TOPAS Sum and for the public chain check (tools/plan_mc/checks/public_chain.py)."""
import json, os, sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_mc import topas_io  # noqa: E402
from plan_mc.checks import public_chain  # noqa: E402
from test_plan_mc_matrad import _mat  # noqa: E402


def test_sum_sd_is_the_spread_of_the_sum_over_repeated_runs():
    """Repeat a run of N histories (most deposit nothing in the voxel, as in a plan): the Sum spreads by sum_sd."""
    rng = np.random.default_rng(1)
    n, runs = 2000, 400
    per_history = rng.exponential(1.0, (runs, n)) * (rng.random((runs, n)) < 0.05)
    sums = per_history.sum(axis=1)
    sd_report = per_history[0].std()          # what TOPAS reports for one run: the spread of one history
    assert topas_io.sum_sd(sd_report, n) == pytest.approx(sums.std(), rel=0.15)
    assert sd_report < 0.1 * sums.std()        # read as the spread of the Sum it would be far too small


HEADER = """# TOPAS Version: 4.3
# Results for scorer: Dose
# X in {nx} bins of 2 mm
# Y in {ny} bins of 3 mm
# Z in {nz} bins of 4 mm
# DoseToWater ( Gy ) : Sum   Standard_Deviation
"""


@pytest.mark.parametrize("n_fields", [1, 3])
def test_check_normalises_where_matrad_has_dose_when_the_target_has_none(tmp_path, n_fields):
    """E2 case: the matRad dose is zero in the target; the factor and the uncertainty still come out right."""
    hu = np.zeros((4, 5, 6), np.int16)                    # [y, x, z]; the cst target is voxel z=3, y=1, x=2
    mat = _mat(str(tmp_path / "plan.mat"), hu, [1.0], [((0, 0, 0), [80.0])])
    shape = (6, 4, 5)                                     # [z, y, x]
    z, y, x = np.indices(shape)
    ref = 2.0 * np.exp(-((x - 0) ** 2 + (y - 2) ** 2 + (z - 1) ** 2) / 4.0)
    ref[3, 1, 2] = 0.0
    out = tmp_path / "chain"; out.mkdir()
    np.save(str(out / "ref_dose.npy"), ref)
    histories, scale = 1000, 1.1                          # TOPAS 10 % above matRad
    fields = []
    for i in range(1, n_fields + 1):
        m = {"number": i, "field": f"f{i}", "histories": histories, "particles": 5e6}
        part = (scale * ref / n_fields / (m["particles"] / histories)).transpose(2, 1, 0)   # [x, y, z]
        (out / f"field{i}_dose.binheader").write_text(HEADER.format(nx=5, ny=4, nz=6))
        np.stack([part.ravel(order="F"), 0.001 * part.ravel(order="F")], axis=1).astype("<f8").ravel() \
            .tofile(str(out / f"field{i}_dose.bin"))
        fields.append(m)
    (out / "chain.json").write_text(json.dumps({"mat": mat, "origin": [-10.0, 5.0, -20.0], "spacing": [2.0, 3.0, 4.0],
                                                "shape_zyx": list(shape), "fields": fields}))
    rep = public_chain.check(str(out))
    assert rep["absolute"]["target_mean_ratio_eval_over_ref"] is None
    assert rep["normalised_to_high_dose_mean"]["factor"] == pytest.approx(scale)
    assert all(v["pass_pct"] == 100.0 for v in rep["normalised_to_high_dose_mean"]["gamma"].values())
    assert rep["rel_uncertainty_rms_above_50pct"] == pytest.approx(0.001 * np.sqrt(histories) / np.sqrt(n_fields))


def _jobs(ws, entries):
    (ws / ".workbench").mkdir(parents=True, exist_ok=True)
    (ws / ".workbench" / "jobs.json").write_text(json.dumps(entries))


def test_seconds_reads_the_workbench_record_and_skips_patient_runs(tmp_path):
    from plan_mc.checks import runtime_split
    ws = tmp_path / "ws"
    _jobs(ws, [{"file": "a/f.txt", "status": "done", "seconds": 10},
               {"file": "a/f.txt", "status": "failed", "seconds": 3},
               {"file": "a/f.txt", "status": "done", "seconds": 12},
               {"file": "patient/f.txt", "status": "done", "seconds": 99}])
    assert runtime_split.seconds(str(ws), str(ws / "a" / "f.txt")) == 12.0
    assert runtime_split.seconds(str(ws), str(ws / "patient" / "f.txt")) is None
    assert runtime_split.seconds(str(ws), str(ws / "a" / "g.txt")) is None


@pytest.mark.parametrize("runs", [3, 5])
def test_split_recovers_start_spot_and_history_cost(runs):
    from plan_mc.checks import runtime_split
    start, per_spot, per_hist = 20.0, 1.2, 2e-4
    rs = [{"spots": s, "histories": h, "seconds": start + per_spot * s + per_hist * h}
          for s, h in [(100, 2e4), (100, 4e5), (300, 2e4), (500, 1e6), (50, 1e5)][:runs]]
    assert runtime_split.split(rs) == pytest.approx((start, per_spot, per_hist), rel=1e-6)
