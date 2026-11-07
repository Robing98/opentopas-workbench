"""Tests for the start script of the patient recalculation (tools/plan_mc/run_patient.py), on synthetic data only."""
import json, os, sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("pydicom")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from plan_mc import beam_model, plan as pl, run_patient, synthetic  # noqa: E402
from test_plan_mc_hlut import TEMPLATE  # noqa: E402

SHAPE, ORIGIN, SPACING = (4, 5, 6), (-7.5, -6.0, -3.0), (3.0, 3.0, 2.0)
HEADER = """# TOPAS Version: 4.3
# Results for scorer: Dose
# X in {nx} bins of 3 mm
# Y in {ny} bins of 3 mm
# Z in {nz} bins of 2 mm
# DoseToWater ( Gy ) : Sum   Standard_Deviation
"""


def make_inputs(tmp_path, n_fields, range_shifter=False):
    ct = tmp_path / "ct"
    synthetic.write_ct(str(ct), np.zeros(SHAPE), ORIGIN, SPACING)
    fields = []
    for i in range(1, n_fields + 1):
        layers = [pl.Layer(120.0, [0.0, 5.0], [0.0, -5.0], [1.0, 2.0])]
        fields.append(pl.Field(f"F{i}", i, 90.0 * (i - 1), 0.0, (0.0, 0.0, 0.0), layers, "MU", None,
                               pl.RangeShifter("RS", 30.0, 200.0) if range_shifter else None))
    rp = pl.write_rtplan(pl.Plan("synth", "HFS", fields, fractions=30), str(tmp_path / "rp.dcm"),
                         mu_per_field={i: 10.0 for i in range(1, n_fields + 1)})
    hl = tmp_path / "test.hlut"; hl.write_text("-1000 0.001\n0 1.0\n1500 1.85\n")
    tp = tmp_path / "schneider_template.txt"; tp.write_text(TEMPLATE.format(ones=" ".join(["1"] * 3000)))
    bm = tmp_path / "model.json"
    beam_model.generic(particles_per_mu=1e7).save(str(bm))
    return {"ct": str(ct), "rtplan": rp, "hlut": str(hl), "template": str(tp), "beam_model": str(bm)}


def args(ws, inp, *extra, rtdose=None):
    a = ["write", "--workspace", str(ws), "--name", "run1", "--ct", inp["ct"], "--rtplan", inp["rtplan"],
         "--hlut", inp["hlut"], "--template", inp["template"], "--beam-model", inp["beam_model"], "--histories", "1000"]
    return a + (["--rtdose", rtdose] if rtdose else []) + list(extra)


def test_everything_lands_below_the_patient_folder(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    inp = make_inputs(tmp_path, 1)
    run_patient.main(args(ws, inp))
    out = ws / "patient" / "plan_mc" / "run1"
    assert sorted(os.listdir(ws)) == ["patient"]
    assert {"schneider.txt", "patient.txt", "field1.txt", "run.json", "beam_model.json"} <= set(os.listdir(out))
    assert f"includeFile = {out / 'patient.txt'}" in (out / "field1.txt").read_text()


def test_range_shifter_needs_its_physical_thickness(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    inp = make_inputs(tmp_path, 1, range_shifter=True)
    with pytest.raises(SystemExit, match="rs-thickness"):
        run_patient.main(args(ws, inp))
    run_patient.main(args(ws, inp, "--rs-material", "Lexan", "--rs-thickness", "26"))
    t = (ws / "patient" / "plan_mc" / "run1" / "field1.txt").read_text()
    assert 's:Ge/RangeShifter/Material = "Lexan"' in t and "d:Ge/RangeShifter/TransZ   = 213 mm" in t


@pytest.mark.parametrize("name", ["", "../x", "a/b", ".hidden", "D:x"])
def test_run_name_cannot_leave_the_patient_folder(tmp_path, name):
    with pytest.raises(SystemExit):
        run_patient.folder(str(tmp_path), name)


@pytest.mark.parametrize("n_fields", [1, 3])
def test_evaluate_sums_fields_and_compares_per_fraction(tmp_path, n_fields):
    """TOPAS dose per field written so that the sum equals the TPS plan dose / fractions: gamma passes everywhere."""
    ws = tmp_path / "ws"; ws.mkdir()
    inp = make_inputs(tmp_path, n_fields)
    z, y, x = np.indices(SHAPE)
    ref_fraction = 2.0 * np.exp(-((x - 3) ** 2 + (y - 2) ** 2 + (z - 2) ** 2) / 8.0)
    rd = str(tmp_path / "rd.dcm")
    synthetic.write_rtdose(rd, 30 * ref_fraction, ORIGIN, SPACING)        # DoseSummationType PLAN, 30 fractions
    run_patient.main(args(ws, inp, rtdose=rd))
    out = ws / "patient" / "plan_mc" / "run1"
    assert 'CloneRTDoseGridFrom' in (out / "patient.txt").read_text()
    assert 's:Sc/Dose/Component = "Patient/RTDoseGrid"' in (out / "patient.txt").read_text()
    run = json.loads((out / "run.json").read_text())
    nz, ny, nx = SHAPE
    for m in run["fields"]:
        k = m["particles"] / m["histories"]
        part = (ref_fraction / n_fields / k).transpose(2, 1, 0)           # [x, y, z] as TOPAS writes it
        (out / f"field{m['number']}_dose.binheader").write_text(HEADER.format(nx=nx, ny=ny, nz=nz))
        data = np.stack([part.ravel(order="F"), 0.01 * part.ravel(order="F")], axis=1)
        data.astype("<f8").ravel().tofile(str(out / f"field{m['number']}_dose.bin"))
    run_patient.main(["evaluate", "--workspace", str(ws), "--name", "run1"])
    res = json.loads((out / "result.json").read_text())
    assert res["topas_max_gy"] == pytest.approx(ref_fraction.max(), rel=1e-6)
    assert all(v["pass_pct"] == 100.0 for v in res["comparison"]["gamma"].values())
    # Standard_Deviation is per history: the Sum of 1000 histories spreads by 0.01 * sqrt(1000) of its value
    assert res["uncertainty_rms"] == pytest.approx(0.01 * np.sqrt(1000) / np.sqrt(n_fields), rel=1e-3)
    assert (out / "topas_dose.dcm").exists() and "Gamma 2%/2mm" in (out / "BERICHT.md").read_text()


def test_abs_dose_reference_gives_dose_scale(tmp_path):
    from plan_mc.checks import abs_dose
    bm = tmp_path / "model.json"; beam_model.generic(particles_per_mu=1e7).save(str(bm))
    out = tmp_path / "abs"
    abs_dose.write(str(out), str(bm), 150.0, 10.0, 2.5, 20.0, 0.01, 1000)
    c = json.loads((out / "absdose.json").read_text())
    assert c["spots"] == 25 and c["mu_total"] == pytest.approx(0.25)
    nx, ny = abs_dose.LAT_BINS, int(abs_dose.DEPTH_MM / abs_dose.DEPTH_BIN)
    s = np.zeros((nx, ny, nx)); s[nx // 2, 39:41, nx // 2] = 2.0     # bins at 19.5 and 20.5 mm depth
    s /= c["field"]["particles"] / c["field"]["histories"]
    (out / "absdose_dose.binheader").write_text(HEADER.format(nx=nx, ny=ny, nz=nx))
    np.stack([s.ravel(order="F"), 0 * s.ravel(order="F")], axis=1).astype("<f8").ravel().tofile(str(out / "absdose_dose.bin"))
    assert abs_dose.check(str(out), None) == pytest.approx(2.0)
    assert abs_dose.check(str(out), 1.0, update=str(bm)) == pytest.approx(0.5)
    assert json.loads(bm.read_text())["dose_scale"] == pytest.approx(0.5)


@pytest.mark.parametrize("wets,refused", [((30.0,), False), ((30.0, 30.0, 30.0), False), ((30.0, 57.0, 30.0), True)])
def test_one_physical_range_shifter_for_fields_with_different_shifters_is_refused(tmp_path, wets, refused):
    ws = tmp_path / "ws"; ws.mkdir()
    inp = make_inputs(tmp_path, 1)
    fields = [pl.Field(f"F{i}", i, 0.0, 0.0, (0.0, 0.0, 0.0), [pl.Layer(120.0, [0.0], [0.0], [1.0])], "MU", None,
                       pl.RangeShifter("RS", w, 200.0)) for i, w in enumerate(wets, 1)]
    inp["rtplan"] = pl.write_rtplan(pl.Plan("synth", "HFS", fields, fractions=30), str(tmp_path / "rp2.dcm"),
                                    mu_per_field={i: 10.0 for i in range(1, len(wets) + 1)})
    a = args(ws, inp, "--rs-material", "Lexan", "--rs-thickness", "26")
    if refused:
        with pytest.raises(SystemExit, match="different range shifters"): run_patient.main(a)
    else:
        run_patient.main(a)


def test_rt_dose_of_one_beam_is_refused(tmp_path):
    import pydicom
    ws = tmp_path / "ws"; ws.mkdir()
    inp = make_inputs(tmp_path, 1)
    rd = str(tmp_path / "rd.dcm")
    synthetic.write_rtdose(rd, np.ones(SHAPE), ORIGIN, SPACING)
    ds = pydicom.dcmread(rd); ds.DoseSummationType = "BEAM"; ds.save_as(rd)
    run_patient.main(args(ws, inp, rtdose=rd))
    out = ws / "patient" / "plan_mc" / "run1"
    run = json.loads((out / "run.json").read_text())
    for m in run["fields"]:
        (out / f"field{m['number']}_dose.binheader").write_text(HEADER.format(nx=SHAPE[2], ny=SHAPE[1], nz=SHAPE[0]))
        np.ones((int(np.prod(SHAPE)), 2)).astype("<f8").ravel().tofile(str(out / f"field{m['number']}_dose.bin"))
    with pytest.raises(SystemExit, match="BEAM"):
        run_patient.main(["evaluate", "--workspace", str(ws), "--name", "run1"])


@pytest.mark.parametrize("change", ["orientation", "descending"])
def test_rt_dose_layouts_that_do_not_match_the_topas_grid_are_refused(tmp_path, change):
    import pydicom
    rd = str(tmp_path / "rd.dcm")
    synthetic.write_rtdose(rd, np.ones(SHAPE), ORIGIN, SPACING)
    ds = pydicom.dcmread(rd)
    if change == "orientation": ds.ImageOrientationPatient = [-1, 0, 0, 0, -1, 0]
    else: ds.GridFrameOffsetVector = [-2.0 * k for k in range(SHAPE[0])]
    ds.save_as(rd)
    with pytest.raises(ValueError, match="orientation" if change == "orientation" else "descending"):
        synthetic.read_rtdose(rd)


def test_beam_model_takes_an_energy_just_above_its_top_row():
    m = beam_model.generic()
    assert m.at(220.0 + 5e-7)["mean"] == pytest.approx(220.0)
    with pytest.raises(ValueError): m.at(220.1)
