"""Tests for the fixes from the first probes of plan_mc on a clinical plan (task 2316), on synthetic plans only.

Each test names the point of the probe it covers: empty DICOM fields (1), NP unit (2), fraction groups (3), range
shifter distance (4), export finder (5), pydicom 2 and 3 (6), body mask (7), no file names (8), RBE (9), diagnostics
(10), spot sizes from the plan (11).
"""
import json, math, os, shlex, sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pydicom = pytest.importorskip("pydicom")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_mc import beam_model, dicom_compat, evaluate, plan as pl, run_patient, synthetic, topas_writer as tw  # noqa: E402
from test_plan_mc_run_patient import HEADER, ORIGIN, SHAPE, SPACING, args, make_inputs  # noqa: E402

NAME = "Mustermann"          # stands for the patient's name in file names, label and PatientName


def fields(n=1, unit="MU", rs=None, size=None):
    return [pl.Field(f"F{i}", i, 0.0, 0.0, (0.0, 0.0, 0.0),
                     [pl.Layer(120.0, [0.0, 5.0], [0.0, -5.0], [1.0, 2.0], spot_size=size)], unit, 400.0, rs)
            for i in range(1, n + 1)]


def write_plan(tmp_path, flds, name="rp.dcm", **kw):
    return pl.write_rtplan(pl.Plan("synth", "HFS", flds, fractions=kw.pop("fractions", 30)), str(tmp_path / name),
                           mu_per_field=kw.pop("mu", {f.number: 10.0 for f in flds}), **kw)


def run_dir(ws): return ws / "patient" / "plan_mc" / "run1"


def fake_topas(out, parts, sd_rel=0.01):
    """Writes the TOPAS dose files of a run so that field i scores parts[i] Gy per fraction ([z, y, x]), with a
    Standard_Deviation of sd_rel times the Sum (scalar or [z, y, x])."""
    run = json.loads((out / "run.json").read_text())
    for m, part in zip(run["fields"], parts):
        k = m["particles"] / m["histories"]
        p = (np.asarray(part, float) / k).transpose(2, 1, 0)
        nx, ny, nz = p.shape
        (out / f"field{m['number']}_dose.binheader").write_text(HEADER.format(nx=nx, ny=ny, nz=nz))
        rel = np.asarray(sd_rel, float)
        sd = (rel if rel.ndim == 0 else rel.transpose(2, 1, 0)) * p      # per history, relative to the Sum
        np.stack([p.ravel(order="F"), np.broadcast_to(sd, p.shape).ravel(order="F")], axis=1).astype("<f8").ravel().tofile(
            str(out / f"field{m['number']}_dose.bin"))


def blob():
    z, y, x = np.indices(SHAPE)
    return 2.0 * np.exp(-((x - 3) ** 2 + (y - 2) ** 2 + (z - 2) ** 2) / 8.0)


# ------------------------------------------------------------------ 1. empty type 2 fields
def test_empty_snout_isocenter_and_spot_size_count_as_missing(tmp_path):
    path = write_plan(tmp_path, fields(size=(7.0, 8.0)))
    ds = pydicom.dcmread(path)
    cps = ds.IonBeamSequence[0].IonControlPointSequence
    cps[0].SnoutPosition = None
    cps[0].ScanningSpotSize = None
    cps[1].IsocenterPosition = None           # a later control point with an empty isocenter keeps the first one
    ds.save_as(path)
    assert pydicom.dcmread(path).IonBeamSequence[0].IonControlPointSequence[0].SnoutPosition in (None, "")
    f = pl.read_rtplan(path).fields[0]
    assert f.snout_position is None and f.iso == (0.0, 0.0, 0.0) and f.layers[0].spot_size is None


def test_spot_size_given_once_holds_for_the_later_layers(tmp_path):
    flds = fields(size=(7.0, 9.0))
    flds[0].layers.append(pl.Layer(110.0, [0.0], [0.0], [1.0], spot_size=(8.0, 10.0)))
    flds[0].layers.append(pl.Layer(100.0, [0.0], [0.0], [1.0], spot_size=(8.0, 10.0)))
    path = write_plan(tmp_path, flds)
    ds = pydicom.dcmread(path)
    cps = ds.IonBeamSequence[0].IonControlPointSequence
    for cp in cps[1:]:                        # DICOM: an attribute left out of a control point keeps its last value
        if "ScanningSpotSize" in cp and list(cp.ScanningSpotSize) == [7.0, 9.0]: del cp.ScanningSpotSize
    third = [cp for cp in cps if float(getattr(cp, "NominalBeamEnergy", 0) or 0) == 100.0]
    for cp in third: del cp.ScanningSpotSize  # the third layer inherits the second one's size
    ds.save_as(path)
    assert [l.spot_size for l in pl.read_rtplan(path).fields[0].layers] == [(7.0, 9.0), (8.0, 10.0), (8.0, 10.0)]


# ------------------------------------------------------------------ 2. NP unit
@pytest.mark.parametrize("unit,particles", [("NP", 10.0), ("MU", 10.0 * 1e7)])
def test_np_meterset_is_taken_as_particles(tmp_path, unit, particles):
    ws = tmp_path / "ws"; ws.mkdir()
    inp = make_inputs(tmp_path, 1)
    inp["rtplan"] = write_plan(tmp_path, fields(unit=unit), name="np.dcm")
    assert pydicom.dcmread(inp["rtplan"]).IonBeamSequence[0].PrimaryDosimeterUnit == unit
    assert pl.read_rtplan(inp["rtplan"]).fields[0].unit == unit
    run_patient.main(args(ws, inp))                     # beam model with 1e7 particles per MU
    m = json.loads((run_dir(ws) / "run.json").read_text())["fields"][0]
    assert m["unit"] == unit and m["particles"] == pytest.approx(particles)


# ------------------------------------------------------------------ 3. fraction groups
@pytest.mark.parametrize("n_fields", [1, 3])
def test_setup_group_without_fractions_does_not_overwrite_the_treatment_group(tmp_path, n_fields):
    path = write_plan(tmp_path, fields(n_fields), setup_beams=2)
    ds = pydicom.dcmread(path)
    assert len(ds.FractionGroupSequence) == 2 and "NumberOfFractionsPlanned" not in ds.FractionGroupSequence[1]
    p = pl.read_rtplan(path)
    assert p.fractions == 30 and [f.number for f in p.fields] == list(range(1, n_fields + 1))


def test_treatment_beams_in_groups_with_different_fractions_are_refused(tmp_path):
    path = write_plan(tmp_path, fields(2))
    ds = pydicom.dcmread(path)
    g1 = ds.FractionGroupSequence[0]
    g2 = pydicom.Dataset(); g2.FractionGroupNumber = 2; g2.NumberOfFractionsPlanned = 5
    g2.ReferencedBeamSequence = pydicom.Sequence([g1.ReferencedBeamSequence.pop()])
    ds.FractionGroupSequence.append(g2); ds.save_as(path)
    with pytest.raises(ValueError, match="different fraction counts"):
        pl.read_rtplan(path)
    ds.FractionGroupSequence[1].NumberOfFractionsPlanned = 30; ds.save_as(path)
    assert pl.read_rtplan(path).fractions == 30


def test_treatment_group_with_an_empty_fraction_count_is_refused(tmp_path):
    path = write_plan(tmp_path, fields(1), setup_beams=2)
    ds = pydicom.dcmread(path)
    ds.FractionGroupSequence[0].NumberOfFractionsPlanned = None   # type 2: may be empty, then the count is unknown
    ds.save_as(path)
    with pytest.raises(ValueError, match="NumberOfFractionsPlanned"):
        pl.read_rtplan(path)


# ------------------------------------------------------------------ 4. range shifter distance
def plan_without_rs_distance(tmp_path, value=None):
    path = write_plan(tmp_path, fields(rs=pl.RangeShifter("RS", 30.0, 200.0)), name="rs.dcm")
    ds = pydicom.dcmread(path)
    s = ds.IonBeamSequence[0].IonControlPointSequence[0].RangeShifterSettingsSequence[0]
    if value is None: del s.IsocenterToRangeShifterDistance
    else: s.IsocenterToRangeShifterDistance = value
    ds.save_as(path)
    return path


@pytest.mark.parametrize("plan_value", [None, 0.0])
def test_range_shifter_distance_falls_back_to_the_option_and_refuses_zero(tmp_path, capsys, plan_value):
    ws = tmp_path / "ws"; ws.mkdir()
    inp = make_inputs(tmp_path, 1)
    inp["rtplan"] = plan_without_rs_distance(tmp_path, plan_value)
    assert pl.read_rtplan(inp["rtplan"]).fields[0].range_shifter.iso_distance == plan_value
    rs = ["--rs-material", "Lexan", "--rs-thickness", "26"]
    with pytest.raises(SystemExit, match="rs-iso-distance"):
        run_patient.main(args(ws, inp, *rs))
    with pytest.raises(SystemExit, match="above 0"):
        run_patient.main(args(ws, inp, *rs, "--rs-iso-distance", "0"))
    capsys.readouterr()
    run_patient.main(args(ws, inp, *rs, "--rs-iso-distance", "173"))
    out = capsys.readouterr().out
    assert "30 mm WET" in out and "173 mm from the isocenter (--rs-iso-distance)" in out
    assert "d:Ge/RangeShifter/TransZ   = 186 mm" in (run_dir(ws) / "field1.txt").read_text()


def test_range_shifter_distance_of_the_plan_wins_and_is_named(tmp_path, capsys):
    ws = tmp_path / "ws"; ws.mkdir()
    inp = make_inputs(tmp_path, 1, range_shifter=True)
    run_patient.main(args(ws, inp, "--rs-material", "Lexan", "--rs-thickness", "26", "--rs-iso-distance", "999"))
    assert "200 mm from the isocenter (plan)" in capsys.readouterr().out


# ------------------------------------------------------------------ 5. find
def export(tmp_path, dose_refs="other", n_plans=1, hlut=True):
    """Synthetic export: CT, plan(s), RT Dose; file names carry the patient's name as in a real export."""
    ex = tmp_path / "export"; (ex / f"CT_{NAME}").mkdir(parents=True)
    study, frame = synthetic.write_ct(str(ex / f"CT_{NAME}"), np.zeros(SHAPE), ORIGIN, SPACING)
    other = tmp_path / "other_ct"
    synthetic.write_ct(str(other), np.zeros(SHAPE), ORIGIN, SPACING)        # another series, other frame
    for k, n in enumerate(sorted(os.listdir(other))[:2]): os.rename(other / n, ex / f"scout_{k}.dcm")
    plans = [write_plan(tmp_path, fields(), name=f"export/RP_{NAME}_{i}.dcm", frame_uid=frame) for i in range(n_plans)]
    uid = str(pydicom.dcmread(plans[0]).SOPInstanceUID)
    if dose_refs is not None:
        synthetic.write_rtdose(str(ex / f"RD_{NAME}.dcm"), np.ones(SHAPE), ORIGIN, SPACING, study, frame,
                               plan_uid=uid if dose_refs == "fits" else "1.2.3.4")
        synthetic.write_rtdose(str(ex / f"RD_{NAME}_beam.dcm"), np.ones(SHAPE), ORIGIN, SPACING, study, frame,
                               plan_uid=uid, summation="BEAM")
    hd = tmp_path / "hluts"; hd.mkdir()
    if hlut: (hd / "clinic.hlut").write_text("-1000 0.001\n0 1.0\n")
    return ex, hd, plans[0]


def run_find(capsys, *argv):
    run_patient.main(["find", *argv])
    c = capsys.readouterr()
    env = dict(tok.split("=", 1) for tok in shlex.split(c.out))
    return env, c.err


@pytest.mark.parametrize("refs", ["fits", "other"])
def test_find_matches_by_reference_and_falls_back_with_a_warning(tmp_path, capsys, refs):
    ex, hd, rp = export(tmp_path, refs)
    env, err = run_find(capsys, str(ex), "--hlut-dir", str(hd))
    assert env["RP"] == rp and env["CT"] == str(ex / f"CT_{NAME}") and env["HLUT"] == str(hd / "clinic.hlut")
    assert env["RD"] == str(ex / f"RD_{NAME}.dcm")                       # the PLAN dose, not the BEAM dose
    assert "CT series: 2" in err and "RT Ion Plans: 1" in err and "RT Doses: 2 (BEAM: 1, PLAN: 1)" in err
    assert ("fits 2" in err) if refs == "fits" else ("fits 1, missing 0, other plan 1" in err)
    assert ("does not reference this plan" in err) == (refs == "other")
    assert NAME not in err                                               # counts only, no file names


def test_find_without_dose_and_hlut_says_so(tmp_path, capsys, monkeypatch):
    ex, hd, _ = export(tmp_path, None, hlut=False)
    public = tmp_path / "matRad_default.hlut"; public.write_text("0 1.0\n")
    monkeypatch.setattr(run_patient, "PUBLIC_HLUT", str(public))
    env, err = run_find(capsys, str(ex), "--hlut-dir", str(hd))
    assert env["RD"] == "" and env["HLUT"] == str(public)
    assert "no RT Dose" in err and "public matRad default HLUT" in err


def test_find_refuses_several_plans_without_a_dose_to_choose(tmp_path, capsys):
    ex, hd, _ = export(tmp_path, None, n_plans=3)
    with pytest.raises(SystemExit, match="3 RT Ion Plans"):
        run_patient.main(["find", str(ex)])


# ------------------------------------------------------------------ 6. pydicom 2 and 3
class Pydicom2Dataset:
    """Stands for a pydicom 2 dataset: save_as knows no enforce_file_format."""
    def __init__(self, ts):
        self.file_meta = type("M", (), {"TransferSyntaxUID": ts})()
        self.calls = []

    def save_as(self, path, write_like_original=True):
        self.calls.append((path, write_like_original, self.is_little_endian, self.is_implicit_VR))


def test_save_falls_back_to_the_pydicom_2_call():
    ds = Pydicom2Dataset("1.2.840.10008.1.2.1")
    dicom_compat.save(ds, "x.dcm")
    assert ds.calls == [("x.dcm", False, True, False)]


@pytest.mark.parametrize("version,ok", [("3.0.1", True), ("2.4.3", True), ("2.3.0", False)])
def test_pydicom_version_check_names_the_install_line(monkeypatch, version, ok):
    monkeypatch.setattr(pydicom, "__version__", version)
    if ok: assert dicom_compat.check() == version
    else:
        with pytest.raises(SystemExit, match="pip install --user --break-system-packages"): dicom_compat.check()


# ------------------------------------------------------------------ 7, 8, 9, 10: evaluate
def evaluated(tmp_path, capsys, parts, ref_plan, *extra, dose_type="PHYSICAL", plan=None, sd_rel=0.01):
    ws = tmp_path / "ws"; ws.mkdir()
    inp = make_inputs(tmp_path, len(parts))
    if plan: inp["rtplan"] = plan
    rd = str(tmp_path / f"RD_{NAME}.dcm")
    synthetic.write_rtdose(rd, ref_plan, ORIGIN, SPACING, dose_type=dose_type)
    run_patient.main(args(ws, inp, rtdose=rd))
    fake_topas(run_dir(ws), parts, sd_rel)
    capsys.readouterr()
    run_patient.main(["evaluate", "--workspace", str(ws), "--name", "run1", *extra])
    out = run_dir(ws)
    return json.loads((out / "result.json").read_text()), (out / "BERICHT.md").read_text(), capsys.readouterr().out


def test_maximum_and_uncertainty_count_only_inside_the_body(tmp_path, capsys):
    air = (0, 0, 0)
    ref = blob(); ref[air] = 0.0                                         # no TPS dose in this air voxel
    topas = ref.copy(); topas[air] = 10.0                                # noisy, and above the true maximum
    rel = np.full(SHAPE, 0.01); rel[air] = 10.0
    res, rep, _ = evaluated(tmp_path, capsys, [topas], 30 * ref, sd_rel=rel)
    assert res["topas_max_gy"] == pytest.approx(ref.max(), rel=1e-6)
    assert res["uncertainty_rms"] == pytest.approx(0.01 * np.sqrt(1000), rel=1e-3)
    assert res["body_mask"].startswith("TPS-Dosis") and "Koerpermaske: TPS-Dosis ueber 1 %" in rep


def test_body_mask_without_tps_dose_comes_from_the_ct(tmp_path):
    hu = np.full(SHAPE, -1000.0); hu[:, 1:4, 1:5] = 0.0
    synthetic.write_ct(str(tmp_path / "ct"), hu, ORIGIN, SPACING)
    mask, name = run_patient.body_mask({"ct": str(tmp_path / "ct")}, np.ones(SHAPE), None)
    assert name == "CT ueber -900 HU" and mask.sum() == SHAPE[0] * 3 * 4


def test_reports_name_no_patient_file_and_no_label_with_the_name(tmp_path, capsys):
    rp = write_plan(tmp_path, fields(), name=f"RP_{NAME}.dcm")
    ds = pydicom.dcmread(rp); ds.PatientName = f"{NAME}^Max"; ds.RTPlanLabel = f"{NAME} 1"; ds.save_as(rp)
    _, rep, out = evaluated(tmp_path, capsys, [blob()], 30 * blob(), plan=rp)
    assert "Plan: Plan, 1 Felder, 30 Fraktionen" in rep
    for text in (rep, out, (tmp_path / "ws" / "patient" / "plan_mc" / "run1" / "field1.txt").read_text()):
        assert NAME not in text


@pytest.mark.parametrize("label,name,shown", [("PT1 Prostata", "Synthetic^Phantom", "PT1 Prostata"),
                                              ("Max", "Mustermann^Max", "Plan"), ("A^B", "X^Y", "Plan"), ("", "X^Y", "Plan"),
                                              ("1234567 Prost", "Synthetic^Phantom", "Plan"),
                                              ("M.M. Prost", "Mustermann^Max", "Plan"), ("MM", "Mustermann^Max", "Plan"),
                                              ("Prost MM2", "Mustermann^Max", "Prost MM2")])
def test_plan_label_is_kept_only_when_it_is_not_a_name(label, name, shown):
    ds = pydicom.Dataset(); ds.RTPlanLabel = label; ds.PatientName = name; ds.PatientID = "1234567"
    assert pl.safe_label(ds) == shown


@pytest.mark.parametrize("dose_type,rbe_arg,rbe,ratio", [("EFFECTIVE", [], 1.1, 1.0), ("PHYSICAL", [], 1.0, 1.0),
                                                          ("EFFECTIVE", ["--rbe", "1.0"], 1.0, 1 / 1.1)])
def test_effective_tps_dose_is_compared_with_rbe_times_topas(tmp_path, capsys, dose_type, rbe_arg, rbe, ratio):
    physical = blob()
    tps = 1.1 * physical if dose_type == "EFFECTIVE" else physical
    res, rep, _ = evaluated(tmp_path, capsys, [physical], 30 * tps, *rbe_arg, dose_type=dose_type)
    assert res["rbe"] == rbe and res["diagnostics"]["high_dose_mean_ratio"] == pytest.approx(ratio, rel=1e-6)
    assert f"verglichen mit RBE {rbe:g} x TOPAS" in rep
    assert f"TOPAS / TPS wie gespeichert ({dose_type}): {1 / (1.1 if dose_type == 'EFFECTIVE' else 1.0):.3f}" in rep
    if ratio == 1.0: assert all(v["pass_pct"] == 100.0 for v in res["comparison"]["gamma"].values())


def test_diagnostics_find_a_rigid_shift_and_the_scale():
    ref = np.zeros((12, 12, 12)); ref[4:8, 4:8, 4:8] = 1.0; ref[5:7, 5:7, 5:7] = 2.0
    ev = 0.9 * np.roll(ref, 1, axis=2)                                   # TOPAS 1 voxel to +x, 10 % low
    d = evaluate.diagnostics(ref, ev, (0.0, 0.0, 0.0), (2.0, 3.0, 4.0))
    r = ref[ref >= 1.0]                                                   # rms of 0.1 * ref over the mean of ref
    assert d["rigid"]["shift_mm"] == [2.0, 0.0, 0.0]
    assert d["rigid"]["rms_rel_best"] == pytest.approx(0.1 * np.sqrt(np.mean(r ** 2)) / r.mean(), rel=1e-6)
    assert d["rigid"]["rms_rel_unshifted"] > 0.3
    assert d["centre_of_mass_shift_mm"] == pytest.approx([2.0, 0.0, 0.0], abs=1e-9)
    same = evaluate.diagnostics(ref, 0.9 * ref, (0.0, 0.0, 0.0), (2.0, 3.0, 4.0))
    assert same["high_dose_mean_ratio"] == pytest.approx(0.9) and same["ratio_q05_q50_q95"] == pytest.approx([0.9] * 3)
    assert same["rigid"]["shift_mm"] == [0.0, 0.0, 0.0]


def test_normalised_gamma_removes_a_global_scale(tmp_path, capsys):
    res, rep, _ = evaluated(tmp_path, capsys, [0.9 * blob()], 30 * blob(), "--normalised-gamma")
    assert res["comparison"]["gamma"]["2%/2mm"]["pass_pct"] < 100.0
    assert all(v["pass_pct"] == 100.0 for v in res["gamma_normalised_to_tps_mean"].values())
    assert "Beste starre Verschiebung" in rep and "Quantile 5/50/95 %" in rep and "normiert" in rep


# ------------------------------------------------------------------ 11. spot sizes from the plan
@pytest.mark.parametrize("corr", [0.0, -0.4])
def test_plan_spot_size_is_reached_at_the_isocenter(corr):
    m = beam_model.generic(energies=[100.0, 140.0])
    for r in m.rows: r["corr_x"] = r["corr_y"] = corr
    m = beam_model.BeamModel(m.d)
    p = pl.Plan("p", "HFS", [pl.Field("F1", 1, 0, 0, (0, 0, 0), [pl.Layer(120.0, [0.0], [0.0], [1.0], spot_size=(7.0, 9.0))])])
    q = beam_model.with_plan_spots(m, p).at(120.0)
    z = m.source_distance
    for k, fwhm in (("x", 7.0), ("y", 9.0)):
        s, d = q[f"sigma_{k}"], q[f"div_{k}"] / 1000
        assert math.sqrt(s * s + 2 * corr * s * d * z + d * d * z * z) == pytest.approx(fwhm / 2.3548, rel=1e-4)
    assert q["mean"] == pytest.approx(120.0)


def test_spots_from_plan_go_into_the_run_folder_only(tmp_path, capsys):
    ws = tmp_path / "ws"; ws.mkdir()
    inp = make_inputs(tmp_path, 1)
    base = json.loads(open(inp["beam_model"]).read())
    inp["rtplan"] = write_plan(tmp_path, fields(size=(7.0, 9.0)), name="spots.dcm")
    capsys.readouterr()
    run_patient.main(args(ws, inp, "--spots-from-plan"))
    out = capsys.readouterr().out
    model = beam_model.BeamModel.load(str(run_dir(ws) / "beam_model.json"))
    sx = model.at(120.0)["sigma_x"]
    assert sx != pytest.approx(2.97) and f"{sx:.6g}" in (run_dir(ws) / "field1.txt").read_text()
    assert json.loads(open(inp["beam_model"]).read()) == base                  # the given model stays as it was
    assert "spot sizes from the plan for 1 energies" in out and f"{sx:.3g}" not in out and "sigma" not in out
    inp["rtplan"] = write_plan(tmp_path, fields(), name="nospots.dcm")
    with pytest.raises(SystemExit, match="without ScanningSpotSize"):
        run_patient.main(args(ws, inp, "--spots-from-plan"))
