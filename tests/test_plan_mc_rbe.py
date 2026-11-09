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


def test_mcnamara_rbemin_uses_the_root_of_alpha_beta():
    # McNamara 2015 eq. 8: RBEmin = 1.1012 - 0.0038703 sqrt(a/b) LETd; a/b 4 Gy gives the root 2
    _, mn = rbe.mcnamara(np.array([0.0, 2.5, 10.0]), 4.0)
    assert mn == pytest.approx([1.1012, 1.1012 - 0.0038703 * 2 * 2.5, 1.1012 - 0.0038703 * 2 * 10.0], abs=1e-9)
    _, mn9 = rbe.mcnamara(10.0, 9.0)
    assert float(mn9) == pytest.approx(1.1012 - 0.0038703 * 3 * 10.0, abs=1e-9)


def test_carabe_scales_with_its_alpha_beta_ratio():
    # Carabe 2012: RBEmax = 0.843 + 0.154 (2.686 / (a/b)) LETd, RBEmin = 1.09 + 0.006 (2.686 / (a/b)) LETd
    mx, mn = rbe.carabe(5.0, 2.0)
    assert float(mx) == pytest.approx(0.843 + 0.154 * 1.343 * 5.0)
    assert float(mn) == pytest.approx(1.09 + 0.006 * 1.343 * 5.0)


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
    for mode in ("absolute", "shape"):
        assert s[mode]["median_mm"] == pytest.approx(shift, abs=0.05) and s[mode]["rays"] == 9


def test_distal_shift_absolute_counts_a_higher_dose_and_shape_does_not():
    ref = falloff(30.0)
    s = rbe.distal_shift(ref, 1.2 * ref, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0))
    # falloff of 0.1 per mm: 1.2 x dose reaches 0.8 where ref is 0.8 / 1.2, (0.8 - 0.667) / 0.1 = 1.33 mm deeper
    assert s["absolute"]["median_mm"] == pytest.approx(4.0 / 3.0, abs=0.05)
    assert s["shape"]["median_mm"] == pytest.approx(0.0, abs=0.05)


def test_distal_shift_looks_along_the_beam_direction():
    ref, ev = falloff(30.0), falloff(32.0)
    back = rbe.distal_shift(ref[::-1], ev[::-1], (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, -1.0))
    assert back["shape"]["median_mm"] == pytest.approx(2.0, abs=0.05)
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


def contour_planes(path, zs, x0, x1, y0, y1):
    """A structure set with one PTV rectangle on each contour plane z in `zs`."""
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    import pydicom
    box = np.zeros((1, 2, 2), bool); box[:] = True
    rtstruct.write(path, [("PTV", "PTV", box)], (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ds = pydicom.dcmread(path)
    cs = []
    for z in zs:
        c = Dataset(); c.ContourGeometricType = "CLOSED_PLANAR"; c.NumberOfContourPoints = 4
        c.ContourData = [float(v) for v in (x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z)]
        cs.append(c)
    ds.ROIContourSequence[0].ContourSequence = Sequence(cs)
    ds.save_as(path)
    return path


def test_rtstruct_fills_every_slice_when_contour_planes_and_dose_slices_differ(tmp_path):
    # contours every 2 mm (z 0..6) on a 3 mm dose grid: planes 2 and 4 both lie nearest to slice 1 and must not cancel
    p = contour_planes(str(tmp_path / "coarse.dcm"), (0.0, 2.0, 4.0, 6.0), -0.5, 2.5, -0.5, 2.5)
    (_, _, m), = rtstruct.read(p, (3, 4, 4), (0.0, 0.0, 0.0), (1.0, 1.0, 3.0))
    assert m.sum(axis=(1, 2)).tolist() == [9, 9, 9]
    # contours every 3 mm (z 0..6) on a 1 mm dose grid: the slices between planes are filled, none beyond 7.5 mm
    p = contour_planes(str(tmp_path / "fine.dcm"), (0.0, 3.0, 6.0), -0.5, 2.5, -0.5, 2.5)
    (_, _, m), = rtstruct.read(p, (10, 4, 4), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert m.sum(axis=(1, 2)).tolist() == [9] * 8 + [0, 0]


# ------------------------------------------------------------------ the rbe command of run_patient.py
def run_with_let(tmp_path, capsys, n_fields=1, hot_air=False, hot_tissue=False):
    """A zoned synthetic run whose field 1 (gantry 0, beam along +y) falls off distally in y with LET rising there.
    `hot_air`: one corner voxel is air in the CT and holds 3 times the target dose, as a noisy voxel outside the body.
    `hot_tissue`: the same voxel holds that dose but stays water in the CT, so only an external contour leaves it out."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plan_mc import run_patient
    from test_plan_mc_probe_fixes import fake_topas
    from test_plan_mc_run_patient import SHAPE, args, make_inputs
    ws = tmp_path / "ws"
    for z in ("offen", "robin", "lokal"): (ws / z).mkdir(parents=True)
    inp = make_inputs(tmp_path, n_fields)
    run_patient.main(args(ws, inp))
    out = ws / "lokal" / "patient" / "plan_mc" / "run1"
    prof = np.array([1.0, 1.0, 1.0, 0.6, 0.02])                                # dose along y, Gy per fraction
    lety = np.array([1.5, 2.0, 3.0, 8.0, 15.0])                                # keV/um, rising distally
    dose = np.broadcast_to(2.0 * prof[None, :, None], SHAPE).copy()
    let = np.broadcast_to(lety[None, :, None], SHAPE).copy()
    if hot_air:
        from plan_mc import synthetic
        from test_plan_mc_run_patient import ORIGIN, SPACING
        hu = np.zeros(SHAPE); hu[3, 4, 5] = -1000
        import shutil
        shutil.rmtree(inp["ct"]); synthetic.write_ct(inp["ct"], hu, ORIGIN, SPACING)
        dose[3, 4, 5] = 6.0
    if hot_tissue: dose[3, 4, 5] = 6.0
    fake_topas(out, [dose] * n_fields)
    run = __import__("json").loads((out / "run.json").read_text())
    for m in run["fields"]:
        write_let(str(out / f"field{m['number']}_let.bin"), let.transpose(2, 1, 0) / m["histories"], "Mean")
    capsys.readouterr()
    return run_patient, ws, out, inp, dose, let


def test_rbe_command_writes_maps_and_a_report_with_the_disclaimer(tmp_path, capsys):
    run_patient, ws, out, inp, dose, let = run_with_let(tmp_path, capsys)
    run_patient.main(["rbe", "--workspace", str(ws), "--name", "run1", "--alpha-beta", "2"])
    printed = capsys.readouterr().out
    rep = (out / "RBE_BERICHT.md").read_text()
    assert run_patient.DISCLAIMER in rep and run_patient.DISCLAIMER_DE in rep and run_patient.DISCLAIMER in printed
    from test_plan_mc_probe_fixes import NAME
    assert NAME not in rep and str(tmp_path) not in printed
    assert np.load(out / "letd.npy") == pytest.approx(let, rel=1e-5)
    from plan_mc import synthetic
    import json, pydicom
    for tag in ("1_1", "mcnamara", "wedenberg", "carabe"):
        w = np.load(out / f"rbe_dose_{tag}.npy")
        assert pydicom.dcmread(str(out / f"rbe_dose_{tag}.dcm")).DoseType == "EFFECTIVE"
        assert synthetic.read_rtdose(str(out / f"rbe_dose_{tag}.dcm"))[0] == pytest.approx(w, rel=1e-4, abs=1e-6)
    assert np.load(out / "rbe_dose_1_1.npy") == pytest.approx(1.1 * dose, rel=1e-5)
    mc = np.load(out / "rbe_mcnamara.npy")
    expect = rbe.rbe("mcnamara", dose[0, 2, 0], let[0, 2, 0], 2.0)
    assert float(mc[0, 2, 0]) == pytest.approx(float(expect), rel=1e-5)
    assert mc[0, 4, 0] == pytest.approx(1.1)                                    # 1 % of the maximum: LET not trusted
    res = json.loads((out / "rbe_result.json").read_text())
    for n in ("mcnamara", "wedenberg", "carabe"):
        s = res["models"][n]["distal_shift"]["1"]
        assert s["absolute"]["median_mm"] > 0 and s["shape"]["median_mm"] > 0    # RBE rises distally: range extends
    assert sorted(os.listdir(ws / "robin")) == [] and os.listdir(ws / "offen") == []


def test_rbe_command_takes_the_maximum_inside_the_body(tmp_path, capsys):
    # a noisy air voxel at 3 times the target dose must not empty the high dose region nor move the LET threshold
    run_patient, ws, out, inp, dose, let = run_with_let(tmp_path, capsys, hot_air=True)
    run_patient.main(["rbe", "--workspace", str(ws), "--name", "run1", "--alpha-beta", "2"])
    import json
    res = json.loads((out / "rbe_result.json").read_text())
    assert res["letd_high_dose_q50_q95"] is not None
    assert res["models"]["mcnamara"]["rbe_high_dose_mean"] > 1.1
    assert res["voxels_variable_rbe"] == int((dose >= 0.05 * 2.0).sum()) - 1          # the air voxel keeps 1.1
    assert float(np.load(out / "rbe_mcnamara.npy")[3, 4, 5]) == pytest.approx(1.1)
    assert res["models"]["mcnamara"]["distal_shift"]["1"]["shape"]["median_mm"] > 0
    assert "CT" in res["body_mask"]


def test_rbe_command_takes_alpha_beta_per_structure(tmp_path, capsys):
    run_patient, ws, out, inp, dose, let = run_with_let(tmp_path, capsys)
    from test_plan_mc_run_patient import ORIGIN, SHAPE, SPACING
    ptv = np.zeros(SHAPE, bool); ptv[1:3, 0:3, 1:4] = True
    rs = rtstruct.write(str(tmp_path / "rs.dcm"), [("PTV", "PTV", ptv)], ORIGIN, SPACING)
    run_patient.main(["rbe", "--workspace", str(ws), "--name", "run1", "--rtstruct", rs])
    rep = (out / "RBE_BERICHT.md").read_text()
    mc = np.load(out / "rbe_mcnamara.npy")
    assert float(mc[1, 1, 1]) == pytest.approx(float(rbe.rbe("mcnamara", dose[1, 1, 1], let[1, 1, 1], 10.0)), rel=1e-5)
    assert float(mc[0, 1, 1]) == pytest.approx(float(rbe.rbe("mcnamara", dose[0, 1, 1], let[0, 1, 1], 2.0)), rel=1e-5)
    import json
    s = json.loads((out / "rbe_result.json").read_text())["models"]
    assert s["1.1"]["structures"]["PTV"]["D50"] == pytest.approx(2.2, rel=1e-4)
    assert s["mcnamara"]["structures"]["PTV"]["kind"] == "TARGET" and "| PTV | TARGET |" in rep


def test_rbe_command_refuses_a_run_without_let(tmp_path, capsys):
    run_patient, ws, out, *_ = run_with_let(tmp_path, capsys)
    (out / "field1_let.bin").unlink()
    with pytest.raises(SystemExit, match="no LET output"):
        run_patient.main(["rbe", "--workspace", str(ws), "--name", "run1"])


# ------------------------------------------------------------------ body from the external contour, organs without targets
def test_external_is_found_by_type_before_name(tmp_path):
    shape, origin, spacing = (3, 4, 4), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    big = np.ones(shape, bool)
    small = np.zeros(shape, bool); small[:, 1:3, 1:3] = True
    p = rtstruct.write(str(tmp_path / "rs.dcm"), [("Body", "ORGAN", big), ("Haut aussen", "EXTERNAL", small)], origin, spacing)
    typed = rtstruct.read(p, shape, origin, spacing, types=True)
    name, how, mask = rtstruct.external(typed)
    assert (name, how) == ("Haut aussen", "Typ EXTERNAL") and np.array_equal(mask, small)
    kinds = {n: k for n, k, _ in rtstruct.read(p, shape, origin, spacing)}
    assert kinds == {"Body": "OAR", "Haut aussen": "EXTERNAL"}


@pytest.mark.parametrize("name", ["BODY", "Körper", "KOERPER_2", "Außenkontur", "External", "outline"])
def test_external_is_found_by_name_without_the_type(tmp_path, name):
    shape, origin, spacing = (3, 4, 4), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    box = np.zeros(shape, bool); box[:, 1:3, 1:3] = True
    p = rtstruct.write(str(tmp_path / "rs.dcm"), [("Rueckenmark", "ORGAN", box), (name, "", box)], origin, spacing)
    assert rtstruct.external(rtstruct.read(p, shape, origin, spacing, types=True))[:2] == (name, "Name")


@pytest.mark.parametrize("name,typ", [("Body", "PTV"), ("Bodyplan", ""), ("Koerper Rest", "ORGAN"), ("Lunge", "ORGAN")])
def test_no_external_from_a_target_or_another_name(tmp_path, name, typ):
    shape, origin, spacing = (3, 4, 4), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    box = np.zeros(shape, bool); box[:, 1:3, 1:3] = True
    p = rtstruct.write(str(tmp_path / "rs.dcm"), [(name, typ, box)], origin, spacing)
    assert rtstruct.external(rtstruct.read(p, shape, origin, spacing, types=True)) is None


def phantom_structures(tmp_path):
    """External without the last row in y and column in x (the hot voxel 3, 4, 5 lies outside, inside the CT body),
    a PTV, and an organ of 12 voxels of which 4 lie in the PTV."""
    from test_plan_mc_run_patient import ORIGIN, SHAPE, SPACING
    ext = np.zeros(SHAPE, bool); ext[:, 0:4, 0:5] = True
    ptv = np.zeros(SHAPE, bool); ptv[1:3, 0:3, 1:4] = True
    organ = np.zeros(SHAPE, bool); organ[1:3, 1:4, 3:5] = True
    rs = rtstruct.write(str(tmp_path / "rs.dcm"), [("Aussenkontur", "EXTERNAL", ext), ("PTV", "PTV", ptv),
                                                   ("Rueckenmark", "ORGAN", organ)], ORIGIN, SPACING)
    return rs, ext, ptv, organ


def test_rbe_command_takes_the_body_from_the_external_contour(tmp_path, capsys):
    import json
    run_patient, ws, out, inp, dose, let = run_with_let(tmp_path, capsys, hot_tissue=True)
    rs, ext, ptv, organ = phantom_structures(tmp_path)
    run_patient.main(["rbe", "--workspace", str(ws), "--name", "run1", "--rtstruct", rs])
    res = json.loads((out / "rbe_result.json").read_text())
    assert res["body_mask"] == "Aussenkontur des RT Structure Set (ROI nach Typ EXTERNAL gewaehlt)"
    assert res["body_roi"] == "Aussenkontur"
    # threshold from the maximum inside the external (2 Gy, not the 6 Gy voxel), model RBE only inside it
    assert res["voxels_variable_rbe"] == int((ext & (dose >= 0.05 * 2.0)).sum())
    mc = np.load(out / "rbe_mcnamara.npy")
    assert mc[0, 0, 5] == pytest.approx(1.1) and mc[0, 0, 4] != pytest.approx(1.1)
    w = np.load(out / "rbe_dose_mcnamara.npy")
    assert res["models"]["mcnamara"]["max_gy"] == pytest.approx(float(w[ext].max()), rel=1e-5)
    assert res["models"]["1.1"]["max_gy"] == pytest.approx(2.2, rel=1e-5)
    assert res["models"]["1.1"]["structures"]["Aussenkontur"]["kind"] == "EXTERNAL"     # the body is no organ
    rep = (out / "RBE_BERICHT.md").read_text()
    assert 'ROI "Aussenkontur"' in rep


def test_rbe_command_gives_organ_values_without_the_target(tmp_path, capsys):
    import json
    from plan_mc import evaluate
    run_patient, ws, out, inp, dose, let = run_with_let(tmp_path, capsys)
    rs, ext, ptv, organ = phantom_structures(tmp_path)
    run_patient.main(["rbe", "--workspace", str(ws), "--name", "run1", "--rtstruct", rs])
    ms = json.loads((out / "rbe_result.json").read_text())["models"]
    o = ms["1.1"]["structures"]["Rueckenmark"]
    assert o["kind"] == "OAR" and o["target_overlap_pct"] == pytest.approx(100 * 4 / 12)
    # 1.1 by hand: outside the PTV 4 voxels at y 3 (1.2 Gy x 1.1) and 4 at y 1..2 (2 Gy x 1.1)
    assert o["outside_targets"] == pytest.approx({"D2": 2.2, "D50": 2.2, "D98": 1.32}, rel=1e-5)
    assert (o["D2"], o["D98"]) == pytest.approx((2.2, 1.32), rel=1e-5)
    for n in ("mcnamara", "wedenberg", "carabe"):
        w = np.load(out / f"rbe_dose_{n.replace('.', '_')}.npy")
        fig = evaluate.dvh_figures(w, organ & ~ptv, 1.0)
        got = ms[n]["structures"]["Rueckenmark"]["outside_targets"]
        assert got == pytest.approx({k: fig[k] for k in ("D2", "D50", "D98")}, rel=1e-5)
        # in the PTV alpha/beta 10 lowers the RBE: the organ's own voxels outside reach a higher D2 than its PTV part
        assert got["D2"] >= float(w[organ & ptv].max()) * (1 - 1e-6)
    assert "PTV" in ms["1.1"]["structures"] and "target_overlap_pct" not in ms["1.1"]["structures"]["PTV"]
    rep = (out / "RBE_BERICHT.md").read_text()
    assert "Risikoorgane ohne die Zielvolumen" in rep and "| Rueckenmark | 33.3 % |" in rep


def test_evaluate_takes_the_body_from_the_external_contour(tmp_path, capsys):
    import json
    run_patient, ws, out, inp, dose, let = run_with_let(tmp_path, capsys, hot_tissue=True)
    rs, ext, ptv, organ = phantom_structures(tmp_path)
    run_patient.main(["evaluate", "--workspace", str(ws), "--name", "run1"])
    assert json.loads((out / "result.json").read_text())["topas_max_gy"] == pytest.approx(6.0, rel=1e-5)
    run_patient.main(["evaluate", "--workspace", str(ws), "--name", "run1", "--rtstruct", rs])
    res = json.loads((out / "result.json").read_text())
    assert res["topas_max_gy"] == pytest.approx(2.0, rel=1e-5) and res["body_roi"] == "Aussenkontur"
    assert 'ROI "Aussenkontur"' in (out / "BERICHT.md").read_text()
    k = json.loads((ws / "robin" / "plan_mc" / "run1" / "kennzahlen.json").read_text())
    assert "Aussenkontur" not in json.dumps(k).replace("Aussenkontur des RT", "")


def test_rbe_command_without_an_external_falls_back_with_a_warning(tmp_path, capsys):
    import json
    run_patient, ws, out, inp, dose, let = run_with_let(tmp_path, capsys)
    from test_plan_mc_run_patient import ORIGIN, SHAPE, SPACING
    ptv = np.zeros(SHAPE, bool); ptv[1:3, 0:3, 1:4] = True
    rs = rtstruct.write(str(tmp_path / "rs.dcm"), [("PTV", "PTV", ptv)], ORIGIN, SPACING)
    run_patient.main(["rbe", "--workspace", str(ws), "--name", "run1", "--rtstruct", rs])
    assert "without an external contour" in capsys.readouterr().err
    res = json.loads((out / "rbe_result.json").read_text())
    assert "CT" in res["body_mask"] and res["body_roi"] is None


@pytest.mark.parametrize("raw", ["K\u006f\u0308rper", "K\u00c3\u00b6rper", "Au\u00c3\u009fenkontur", "Au\u00dfenkontur"])
def test_external_names_with_decomposed_or_misdecoded_umlauts(raw):
    # o + combining diaeresis, and UTF-8 bytes read as Latin-1 from a file without SpecificCharacterSet
    assert rtstruct.is_external_name(raw)


def test_a_second_external_is_no_organ(tmp_path):
    shape, origin, spacing = (3, 4, 4), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    big = np.ones(shape, bool)
    small = np.zeros(shape, bool); small[:, 1:3, 1:3] = True
    p = rtstruct.write(str(tmp_path / "rs.dcm"), [("Body", "EXTERNAL", big), ("Body alt", "EXTERNAL", small)], origin, spacing)
    assert {n: k for n, k, _ in rtstruct.read(p, shape, origin, spacing)} == {"Body": "EXTERNAL", "Body alt": "EXTERNAL"}
    assert rtstruct.external(rtstruct.read(p, shape, origin, spacing, types=True))[0] == "Body"
