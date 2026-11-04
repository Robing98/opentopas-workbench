"""Tests for the spot-by-spot plan chain (tools/plan_mc): plan import, beam model, TOPAS placement, field files."""
import math, os, re, sys

import pytest

np = pytest.importorskip("numpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from plan_mc import beam_model, iec, plan as pl, topas_writer as tw  # noqa: E402


def rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def topas(rot_x=0.0, rot_y=0.0, rot_z=0.0):
    """Local-to-parent matrix of a TOPAS component: G4PVPlacement turns the daughter by the inverse of Rz*Ry*Rx."""
    r = np.radians
    return rx(-r(rot_x)) @ iec.ry(-r(rot_y)) @ iec.rz(-r(rot_z))


def small_plan(position="HFS", gantry=30.0, couch=310.0):
    layers = [pl.Layer(100.0, [-20.0, 0.0, 35.0], [10.0, 0.0, -40.0], [1.0, 2.0, 3.0], "3.0"),
              pl.Layer(120.0, [80.0, -60.0], [70.0, 5.0], [0.5, 0.25], "3.0")]
    f = pl.Field("F1", 1, gantry, couch, (12.5, -30.0, 41.0), layers, "MU", 421.0,
                 pl.RangeShifter("RS3", 30.5, 250.0))
    return pl.Plan("synth", position, [f], fractions=30)


# ------------------------------------------------------------------ plan import
def test_rtplan_round_trip(tmp_path):
    pytest.importorskip("pydicom")
    p = small_plan()
    path = pl.write_rtplan(p, str(tmp_path / "plan.dcm"), mu_per_field={1: 13.5})
    q = pl.read_rtplan(path)
    assert q.patient_position == "HFS" and q.fractions == 30 and len(q.fields) == 1
    f = q.fields[0]
    assert (f.gantry, f.couch, f.iso, f.snout_position) == (30.0, 310.0, (12.5, -30.0, 41.0), 421.0)
    assert f.range_shifter.rs_id == "RS3" and f.range_shifter.wet == 30.5 and f.range_shifter.iso_distance == 250.0
    assert [l.energy for l in f.layers] == [100.0, 120.0]
    assert f.layers[1].x == [80.0, -60.0] and f.layers[1].y == [70.0, 5.0]
    # spot MU = BeamMeterset * weight / FinalCumulativeMetersetWeight
    assert f.total_weight == pytest.approx(13.5)
    assert f.layers[0].w[2] == pytest.approx(13.5 * 3.0 / 6.75)


def test_rtplan_skips_zero_weight_spots_and_setup_beams(tmp_path):
    pytest.importorskip("pydicom")
    import pydicom
    p = small_plan()
    p.fields[0].layers[0].w[1] = 0.0
    path = pl.write_rtplan(p, str(tmp_path / "plan.dcm"))
    ds = pydicom.dcmread(path)
    setup = pydicom.Dataset(); setup.update(ds.IonBeamSequence[0]); setup.BeamNumber = 2; setup.TreatmentDeliveryType = "SETUP"
    ds.IonBeamSequence.append(setup); ds.save_as(path)
    q = pl.read_rtplan(path)
    assert len(q.fields) == 1 and q.fields[0].layers[0].x == [-20.0, 35.0]


# ------------------------------------------------------------------ beam model
def test_beam_model_interpolates_and_refuses_extrapolation():
    m = beam_model.generic(energies=[100.0, 120.0], sigma=3.0)
    m.rows[1]["sigma_x"] = 5.0
    assert m.at(110.0)["sigma_x"] == pytest.approx(4.0)
    assert m.at(100.0)["sigma_x"] == 3.0
    with pytest.raises(ValueError): m.at(99.0)
    with pytest.raises(ValueError): m.at(121.0)


def test_spot_at_source_points_at_the_virtual_source():
    m = beam_model.generic()
    xs, ys, ax, ay = m.spot_at_source(50.0, -40.0)
    # extending the ray from the source plane by the source distance reaches the spot at the isocenter
    assert xs + math.tan(ax) * 1150.0 == pytest.approx(50.0)
    assert ys + math.tan(ay) * 1150.0 == pytest.approx(-40.0)
    assert xs == pytest.approx(50.0 * (7549 - 1150) / 7549)


# ------------------------------------------------------------------ placement
@pytest.mark.parametrize("position", ["HFS", "FFS", "HFP", "FFP"])
def test_patient_rotation_table_matches_iec_model(position):
    assert np.allclose(topas(*tw.PATIENT_ROT[position]), iec.POSITION[position])


def test_iec_hfs_basics():
    # gantry 0: beam from above runs posterior (+y DICOM); gantry 90: from the patient's left (+x) to the right
    _, d0 = iec.spot_ray(0, 0, 0, 0, (0, 0, 0))
    _, d90 = iec.spot_ray(0, 0, 90, 0, (0, 0, 0))
    assert np.allclose(d0, [0, 1, 0]) and np.allclose(d90, [-1, 0, 0])
    # couch 90 with gantry 90: the beam comes from the feet (patient support turned counterclockwise)
    _, d = iec.spot_ray(0, 0, 90, 90, (0, 0, 0))
    assert np.allclose(d, [0, 0, 1])


@pytest.mark.parametrize("gantry,couch,position", [(0, 0, "HFS"), (90, 0, "HFS"), (30, 310, "HFS"), (240, 45, "FFS"), (180, 0, "HFP")])
def test_writer_spot_ray_matches_iec_model(gantry, couch, position):
    """Chains the TOPAS transforms the writer sets (Spot in Nozzle in IEC_G; patient in DICOM_to_IEC in IEC_S) and
    compares where each spot crosses the isocenter plane and where it points with the independent IEC model."""
    m = beam_model.generic()
    p = small_plan(position, gantry, couch)
    f = p.fields[0]
    text, _ = tw.field_file(p, f, m, {}, "/x/patient.txt", 1000)
    vals = lambda name: [float(v) for v in re.search(rf"Tf/{name}/Values = \d+ ([^\n]*?)(?: mm| rad)?\n", text).group(1).split()]
    px, py, ax, ay = vals("PosX"), vals("PosY"), vals("AngX"), vals("AngY")
    spots = [(x, y) for l in f.layers for x, y in zip(l.x, l.y)]
    g_mat = topas(rot_y=-gantry)
    nozzle = topas(rot_x=180)
    s_to_fixed = topas(rot_z=-couch) @ topas(*tw.PATIENT_ROT[position])   # DICOM offset -> fixed
    for (x, y), a, b, c, d in zip(spots, px, py, ax, ay):
        src_f = g_mat @ (nozzle @ np.array([a, b, 0.0]) + np.array([0, 0, 1150.0]))
        dir_f = g_mat @ nozzle @ topas(math.degrees(c), math.degrees(d)) @ np.array([0, 0, 1.0])
        src = np.asarray(f.iso) + np.linalg.solve(s_to_fixed, src_f)
        u = np.linalg.solve(s_to_fixed, dir_f)
        p_iec, d_iec = iec.spot_ray(x, y, gantry, couch, f.iso, position=position)
        assert np.allclose(u, d_iec, atol=1e-7)
        # the TOPAS ray from the source plane passes through the spot point at the isocenter plane; 1 um covers the
        # 6 significant digits of the parameter file
        t = np.dot(p_iec - src, d_iec)
        assert np.allclose(src + t * u, p_iec, atol=1e-3)


# ------------------------------------------------------------------ field files
def test_histories_follow_particles_without_bias():
    m = beam_model.generic(particles_per_mu=1e6)
    f = small_plan().fields[0]
    spots, total = tw.spot_sequence(f, m, 10_000_000, seed=3)
    assert total == pytest.approx(f.total_weight * 1e6)
    hist = np.array([s[4] for s in spots]); part = np.array([s[3] for s in spots])
    assert abs(hist.sum() - 10_000_000) <= len(spots)
    assert np.allclose(hist / hist.sum(), part / part.sum(), rtol=1e-4)


def test_field_file_vectors_have_one_value_per_spot():
    p = small_plan()
    text, meta = tw.field_file(p, p.fields[0], beam_model.generic(), {}, "/x/patient.txt", 5000)
    n = meta["spots"]
    assert n == 5 and "includeFile = /x/patient.txt" in text
    for name in ("Energy", "Spread", "Histories", "PosX", "PosY", "AngX", "AngY", "SigmaX", "SigmaY", "SigmaXprime",
                 "SigmaYprime", "CorrX", "CorrY"):
        m = re.search(rf"v:Tf/{name}/Values = (\d+) ", text)
        assert m and int(m.group(1)) == n, name
    assert f"i:Tf/NumberOfSequentialTimes = {n}" in text
    assert "d:Ge/IEC_G/RotY   = -30 deg" in text and "d:Ge/IEC_S/RotZ   = -310 deg" in text


def test_beamline_places_range_shifter_and_ripple_filter():
    text = tw.beamline_block({"range_shifter": {"material": "G4_PLEXIGLASS", "thickness_mm": 30, "iso_distance_mm": 250},
                              "ripple_filter": {"variant": "t4", "iso_distance_mm": 1080, "period_mm": 1.5}})
    assert "d:Ge/RangeShifter/TransZ   = 265 mm" in text
    assert 's:Ge/RiFi1/Parent   = "IEC_G"' in text and "d:Ge/RiFi1/TransZ   = 1080 mm" in text
    assert "d:Ge/RiFi1/Length   = 20.0 cm" in text
    disp = re.search(r"dv:Ge/RiFi1/Displacement = (\d+) ", text)
    assert int(disp.group(1)) == int(200 / 1.5) + 1
    with pytest.raises(ValueError): tw.beamline_block({"ripple_filter": {"variant": "pin4", "iso_distance_mm": 1080}})
