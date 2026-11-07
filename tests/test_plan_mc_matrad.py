"""Tests for the matRad reader (tools/plan_mc/matrad.py)."""
import os, sys

import pytest

np = pytest.importorskip("numpy")
sio = pytest.importorskip("scipy.io")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from plan_mc import iec, matrad  # noqa: E402

SAD = 10000.0


@pytest.mark.parametrize("gantry,couch", [(0, 0), (90, 0), (270, 0), (45, 310), (300, 240), (180, 90), (120, 20)])
def test_matrad_ray_equals_the_iec_ray_with_x_bev_as_iec_x_and_z_bev_as_iec_y(gantry, couch):
    # matRad: source at R @ (0, -SAD, 0), spot at R @ rayPos_bev, both relative to the isocenter (LPS = DICOM)
    iso = np.array([12.0, -30.0, 47.0])
    r = matrad.rotation(gantry, couch)
    for xb, zb in [(20.0, -15.0), (-40.0, 33.0), (0.0, 0.0)]:
        spot = iso + r @ np.array([xb, 0.0, zb])
        src = iso + r @ np.array([0.0, -SAD, 0.0])
        d = (spot - src) / np.linalg.norm(spot - src)
        p, d_iec = iec.spot_ray(xb, zb, gantry, couch, iso, vsad=(SAD, SAD), position="HFS")
        assert p == pytest.approx(spot, abs=1e-9)
        assert d_iec == pytest.approx(d, abs=1e-9)


def test_gantry_90_beam_comes_from_patient_left():
    # LPS: +x is patient left; the beam runs from the source towards the isocenter, so along -x
    r = matrad.rotation(90, 0)
    assert r @ np.array([0.0, 1.0, 0.0]) == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)


def _cell(v):
    c = np.empty((1,), dtype=object); c[0] = v
    return c


def _mat(path, hu_yxz, weights, rays, position=None, dose=None, axes=True, rs=None):
    ny, nx, nz = hu_yxz.shape
    ct = {"cubeHU": _cell(hu_yxz), "resolution": {"x": 2.0, "y": 3.0, "z": 4.0},
          "cubeDim": np.array([ny, nx, nz])}
    if axes:
        ct.update(x=-10.0 + 2.0 * np.arange(nx), y=5.0 + 3.0 * np.arange(ny), z=-20.0 + 4.0 * np.arange(nz))
    if position: ct["dicomInfo"] = {"PatientPosition": position}
    ray = []
    for pos, energies in rays:
        r = {"rayPos_bev": np.array(pos, float), "energy": np.array(energies, float)}
        if rs is not None:
            r["rangeShifter"] = np.array([{"ID": 1, "eqThickness": rs, "sourceRashiDistance": SAD - 300.0}
                                          for _ in energies], dtype=object)
        ray.append(r)
    stf = [{"gantryAngle": 90.0, "couchAngle": 0.0, "SAD": SAD, "isoCenter": np.array([1.0, 2.0, 3.0]),
            "ray": np.array(ray, dtype=object)}]
    cst = np.empty((1, 6), dtype=object)
    target = np.ravel_multi_index(([1], [2], [3]), (ny, nx, nz), order="F") + 1   # voxel y=1, x=2, z=3
    cst[0, :] = [0, "PTV", "TARGET", _cell(target.astype(float)), {}, {}]
    res = {"w": np.array(weights, float)}
    if dose is not None: res["physicalDose"] = dose
    sio.savemat(path, {"ct": ct, "cst": cst, "stf": np.array(stf, dtype=object), "resultGUI": res})
    return path


@pytest.fixture
def cube():
    hu = np.zeros((4, 5, 6), np.int16)   # [y, x, z]
    hu[1, 2, 3] = 1000
    return hu


def test_ct_and_cst_land_on_the_same_voxel_and_axes(cube, tmp_path):
    p = _mat(str(tmp_path / "a.mat"), cube, [1.0, 2.0], [((0, 0, 0), [100.0, 90.0])])
    r = matrad.read(p)
    assert r["hu"].shape == (6, 4, 5)                 # [z, y, x]
    assert r["hu"][3, 1, 2] == 1000 and r["hu"].sum() == 1000
    assert r["origin"] == (-10.0, 5.0, -20.0) and r["spacing"] == (2.0, 3.0, 4.0)
    name, typ, mask = r["cst"][0]
    assert (name, typ, int(mask.sum())) == ("PTV", "TARGET", 1) and mask[3, 1, 2]


def test_spots_in_bixel_order_with_particles_and_zero_weights_dropped(cube, tmp_path):
    rays = [((10, 0, -5), [120.0, 100.0]), ((-4, 0, 7), [120.0])]
    p = _mat(str(tmp_path / "a.mat"), cube, [1.5, 0.0, 2.5], rays)
    f = matrad.read(p)["plan"].fields[0]
    assert (f.gantry, f.couch, f.iso, f.unit) == (90.0, 0.0, (1.0, 2.0, 3.0), "particles")
    assert [l.energy for l in f.layers] == [120.0]    # the 100 MeV spot has weight 0
    l = f.layers[0]
    assert l.x == [10.0, -4.0] and l.y == [-5.0, 7.0]  # IEC X = x_bev, IEC Y = z_bev
    assert l.w == pytest.approx([1.5e6, 2.5e6])


def test_range_shifter_is_taken_over(cube, tmp_path):
    p = _mat(str(tmp_path / "a.mat"), cube, [1.0], [((0, 0, 0), [80.0])], rs=40.0)
    rs = matrad.read(p)["plan"].fields[0].range_shifter
    assert rs.wet == 40.0 and rs.iso_distance == pytest.approx(300.0)


def test_dose_is_returned_as_zyx(cube, tmp_path):
    dose = np.zeros(cube.shape); dose[1, 2, 3] = 2.0
    r = matrad.read(_mat(str(tmp_path / "a.mat"), cube, [1.0], [((0, 0, 0), [80.0])], dose=dose))
    assert r["dose"][3, 1, 2] == 2.0


@pytest.mark.parametrize("kw,msg", [({"weights": [1.0, 2.0, 3.0]}, "bixels"), ({"position": "FFS"}, "HFS"),
                                    ({"axes": False}, "world axes")])
def test_refuses_what_it_cannot_map(cube, tmp_path, kw, msg):
    args = {"weights": [1.0, 2.0], "rays": [((0, 0, 0), [100.0, 90.0])]}
    args.update(kw)
    with pytest.raises(ValueError, match=msg):
        matrad.read(_mat(str(tmp_path / "a.mat"), cube, **args))
