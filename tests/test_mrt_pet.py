"""Tests for the PET layer of the MRI viewer, on synthetic data only (phantom.py).

The requirements tested: a PET whose header gives no position (origin 0, unit orientation, shuffled files, a slope
per slice) is read in slice order with its values; the registration finds a known rotation and shift; a PET with
left and right swapped, or standing on its head, stops the run; the ratio volume lands in the viewer files with one entry per PET date.
"""
import base64
import io
import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
sitk = pytest.importorskip("SimpleITK")
pydicom = pytest.importorskip("pydicom")
pytest.importorskip("PIL")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from mrt_overlay import geometry as geo, pet as pe, phantom as ph  # noqa: E402

DAY = "2020-03-01"


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    """A fake viewer cache: reference T1 grid, brain mask, two 3D T1 exams and one 2D exam."""
    d = tmp_path_factory.mktemp("cacheF")
    t1, brain = ph.t1_patient()
    sitk.WriteImage(t1, str(d / "ref_t1.nii.gz"))
    a = sitk.GetArrayFromImage(t1)
    np.save(d / "t1_2019-12-01.npy", np.clip(a / a.max() * 255, 0, 255).astype(np.uint8))
    np.save(d / "t1_2021-01-01.npy", np.zeros(a.shape, np.uint8))      # far in time: must not be chosen
    np.save(d / "mask.npy", brain)
    json.dump({"size": list(t1.GetSize()), "spacing": 1.6,
               "t1 2019-12-01": {"series": "x", "kind": "3D", "native": [1.0, 1.0, 1.0]},
               "t1 2021-01-01": {"series": "y", "kind": "3D", "native": [1.0, 1.0, 1.0]},
               "flair 2020-03-01": {"series": "z", "kind": "2D", "native": [0.5, 0.5, 5.0]}},
              open(d / "log.json", "w"))
    return str(d), t1, brain


@pytest.fixture(scope="module")
def pet_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pet")
    v = ph.write_pet(str(d))
    return str(d), v


def decode(js, shape, cols=12):
    from PIL import Image
    url = js.split('"')[5]
    m = np.array(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))
    nz, ny, nx = shape
    return np.stack([m[(z // cols) * ny:(z // cols + 1) * ny, (z % cols) * nx:(z % cols + 1) * nx] for z in range(nz)])


def test_read_pet_without_header_position(pet_dir):
    d, v = pet_dir
    img, facts = pe.read_pet(d)
    assert img.GetSize() == ph.PET_SIZE and np.allclose(img.GetSpacing(), ph.PET_SPACING)
    assert img.GetOrigin() == (0.0, 0.0, 0.0) and np.allclose(img.GetDirection(), np.eye(3).ravel())
    # slices in position order although the file names are shuffled, each with its own slope
    a = sitk.GetArrayFromImage(img)
    assert np.allclose(a, v, atol=v.max() / 65535 * 1.01)
    assert facts["sorted_by"] == "ImagePositionPatient" and facts["day"] == DAY
    assert facts["units"] == "NONE" and facts["tracer"] == "Phantom-Tracer"


def test_read_pet_falls_back_to_instance_number(tmp_path, pet_dir):
    d, v = pet_dir
    for n in os.listdir(d):
        ds = pydicom.dcmread(os.path.join(d, n)); ds.ImagePositionPatient = [0.0, 0.0, 0.0]
        ds.save_as(str(tmp_path / n))
    img, facts = pe.read_pet(str(tmp_path))
    assert facts["sorted_by"] == "InstanceNumber" and np.allclose(img.GetSpacing()[2], ph.PET_SPACING[2])
    assert np.allclose(sitk.GetArrayFromImage(img), v, atol=v.max() / 65535 * 1.01)


def test_pet_command_finds_known_transform_and_writes_viewer_files(cache, pet_dir, tmp_path):
    from mrt_overlay import cli
    c, t1, brain = cache
    data, out = str(tmp_path / "data"), str(tmp_path / "out")
    cli.main(["pet", pet_dir[0], c, data, "--out", out])
    t = sitk.ReadTransform(os.path.join(out, f"grid_to_pet_{DAY}.tfm"))
    pts = geo.brain_points(geo.like(brain.astype(np.uint8), t1), 6)
    rms, mx = geo.displacement_mm(t, ph.pet_true_transform().GetInverse(), pts)
    assert rms < 1.0 and mx < 1.5                    # PET voxels are 2 to 2.5 mm
    meta = json.load(open(os.path.join(data, "pet.json"), encoding="utf-8"))
    (s,) = meta["scans"]
    assert s["day"] == DAY and s["fixed_t1"] == "2019-12-01" and s["suv_per_ratio"] is None
    assert s["registration"]["abstand_relativ"] >= pe.MIRROR_MARGIN and s["registration"]["starts"] == 9
    assert s["registration"]["abstand_kopfstehend_relativ"] >= pe.MIRROR_MARGIN
    # the control sheet: T1 above, PET on T1 below, three views; colour only in the lower row
    from PIL import Image
    run = json.load(open(os.path.join(out, f"pet_{DAY}.json"), encoding="utf-8"))
    im = np.asarray(Image.open(run["sheet"]).convert("RGB")).astype(int)
    assert im.shape[0] == 24 + 2 * pe.SHEET_PX + 3 * 6
    colour = im[..., 0] - im[..., 1] > 40                      # red to orange; the yellow R and L marks are not
    half = 24 + 6 + pe.SHEET_PX
    assert colour[:half].sum() == 0 and colour[half:].sum() > 200
    assert "sheet" not in s                                      # the viewer's pet.json names no local path
    # the viewer file: ratio bytes on the grid; the lesion on the patient's right is hot, its mirror place is not
    vol = decode(open(os.path.join(data, f"pet_{DAY}.js")).read(), brain.shape) * meta["per_byte"]
    c_r = ph.PET_HOT[2][0]
    lesion = t1.TransformPhysicalPointToIndex(c_r)
    mirror = t1.TransformPhysicalPointToIndex((-c_r[0], c_r[1], c_r[2]))
    assert vol[lesion[::-1]] > 2.0 and vol[mirror[::-1]] < 1.3
    assert abs(np.median(vol[brain]) - 1.0) < 0.02     # reference: median in the brain


def test_mirrored_pet_stops_the_run(cache, tmp_path):
    c, *_ = cache
    d, data = str(tmp_path / "pet"), str(tmp_path / "data")
    ph.write_pet(d, mirrored=True)
    with pytest.raises(SystemExit, match="Seitenprüfung"):
        pe.run(d, c, data)
    assert not os.path.exists(data)                  # nothing written


def test_upside_down_pet_stops_the_run(cache, tmp_path):
    # slice order reversed (z falls with the anatomy): no rigid turn undoes it, the head to foot check must stop
    c, *_ = cache
    d, data = str(tmp_path / "pet"), str(tmp_path / "data")
    ph.write_pet(d, upside_down=True)
    with pytest.raises(SystemExit, match="Kopf-Fuß"):
        pe.run(d, c, data, start_deg=(0.0,))
    assert not os.path.exists(data)


def test_second_pet_date_adds_an_entry(cache, pet_dir, tmp_path, monkeypatch):
    c, *_ = cache
    data = str(tmp_path / "data")
    truth = ph.pet_true_transform().GetInverse()
    monkeypatch.setattr(pe, "side_check", lambda *a, **k: (truth, {"mi": -1.0}))
    pe.run(pet_dir[0], c, data)
    pe.run(pet_dir[0], c, data, day="2020-09-01", ref="contralateral", side="rechts")
    meta = json.load(open(os.path.join(data, "pet.json"), encoding="utf-8"))
    assert [s["day"] for s in meta["scans"]] == [DAY, "2020-09-01"]
    assert meta["scans"][1]["fixed_t1"] == "2021-01-01"      # nearer in time to September than December 2019 is
    assert "linken" in meta["scans"][1]["reference"]
    assert {f for f in os.listdir(data)} == {"pet.json", f"pet_{DAY}.js", "pet_2020-09-01.js"}


def test_reference_regions_and_suv_refusal(cache, pet_dir, tmp_path, monkeypatch):
    c, t1, brain = cache
    m, _ = pe.reference_mask("contralateral", brain, t1, side="links")
    x_mm = t1.GetOrigin()[0] + np.nonzero(m)[2] * t1.GetSpacing()[0]
    assert m.sum() > 0.4 * brain.sum() and (x_mm < 1.0).all()      # side links: the right half, LPS x below 0
    lab = np.zeros(brain.shape, np.int16); lab[brain] = 2
    z, y, x = np.nonzero(brain); lab[z[:50], y[:50], x[:50]] = 11
    sitk.WriteImage(geo.like(lab, t1), str(tmp_path / "seg.nii.gz"))
    m, name = pe.reference_mask("striatum", brain, t1, labels=str(tmp_path / "seg.nii.gz"))
    assert m.sum() == 50 and "Striatum" in name
    monkeypatch.setattr(pe, "side_check", lambda *a, **k: (ph.pet_true_transform().GetInverse(), {}))
    with pytest.raises(SystemExit, match="Bq/ml"):
        pe.run(pet_dir[0], c, str(tmp_path / "d"), weight_kg=70, activity_mbq=200)


def test_reference_median_ignores_grid_outside_the_pet():
    # half the brain lies outside the PET field of view and resamples to 0; the median must come from the covered half
    pet_on_grid = np.zeros((4, 4, 4), np.float32)
    pet_on_grid[:2] = 2.0
    brain = np.ones((4, 4, 4), bool)
    covered = np.zeros((4, 4, 4), bool)
    covered[:2] = True
    ratio, ref = pe.ratio_volume(pet_on_grid, brain, covered)
    assert ref == 2.0 and float(ratio[0, 0, 0]) == 1.0
    with pytest.raises(SystemExit):
        pe.ratio_volume(pet_on_grid, brain, np.zeros_like(covered))
