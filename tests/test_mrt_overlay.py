"""Tests for the LETd layer on follow-up MRIs: geometry, export, ventricle margin, registration, look-list, viewer files.

The requirements tested: the layer lands where the anatomy is (known transform), the margin is 4 mm of brain around
the ventricles, LETd is hidden below 5 % of the maximum dose, the look-list finds the region and nothing else.
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
pytest.importorskip("pydicom")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from mrt_overlay import export as ex, geometry as geo, layer as ly, phantom as ph, register as rg  # noqa: E402
from mrt_overlay import series as se, ventricles as ve  # noqa: E402


def grid(size=(20, 20, 10), spacing=(1.0, 1.0, 2.0), origin=(0.0, 0.0, 0.0)):
    img = sitk.Image(list(size), sitk.sitkUInt8)
    img.SetSpacing(spacing); img.SetOrigin(origin)
    return img


def test_contour_mask_fills_square_on_its_plane():
    ref = grid()
    sq = np.array([[4, 4, 6], [13, 4, 6], [13, 9, 6], [4, 9, 6]], float)
    m = geo.contour_mask([sq], ref)
    assert m[3].sum() == 9 * 5 and m.sum() == 45            # the polygon's area, on the slice at z = 6 mm
    hole = np.array([[7, 6, 6], [9, 6, 6], [9, 7, 6], [7, 7, 6]], float)
    assert geo.contour_mask([sq, hole], ref).sum() == 45 - 2    # an inner contour cuts a hole


def test_rtdose_roundtrip_keeps_geometry(tmp_path):
    d = np.random.default_rng(1).random((4, 5, 6)) * 3
    p = str(tmp_path / "d.dcm")
    geo.write_rtdose(p, d, (-10.0, 20.0, 30.0), (2.0, 2.5, 3.0))
    img = geo.read_rtdose(p)
    assert img.GetSize() == (6, 5, 4)
    assert np.allclose(img.GetOrigin(), (-10, 20, 30)) and np.allclose(img.GetSpacing(), (2, 2.5, 3))
    assert np.allclose(sitk.GetArrayFromImage(img), d, atol=1e-6)


def _set_offsets(path, offsets, ipp_z=None):
    import pydicom
    ds = pydicom.dcmread(path)
    ds.GridFrameOffsetVector = [float(v) for v in offsets]
    if ipp_z is not None: ds.ImagePositionPatient = [float(ds.ImagePositionPatient[0]), float(ds.ImagePositionPatient[1]), float(ipp_z)]
    ds.save_as(path)


def test_rtdose_absolute_offsets_start_at_image_position(tmp_path):
    d = np.random.default_rng(2).random((4, 5, 6))
    p = str(tmp_path / "d.dcm")
    geo.write_rtdose(p, d, (-10.0, 20.0, 30.0), (2.0, 2.5, 3.0))
    _set_offsets(p, [30, 33, 36, 39])          # absolute form: the first offset is the z of ImagePositionPatient
    img = geo.read_rtdose(p)
    assert np.allclose(img.GetOrigin(), (-10, 20, 30)) and np.allclose(img.GetSpacing(), (2, 2.5, 3))


def test_export_keeps_npy_on_descending_frames(tmp_path):
    import pydicom
    run = str(tmp_path / "run")
    ph.write_run(run)
    raw = np.load(os.path.join(run, "letd.npy"))
    # LETd rising along z, so that a head to feet mirror cannot pass
    np.save(os.path.join(run, "letd.npy"), raw + np.arange(raw.shape[0])[:, None, None])
    ex.export(run, str(tmp_path / "asc"))
    ref = geo.read_rtdose(os.path.join(run, "rbe_dose_1_1.dcm"))
    nz, dz = ref.GetSize()[2], ref.GetSpacing()[2]
    top = ref.TransformIndexToPhysicalPoint((0, 0, nz - 1))[2]
    # the same run stored head first: dcm pixel frames and every npy reversed, offsets descending
    p = os.path.join(run, "rbe_dose_1_1.dcm"); ds = pydicom.dcmread(p)
    ds.PixelData = np.ascontiguousarray(ds.pixel_array[::-1]).tobytes(); ds.save_as(p)
    _set_offsets(p, [-k * dz for k in range(nz)], ipp_z=top)
    for f in os.listdir(run):
        if f.endswith(".npy"): np.save(os.path.join(run, f), np.load(os.path.join(run, f))[::-1])
    ex.export(run, str(tmp_path / "desc"))
    for name in ("letd", "dose_phys"):
        i1 = sitk.ReadImage(str(tmp_path / "asc" / f"{name}.nii.gz"))
        i2 = sitk.ReadImage(str(tmp_path / "desc" / f"{name}.nii.gz"))
        assert np.allclose(i1.GetOrigin(), i2.GetOrigin()) and np.allclose(i1.GetDirection(), i2.GetDirection())
        assert np.allclose(sitk.GetArrayFromImage(i1), sitk.GetArrayFromImage(i2), rtol=1e-5)


def test_export_hides_low_dose_letd_and_applies_bahn_line(tmp_path):
    run, out = str(tmp_path / "run"), str(tmp_path / "frame")
    ph.write_run(run)
    s = ex.export(run, out, fractions=30)
    letd = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(out, "letd.nii.gz")))
    phys = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(out, "dose_phys.nii.gz")))
    bahn = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(out, "dose_rbe_bahn_linear.nii.gz")))
    raw = np.load(os.path.join(run, "letd.npy"))
    low = phys < 0.05 * phys.max()
    assert low.any() and (letd[low] == 0).all() and np.allclose(letd[~low], raw[~low])
    assert np.allclose(bahn[~low], (1 + 0.11 * raw[~low]) * phys[~low], rtol=1e-4)
    assert np.allclose(bahn[low], 1.1 * phys[low], rtol=1e-4)
    assert abs(s["dose_max_phys_gy"] - phys.max()) < 1e-3 and 55 < phys.max() <= 60   # 2 Gy x 30 at the centre
    ref = sitk.ReadImage(os.path.join(out, "letd.nii.gz"))
    assert np.allclose(ref.GetSpacing(), (2.5, 2.5, 3.0)) and np.allclose(ref.GetOrigin(), (-75, -87, -84))


def test_margin_is_4mm_of_brain_around_ventricles():
    n, sp = 61, 1.0
    z, y, x = np.indices((n, n, n)) - 30
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lab = np.where(r <= 25, 2, 0).astype(np.int16)
    lab[r <= 8] = 4                                       # a ventricle ball of 8 mm radius in brain
    lab[(r <= 25) & (x > 10)] = 24                        # CSF there is not brain
    img = geo.like(lab, grid((n, n, n), (sp, sp, sp)))
    vent, rim = ve.margin(img)
    rim = sitk.GetArrayFromImage(rim).astype(bool)
    assert not (rim & (r <= 8)).any() and not (rim & (lab == 24)).any()
    assert rim[r <= 11.5].sum() > 0.9 * ((r > 8) & (r <= 11.5)).sum()
    assert not rim[r > 12.01].any()
    # distances run between voxel centres, so the outer edge lies between 11.5 and 12 mm
    shell = lambda out: int(((r > 8) & (r <= out) & (lab == 2)).sum())
    assert shell(11.5) <= int(rim.sum()) <= shell(12.0)


def test_margin_uses_mm_on_anisotropic_grid():
    lab = np.full((21, 41, 41), 2, np.int16)
    lab[10, 20, 20] = 4
    img = geo.like(lab, grid((41, 41, 21), (0.5, 0.5, 2.0)))
    _, rim = ve.margin(img)
    rim = sitk.GetArrayFromImage(rim).astype(bool)
    assert rim[10, 20, 20 + 8] and not rim[10, 20, 20 + 9]      # 4 mm in x is 8 voxels of 0.5 mm
    assert rim[12, 20, 20] and not rim[13, 20, 20]              # 4 mm in z is 2 slices of 2 mm


@pytest.fixture(scope="module")
def head():
    return ph.make()


def test_registration_recovers_known_transform(head):
    ct, mri, labels, truth = head
    pts = geo.brain_points(geo.like((sitk.GetArrayFromImage(labels) > 0).astype(np.uint8), mri), 6)
    t, q = rg.rigid(mri, ct, mask_points=pts)
    rms, mx = geo.displacement_mm(t, truth, pts)
    assert rms < rg.ACCEPT_MM and mx < 2.0
    assert q["starts_same_optimum"] >= 2 and q["spread_rms_mm"] < rg.ACCEPT_MM


def test_layer_lands_on_anatomy_and_look_list_finds_region(head, tmp_path):
    ct, mri, labels, truth = head
    run, frame = str(tmp_path / "run"), str(tmp_path / "frame")
    # the blob sits next to the left ventricle in CT coordinates of the phantom anatomy
    ph.write_run(run, centre_mm=(-11.0, 4.0, 24.0), sigma_mm=8.0)
    ex.export(run, frame)
    layers = ly.to_grid(frame, mri, truth)
    w = layers["dose_phys"]
    k, j, i = np.indices(w.shape)
    cen = np.array(mri.TransformContinuousIndexToPhysicalPoint([float((w * g).sum() / w.sum()) for g in (i, j, k)]))
    # on the MRI the blob is where the CT point lies after the inverse of MRI->CT
    expect = np.array(truth.GetInverse().TransformPoint((-11.0, 4.0, 24.0)))
    assert np.linalg.norm(cen - expect) < 1.0
    _, rim = ve.margin(labels)
    rim = sitk.GetArrayFromImage(rim).astype(bool)
    look, hit = ly.look_list(layers["letd"], layers["dose_rbe_1_1"], rim, 2.2, mri.GetSpacing())
    assert len(look) >= 1 and hit.any()
    assert (hit <= rim).all() and (layers["letd"][hit] >= 4).all() and (layers["dose_rbe_1_1"][hit] >= 1.1).all()
    lo, hi = look[0]["slices"]
    assert 1 <= lo <= hi <= w.shape[0]
    # nothing to look at where the LETd threshold is not reached
    none, _ = ly.look_list(np.minimum(layers["letd"], 3.9), layers["dose_rbe_1_1"], rim, 2.2, mri.GetSpacing())
    assert none == []


def test_marks_bits_and_viewer_files_roundtrip(tmp_path):
    d = np.zeros((6, 8, 10), np.float32); d[1:5, 2:6, 2:8] = 2.0; d[2:4, 3:5, 3:7] = 2.1
    layers = {"dose_rbe_1_1": d, "letd": np.full(d.shape, 3.0, np.float32), "ptv": d >= 2.1}
    margin = np.zeros(d.shape, bool); margin[0] = True
    hit = np.zeros(d.shape, bool); hit[3, 4, 4] = True
    mk = ly.marks(layers, margin, 2.0, hit)
    assert (mk[0] & ly.MARK["margin"]).all() and mk[3, 4, 4] & ly.MARK["look"]
    assert mk[1, 2, 2] & ly.MARK["iso95"] and not mk[2, 3, 3] & ly.MARK["iso95"]   # a shell, not the inside
    assert (mk[2:4, 3:5, 3:7] & ly.MARK["ptv"]).any() and not (mk[d == 0] & ly.MARK["iso50"]).any()
    meta = ly.write_viewer(str(tmp_path), layers, mk, {"exams": []}, cols=4)
    js = open(tmp_path / "letd_all.js").read()
    assert js.startswith('MRT.vol("letd","all","data:image/png;base64,')
    from PIL import Image
    png = Image.open(io.BytesIO(base64.b64decode(js.split("base64,")[1].rsplit('"', 1)[0])))
    a = np.array(png)
    assert a.shape == (2 * 8, 4 * 10) and a[0, 0] == round(3.0 * ly.LET_SCALE)
    assert json.load(open(tmp_path / "letd.json"))["let_steps"] == [2.0, 3.0, 5.0] == meta["let_steps"]


def test_series_kind_from_headers():
    from pydicom.dataset import Dataset

    def ds(desc, agent="", image_type=("ORIGINAL", "PRIMARY")):
        d = Dataset(); d.SeriesDescription = desc; d.ContrastBolusAgent = agent; d.ImageType = list(image_type)
        return d
    assert se.kind(ds("t1_mprage_sag")) == "t1"
    assert se.kind(ds("t1_mprage_sag", agent="Gadovist")) == "t1_km"
    assert se.kind(ds("T1 3D VIBE KM")) == "t1_km"           # Philips and others: contrast only in the name
    assert se.kind(ds("t2_flair_3d")) == "flair"
    assert se.kind(ds("t1_mprage MPR tra")) is None          # scanner reformat
    assert se.kind(ds("t1_mprage", image_type=("DERIVED", "SECONDARY", "MPR"))) is None
    assert se.kind(ds("t1 sub")) is None and se.kind(ds("ep2d_diff")) is None
    e = [{"date": "1", "kind": "t1", "suitable": True, "slice_thickness_mm": 1.0, "n": 1},
         {"date": "1", "kind": "t1", "suitable": True, "slice_thickness_mm": 0.9, "n": 2},
         {"date": "1", "kind": "flair", "suitable": False, "slice_thickness_mm": 5.0, "n": 3}]
    best = se.pick(e)
    assert best[("1", "t1")]["n"] == 2 and ("1", "flair") not in best


def test_thick_series_marked_unsuitable():
    assert ly.exam_suitable([0.9, 0.9, 1.0]) and not ly.exam_suitable([0.45, 0.45, 5.5])


def test_viewer_command_end_to_end(head, tmp_path):
    """The viewer route on a fake viewer cache: reference T1 on the fine grid, one exam, SynthSeg-like labels."""
    from mrt_overlay import cli
    ct, mri, labels, truth = head
    cache, run, frame, out, data = (str(tmp_path / n) for n in ("cacheF", "run", "frame", "out", "data"))
    os.makedirs(cache)
    sitk.WriteImage(mri, os.path.join(cache, "ref_t1.nii.gz"))
    a = sitk.GetArrayFromImage(mri)
    a8 = np.clip(a / a.max() * 255, 0, 255).astype(np.uint8)
    np.save(os.path.join(cache, "t1_2020-01-01.npy"), a8)
    # a later exam that the viewer cache holds 3 voxels (4.8 mm) off the reference: its check must fail
    np.save(os.path.join(cache, "t1_2020-06-01.npy"), np.roll(a8, 3, axis=2))
    json.dump({"size": list(mri.GetSize()), "spacing": 1.6,
               "t1 2020-01-01": {"series": "x", "kind": "3D", "native": [1.0, 1.0, 1.0]},
               "t1 2020-06-01": {"series": "z", "kind": "3D", "native": [1.0, 1.0, 1.0]},
               "flair 2020-01-01": {"series": "y", "kind": "2D", "native": [0.5, 0.5, 5.0]}},
              open(os.path.join(cache, "log.json"), "w"))
    sitk.WriteImage(ct, str(tmp_path / "ct.nii.gz"))
    sitk.WriteImage(labels, str(tmp_path / "seg.nii.gz"))
    ph.write_run(run, centre_mm=(-11.0, 4.0, 24.0), sigma_mm=8.0)
    cli.main(["export", run, frame, "--fractions", "1"])
    cli.main(["viewer", cache, str(tmp_path / "ct.nii.gz"), frame, out, "--labels", str(tmp_path / "seg.nii.gz"),
              "--prescription", "2.2", "--viewer-data", data, "--kennzahlen", str(tmp_path / "robin" / "kennzahlen.json")])
    info = json.load(open(os.path.join(out, "overlay.json")))
    exams = {(e["track"], e["day"]): e for e in info["exams"]}
    same, moved = exams[("t1", "2020-01-01")], exams[("t1", "2020-06-01")]
    assert same["suitable"] and not exams[("flair", "2020-01-01")]["suitable"]
    assert same["check_rms_mm"] < 1.5 and same["check_ok"]
    assert 3.8 < moved["check_rms_mm"] < 5.8 and not moved["check_ok"]
    assert info["look_list"] and os.path.exists(os.path.join(data, "letd_all.js"))
    # the viewer's legend names only the contours that exist
    meta = json.load(open(os.path.join(data, "letd.json")))
    assert meta["margin"] is True and meta["ptv"] is False
    k = json.load(open(tmp_path / "robin" / "kennzahlen.json"))
    assert k["exams_unsuitable"] == 1 and k["exams_check_over_1_5mm"] == 1 and k["look_regions"] == len(info["look_list"])
    assert not any(isinstance(v, str) for v in k.values())         # numbers only, no names or paths


def test_synthseg_call_is_robust_on_cpu_and_cropped(tmp_path):
    """The call must fit the 15 GB of the WSL: robust model on the CPU, cropped to a 192 mm cube."""
    seen = tmp_path / "argv.json"
    fake = tmp_path / "fake_synthseg.py"
    fake.write_text(f"import json, sys\njson.dump(sys.argv[1:], open({str(seen)!r}, 'w'))\n")
    ve.run_synthseg("in.nii.gz", "out.nii.gz", command=f"{sys.executable} {fake}")
    argv = json.load(open(seen))
    assert argv[:4] == ["--i", "in.nii.gz", "--o", "out.nii.gz"]
    assert "--robust" in argv and "--cpu" in argv and argv[argv.index("--crop") + 1] == "192"


def _case(folder, d_change):
    sys.path.insert(0, ROOT)
    from opentopas_workbench import fall
    fall.update(str(folder), d_change, create="test-fall")
    return fall


def test_viewer_command_from_case_file(head, tmp_path):
    """--fall: every path from fall.json, the export done when ct_frame is missing, and the registration of each
    exam and the viewer result written back, so that fall_check sees the result as current."""
    from mrt_overlay import cli
    ct, mri, labels, truth = head
    case = tmp_path / "fall"
    cache = case / "cacheF"; os.makedirs(cache)
    sitk.WriteImage(mri, str(cache / "ref_t1.nii.gz"))
    a = sitk.GetArrayFromImage(mri)
    a8 = np.clip(a / a.max() * 255, 0, 255).astype(np.uint8)
    np.save(cache / "t1_2020-01-01.npy", a8)
    np.save(cache / "t1_2020-06-01.npy", np.roll(a8, 3, axis=2))
    json.dump({"t1 2020-01-01": {"series": "x", "kind": "3D", "native": [1.0, 1.0, 1.0]},
               "t1 2020-06-01": {"series": "z", "kind": "3D", "native": [1.0, 1.0, 1.0]}}, open(cache / "log.json", "w"))
    os.makedirs(case / "ct")
    sitk.WriteImage(ct, str(case / "ct" / "ct.nii.gz"))
    sitk.WriteImage(labels, str(tmp_path / "seg.nii.gz"))
    ph.write_run(str(case / "run"), centre_mm=(-11.0, 4.0, 24.0), sigma_mm=8.0)

    def setup(d):
        d["bilder"] = [{"serie": "ct_plan", "modalitaet": "CT", "datum": None, "ordner": "ct/ct.nii.gz",
                        "nach_bezug": None, "guete": None}]
        d["bezug"] = {"serie": "ct_plan"}
        d["plan_ohne_datei"] = {"fraktionen": 1, "verschreibung_gy_rbe": 2.2}
        d["viewer"] = {"cache": "cacheF", "daten": "data"}
        d["einstellungen"] = {"ausgabe": "out", "synthseg": False, "lauf": "mc-1"}
        d["ergebnisse"]["workbench"] = [{"id": "mc-1", "ordner": "run", "eingaben": {}, "dateien": {}}]
    fall = _case(case, setup)
    cli.main(["viewer", "--fall", str(case), "--labels", str(tmp_path / "seg.nii.gz")])
    assert os.path.isfile(case / "out" / "ct_frame" / "dose_rbe_1_1.nii.gz")
    info = json.load(open(case / "out" / "viewer_grid" / "overlay.json"))
    assert info["prescription_gy_rbe"] == 2.2 and info["margin"] is True
    d = fall.load(str(case))
    mr = {b["serie"]: b for b in d["bilder"] if b["modalitaet"] == "MR"}
    assert set(mr) == {"t1_2020-01-01", "t1_2020-06-01"}
    assert mr["t1_2020-01-01"]["guete"]["ok"] is True and mr["t1_2020-06-01"]["guete"]["ok"] is False
    assert mr["t1_2020-01-01"]["nach_bezug"] == "out/viewer_grid/mri_to_ct.tfm" and mr["t1_2020-01-01"]["datum"] == "2020-01-01"
    (v,) = d["ergebnisse"]["viewer"]
    assert v["aus"] == ["mc-1"] and v["ordner"] == "data" and set(v["dateien"]) == {"letd_all.js", "marks_all.js", "letd.json"}
    rep = fall.check(str(case))
    assert {(r["art"], r["zustand"]) for r in rep["ergebnisse"]} >= {("viewer", "aktuell")}
    assert "bilder t1_2020-06-01: Registrierung nicht ok" in rep["warnungen"]
    # a viewer file changed after the run: the result is broken, and fall_check says so
    open(case / "data" / "letd.json", "a").write(" ")
    assert ("viewer", "kaputt") in {(r["art"], r["zustand"]) for r in fall.check(str(case))["ergebnisse"]}


def test_viewer_command_needs_paths_or_case():
    from mrt_overlay import cli
    with pytest.raises(SystemExit, match="--fall"):
        cli.main(["viewer", "only-a-cache"])
