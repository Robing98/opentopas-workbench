"""Tests for the synthetic DICOM writers (tools/plan_mc/synthetic.py) and the queue helper (wbqueue.py)."""
import json, os, sys

import pytest

np = pytest.importorskip("numpy")
pydicom = pytest.importorskip("pydicom")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from plan_mc import synthetic, wbqueue  # noqa: E402


@pytest.mark.parametrize("nz", [1, 4])
def test_ct_round_trip_keeps_hu_and_geometry(tmp_path, nz):
    hu = np.arange(nz * 3 * 5, dtype=np.int16).reshape(nz, 3, 5) * 37 - 1000
    synthetic.write_ct(str(tmp_path / "ct"), hu, (-10.0, 20.0, -5.0), (2.0, 0.5, 3.0))
    back, origin, sp = synthetic.read_ct(str(tmp_path / "ct"))
    assert np.array_equal(back, hu)
    assert origin == pytest.approx((-10.0, 20.0, -5.0))
    if nz > 1: assert sp == pytest.approx((2.0, 0.5, 3.0))
    ds = pydicom.dcmread(str(tmp_path / "ct" / "CT_0000.dcm"))
    # DICOM PixelSpacing is (row spacing = y, column spacing = x)
    assert [float(v) for v in ds.PixelSpacing] == [0.5, 2.0] and (ds.Rows, ds.Columns) == (3, 5)


@pytest.mark.parametrize("nz", [1, 3])
def test_rtdose_round_trip(tmp_path, nz):
    rng = np.random.default_rng(1)
    d = rng.random((nz, 4, 6)) * 2.5
    p = str(tmp_path / "rd.dcm")
    synthetic.write_rtdose(p, d, (1.0, 2.0, 3.0), (2.5, 2.5, 2.0), plan_uid="1.2.3")
    back, origin, sp = synthetic.read_rtdose(p)
    assert back.shape == d.shape and np.allclose(back, d, atol=2.5 / 2 ** 31)
    assert origin == pytest.approx((1.0, 2.0, 3.0))
    if nz > 1: assert sp == pytest.approx((2.5, 2.5, 2.0))
    assert pydicom.dcmread(p).ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID == "1.2.3"


def test_slab_phantom_rows():
    hu = synthetic.slab_phantom((2, 10, 3), 0, [(2, 4, 1000), (6, 9, -500)])
    assert list(hu[0, :, 0]) == [0, 0, 1000, 1000, 0, 0, -500, -500, -500, 0]


def test_queue_paths_are_workspace_relative(tmp_path):
    ws = tmp_path / "ws"; (ws / "a").mkdir(parents=True)
    assert wbqueue.rel(str(ws), str(ws / "a" / "f.txt")) == "a/f.txt"
    assert wbqueue.rel(str(ws), "a/f.txt") == "a/f.txt"
    with pytest.raises(ValueError): wbqueue.rel(str(ws), str(tmp_path / "other.txt"))


def test_queue_without_server_stops(tmp_path):
    with pytest.raises(SystemExit): wbqueue.enqueue(str(tmp_path), ["x.txt"])
