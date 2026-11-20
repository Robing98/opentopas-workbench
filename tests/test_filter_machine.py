"""Tests for tools/filter_machine.py: constant WET grid, filter folding, and added spot size on a synthetic machine."""
import os, sys

import pytest

np = pytest.importorskip("numpy")
sio = pytest.importorskip("scipy.io")
from scipy.special import erfc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import filter_machine as fm  # noqa: E402


def bragg(z, r):
    peak = np.exp(-0.5 * ((z - r + 1.0) / (0.6 + 0.012 * r)) ** 2) * 0.9 + 0.35 * np.exp(-(r - z).clip(0) / 6.0)
    return (0.3 + 0.2 * z / r + peak) * 0.5 * erfc((z - r) / (np.sqrt(2) * (0.3 + 0.011 * r)))


def template(tmp_path):
    """Machine like protons_Generic: about 3 mm between peaks, irregular depth grid, focus and spectrum structs."""
    ranges = np.arange(20.0, 120.0, 3.1)
    names = ["energy", "range", "peakPos", "energySpectrum", "initFocus", "offset", "depths", "Z", "sigma", "sigma1", "sigma2", "weight", "LET"]
    data = np.zeros((1, len(ranges)), dtype=[(n, object) for n in names])
    for j, r in enumerate(ranges):
        z = np.unique(np.round(np.concatenate([np.arange(0, r - 12, 2.0), np.arange(r - 12, r + 6, 0.2), np.arange(r + 6, r + 20, 1.0)]), 3))
        Z = bragg(z, r)
        e = data[0, j]
        e["energy"] = np.array([[(r / 0.0022) ** (1 / 1.77)]]); e["range"] = np.array([[r]])
        e["peakPos"] = np.array([[z[np.argmax(Z)] + 0.3]])          # deliberately off the curve maximum, as in Generic
        e["energySpectrum"] = {"type": "gaussian", "mean": e["energy"], "sigma": e["energy"] * 0.01}
        e["initFocus"] = {"dist": np.array([[9000.0, 10000.0]]), "sigma": np.array([[3.0, 3.0]]), "SisFWHMAtIso": np.array([[3.0 * 2.3548]]),
                          "emittance": {"type": "bigaussian", "sigmaX": 3.0, "divX": 0.0, "corrX": 0.0, "sigmaY": 3.0, "divY": 0.0, "corrY": 0.0}}
        e["offset"] = np.array([[0.0]])
        e["depths"] = z.reshape(-1, 1); e["Z"] = Z.reshape(-1, 1)
        for k in ("sigma", "sigma1", "sigma2"): e[k] = (0.02 * z).reshape(-1, 1)
        e["weight"] = np.zeros((len(z), 1)); e["LET"] = (1 + z / r).reshape(-1, 1)
    p = tmp_path / "protons_T.mat"
    sio.savemat(p, {"machine": {"meta": {"name": "T", "SAD": 10000.0}, "data": data}})
    return sio.loadmat(p)


def peaks(arr): return np.array([float(arr[0, j]["peakPos"].ravel()[0]) for j in range(arr.shape[1])])


@pytest.mark.parametrize("spacing", [2.0, 4.64])
def test_unfiltered_grid_is_exact(tmp_path, spacing):
    arr, info = fm.build(template(tmp_path), spacing)
    assert np.allclose(np.diff(peaks(arr)), spacing, atol=1e-6)
    # the peak written to the machine is where the curve actually peaks (within the 0.1 mm output grid)
    for j in range(0, arr.shape[1], 5):
        z = arr[0, j]["depths"].ravel(); Z = arr[0, j]["Z"].ravel()
        assert abs(z[np.argmax(Z)] - peaks(arr)[j]) <= 0.15


def test_steps4_kernel_moments():
    t, w = fm.filter_kernel("steps4:1.74:1.30")
    assert np.sum(t * w) == pytest.approx(1.74) and np.sqrt(np.sum(w * (t - 1.74) ** 2)) == pytest.approx(1.30)
    assert t[0] == pytest.approx(0.0, abs=0.01)                    # stp4 has an open lowest step


def test_filter_pulls_peak_back_and_widens_it(tmp_path):
    tpl = template(tmp_path)
    a0, i0 = fm.build(tpl, 4.64)
    a1, i1 = fm.build(tpl, 4.64, "steps4:1.74:1.30")
    # same grid phase: the filter machine's folded peaks sit on the unfiltered grid
    common = np.intersect1d(np.round(peaks(a0), 2), np.round(peaks(a1), 2))
    assert len(common) >= len(i1) - 1 and np.allclose(np.diff(peaks(a1)), 4.64, atol=0.05)
    # for the same peak the filter needs more energy: the unfiltered energy of a peak 1.74 mm (mean thickness) deeper
    E0 = np.array([x[0] for x in i0]); P0 = peaks(a0)
    for x, p in zip(i1, peaks(a1)):
        if P0[0] < p + 1.74 < P0[-1]: assert x[0] == pytest.approx(np.interp(p + 1.74, P0, E0), rel=0.01)
    j1 = a1.shape[1] // 2; j0 = int(np.argmin(abs(P0 - peaks(a1)[j1])))
    def width(e):
        z = e["depths"].ravel(); Z = e["Z"].ravel(); zz = np.arange(0, z[-1], 0.01); y = np.interp(zz, z, Z)
        return np.ptp(zz[y >= 0.8 * y.max()])
    assert width(a1[0, j1]) > width(a0[0, j0]) + 1.0
    # LET and lateral tables move with the mean thickness: the filter machine at depth 10 sees the template at 11.74
    j = a1.shape[1] // 2
    e1 = a1[0, j]; src = fm.build(tpl, 4.64, anchor=peaks(a1)[j] + 1.74)[0]
    k = int(np.argmin(abs(peaks(src) - (peaks(a1)[j] + 1.74))))
    assert np.interp(10.0, e1["depths"].ravel(), e1["sigma"].ravel()) == pytest.approx(
        np.interp(11.74, src[0, k]["depths"].ravel(), src[0, k]["sigma"].ravel()), rel=0.03)


@pytest.mark.parametrize("kernel", ["none", "steps4:1.74:1.30", "box:2.58:1.32"])
def test_anchor_gets_a_peak(tmp_path, kernel):
    arr, _ = fm.build(template(tmp_path), 4.64, kernel, anchor=60.0)
    assert np.min(abs(peaks(arr) - 60.0)) < 0.05
    assert np.allclose(np.diff(peaks(arr)), 4.64, atol=0.05)


def test_extra_sigma_in_quadrature(tmp_path):
    arr, info = fm.build(template(tmp_path), 3.0, "none", "4.0:-1.0")
    for j in (0, arr.shape[1] - 1):
        e = arr[0, j]; E = float(e["energy"].ravel()[0]); add = 4.0 * (E / 100.0) ** -1.0
        assert float(e["initFocus"][0, 0]["sigma"].ravel()[0]) == pytest.approx(np.hypot(3.0, add))
        assert float(e["initFocus"][0, 0]["emittance"][0, 0]["sigmaX"].ravel()[0]) == pytest.approx(np.hypot(3.0, add))
