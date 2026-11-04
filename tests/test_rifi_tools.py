"""Tests for the ripple filter tools: kernel geometry, STL thickness, 1D equivalents, and the kernel fit."""
import importlib.util, os, struct, sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rifi_kernel_fit as kf  # noqa: E402

spec = importlib.util.spec_from_file_location("ripple_filter", os.path.join(ROOT, "opentopas_workbench", "generators", "ripple_filter.py"))
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)


def bragg(z, r=60.0, s=0.9):
    """Rough Bragg curve: rising plateau plus a peak that falls off as an error function (enough for the fit tests)."""
    from scipy.special import erfc
    plateau = 0.3 + 0.2 * z / r
    peak = np.exp(-0.5 * ((z - r + 1.0) / 1.6) ** 2) * 0.9 + 0.35 * np.exp(-(r - z).clip(0) / 6.0)
    return (plateau + peak) * 0.5 * erfc((z - r) / (np.sqrt(2) * s))


def convolved(z, d, samples):
    t, w, m, s = kf.hist_kernel(samples)
    return sum(wi * kf.shifted(z, z, d, ti) for ti, wi in zip(t, w)), m, s


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("kind,mean,sd", [("pyramid", 1 / 3, 1 / np.sqrt(18)), ("uniform", 1 / 2, 1 / np.sqrt(12))])
def test_pin_kernel_moments(kind, mean, sd):
    s = kf.profile_samples(kind, 4.0, base=0.2)
    assert s.mean() == pytest.approx(0.2 + 4.0 * mean, abs=1e-3)
    assert s.std() == pytest.approx(4.0 * sd, abs=1e-3)


def test_one_d_triangle_is_uniform():
    s = kf.ridge_samples([0, 0.75, 1.5], [0.0, 3.0, 0.0], 1.5)
    t, w, m, sd = kf.hist_kernel(s, step=0.1)
    assert m == pytest.approx(1.5, abs=1e-3) and sd == pytest.approx(3 / np.sqrt(12), abs=1e-3)
    assert w[1:-1].std() / w[1:-1].mean() < 0.02            # flat


def test_crossed_filters_convolve():
    a = kf.hist_kernel(kf.ridge_samples([0, 0.5, 1.0], [0.0, 2.0, 0.0], 1.0))
    c = kf.convolve_kernels(a, a)
    assert c[2] == pytest.approx(2 * a[2], abs=1e-3)
    assert c[3] == pytest.approx(np.sqrt(2) * a[3], abs=2e-3)


def mesh_ok(stl):
    n = struct.unpack("<I", stl[80:84])[0]
    key = lambda p: tuple(round(c, 4) for c in p)
    edges = {}
    vol = 0.0
    for k in range(n):
        v = struct.unpack("<9f", stl[84 + 50 * k + 12: 84 + 50 * k + 48])
        a, b, c = np.array(v[0:3]), np.array(v[3:6]), np.array(v[6:9])
        vol += np.dot(a, np.cross(b, c)) / 6
        tri = [key(a), key(b), key(c)]
        for i, j in ((0, 1), (1, 2), (2, 0)): edges[(tri[i], tri[j])] = edges.get((tri[i], tri[j]), 0) + 1
    closed = all(cnt == 1 and edges.get((b, a)) == 1 for (a, b), cnt in edges.items())
    return closed, vol


@pytest.mark.parametrize("profile", ["uniform", "pyramid"])
def test_pin_stl_closed_and_thickness_matches_rings(tmp_path, profile):
    stl = rf.pins_stl(4.0, half=4.5, profile=profile)
    closed, vol = mesh_ok(stl)
    assert closed and vol > 0                                  # watertight, normals outwards
    p = tmp_path / "pins.stl"; p.write_bytes(stl)
    T = kf.stl_thickness(kf.read_stl(str(p)), region=6.0)
    zr, sr = rf.pin_rings(4.0, profile=profile)
    zz = np.linspace(zr[0], zr[-1], 100001); a = (np.interp(zz, zr, sr) / 0.75) ** 2
    mean = zr[0] + np.trapezoid(a, zz); sd = np.sqrt(zr[0] ** 2 + np.trapezoid(2 * zz * a, zz) - mean ** 2)
    assert T.mean() == pytest.approx(mean, abs=2e-3) and T.std() == pytest.approx(sd, abs=2e-3)
    assert vol / (9.0 * 9.0) == pytest.approx(mean, abs=2e-3)  # volume per area = mean thickness


@pytest.mark.parametrize("profile", ["uniform", "pyramid"])
def test_equivalent_ridge_has_same_distribution(profile):
    xs, zs = rf.equivalent_ridge(4.0, profile=profile)
    assert all(b > a for a, b in zip(xs, xs[1:]))
    r = kf.ridge_samples(np.round(xs, 3), np.round(zs, 3), 1.5)  # as written into the parameter file
    zr, sr = rf.pin_rings(4.0, profile=profile)
    zz = np.linspace(zr[0], zr[-1], 100001); a = (np.interp(zz, zr, sr) / 0.75) ** 2
    mean = zr[0] + np.trapezoid(a, zz)
    assert r.mean() == pytest.approx(mean, abs=2e-3)


def test_generator_writes_pyramids_and_equivalents():
    v = {f["key"]: f.get("default") for f in rf.FIELDS}
    v.update(energies="100", variants=["pyr4", "pyr4e", "pin4e"])
    files = dict(rf.generate(v))
    assert set(files) == {"mess_p100_pyr4.txt", "mess_p100_pyr4e.txt", "mess_p100_pin4e.txt", "pyr_4mm.stl"}
    assert '"pyr_4mm"' in files["mess_p100_pyr4.txt"] and "TsRidgeFilter" in files["mess_p100_pyr4e.txt"]


def test_true_kernel_from_parameter_files(tmp_path):
    v = {f["key"]: f.get("default") for f in rf.FIELDS}
    v.update(energies="100", variants=["pyr4", "pyr4e", "rifi2"])
    for name, content in rf.generate(v):
        (tmp_path / name).write_bytes(content) if isinstance(content, bytes) else (tmp_path / name).write_text(content)
    args = type("A", (), {"pin": None, "stl": None, "heightmap": None, "ridge": None, "spr": 1.16, "stl_units": 1.0})()
    k2d = kf.true_kernel(kf.topas_info(str(tmp_path / "IDD_mess_p100_pyr4.csv")), args)
    k1d = kf.true_kernel(kf.topas_info(str(tmp_path / "IDD_mess_p100_pyr4e.csv")), args)
    assert k2d[2] == pytest.approx(k1d[2], abs=3e-3) and k2d[3] == pytest.approx(k1d[3], abs=3e-3)
    kx = kf.true_kernel(kf.topas_info(str(tmp_path / "IDD_mess_p100_rifi2.csv")), args)
    single = kf.hist_kernel(kf.ridge_samples([0.0, 0.009, 0.46, 0.54, 0.991, 1.0], [0.4, 0.4, 2.0, 2.0, 0.4, 0.4], 1.0) * 1.16)
    assert kx[2] == pytest.approx(2 * single[2], abs=3e-3) and "⊗" in kx[4]


# ------------------------------------------------------------------ fit
@pytest.mark.parametrize("kind,shape", [("pyramid", "pyramid"), ("uniform", "box")])
def test_fit_recovers_kernel_and_shape(kind, shape):
    z = np.arange(0.05, 90, 0.1); d0 = bragg(z)
    d1, m, s = convolved(z, d0, kf.profile_samples(kind, 3.9, 0.1) * 1.16)
    d0n, d1n = d0 / d0.max(), d1 / d1.max()
    res = {sh: kf.fit_param(z, d0n, z, d1n, sh, [1.0, m - 0.2, s * 1.1]) for sh in kf.SHAPES}
    best = min(res, key=lambda k: res[k][2])
    assert best == shape
    p = res[best][0]
    assert p[1] == pytest.approx(m, abs=0.01) and p[2] == pytest.approx(s, abs=0.01)
    t, k, A, tm, sg, rms = kf.fit_free(z, d0n, z, d1n, tmax=6.5)
    assert tm == pytest.approx(m, abs=0.03) and sg == pytest.approx(s, abs=0.03)


def test_quadrature_shortcut_overestimates():
    z = np.arange(0.05, 90, 0.1); d0 = bragg(z)
    d1, m, s = convolved(z, d0, kf.profile_samples("uniform", 3.7, 0.3, dim=1) * 1.16)
    q = np.sqrt(kf.metrics(z, d1)["BPW80"] ** 2 - kf.metrics(z, d0)["BPW80"] ** 2)
    assert q > 1.3 * s


def test_readers(tmp_path):
    (tmp_path / "a.mcc").write_text("BEGIN_SCAN\nBEGIN_DATA\n\t0.0\t1.0\n\t1.5\t2.0\nEND_DATA\nEND_SCAN\n")
    z, d, _ = kf.read_curve(str(tmp_path / "a.mcc")); assert list(z) == [0.0, 1.5] and list(d) == [1.0, 2.0]
    (tmp_path / "b.txt").write_text("# depth; dose\n0,5; 10,0\n1,5; 12,5\n")
    z, d, _ = kf.read_curve(str(tmp_path / "b.txt")); assert list(z) == [0.5, 1.5] and list(d) == [10.0, 12.5]
    assert kf.energy_from_name("100MeV_ohne.mcc") == 100 and kf.energy_from_name("IDD_mess_a058_pin3.csv") == 58


def test_discover_pairs_across_folders(tmp_path):
    (tmp_path / "IDD_x_p063_rifi0.csv").write_text("")
    sub = tmp_path / "rifi2d"; sub.mkdir()
    for v in ("pyr4", "pin4e"): (sub / f"IDD_x_p063_{v}.csv").write_text("")
    pairs = kf.discover(str(sub), None, "rifi0", None)
    assert sorted(p[2] for p in pairs) == ["pin4e", "pyr4"] and all(p[0].endswith("IDD_x_p063_rifi0.csv") for p in pairs)
