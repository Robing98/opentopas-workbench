"""Tests for topas2matrad.py: the ripple filter's share of the lateral kernel in depth."""
import os, sys

import pytest

np = pytest.importorskip("numpy")
sio = pytest.importorskip("scipy.io")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import topas2matrad as tm  # noqa: E402

DIST = 1080.0          # mm, filter before the isocentre
INC2 = 10.0            # mm2, share of the filter in the spot variance at the isocentre
WET = (0.5, 4.5)       # mm, water equivalent thickness of the ridges


def bragg(z, r=60.0):
    """Rough Bragg curve with its 80 % point near r."""
    from scipy.special import erfc
    return (0.3 + 0.2 * z / r + 0.9 * np.exp(-0.5 * ((z - r + 1.0) / 1.6) ** 2)) * 0.5 * erfc((z - r) / (np.sqrt(2) * 0.9))


# ------------------------------------------------------------------ filter_variance
def test_geometric_share_grows_with_the_lever_arm():
    z = np.array([0.0, 50.0, 100.0])
    add = tm.filter_variance(z, INC2, DIST)
    assert add[0] == 0.0                                   # the isocentre value already sits in the spot size in air
    assert add[1] == pytest.approx(0.947, abs=5e-4)        # 10 * ((1 + 50 / 1080)^2 - 1)
    assert np.allclose(INC2 + add, INC2 * ((DIST + z) / DIST) ** 2)


@pytest.mark.parametrize("dist", [200.0, 1080.0])
def test_flat_dose_gives_the_geometric_share(dist):
    z = np.linspace(0, 80, 81)
    zo = np.linspace(0, 200, 401)
    flat = tm.filter_variance(z, INC2, dist, WET, zo, np.ones(len(zo)))
    assert np.allclose(flat, tm.filter_variance(z, INC2, dist))


def test_peaked_dose_widens_before_the_peak_and_narrows_behind_it():
    z = np.linspace(0, 70, 141)
    zo = np.linspace(0, 100, 1001)
    geo = tm.filter_variance(z, INC2, DIST)
    dw = tm.filter_variance(z, INC2, DIST, WET, zo, bragg(zo))
    assert dw[0] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(dw[z < 30], geo[z < 30], atol=0.15)                 # plateau: all thicknesses weigh the same
    before = (z > 53) & (z < 55.5)                                         # thick sub-beams are in their peak
    assert (dw[before] > geo[before] + 0.5).all()
    behind = (z > 58.5) & (z < 59.5)                                       # they have stopped, thin ones remain
    assert (dw[behind] < geo[behind] - 2.0).all()
    assert dw.min() > -INC2 * (1 + 1e-9)                                   # at most the whole share is taken back


def weight(add, z):
    """Dose weight c(z) / c(0) of the filter share, without the lever arm."""
    return (add / INC2 + 1) / (1 + z / DIST) ** 2


def test_behind_the_range_the_last_weight_is_held():
    z = np.linspace(0, 70, 141)
    zo = np.linspace(0, 100, 1001)
    c = weight(tm.filter_variance(z, INC2, DIST, WET, zo, bragg(zo)), z)
    assert c[z > 62].min() == pytest.approx(c[z > 62].max())               # the thinnest sub-beam ends near 59.5 + 2
    assert c[-1] == pytest.approx(c.min(), rel=0.02)                       # held at the weight of the thin sub-beams


def test_a_dose_tail_behind_the_range_does_not_bring_the_share_back():
    # 5e-4 of the peak behind the range, as from neutrons: there every thickness has the same dose, which would
    # give the geometric share again and so a spot that widens behind the peak
    z = np.linspace(0, 70, 141)
    zo = np.linspace(0, 100, 1001)
    clean = tm.filter_variance(z, INC2, DIST, WET, zo, bragg(zo))
    tail = tm.filter_variance(z, INC2, DIST, WET, zo, bragg(zo) + 5e-4 * bragg(zo).max())
    assert np.allclose(tail[z < 55], clean[z < 55], atol=0.05)
    assert (tail[z > 62] < tm.filter_variance(z, INC2, DIST)[z > 62] - 2.0).all()
    assert weight(tail, z)[z > 62].max() == pytest.approx(weight(tail, z)[z > 62].min())


def test_open_depth_dose_that_ends_inside_the_range_is_refused():
    # grid cut at 57 mm, R80 at 60 mm: behind 52.5 mm the thick sub-beams would count as stopped
    z = np.linspace(0, 56, 113)
    zo = np.linspace(0, 57, 571)
    with pytest.raises(ValueError, match="ends at 57.0 mm"):
        tm.filter_variance(z, INC2, DIST, WET, zo, bragg(zo))
    ok = tm.filter_variance(z[z <= 52.5], INC2, DIST, WET, zo, bragg(zo))  # every sub-beam still on the grid
    full = tm.filter_variance(z[z <= 52.5], INC2, DIST, WET, np.linspace(0, 100, 1001), bragg(np.linspace(0, 100, 1001)))
    assert np.allclose(ok, full)


@pytest.mark.parametrize("end", [71.0, 62.5])
def test_open_depth_dose_may_end_where_no_dose_is_left(end):
    # 62.5 mm: the dose with the filter reaches 59.5 mm, so the thickest sub-beam looks 1.5 mm past the grid, but
    # the open dose is below 1 % there (scoring boxes of the spot runs end this close behind the range)
    z = np.linspace(0, 70, 141)
    zo = np.linspace(0, end, int(end * 10) + 1)
    assert bragg(zo)[-1] < 1e-2 * bragg(zo).max()
    assert np.allclose(tm.filter_variance(z, INC2, DIST, WET, zo, bragg(zo)),
                       tm.filter_variance(z, INC2, DIST, WET, np.linspace(0, 100, 1001), bragg(np.linspace(0, 100, 1001))), atol=0.1)   # the cut tail moves the held value by 0.05 of 6.2 mm2


# ------------------------------------------------------------------ end to end
def write_case(folder, name, energy, r, hist=1000):
    """A parameter file and the IDD csv of its scorer: 0.5 mm bins to 100 mm, 10 x 10 cm2 lateral."""
    z = (np.arange(200) + 0.5) * 0.5
    with open(os.path.join(folder, f"{name}.txt"), "w") as fh:
        fh.write(f's:So/Beam/BeamParticle = "proton"\nd:So/Beam/BeamEnergy = {energy} MeV\nu:So/Beam/BeamEnergySpread = 0.5\n'
                 f'i:So/Beam/NumberOfHistoriesInRun = {hist}\ns:Sc/IDD/OutputFile = "{name}"\n')
    with open(os.path.join(folder, f"{name}.csv"), "w") as fh:
        fh.write("# X in 1 bin of 10 cm\n# Y in 1 bin of 10 cm\n# Z in 200 bins of 0.05 cm\n# DoseToWater ( Gy ) : Sum\n")
        for k, d in enumerate(bragg(z, r) * 1e-9 * hist):
            fh.write(f"0, 0, {k}, {d:.8e}\n")
    return os.path.join(folder, f"{name}.csv")


def write_template(path, ranges=(40.0, 60.0)):
    """A two-energy matRad machine with constant double Gaussian kernels, so that every change comes from the filter."""
    data = np.zeros((1, len(ranges)), dtype=[(k, object) for k in ("energy", "range", "depths", "sigma1", "sigma2", "weight", "LET", "initFocus")])
    for j, r in enumerate(ranges):
        d = np.arange(0.0, r + 20.0, 1.0)
        for k, v in (("energy", 50.0 + r), ("range", r), ("depths", d.reshape(-1, 1)), ("sigma1", np.full((len(d), 1), 2.0)),
                     ("sigma2", np.full((len(d), 1), 6.0)), ("weight", np.full((len(d), 1), 0.05)), ("LET", np.full((len(d), 1), 1.0)),
                     ("initFocus", {"dist": np.array([[0.0, 20000.0]]), "sigma": np.array([[3.0, 3.0]])})):
            data[0, j][k] = v
    meta = {"name": "Template", "dataType": "doubleGauss", "SAD": 10000.0, "BAMStoIsoDist": 2000.0}
    sio.savemat(path, {"machine": {"meta": meta, "data": data}})


@pytest.fixture
def setup(tmp_path):
    write_template(str(tmp_path / "tpl.mat"))
    for sub, r in (("filt", (37.5, 57.5)), ("open", (40.0, 60.0))):      # the filter costs 2.5 mm of range
        os.mkdir(tmp_path / sub)
        for e, rr in zip((70.0, 90.0), r):
            write_case(str(tmp_path / sub), f"IDD_{sub}_p{e:.0f}", e, rr)
    with open(tmp_path / "sigma.csv", "w") as fh:
        fh.write("energy_MeV,sigma_noRiFi_mm,sigma_rs_mm,sigma_RiFi4_mm\n70,4.0,4.5,5.0\n90,3.5,3.75,4.0\n")
    return tmp_path


def run(tmp_path, monkeypatch, *extra, sigma=True):
    out = str(tmp_path / "out.mat")
    argv = ["topas2matrad.py", "--template", str(tmp_path / "tpl.mat"), "--name", "T", "--out", out, str(tmp_path / "filt" / "IDD_*.csv")]
    if sigma: argv += ["--sigma-table", str(tmp_path / "sigma.csv"), "--sigma-column", "sigma_RiFi4_mm"]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    tm.main()
    m = sio.loadmat(out, squeeze_me=True, struct_as_record=False)["machine"]
    return m, list(m.data)


def test_without_filter_distance_the_kernels_are_the_template(setup, monkeypatch):
    m, data = run(setup, monkeypatch)
    for e in data:
        assert e.sigma1[0] == 0.0 and np.allclose(e.sigma1[1:], 2.0) and np.allclose(e.sigma2, 6.0)
    assert data[0].initFocus.SisFWHMAtIso == pytest.approx(2.3548 * np.sqrt((7.0 / 2.3548) ** 2 + 9.0))   # 5^2 - 4^2
    assert "growing in depth" not in m.meta.description


def test_filter_distance_adds_the_lever_arm_to_both_kernels(setup, monkeypatch):
    _, plain = run(setup, monkeypatch)
    m, data = run(setup, monkeypatch, "--filter-distance", "108")
    for e, p, inc2 in zip(data, plain, (9.0, 3.75)):
        add = inc2 * ((1 + e.depths / DIST) ** 2 - 1)
        assert np.allclose(e.depths, p.depths) and np.allclose(e.Z, p.Z)         # depth dose and focus stay
        assert e.initFocus.SisFWHMAtIso == pytest.approx(p.initFocus.SisFWHMAtIso)
        assert np.allclose(e.sigma1 ** 2, p.sigma1 ** 2 + add) and np.allclose(e.sigma2 ** 2, p.sigma2 ** 2 + add)
        assert np.allclose(e.weight, p.weight) and np.allclose(e.LET, p.LET)
        assert e.sigma1[0] == 0.0 and (np.diff(e.sigma1 ** 2 - p.sigma1 ** 2) > 0).all()
    assert "growing in depth from a filter 108 cm" in m.meta.description


def test_sigma_base_takes_the_range_shifter_out_of_the_filter_share(setup, monkeypatch):
    _, plain = run(setup, monkeypatch)
    _, data = run(setup, monkeypatch, "--filter-distance", "108", "--sigma-base", "sigma_rs_mm")
    for e, p, inc2 in zip(data, plain, (25.0 - 20.25, 16.0 - 14.0625)):
        assert np.allclose(e.sigma1 ** 2, p.sigma1 ** 2 + inc2 * ((1 + e.depths / DIST) ** 2 - 1))
        assert e.initFocus.SisFWHMAtIso == pytest.approx(p.initFocus.SisFWHMAtIso)   # in air the whole increase stays


def test_base_distance_lets_the_range_shifter_share_grow_on_its_own_lever(setup, monkeypatch):
    _, plain = run(setup, monkeypatch)
    m, data = run(setup, monkeypatch, "--filter-distance", "108", "--sigma-base", "sigma_rs_mm", "--base-distance", "18.46")
    for e, p, f2, b2 in zip(data, plain, (4.75, 1.9375), (20.25 - 16.0, 14.0625 - 12.25)):
        add = f2 * ((1 + e.depths / DIST) ** 2 - 1) + b2 * ((1 + e.depths / 184.6) ** 2 - 1)
        assert np.allclose(e.sigma1 ** 2, p.sigma1 ** 2 + add) and np.allclose(e.sigma2 ** 2, p.sigma2 ** 2 + add)
    assert "and from sigma_rs_mm at 18.46 cm" in m.meta.description


def test_filter_wet_weights_with_the_open_dose(setup, monkeypatch):
    _, geo = run(setup, monkeypatch, "--filter-distance", "108")
    _, dw = run(setup, monkeypatch, "--filter-distance", "108", "--filter-wet", "0.5", "4.5", "--open-idd", str(setup / "open" / "IDD_*.csv"))
    for e, g in zip(dw, geo):
        z = e.depths; r = float(e.range)
        assert np.allclose(e.sigma1[z < r / 2], g.sigma1[z < r / 2], atol=0.02)
        assert e.sigma1[(z > r - 4) & (z < r - 2)].min() > g.sigma1[(z > r - 4) & (z < r - 2)].max()
        assert e.sigma1[np.argmin(abs(z - (r + 1.5)))] < g.sigma1[np.argmin(abs(z - (r + 1.5)))]
        assert np.isfinite(e.sigma1).all() and (e.sigma1 >= 0).all()


@pytest.mark.parametrize("extra, text, sigma", [
    (["--filter-distance", "108"], "needs --sigma-table and --sigma-column", False),
    (["--filter-distance", "0"], "must be positive", True),
    (["--filter-distance", "108", "--filter-wet", "0.5", "4.5"], "belong together", True),
    (["--filter-wet", "0.5", "4.5", "--open-idd", "x.csv"], "needs --filter-distance", True),
    (["--filter-distance", "108", "--filter-wet", "4.5", "0.5", "--open-idd", "x.csv"], "0 <= MIN < MAX", True),
    (["--sigma-base", "sigma_rs_mm"], "--sigma-base needs --filter-distance", True),
    (["--filter-distance", "108", "--base-distance", "18.46"], "--base-distance needs --sigma-base", True),
    (["--filter-distance", "108", "--sigma-base", "sigma_rs_mm", "--base-distance", "-1"], "--base-distance must be positive", True),
    (["--filter-distance", "108", "--sigma-base", "nope"], "column nope not in", True),
])
def test_option_errors(setup, monkeypatch, extra, text, sigma):
    with pytest.raises(SystemExit) as e:
        run(setup, monkeypatch, *extra, sigma=sigma)
    assert text in str(e.value)


def test_open_idd_must_cover_every_energy(setup, monkeypatch):
    with pytest.raises(SystemExit) as e:
        run(setup, monkeypatch, "--filter-distance", "108", "--filter-wet", "0.5", "4.5", "--open-idd", str(setup / "open" / "IDD_open_p70.csv"))
    assert "no file of --open-idd has 90.0 MeV" in str(e.value)


def test_open_idd_that_ends_inside_the_range_exits_with_the_file(setup, monkeypatch):
    path = setup / "open" / "IDD_open_p90.csv"
    lines = open(path).read().splitlines()
    with open(path, "w") as fh:                                            # 110 bins of 0.5 mm: ends at 55 mm, R80 60 mm
        fh.write("\n".join(l.replace("200 bins", "110 bins") for l in lines[:4] + lines[4:114]) + "\n")
    with pytest.raises(SystemExit) as e:
        run(setup, monkeypatch, "--filter-distance", "108", "--filter-wet", "0.5", "4.5", "--open-idd", str(setup / "open" / "IDD_*.csv"))
    assert "IDD_filt_p90.csv" in str(e.value) and "ends at" in str(e.value)


def test_a_second_open_idd_of_the_same_energy_is_skipped(setup, monkeypatch, capsys):
    one = ["--filter-distance", "108", "--filter-wet", "0.5", "4.5", "--open-idd", str(setup / "open" / "IDD_*.csv")]
    _, first = run(setup, monkeypatch, *one)
    os.mkdir(setup / "open2")
    write_case(str(setup / "open2"), "IDD_zz_p90", 90.0, 75.0)             # same energy, another range: must not win
    _, both = run(setup, monkeypatch, *one, "--open-idd", str(setup / "open2" / "IDD_*.csv"))
    assert "skip duplicate energy 90.0 MeV of --open-idd" in capsys.readouterr().out
    assert np.allclose(both[1].sigma1, first[1].sigma1)


def test_energy_between_the_table_rows_interpolates_the_variance(setup, monkeypatch):
    os.remove(setup / "filt" / "IDD_filt_p70.csv")
    write_case(str(setup / "filt"), "IDD_filt_p80", 80.0, 47.5)
    _, plain = run(setup, monkeypatch)
    _, data = run(setup, monkeypatch, "--filter-distance", "108")
    e, p = data[0], plain[0]
    assert float(e.energy) == 80.0
    assert np.allclose(e.sigma1 ** 2, p.sigma1 ** 2 + 6.375 * ((1 + e.depths / DIST) ** 2 - 1))   # (9 + 3.75) / 2


def test_filter_wet_and_base_distance_add_up(setup, monkeypatch):
    wet = ["--filter-distance", "108", "--sigma-base", "sigma_rs_mm", "--filter-wet", "0.5", "4.5", "--open-idd", str(setup / "open" / "IDD_*.csv")]
    _, a = run(setup, monkeypatch, *wet)
    _, b = run(setup, monkeypatch, *wet, "--base-distance", "18.46")
    for x, y, b2 in zip(a, b, (20.25 - 16.0, 14.0625 - 12.25)):
        grow = b2 * ((1 + x.depths / 184.6) ** 2 - 1)
        free = x.sigma1 ** 2 > 0.5                                         # not clipped at zero behind the range
        assert free.sum() > 100 and np.allclose(y.sigma1[free] ** 2, x.sigma1[free] ** 2 + grow[free])
        assert np.allclose(y.sigma2 ** 2, x.sigma2 ** 2 + grow)


def test_sigma_base_above_the_column_grows_no_more_than_the_spot_has(setup, monkeypatch):
    # noise can put the spot with the range shifter alone above the one with the filter: the increase in air is
    # 5^2 - 4^2 = 9 mm2, so the range shifter's share must not grow from 5.2^2 - 4^2 = 11.04 mm2
    with open(setup / "sigma.csv", "w") as fh:
        fh.write("energy_MeV,sigma_noRiFi_mm,sigma_rs_mm,sigma_RiFi4_mm\n70,4.0,5.2,5.0\n90,3.5,3.75,4.0\n")
    _, plain = run(setup, monkeypatch)
    _, data = run(setup, monkeypatch, "--filter-distance", "108", "--sigma-base", "sigma_rs_mm", "--base-distance", "18.46")
    e, p = data[0], plain[0]
    assert np.allclose(e.sigma1 ** 2, p.sigma1 ** 2 + 9.0 * ((1 + e.depths / 184.6) ** 2 - 1))
