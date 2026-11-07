"""Beam profile monitors and lateral dose profiles: readers, fits, folder scan, generator output and /api/beam/*."""
import array, importlib.util, json, math, os, threading, urllib.parse, urllib.request

import pytest

from opentopas_workbench import beamprofile as bp
from opentopas_workbench import server as wb

GEN_DIR = os.path.join(os.path.dirname(wb.__file__), "generators")


def load_gen(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(GEN_DIR, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def defaults(mod):
    return {f["key"]: f.get("default") for f in mod.FIELDS}


def q(z):
    """Upper tail of the standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def gauss2(x, s1=3.0, s2=10.0, w=0.1, mu=0.0):
    """Area-normalized core plus halo: w is the halo's share of the area."""
    n = 1 / math.sqrt(2 * math.pi)
    return ((1 - w) * n / s1 * math.exp(-0.5 * ((x - mu) / s1) ** 2)
            + w * n / s2 * math.exp(-0.5 * ((x - mu) / s2) ** 2))


def axis(n, width):
    return [(i + 0.5) * width - n * width / 2 for i in range(n)]


# ------------------------------------------------------------------ synthetic TOPAS files
def header(scorer, dims, quantity="Fluence ( /mm2 ) : Sum", binary=None):
    """dims: [(name, n, width, unit)]"""
    h = ["# TOPAS Version: test", f"# Results for scorer: {scorer}"]
    h += [f"# {a} in {n} {'bin ' if n == 1 else 'bins'} of {w:g} {u}" for a, n, w, u in dims]
    h.append(f"# {quantity}   ")
    if binary: h.append(f"# Binary file: {binary}")
    return "\n".join(h) + "\n"


def write_bin(folder, base, dims, values, quantity="Fluence ( /mm2 ) : Sum"):
    (folder / (base + ".binheader")).write_text(header(base, dims, quantity, base + ".bin"))
    array.array("d", values).tofile(open(folder / (base + ".bin"), "wb"))


def write_grid_csv(folder, base, dims, values, quantity="DoseToMedium ( Gy ) : Sum"):
    nx, ny, nz = (d[1] for d in dims)
    rows = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v = values[i + nx * (j + ny * k)]
                if v: rows.append(f"{i}, {j}, {k}, {v!r}")
    (folder / (base + ".csv")).write_text(header(base, dims, quantity) + "\n".join(rows) + "\n")


def map_values(n, width, fx, fy=None, scale=1.0):
    """Separable 2D map, x fastest, from 1D functions of x and y [mm]."""
    fy = fy or fx
    xs = axis(n, width)
    return [scale * fx(x) * fy(y) for y in xs for x in xs]


def write_spectrum(folder, base, n, w, e0, values, prefix=False):
    vals = [0.0] + list(values) + [0.0, 0.0]
    row = ("0, 0, 0, " if prefix else "") + ", ".join(repr(v) for v in vals)
    (folder / (base + ".csv")).write_text(
        f"# TOPAS Version: test\n# Results for scorer: {base}\n# Fluence ( /mm2 ) : Sum   \n"
        f"# Binned by incident track energy in {n} bins of {w:g} MeV from {e0:g} MeV to {e0 + n * w:g} MeV\n"
        "# First bin is underflow, next to last bin is overflow, final bin is for case of no incident track.\n"
        + row + "\n")


# ------------------------------------------------------------------ fits
def test_gauss_core_recovers_sigma_and_center():
    xs = axis(160, 0.5)
    ys = [1000 * math.exp(-0.5 * ((x - 1.3) / 2.5) ** 2) for x in xs]
    a, mu, s = bp.gauss_core(xs, ys)
    assert mu == pytest.approx(1.3, abs=1e-3)
    assert s == pytest.approx(2.5, rel=1e-3)
    assert a == pytest.approx(1000, rel=1e-3)


def test_fwhm_of_a_gaussian():
    xs = axis(400, 0.1)
    ys = [math.exp(-0.5 * (x / 3) ** 2) for x in xs]
    assert bp.fwhm(xs, ys) == pytest.approx(2 * math.sqrt(2 * math.log(2)) * 3, rel=2e-3)


def test_fwhm_of_a_modulated_gaussian():
    """Behind a ripple filter the profile dips below half maximum between grooves near the peak; the width is still
    that of the envelope."""
    xs = axis(400, 0.1)
    ys = [math.exp(-0.5 * (x / 3) ** 2) * (0.75 + 0.25 * math.cos(2 * math.pi * x / 1.5)) for x in xs]
    # the walk from the peak stopped in the first dip at 1.4 mm; the envelope's own FWHM is 7.06 mm
    assert bp.fwhm(xs, ys) == pytest.approx(6.47, abs=0.05)


def test_profile_stats_core_and_halo():
    """sigma1 3 mm, sigma2 10 mm, halo weight 0.1: the double Gaussian finds them, halo3s matches the analytic share."""
    xs = axis(160, 0.5)
    ys = [gauss2(x) for x in xs]
    st = bp.profile_stats(xs, ys)
    assert st["sigma1"] == pytest.approx(3.0, rel=1e-2)
    assert st["sigma2"] == pytest.approx(10.0, rel=1e-2)
    assert st["halo_dg"] == pytest.approx(0.1, abs=2e-3)
    assert st["centroid"] == pytest.approx(0.0, abs=1e-9)
    # the core fit above 10 % sees a little of the halo, so its sigma is a bit wider than 3 mm
    s = st["sigma"]
    assert 3.0 < s < 3.3
    # share outside 3 sigma of the fitted core: analytic for the two Gaussians; the profile ends at +-40 mm
    want = 0.9 * 2 * q(3 * s / 3) + 0.1 * 2 * q(3 * s / 10) - 0.1 * 2 * q(40 / 10)
    assert st["halo3s"] == pytest.approx(want / (1 - 0.1 * 2 * q(4)), abs=2e-3)


def test_profile_stats_without_halo():
    xs = axis(160, 0.5)
    ys = [math.exp(-0.5 * (x / 4) ** 2) for x in xs]
    st = bp.profile_stats(xs, ys)
    assert st["sigma"] == pytest.approx(4, rel=1e-3)
    assert st["halo3s"] == pytest.approx(2 * q(3), abs=2e-3)
    assert st["rms"] == pytest.approx(4, rel=1e-2)


# x profile of the 112 cm monitor plane of the OTPAS-1866 case A_p132p06_t4_d108 (protons, 0.5 mm bins over
# +-40 mm): a Gaussian core without a visible halo, with single counts in the tails
REAL_112CM_X = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.79847, 3.55068, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.46721, 0.0, 0.0,
    0.0, 2.50981, 0.0, 2.62059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.17063, 0.0, 0.0, 0.0, 2.33512, 2.28927,
    2.4161, 0.0, 2.16651, 0.0, 0.0, 0.0, 0.0, 5.19269, 2.26521, 0.0, 0.0, 2.52784, 0.0, 2.23398, 4.76518, 0.0,
    2.0, 0.0, 2.0, 10.0, 14.633, 22.9067, 49.0572, 114.655, 220.0, 320.0, 564.001, 878.002, 1346.0, 2074.62,
    3082.0, 4406.01, 6164.01, 8434.92, 11398.5, 14654.1, 17817.6, 22259.1, 26171.6, 30534.1, 34003.5, 36548.4,
    39334.9, 40113.0, 40247.2, 38746.9, 36999.9, 33535.1, 30390.3, 26180.7, 22458.2, 18004.7, 14246.0, 11172.7,
    8416.77, 6128.04, 4418.01, 3006.0, 2104.3, 1395.71, 882.001, 470.001, 298.0, 178.433, 108.093, 66.6873,
    32.0, 6.00001, 18.0, 2.0, 0.0, 0.0, 4.75247, 2.23402, 0.0, 0.0, 0.0, 4.46706, 0.0, 0.0, 2.32888, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.10243, 0.0, 0.0, 0.0,
    5.74397, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]


def test_no_double_gaussian_on_a_real_plane_without_halo():
    """On this plane the free fit split the one Gaussian into sigma1 1.88 mm and sigma2 2.97 mm with a "halo" of
    99.9 % of the area. Such a split is no halo, so the double Gaussian is left out."""
    xs = axis(160, 0.5)
    st = bp.profile_stats(xs, REAL_112CM_X)
    assert st["sigma"] == pytest.approx(2.97, abs=0.01)
    assert "halo_dg" not in st and "sigma1" not in st
    assert st["halo3s"] < 0.01


def test_profile_stats_of_an_empty_profile():
    assert bp.profile_stats(axis(10, 1), [0.0] * 10) == {}


# ------------------------------------------------------------------ readers
def test_binary_and_csv_grid_agree(tmp_path):
    dims = [("X", 8, 0.1, "cm"), ("Y", 6, 0.1, "cm"), ("Z", 3, 0.2, "cm")]
    vals = [float(i * 7 % 11) for i in range(8 * 6 * 3)]
    write_bin(tmp_path, "dose", dims, vals, "DoseToMedium ( Gy ) : Sum")
    write_grid_csv(tmp_path, "dosec", dims, vals)
    gb, gc = bp.read_grid(str(tmp_path / "dose.bin")), bp.read_grid(str(tmp_path / "dosec.csv"))
    assert list(gb.data) == list(gc.data) == vals
    assert (gb.nx, gb.ny, gb.nz) == (8, 6, 3)
    assert (gb.dx, gb.dy, gb.dz) == (1.0, 1.0, 2.0)          # mm
    assert gb.axis(0)[0] == pytest.approx(-3.5) and gb.axis(1)[-1] == pytest.approx(2.5)
    assert gb.quantity.startswith("DoseToMedium")


def test_csv_with_a_bin_outside_the_header(tmp_path):
    """A row outside the header's bins is a 400 with a reason, not an IndexError that drops the connection."""
    f = tmp_path / "g.csv"
    f.write_text(header("g", [("X", 2, 0.1, "cm"), ("Y", 2, 0.1, "cm"), ("Z", 2, 0.1, "cm")]) + "0, 0, 5, 1.0\n")
    with pytest.raises(ValueError, match="outside"):
        bp.read_grid(str(f))
    f.write_text(header("g", [("X", 2, 0.1, "cm"), ("Y", 2, 0.1, "cm"), ("Z", 2, 0.1, "cm")]) + "-1, 0, 0, 1.0\n")
    with pytest.raises(ValueError, match="outside"):
        bp.read_grid(str(f))


def test_profile_stats_survive_a_diverging_core_fit(monkeypatch):
    def boom(*a, **k): raise OverflowError("math range error")
    monkeypatch.setattr(bp, "gauss_core", boom)
    st = bp.profile_stats(axis(10, 1), [0, 0, 1, 5, 9, 5, 1, 0, 0, 0])
    assert "sigma" not in st and st["rms"] > 0


def test_binary_without_header_or_with_several_statistics(tmp_path):
    dims = [("X", 2, 1, "mm"), ("Y", 2, 1, "mm"), ("Z", 1, 1, "mm")]
    write_bin(tmp_path, "m", dims, [1.0] * 8)               # two statistics' worth of doubles
    with pytest.raises(ValueError, match="statistic"):
        bp.read_grid(str(tmp_path / "m.bin"))
    array.array("d", [1.0] * 4).tofile(open(tmp_path / "lone.bin", "wb"))
    with pytest.raises(ValueError, match="binheader"):
        bp.read_grid(str(tmp_path / "lone.bin"))


def test_monitor_profiles_integrated_and_cut(tmp_path):
    n, w = 80, 0.5
    write_bin(tmp_path, "mon_5_c_p", [("X", n, w, "mm"), ("Y", n, w, "mm"), ("Z", 1, 0.1, "mm")],
              map_values(n, w, lambda x: math.exp(-0.5 * ((x - 2) / 3) ** 2), lambda y: math.exp(-0.5 * (y / 5) ** 2)))
    for mode in ("int", "cut"):
        m = bp.monitor(str(tmp_path / "mon_5_c_p.bin"), mode)
        assert len(m["px"]) == n and len(m["map"]) == n and len(m["map"][0]) == n
        assert m["fx"]["sigma"] == pytest.approx(3, rel=1e-3) and m["fx"]["mu"] == pytest.approx(2, abs=1e-3)
        assert m["fy"]["sigma"] == pytest.approx(5, rel=1e-3)
    # integrated: value times the bin width of the other axis; total = sum of the map times both bin widths
    m = bp.monitor(str(tmp_path / "mon_5_c_p.bin"), "int")
    assert m["fx"]["total"] == pytest.approx(sum(sum(r) for r in m["map"]) * w * w / w, rel=1e-3)


def test_monitor_rejects_a_3d_grid(tmp_path):
    write_bin(tmp_path, "g", [("X", 2, 1, "mm"), ("Y", 2, 1, "mm"), ("Z", 2, 1, "mm")], [1.0] * 8)
    with pytest.raises(ValueError, match="one Z bin"):
        bp.monitor(str(tmp_path / "g.bin"))


@pytest.mark.parametrize("prefix", [False, True])
def test_spectrum_with_and_without_bin_indices(tmp_path, prefix):
    n, w = 200, 0.05
    vals = [math.exp(-0.5 * (((i + 0.5) * w - 5) / 0.5) ** 2) for i in range(n)]
    write_spectrum(tmp_path, "mon_5s_c_spec", n, w, 0, vals, prefix)
    s = bp.read_spectrum(str(tmp_path / "mon_5s_c_spec.csv"))
    assert len(s["e"]) == n and s["e"][0] == pytest.approx(0.025)
    assert s["peak"] == pytest.approx(5, abs=w) and s["mean"] == pytest.approx(5, abs=0.01)
    assert s["fwhm"] == pytest.approx(2.3548 * 0.5, rel=0.02)
    assert s["quantity"].startswith("Fluence")


def test_spectrum_in_kev_and_wrong_length(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("# Fluence ( /mm2 ) : Sum\n# Binned by incident track energy in 2 bins of 500 keV from 0 keV to 1000 keV\n"
                 "0, 1, 2, 0, 0\n")
    s = bp.read_spectrum(str(p))
    assert s["e"] == pytest.approx([0.25, 0.75]) and s["width"] == pytest.approx(0.5)
    p.write_text("# Binned by incident track energy in 3 bins of 1 MeV from 0 MeV to 3 MeV\n0, 1, 2\n")
    with pytest.raises(ValueError, match="expected 6"):
        bp.read_spectrum(str(p))


# ------------------------------------------------------------------ folder scan and envelope
def mon_dims(n=20, w=1.0):
    return [("X", n, w, "mm"), ("Y", n, w, "mm"), ("Z", 1, 0.1, "mm")]


def gmap(sigma, n=20, w=1.0):
    return map_values(n, w, lambda x: math.exp(-0.5 * (x / sigma) ** 2))


def test_scan_one_case_one_plane(tmp_path):
    write_bin(tmp_path, "mon_0_solo_all", mon_dims(), gmap(2))
    res = bp.scan(str(tmp_path), "f")
    assert [c["name"] for c in res["cases"]] == ["solo"]
    (c,) = res["cases"]
    assert c["kinds"] == ["all"] and c["dose"] == []
    assert c["monitors"] == [{"pos": 0.0, "maps": {"all": "f/mon_0_solo_all.bin"}, "spectra": {}}]


def test_scan_names_cases_planes_and_dose(tmp_path):
    cases = ["A_p62p99_t4_d20", "A_p62p99_t4_d20_rs", "rifi0", "rifi0_rs"]
    for case in cases:
        for pos in ("112", "19p52", "0"):
            for k in ("p", "e", "o"):
                write_bin(tmp_path, f"mon_{pos}_{case}_{k}", mon_dims(), gmap(2 + float(pos.replace("p", ".")) / 50))
            write_spectrum(tmp_path, f"mon_{pos}s_{case}_spec", 4, 1, 0, [1, 2, 3, 4])
    dd = [("X", 4, 1, "mm"), ("Y", 4, 1, "mm"), ("Z", 3, 1, "mm")]
    for d in ("A_p62p99_t4_d20", "x_rifi0", "x_rifi0_rs"):
        write_bin(tmp_path, d, dd, [1.0] * 48, "DoseToMedium ( Gy ) : Sum")
    write_bin(tmp_path, "flat", [("X", 1, 1, "mm"), ("Y", 1, 1, "mm"), ("Z", 3, 1, "mm")], [1.0] * 3)   # a depth curve
    (tmp_path / "mon_3_bad_spec.csv").write_text("x")       # a spectrum name without the "s": ignored
    res = bp.scan(str(tmp_path), "")
    by = {c["name"]: c for c in res["cases"]}
    assert sorted(by) == sorted(cases)
    c = by["A_p62p99_t4_d20_rs"]
    assert [m["pos"] for m in c["monitors"]] == [112.0, 19.52, 0.0]           # upstream first
    assert c["kinds"] == ["p", "e", "o"]
    assert c["monitors"][1]["spectra"] == {"spec": "mon_19p52s_A_p62p99_t4_d20_rs_spec.csv"}
    assert by["A_p62p99_t4_d20"]["dose"] == ["A_p62p99_t4_d20.bin"]
    assert c["dose"] == []
    assert by["rifi0"]["dose"] == ["x_rifi0.bin"] and by["rifi0_rs"]["dose"] == ["x_rifi0_rs.bin"]
    assert sorted(d["name"] for d in res["dose"]) == ["A_p62p99_t4_d20", "x_rifi0", "x_rifi0_rs"]
    env = bp.envelope(str(tmp_path), "", "rifi0", "p")
    assert [r["pos"] for r in env["rows"]] == [112, 19.52, 0]
    assert [r["fx"]["sigma"] for r in env["rows"]] == pytest.approx([2 + 112 / 50, 2 + 19.52 / 50, 2], rel=1e-3)
    with pytest.raises(ValueError, match="no monitors"):
        bp.envelope(str(tmp_path), "", "nope")


def test_dose_depth_and_sigma_vs_depth(tmp_path):
    n, w, nz = 40, 0.5, 4
    sig = [2.0, 2.5, 3.0, 0.0]
    vals = []
    for k in range(nz):
        vals += map_values(n, w, lambda x, s=sig[k]: math.exp(-0.5 * (x / s) ** 2) if s else 0.0)
    write_bin(tmp_path, "dose", [("X", n, w, "mm"), ("Y", n, w, "mm"), ("Z", nz, 0.2, "cm")], vals,
              "DoseToMedium ( Gy ) : Sum")
    d = bp.dose_depth(str(tmp_path / "dose.bin"), 1)
    assert d["k"] == 1 and d["depth"] == pytest.approx(3.0)
    assert d["fx"]["sigma"] == pytest.approx(2.5, rel=1e-3)
    assert [r["k"] for r in d["sigma_z"]] == [0, 1, 2]                      # the empty last plane is left out
    assert [r["fy"]["sigma"] for r in d["sigma_z"]] == pytest.approx(sig[:3], rel=1e-3)
    assert bp.dose_depth(str(tmp_path / "dose.bin"), 99)["k"] == nz - 1     # clamped


# ------------------------------------------------------------------ generator output
def monitor_scorers(text):
    return [o for o in wb.scorer_outputs(text) if o.startswith("mon_")]


def test_no_monitors_by_default():
    for name in ("pencil_beam", "ripple_filter"):
        mod = load_gen(name)
        for _, t in mod.generate(defaults(mod)):
            if isinstance(t, str): assert "Monitors" not in t and not monitor_scorers(t)


def test_ripple_filter_auto_monitors():
    mod = load_gen("ripple_filter")
    v = dict(defaults(mod), mon_auto=True)
    files = [(n, t) for n, t in mod.generate(v) if isinstance(t, str)]
    assert len(files) >= 3                                                    # several variants, same planes in each
    for n, t in files:
        case = n[:-4]
        outs = monitor_scorers(t)
        for pos in ("84p5", "81p5", "77", "0"):
            for k in ("p", "e", "o"): assert f"mon_{pos}_{case}_{k}" in outs
            assert f"mon_{pos}s_{case}_spec" in outs and f"mon_{pos}s_{case}_espec" in outs
        assert len(outs) == 4 * 5
        assert 'b:Ge/Monitors/IsParallel = "True"' in t
        assert t.count('sv:Sc/Mon0O/OnlyIncludeParticlesNotNamed = 2 "proton" "e-"') == 1
        assert [i for i in wb.lint(t) if i[1] == "error"] == [], n
        # every output name the generator writes is one the view reads back as this case
        for o in outs:
            m = bp.MON_RE.match(o)
            assert m and m.group(3) == case


def test_pencil_beam_monitor_positions_and_options():
    mod = load_gen("pencil_beam")
    v = dict(defaults(mod), mon_pos="9.52, 5", mon_split=False, mon_spec=False, mon_bin=1, mon_half=10)
    files = [(n, t) for n, t in mod.generate(v) if isinstance(t, str)]
    for n, t in files:
        case = n[:-4]
        assert monitor_scorers(t) == [f"mon_9p52_{case}_all", f"mon_5_{case}_all"]
        assert "OnlyIncludeParticles" not in t
        assert "i:Ge/Mon9p52/XBins      = 20" in t and "d:Ge/Mon9p52/HLX        = 10 mm" in t
        assert "d:Ge/Mon5/TransZ     = 50.05 mm" in t
    with pytest.raises(ValueError, match="outside the world"):
        mod.generate(dict(v, mon_pos="19.52"))                               # 10 cm source distance by default


def test_monitor_positions_rejected():
    mon = load_gen("_monitors")
    v = {"mon_pos": "200"}
    with pytest.raises(ValueError, match="outside the world"):
        mon.block(v, "c", [], 100, 300, 1400)
    with pytest.raises(ValueError, match="0 or more"):
        mon.block({"mon_pos": "-1"}, "c", [], 100, 300, 1400)
    with pytest.raises(ValueError, match="closer than 1 mm"):
        mon.block({"mon_pos": "5, 5.05"}, "c", [], 100, 300, 1400)
    assert mon.block({"mon_pos": ""}, "c", [1, 2], 100, 300, 1400) == ""    # auto positions only when mon_auto is on
    assert [mon.tag(p) for p in (19.52, 5.0, 0.5, 0)] == ["19p52", "5", "0p5", "0"]


# ------------------------------------------------------------------ HTTP
@pytest.fixture
def srv(tmp_path):
    wb.STATE["root"] = str(tmp_path)
    wb.STATE["cfg"] = {"binaries": [], "lang": "en"}
    wb.STATE["jobs"] = []
    s = wb.Server(("127.0.0.1", 0), wb.H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


def get(url, **params):
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params))
    req.add_header("X-Token", wb.TOKEN)
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_beam_api(srv, tmp_path):
    f = tmp_path / "beam"
    f.mkdir()
    for pos, s in (("10", 3.0), ("5", 3.5), ("0", 4.0)):
        write_bin(f, f"mon_{pos}_c_p", mon_dims(40, 0.5), gmap(s, 40, 0.5))
    write_spectrum(f, "mon_0s_c_spec", 4, 1, 0, [1, 2, 3, 4])
    write_bin(f, "c", [("X", 40, 0.5, "mm"), ("Y", 40, 0.5, "mm"), ("Z", 2, 1, "mm")], gmap(2, 40, 0.5) * 2,
              "DoseToMedium ( Gy ) : Sum")
    st, res = get(srv + "/api/beam/scan", p="beam")
    assert st == 200 and [c["name"] for c in res["cases"]] == ["c"] and res["cases"][0]["dose"] == ["beam/c.bin"]
    st, res = get(srv + "/api/beam/monitor", p="beam/mon_5_c_p.bin", mode="cut")
    assert st == 200 and res["fx"]["sigma"] == pytest.approx(3.5, rel=1e-3)
    st, res = get(srv + "/api/beam/spectrum", p="beam/mon_0s_c_spec.csv")
    assert st == 200 and res["peak"] == 3.5
    st, res = get(srv + "/api/beam/envelope", p="beam", case="c", kind="p")
    assert st == 200 and [r["pos"] for r in res["rows"]] == [10, 5, 0]
    st, res = get(srv + "/api/beam/dose", p="beam/c.bin", k=1)
    assert st == 200 and res["k"] == 1 and len(res["sigma_z"]) == 2
    assert get(srv + "/api/beam/scan", p="beam/c.bin")[0] >= 400
    assert get(srv + "/api/beam/nope", p="beam")[0] >= 400
    assert get(srv + "/api/beam/scan", p="../..")[0] >= 400
