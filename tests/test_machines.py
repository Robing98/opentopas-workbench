"""Tests for the Machines view: reading matRad machine files (v7 and v7.3), the folder list, one machine, and comparisons.

Synthetic machines are written by the tests in both formats. The public matRad machines (protons_Generic.mat,
carbon_Generic.mat) are tested when MATRAD_BASEDATA names a folder that holds them; CI downloads them from matRad.
"""
import importlib.util, json, os, sys, threading, urllib.parse, urllib.request, urllib.error

import pytest

np = pytest.importorskip("numpy")
sio = pytest.importorskip("scipy.io")

from opentopas_workbench import machines as M, server as wb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENERGIES = [70.0, 100.0, 150.0]


def peak_of(e):
    """Bragg peak depth in mm of the synthetic machine: rises with the energy, as a range-energy relation does."""
    return 0.022 * e ** 1.77


def synthetic(shift=0.0, energies=ENERGIES):
    """A doubleGauss proton machine: plateau plus a Gaussian peak at peak_of(E) + shift, cut off behind it."""
    data = []
    for e in energies:
        p = peak_of(e) + shift
        z = np.arange(0.0, p + 30.0, 0.25)
        y = (0.3 + 0.2 * z / p + np.exp(-0.5 * ((z - p) / 2.0) ** 2)) * (z < p + 6) + np.exp(-0.5 * ((z - p) / 2.0) ** 2) * (z >= p + 6)
        data.append({"energy": e, "range": p + 1.0, "peakPos": p, "offset": 0.0, "depths": z, "Z": y,
                     "sigma1": 1.0 + 0.01 * z, "sigma2": 5.0 + 0.02 * z, "weight": 0.1 + 0 * z,
                     "initFocus": {"dist": np.array([0.0, 20000.0]), "sigma": np.array([6.0, 2.0]), "SisFWHMAtIso": 4.0}})
    meta = {"name": "Synth", "radiationMode": "protons", "dataType": "doubleGauss", "SAD": 10000.0}
    return meta, data


def write_v7(path, meta, data):
    keys = list(data[0])
    arr = np.zeros((1, len(data)), dtype=[(k, "O") for k in keys])
    for i, d in enumerate(data):
        for k in keys: arr[0, i][k] = d[k]
    sio.savemat(str(path), {"machine": {"meta": meta, "data": arr}}, do_compression=True)


def write_v73(path, meta, data):
    """The layout MATLAB writes with -v7.3: an HDF5 file behind a 512 byte header, struct arrays as reference fields."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "w", userblock_size=512) as f:
        refs = f.create_group("#refs#")
        count = [0]

        def put(group, name, v):
            if isinstance(v, dict):
                g = group.create_group(name); g.attrs["MATLAB_class"] = np.bytes_("struct")
                for k, x in v.items(): put(g, k, x)
                return g
            if isinstance(v, str):
                ds = group.create_dataset(name, data=np.array([[ord(c)] for c in v], dtype=np.uint16).T)
                ds.attrs["MATLAB_class"] = np.bytes_("char")
                return ds
            a = np.atleast_2d(np.asarray(v, float))
            ds = group.create_dataset(name, data=a); ds.attrs["MATLAB_class"] = np.bytes_("double")
            return ds

        m = f.create_group("machine"); m.attrs["MATLAB_class"] = np.bytes_("struct")
        put(m, "meta", meta)
        dg = m.create_group("data"); dg.attrs["MATLAB_class"] = np.bytes_("struct")
        for k in data[0]:
            rs = []
            for d in data:
                count[0] += 1
                rs.append(put(refs, f"r{count[0]}", d[k]).ref)
            dg.create_dataset(k, data=np.array([rs], dtype=h5py.ref_dtype).T)
    head = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: test HDF5 schema 1.00 ."
    with open(path, "r+b") as fh:
        fh.write(head.ljust(116, b" ") + b"\0" * 8 + b"\x00\x02" + b"IM")


@pytest.fixture
def folder(tmp_path):
    meta, data = synthetic()
    write_v7(tmp_path / "protons_SynthA.mat", meta, data)
    try:
        import h5py  # noqa: F401
        write_v73(tmp_path / "protons_SynthA73.mat", meta, data)
    except ImportError:
        pass
    m2, d2 = synthetic(shift=2.0, energies=[71.0, 99.0, 150.0])
    write_v7(tmp_path / "protons_SynthB.mat", dict(m2, name="SynthB"), d2)
    (tmp_path / "broken.mat").write_bytes(b"not a mat file at all")
    (tmp_path / "notes.txt").write_text("ignored")
    return tmp_path


# ------------------------------------------------------------------ reading and the list
def test_list_reads_both_formats_and_reports_broken_files(folder):
    rows = {r["file"]: r for r in M.list_dir(str(folder))}
    assert "notes.txt" not in rows and "error" in rows["broken.mat"]
    a = rows["protons_SynthA.mat"]
    assert (a["format"], a["radiation"], a["data_type"], a["n"], a["e_min"], a["e_max"]) == ("v7", "protons", "doubleGauss", 3, 70.0, 150.0)
    assert a["name"] == "Synth" and a["depth_n_min"] < a["depth_n_max"]
    if "protons_SynthA73.mat" in rows:
        b = rows["protons_SynthA73.mat"]
        assert b["format"] == "v7.3" and {k: v for k, v in b.items() if k not in ("file", "format", "size")} == \
            {k: v for k, v in a.items() if k not in ("file", "format", "size")}


def test_view_curves_fields_and_scalars(folder):
    v = M.view(str(folder / "protons_SynthA.mat"))
    assert v["energies"] == ENERGIES
    c = v["curves"][1]
    assert set(c["fields"]) == {"sigma1", "sigma2", "weight"} and len(c["fields"]["sigma1"]) == len(c["depths"]) == len(c["Z"])
    assert list(v["scalars"])[:4] == ["range", "peakPos", "offset", "initFocus.sigma@iso"]
    assert v["scalars"]["initFocus.sigma@iso"] == [4.0] * 3            # 6 mm at the source, 2 mm at 20 m: 4 mm at SAD 10 m
    for e, m in zip(ENERGIES, v["metrics"]):
        assert m["peak"] == pytest.approx(peak_of(e), abs=0.26) and m["r80"] > m["peak"]


def test_v73_view_matches_v7(folder):
    pytest.importorskip("h5py")
    a, b = M.view(str(folder / "protons_SynthA.mat")), M.view(str(folder / "protons_SynthA73.mat"))
    assert a["curves"] == b["curves"] and a["scalars"] == b["scalars"] and a["metrics"] == b["metrics"]


def test_tree_shows_every_field(folder):
    tr = M.tree(str(folder / "protons_SynthA.mat"), 2)
    assert tr["index"] == 2 and tr["energy"] == 150.0
    meta = {n["name"]: n for n in tr["meta"]["children"]}
    assert meta["radiationMode"]["value"] == "protons" and meta["SAD"]["value"] == 10000.0
    data = {n["name"]: n for n in tr["data"]["children"]}
    assert data["depths"]["kind"] == "array" and data["depths"]["min"] == 0.0
    assert {n["name"] for n in data["initFocus"]["children"]} == {"dist", "sigma", "SisFWHMAtIso"}


# ------------------------------------------------------------------ comparison
def test_compare_with_itself_is_zero(folder):
    p = str(folder / "protons_SynthA.mat")
    r = M.compare(p, p, [100.0])
    assert all(w["d_r80"] == 0 and w["d_peak"] == 0 for w in r["rows"])
    assert len(r["curves"]) == 1 and max(abs(x) for x in r["curves"][0]["diff"] if x is not None) == 0


def test_compare_takes_nearest_energy_and_finds_the_shift(folder):
    r = M.compare(str(folder / "protons_SynthA.mat"), str(folder / "protons_SynthB.mat"), [150.0])
    assert [(w["energy_a"], w["energy_b"]) for w in r["rows"]] == [(70.0, 71.0), (100.0, 99.0), (150.0, 150.0)]
    same = r["rows"][2]
    assert same["d_r80"] == pytest.approx(2.0, abs=0.05) and same["d_peak"] == pytest.approx(2.0, abs=0.3)
    c = r["curves"][0]
    assert c["energy_b"] == 150.0 and len(c["diff"]) == len(c["za"]) and max(abs(x) for x in c["diff"] if x is not None) > 10


def test_metrics_shared_with_the_compare_tool():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    spec = importlib.util.spec_from_file_location("compare_idd_matrad", os.path.join(ROOT, "tools", "compare_idd_matrad.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert mod.metrics is M.metrics and mod.load_machine is M.idd_entries
    import matload
    assert matload.load_struct is M.load_struct


def test_local_path_turns_windows_paths_into_mounts(monkeypatch):
    if os.sep != "/": pytest.skip("POSIX only")
    assert M.local_path(r"C:\Users\x\matRad\machines") == "/mnt/c/Users/x/matRad/machines"
    monkeypatch.setitem(M._WINHOME, "h", "/mnt/c/Users/x")
    assert M.local_path(r"%USERPROFILE%\matRad\machines") == "/mnt/c/Users/x/matRad/machines"
    assert M.default_dir() == "/mnt/c/Users/x/matRad/machines"


# ------------------------------------------------------------------ HTTP
@pytest.fixture
def srv(tmp_path_factory):
    wb.STATE["root"] = str(tmp_path_factory.mktemp("ws"))
    wb.STATE["cfg"] = {"binaries": [], "lang": "en"}
    wb.STATE["jobs"] = []
    s = wb.Server(("127.0.0.1", 0), wb.H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


def call(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("X-Token", wb.TOKEN); req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_list_view_compare(srv, folder):
    q = urllib.parse.quote
    assert call(srv + "/api/config", {"machines_dir": str(folder)})[0] == 200
    code, r = call(srv + "/api/machines/list")
    assert code == 200 and r["dir"] == os.path.realpath(str(folder)) and "protons_SynthB.mat" in [f["file"] for f in r["files"]]
    a, b = q(str(folder / "protons_SynthA.mat")), q(str(folder / "protons_SynthB.mat"))
    code, v = call(srv + f"/api/machines/view?p={a}")
    assert code == 200 and v["summary"]["n"] == 3
    code, t = call(srv + f"/api/machines/tree?p={a}&i=1")
    assert code == 200 and t["energy"] == 100.0
    code, c = call(srv + f"/api/machines/compare?a={a}&b={b}&e=150&norm=abs&offset=0")
    assert code == 200 and c["norm"] == "abs" and c["offset"] is False and len(c["curves"]) == 1
    assert call(srv + f"/api/machines/view?p={q(str(folder / 'notes.txt'))}")[0] == 400
    assert call(srv + f"/api/machines/view?p={q(str(folder / 'broken.mat'))}")[0] == 400
    assert call(srv + f"/api/machines/list?dir={q(str(folder / 'missing'))}")[0] == 400
    req = urllib.request.Request(srv + f"/api/machines/view?p={a}")   # no token
    with pytest.raises(urllib.error.HTTPError): urllib.request.urlopen(req)


# ------------------------------------------------------------------ public matRad machines
PUBLIC = os.environ.get("MATRAD_BASEDATA", "")


def public(name):
    p = os.path.join(PUBLIC, name)
    if not PUBLIC or not os.path.isfile(p): pytest.skip(f"{name}: set MATRAD_BASEDATA to a folder with the public matRad machines")
    return p


def test_public_proton_machine():
    p = public("protons_Generic.mat")
    row = next(r for r in M.list_dir(os.path.dirname(p)) if r["file"] == "protons_Generic.mat")
    assert (row["radiation"], row["data_type"], row["format"]) == ("protons", "doubleGauss", "v7") and row["n"] > 100
    v = M.view(p)
    assert {"sigma1", "sigma2", "weight", "LET"} <= set(v["curves"][0]["fields"])
    assert {"range", "peakPos", "initFocus.sigma@iso"} <= set(v["scalars"])
    r80 = [m["r80"] for m in v["metrics"]]
    assert all(b > a for a, b in zip(r80, r80[1:]))                  # range grows with the energy
    r = M.compare(p, p, [v["energies"][10]])
    assert all(w["d_r80"] == 0 for w in r["rows"]) and len(r["curves"]) == 1


def test_public_carbon_machine_has_let_alpha_beta():
    p = public("carbon_Generic.mat")
    v = M.view(p)
    f = set(v["curves"][0]["fields"])
    assert "LET" in f and any(k.startswith("alpha (") for k in f) and any(k.startswith("beta (") for k in f)
    tr = M.tree(p, 0)
    assert "alphaX" in {n["name"] for n in tr["data"]["children"]}
    pr = public("protons_Generic.mat")
    r = M.compare(pr, p)
    assert len(r["rows"]) == len(M.view(pr)["energies"])
