"""Tests for the FLUKA backend: inputs written from the case model (generators/_fluka.py), checked by reading the
written geometry back and classifying points, against the TOPAS profile; snapshots; the usbrea to CSV converter."""
import bisect, hashlib, importlib.util, json, math, os, random, re

import pytest

np = pytest.importorskip("numpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "opentopas_workbench", "generators")
SNAP = os.path.join(ROOT, "tests", "data", "fluka")


def load(name):
    spec = importlib.util.spec_from_file_location("tf_" + name, os.path.join(GEN, name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


RF, FK, R = load("ripple_filter"), load("_fluka"), load("_rifi")
CTRL = {"ions": ["p"], "energies": "63, 132", "variants": ["rifi0", "t4"], "histories": 200000, "spread": 0.2,
        "period": 1.5, "prefix": "fk"}


# ------------------------------------------------------------------ a small reader of the written geometry
def geometry(text):
    lines = text.splitlines()
    g0 = next(i for i, l in enumerate(lines) if l.startswith("GEOBEGIN")) + 2
    e1 = lines.index("END", g0); e2 = lines.index("END", e1 + 1)
    bodies = {}
    for l in lines[g0:e1]:
        t = l.split(); bodies[t[1]] = (t[0], [float(v) for v in t[2:]])
    regions, cur = {}, None
    for l in lines[e1 + 1:e2]:
        if not l.startswith(" "): cur = l.split()[0]; regions[cur] = l.split()[2:]
        else: regions[cur] += l.split()
    return bodies, regions


class Points:
    """Many points at once, each body's inside test computed once for all of them (numpy), so a region of a thousand
    zones is checked for thousands of points in a fraction of a second."""
    def __init__(self, bodies, pts):
        self.bodies, self.p, self.memo = bodies, np.asarray(pts, dtype=float).reshape(-1, 3), {}

    def inside(self, name):
        if name not in self.memo:
            kind, v = self.bodies[name]; p = self.p
            if kind == "RPP":
                r = np.ones(len(p), bool)
                for k in range(3): r &= (v[2 * k] < p[:, k]) & (p[:, k] < v[2 * k + 1])
            else:
                r = (p - np.array(v[3:])) @ np.array(v[:3]) < 0      # PLA: inside is opposite the normal
            self.memo[name] = r
        return self.memo[name]


def evaluate(tokens, pts):
    """Union of zones ('|'), each an intersection of signed bodies or signed parenthesised expressions."""
    n = len(pts.p)
    def expr(i):
        union, ok, open_ = np.zeros(n, bool), np.ones(n, bool), False
        while i < len(tokens) and tokens[i] != ")":
            t = tokens[i]
            if t == "|":
                if open_: union |= ok
                ok, open_ = np.ones(n, bool), False; i += 1; continue
            sign = t[0]
            if t[1:] == "(" or (t[1:] == "" and tokens[i + 1] == "("):
                val, i = expr(i + (1 if t[1:] == "(" else 2)); i += 1
            else:
                val = pts.inside(t[1:]); i += 1
            ok = ok & (val if sign == "+" else ~val); open_ = True
        if open_: union |= ok
        return union, i
    return expr(0)[0]


def where(bodies, regions, pts):
    """For each point, the number of regions that hold it, and the name of the last one."""
    pts = Points(bodies, pts)
    count, last = np.zeros(len(pts.p), int), np.full(len(pts.p), "", dtype=object)
    for r, toks in regions.items():
        hit = evaluate(toks, pts); count += hit; last[hit] = r
    return count, last


# ------------------------------------------------------------------ geometry checks
SHAPES = ["t4", "stk4", "stk4x3", "stp4", "rifi2", "pair4"]


@pytest.mark.parametrize("v", SHAPES)
def test_regions_tile_the_world(v):
    """Every point of the world lies in exactly one region: no overlap, no hole (FLUKA stops on both)."""
    text = FK.write(RF.case_model("p", 63, v, 1000, 0.2, 1.5, "fk"))
    bodies, regions = geometry(text)
    rnd = random.Random(1)
    pts = []
    for k in range(1500):
        z = rnd.uniform(-80.8, -79.0) if k % 3 else rnd.uniform(-139, 139)   # mostly in the filter layers
        pts.append((rnd.uniform(-7, 7) if k % 2 else rnd.uniform(-29, 29), rnd.uniform(-7, 7), z))
    count, last = where(bodies, regions, pts)
    bad = [(pts[k], count[k], last[k]) for k in range(len(pts)) if count[k] != 1 or last[k] == "BLKHOLE"]
    assert not bad, bad[:5]


def expected_thickness(r, x, y):
    """PMMA along the beam at (x, y) of the TOPAS frame (mm), from the TsRidgeFilter record as TOPAS reads it."""
    th = math.radians(r["rotz"]); u = (x * math.cos(th) + y * math.sin(th)) * 10
    if abs(-x * math.sin(th) + y * math.cos(th)) * 10 > 60: return 0.0
    k = math.floor((u - r["x0"]) / r["pitch"])
    if k < 0 or k > r["n"]: return 0.0
    q = u - r["x0"] - k * r["pitch"]
    if q > r["width"]: return 0.0
    xs, zs = [float(a) for a in r["x"]], [float(a) for a in r["z"]]
    if xs[0] > 0: xs, zs = [0.0] + xs, [0.0] + zs          # closed at 0, as _rifi.ridge_samples reads it
    if xs[-1] < r["width"]: xs, zs = xs + [r["width"]], zs + [0.0]
    return R.interp(q, xs, zs)


@pytest.mark.parametrize("v", ["t4", "stk4", "stk4x3", "rifi1"])
def test_thickness_along_the_beam_matches_topas(v):
    m = RF.case_model("p", 63, v, 1000, 0.2, 1.5, "fk")
    bodies, regions = geometry(FK.write(m))
    rnd = random.Random(2)
    for _ in range(25):
        x, y = rnd.uniform(-2, 2), rnd.uniform(-2, 2)      # TOPAS frame
        for i, r in enumerate(m["filters"], 1):
            z0 = r["tz"] * 10 - r["height"] / 2; nz = 400
            dz = r["height"] / nz
            col = [FK.to_fluka((x, y, (z0 + (j + 0.5) * dz) / 10)) for j in range(nz)]
            got = dz * int(evaluate(regions[f"RIFI{i}"], Points(bodies, col)).sum())
            assert got == pytest.approx(expected_thickness(r, x, y), abs=2 * dz), (v, i, x, y)


# ------------------------------------------------------------------ cards
def test_cards_keep_the_fixed_format():
    text = FK.write(RF.case_model("p", 132, "t4", 200000, 0.2, 1.5, "fk"))
    lines = text.splitlines()
    g0 = next(i for i, l in enumerate(lines) if l.startswith("GEOBEGIN"))
    g1 = next(i for i, l in enumerate(lines) if l.startswith("GEOEND"))
    for l in lines[2:g0] + lines[g1:]:
        if l.startswith("*"): continue
        assert len(l) <= 78 and re.fullmatch(r"[A-Z-]{3,8}", l[:8].strip()), l
        for k in range(6):
            f = l[10 + 10 * k:20 + 10 * k].strip()
            assert not f or re.fullmatch(r"-?\d+\.\d*(E-?\d+)?|[A-Z][A-Za-z0-9-]{0,9}", f), (l, f)
    bodies, regions = geometry(text)
    assert all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,7}", n) for n in list(bodies) + list(regions))
    assert all(len(l) <= 132 for l in lines)


def test_beam_card_turns_sigma_into_fwhm():
    text = FK.write(RF.case_model("p", 132, "rifi0", 200000, 0.2, 1.5, "fk"))
    beam = next(l for l in text.splitlines() if l.startswith("BEAM "))
    w = [float(beam[10 + 10 * k:20 + 10 * k]) for k in range(5)]
    t, m = 0.132, FK.MASS_GEV["proton"]
    p = math.sqrt(t * t + 2 * t * m)
    assert w[0] == -0.132 and w[1] == pytest.approx(-2.35482 * (t + m) / p * t * 0.002, rel=1e-4)
    assert w[2] == pytest.approx(-2.35482, rel=1e-5) and w[3] == w[4] == pytest.approx(-0.706446, rel=1e-5)
    assert beam.rstrip().endswith("PROTON")
    pos = next(l for l in text.splitlines() if l.startswith("BEAMPOS"))
    assert float(pos[30:40]) == -85.0                      # source upstream, beam along +z of the FLUKA frame


def test_unsupported_cases_are_named():
    files = dict(RF.generate_fluka(dict(CTRL, variants=["pin4", "stp4n15", "t4"], ions=["p", "c"])))
    note = files["FLUKA_not_written.txt"]
    assert "pin4" in note and "stp4n15" in note and "c 63" in note
    assert set(files) == {"fk_p063_t4.inp", "fk_p132_t4.inp", "FLUKA_not_written.txt"}


# ------------------------------------------------------------------ snapshots of the control cases
def test_control_case_snapshots():
    """The four control inputs (63 and 132 MeV, with t4 and without filter) as written: a change of the writer shows
    here first. The cases without filter are kept in full, the filter cases by hash (about 1100 lines each)."""
    files = dict(RF.generate_fluka(CTRL))
    snap = json.load(open(os.path.join(SNAP, "snapshots.json")))
    assert sorted(files) == sorted(snap)
    for n, t in files.items():
        if n.endswith("rifi0.inp"): assert t == open(os.path.join(SNAP, n)).read(), n
        assert hashlib.sha256(t.encode()).hexdigest() == snap[n], n


def test_topas_generate_unchanged_by_the_fluka_writer():
    """generate() for TOPAS is byte-identical with the snapshot taken before the case model existed."""
    snap = json.load(open(os.path.join(SNAP, "topas_before.json")))
    files = RF.generate(dict(CTRL, variants=["rifi0", "t4", "stk4", "stp4n15", "pin4"]))
    got = {n: hashlib.sha256(t if isinstance(t, bytes) else t.encode()).hexdigest() for n, t in files}
    assert got == snap


# ------------------------------------------------------------------ results
USBREA = """ ****  Cartesian binning n.   1  "IDD       " , generalized particle n.  228
      X coordinate: from -1.0000E+01 to  1.0000E+01 cm,     1 bins ( 2.0000E+01 cm wide)
      Y coordinate: from -1.0000E+01 to  1.0000E+01 cm,     1 bins ( 2.0000E+01 cm wide)
      Z coordinate: from  0.0000E+00 to  1.2000E+01 cm,     4 bins ( 3.0000E+00 cm wide)
      Data follow in a matrix A(ix,iy,iz), format (1(5x,1p,10(1x,e11.4)))

      accurate deposition along the tracks requested
  1.0000E-03  2.0000E-03  4.0000E-03  0.0000E+00

 Percentage errors follow in a matrix A(ix,iy,iz), format (1(5x,1p,10(1x,e11.4)))

  1.0000E+00  2.0000E+00  5.0000E-01  0.0000E+00
"""


def test_usbrea_to_workbench_csv():
    m = RF.case_model("p", 63, "rifi0", 1000, 0.2, 1.5, "fk")
    m["phantom"]["zbins"] = 4
    vals, errs = FK.read_usbrea(USBREA, 4)
    assert vals == [1e-3, 2e-3, 4e-3, 0.0] and errs == [1.0, 2.0, 0.5, 0.0]
    csv = FK.to_csv(vals, errs, m).splitlines()
    assert "# Z in 4 bins of 3 cm" in csv and csv[5].startswith("# DoseToMedium ( Gy ) : Sum")
    k2 = [float(v) for v in csv[8].split(",")]
    assert k2[:3] == [0, 0, 2] and k2[3] == pytest.approx(4e-3 * 1.602176634e-7 * 1000) and k2[4] == pytest.approx(k2[3] * 0.005)
    with pytest.raises(ValueError, match="4 bins"): FK.read_usbrea("no numbers here\n1.0 2.0\n", 4)


# ------------------------------------------------------------------ the FLUKA job of the workbench queue
# Stand-ins for rfluka, usbsuw and usbrea: shell scripts that do what the job expects of the real programs. The tests
# never start FLUKA; only its user does, from the queue.
FAKE = {
    "rfluka": '#!/bin/bash\necho "rfluka $@" >> calls.txt\nfor c in 001 002 003 004 005; do echo bin > "${@: -1}${c}_fort.21"; done\n',
    "usbsuw": '#!/bin/bash\ncat > usbsuw.stdin\necho bnn > "$(tail -n 1 usbsuw.stdin)"\n',
    "usbrea": '#!/bin/bash\nread inp; read out\n[ -f "$inp" ] || exit 3\ncat "$USBREA_TEXT" > "$out"\n',
}


@pytest.fixture
def queue(tmp_path, monkeypatch):
    from opentopas_workbench import server as wb
    fk = tmp_path / "fluka_bin"; fk.mkdir()
    for n, body in FAKE.items(): (fk / n).write_text(body); (fk / n).chmod(0o755)
    (tmp_path / "usbrea.txt").write_text(USBREA)
    monkeypatch.setenv("USBREA_TEXT", str(tmp_path / "usbrea.txt"))
    ws = tmp_path / "ws"; ws.mkdir()
    monkeypatch.setitem(wb.STATE, "root", str(ws))
    monkeypatch.setitem(wb.STATE, "cfg", {"binaries": [], "lang": "en", "fluka_dir": str(fk), "keep_awake": False, "toast": False})
    monkeypatch.setitem(wb.STATE, "jobs", [])
    monkeypatch.setitem(wb.STATE, "proc", None)
    m = RF.case_model("p", 63, "rifi0", 1003, 0.2, 1.5, "fk"); m["phantom"]["zbins"] = 4
    (ws / "cases").mkdir(); (ws / "cases" / "fk_p063_rifi0.inp").write_text(FK.write(m))
    return wb, ws


def test_fluka_job_waits_for_the_start_by_hand(queue):
    wb, ws = queue
    wb.enqueue(["cases/fk_p063_rifi0.inp"], "OpenTOPAS (topas4)", "", False)
    assert [(j["binary"], j["status"]) for j in wb.STATE["jobs"]] == [("FLUKA", "held")]
    wb.worker_step()
    assert wb.STATE["jobs"][0]["status"] == "held" and not (ws / "cases" / "calls.txt").exists()
    assert wb.release_jobs({999}) == 0 and wb.release_jobs({wb.STATE["jobs"][0]["id"]}) == 1
    assert wb.STATE["jobs"][0]["status"] == "queued"


def test_fluka_job_runs_merges_and_writes_the_csv(queue):
    wb, ws = queue
    wb.enqueue(["cases/fk_p063_rifi0.inp"], "", "out", False)
    wb.release_jobs()
    wb.worker_step()
    job = wb.STATE["jobs"][0]
    out = ws / "cases" / "out"
    assert job["status"] == "done", open(ws / job["log"]).read()
    assert (out / "calls.txt").read_text() == "rfluka -N0 -M5 fk_p063_rifi0\n"
    assert "PATH" not in wb.fluka_script("x")
    sw = (out / "usbsuw.stdin").read_text().splitlines()
    assert sw == [f"fk_p063_rifi0{c:03d}_fort.21" for c in range(1, 6)] + ["", "fk_p063_rifi0_21.bnn"]
    csv = (out / "IDD_fk_p063_rifi0_fluka.csv").read_text().splitlines()
    assert "cases/out/IDD_fk_p063_rifi0_fluka.csv" in job["outputs"]
    k2 = [float(v) for v in csv[8].split(",")]
    assert k2[3] == pytest.approx(4e-3 * 1.602176634e-7 * 1005)     # summed over the 5 x 201 primaries that ran
    # and the workbench reads it like a TOPAS result
    r = wb.read_topas_csv(str(out / "IDD_fk_p063_rifi0_fluka.csv"))
    assert r is not None


def test_fluka_job_fails_when_usbrea_gives_nothing(queue, monkeypatch, tmp_path):
    wb, ws = queue
    (tmp_path / "usbrea.txt").write_text("no numbers\n")
    wb.enqueue(["cases/fk_p063_rifi0.inp"], "", "", False); wb.release_jobs(); wb.worker_step()
    job = wb.STATE["jobs"][0]
    assert job["status"] == "failed" and "not converted" in open(ws / job["log"]).read()


def test_fluka_job_never_falls_back_to_path(queue, monkeypatch, tmp_path):
    """A wrong fluka_dir fails the job; a FLUKA on PATH is never started instead (licence: its user starts each run)."""
    wb, ws = queue
    monkeypatch.setenv("PATH", str(tmp_path / "fluka_bin") + os.pathsep + os.environ["PATH"])
    wb.STATE["cfg"]["fluka_dir"] = str(tmp_path / "missing")
    wb.enqueue(["cases/fk_p063_rifi0.inp"], "", "", False); wb.release_jobs(); wb.worker_step()
    job = wb.STATE["jobs"][0]
    assert job["status"] == "failed" and job["rc"] == 127 and not (ws / "cases" / "calls.txt").exists()
    assert "rfluka not found" in open(ws / job["log"]).read()


def test_fluka_job_is_never_started_without_the_start_button(queue):
    """Neither a TOPAS file sent to FLUKA nor a server restart during a FLUKA run lets FLUKA start on its own."""
    wb, ws = queue
    (ws / "cases" / "a.txt").write_text("s:Ge/World/Material = \"G4_AIR\"\n")
    with pytest.raises(ValueError, match=".inp inputs only"): wb.enqueue(["cases/a.txt"], "FLUKA", "", False)
    wb.enqueue(["cases/fk_p063_rifi0.inp"], "", "", False); wb.release_jobs()
    wb.STATE["jobs"][0]["status"] = "running"; wb.save_jobs()
    wb.load_jobs()
    assert wb.STATE["jobs"][0]["status"] == "held"
    wb.worker_step()
    assert not (ws / "cases" / "calls.txt").exists()
