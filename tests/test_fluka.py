"""Tests for the FLUKA backend: inputs written from the case model (generators/_fluka.py), checked by reading the
written geometry back and classifying points, against the TOPAS profile; snapshots; the usbrea to CSV converter."""
import bisect, hashlib, importlib.util, json, math, os, random, re, sys, threading, time, urllib.error, urllib.request

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
            elif kind == "RCC":                                          # base V, height H, radius R
                d, h = p - np.array(v[:3]), np.array(v[3:6])
                t = d @ h / (h @ h)
                r = (0 < t) & (t < 1) & (np.linalg.norm(d - np.outer(t, h), axis=1) < v[6])
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
    assert "pin4" in note and "stp4n15" in note and "c 132 t4" in note and "DPMJET" in note and "c 63 t4" not in note
    assert set(files) == {"fk_p063_t4.inp", "fk_p132_t4.inp", "fk_c063_t4.inp", "FLUKA_not_written.txt"}


def test_carbon_beam_is_a_heavy_ion_per_nucleon():
    """12C: BEAM HEAVYION takes kinetic energy and momentum per nucleon (manual, BEAM), HI-PROPE names Z and A."""
    m = RF.case_model("c", 88, "t3", 100000, 0.2, 1.5, "fk")
    assert m["beam"]["particle"] == "GenericIon(6,12)" and m["beam"]["energy_mev"] == 12 * 88
    lines = FK.write(m).splitlines()
    beam = next(l for l in lines if l.startswith("BEAM "))
    w = [float(beam[10 + 10 * k:20 + 10 * k]) for k in range(2)]
    t, mu = 0.088, 11.17486 / 12
    p = math.sqrt(t * t + 2 * t * mu)
    assert beam.rstrip().endswith("HEAVYION") and w[0] == pytest.approx(-0.088)
    assert w[1] == pytest.approx(-2.35482 * (t + mu) / p * t * 0.002, rel=1e-4)
    hi = next(l for l in lines if l.startswith("HI-PROPE"))
    assert [float(hi[10 + 10 * k:20 + 10 * k]) for k in range(2)] == [6.0, 12.0]
    assert not any(l.startswith("HI-PROPE") for l in FK.write(RF.case_model("p", 63, "t3", 1000, 0.2, 1.5, "fk")).splitlines())


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
# Layout of usbrea's ASCII output: a Fortran carriage-control "1" on the first line, then the detector header.
USBREA = """1
   Cartesian binning n.   1  "IDD       " , generalized particle n.  228
      X coordinate: from -1.0000E+01 to  1.0000E+01 cm,     1 bins ( 2.0000E+01 cm wide)
      Y coordinate: from -1.0000E+01 to  1.0000E+01 cm,     1 bins ( 2.0000E+01 cm wide)
      Z coordinate: from  0.0000E+00 to  1.2000E+01 cm,     4 bins ( 3.0000E+00 cm wide)
      Data follow in a matrix A(ix,iy,iz), format (1(5x,1p,10(1x,e11.4)))

      accurate deposition along the tracks requested
      this is a track-length binning
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


def usbrea_text(vals, errs):
    """A usbrea ASCII file of one 1 x 1 x len(vals) detector, ten numbers a line as usbrea prints them."""
    def rows(xs): return "".join("     " + "".join(f" {x:11.4E}" for x in xs[i:i + 10]) + "\n" for i in range(0, len(xs), 10))
    n = len(vals)
    head = ("1\n   Cartesian binning n.   1  \"IDD       \" , generalized particle n.  228\n"
            f"      Z coordinate: from  0.0000E+00 to  1.2000E+01 cm,  {n:4d} bins\n"
            "      Data follow in a matrix A(ix,iy,iz), format (1(5x,1p,10(1x,e11.4)))\n\n"
            "      accurate deposition along the tracks requested\n      this is a track-length binning\n")
    tail = "\n      Percentage errors follow in a matrix A(ix,iy,iz), format (1(5x,1p,10(1x,e11.4)))\n\n" + rows(errs) if errs else ""
    return head + rows(vals) + tail


def test_usbrea_first_and_last_bin_kept():
    """Task 3357: the carriage-control "1" that opens the file was read as bin 0 (Bin0/Bin1 = 4e4 in Robin's first
    run), every bin moved one deeper and the last one fell off. Bin 0 and the last bin are the file's own now."""
    vals = [2.65e-5 + 1e-7 * k for k in range(12)]
    errs = [0.1 * (k + 1) for k in range(12)]
    got, gerr = FK.read_usbrea(usbrea_text(vals, errs), 12)
    assert got == pytest.approx(vals, rel=1e-4) and gerr == pytest.approx(errs, rel=1e-4)
    got, gerr = FK.read_usbrea(usbrea_text(vals, None), 12)
    assert got == pytest.approx(vals, rel=1e-4) and gerr is None
    with pytest.raises(ValueError, match="11 values .* 12 bins"): FK.read_usbrea(usbrea_text(vals[:11], None), 12)


# ------------------------------------------------------------------ the FLUKA job of the workbench queue
# Stand-ins for rfluka, usbsuw and usbrea: shell scripts that do what the job expects of the real programs. The tests
# never start FLUKA; only its user does, from the queue.
FAKE = {
    "rfluka": r"""#!/bin/bash
echo "rfluka $@" >> calls.txt
name="${@: -1}"; cycles="${2#-M}"; me=$(basename "$PWD")
[ -f "$name.inp" ] || exit 4
echo $$ > pid
if [ -n "$FAKE_TOGETHER" ]; then
  n=0
  while [ "$(ls ../*/pid | wc -l)" -lt "$FAKE_TOGETHER" ]; do n=$((n+1)); [ $n -lt 100 ] || exit 9; sleep 0.1; done
fi
if [ "$FAKE_FAIL" = "$me" ]; then echo "FLUKA stopped: geometry"; exit 3; fi
if [ "$FAKE_EMPTY" = "$me" ]; then exit 0; fi
if [ -n "$FAKE_HANG" ]; then
  # one child that ignores SIGTERM, one that leaves the process group
  ( trap "" TERM; exec sleep 60 ) & echo $! > stubborn.pid
  if [ -d /proc ]; then "$FAKE_PYTHON" -c 'import os, time; os.setsid(); open("detached.pid", "w").write(str(os.getpid())); time.sleep(60)' & fi
  sleep 60
fi
for ((c=1; c<=cycles; c++)); do f=$(printf '%s%03d_fort.21' "$name" $c); echo "bin $me $c" > "$f"; done
""",
    "usbsuw": '#!/bin/bash\ncat > usbsuw.stdin\nout=$(tail -n 1 usbsuw.stdin)\nwhile read f; do [ -n "$f" ] || break; cat "$f" >> "$out" || exit 5; done < usbsuw.stdin\n',
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
    monkeypatch.setattr(wb, "FLUKA_DIR_DEFAULT", str(tmp_path / "no_installation"))   # no test may reach a real FLUKA
    monkeypatch.setitem(wb.STATE, "cfg", {"binaries": [], "lang": "en", "fluka_dir": str(fk), "fluka_parts": 1,
                                          "keep_awake": False, "toast": False})
    monkeypatch.setitem(wb.STATE, "jobs", [])
    monkeypatch.setitem(wb.STATE, "proc", None)
    monkeypatch.setitem(wb.STATE, "stop", False)
    monkeypatch.setattr(wb, "free_cores", lambda seconds=1.0: pytest.fail("a fixed setting needs no measurement"))
    monkeypatch.setattr(wb, "LIVE", {})
    for var in ("FAKE_TOGETHER", "FAKE_FAIL", "FAKE_EMPTY", "FAKE_HANG"): monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FAKE_PYTHON", sys.executable)
    m = RF.case_model("p", 63, "rifi0", 1003, 0.2, 1.5, "fk"); m["phantom"]["zbins"] = 4
    (ws / "cases").mkdir(); (ws / "cases" / "fk_p063_rifi0.inp").write_text(FK.write(m))
    return wb, ws


def test_fluka_job_waits_for_the_start_by_hand(queue):
    wb, ws = queue
    wb.enqueue(["cases/fk_p063_rifi0.inp"], "OpenTOPAS (topas4)", "", False)
    assert [(j["binary"], j["status"]) for j in wb.STATE["jobs"]] == [("FLUKA", "held")]
    wb.worker_step()
    assert wb.STATE["jobs"][0]["status"] == "held" and not (ws / "cases" / "fk_p063_rifi0_fluka_runs").exists()
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
    runs = out / "fk_p063_rifi0_fluka_runs"
    assert (runs / "01" / "calls.txt").read_text() == "rfluka -N0 -M5 run\n"
    assert (runs / "01" / "run.inp").read_text() == (ws / "cases" / "fk_p063_rifi0.inp").read_text()
    assert "PATH" not in wb.fluka_script("x", 3, 5)
    sw = (runs / "usbsuw.stdin").read_text().splitlines()
    assert sw == [f"01/run{c:03d}_fort.21" for c in range(1, 6)] + ["", "merged.bnn"]
    assert (out / "fk_p063_rifi0_21.bnn").is_file() and (out / "fk_p063_rifi0_21.txt").is_file()
    csv = (out / "IDD_fk_p063_rifi0_fluka.csv").read_text().splitlines()
    assert "cases/out/IDD_fk_p063_rifi0_fluka.csv" in job["outputs"]
    k2 = [float(v) for v in [l for l in csv if not l.startswith("#")][2].split(",")]
    assert csv[1:3] == ["# Histories: 1005", "# Seed: 54217137"] and not any("Version" in l for l in csv)   # no version in "fluka_bin"
    assert job["code"] == {"code": "FLUKA", "version": None, "histories": 1005, "seed": 54217137, "parts": None}
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
    assert job["status"] == "failed" and "rfluka not found" in job["error"]
    assert not list((ws / "cases").glob("*_fluka_runs/*/calls.txt"))


@pytest.mark.parametrize("why", ["missing", "relative", "stop"])
def test_an_earlier_run_survives_a_start_that_cannot_run(queue, monkeypatch, tmp_path, why):
    """A wrong FLUKA folder, or a stop while the job prepares, deletes nothing of the sub-runs of an earlier run."""
    wb, ws = queue
    wb.STATE["cfg"]["fluka_parts"] = 2
    t, job = start_job(wb); t.join(30)
    runs = ws / "cases" / "fk_p063_rifi0_fluka_runs"
    assert job["status"] == "done" and (runs / "02" / "run005_fort.21").is_file()
    if why == "missing": wb.STATE["cfg"]["fluka_dir"] = str(tmp_path / "missing")
    if why == "relative":
        monkeypatch.chdir(tmp_path); wb.STATE["cfg"]["fluka_dir"] = "fluka_bin"   # a working installation, but relative
    if why == "stop":
        wb.STATE["cfg"]["fluka_parts"] = 0
        def measure(seconds=1.0): wb.stop_current(); return 6
        monkeypatch.setattr(wb, "free_cores", measure)
    t, job = start_job(wb); t.join(30)
    want = {"missing": "rfluka not found", "relative": "not an absolute path"}
    assert job["status"] == ("stopped" if why == "stop" else "failed") and want.get(why, "") in job.get("error", "")
    assert (runs / "02" / "run005_fort.21").is_file() and (runs / "02" / "calls.txt").read_text().count("rfluka") == 1
    assert wb.STATE["stop"] is False and "subruns" not in job


def test_a_folder_that_was_emptied_half_stays_the_workbench_s(queue, monkeypatch):
    """rmtree that fails halfway must not leave a folder without its mark, which every later run would refuse."""
    wb, ws = queue
    t, job = start_job(wb); t.join(30)
    runs = ws / "cases" / "fk_p063_rifi0_fluka_runs"
    real = wb.shutil.rmtree
    def half(path, *a, **k):
        (runs / wb.FLUKA_MARK).unlink(); real(runs / "01")
        raise PermissionError("a file is open elsewhere")
    monkeypatch.setattr(wb.shutil, "rmtree", half)
    t, job = start_job(wb); t.join(30)
    assert job["status"] == "failed" and "open elsewhere" in job["error"] and (runs / wb.FLUKA_MARK).is_file()
    monkeypatch.setattr(wb.shutil, "rmtree", real)
    t, job = start_job(wb); t.join(30)
    assert job["status"] == "done", job.get("error")


def test_fluka_job_is_never_started_without_the_start_button(queue):
    """Neither a TOPAS file sent to FLUKA nor a server restart during a FLUKA run lets FLUKA start on its own."""
    wb, ws = queue
    (ws / "cases" / "a.txt").write_text("s:Ge/World/Material = \"G4_AIR\"\n")
    assert wb.enqueue(["cases/a.txt"], "FLUKA", "", False)[2][0]["file"] == "cases/a.txt" and not wb.STATE["jobs"]
    wb.enqueue(["cases/fk_p063_rifi0.inp"], "", "", False); wb.release_jobs()
    wb.STATE["jobs"][0]["status"] = "running"; wb.save_jobs()
    wb.load_jobs()
    assert wb.STATE["jobs"][0]["status"] == "held"
    wb.worker_step()
    assert not (ws / "cases" / "fk_p063_rifi0_fluka_runs").exists()


# ------------------------------------------------------------------ sub-runs on several cores
def field(text, name, col):
    return float(next(l for l in text.splitlines() if l.startswith(name))[col:col + 10])


@pytest.mark.parametrize("n, cycles", [(2, 5), (3, 5), (4, 5), (5, 1), (7, 1), (16, 1)])
def test_sub_run_inputs_differ_only_in_seed_and_primaries(n, cycles):
    text = FK.write(RF.case_model("c", 88, "t3", 100003, 0.2, 1.5, "fk"))
    inputs, starts, m = FK.sub_runs(text, n)
    assert len(inputs) == len(starts) == n and m == cycles
    assert sum(starts) * m == 100005                   # the total of the case, as FLUKA ran it on one core
    assert max(starts) - min(starts) <= 1
    for part, start in zip(inputs, starts):
        a, b = text.splitlines(), part.splitlines()
        assert len(a) == len(b)
        changed = [(x, y) for x, y in zip(a, b) if x != y]
        same_start = start == 20001                    # five sub-runs of one cycle each keep the START of the case
        assert [x[:8] for x, _ in changed] == ["RANDOMIZ"] + ([] if same_start else ["START   "])
        (r0, r1) = changed[0]
        (s0, s1) = next((x, y) for x, y in zip(a, b) if x.startswith("START "))
        assert r0[:20] == r1[:20] and r0[30:].rstrip() == r1[30:].rstrip()
        assert s0[:10] == s1[:10] and s0[20:].rstrip() == s1[20:].rstrip()
        assert float(s1[10:20]) == start
        seed = float(r1[20:30])
        assert seed == int(seed) and 1 <= seed < 9e8
    assert len({field(p, "RANDOMIZ", 20) for p in inputs}) == n


def test_sub_run_seeds_of_repeated_cases_do_not_overlap():
    """Repeats use the seeds 1, 2, 3: seed + k would give them the same sub-run sequences."""
    def seeds(seed):
        m = RF.case_model("p", 63, "rifi0", 100000, 0.2, 1.5, "fk"); m["beam"]["seed"] = seed
        return [int(field(p, "RANDOMIZ", 20)) for p in FK.sub_runs(FK.write(m), 16)[0]]
    every = seeds(1) + seeds(2) + seeds(3)
    assert len(set(every) | {1, 2, 3}) == 3 * 16 + 3
    assert seeds(2) == every[16:32]                    # and the same case gives the same seeds again


def test_sub_runs_of_tiny_cases_and_of_inputs_that_cannot_be_split():
    def split(histories, n): return FK.sub_runs(FK.write(RF.case_model("p", 63, "rifi0", histories, 0.2, 1.5, "fk")), n)[1:]
    assert split(5, 16) == ([1] * 5, 1) and split(5, 3) == ([1], 5) and split(10, 4) == ([1, 1], 5)
    text = FK.write(RF.case_model("p", 63, "rifi0", 1000, 0.2, 1.5, "fk"))
    inputs, starts, _ = FK.sub_runs(text.replace("\n", "\r\n"), 5)
    assert starts == [200] * 5 and inputs[0].count("\r\n") == inputs[0].count("\n")
    card = next(l for l in text.splitlines(True) if l.startswith("RANDOMIZ"))
    for bad in (text.replace(card, ""), text.replace(card, card + card), text.replace(card, card[:20] + "  not.num\n")):
        with pytest.raises(ValueError): FK.sub_runs(bad, 4)


def test_progress_line_of_the_standard_output():
    text = (" NEXT SEEDS:       0       0       0       0       0       0       1    3039       0       0\n"
            "          1                 79999                 79999             5.2595139E-04         1.0000000E+30                 0\n"
            " NEXT SEEDS:    13E1       0       0       0       0       0       1    3039       0       0\n"
            "      11000                 69000                 69000             4.6403818E-04         1.0000000E+30              1657\n"
            " NEXT SEEDS: 3889978       0       0       0       0       0       1    3039       0       0\n")
    assert FK.read_handled(text, 80000) == 11000
    assert FK.read_handled(text, 40000) == 0           # lines of another run
    assert FK.read_handled("no progress yet\n", 80000) == 0


def start_job(wb, subdir=""):
    wb.enqueue(["cases/fk_p063_rifi0.inp"], "", subdir, False); wb.release_jobs()
    t = threading.Thread(target=wb.worker_step, daemon=True); t.start()
    return t, wb.STATE["jobs"][-1]


def wait_for(cond, what, seconds=30):
    end = time.monotonic() + seconds
    while not cond():
        assert time.monotonic() < end, what
        time.sleep(0.05)


def alive(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    if os.path.isdir("/proc"):
        try:
            with open(f"/proc/{pid}/stat") as fh: return fh.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
        except OSError: return False
    return True


@pytest.mark.parametrize("n, cycles", [(1, 5), (3, 5), (7, 1), (16, 1)])
def test_merge_gets_every_sub_run(queue, monkeypatch, n, cycles):
    wb, ws = queue
    wb.STATE["cfg"]["fluka_parts"] = n
    monkeypatch.setenv("FAKE_TOGETHER", str(n))        # each waits for all others, so they have to run at the same time
    t, job = start_job(wb, "out"); t.join(40)
    assert job["status"] == "done", open(ws / job["log"]).read()
    out = ws / "cases" / "out"; runs = out / "fk_p063_rifi0_fluka_runs"
    dirs = [f"{k:02d}" for k in range(1, n + 1)]
    files = [f"{d}/run{c:03d}_fort.21" for d in dirs for c in range(1, cycles + 1)]
    assert (runs / "usbsuw.stdin").read_text().splitlines() == files + ["", "merged.bnn"]
    assert (out / "fk_p063_rifi0_21.bnn").read_text().splitlines() == [f"bin {d} {c}" for d in dirs for c in range(1, cycles + 1)]
    for d in dirs: assert (runs / d / "calls.txt").read_text() == f"rfluka -N0 -M{cycles} run\n"
    assert job["threads"] == n and job["total"] == 1005 and not job["hints"]
    csv = (out / "IDD_fk_p063_rifi0_fluka.csv").read_text().splitlines()
    assert ("# Sub-runs: %d" % n in csv) == (n > 1) and job["code"]["parts"] == (n if n > 1 else None)
    k2 = [float(v) for v in [l for l in csv if not l.startswith("#")][2].split(",")]
    assert k2[3] == pytest.approx(4e-3 * 1.602176634e-7 * 1005) and len(k2) == 5     # sum and its error


@pytest.mark.parametrize("how", ["FAKE_FAIL", "FAKE_EMPTY"])
def test_a_failed_sub_run_ends_the_others_and_merges_nothing(queue, monkeypatch, how):
    wb, ws = queue
    wb.STATE["cfg"]["fluka_parts"] = 4
    monkeypatch.setattr(wb, "STOP_GRACE", 0.5)
    monkeypatch.setenv(how, "03"); monkeypatch.setenv("FAKE_HANG", "1"); monkeypatch.setenv("FAKE_TOGETHER", "4")
    (ws / "cases" / "IDD_fk_p063_rifi0_fluka.csv").write_text("earlier result\n")
    t, job = start_job(wb); t.join(30)
    assert not t.is_alive() and job["status"] == "failed" and job["rc"] == (3 if how == "FAKE_FAIL" else 1)
    runs = ws / "cases" / "fk_p063_rifi0_fluka_runs"
    assert not (runs / "usbsuw.stdin").exists() and not (ws / "cases" / "fk_p063_rifi0_21.txt").exists()
    assert (ws / "cases" / "IDD_fk_p063_rifi0_fluka.csv").read_text() == "earlier result\n"
    assert any("sub-run 03 of 4 failed" in h for h in job["hints"])
    pids = [int(p.read_text()) for p in runs.glob("*/*pid") if p.read_text().strip()]
    assert len(pids) >= 4
    wait_for(lambda: not any(alive(p) for p in pids), "a process of the failed job still runs", 10)
    assert wb.STATE["proc"] is None and wb.STATE["stop"] is False


def test_sub_runs_ignore_cdpath(queue, monkeypatch, tmp_path):
    """bash looks a plain folder name up in CDPATH first: the job must not land in a folder of the same name there."""
    wb, ws = queue
    other = tmp_path / "elsewhere" / "fk_p063_rifi0_fluka_runs"; (other / "01").mkdir(parents=True)
    monkeypatch.setenv("CDPATH", str(tmp_path / "elsewhere"))
    t, job = start_job(wb); t.join(30)
    assert job["status"] == "done", open(ws / job["log"]).read()
    assert (ws / "cases" / "fk_p063_rifi0_fluka_runs" / "01" / "calls.txt").is_file() and not list(other.glob("01/*"))


def test_stop_ends_every_process(queue, monkeypatch):
    wb, ws = queue
    wb.STATE["cfg"]["fluka_parts"] = 3
    monkeypatch.setattr(wb, "STOP_GRACE", 0.5)
    monkeypatch.setenv("FAKE_HANG", "1")
    t, job = start_job(wb)
    runs = ws / "cases" / "fk_p063_rifi0_fluka_runs"
    names = ["pid", "stubborn.pid"] + (["detached.pid"] if os.path.isdir("/proc") else [])
    files = [runs / f"{k:02d}" / n for k in range(1, 4) for n in names]
    wait_for(lambda: all(f.is_file() and f.read_text().strip() for f in files), "the sub-runs did not start")
    pids = [int(f.read_text()) for f in files]
    assert all(alive(p) for p in pids)
    wb.stop_current(); t.join(30)
    assert not t.is_alive() and job["status"] == "stopped" and not (runs / "usbsuw.stdin").exists()
    wait_for(lambda: not any(alive(p) for p in pids), "a process of the stopped job still runs", 10)


def test_stop_during_the_preparation_starts_nothing(queue, monkeypatch):
    wb, ws = queue
    wb.STATE["cfg"]["fluka_parts"] = 0                  # automatic: the cores are measured first
    def measure(seconds=1.0):
        wb.stop_current()
        return 6
    monkeypatch.setattr(wb, "free_cores", measure)
    t, job = start_job(wb); t.join(30)
    runs = ws / "cases" / "fk_p063_rifi0_fluka_runs"
    assert job["status"] == "stopped" and not runs.exists() and "stopped before" in open(ws / job["log"]).read()
    assert wb.STATE["stop"] is False


def test_a_rerun_empties_only_a_folder_of_its_own(queue):
    wb, ws = queue
    runs = ws / "cases" / "fk_p063_rifi0_fluka_runs"
    wb.STATE["cfg"]["fluka_parts"] = 6
    t, job = start_job(wb); t.join(30)
    assert job["status"] == "done" and (runs / "06").is_dir()
    (runs / "02" / "ranrun001").write_text("seeds of the earlier run")
    wb.STATE["cfg"]["fluka_parts"] = 2
    t, job = start_job(wb); t.join(30)
    assert job["status"] == "done" and not (runs / "06").exists() and not (runs / "02" / "ranrun001").exists()
    (runs / wb.FLUKA_MARK).unlink(); (runs / "notes.md").write_text("mine")
    t, job = start_job(wb); t.join(30)
    assert job["status"] == "failed" and "not made by the workbench" in job["error"] and (runs / "notes.md").read_text() == "mine"


@pytest.mark.parametrize("free, parts", [(0, 1), (2, 1), (3, 1), (7, 5), (18, 16), (64, 16)])
def test_default_is_the_free_cores_minus_two_at_most_16(queue, monkeypatch, free, parts):
    wb, _ = queue
    monkeypatch.setattr(wb, "free_cores", lambda seconds=1.0: free)
    wb.STATE["cfg"]["fluka_parts"] = 0
    assert wb.fluka_parts() == parts


def test_free_cores_count_what_idles_now(tmp_path, monkeypatch):
    from opentopas_workbench import server as wb
    stat = tmp_path / "stat"; stat.write_text("cpu  0 0 0 0 0 0 0 0 0 0\n")
    monkeypatch.setattr(wb, "PROC_DIR", str(tmp_path))
    monkeypatch.setattr(wb, "usable_cores", lambda: 18)
    monkeypatch.setattr(wb.os, "cpu_count", lambda: 18)
    monkeypatch.setattr(wb.time, "sleep", lambda s: stat.write_text("cpu  600 0 0 1100 100 0 0 0 0 0\n"))
    assert wb.free_cores() == 12                        # two thirds of 18 idled or waited for the disk


def test_api_reports_progress_cores_and_the_setting(queue):
    wb, ws = queue
    top = ws / "cases" / "x_fluka_runs"
    line = "   {}        {}        {}     4.6E-04     1.0000000E+30      12\n"
    for d in ("01", "02/fluka_4711", "03"): (top / d).mkdir(parents=True)
    (top / "01" / "run001_fort.21").write_text("bin")                                     # finished
    (top / "02" / "fluka_4711" / "run001.out").write_text(line.format(250, 751, 751))     # runs
    log = ws / "job.log"; log.write_text("# start\nStarted 3 sub-runs\n")
    wb.STATE["next_id"] = 8
    wb.STATE["jobs"].append({"id": 7, "file": "cases/fk_p063_rifi0.inp", "binary": "FLUKA", "status": "running",
                             "log": "job.log", "t0": time.time(), "total": 3003, "threads": 3,
                             "subruns": {"dir": "cases/x_fluka_runs", "cycles": 1, "starts": [1001, 1001, 1001]}})
    srv = wb.Server(("127.0.0.1", 0), wb.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    def api(path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}{path}",
                                     None if body is None else json.dumps(body).encode(),
                                     {"X-Token": wb.TOKEN, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r: return json.load(r)
    try:
        j = api("/api/jobs")["jobs"][0]
        assert j["threads"] == 3 and len(j["subruns"]["starts"]) == 3
        assert j["done"] == 1251 and j["live"]["phase"] == "run" and j["progress"] == pytest.approx(1251 / 3003)
        (top / "02" / "fluka_4711" / "run001.out").unlink()      # while a cycle's files move, progress does not fall back
        assert api("/api/jobs")["jobs"][0]["done"] == 1251
        api("/api/config", {"fluka_parts": "7"})
        info = api("/api/info")
        assert info["cfg"]["fluka_parts"] == 7 and 1 <= info["fluka_parts_auto_max"] <= 16
        for bad in ("17", "-1", "2.5", "x"):
            with pytest.raises(urllib.error.HTTPError): api("/api/config", {"fluka_parts": bad, "fluka_dir": "/changed"})
        cfg = api("/api/info")["cfg"]
        assert cfg["fluka_parts"] == 7 and cfg["fluka_dir"] != "/changed"
        api("/api/config", {"fluka_parts": ""})
        assert api("/api/info")["cfg"]["fluka_parts"] == 0   # automatic again
    finally:
        srv.shutdown(); srv.server_close()


def test_queue_finish_time_counts_waiting_cases(queue):
    wb, _ = queue
    wb.enqueue(["cases/fk_p063_rifi0.inp"], "", "", False); wb.release_jobs()
    waiting = wb.STATE["jobs"][0]
    assert waiting["total"] == 1005
    done = {"binary": "FLUKA", "status": "done", "seconds": 100, "total": 2010, "energy": None}
    est = wb.queue_estimate([done, dict(waiting)])
    assert est["seconds"] == 50 and est["unknown"] == 0


# ------------------------------------------------------------------ review findings of the sub-runs
def test_a_single_sub_run_runs_an_input_without_seed_or_start_card_unchanged(queue):
    """FLUKA runs such an input with the manual's defaults; with 1 sub-run the workbench must not refuse it."""
    wb, ws = queue
    text = (ws / "cases" / "fk_p063_rifi0.inp").read_text()
    bare = "".join(l for l in text.splitlines(True) if not l.startswith(("RANDOMIZ", "START")))
    assert FK.sub_runs(bare, 1) == ([bare], [5000], 5)
    assert FK.sub_runs(text, 1)[0] == [text]
    with pytest.raises(ValueError, match="RANDOMIZ"): FK.sub_runs(bare, 2)
    (ws / "cases" / "fk_p063_rifi0.inp").write_text(bare)
    t, job = start_job(wb); t.join(30)
    assert job["status"] == "done", job.get("error")
    assert (ws / "cases" / "fk_p063_rifi0_fluka_runs" / "01" / "run.inp").read_text() == bare and job["total"] == 25000


def test_sub_run_folders_stay_out_of_the_folder_pickers(queue):
    wb, ws = queue
    for k in range(40): (ws / "a" / f"case{k:02d}_fluka_runs" / "01").mkdir(parents=True)
    (ws / "z_real").mkdir()
    wb.invalidate_cache()
    assert wb.list_folders(limit=3) == ["a", "cases", "z_real"]


def test_the_held_cores_are_saved_with_the_job(queue, monkeypatch):
    wb, ws = queue
    wb.STATE["cfg"]["fluka_parts"] = 3
    monkeypatch.setattr(wb, "STOP_GRACE", 0.5)
    monkeypatch.setenv("FAKE_HANG", "1")
    t, job = start_job(wb)
    def saved():
        try:
            with open(os.path.join(wb.wb_dir(), "jobs.json")) as fh: return json.load(fh)[-1].get("threads")
        except (OSError, ValueError): return None
    wait_for(lambda: saved() == 3, "jobs.json does not name the 3 cores of the running job")
    wb.stop_current(); t.join(30)
    assert job["status"] == "stopped"


def test_free_cores_without_proc_count_only_processes_alive_now(tmp_path, monkeypatch):
    """The load average still counts the case that just ended (macOS has no /proc): ps does not."""
    from opentopas_workbench import server as wb
    monkeypatch.setattr(wb, "PROC_DIR", str(tmp_path / "no_proc"))
    monkeypatch.setattr(wb, "usable_cores", lambda: 8)
    monkeypatch.setattr(wb.os, "getloadavg", lambda: (16.0, 16.0, 16.0), raising=False)
    bin_ = tmp_path / "bin"; bin_.mkdir()
    (bin_ / "ps").write_text("#!/bin/sh\nprintf ' 250.0\\n  50,0\\n   0.0\\n'\n"); (bin_ / "ps").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_))
    assert wb.free_cores() == 5
    (bin_ / "ps").write_text("#!/bin/sh\nexit 1\n")
    assert wb.free_cores() == 8


def test_shut_down_ignores_a_second_signal(queue, monkeypatch):
    """A second SIGTERM, SIGHUP or Ctrl+C during the last end_run would skip its SIGKILL and leave sub-runs alive."""
    import signal, subprocess
    wb, ws = queue
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    def on_term(*_): raise KeyboardInterrupt
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    ended = []
    def end_run(pgid, mark):
        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT): os.kill(os.getpid(), s)
        time.sleep(0.2)
        ended.append((pgid, mark))
    try:
        for s in old: signal.signal(s, on_term)
        monkeypatch.setitem(wb.STATE, "closing", False)
        monkeypatch.setitem(wb.STATE, "proc", p); monkeypatch.setitem(wb.STATE, "mark", "m")
        wb.STATE["jobs"].append({"id": 1, "status": "running"})
        monkeypatch.setattr(wb, "end_run", end_run)
        wb.shut_down()
        assert ended == [(p.pid, "m")]
        # nothing starts while the server ends: a job would inherit the ignored signals and outlive it
        wb.STATE["jobs"][-1]["status"] = "stopped"; monkeypatch.setitem(wb.STATE, "proc", None)
        wb.enqueue(["cases/fk_p063_rifi0.inp"], "", "", False); wb.release_jobs(); wb.worker_step()
        job = wb.STATE["jobs"][-1]
        assert job["status"] == "running" and "end" not in job and wb.STATE["proc"] is None   # left for the next start
        assert not list(ws.glob("cases/*_fluka_runs"))
    finally:
        for s, h in old.items(): signal.signal(s, h)
        p.kill(); p.wait()


def test_fluka_version_comes_from_the_name_of_its_folder_only(queue):
    wb, ws = queue
    for d, v in [("/home/x/fluka/fluka4-5.2/bin", "4-5.2"), ("C:\\FLUKA\\fluka_4.4.1", "4.4.1"), ("/opt/fluka/bin", None),
                 ("/tmp/test_fluka_2/fluka_bin", None)]:
        wb.STATE["cfg"]["fluka_dir"] = d
        assert wb.fluka_version() == v, d


def test_code_of_a_result_comes_from_its_head_and_its_case(queue):
    wb, ws = queue
    head = "# TOPAS Version: {}\n# Parameter File: /elsewhere/c.txt\n# Results for scorer: IDD_c\n# Scored in component: Phantom\n" \
           "# X in 1 bin  of 20 cm\n# Y in 1 bin  of 20 cm\n# Z in 3 bins of 0.1 cm\n# DoseToMedium ( Gy ) : Sum\n0, 0, 0, 1\n0, 0, 1, 2\n0, 0, 2, 1\n"
    d = ws / "codes"; d.mkdir()
    (d / "c.txt").write_text("i:Ts/Seed = 7\ni:So/Demo/NumberOfHistoriesInRun = 30000\ns:Sc/IDD_c/Quantity = \"DoseToMedium\"\n"
                             "s:Sc/IDD_c/Component = \"Phantom\"\ns:Sc/IDD_c/OutputFile = \"IDD_c\"\ns:So/Demo/BeamParticle = \"proton\"\n"
                             "d:So/Demo/BeamEnergy = 100 MeV\n")
    (d / "IDD_c.csv").write_text(head.format("4.3"))
    (d / "old.csv").write_text(head.format("3.9").replace("IDD_c", "old"))
    (d / "m.csv").write_text("0 1\n1 2\n")
    assert wb.result_code(str(d / "IDD_c.csv"))["code"] == "OpenTOPAS"
    assert wb.result_code(str(d / "old.csv"))["code"] == "TOPAS" and wb.result_code(str(d / "m.csv")) is None
    items = {i["name"]: i for i in wb.build_index("")}
    c = items["IDD_c.csv"]["code"]
    assert (c["code"], c["version"], c["histories"], c["seed"]) == ("OpenTOPAS", "4.3", 30000, 7)
    assert items["old.csv"]["code"]["histories"] is None and "code" not in items["m.csv"]
    assert wb.with_case(wb.result_code(str(d / "IDD_c.csv")), wb.producer_of("codes/IDD_c.csv"))["seed"] == 7


# ------------------------------------------------------------------ FLUKA as a program in the Runs tab (task 3375, point 1)
def test_rifi_topas_file_reads_back_to_the_fluka_input_of_its_case():
    """fluka_from_topas gives byte for byte what generate_fluka writes, also for a prefix that looks like a case tag and
    a period other than the default, for every ion; and the reason for each case without a 1:1 translation."""
    v = dict(CTRL, ions=["p", "a", "c"], energies="63, 132.5", variants=["rifi0", "rifi1", "t4", "stk4", "stp4", "pin4", "stp4n15"],
             period=2.0, prefix="fk_p063_x", histories=1000)
    inps = dict(RF.generate_fluka(v))
    n_ok = 0
    for name, text in RF.generate(v):
        if not name.endswith(".txt"): continue
        inp = name[:-4] + ".inp"
        if inp in inps:
            assert RF.fluka_from_topas(text) == inps[inp], name; n_ok += 1
        else:
            with pytest.raises(ValueError, match="rifi/v1|DPMJET"): RF.fluka_from_topas(text)
    assert n_ok == 3 * 2 * 5 - 5    # carbon at 132.5 MeV/u is above the BME limit for each of the five plain variants


def test_edited_rifi_topas_file_is_read_as_it_is_or_refused():
    text = dict(RF.generate(dict(CTRL, variants=["t4"], energies="63")))["fk_p063_t4.txt"]
    more = text.replace("NumberOfHistoriesInRun   = 200000", "NumberOfHistoriesInRun   = 400000")
    assert more != text and FK.read_case(RF.fluka_from_topas(more))["beam"]["histories"] == 400000
    for bad, why in [(text.replace("d:Ge/Phantom/HLX      = 10.0 cm", "d:Ge/Phantom/HLX      = 5.0 cm"), "differs"),
                     (text + "d:Ge/Phantom/HLY = 5 cm\n", "was added after"),
                     (text.replace("Generated by OpenTOPAS Workbench (ripple_filter)", "Written by hand"), "header")]:
        assert bad != text
        with pytest.raises(ValueError, match=why): RF.fluka_from_topas(bad)
    mon = dict(RF.generate(dict(CTRL, variants=["t4"], energies="63", mon_pos="80")))["fk_p063_t4.txt"]
    assert "Ge/Monitors" in mon and RF.fluka_from_topas(mon) == RF.fluka_from_topas(text)


def test_topas_file_sent_to_fluka_queues_the_input_of_its_case_on_hold(queue):
    wb, ws = queue
    for name, text in RF.generate(dict(CTRL, variants=["t4"], energies="63")): (ws / "cases" / name).write_text(text)
    (ws / "cases" / "other.txt").write_text("s:Ge/World/Material = \"G4_AIR\"\n")
    (ws / "cases" / "pb.txt").write_text("# Proton 100 MeV/u. Generated by OpenTOPAS Workbench (pencil_beam).\ns:Ge/World/Material = \"G4_AIR\"\n")
    added, skipped, refused = wb.enqueue(["cases/other.txt", "cases/fk_p063_t4.txt", "cases/pb.txt"], "FLUKA", "", False)
    assert added == ["cases/fk_p063_t4.inp"] and skipped == []
    assert [r["file"] for r in refused] == ["cases/other.txt", "cases/pb.txt"]
    assert all("no FLUKA translation" in r["reason"] and "Ge/World/HLX is missing" in r["reason"] for r in refused)
    inp = (ws / "cases" / "fk_p063_t4.inp").read_text()
    assert inp == dict(RF.generate_fluka(dict(CTRL, variants=["t4"], energies="63")))["fk_p063_t4.inp"]
    j = wb.STATE["jobs"][0]
    assert (j["binary"], j["status"], j["source"], j["total"]) == ("FLUKA", "held", "cases/fk_p063_t4.txt", 200000)
    wb.worker_step()
    assert j["status"] == "held"                            # only the Start button starts FLUKA
    assert wb.enqueue(["cases/fk_p063_t4.txt"], "FLUKA", "", False)[:2] == ([], ["cases/fk_p063_t4.inp"])   # waits already
    assert len(wb.STATE["jobs"]) == 1
    j["status"] = "done"
    assert wb.enqueue(["cases/fk_p063_t4.txt"], "FLUKA", "", False)[0] == ["cases/fk_p063_t4.inp"]   # same input again
    (ws / "cases" / "fk_p063_t4.inp").write_text(inp.replace("RANDOMIZ", "RANDOMIZ ", 1))
    a, _, r = wb.enqueue(["cases/fk_p063_t4.txt"], "FLUKA", "", False)
    assert a == [] and "exists with other content" in r[0]["reason"]
    assert (ws / "cases" / "fk_p063_t4.inp").read_text() != inp   # an edited input is never overwritten


def test_unwritable_fluka_input_is_refused_and_the_rest_still_queued(queue):
    wb, ws = queue
    (ws / "ro").mkdir()
    for name, text in RF.generate(dict(CTRL, variants=["t4"], energies="63")):
        (ws / "ro" / name).write_text(text); (ws / "cases" / name).write_text(text)
    (ws / "ro").chmod(0o500)
    try:
        added, skipped, refused = wb.enqueue(["cases/fk_p063_t4.txt", "ro/fk_p063_t4.txt"], "FLUKA", "", False)
    finally: (ws / "ro").chmod(0o700)
    assert added == ["cases/fk_p063_t4.inp"] and skipped == [] and [r["file"] for r in refused] == ["ro/fk_p063_t4.txt"]
    assert not (ws / "ro" / "fk_p063_t4.inp").exists()
    assert [j["file"] for j in json.loads((ws / ".workbench" / "jobs.json").read_text())] == ["cases/fk_p063_t4.inp"]


def test_fluka_inputs_are_listed_with_their_results(queue):
    wb, ws = queue
    for name, text in RF.generate(dict(CTRL, variants=["t4", "rifi0"], energies="63")): (ws / "cases" / name).write_text(text)
    (ws / "cases" / "elsewhere.inp").write_text("TITLE\nnot from the workbench\nSTOP\n")
    wb.enqueue(["cases/fk_p063_t4.txt"], "FLUKA", "", False)
    (ws / "cases" / "out").mkdir(); (ws / "cases" / "out" / "IDD_fk_p063_t4_fluka.csv").write_text("x")
    wb.invalidate_cache()
    p = wb.list_params()
    assert {"cases/fk_p063_t4.inp", "cases/elsewhere.inp", "cases/fk_p063_t4.txt", "cases/fk_p063_rifi0.txt"} <= set(p["files"])
    assert p["results"].get("cases/fk_p063_t4.inp") == ["out"] and "cases/elsewhere.inp" not in p["results"]
    assert "cases/fk_p063_t4.txt" not in p["results"]       # the TOPAS result is another file
    assert p["last"]["cases/fk_p063_t4.txt|FLUKA"]["status"] == "held" == p["last"]["cases/fk_p063_t4.inp|FLUKA"]["status"]


# ------------------------------------------------------------------ compute in both codes, in turns (task 3375, point 4)
def both_cases(ws, variants=("t4", "rifi0")):
    fixed = ws / "cases" / "fk_p063_rifi0.inp"   # the fixture's hand-made input: kept as alone.inp, out of the generator's way
    if fixed.exists(): fixed.rename(ws / "cases" / "alone.inp")
    for name, text in RF.generate(dict(CTRL, variants=list(variants), energies="63")): (ws / "cases" / name).write_text(text)
    return [f"cases/fk_p063_{v}.txt" for v in variants]


def fake_runs(wb, monkeypatch):
    """run_job without TOPAS or FLUKA: the order of the runs, and per run the CSV that the real code would write."""
    order = []
    def run(job):
        order.append((job["file"], job["binary"]))
        out = os.path.splitext(job["file"])[0] + ("_fluka.csv" if job["binary"] == "FLUKA" else ".csv")
        job.update(status="done", outputs=[out], end="now")
    monkeypatch.setattr(wb, "run_job", run)
    return order


def test_both_codes_queue_each_case_as_a_pair_in_turns(queue):
    wb, ws = queue
    t4, r0 = both_cases(ws)
    (ws / "cases" / "other.txt").write_text("s:Ge/World/Material = \"G4_AIR\"\n")
    wb.STATE["next_id"] = 5
    added, skipped, refused, group = wb.enqueue_both([t4, "cases/other.txt", "cases/fk_p063_rifi0.inp", r0], "OpenTOPAS", "", False)
    assert [(j["file"], j["binary"], j["status"]) for j in wb.STATE["jobs"]] == [
        (t4, "OpenTOPAS", "queued"), ("cases/fk_p063_t4.inp", "FLUKA", "held"),
        (r0, "OpenTOPAS", "queued"), ("cases/fk_p063_rifi0.inp", "FLUKA", "held")]
    assert group == 5 and {j["group"] for j in wb.STATE["jobs"]} == {5} and skipped == []
    assert [r["file"] for r in refused] == ["cases/other.txt", "cases/fk_p063_rifi0.inp"]
    assert "no FLUKA translation" in refused[0]["reason"] and "no TOPAS file" in refused[1]["reason"]
    for bad in ("FLUKA", "", None):
        with pytest.raises(ValueError, match="TOPAS program"): wb.enqueue_both([t4], bad, "", False)


def test_topas_runs_past_held_fluka_and_start_comparison_keeps_the_group_order(queue, monkeypatch):
    wb, ws = queue
    t4, r0 = both_cases(ws)
    order = fake_runs(wb, monkeypatch)
    wb.enqueue(["cases/alone.inp"], "", "", False)            # a FLUKA job of no group, held before the group came
    group = wb.enqueue_both([t4, r0], "OpenTOPAS", "", False)[3]
    for _ in range(4): wb.worker_step()
    assert order == [(t4, "OpenTOPAS"), (r0, "OpenTOPAS")]     # the queue does not stand in front of a held job
    assert wb.release_jobs(group=group) == 2
    for _ in range(4): wb.worker_step()
    assert order[2:] == [("cases/fk_p063_t4.inp", "FLUKA"), ("cases/fk_p063_rifi0.inp", "FLUKA")]
    assert wb.STATE["jobs"][0]["status"] == "held"             # Start comparison releases its own group only


def test_started_right_away_the_codes_take_turns(queue, monkeypatch):
    wb, ws = queue
    t4, r0 = both_cases(ws)
    order = fake_runs(wb, monkeypatch)
    wb.release_jobs(group=wb.enqueue_both([t4, r0], "OpenTOPAS", "", False)[3])
    for _ in range(5): wb.worker_step()
    assert [b for _, b in order] == ["OpenTOPAS", "FLUKA", "OpenTOPAS", "FLUKA"]
    assert [f for f, _ in order] == [t4, "cases/fk_p063_t4.inp", r0, "cases/fk_p063_rifi0.inp"]


def test_a_pair_is_skipped_whole_never_by_half(queue):
    wb, ws = queue
    t4, r0 = both_cases(ws)
    for name in ("IDD_fk_p063_t4.csv", "IDD_fk_p063_t4_fluka.csv", "IDD_fk_p063_rifi0.csv"): (ws / "cases" / name).write_text("x")
    wb.invalidate_cache()
    added, skipped, _, _ = wb.enqueue_both([t4, r0], "OpenTOPAS", "", True)
    assert skipped == [t4] and added == [r0, "cases/fk_p063_rifi0.inp"]   # rifi0 has its TOPAS result only: both run
    added, skipped, _, _ = wb.enqueue_both([r0], "OpenTOPAS", "", False)
    assert added == [] and skipped == [r0]                    # its FLUKA half waits already


def test_groups_in_the_api_count_finished_pairs_and_release_by_group(queue, monkeypatch):
    wb, ws = queue
    t4, r0 = both_cases(ws)
    order = fake_runs(wb, monkeypatch)
    srv = wb.Server(("127.0.0.1", 0), wb.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    def api(path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}{path}",
                                     None if body is None else json.dumps(body).encode(),
                                     {"X-Token": wb.TOKEN, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r: return json.load(r)
    try:
        r = api("/api/enqueue_both", {"files": [t4, r0], "binary": "OpenTOPAS", "subdir": "", "skip": False})
        g = str(r["group"])
        assert r["added"] == [t4, "cases/fk_p063_t4.inp", r0, "cases/fk_p063_rifi0.inp"]
        assert api("/api/jobs")["groups"][g] == {"held": 2, "jobs": 4, "cases": 2, "pairs": 0, "active": 4, "outputs": []}
        wb.worker_step()
        assert api("/api/jobs")["groups"][g]["outputs"] == []    # one half of a pair compares nothing yet
        assert api("/api/release", {"group": r["group"]})["released"] == 2
        wb.worker_step()
        grp = api("/api/jobs")["groups"][g]
        assert grp["pairs"] == 1 and grp["held"] == 0 and grp["outputs"] == ["cases/fk_p063_t4.csv", "cases/fk_p063_t4_fluka.csv"]
        wb.worker_step(); wb.worker_step()
        grp = api("/api/jobs")["groups"][g]
        assert grp["pairs"] == 2 and grp["active"] == 0 and len(grp["outputs"]) == 4 and len(order) == 4
        with pytest.raises(urllib.error.HTTPError):
            api("/api/enqueue_both", {"files": [t4], "binary": "FLUKA"})
    finally:
        srv.shutdown(); srv.server_close()


def test_two_requests_at_once_put_each_case_into_one_group_once(queue, monkeypatch):
    """A double click on Compute in both codes: the second request waits for the first, so its pairs are skipped as
    already waiting instead of putting a third job into the case and leaving the group without a comparison."""
    wb, ws = queue
    t4, r0 = both_cases(ws)
    slow = wb.fluka_input_from_topas
    monkeypatch.setattr(wb, "fluka_input_from_topas", lambda f: (time.sleep(0.2), slow(f))[1])
    res = []
    ts = [threading.Thread(target=lambda: res.append(wb.enqueue_both([t4, r0], "OpenTOPAS", "", False))) for _ in range(2)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert len(wb.STATE["jobs"]) == 4 and sorted(len(r[0]) for r in res) == [0, 4]
    assert wb.job_groups(wb.STATE["jobs"])[str(wb.STATE["jobs"][0]["group"])]["cases"] == 2


# ------------------------------------------------------------------ plain TOPAS files (generators/_topas_fluka.py)
TF = load("_topas_fluka")


def body_cards(text):
    """The cards and geometry of an input without its title and comment lines."""
    return [l for k, l in enumerate(text.splitlines()) if k > 1 and not l.startswith("*") and not l.startswith("    0    0")]


def test_generator_cases_convert_to_the_generators_own_input():
    """A file of the ripple filter generator gives, read as a plain TOPAS file, the input the generator writes for its
    case. Left out: the tilted variants, which neither writes, and helium without a filter, whose TOPAS file has a
    1 um foil for TOPAS 3.9 that the generator's FLUKA writer leaves out and the converter keeps."""
    ctrl = dict(CTRL, ions=["p", "a", "c"], energies="63", variants=list(RF.VARIANTS), histories=1000, mon_pos="80")
    txt, inp = dict(RF.generate(ctrl)), dict(RF.generate_fluka(ctrl))
    same = 0
    for name, text in txt.items():
        if not name.endswith(".txt") or name[:-4] + ".inp" not in inp: continue
        conv = TF.convert(text, name[:-4])
        if name == "fk_a063_rifi0.txt":
            assert "RPP SB001 -6 6 -6 6 -79.00005 -78.99995" in conv; continue
        assert body_cards(conv) == body_cards(inp[name[:-4] + ".inp"]), name
        same += 1
    assert same >= 30
    tilted = dict(RF.generate(dict(ctrl, ions=["p"], variants=["stp4s10n15"])))["fk_p063_stp4s10n15.txt"]
    with pytest.raises(TF.Refused) as e: TF.convert(tilted, "x")
    assert [n for n, _ in e.value.reasons] == [38, 51, 64] and "tilted" in str(e.value)


PLAIN = """# A range shifter, an aperture and a shifted ridge plate before a water phantom, written by hand.
i:Ts/NumberOfThreads = 4
i:Ts/Seed = 4711
s:Gr/View/Type = "OpenGL"
d:Ge/World/HLX = 200 mm
d:Ge/World/HLY = 0.2 m
d:Ge/World/HLZ = 60 cm
s:Ge/World/Material = "G4_AIR"
s:Ge/Water/Type = "TsBox"
s:Ge/Water/Parent = "World"
s:Ge/Water/Material = "G4_WATER"
d:Ge/Water/HLX = 5 cm
d:Ge/Water/HLY = 5 cm
d:Ge/Water/HLZ = 10 cm
d:Ge/Water/TransZ = -10 cm
d:Ge/Water/RotX = 180 deg
i:Ge/Water/ZBins = 200
s:Ge/Shifter/Type = "TsBox"
s:Ge/Shifter/Parent = "World"
s:Ge/Shifter/Material = "G4_PLEXIGLASS"
d:Ge/Shifter/HLX = 4 cm
d:Ge/Shifter/HLY = 3 cm
d:Ge/Shifter/HLZ = 1 cm
d:Ge/Shifter/TransX = 1 cm
d:Ge/Shifter/TransZ = 10 cm
s:Ge/Shifter/Color = "blue"
s:Ge/Hole/Type = "TsCylinder"
s:Ge/Hole/Parent = "World"
s:Ge/Hole/Material = "G4_PLEXIGLASS"
d:Ge/Hole/RMin = 1 cm
d:Ge/Hole/RMax = 6 cm
d:Ge/Hole/HL = 0.5 cm
d:Ge/Hole/TransZ = 20 cm
s:Ge/Plate/Type = "TsRidgeFilter"
s:Ge/Plate/Parent = "World"
s:Ge/Plate/Material = "G4_PLEXIGLASS"
d:Ge/Plate/TransZ = 30 cm
d:Ge/Plate/TransY = 0.3 mm
d:Ge/Plate/RotZ = 90 deg
dv:Ge/Plate/XPoints = 4 0 0.5 1.5 2 mm
dv:Ge/Plate/ZPoints = 4 1 3 3 1 mm
d:Ge/Plate/Width = 2 mm
d:Ge/Plate/Length = 8 cm
dv:Ge/Plate/Displacement = 5 -5 -2 1 4 7 mm
s:Ge/Mon/Type = "TsBox"
s:Ge/Mon/Parent = "World"
b:Ge/Mon/IsParallel = "True"
d:Ge/Mon/HLX = 5 cm
d:Ge/Mon/HLY = 5 cm
d:Ge/Mon/HLZ = 0.1 mm
d:Ge/Mon/TransZ = 30 cm
s:Sc/Fluence/Quantity = "Fluence"
s:Sc/Fluence/Component = "Mon"
s:Sc/Fluence/OutputType = "binary"
d:Ge/BeamPosition/TransZ = 50 cm
s:So/Demo/BeamParticle = "proton"
d:So/Demo/BeamEnergy = 0.12 GeV
u:So/Demo/BeamEnergySpread = 0.5
d:So/Demo/BeamPositionSpreadX = 4 mm
d:So/Demo/BeamPositionSpreadY = 2 mm
d:So/Demo/BeamAngularSpreadX = 2 mrad
d:So/Demo/BeamAngularSpreadY = 2 mrad
i:So/Demo/NumberOfHistoriesInRun = 30000
s:Sc/Dose/Quantity = "DoseToMedium"
s:Sc/Dose/Component = "Water"
s:Sc/Dose/OutputType = "csv"
s:Sc/Dose/OutputFile = "IDD_plain"
"""


def card_fields(text, name):
    line = next(l for l in text.splitlines() if l.startswith(name))
    return [line[10 + 10 * k:20 + 10 * k].strip() for k in range(6)] + [line[70:].strip()]


def test_plain_topas_file_with_box_cylinder_and_ridge_plate():
    text = TF.convert(PLAIN, "plain")
    rec = FK.read_case(text)
    assert rec["score"]["name"] == "IDD_plain" and rec["beam"]["histories"] == 30000 and rec["phantom"]["zbins"] == 200
    assert "Left out, in a parallel world without matter: Mon." in text
    b = card_fields(text, "BEAM")
    assert float(b[0]) == pytest.approx(-0.12) and b[6] == "PROTON"
    assert float(b[2]) == pytest.approx(-FK.FWHM * 2) and float(b[3]) == pytest.approx(-FK.FWHM * 0.4)
    assert float(b[4]) == pytest.approx(-FK.FWHM * 0.2)
    assert float(card_fields(text, "BEAMPOS")[2]) == -50.0 and float(card_fields(text, "RANDOMIZ")[1]) == 4711
    bodies, regions = geometry(text)
    rnd = random.Random(3)
    pts = [(rnd.uniform(-19.9, 19.9), rnd.uniform(-19.9, 19.9), rnd.uniform(-59.9, 59.9)) for _ in range(3000)]
    pts += [(rnd.uniform(-6, 6), rnd.uniform(-6, 6), rnd.uniform(-30.4, -29.6)) for _ in range(3000)]   # the plate
    count, last = where(bodies, regions, pts)
    assert all(count == 1) and "BLKHOLE" not in set(last)
    # FLUKA frame: x and z turned. Shifter at TOPAS x = 1 cm, z = 10 cm; the aperture ring between 1 and 6 cm.
    probe = [(-1.0 - 3.9, 0, -10), (-1.0 + 3.9, 0, -10), (0, 0, -20), (3.5, 0, -20), (0, 6.5, -20), (0, 0, 5)]
    _, at = where(bodies, regions, probe)
    assert list(at) == ["SOL001", "SOL001", "AIR", "SOL002", "AIR", "PHANTOM"]
    # Ridge plate turned by 90 deg: its profile runs along TOPAS y. Ridge k stands centred on -5 + 3 k mm plus the
    # 0.3 mm TransY; it rises from 1 to 3 mm over 0.5 mm and has 1 mm gaps of air between the ridges.
    for y_mm, want in [(-5.3 + 0.3, 3.0), (-5.0 + 0.3 + 0.75, 1.0 + 2 * 0.25 / 0.5), (-3.5 + 0.3, 0.0), (7.0 + 0.3, 3.0),
                       (8.5 + 0.3, 0.0)]:
        zs = np.linspace(-30.0 - 0.15 + 1e-5, -30.0 + 0.15 - 1e-5, 3001)
        _, at = where(bodies, regions, [(0.7, y_mm / 10, z) for z in zs])
        assert (at == "RIFI1").sum() * (0.3 / 3000) * 10 == pytest.approx(want, abs=0.002), y_mm


@pytest.mark.parametrize("edit,marker,why", [
    (("d:Ge/Plate/RotZ = 90 deg", "d:Ge/Plate/RotY = 10 mrad"), "Ge/Plate/RotY", "tilted plates"),
    (("i:Ts/Seed = 4711", "i:Ts/Seed = 4711\ns:Ph/ListName = \"Mine\""), "Ph/ListName", "not translated"),
    (("d:Ge/Shifter/HLZ = 1 cm", "d:Ge/Shifter/HLZ = Ge/Hole/HL cm"), "Ge/Shifter/HLZ", "expressions"),
    (("i:Ts/Seed = 4711", "includeFile = more.txt"), "includeFile", "includeFile"),
    (('s:So/Demo/BeamParticle = "proton"', 's:So/Demo/BeamParticle = "e-"'), "BeamParticle", "only proton"),
    (("d:So/Demo/BeamAngularSpreadY = 2 mrad", "d:So/Demo/BeamAngularSpreadY = 3 mrad"), "BeamAngularSpreadY", "one divergence"),
    (("i:Ge/Water/ZBins = 200", "i:Ge/Water/ZBins = 200\ni:Ge/Water/XBins = 10"), "Ge/Water/XBins", "depth grid"),
    (('s:Ge/Hole/Material = "G4_PLEXIGLASS"', 's:Ge/Hole/Material = "G4_Pb"'), "Ge/Hole/Material", "no FLUKA definition"),
    (("d:Ge/Shifter/TransZ = 10 cm", "d:Ge/Shifter/TransZ = 0.5 cm"), "Ge/Shifter/Type", "overlaps Water"),
    (('s:Sc/Dose/Quantity = "DoseToMedium"', 's:Sc/Dose/Quantity = "DoseToWater"'), "Sc/Dose/Quantity", "only DoseToMedium"),
    (("d:Ge/Shifter/HLY = 3 cm", "d:Ge/Shifter/HLY = 3 cm\nd:Ge/Shifter/HLY = 4 cm"), "HLY = 4 cm", "set again"),
    (("5 -5 -2 1 4 7 mm", "5 -5 -2 1 4 8 mm"), "Displacement", "evenly spaced"),
    (('s:Sc/Dose/OutputType = "csv"', 's:Sc/Dose/OutputType = "root"'), "Sc/Dose/OutputType", "must be csv"),
    (("d:Ge/Water/RotX = 180 deg", "d:Ge/Water/RotX = 0 deg"), "Ge/Water/RotX", "bin 0 is the entrance"),
    (("d:Ge/Hole/RMax = 6 cm", "d:Ge/Hole/RMax = 6 cm\ns:So/Demo/BeamPositionDistribution = \"Flat\""), "BeamPositionDistribution", "Gaussian"),
])
def test_plain_topas_file_refused_with_its_line(edit, marker, why):
    text = PLAIN.replace(*edit)
    assert text != PLAIN
    line = next(n for n, l in enumerate(text.splitlines(), 1) if marker in l)
    with pytest.raises(TF.Refused) as e: TF.convert(text, "plain")
    assert any(n == line and why in r for n, r in e.value.reasons), e.value.reasons
    assert f"line {line}: " in str(e.value)


def test_plain_topas_file_queues_for_fluka_and_its_result_pairs_with_topas(queue):
    wb, ws = queue
    (ws / "cases" / "plain.txt").write_text(PLAIN)
    (ws / "cases" / "tilted.txt").write_text(PLAIN.replace("d:Ge/Plate/RotZ = 90 deg", "d:Ge/Plate/RotY = 10 mrad"))
    added, skipped, refused = wb.enqueue(["cases/plain.txt", "cases/tilted.txt"], "FLUKA", "", False)
    assert added == ["cases/plain.inp"] and [r["file"] for r in refused] == ["cases/tilted.txt"]
    assert "line 39: Ge/Plate/RotY: tilted plates" in refused[0]["reason"]
    assert (ws / "cases" / "plain.inp").read_text() == TF.convert(PLAIN, "plain")
    assert wb.STATE["jobs"][0]["status"] == "held" and wb.STATE["jobs"][0]["total"] == 30000
    assert wb.file_info(str(ws / "cases" / "plain.inp"))["outs"] == ["IDD_plain_fluka"]
    pb = "# Proton 100 MeV/u. Generated by OpenTOPAS Workbench (pencil_beam).\n" + PLAIN.split("\n", 1)[1]
    (ws / "cases" / "pb.txt").write_text(pb)
    assert wb.enqueue(["cases/pb.txt"], "FLUKA", "", False)[0] == ["cases/pb.inp"]   # no writer of its own: converted
    bad = dict(RF.generate(dict(CTRL, variants=["t4"], energies="63")))["fk_p063_t4.txt"]
    (ws / "cases" / "edited.txt").write_text(bad.replace("d:Ge/RiFi1/RotZ     = 0 deg", "d:Ge/RiFi1/RotZ     = 0 deg\nd:Ge/RiFi1/RotY = 1 mrad"))
    _, _, refused = wb.enqueue(["cases/edited.txt"], "FLUKA", "", False)
    assert refused[0]["reason"].startswith("the generator ripple_filter: ") and "as a plain TOPAS file, " in refused[0]["reason"]
