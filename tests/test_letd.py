"""The tab "LETd on MRI": paths as typed, paths found in the viewer's folder, the checks before a start, the run with
its steps, and the result on the page. Synthetic folders only; the last test runs the real run.sh on phantom.py data.

Requirements tested: one form is enough (cacheF and Verlauf-Daten are found from the MRI folder, build_page.py comes
from the viewer's own clone, the values come back on the next visit), a wrong path is named before anything runs, results never go to workspace/offen,
the page sees the steps while it runs and the exams and Verlauf.html after it.
"""
import json
import os
import stat
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from opentopas_workbench import letd, server as wb

REPO_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "mrt_overlay", "run.sh")


@pytest.fixture(autouse=True)
def script_of_the_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(letd, "VIEWER_DIR_DEFAULT", str(tmp_path / "mrt-viewer"))
    # CI installs the package into site-packages, where tools/ does not come along; the tab runs from a checkout
    monkeypatch.setattr(letd, "SCRIPT", REPO_SCRIPT)


@pytest.fixture
def ws(tmp_path):
    wb.STATE["root"] = str(tmp_path / "ws")
    os.makedirs(wb.STATE["root"])
    wb.STATE["cfg"] = {"binaries": [], "lang": "en"}
    wb.STATE["jobs"] = []
    letd.JOB.update(proc=None, thread=None, stop=False, state=None)
    yield tmp_path
    t = letd.JOB["thread"]
    if t is not None: t.join(30)


@pytest.fixture
def srv(ws):
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


def touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh: fh.write(text)
    return str(path)


def viewer_tree(top):
    """The layout of a case: Analyse/Skripte (case files and an old build_page.py), Analyse/Verlauf-Daten, the caches."""
    touch(top / "Analyse" / "Skripte" / "build_page.py", "")
    os.makedirs(top / "Analyse" / "Verlauf-Daten")
    touch(top / "Analyse" / "cacheF" / "log.json", "{}")
    touch(top / "Analyse" / "cacheF" / "ref_t1.nii.gz")
    for n in ("viewer.json", "befunde.json"): touch(top / "Analyse" / "Skripte" / n, "{}")
    for n in ("log.json", "vent.json"): touch(top / "Analyse" / "cacheG" / n, "{}")
    return top


def full_form(tmp):
    """A form whose every path exists."""
    run = tmp / "run"
    for n in ("letd.npy", "rbe_dose_1_1.npy", "rbe_dose_1_1.dcm"): touch(run / n)
    touch(tmp / "ct" / "CT1.dcm")
    top = viewer_tree(tmp / "MRT")
    touch(tmp / "mrt-viewer" / "build_page.py")
    return {"mrt": str(top), "run": str(run), "ct": str(tmp / "ct"), "rs": "", "cache": str(top / "Analyse" / "cacheF"),
            "view": str(top / "Analyse" / "Verlauf-Daten"), "rx": "54", "fx": "30", "out": str(tmp / "out"), "kz": str(tmp / "kz" / "kennzahlen.json"),
            "build": "1", "synthseg": "0"}


# ------------------------------------------------------------------ paths
def test_path_as_typed_drops_quotes_and_maps_windows_drive():
    assert letd.to_local('  "/data/a b/run"  ') == "/data/a b/run"
    assert letd.to_local("") == ""
    if os.path.isdir("/mnt"):   # WSL and most Linux: a drive path becomes its mount
        assert letd.to_local(r'"C:\Users\x y\MRT Bilder\"') == "/mnt/c/Users/x y/MRT Bilder"
    assert letd.to_windows("/mnt/c/Users/x/Verlauf.html") == r"C:\Users\x\Verlauf.html"
    assert letd.to_windows("/home/x") == "/home/x"


def test_public_zone_is_refused_wherever_the_workspace_lies():
    assert letd.in_public_zone("/mnt/c/dev/w/workspace/offen/letd")
    assert letd.in_public_zone("/x/Workspace/Offen")
    assert not letd.in_public_zone("/x/workspace/offenbar")
    assert not letd.in_public_zone("/x/offen")


def test_derive_finds_the_two_viewer_paths_and_no_old_build_page(tmp_path):
    top = viewer_tree(tmp_path / "MRT Bilder")
    # a second, deeper cacheF (an old copy) must lose against the nearer one
    touch(top / "Analyse" / "alt" / "x" / "cacheF" / "log.json"); touch(top / "Analyse" / "alt" / "x" / "cacheF" / "ref_t1.nii.gz")
    got = letd.derive(f'"{top}"')
    assert got == {"cache": str(top / "Analyse" / "cacheF"), "view": str(top / "Analyse" / "Verlauf-Daten")}
    assert letd.derive(str(tmp_path / "nothing")) == {"error": "not_found"}
    assert letd.derive(str(tmp_path / "MRT Bilder" / "Analyse" / "Skripte")) == {}


def test_default_out_uses_the_private_zones_only_when_they_exist(tmp_path):
    assert letd.default_out(str(tmp_path)) == {}
    base = tmp_path / "workspace"
    os.makedirs(base / "lokal"); os.makedirs(base / "robin")
    got = letd.default_out(str(tmp_path), today="2026-11-21")
    assert got["out"] == os.path.join(str(base), "lokal", "letd_mrt", "2026-11-21")
    assert got["kz"].endswith(os.path.join("letd_mrt", "kennzahlen.json")) and os.sep + "robin" + os.sep in got["kz"]
    assert letd.default_out(str(base), today="d")["out"] == os.path.join(str(base), "lokal", "letd_mrt", "d")


# ------------------------------------------------------------------ checks
def test_check_names_each_missing_field():
    bad, _ = letd.check({"fx": "x"})
    by = {m["field"]: m["msg"] for m in bad}
    for k in ("run", "ct", "cache", "view", "out", "kz"): assert by[k] == "empty"
    assert by["fx"] == "number" and by["rx"] == "rx_or_rs"


def test_check_passes_a_full_form_and_names_missing_files(tmp_path):
    f = full_form(tmp_path)
    assert letd.check(f)[0] == []
    os.remove(tmp_path / "run" / "letd.npy"); os.remove(tmp_path / "MRT" / "Analyse" / "cacheF" / "log.json")
    by = {m["field"]: m for m in letd.check(f)[0]}
    assert by["run"]["msg"] == "missing" and by["run"]["args"] == ["letd.npy"]
    assert by["cache"]["args"] == ["log.json"]


def test_check_refuses_results_in_the_public_zone(tmp_path):
    f = full_form(tmp_path)
    f["out"] = str(tmp_path / "workspace" / "offen" / "letd")
    f["kz"] = str(tmp_path / "workspace" / "offen" / "k.json")
    by = {m["field"]: m["msg"] for m in letd.check(f)[0]}
    assert by == {"out": "public_zone", "kz": "public_zone"}


def test_check_needs_build_page_only_when_the_page_is_rebuilt(tmp_path):
    f = full_form(tmp_path)
    bad = letd.check(f, viewer=str(tmp_path / "gone"))[0]
    assert bad == [{"field": "build", "msg": "no_viewer", "args": [str(tmp_path / "gone" / "build_page.py")]}]
    os.remove(tmp_path / "MRT" / "Analyse" / "Skripte" / "viewer.json")
    bad = letd.check(f)[0]
    assert [(m["field"], m["msg"]) for m in bad] == [("build", "no_case")] and "viewer.json" in bad[0]["args"][0]
    f["build"] = "0"
    assert letd.check(f)[0] == []


# ------------------------------------------------------------------ progress
def test_progress_from_the_step_lines():
    lines = ["## STEP export", "## STEP register", "## EXAM 2/5"]
    steps, exam = letd._progress(lines, build=True, synthseg=False)
    assert steps == {"check": "done", "export": "done", "synthseg": "skip", "register": "done", "exams": "now",
                     "write": "todo", "page": "todo"}
    assert exam == [2, 5]
    steps, _ = letd._progress([], build=False, synthseg=True)
    assert steps["page"] == "skip" and steps["export"] == "todo"


# ------------------------------------------------------------------ the run through the server
FAKE_RUN = """#!/usr/bin/env bash
# stands in for tools/mrt_overlay/run.sh: the same arguments, the same progress lines, an overlay.json
OUT=$8
echo "## STEP export"; echo "## STEP register"
for i in 1 2; do echo "## EXAM $i/2"; done
echo "## STEP write"
mkdir -p "$OUT/viewer_grid"
printf '%s' '{"registration": {"spread_rms_mm": 0.3}, "prescription_gy_rbe": 54.0, "prescription_from": "angegeben",
 "exams": [{"track": "t1", "day": "2020-01-01", "native_mm": [1, 1, 1], "suitable": true, "check_rms_mm": 0.4, "check_ok": true}],
 "look_list": [{"volume_ml": 1.5}], "hinweis": "Nur Anzeige"}' > "$OUT/viewer_grid/overlay.json"
printf '%s\\n' "$@" > "$OUT/args.txt"
echo "SYNTHSEG=${SYNTHSEG:-}" >> "$OUT/args.txt"
"""


def wait_done(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = call(url + "/api/letd")[1]["job"]
        if not job.get("running") and job.get("status") != "running": return job
        time.sleep(0.2)
    raise AssertionError("run did not end")


def test_start_runs_the_script_then_build_page_and_shows_the_result(srv, ws, monkeypatch):
    fake = ws / "fake_run.sh"; fake.write_text(FAKE_RUN); fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(letd, "SCRIPT", str(fake))
    monkeypatch.setattr(letd.shutil, "which", lambda n: sys.executable if n == "python3" else None)
    f = full_form(ws)
    # build_page.py of the viewer's clone writes Verlauf.html next to the data folder it is given
    touch(ws / "mrt-viewer" / "build_page.py", "import os, sys\na = sys.argv\nassert os.path.isdir(a[a.index('--cache') + 1])\n"
          "open(os.path.join(os.path.dirname(a[a.index('--view') + 1]), 'Verlauf.html'), 'w').write('<html>')\nprint('built')\n")
    code, r = call(srv + "/api/letd/start", {"form": f})
    assert code == 200 and not r.get("problems"), r
    job = wait_done(srv)
    assert job["status"] == "done", job["log"]
    assert set(job["steps"].values()) == {"done", "skip"} and job["steps"]["page"] == "done"
    assert job["result"]["exams"][0]["check_ok"] is True and job["result"]["look_list"]
    assert job["html"] == str(ws / "MRT" / "Analyse" / "Verlauf.html")
    assert "built" in job["log"] and "## STEP page" in job["log"]
    args = (ws / "out" / "args.txt").read_text().splitlines()
    assert args[:5] == [f["run"], f["ct"], "", f["cache"], f["view"]] and args[5:9] == ["54.0", "30", f["out"], f["kz"]]
    assert args[9] == "SYNTHSEG=none"   # the box was not ticked
    # the log of the run lies in the output folder, not in the workspace
    assert (ws / "out" / letd.LOG_NAME).exists() and not os.path.exists(os.path.join(wb.wb_dir(), letd.LOG_NAME))
    # the next visit starts from the same values and the last result, also after a server restart
    saved = wb.STATE["cfg"]["letd"]
    assert saved["form"]["run"] == f["run"] and saved["last"]["status"] == "done"
    letd.JOB.update(state=None, thread=None)
    info = call(srv + "/api/letd")[1]
    assert info["form"]["cache"] == f["cache"] and info["job"]["result"]["exams"]


def test_start_runs_build_page_of_the_clone_that_the_setting_names(srv, ws, monkeypatch):
    fake = ws / "fake_run.sh"; fake.write_text(FAKE_RUN); fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(letd, "SCRIPT", str(fake))
    monkeypatch.setattr(letd.shutil, "which", lambda n: sys.executable if n == "python3" else None)
    f = full_form(ws)
    for clone in ("mrt-viewer", "second clone"):
        touch(ws / clone / "build_page.py", f"print('built by {clone}')\n")
    assert call(srv + "/api/config", {"mrt_viewer_dir": str(ws / "second clone")})[0] == 200
    code, r = call(srv + "/api/letd/start", {"form": f})
    assert code == 200 and not r.get("problems"), r
    job = wait_done(srv)
    assert job["status"] == "done", job["log"]
    assert "built by second clone" in job["log"] and "built by mrt-viewer" not in job["log"]


def test_start_refuses_a_bad_form_and_runs_nothing(srv, ws):
    f = full_form(ws); f["run"] = str(ws / "missing")
    code, r = call(srv + "/api/letd/start", {"form": f})
    assert code == 200 and [m["field"] for m in r["problems"]] == ["run"]
    assert letd.JOB["thread"] is None and not (ws / "out").exists()
    # the typed values are kept even so, so a fix needs one field, not the whole form again
    assert wb.STATE["cfg"]["letd"]["form"]["ct"] == f["ct"]


def test_stop_ends_the_whole_process_group(srv, ws, monkeypatch):
    fake = ws / "slow.sh"; fake.write_text("#!/usr/bin/env bash\necho '## STEP export'\nsleep 60 & wait\n")
    monkeypatch.setattr(letd, "SCRIPT", str(fake))
    f = full_form(ws); f["build"] = "0"
    assert call(srv + "/api/letd/start", {"form": f})[0] == 200
    assert call(srv + "/api/letd/start", {"form": f})[0] == 409   # one run at a time
    for _ in range(50):
        if letd.JOB["proc"] is not None: break
        time.sleep(0.1)
    t0 = time.time()
    assert call(srv + "/api/letd/stop", {})[1]["stopped"] is True
    job = wait_done(srv, timeout=20)
    assert job["status"] == "stopped" and time.time() - t0 < 15


def test_stop_right_after_the_start_still_ends_the_run(ws, monkeypatch):
    # Stop pressed before the run's first process exists: the process must not run to its end
    fake = ws / "slow.sh"; fake.write_text("#!/usr/bin/env bash\nsleep 60 & wait\n")
    monkeypatch.setattr(letd, "SCRIPT", str(fake))
    gate = threading.Event()
    real_spawn = letd._spawn
    monkeypatch.setattr(letd, "_spawn", lambda *a, **k: (gate.wait(10), real_spawn(*a, **k))[1])
    f = full_form(ws); f["build"] = "0"
    letd.start(f)
    assert letd.stop() is True
    gate.set()
    t0 = time.time(); letd.JOB["thread"].join(20)
    assert letd.JOB["state"]["status"] == "stopped" and time.time() - t0 < 15


def test_stop_while_a_process_is_being_started_ends_it(ws):
    letd.JOB["stop"] = True
    t0 = time.time()
    with open(os.devnull, "w") as log:
        rc = letd._spawn(["bash", "-c", "sleep 60 & wait"], log, dict(os.environ))
    assert rc != 0 and time.time() - t0 < 15


def test_relative_paths_become_absolute_so_run_sh_writes_where_they_were_checked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert letd.to_local("out/x") == str(tmp_path / "out" / "x")


def test_public_zone_follows_links_and_the_workbench_root(tmp_path):
    pub = tmp_path / "workspace" / "offen"; pub.mkdir(parents=True)
    link = tmp_path / "res"; link.symlink_to(pub)
    assert letd.in_public_zone(str(link / "x"))
    root = tmp_path / "ws"; (root / "offen").mkdir(parents=True)
    assert letd.in_public_zone(str(root / "offen" / "x"), str(root))
    assert not letd.in_public_zone(str(root / "offenbar"), str(root))


def test_check_refuses_viewer_data_in_the_public_zone(tmp_path):
    f = full_form(tmp_path)
    view = tmp_path / "workspace" / "offen" / "Verlauf-Daten"; view.mkdir(parents=True)
    f["view"] = str(view)
    bad, _ = letd.check(f)
    assert {"field": "view", "msg": "public_zone", "args": []} in bad


def test_open_uses_the_page_of_the_last_run_only(srv, ws):
    other = ws / "other.html"; other.write_text("<html>")
    code, r = call(srv + "/api/letd/open", {"p": str(other)})
    assert code != 200 or not r.get("ok")   # no run yet: nothing to open, whatever the page sends


def test_derive_and_check_over_http(srv, ws):
    top = viewer_tree(ws / "MRT")
    r = call(srv + "/api/letd/derive", {"mrt": str(top)})[1]
    assert r["view"].endswith("Verlauf-Daten")
    r = call(srv + "/api/letd/check", {"form": {}})[1]
    assert {m["field"] for m in r["problems"]} >= {"run", "ct", "cache"}


# ------------------------------------------------------------------ the viewer's own repository
def test_default_viewer_is_a_clone_next_to_the_workbench(monkeypatch):
    monkeypatch.undo()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(letd.__file__)))
    assert letd.VIEWER_DIR_DEFAULT == os.path.join(os.path.dirname(repo), "mrt-viewer")


def test_page_command_takes_windows_python_only_when_it_reaches_every_path(monkeypatch):
    monkeypatch.setattr(letd.shutil, "which", lambda n: "/bin/" + n)
    page = "/mnt/c/dev/mrt-viewer/build_page.py"
    got = letd.page_command(page, ("--cache", "/mnt/c/w/cacheF", "--view", "/mnt/c/a b/Verlauf-Daten"))
    assert got == ["/bin/py.exe", "-3", r"C:\dev\mrt-viewer\build_page.py", "--cache", r"C:\w\cacheF", "--view", r"C:\a b\Verlauf-Daten"]
    assert letd.page_command(page, ("--cache", "/home/x/cacheF"))[0] == "/bin/python3"
    # a mount that is no drive letter (a network share, say) is no path that Windows Python can open
    assert letd.page_command(page, ("--cache", "/mnt/wsl/x/cacheF"))[0] == "/bin/python3"


def test_the_setting_names_the_clone_that_the_tab_runs(srv, ws):
    f = full_form(ws)
    other = ws / "elsewhere"
    assert call(srv + "/api/config", {"mrt_viewer_dir": f" {other} "})[0] == 200
    assert wb.STATE["cfg"]["mrt_viewer_dir"] == str(other)
    r = call(srv + "/api/letd/check", {"form": f})[1]
    assert [(m["field"], m["args"]) for m in r["problems"]] == [("build", [str(other / "build_page.py")])]
    touch(other / "build_page.py")
    assert call(srv + "/api/letd/check", {"form": f})[1]["problems"] == []
    assert call(srv + "/api/info")[1]["mrt_viewer_dir_default"] == letd.VIEWER_DIR_DEFAULT


# ------------------------------------------------------------------ the real run.sh on phantom data
def test_real_run_on_phantom_data(ws, monkeypatch):
    np = pytest.importorskip("numpy")
    sitk = pytest.importorskip("SimpleITK")
    pytest.importorskip("scipy"); pytest.importorskip("pydicom"); pytest.importorskip("PIL")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    from mrt_overlay import phantom as ph
    ct, mri, _, _ = ph.make()
    cache = ws / "MRT" / "cacheF"; os.makedirs(cache)
    sitk.WriteImage(mri, str(cache / "ref_t1.nii.gz"))
    a = sitk.GetArrayFromImage(mri)
    np.save(cache / "t1_2020-01-01.npy", np.clip(a / a.max() * 255, 0, 255).astype(np.uint8))
    json.dump({"t1 2020-01-01": {"series": "x", "kind": "3D", "native": [1.0, 1.0, 1.0]}}, open(cache / "log.json", "w"))
    os.makedirs(ws / "MRT" / "Verlauf-Daten")
    sitk.WriteImage(ct, str(ws / "ct.nii.gz"))
    ph.write_run(str(ws / "run"), centre_mm=(-11.0, 4.0, 24.0), sigma_mm=8.0)
    f = {"run": str(ws / "run"), "ct": str(ws / "ct.nii.gz"), "rs": "", "cache": str(cache), "view": str(ws / "MRT" / "Verlauf-Daten"),
         "rx": "2.2", "fx": "1", "out": str(ws / "out"), "kz": str(ws / "kz" / "kennzahlen.json"), "build": "0", "synthseg": "0"}
    assert letd.check(f)[0] == []
    monkeypatch.setenv("MRT_PY", sys.executable)
    letd.start(f)
    letd.JOB["thread"].join(600)
    st = letd.status()
    assert st["status"] == "done", st["log"]
    assert st["steps"]["exams"] == "done" and st["steps"]["synthseg"] == "skip"
    assert st["result"]["exams"][0]["check_ok"] and st["result"]["prescription_gy_rbe"] == 2.2
    assert (ws / "MRT" / "Verlauf-Daten" / "letd_all.js").exists()
    assert not any(isinstance(v, str) for v in json.load(open(ws / "kz" / "kennzahlen.json")).values())
