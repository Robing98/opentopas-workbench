"""The PET part of the tab "LETd/PET on MRI": the checks before a start, the run with its steps, the control sheet
and the result on the page. Synthetic folders only; the last test runs the real run_pet.sh on phantom.py data.

Requirements tested: one form is enough (the viewer's folders come from the MRI folder or the LETd form, the values
come back on the next visit), a wrong path is named before anything runs, results never go to workspace/offen, the
reference region defaults to the brain median, the page sees the steps while it runs and the MI values, the control
sheet and Verlauf.html after it, and the server serves no file but that sheet.
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

from opentopas_workbench import letd, pet_run, server as wb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_SCRIPT = os.path.join(ROOT, "tools", "mrt_overlay", "run_pet.sh")
PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
       b"\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x35\x81\x84\x00\x00\x00\x00IEND\xaeB`\x82")


@pytest.fixture(autouse=True)
def script_of_the_checkout(monkeypatch):
    monkeypatch.setattr(pet_run, "SCRIPT", REPO_SCRIPT)


@pytest.fixture
def ws(tmp_path):
    wb.STATE["root"] = str(tmp_path / "ws")
    os.makedirs(wb.STATE["root"])
    wb.STATE["cfg"] = {"binaries": [], "lang": "en"}
    wb.STATE["jobs"] = []
    pet_run.JOB.update(proc=None, thread=None, stop=False, state=None)
    yield tmp_path
    t = pet_run.JOB["thread"]
    if t is not None: t.join(30)


@pytest.fixture
def srv(ws):
    s = wb.Server(("127.0.0.1", 0), wb.H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


def call(url, body=None, raw=False):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("X-Token", wb.TOKEN); req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            data = r.read()
            return r.status, (data if raw else json.loads(data))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh: fh.write(text)
    return str(path)


def full_form(tmp):
    top = tmp / "MRT"
    touch(top / "Analyse" / "Skripte" / "build_page.py")
    os.makedirs(top / "Analyse" / "Verlauf-Daten")
    for n in ("log.json", "ref_t1.nii.gz", "mask.npy", "t1_2020-01-01.npy"): touch(top / "Analyse" / "cacheF" / n)
    touch(tmp / "pet" / "sub" / "1.dcm")
    return {"mrt": str(top), "pet": str(tmp / "pet"), "cache": str(top / "Analyse" / "cacheF"),
            "view": str(top / "Analyse" / "Verlauf-Daten"), "page": str(top / "Analyse" / "Skripte" / "build_page.py"),
            "ref": "brain", "side": "", "labels": "", "out": str(tmp / "out"), "build": "1"}


# ------------------------------------------------------------------ checks
def test_full_form_passes_and_defaults_to_the_brain_median(tmp_path):
    f = full_form(tmp_path); f["ref"] = ""
    assert pet_run.check(f) == ([], [])


def test_check_names_each_wrong_field(tmp_path):
    f = full_form(tmp_path)
    os.remove(os.path.join(f["cache"], "mask.npy"))
    f.update(pet=str(tmp_path / "nothing"), ref="contralateral", side="", page="")
    bad, _ = pet_run.check(f)
    by = {m["field"]: m["msg"] for m in bad}
    assert by == {"pet": "not_found", "cache": "missing", "side": "choice", "page": "empty"}
    f = full_form(tmp_path / "b"); f["ref"] = "striatum"
    assert {m["field"] for m in pet_run.check(f)[0]} == {"labels"}
    f["ref"] = "anything"
    assert {m["field"] for m in pet_run.check(f)[0]} == {"ref"}


def test_check_needs_a_3d_t1_and_files_in_the_pet_folder(tmp_path):
    f = full_form(tmp_path)
    os.remove(os.path.join(f["cache"], "t1_2020-01-01.npy"))
    os.remove(os.path.join(f["pet"], "sub", "1.dcm"))
    by = {m["field"]: m["msg"] for m in pet_run.check(f)[0]}
    assert by == {"cache": "no_t1", "pet": "no_dicom"}


def test_check_refuses_the_public_zone(tmp_path):
    f = full_form(tmp_path)
    pub = tmp_path / "workspace" / "offen"; (pub / "v").mkdir(parents=True)
    f.update(out=str(pub / "pet"), view=str(pub / "v"))
    assert {(m["field"], m["msg"]) for m in pet_run.check(f)[0]} == {("out", "public_zone"), ("view", "public_zone")}


def test_default_out_goes_to_eigen_else_lokal(tmp_path):
    root = tmp_path / "ws"; (root / "workspace" / "lokal").mkdir(parents=True)
    assert pet_run.default_out(str(root), "2020-01-01")["out"] == str(root / "workspace" / "lokal" / "pet_mrt" / "2020-01-01")
    (root / "workspace" / "eigen").mkdir()
    assert pet_run.default_out(str(root), "2020-01-01")["out"] == str(root / "workspace" / "eigen" / "pet_mrt" / "2020-01-01")
    assert pet_run.default_out(str(tmp_path / "none")) == {}


def test_progress_from_the_log_lines():
    lines = ["## STEP read", "## STEP register", "## START 1/27", "## START 9/27", "## STEP headfoot", "## START 12/27"]
    steps, start = pet_run.progress(lines, build=True)
    assert steps == {"check": "done", "read": "done", "register": "done", "headfoot": "now", "side": "todo",
                     "write": "todo", "page": "todo"}
    assert start == [12, 27]
    steps, start = pet_run.progress([], build=False)
    assert steps["page"] == "skip" and steps["read"] == "todo" and start is None


# ------------------------------------------------------------------ the run through the server
FAKE_RUN = r"""#!/usr/bin/env bash
# stands in for tools/mrt_overlay/run_pet.sh: the same arguments, the same progress lines, pet_<day>.json and a sheet
OUT=$4
echo "## STEP read"; echo "## STEP register"; echo "## START 1/3"; echo "## STEP headfoot"; echo "## START 2/3"
echo "## STEP side"; echo "## START 3/3"; echo "## STEP write"
cp "$SHEET_SRC" "$OUT/pet_kontrolle_2020-03-01.png"
printf '{"day": "2020-03-01", "tracer": "Phantom", "reference": "Median im Hirn", "fixed_t1": "2020-01-01",
 "registration": {"mi": -0.5, "mi_gespiegelt": -0.4, "abstand_relativ": 0.2, "mi_kopfstehend": -0.3,
 "abstand_kopfstehend_relativ": 0.4}, "hinweis": "kein SUV", "sheet": "%s"}' "$OUT/pet_kontrolle_2020-03-01.png" > "$OUT/pet_2020-03-01.json"
printf '%s\n' "$@" > "$OUT/args.txt"
echo "LABELS=${LABELS:-}" >> "$OUT/args.txt"
"""


def wait_done(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = call(url + "/api/pet")[1]["job"]
        if not job.get("running") and job.get("status") != "running": return job
        time.sleep(0.2)
    raise AssertionError("run did not end")


def fake_script(ws, monkeypatch):
    fake = ws / "fake_run_pet.sh"; fake.write_text(FAKE_RUN); fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    src = ws / "sheet.png"; src.write_bytes(PNG)
    monkeypatch.setenv("SHEET_SRC", str(src))
    monkeypatch.setattr(pet_run, "SCRIPT", str(fake))
    monkeypatch.setattr(letd.shutil, "which", lambda n: sys.executable if n == "python3" else None)


def test_start_runs_the_script_then_build_page_and_shows_the_result(srv, ws, monkeypatch):
    fake_script(ws, monkeypatch)
    f = full_form(ws)
    touch(f["page"], "import os\nopen(os.path.join('..', 'Verlauf.html'), 'w').write('<html>')\nprint('built')\n")
    code, r = call(srv + "/api/pet/start", {"form": f})
    assert code == 200 and not r.get("problems"), r
    job = wait_done(srv)
    assert job["status"] == "done", job["log"]
    assert set(job["steps"].values()) == {"done"} and job["start_n"] == [3, 3]
    res = job["result"]
    assert res["day"] == "2020-03-01" and res["sheet"] is True and res["registration"]["mi_kopfstehend"] == -0.3
    assert "sheet" not in job and str(ws) not in json.dumps(res)     # the page gets no local path of the sheet
    assert job["html"] == str(ws / "MRT" / "Analyse" / "Verlauf.html")
    args = (ws / "out" / "args.txt").read_text().splitlines()
    assert args == [f["pet"], f["cache"], f["view"], f["out"], "brain", "LABELS="]
    code, png = call(srv + "/api/pet/sheet", raw=True)
    assert code == 200 and png == PNG
    assert (ws / "out" / pet_run.LOG_NAME).exists()
    # the next visit starts from the same values and the last result, also after a server restart
    assert wb.STATE["cfg"]["pet"]["form"]["pet"] == f["pet"] and wb.STATE["cfg"]["pet"]["last"]["status"] == "done"
    pet_run.JOB.update(state=None, thread=None)
    info = call(srv + "/api/pet")[1]
    assert info["form"]["cache"] == f["cache"] and info["job"]["result"]["day"] == "2020-03-01"


def test_striatum_and_opposite_side_reach_the_script(srv, ws, monkeypatch):
    fake_script(ws, monkeypatch)
    f = full_form(ws); f.update(build="0", ref="striatum", labels=touch(ws / "seg.nii.gz"))
    assert call(srv + "/api/pet/start", {"form": f})[0] == 200
    assert wait_done(srv)["status"] == "done"
    assert (ws / "out" / "args.txt").read_text().splitlines()[4:] == ["striatum", "LABELS=" + f["labels"]]
    f.update(ref="contralateral", side="rechts", labels="")
    assert call(srv + "/api/pet/start", {"form": f})[0] == 200
    assert wait_done(srv)["status"] == "done"
    assert (ws / "out" / "args.txt").read_text().splitlines()[4:] == ["contralateral", "rechts", "LABELS="]


def test_a_failed_run_shows_its_message_and_no_old_result(srv, ws, monkeypatch):
    fake = ws / "fail.sh"
    fake.write_text("#!/usr/bin/env bash\necho '## STEP read'\necho 'Kopf-Fuß-Prüfung nicht bestanden' >&2\nexit 1\n")
    monkeypatch.setattr(pet_run, "SCRIPT", str(fake))
    f = full_form(ws); f["build"] = "0"
    os.makedirs(f["out"]); touch(os.path.join(f["out"], "pet_2019-01-01.json"), '{"day": "2019-01-01", "sheet": ""}')
    old = time.time() - 3600
    os.utime(os.path.join(f["out"], "pet_2019-01-01.json"), (old, old))
    assert call(srv + "/api/pet/start", {"form": f})[0] == 200
    job = wait_done(srv)
    assert job["status"] == "failed" and "Kopf-Fuß" in job["log"] and job["result"] is None
    assert call(srv + "/api/pet/sheet")[0] == 404


def test_start_refuses_a_bad_form_and_runs_nothing(srv, ws):
    f = full_form(ws); f["cache"] = str(ws / "missing")
    code, r = call(srv + "/api/pet/start", {"form": f})
    assert code == 200 and [m["field"] for m in r["problems"]] == ["cache"]
    assert pet_run.JOB["thread"] is None and not (ws / "out").exists()
    assert wb.STATE["cfg"]["pet"]["form"]["pet"] == f["pet"]


def test_stop_ends_the_run(srv, ws, monkeypatch):
    fake = ws / "slow.sh"; fake.write_text("#!/usr/bin/env bash\necho '## STEP read'\nsleep 60 & wait\n")
    monkeypatch.setattr(pet_run, "SCRIPT", str(fake))
    f = full_form(ws); f["build"] = "0"
    assert call(srv + "/api/pet/start", {"form": f})[0] == 200
    assert call(srv + "/api/pet/start", {"form": f})[0] == 409
    for _ in range(50):
        if pet_run.JOB["proc"] is not None: break
        time.sleep(0.1)
    assert call(srv + "/api/pet/stop", {})[1]["stopped"] is True
    assert wait_done(srv, timeout=20)["status"] == "stopped"


def test_sheet_is_only_the_run_s_own_png(srv, ws):
    assert call(srv + "/api/pet/sheet")[0] == 404                  # no run yet
    out = ws / "out"; out.mkdir()
    other = ws / "elsewhere.png"; other.write_bytes(PNG)
    touch(str(out / "pet_2020-03-01.json"), json.dumps({"day": "2020-03-01", "sheet": str(other)}))
    pet_run.JOB["state"] = {"status": "done", "out_local": str(out), "t0": time.time() - 10}
    assert call(srv + "/api/pet/sheet?p=" + str(other))[0] == 404  # a sheet outside the run's folder is not served


def test_pet_form_takes_the_viewer_folders_of_the_letd_form(srv, ws):
    wb.STATE["cfg"]["letd"] = {"form": {"mrt": "/m", "cache": "/m/c", "view": "/m/v", "page": "/m/p.py", "run": "/r"}}
    form = call(srv + "/api/pet")[1]["form"]
    assert (form["mrt"], form["cache"], form["view"], form["page"], form["ref"]) == ("/m", "/m/c", "/m/v", "/m/p.py", "brain")
    assert "run" not in form


def test_ui_has_every_pet_text_in_both_languages():
    import re
    html = open(os.path.join(ROOT, "opentopas_workbench", "ui.html"), encoding="utf-8").read()
    used = {k for k in re.findall(r'(?:\bt\(|data-i18n=)"(pt_[a-z_]+)"', html) if not k.endswith("_")}
    used |= {f"pt_f_{k}" for k in pet_run.FIELDS} | {f"pt_h_{k}" for k in pet_run.FIELDS}
    used |= {f"lt_m_{m}" for m in ("no_t1", "choice", "pet_adds", "no_dicom", "public_zone", "missing", "empty", "not_found")}
    used |= {f"pt_s_{s}" for s in pet_run.STEPS} | {f"pt_ref_{r}" for r in pet_run.REFS} | {f"pt_side_{s}" for s in pet_run.SIDES}
    for lang in ("de", "en"):
        block = "".join(re.findall(r"Object\.assign\(I18N\.%s,\{(.*?)\}\);" % lang, html, re.S))
        keys = set(re.findall(r"([a-z_0-9]+):", block))
        assert used - keys == set(), lang


# ------------------------------------------------------------------ the real run_pet.sh on phantom data
def test_real_run_on_phantom_data(ws, monkeypatch):
    np = pytest.importorskip("numpy")
    sitk = pytest.importorskip("SimpleITK")
    pytest.importorskip("scipy"); pytest.importorskip("pydicom"); pytest.importorskip("PIL")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from mrt_overlay import phantom as ph
    t1, brain = ph.t1_patient()
    cache = ws / "MRT" / "cacheF"; os.makedirs(cache)
    sitk.WriteImage(t1, str(cache / "ref_t1.nii.gz"))
    a = sitk.GetArrayFromImage(t1)
    np.save(cache / "t1_2020-01-01.npy", np.clip(a / a.max() * 255, 0, 255).astype(np.uint8))
    np.save(cache / "mask.npy", brain)
    json.dump({"t1 2020-01-01": {"series": "x", "kind": "3D", "native": [1.0, 1.0, 1.0]}}, open(cache / "log.json", "w"))
    view = ws / "MRT" / "Verlauf-Daten"; os.makedirs(view)
    ph.write_pet(str(ws / "pet"))
    f = {"pet": str(ws / "pet"), "cache": str(cache), "view": str(view), "ref": "brain", "out": str(ws / "out"), "build": "0"}
    assert pet_run.check(f)[0] == []
    monkeypatch.setenv("MRT_PY", sys.executable)
    monkeypatch.setenv("PET_STARTS", "0")          # one start per search: the search itself is tested in test_mrt_pet
    pet_run.start(f)
    pet_run.JOB["thread"].join(900)
    st = pet_run.status()
    assert st["status"] == "done", st["log"]
    assert set(st["steps"].values()) == {"done", "skip"} and st["start_n"] == [3, 3]
    r = st["result"]
    assert r["day"] == "2020-03-01" and r["sheet"] is True
    assert r["registration"]["abstand_relativ"] >= 0.01 and r["registration"]["abstand_kopfstehend_relativ"] >= 0.01
    assert pet_run.sheet_path().endswith("pet_kontrolle_2020-03-01.png")
    assert {p.name for p in view.iterdir()} == {"pet.json", "pet_2020-03-01.js"}
