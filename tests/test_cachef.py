"""cacheF from the viewer's page (tools/mrt_overlay/cachef.py) and the job of the case assistant that runs it.
Every volume here is synthetic; the page and data folder are written the way build_page.py of the Timeline Viewer
writes them."""
import base64
import io
import json
import os
import sys
import time

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
sitk = pytest.importorskip("SimpleITK")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from mrt_overlay import cachef as cf  # noqa: E402
from opentopas_workbench import fall_setup  # noqa: E402
from opentopas_workbench import server as wb  # noqa: E402

NX, NY, NZ, COLS = 10, 8, 14, 4          # 14 slices in 4 columns: the last mosaic row is half empty


def volume(seed):
    z, y, x = np.mgrid[0:NZ, 0:NY, 0:NX]
    return ((x * 7 + y * 13 + z * 11 + seed * 29) % 256).astype(np.uint8)


def js(track, day, vol, kind="PNG", **opt):
    m = np.zeros((-(-NZ // COLS) * NY, COLS * NX), np.uint8)
    for z in range(NZ):
        m[(z // COLS) * NY:(z // COLS + 1) * NY, (z % COLS) * NX:(z % COLS + 1) * NX] = vol[z]
    b = io.BytesIO()
    Image.fromarray(m).save(b, kind, **opt)
    return f'MRT.vol("{track}","{day}","data:image/{kind.lower()};base64,{base64.b64encode(b.getvalue()).decode()}");'


def viewer(top, days=("2001-02-01", "2001-05-01"), refday="2001-05-01", kind="PNG", sp=1.5):
    """A viewer folder: Analyse/Verlauf.html with its data, Verlauf-Daten with T1 and FLAIR per day and the mask,
    Skripte/stand/cacheF_log.json. Returns (page, daten, skripte, {name: volume})."""
    an = os.path.join(top, "Analyse")
    daten, stand = os.path.join(an, "Verlauf-Daten"), os.path.join(an, "Skripte", "stand")
    os.makedirs(daten); os.makedirs(stand)
    vols, tracks, log = {}, {}, {"size": [1, 1, 1], "spacing": 9.0}
    for i, day in enumerate(days):
        for j, tr in enumerate(("t1", "flair")):
            v = volume(i * 2 + j); vols[f"{tr}_{day}"] = v
            open(os.path.join(daten, f"{tr}_{day}.js"), "w").write(js(tr, day, v, kind))
            tracks.setdefault(tr, {"label": tr, "exams": []})["exams"].append({"day": day})
            log[f"{tr} {day}"] = {"series": "x", "kind": "3D", "native": [1, 1, 1]}
    mask = np.zeros((NZ, NY, NX), np.uint8); mask[3:10, 2:6, 2:8] = 128; mask[5:8, 3:5, 4:6] = 255
    open(os.path.join(daten, "mask_all.js"), "w").write(js("mask", "all", mask))
    vols["mask"] = mask > 0
    data = {"dim": [NX, NY, NZ], "sp": sp, "cols": COLS, "dir": "Verlauf-Daten", "tracks": tracks, "refday": refday,
            "reports": [{"text": "a </script> inside"}]}
    page = os.path.join(an, "Verlauf.html")
    open(page, "w", encoding="utf-8").write("<!doctype html><script>\nconst D = "
                                             + json.dumps(data).replace("</", "<\\/") + ";\nMRT.go(D);\n</script>")
    json.dump(log, open(os.path.join(stand, "cacheF_log.json"), "w"))
    return page, daten, os.path.join(an, "Skripte"), vols


def test_lossless_page_gives_the_volumes_back_exactly(tmp_path):
    page, daten, sk, vols = viewer(str(tmp_path / "v"))
    out = str(tmp_path / "cacheF")
    r = cf.build(page, daten, os.path.join(sk, "stand", "cacheF_log.json"), out, say=lambda m: None)
    assert r == {"volumen": 4, "fehlen": [], "ref_t1": "2001-05-01", "maske": True}
    for name in ("t1_2001-02-01", "flair_2001-05-01"):
        a = np.load(os.path.join(out, name + ".npy"))
        assert a.dtype == np.uint8 and a.shape == (NZ, NY, NX) and (a == vols[name]).all()
    assert (np.load(os.path.join(out, "mask.npy")).astype(bool) == vols["mask"]).all()
    ref = sitk.ReadImage(os.path.join(out, "ref_t1.nii.gz"))
    assert ref.GetSize() == (NX, NY, NZ) and ref.GetSpacing() == (1.5, 1.5, 1.5)
    assert ref.GetDirection() == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert (sitk.GetArrayFromImage(ref) == vols["t1_2001-05-01"]).all()
    log = json.load(open(os.path.join(out, "log.json")))
    assert (log["size"], log["spacing"]) == ([NX, NY, NZ], 1.5)       # the page's grid wins over the copy
    assert "t1 2001-02-01" in log and log["aus_viewer"]["verlustbehaftet"]
    # what the commands viewer and pet read: the assistant's check takes the folder as a cacheF
    assert fall_setup.check_path("cache", out)["befund"] == "ok"


def test_lossy_webp_stays_close_and_ref_day_is_the_nearest_t1(tmp_path):
    page, daten, sk, vols = viewer(str(tmp_path / "v"), refday="2001-03-01", kind="WEBP", sp=[1.0, 1.0, 2.0])
    out = str(tmp_path / "c")
    r = cf.build(page, daten, os.path.join(sk, "stand", "cacheF_log.json"), out, say=lambda m: None)
    assert r["ref_t1"] == "2001-02-01"
    a = np.load(os.path.join(out, "t1_2001-02-01.npy")).astype(int)
    assert np.abs(a - vols["t1_2001-02-01"]).mean() < 12
    assert sitk.ReadImage(os.path.join(out, "ref_t1.nii.gz")).GetSpacing() == (1.0, 1.0, 2.0)


def test_a_missing_volume_is_named_and_a_page_without_data_stops(tmp_path):
    page, daten, sk, _ = viewer(str(tmp_path / "v"))
    os.remove(os.path.join(daten, "flair_2001-02-01.js"))
    out = str(tmp_path / "c")
    assert cf.build(page, daten, os.path.join(sk, "stand", "cacheF_log.json"), out, say=lambda m: None)["fehlen"] \
        == ["flair_2001-02-01"]
    open(page, "w").write("<html>nothing</html>")
    with pytest.raises(cf.CacheError):
        cf.build(page, daten, os.path.join(sk, "stand", "cacheF_log.json"), str(tmp_path / "d"), say=lambda m: None)
    assert not os.path.exists(os.path.join(str(tmp_path / "d"), "log.json"))


def test_the_assistant_makes_cachef_and_the_next_search_finds_it(tmp_path, monkeypatch):
    """From a subfolder of the viewer with no cacheF anywhere: the row offers the button, the job writes cacheF into
    lokal, and a new search finds it there."""
    view = tmp_path / "Sync" / "MRT Bilder"
    page, daten, sk, _ = viewer(str(view))
    exam = view / "2001-02-01 Klinik" / "Bilder"; os.makedirs(exam)
    ws = tmp_path / "ws"; os.makedirs(ws / "workspace" / "lokal")
    root = str(ws)
    r = fall_setup.scan(str(exam), root, default_roots=[])["zeilen"]["cache"]
    target = fall_setup.cachef_target(root, page)
    assert os.path.dirname(os.path.dirname(target)) == str(ws / "workspace" / "lokal" / "viewer")
    assert (r["wert"], r["quelle"], r["erzeugbar"], r["ziel"]) == (None, "fehlt", True, target)
    monkeypatch.setenv("MRT_PY", sys.executable)
    fall_setup.cachef_start({"seite": page, "daten": daten, "skripte": sk}, root)
    end = time.time() + 120
    while fall_setup.cachef_status()["running"] and time.time() < end: time.sleep(0.2)
    st = fall_setup.cachef_status()
    assert st["status"] == "done", st.get("log")
    assert st["cache"] == target and st["vol"] == [4, 4] and set(st["steps"].values()) == {"done"}
    r = fall_setup.scan(str(exam), root, default_roots=[])["zeilen"]["cache"]
    assert (r["wert"], r["quelle"]) == (target, "gemerkt") and "erzeugbar" not in r


def test_the_job_refuses_without_sources_or_lokal(tmp_path):
    page, daten, sk, _ = viewer(str(tmp_path / "v"))
    with pytest.raises(ValueError, match="quellen"):
        fall_setup.cachef_start({"seite": page, "daten": daten, "skripte": str(tmp_path)}, str(tmp_path))
    with pytest.raises(ValueError, match="kein_lokal"):
        fall_setup.cachef_start({"seite": page, "daten": daten, "skripte": sk}, str(tmp_path / "leer"))


def test_each_viewer_gets_its_own_cachef(tmp_path):
    """A second patient's viewer must never write into the first one's cacheF."""
    os.makedirs(tmp_path / "ws" / "lokal")
    a = fall_setup.cachef_target(str(tmp_path / "ws"), str(tmp_path / "a" / "Analyse" / "Verlauf.html"))
    b = fall_setup.cachef_target(str(tmp_path / "ws"), str(tmp_path / "b" / "Analyse" / "Verlauf.html"))
    assert a != b and os.path.basename(a) == os.path.basename(b) == "cacheF"
    assert fall_setup.cachef_target(str(tmp_path / "ws"), None) is None


def test_a_failed_rerun_leaves_the_earlier_cachef_whole(tmp_path):
    page, daten, sk, vols = viewer(str(tmp_path / "v"))
    log, out = os.path.join(sk, "stand", "cacheF_log.json"), str(tmp_path / "c" / "cacheF")
    cf.build(page, daten, log, out, say=lambda m: None)
    before = {n: open(os.path.join(out, n), "rb").read() for n in os.listdir(out)}
    for n in os.listdir(daten):
        if n.startswith("t1_"): os.remove(os.path.join(daten, n))     # no T1: the rerun stops at ref_t1
    with pytest.raises(cf.CacheError):
        cf.build(page, daten, log, out, say=lambda m: None)
    assert {n: open(os.path.join(out, n), "rb").read() for n in os.listdir(out)} == before
    assert not os.path.exists(out + ".part")
    open(os.path.join(out, "notizen.txt"), "w").write("x")
    with pytest.raises(cf.CacheError, match="fremde"):
        cf.build(page, daten, log, out, say=lambda m: None)


def test_the_job_can_be_stopped_says_why_it_failed_and_runs_once(tmp_path, monkeypatch):
    page, daten, sk, _ = viewer(str(tmp_path / "v"))
    os.makedirs(tmp_path / "ws" / "lokal")
    slow = tmp_path / "slow.sh"; slow.write_text("echo '## STEP read'; sleep 30\n")
    monkeypatch.setattr(fall_setup, "CACHEF_SCRIPT", str(slow))
    vals = {"seite": page, "daten": daten, "skripte": sk}
    assert fall_setup.cachef_start(vals, str(tmp_path / "ws"))["running"]
    with pytest.raises(RuntimeError):
        fall_setup.cachef_start(vals, str(tmp_path / "ws"))
    assert wb.fall_post("cachef_stop", {})[1]["stopped"] is True
    end = time.time() + 20
    while fall_setup.cachef_status()["running"] and time.time() < end: time.sleep(0.1)
    assert fall_setup.cachef_status()["status"] == "stopped"
    with fall_setup.letd.LOCK: fall_setup.CACHEF_JOB["state"].update(status="failed", error="Platte voll")
    assert fall_setup.cachef_status()["meldung"] == "Platte voll"


def test_an_own_cachef_that_lacks_a_new_exam_is_made_again(tmp_path):
    """The viewer gained an exam after its cacheF was made: the row offers the button again instead of keeping it."""
    page, daten, sk, _ = viewer(str(tmp_path / "v"))
    ws = tmp_path / "ws"; os.makedirs(ws / "lokal")
    target = fall_setup.cachef_target(str(ws), page)
    cf.build(page, daten, os.path.join(sk, "stand", "cacheF_log.json"), target, say=lambda m: None)
    r = fall_setup.scan(os.path.dirname(page), str(ws), default_roots=[])["zeilen"]["cache"]
    assert (r["wert"], r["quelle"]) == (target, "gemerkt")
    log = os.path.join(sk, "stand", "cacheF_log.json")
    d = json.load(open(log)); d["t1 2001-09-01"] = {}; json.dump(d, open(log, "w"))
    r = fall_setup.scan(os.path.dirname(page), str(ws), default_roots=[])["zeilen"]["cache"]
    assert r["wert"] is None and r["erzeugbar"] and r["ziel"] == target


def test_a_rerun_swaps_the_folder_whole(tmp_path):
    page, daten, sk, _ = viewer(str(tmp_path / "v"))
    log, out = os.path.join(sk, "stand", "cacheF_log.json"), str(tmp_path / "c" / "cacheF")
    cf.build(page, daten, log, out, say=lambda m: None)
    open(os.path.join(out, "desktop.ini"), "w").write("x")       # Explorer's file does not block a rebuild
    os.remove(os.path.join(daten, "flair_2001-02-01.js"))
    r = cf.build(page, daten, log, out, say=lambda m: None)
    assert r["fehlen"] == ["flair_2001-02-01"] and "flair_2001-02-01.npy" not in os.listdir(out)
    assert sorted(os.listdir(tmp_path / "c")) == ["cacheF"]
