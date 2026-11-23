"""Case file fall/v1: schema, fall_check.py, the demo case, the search of the assistant and the start from a case.
Synthetic data only: the demo case and folder trees built here."""
import json
import os
import shutil
import struct
import sys

import pytest

from opentopas_workbench import dicomkopf, fall, fall_setup, letd, pet_run, server as wb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import fall_check  # noqa: E402


@pytest.fixture
def demo(tmp_path):
    return fall_setup.demo(str(tmp_path / "demo"))


def rc(folder, *a):
    return fall_check.main([str(folder), *a])


def edit(folder, fn):
    p = os.path.join(folder, fall.NAME)
    d = json.load(open(p, encoding="utf-8")); fn(d)
    json.dump(d, open(p, "w", encoding="utf-8"))


# ------------------------------------------------------------------ DICOM headers
def test_rtplan_fields_are_read(tmp_path):
    p = str(tmp_path / "plan.dcm")
    dicomkopf.write(p, {"Modality": "RTPLAN", "SOPInstanceUID": "1.2",
                        "FractionGroupSequence": [{"NumberOfFractionsPlanned": 28}],
                        "DoseReferenceSequence": [{"DoseReferenceType": "ORGAN_AT_RISK", "DeliveryMaximumDose": 12.0},
                                                  {"DoseReferenceType": "TARGET", "TargetPrescriptionDose": 54.0}],
                        "IonBeamSequence": [{"RadiationType": "PROTON"}],
                        "ReferencedStructureSetSequence": [{"ReferencedSOPInstanceUID": "7.7"}]})
    f = fall_setup.plan_facts(p)
    assert (f["fraktionen"], f["verschreibung_gy_rbe"], f["teilchen"], f["rtstruct_uid"]) == (28, 54.0, "proton", "7.7")


def test_plan_without_target_takes_the_delivery_maximum_and_a_ct_is_no_plan(tmp_path):
    p = str(tmp_path / "plan.dcm")
    dicomkopf.write(p, {"Modality": "RTPLAN", "DoseReferenceSequence": [{"DeliveryMaximumDose": 50.4}]})
    assert fall_setup.plan_facts(p)["verschreibung_gy_rbe"] == 50.4
    dicomkopf.write(p, {"Modality": "CT"})
    assert fall_setup.plan_facts(p) == {}
    (tmp_path / "x.dcm").write_bytes(b"not dicom")
    assert dicomkopf.read(str(tmp_path / "x.dcm")) is None


def _implicit(tag, raw):
    return struct.pack("<HHI", *tag, len(raw)) + raw


def test_implicit_vr_with_an_unknown_sequence_of_undefined_length(tmp_path):
    """Planning systems write implicit VR with sequences of undefined length; one the reader does not know by name must
    be skipped, not end the reading before the fractions."""
    item_end, seq_end = struct.pack("<HHI", 0xFFFE, 0xE00D, 0), struct.pack("<HHI", 0xFFFE, 0xE0DD, 0)
    unknown = struct.pack("<HHI", 0x300A, 0x03A8, 0xFFFFFFFF) + struct.pack("<HHI", 0xFFFE, 0xE000, 0xFFFFFFFF) \
        + _implicit((0x300A, 0x0114), b"100 ") + item_end + seq_end
    fg = struct.pack("<HHI", 0x300A, 0x0070, 0xFFFFFFFF) + struct.pack("<HHI", 0xFFFE, 0xE000, 0xFFFFFFFF) \
        + _implicit((0x300A, 0x0078), b"30") + item_end + seq_end
    meta_uid = b"1.2.840.10008.1.2\x00"
    meta = struct.pack("<HH", 0x0002, 0x0010) + b"UI" + struct.pack("<H", len(meta_uid)) + meta_uid
    body = _implicit((0x0008, 0x0060), b"RTPLAN") + unknown + fg
    p = tmp_path / "impl.dcm"
    p.write_bytes(b"\x00" * 128 + b"DICM" + meta + body)
    assert dicomkopf.plan_facts(dicomkopf.read(str(p)))["fraktionen"] == 30


# ------------------------------------------------------------------ schema and fall_check
def test_demo_case_passes(demo, capsys):
    assert rc(demo) == 0
    out = capsys.readouterr().out
    assert "AKTUELL  workbench/mc-1" in out and out.rstrip().endswith("OK")
    d = fall.load(demo)
    assert d["plan"]["fraktionen"] == 30 and d["plan"]["oeffentlich"] is True
    assert not any(os.path.isabs(str(v)) for v in json.dumps(d).split('"') if v.startswith("/"))


def test_schema_errors_fail_loud_with_rc_2(demo, capsys):
    edit(demo, lambda d: d.update(schema="fall/v0", rev=-1))
    assert rc(demo) == 2
    out = capsys.readouterr().out
    assert "schema" in out and "rev" in out and "KAPUTT" in out


@pytest.mark.parametrize("bad", ["/abs/plan.dcm", "C:/x/plan.dcm", "../plan.dcm", "plan\\rtplan.dcm", "@unbekannt/x"])
def test_bad_paths_are_refused(demo, bad):
    edit(demo, lambda d: d["plan"].update(datei=bad))
    assert rc(demo) in (1, 2)


def test_missing_file_and_changed_result_are_kaputt(demo, capsys):
    os.remove(os.path.join(demo, "laeufe", "mc-ohne-rifi", "letd.npy"))
    assert rc(demo) == 1
    assert "KAPUTT   workbench/mc-1  (fehlt: letd.npy)" in capsys.readouterr().out


def test_changed_input_makes_the_result_veraltet_but_not_broken(demo, capsys):
    p = os.path.join(demo, "plan", "rtplan.dcm")
    with open(p, "ab") as fh: fh.write(b"  ")
    assert rc(demo) == 0
    out = capsys.readouterr().out
    assert "VERALTET workbench/mc-1  (Eingabe plan geändert)" in out and "Hash weicht" in out


def test_quick_check_reads_no_hashes(demo):
    with open(os.path.join(demo, "plan", "rtplan.dcm"), "ab") as fh: fh.write(b"  ")
    rep = fall.check(demo, hashes=False)
    assert fall.ok(rep) and rep["ergebnisse"][0]["zustand"] == "aktuell"


@pytest.mark.parametrize("n", [1, 3])
def test_one_and_three_series(demo, n):
    def add(d):
        for i in range(n):
            os.makedirs(os.path.join(demo, "bilder", f"t1_{i}"), exist_ok=True)
            d["bilder"].append({"serie": f"t1_{i}", "modalitaet": "MR", "datum": f"2001-0{i + 2}-01", "ordner": f"bilder/t1_{i}",
                                "nach_bezug": None, "guete": None})
    fall.update(demo, add)
    rep = fall.check(demo)
    assert fall.ok(rep)
    assert sum("noch keine Transformation" in w for w in rep["warnungen"]) == n + 1     # + the PET
    edit(demo, lambda d: d["bilder"][-1].update(serie="ct_plan"))
    assert "doppelt" in " ".join(fall.check(demo)["fehler"])


def test_public_zone_warns_about_a_plan_not_marked_public(tmp_path):
    folder = fall_setup.demo(str(tmp_path / "workspace" / "offen" / "faelle" / "x"))
    fall.update(folder, lambda d: d["plan"].pop("oeffentlich"))
    assert any("Zone lokal" in w for w in fall.check(folder)["warnungen"])


def test_update_raises_rev_and_keeps_other_sections(demo):
    before = fall.load(demo)
    after = fall.update(demo, lambda d: d.setdefault("lieferung", {}).update(kalibrierung=None))
    assert after["rev"] == before["rev"] + 1 and after["plan"] == before["plan"]


# ------------------------------------------------------------------ the search
def touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh: fh.write(text)


@pytest.fixture
def tree(tmp_path):
    """A computer as the assistant meets it: MRI folder in OneDrive, cacheF elsewhere, two plan_mc runs in lokal (the
    newer with run.json), a PET series in the MRI folder, and decoys."""
    ws = tmp_path / "ws"; src = fall_setup.demo(str(tmp_path / "src"))
    mrt = tmp_path / "OneDrive" / "MRT Bilder"
    touch(str(mrt / "Verlauf.html")); touch(str(mrt / "alt" / "Verlauf.html"))
    os.utime(str(mrt / "alt" / "Verlauf.html"), (1, 1))
    os.makedirs(mrt / "Analyse" / "Verlauf-Daten"); os.makedirs(mrt / "Analyse" / "Skripte")
    shutil.copytree(os.path.join(src, "bilder", "pet_2001-04-01"), str(mrt / "PET" / "2001" / "serie1"))
    shutil.copytree(os.path.join(src, "bilder", "ct_plan"), str(mrt / "CT_kein_PET"))
    cache_root = tmp_path / "work"; shutil.copytree(os.path.join(src, "cacheF"), str(cache_root / "cacheF"))
    plan = ws / "lokal" / "plan"; shutil.copytree(os.path.join(src, "plan"), str(plan))
    shutil.copytree(os.path.join(src, "bilder", "ct_plan"), str(ws / "lokal" / "ct"))
    old = ws / "lokal" / "plan_mc" / "alt"; new = ws / "lokal" / "plan_mc" / "neu"
    for r in (old, new): shutil.copytree(os.path.join(src, "laeufe", "mc-ohne-rifi"), str(r))
    os.remove(old / "run.json"); os.utime(str(old / "letd.npy"), (1, 1))
    json.dump({"eingaben": {"ct": str(ws / "lokal" / "ct"), "rtstruct": str(plan / "rtstruct.dcm"),
                            "rtplan": str(plan / "rtplan.dcm")}}, open(new / "run.json", "w"))
    os.makedirs(ws / "offen" / "plan_mc" / "public")   # runs in offen are not searched
    shutil.copytree(os.path.join(src, "laeufe", "mc-ohne-rifi"), str(ws / "offen" / "plan_mc" / "public" / "r"))
    return {"ws": ws, "mrt": mrt, "cache_root": cache_root, "new": new, "old": old, "plan": plan}


def test_search_finds_the_right_folders(tree):
    r = fall_setup.scan(str(tree["mrt"]), str(tree["ws"]), [str(tree["cache_root"])])["zeilen"]
    m = tree["mrt"]
    assert r["seite"]["wert"] == str(m / "Verlauf.html")
    assert r["daten"]["wert"] == str(m / "Analyse" / "Verlauf-Daten") and r["skripte"]["wert"] == str(m / "Analyse" / "Skripte")
    assert r["cache"] == {"wert": str(tree["cache_root"] / "cacheF"), "quelle": "gemerkt"}
    assert r["lauf"]["wert"] == str(tree["new"]) and [c["wert"] for c in r["lauf"]["kandidaten"]] == [str(tree["new"]), str(tree["old"])]
    assert r["rtplan"] == {"wert": str(tree["plan"] / "rtplan.dcm"), "quelle": "lauf"}
    assert r["ct"]["wert"] == str(tree["ws"] / "lokal" / "ct")
    assert (r["fraktionen"]["wert"], r["verschreibung"]["wert"]) == (30, 60.0)
    assert r["pet"]["wert"] == [str(m / "PET" / "2001" / "serie1")] and r["pet"]["kandidaten"][0]["info"].startswith("2001-04-01")


def test_search_without_cache_root_and_with_an_old_run_asks(tree):
    shutil.rmtree(tree["new"])
    r = fall_setup.scan(str(tree["mrt"]), str(tree["ws"]))["zeilen"]
    assert r["cache"]["quelle"] == "fehlt" and r["lauf"]["wert"] == str(tree["old"])
    assert r["rtplan"]["quelle"] == r["ct"]["quelle"] == r["fraktionen"]["quelle"] == "fehlt"
    assert fall_setup.scan(str(tree["mrt"] / "nichts"), str(tree["ws"])) == {"fehler": "mrt_nicht_gefunden"}


def test_save_case_keeps_machine_paths_out_of_fall_json(tree, tmp_path):
    r = fall_setup.scan(str(tree["mrt"]), str(tree["ws"]), [str(tree["cache_root"])])["zeilen"]
    vals = {k: v["wert"] for k, v in r.items()}
    folder = str(tree["ws"] / "lokal" / "faelle" / "mein-fall")
    fall_setup.save_case(folder, "mein-fall", vals)
    assert rc(folder) == 0
    text = open(os.path.join(folder, fall.NAME), encoding="utf-8").read()
    assert str(tmp_path) not in text and "OneDrive" not in text
    orte = fall.load_orte(folder)
    assert any(v == str(tree["mrt"]) for v in orte.values())
    f = fall_setup.letd_form(folder)
    assert (f["run"], f["fx"], f["rx"], f["cache"]) == (str(tree["new"]), "30", "60", str(tree["cache_root"] / "cacheF"))
    assert f["build"] == "1" and f["synthseg"] == "1" and f["oeffentlich"] is False
    p = fall_setup.pet_form(folder)
    assert p["pet"] == str(tree["mrt"] / "PET" / "2001" / "serie1") and p["view"] == f["view"]


def test_save_case_names_what_is_missing(tree):
    with pytest.raises(fall.FallError, match="fehlt: lauf, ct"):
        fall_setup.save_case(str(tree["ws"] / "lokal" / "faelle" / "x"), "x", {"cache": "c", "daten": "d", "fraktionen": 30})


def test_browse_lists_folders_and_marks_a_run(tree):
    b = fall_setup.browse(str(tree["new"]), str(tree["ws"]))
    assert b["lauf"] and "run.json" in b["dateien"] and b["oben"] == str(tree["new"].parent)
    assert "neu" in fall_setup.browse(str(tree["new"].parent), str(tree["ws"]))["ordner"]


# ------------------------------------------------------------------ server: choose and start
@pytest.fixture
def ws(tmp_path):
    wb.STATE["root"] = str(tmp_path / "ws")
    os.makedirs(os.path.join(wb.STATE["root"], "workspace", "lokal"))
    os.makedirs(os.path.join(wb.STATE["root"], "workspace", "offen"))
    wb.STATE["cfg"] = {"binaries": [], "lang": "en"}
    wb.STATE["jobs"] = []
    yield tmp_path


def test_demo_case_is_registered_and_started_without_a_typed_path(ws, monkeypatch):
    started = []
    monkeypatch.setattr(letd, "start", lambda form, on_end=None, viewer=None: started.append(("letd", form)) or {})
    monkeypatch.setattr(letd, "check", lambda form, root=None, viewer=None: ([], []))
    monkeypatch.setattr(pet_run, "check", lambda form, root=None, viewer=None: ([], []))
    monkeypatch.setattr(pet_run, "start", lambda form, on_end=None, viewer=None: started.append(("pet", form)) or {})
    monkeypatch.setattr(letd, "status", lambda: {}); monkeypatch.setattr(pet_run, "status", lambda: {})
    code, r = wb.fall_post("demo", {})
    assert code == 200 and r["ordner"].endswith(os.path.join("offen", "faelle", "demo-tg119"))
    info = wb.fall_info()
    assert info["zuletzt"] == r["ordner"] and info["faelle"][0]["fall_id"] == "demo-tg119" and info["faelle"][0]["pet"]
    code, s = wb.fall_post("start", {"ordner": r["ordner"], "letd": True, "pet": True})
    assert code == 200 and [k for k, _ in started] == ["letd"]           # PET waits for LETd
    f = started[0][1]
    assert f["oeffentlich"] is True and f["fx"] == "30" and os.path.isdir(f["run"])
    assert "oeffentlich" not in wb.STATE["cfg"]["letd"]["form"]
    assert wb.fall_post("start", {"ordner": "/elsewhere", "letd": True})[0] == 404


def test_pet_starts_after_a_finished_letd_run(ws, monkeypatch):
    ends = []
    monkeypatch.setattr(letd, "start", lambda form, on_end=None, viewer=None: ends.append(on_end) or {})
    monkeypatch.setattr(letd, "check", lambda form, root=None, viewer=None: ([], []))
    monkeypatch.setattr(pet_run, "check", lambda form, root=None, viewer=None: ([], []))
    pets = []
    monkeypatch.setattr(pet_run, "start", lambda form, on_end=None, viewer=None: pets.append(form) or {})
    monkeypatch.setattr(letd, "status", lambda: {}); monkeypatch.setattr(pet_run, "status", lambda: {})
    _, r = wb.fall_post("demo", {})
    wb.fall_post("start", {"ordner": r["ordner"], "letd": True, "pet": True})
    ends[0]({"status": "failed"}); assert pets == []
    ends[0]({"status": "done"}); assert len(pets) == 1 and pets[0]["pet"].endswith("pet_2001-04-01")


def test_the_form_cannot_claim_a_public_case(ws, monkeypatch):
    seen = []
    monkeypatch.setattr(letd, "check", lambda form, root=None, viewer=None: seen.append(form) or ([], []))
    wb.letd_post("check", {"form": {"oeffentlich": True}})
    assert "oeffentlich" not in seen[0]


def test_save_without_lokal_and_with_a_bad_name(ws, tmp_path):
    assert wb.fall_post("save", {"fall_id": "a b", "werte": {}})[1] == {"fehler": "fall_id"}
    shutil.rmtree(os.path.join(wb.STATE["root"], "workspace", "lokal"))
    assert wb.fall_post("save", {"fall_id": "x", "werte": {}})[1] == {"fehler": "kein_lokal"}


def test_case_without_results_section_loads_with_empty_lists(tmp_path):
    """ergebnisse is optional in fall/v1: a hand-written case without it must not crash the readers."""
    d = fall.new("ohne-ergebnisse"); del d["ergebnisse"]; del d["bilder"]
    (tmp_path / "fall.json").write_text(json.dumps(d))
    got = fall.load(str(tmp_path))
    assert got["ergebnisse"] == {} and got["bilder"] == []
