import importlib.util, io, json, os

import pytest

spec = importlib.util.spec_from_file_location("zonen_umzug", os.path.join(os.path.dirname(__file__), "..", "tools", "zonen_umzug.py"))
zu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zu)

# Synthetic workspace with the layout of the thesis workspace; every file holds a marker, never real data.
FILES = [
    "00_Start/10 bauen.bat", "STUDIEN.md", "studien/check.py", "studien/s1.json",
    "patient/plan_mc/case1/dose.dcm", "patient/ct/a.dcm",
    "plan_peaks/run.py", "plan_peaks/planF1_120.txt", "plan_peaks/protons_MIT_noRiFi.mat",
    "plan_peaks/kalibriert/planF1_120.txt", "plan_peaks/crosscheck/vergleich_topas/kalibrierung.csv",
    "plan_peaks/crosscheck/report.md", "plan_peaks/crosscheck/vergleich_topas_matrad.md", "plan_peaks/rs20/out.csv", "plan_peaks/mit_hlut_v2.csv",
    "plan_mc/public/a.txt", "plan_mc/run_patient.py",
    "ba_rifi/messenergien/IDD_mess_120.csv", "ba_rifi/messenergien/IDD_topas_120.csv", "ba_rifi/scripts/fit.py",
    "matrad_phantom/typisch/x.txt", "_to_delete/sonstiges/y.txt", ".workbench/jobs.json",
]


@pytest.fixture
def ws(tmp_path):
    for f in FILES:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("marker")
    return tmp_path


def zones(moves): return {m["src"]: m["zone"] for m in moves}


def test_plan_sorts_by_name(ws):
    z = zones(zu.plan(str(ws)))
    for src in ("patient", "plan_peaks/kalibriert", "plan_peaks/protons_MIT_noRiFi.mat", "plan_peaks/crosscheck/vergleich_topas",
                "ba_rifi/messenergien/IDD_mess_120.csv", "_to_delete", "plan_peaks/mit_hlut_v2.csv"):
        assert z[src] == "lokal", src
    for src in ("00_Start", "STUDIEN.md", "studien", "plan_mc", "matrad_phantom", "plan_peaks/run.py", "plan_peaks/planF1_120.txt",
                "plan_peaks/rs20", "plan_peaks/crosscheck/report.md", "ba_rifi/messenergien/IDD_topas_120.csv", "ba_rifi/scripts"):
        assert z[src] == "offen", src
    assert ".workbench" not in z and not any(s.startswith("patient/") for s in z)


def test_unsure_marked_and_lokal_folder_not_listed(ws):
    moves = zu.plan(str(ws))
    unsure = {m["src"] for m in moves if m["unsure"]}
    assert unsure == {"_to_delete", "plan_peaks/mit_hlut_v2.csv", "plan_peaks/crosscheck/vergleich_topas_matrad.md"}
    out = io.StringIO(); zu.show(moves, out)
    text = out.getvalue()
    assert "patient  ->  lokal/patient  2 Dateien" in text
    assert "case1" not in text and "dose.dcm" not in text and "ct" not in text.split("patient")[1].split("\n")[0]


def test_dry_run_moves_nothing(ws, capsys):
    before = sorted(str(p) for p in ws.rglob("*"))
    assert zu.main([str(ws)]) == 0
    assert sorted(str(p) for p in ws.rglob("*")) == before
    assert "Probelauf" in capsys.readouterr().out


def test_apply_moves_and_reports(ws):
    path = zu.apply(str(ws), zu.plan(str(ws)), today="2026-11-07")
    assert (ws / "lokal/patient/plan_mc/case1/dose.dcm").is_file()
    assert (ws / "lokal/plan_peaks/kalibriert/planF1_120.txt").is_file()
    assert (ws / "lokal/plan_peaks/crosscheck/vergleich_topas/kalibrierung.csv").is_file()
    assert (ws / "offen/plan_peaks/crosscheck/report.md").is_file()
    assert (ws / "offen/plan_peaks/run.py").is_file() and (ws / "offen/00_Start/10 bauen.bat").is_file()
    assert (ws / "robin").is_dir() and not any((ws / "robin").iterdir())
    assert (ws / ".workbench/jobs.json").is_file()
    left = sorted(p.name for p in ws.iterdir())
    assert left == [".workbench", "lokal", "offen", "robin"]
    for f in FILES:
        if f.startswith(".workbench"): continue
        assert (ws / "offen" / f).exists() != (ws / "lokal" / f).exists(), f
    text = open(path, encoding="utf-8").read()
    assert path.endswith(os.path.join("offen", "ZONEN_UMZUG_2026-11-07.txt"))
    assert "plan_peaks/run.py -> offen/plan_peaks/run.py" in text
    # lokal only as counts: no lokal path or name shows in the report that sits in offen
    for word in ("patient", "kalibriert", "protons_MIT", "IDD_mess", "vergleich_topas", "_to_delete", "hlut"):
        assert word not in text, word
    assert "lokal: 8 Verschiebungen, 9 Dateien" in text


def test_second_run_finds_nothing(ws):
    zu.apply(str(ws), zu.plan(str(ws)), today="2026-11-07")
    assert zu.plan(str(ws)) == []


def test_existing_target_refuses(ws):
    (ws / "offen/studien").mkdir(parents=True)
    with pytest.raises(SystemExit):
        zu.apply(str(ws), zu.plan(str(ws)))
    assert (ws / "patient/ct/a.dcm").is_file()


def test_job_records_follow_their_files(ws):
    wbd = ws / ".workbench"; (wbd / "logs").mkdir()
    (wbd / "logs/job00001.log").write_text("log1"); (wbd / "logs/job00002.log").write_text("log2")
    (wbd / "index.json").write_text("{}")
    (ws / "plan_peaks/o.csv").write_text("m")
    jobs = [{"id": 1, "file": "patient/plan_mc/case1/a.txt", "log": ".workbench/logs/job00001.log", "outputs": ["patient/plan_mc/case1/d.csv"]},
            {"id": 2, "file": "plan_peaks/planF1_120.txt", "log": ".workbench/logs/job00002.log", "outputs": ["plan_peaks/o.csv"]}]
    (wbd / "jobs.json").write_text(json.dumps(jobs))
    zu.apply(str(ws), zu.plan(str(ws)), today="2026-11-07")
    shared = (wbd / "jobs.json").read_text()
    assert "patient" not in shared and json.loads(shared)[0]["file"] == "offen/plan_peaks/planF1_120.txt"
    assert json.loads(shared)[0]["outputs"] == ["offen/plan_peaks/o.csv"]
    lok = json.loads((ws / "lokal/.workbench/jobs.json").read_text())
    assert lok[0]["file"] == "lokal/patient/plan_mc/case1/a.txt" and lok[0]["log"] == "lokal/.workbench/logs/job00001.log"
    assert (ws / "lokal/.workbench/logs/job00001.log").read_text() == "log1" and not (wbd / "logs/job00001.log").exists()
    assert (wbd / "logs/job00002.log").exists() and not (wbd / "index.json").exists()


def test_case_trash_and_kalib_names(tmp_path):
    for f in ("Patient/a.dcm", "_trash/x.txt", "plan_peaks/Kalibriert/a.txt", "plan_peaks/kalibrierung.csv", "plan_peaks/ok.csv"):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True); (tmp_path / f).write_text("m")
    z = zones(zu.plan(str(tmp_path)))
    assert z == {"Patient": "lokal", "_trash": "lokal", "plan_peaks/Kalibriert": "lokal", "plan_peaks/kalibrierung.csv": "lokal",
                 "plan_peaks/ok.csv": "offen"}
    (tmp_path / "Lokal").mkdir()
    with pytest.raises(SystemExit): zu.plan(str(tmp_path))


def test_refuses_while_a_workbench_runs(ws):
    (ws / ".workbench/server.json").write_text("{}")
    with pytest.raises(SystemExit): zu.apply(str(ws), zu.plan(str(ws)))
    assert (ws / "patient").is_dir()


def test_failed_move_is_reported_and_second_run_appends(ws, monkeypatch):
    real = zu.shutil.move; calls = []
    def flaky(a, b):
        calls.append(a)
        if len(calls) == 3: raise OSError("locked")
        return real(a, b)
    monkeypatch.setattr(zu.shutil, "move", flaky)
    with pytest.raises(OSError): zu.apply(str(ws), zu.plan(str(ws)), today="2026-11-07")
    monkeypatch.setattr(zu.shutil, "move", real)
    zu.apply(str(ws), zu.plan(str(ws)), today="2026-11-07")
    text = (ws / "offen/ZONEN_UMZUG_2026-11-07.txt").read_text(encoding="utf-8")
    assert text.count("Umzug in die Datenzonen") == 2 and "ABGEBROCHEN" in text
    assert sorted(p.name for p in ws.iterdir()) == [".workbench", "lokal", "offen", "robin"]
