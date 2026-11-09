"""plan_mc in a workspace with data zones (offen/, robin/, lokal/), and the centre of mass inside the body (task 2351).

Synthetic data only; every workspace is a tmp folder.
"""
import json, os, sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("pydicom")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_mc import evaluate, run_patient, synthetic  # noqa: E402
from test_plan_mc_probe_fixes import NAME, blob, export, fake_topas, run_find  # noqa: E402
from test_plan_mc_run_patient import ORIGIN, SHAPE, SPACING, args, make_inputs  # noqa: E402


def zoned(tmp_path):
    ws = tmp_path / "ws"
    for z in ("offen", "robin", "lokal"): (ws / z).mkdir(parents=True)
    return ws


def test_run_folder_lies_in_lokal_when_the_workspace_has_zones(tmp_path):
    ws = zoned(tmp_path)
    inp = make_inputs(tmp_path, 1)
    run_patient.main(args(ws, inp))
    out = ws / "lokal" / "patient" / "plan_mc" / "run1"
    assert (out / "field1.txt").exists() and (out / "run.json").exists()
    assert sorted(os.listdir(ws)) == ["lokal", "offen", "robin"]           # no patient/ outside the zones
    assert os.listdir(ws / "offen") == [] and os.listdir(ws / "robin") == []


def test_run_folder_keeps_the_former_place_without_zones(tmp_path):
    assert run_patient.folder(str(tmp_path), "r") == os.path.join(str(tmp_path), "patient", "plan_mc", "r")


def test_find_takes_the_public_hlut_from_offen_of_the_workspace(tmp_path, capsys, monkeypatch):
    ex, hd, _ = export(tmp_path, None, hlut=False)
    ws = zoned(tmp_path)
    hl = ws / "offen" / "plan_mc" / "hluts" / "matRad_default.hlut"; hl.parent.mkdir(parents=True); hl.write_text("0 1.0\n")
    monkeypatch.setattr(run_patient, "ROOT", str(tmp_path / "no_checkout"))     # as from a git archive
    env, err = run_find(capsys, str(ex), "--hlut-dir", str(hd), "--workspace", str(ws))
    assert env["HLUT"] == str(hl) and "public matRad default HLUT" in err
    env, err = run_find(capsys, str(ex), "--hlut-dir", str(hd))
    assert env["HLUT"] == "" and "no HLUT found" in err


def test_generic_beam_model_by_name_warns_and_lands_in_the_run(tmp_path, capsys):
    ws = zoned(tmp_path)
    inp = make_inputs(tmp_path, 1); inp["beam_model"] = "generic"
    run_patient.main(args(ws, inp))
    m = json.loads((ws / "lokal" / "patient" / "plan_mc" / "run1" / "beam_model.json").read_text())
    assert m["name"].startswith("generic") and "generic beam model" in capsys.readouterr().err


def test_centre_of_mass_ignores_dose_outside_the_body():
    ref = blob(); ref[0, 0, 0] = 0.0
    body = ref > 0.01 * ref.max()
    assert not body[0, 0, 0]
    ev = ref.copy(); ev[0, 0, 0] = 20.0 * ref.max()                             # couch or air voxel
    free = evaluate.diagnostics(ref, ev, ORIGIN, SPACING)
    assert max(abs(v) for v in free["centre_of_mass_shift_mm"]) > 1.0          # the old way: pulled away
    d = evaluate.diagnostics(ref, ev, ORIGIN, SPACING, body=body)
    assert d["centre_of_mass_shift_mm"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert d["rigid"]["shift_mm"] == [0.0, 0.0, 0.0]


def evaluate_zoned(tmp_path, capsys, topas, ref):
    ws = zoned(tmp_path)
    inp = make_inputs(tmp_path, 1)
    rd = str(tmp_path / f"RD_{NAME}.dcm")
    synthetic.write_rtdose(rd, ref, ORIGIN, SPACING, dose_type="PHYSICAL")
    run_patient.main(args(ws, inp, rtdose=rd))
    out = ws / "lokal" / "patient" / "plan_mc" / "run1"
    fake_topas(out, [topas])
    capsys.readouterr()
    run_patient.main(["evaluate", "--workspace", str(ws), "--name", "run1"])
    return ws, out, capsys.readouterr().out


def test_evaluate_reports_the_centre_inside_the_body_and_writes_kennzahlen(tmp_path, capsys):
    ref = blob(); ref[0, 0, 0] = 0.0
    topas = ref.copy(); topas[0, 0, 0] = 5.0                                   # dose outside the body
    ws, out, printed = evaluate_zoned(tmp_path, capsys, topas, 30 * ref)
    res = json.loads((out / "result.json").read_text())
    assert res["diagnostics"]["centre_of_mass_shift_mm"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert "nur in der Koerpermaske" in (out / "BERICHT.md").read_text()
    k = json.loads((ws / "robin" / "plan_mc" / "run1" / "kennzahlen.json").read_text())
    # the hub's format: at most 200 figures, names of 1 to 80 characters, finite numbers or texts of at most 200
    assert 0 < len(k) <= 200
    for n, v in k.items():
        assert 1 <= len(n) <= 80
        assert (isinstance(v, str) and len(v) <= 200) or (isinstance(v, (int, float)) and np.isfinite(v))
    assert k["gamma_2%/2mm_pass_pct"] == 100.0 and k["centre_of_mass_shift_x_mm"] == pytest.approx(0.0, abs=1e-9)
    text = json.dumps(k)
    assert NAME not in text and "/" not in text.replace("2%/2mm", "").replace("3%/3mm", "")    # no path, no name
    assert "robin/plan_mc/run1/kennzahlen.json" in printed
    assert not (out / "kennzahlen.json").exists() and (out / "result.json").exists()


def test_no_kennzahlen_without_zones(tmp_path, capsys):
    ws = tmp_path / "ws"; ws.mkdir()
    assert run_patient.write_kennzahlen(str(ws), "run1", {"a": 1.0}) is None and os.listdir(ws) == []
