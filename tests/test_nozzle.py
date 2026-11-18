"""Tests for the HIT nozzle generator: element order and water equivalent, wire grids, filter placement, energy spread."""
import importlib.util, math, os, re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("nozzle", os.path.join(ROOT, "opentopas_workbench", "generators", "nozzle.py"))
nz = importlib.util.module_from_spec(spec); spec.loader.exec_module(nz)


def values(**kw):
    v = {f["key"]: f.get("default") for f in nz.FIELDS}
    v.update({"energies": "132"}, **kw)
    return v


def text(**kw):
    files = nz.generate(values(**kw))
    assert len(files) == 1
    return files[0][1]


def boxes(t):
    """{name: (material, half z mm, centre z mm)} of every TsBox in a parameter text."""
    out = {}
    for name in re.findall(r's:Ge/(\w+)/Type\s*=\s*"TsBox"', t):
        mat = re.search(rf's:Ge/{name}/Material\s*=\s*"([^"]+)"', t).group(1)
        h = re.search(rf"d:Ge/{name}/HLZ\s*=\s*([\d.e-]+) (mm|cm)", t)
        hz = float(h.group(1)) * (10 if h.group(2) == "cm" else 1)
        tz = re.search(rf"d:Ge/{name}/TransZ\s*=\s*(-?[\d.e-]+) (mm|cm)", t)
        out[name] = (mat, hz, float(tz.group(1)) * (10 if tz.group(2) == "cm" else 1))
    return out


def test_m1_order_and_water_equivalent_sum():
    b = boxes(text(models=["m1"]))
    water = {k: v for k, v in b.items() if k in nz.ELEMENTS}
    assert sorted(water, key=lambda k: -water[k][2]) == nz.ELEMENTS        # window first, MWPC 2 last along the beam
    assert all(m == "G4_WATER" for m, _, _ in water.values())
    assert sum(2 * hz for _, hz, _ in water.values()) == pytest.approx(1.37, abs=1e-6)   # mm, Parodi 2012
    mat, hz, z = b["MWPC2"]
    assert z - hz == pytest.approx(1120.0)                                   # end of the monitors at 112 cm


def test_m2_mwpc_is_tungsten_on_water():
    b = boxes(text(models=["m2"]))
    for e in ("MWPC1", "MWPC2"):
        assert b[e + "W"][0] == "G4_W" and 2 * b[e + "W"][1] == pytest.approx(0.003)
        assert b[e][0] == "G4_WATER" and 2 * b[e][1] == pytest.approx(0.130)
        assert b[e][2] - b[e][1] == pytest.approx(b[e + "W"][2] + b[e + "W"][1])   # touching, not overlapping
    assert 2 * b["IC2"][1] == pytest.approx(0.230) and 2 * b["Window"][1] == pytest.approx(0.360)


def test_none_has_no_elements_but_the_same_source():
    t0, t1 = text(models=["none"]), text(models=["m1"])
    assert not set(nz.ELEMENTS) & set(boxes(t0))
    src = lambda t: re.search(r"BeamPosition/TransZ = (.*)", t).group(1)
    assert src(t0) == src(t1)


def test_source_sits_in_the_vacuum_upstream_of_the_window():
    t = text(models=["m1"])
    b = boxes(t)
    z = float(re.search(r"BeamPosition/TransZ = ([\d.]+) mm", t).group(1))
    mat, hz, c = b["Vacuum"]
    assert mat == "G4_Galactic" and c - hz < z < c + hz
    assert c - hz > b["Window"][2] + b["Window"][1]                          # vacuum ends before the window
    assert c + hz < nz.WORLD_HZ


def wires(t):
    """{plane: [(offset, half length, radius)]} and {plane: angle} of the wire model."""
    planes = dict(re.findall(r's:Ge/(\w+)/Type\s*=\s*"Group"\ns:Ge/\1/Parent = "World"\nd:Ge/\1/RotZ\s*=\s*(-?\d+) deg', t))
    out = {p: [] for p in planes}
    for w, parent in re.findall(r's:Ge/(\w+)/Parent\s*=\s*"(\w+P\d+)"', t):
        g = lambda k: float(re.search(rf"d:Ge/{w}/{k}\s*=\s*(-?[\d.]+) mm", t).group(1))
        out[parent].append((g("TransX"), g("HL"), g("RMax")))
    return out, {p: int(a) for p, a in planes.items()}


@pytest.mark.parametrize("hv,expected", [(2, [0, 45, 90, -45]), (1, [0, 45, 90]), (0, [0, 90])])
def test_wire_planes_per_mwpc(hv, expected):
    w, ang = wires(text(models=["wire"], hv_planes=hv))
    for e in ("MWPC1", "MWPC2"):
        assert [ang[f"{e}P{k}"] for k in range(len(expected))] == expected
        for k, a in enumerate(expected):
            ws = w[f"{e}P{k}"]
            pitch, r = (1.0, 0.025) if a in (0, 90) else (2.0, 0.010)
            offs = sorted(o for o, _, _ in ws)
            assert all(abs(b - a_ - pitch) < 1e-9 for a_, b in zip(offs, offs[1:]))
            assert all(rr == r for _, _, rr in ws)
            assert all(o * o + hl * hl == pytest.approx(40.0 ** 2, abs=0.01) for o, hl, _ in ws)   # ends on the 40 mm circle


def test_wire_planes_do_not_touch_each_other_or_the_water():
    t = text(models=["wire"])
    zs = sorted({float(z) for z in re.findall(r"d:Ge/MWPC2P\d+W\d+/TransZ\s*=\s*([\d.]+) mm", t)})
    assert len(zs) == 4 and min(b - a for a, b in zip(zs, zs[1:])) == pytest.approx(nz.WIRE_GAP, abs=0.03)
    b = boxes(t)
    assert b["MWPC2"][2] - b["MWPC2"][1] > zs[-1] + 0.025
    assert min(zs) - 0.025 == pytest.approx(1120.0)                          # first plane ends the monitors


def test_mean_wire_tungsten_against_parodi():
    # two signal planes alone: 2 pi 25^2 / 1000 um = 3.93 um, Parodi method 2 rounds the MWPC to 3 um tungsten
    assert nz.mean_wire_um(0) == pytest.approx(2 * math.pi * 0.025 ** 2 * 1000, rel=1e-9)
    assert nz.mean_wire_um(2) == pytest.approx(nz.mean_wire_um(0) + 2 * math.pi * 0.01 ** 2 / 2 * 1000, rel=1e-9)


@pytest.mark.parametrize("at", [108.0, 80.0, 55.5])
@pytest.mark.parametrize("variant", ["t4", "rifi2"])
def test_filter_moves_with_its_plates(at, variant):
    t = text(variants=[variant], filter_pos=str(at))
    z = [float(x) for x in re.findall(r"d:Ge/RiFi\d/TransZ\s*=\s*([\d.]+) cm", t)]
    ref = [float(x) for x in re.findall(r"TransZ\s*=\s*([\d.]+) cm", nz.RF.variant_block(variant, 1.5))]
    assert z == pytest.approx([r - nz.RF_REF_CM + at for r in ref])
    assert 'Material = "PMMA_HIT"' in t and "d:Ma/PMMA_HIT/Density = 1.20 g/cm3" in t
    assert "MeanExcitationEnergy = 74.0 eV" in t and "G4_PLEXIGLASS" not in t


def test_filter_material_can_stay_g4():
    t = text(variants=["t4"], filter_mat="G4_PLEXIGLASS")
    assert '"G4_PLEXIGLASS"' in t and "PMMA_HIT" not in t


def test_every_ripple_filter_variant_generates():
    for var in nz.RF.VARIANTS:
        files = nz.generate(values(variants=[var], filter_pos="108, 80"))
        cases = [n for n, _ in files if n.endswith(".txt")]
        assert len(cases) == (1 if var == "rifi0" else 2)
    assert any(n.endswith(".stl") for n, _ in nz.generate(values(variants=["pin4"])))


@pytest.mark.parametrize("bad", ["111.5", "112", "2"])
def test_filter_must_lie_behind_the_monitors(bad):
    with pytest.raises(ValueError):
        nz.generate(values(variants=["t4"], filter_pos=bad))


def test_element_positions_are_checked():
    with pytest.raises(ValueError): nz.generate(values(positions="132, 128, 124"))
    with pytest.raises(ValueError): nz.generate(values(positions="112, 116, 120, 124, 128, 132"))


def test_hit_momentum_spread():
    assert nz.momentum_fwhm("p", 48.0) == pytest.approx(0.30)
    assert nz.momentum_fwhm("p", 221.0) == pytest.approx(0.05)
    assert nz.momentum_fwhm("c", 430.0) == pytest.approx(0.10)
    mid = nz.momentum_fwhm("c", (88 + 430) / 2)
    assert mid == pytest.approx(math.sqrt(0.30 * 0.10))                      # exponential: geometric mean halfway
    # energy sigma: dE/E = (gamma + 1)/gamma dp/p, close to 2 dp/p for slow ions
    assert nz.energy_sigma("p", 48.0) == pytest.approx(0.30 / 2.3548 * (2 + 48 / 938.27) / (1 + 48 / 938.27), rel=2e-3)
    assert "BeamEnergySpread         = 0.25" in text(spread="0.25")


def test_parallel_beam_has_no_angular_spread():
    t = text(divergence=0)
    assert 'BeamAngularDistribution  = "None"' in t and "AngularSpreadX" not in t
    assert "BeamAngularSpreadX       = 1.5 mrad" in text(divergence=1.5)


def test_phase_space_behind_the_monitors():
    assert "PhaseSpace" not in text()
    t = text(phsp="write")
    assert 'Surface    = "PhspPlane/ZMinusSurface"' in t and 'KillAfterPhaseSpace = "False"' in t
    b = boxes(t)["PhspPlane"]
    assert b[2] + b[1] == pytest.approx(1119.0)                              # 1 mm behind MWPC 2
    assert 'KillAfterPhaseSpace = "True"' in text(phsp="only")


def test_case_names_and_count():
    files = nz.generate(values(ions=["p", "c"], energies="63, 132", models=["none", "m1"], variants=["rifi0", "t4"], filter_pos="108, 80"))
    names = [n for n, _ in files]
    assert len(names) == len(set(names)) == 2 * 2 * 2 * (1 + 2)
    assert "noz_c132_m1_t4_d80.txt" in names and "noz_p63_none.txt" in names
