"""matRad plans into the plan_mc chain (part C2): CT, structures, steering information (stf) and weights from a matRad
.mat file, so matRad variants run through the same TOPAS chain as the clinical plan.

    python3 matrad.py summary FILE.mat
    python3 matrad.py ct FILE.mat OUTDIR     # write the CT as DICOM slices for TsDicomPatient

Expected variables (matRad workspace as saved from the GUI or by pyRadPlan's matlab export): ct with cubeHU, x, y, z
and resolution; cst; stf; and the weights either as resultGUI.w or as a variable w. An optional resultGUI.physicalDose
is read as the matRad dose for comparison.

Geometry (matRad_getRotationMatrix, pyRadPlan geometry/lps.py): a point in the beam's eye view (BEV) goes to the LPS
world as R_couch(about +y) @ R_gantry(about +z) @ p_bev; the source sits at (0, -SAD, 0) in the BEV and the beam runs
along +y_bev. ct.x, ct.y and ct.z are DICOM patient coordinates. For a head first supine patient this equals IEC 61217
(iec.py) with IEC gantry X = x_bev and IEC gantry Y = z_bev at the isocenter plane, for every gantry and couch angle:
tests/test_plan_mc_matrad.py checks it against iec.spot_ray, which the TOPAS alignment check verified. matRad has no
patient position; other positions than HFS are refused.

Weights: matRad and pyRadPlan compute a bixel's dose for 1e6 primaries (pyRadPlan dose/engines/_topasmc.py,
scaling_factor 1e6), so the particle number of a spot is w * 1e6 (PARTICLES_PER_W).
Index convention of cubes: matRad stores [y, x, z] with column major 1-based voxel indices in cst; this module returns
cubes as [z, y, x] like synthetic.read_ct.
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan_mc import plan as pl  # noqa: E402

PARTICLES_PER_W = 1e6


def _load(path):
    import scipy.io as sio
    try:
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:   # MATLAB v7.3 (HDF5)
        raise SystemExit(f"{path}: MATLAB v7.3 file; save it in matRad with save(FILE, ..., '-v7')")


def _cube(c):
    """matRad keeps cubes in a cell (one per CT scenario); take the first."""
    if isinstance(c, np.ndarray) and c.dtype == object: c = c.flat[0]
    return np.asarray(c)


def _arr(v): return np.atleast_1d(np.asarray(v, float))


def _structs(v):
    return list(np.atleast_1d(v))


def rotation(gantry, couch):
    """matRad beam rotation: LPS = R @ BEV (matRad_getRotationMatrix; pyRadPlan get_beam_rotation_matrix)."""
    g, c = np.radians(gantry), np.radians(couch)
    rg = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
    rc = np.array([[np.cos(c), 0, np.sin(c)], [0, 1, 0], [-np.sin(c), 0, np.cos(c)]])
    return rc @ rg


def read_ct(m):
    """(HU [z, y, x], origin of the first voxel centre (x, y, z), spacing (dx, dy, dz)) in DICOM coordinates."""
    ct = m["ct"]
    hu = _cube(ct.cubeHU)
    if hu.ndim != 3: raise ValueError(f"ct.cubeHU has shape {hu.shape}")
    if not all(hasattr(ct, a) for a in "xyz"):
        raise ValueError("ct has no x, y, z world axes; import the CT with a matRad that sets them (matRad 3)")
    ax = [np.asarray(getattr(ct, a), float).ravel() for a in "xyz"]
    ny, nx, nz = hu.shape
    if (len(ax[0]), len(ax[1]), len(ax[2])) != (nx, ny, nz):
        raise ValueError(f"ct axes {[len(a) for a in ax]} do not fit cubeHU [y, x, z] {hu.shape}")
    sp = []
    for a in ax:
        d = np.diff(a) if len(a) > 1 else np.array([1.0])
        if np.ptp(d) > 1e-3 * abs(d[0]): raise ValueError("non-uniform CT axis")
        if d[0] <= 0: raise ValueError("CT axis not increasing")
        sp.append(float(d[0]))
    return np.ascontiguousarray(hu.transpose(2, 0, 1)), (ax[0][0], ax[1][0], ax[2][0]), tuple(sp)


def read_cst(m, shape_zyx):
    """[(name, type, mask [z, y, x])] from the matRad cst; voxel indices are 1-based, column major in [y, x, z]."""
    nz, ny, nx = shape_zyx
    out = []
    for row in np.atleast_2d(m["cst"]):
        idx = row[3]
        if isinstance(idx, np.ndarray) and idx.dtype == object: idx = idx.flat[0]
        idx = np.asarray(idx, np.int64).ravel() - 1
        iy, ix, iz = np.unravel_index(idx, (ny, nx, nz), order="F")
        mask = np.zeros(shape_zyx, bool)
        mask[iz, iy, ix] = True
        out.append((str(row[1]), str(row[2]), mask))
    return out


def weights(m):
    if "resultGUI" in m and hasattr(m["resultGUI"], "w"): return _arr(m["resultGUI"].w)
    if "w" in m: return _arr(m["w"])
    raise ValueError("no weights: neither resultGUI.w nor w in the file")


def _position(m):
    info = getattr(m["ct"], "dicomInfo", None)
    pos = getattr(info, "PatientPosition", None) if info is not None else None
    return str(pos).strip().upper() if isinstance(pos, str) and pos.strip() else "HFS"


def to_plan(m, label="matRad", w=None):
    """plan.Plan from stf and weights; spot weights in particles (w * PARTICLES_PER_W), zero weights dropped."""
    pos = _position(m)
    if pos != "HFS": raise ValueError(f"patient position {pos}: the matRad to IEC mapping is checked for HFS only")
    w = weights(m) if w is None else _arr(w)
    stf = _structs(m["stf"])
    n = sum(int(np.size(r.energy)) for b in stf for r in _structs(b.ray))
    if n != len(w): raise ValueError(f"stf has {n} bixels, the weights {len(w)}")
    fields, k = [], 0
    for i, b in enumerate(stf, 1):
        layers, shifters = {}, set()
        for r in _structs(b.ray):
            pos_bev = _arr(r.rayPos_bev)
            if abs(pos_bev[1]) > 1e-6: raise ValueError("rayPos_bev not at the isocenter plane (y_bev != 0)")
            rs = _structs(getattr(r, "rangeShifter", [])) or [None]
            for j, e in enumerate(_arr(r.energy)):
                wj = float(w[k]); k += 1
                s = rs[j] if j < len(rs) else rs[-1]
                if s is not None and float(getattr(s, "eqThickness", 0) or 0) > 0:
                    shifters.add((str(getattr(s, "ID", "")), float(s.eqThickness),
                                  float(b.SAD) - float(getattr(s, "sourceRashiDistance", 0) or 0)))
                if wj <= 0: continue
                l = layers.setdefault(round(float(e), 4), pl.Layer(float(e), [], [], []))
                l.x.append(float(pos_bev[0])); l.y.append(float(pos_bev[2])); l.w.append(wj * PARTICLES_PER_W)
        if len(shifters) > 1:
            raise ValueError(f"beam {i}: several range shifters in one field are not supported: {sorted(shifters)}")
        rsh = None
        if shifters:
            rid, wet, dist = shifters.pop()
            # matRad gives the source to range shifter distance; the DICOM isocenter distance is taken from it as an
            # approximation (thickness of the shifter neglected)
            rsh = pl.RangeShifter(rid, wet, dist)
        fields.append(pl.Field(f"{label}_b{i}", i, float(b.gantryAngle), float(b.couchAngle),
                               tuple(float(v) for v in _arr(b.isoCenter)),
                               [layers[e] for e in sorted(layers, reverse=True)], "particles", range_shifter=rsh))
    return pl.Plan(label, pos, fields)


def read(path, label=None):
    """{"plan", "hu", "origin", "spacing", "cst", "dose"} from a matRad .mat; dose is resultGUI.physicalDose [z, y, x]
    on the CT grid or None."""
    m = _load(path)
    hu, origin, spacing = read_ct(m)
    dose = None
    if "resultGUI" in m and hasattr(m["resultGUI"], "physicalDose"):
        d = _cube(m["resultGUI"].physicalDose)
        if d.shape == hu.shape[1:] + hu.shape[:1]: dose = np.ascontiguousarray(d.transpose(2, 0, 1))
    return {"plan": to_plan(m, label or os.path.splitext(os.path.basename(path))[0]), "hu": hu, "origin": origin,
            "spacing": spacing, "cst": read_cst(m, hu.shape), "dose": dose}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "summary":
        r = read(sys.argv[2])
        print(f"CT {r['hu'].shape} [z, y, x], origin {np.round(r['origin'], 2).tolist()}, spacing {r['spacing']}")
        for name, typ, mask in r["cst"]: print(f"  {name} ({typ}): {int(mask.sum())} voxels")
        for f in r["plan"].fields:
            print(f"  {f.name}: gantry {f.gantry:g}, couch {f.couch:g}, iso {np.round(f.iso, 2).tolist()}, "
                  f"{len(f.layers)} layers, {f.n_spots} spots, {f.total_weight:.4g} particles"
                  + (f", range shifter {f.range_shifter.wet:g} mm WET" if f.range_shifter else ""))
    elif len(sys.argv) >= 4 and sys.argv[1] == "ct":
        from plan_mc import synthetic
        r = read(sys.argv[2])
        synthetic.write_ct(sys.argv[3], r["hu"], r["origin"], r["spacing"])
        print(f"{r['hu'].shape[0]} slices in {sys.argv[3]}")
    else:
        sys.exit(__doc__)
