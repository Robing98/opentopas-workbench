"""Readers for TOPAS scorer output (csv and binary) with any set of reports (Sum, Standard_Deviation, ...).

Both formats carry the same header: the bin lines ("# X in 80 bins of 0.2 cm") and the report line
("# DoseToWater ( Gy ) : Sum   Standard_Deviation"). A binary file `NAME.bin` has its header in `NAME.binheader`.
Arrays come back indexed [x, y, z] (the csv lists i, j, k with k fastest).
"""
import os, re

import numpy as np

UNIT_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "um": 1e-3}


def sum_sd(sd, histories):
    """Standard deviation of the TOPAS Sum of `histories` histories from its Standard_Deviation report.

    Standard_Deviation is the spread of the dose of one history, not of the Sum: the Sum of N histories spreads
    by sd * sqrt(N). Taking sd as the spread of the Sum understates the uncertainty by sqrt(N); on the TG119 runs
    (200k histories per field) the voxel noise was 10 %, sd / Sum gave 0.02 %, sd * sqrt(N) / Sum gave 10 %.
    """
    return np.asarray(sd, float) * np.sqrt(histories)


def parse_header(lines):
    bins, reports, quantity, unit = {}, [], None, None
    for line in lines:
        if not line.startswith("#"): break
        m = re.match(r"#\s*(X|Y|Z|R|Phi|Theta|E)\s+in\s+(\d+)\s+bins?\s+of\s+([\d.eE+-]+)\s*(\w+)", line)
        if m:
            a, n, w, u = m.groups()
            bins[a] = (int(n), float(w) * UNIT_MM.get(u, 1.0) if a in "XYZR" else float(w), u)
            continue
        m = re.match(r"#\s*(.+?)\s*\(\s*(.*?)\s*\)\s*:\s*(.+)$", line)
        if m and not line.startswith("# Parameter") and not line.startswith("# TOPAS"):
            quantity, unit, rest = m.groups()
            reports = rest.split()
    return {"bins": bins, "reports": reports, "quantity": quantity, "unit": unit}


def shape(h):
    """Cartesian (X, Y, Z) or cylindrical (R, Phi, Z) bins; an axis the header leaves out has one bin."""
    first = ("R", "Phi") if "R" in h["bins"] else ("X", "Y")
    return tuple(h["bins"][a][0] if a in h["bins"] else 1 for a in first + ("Z",))


def read(path):
    """Read a TOPAS csv or binary scorer file. Returns (header, {report: ndarray})."""
    if path.endswith(".bin") or os.path.exists(path + ".bin"):
        base = path[:-4] if path.endswith(".bin") else path
        with open(base + ".binheader") as fh: h = parse_header(fh.readlines())
        raw = np.fromfile(base + ".bin", dtype="<f8")
        n, k = int(np.prod(shape(h))), max(1, len(h["reports"]))
        if raw.size != n * k: raise ValueError(f"{base}.bin: {raw.size} values, header says {n} bins x {k} reports")
        # TOPAS binary: all reports of one bin together, bins with X (or R) fastest, Z slowest. Verified against the
        # csv of the same scorer in TOPAS 4.3 (checks/ct_range.py, 50 x 320 x 40 bins, Sum and Standard_Deviation).
        data = raw.reshape(n, k)
        return h, {r: data[:, i].reshape(shape(h), order="F") for i, r in enumerate(h["reports"] or ["Sum"])}
    with open(path) as fh: lines = fh.readlines()
    h = parse_header(lines)
    rows = np.array([[float(v) for v in l.split(",")] for l in lines if l.strip() and not l.startswith("#")])
    idx = tuple(rows[:, j].astype(int) for j in range(3))   # csv rows: i, j, k, then one column per report
    out = {}
    for i, r in enumerate((h["reports"] or ["Sum"])[:rows.shape[1] - 3]):
        a = np.zeros(shape(h)); a[idx] = rows[:, 3 + i]; out[r] = a
    return h, out


def depth_curve(path, report="Sum"):
    """Depth dose from a scorer binned in Z only (or in X, Y, Z: summed over x and y). Returns z centres [mm], values."""
    h, d = read(path)
    a = d[report]
    if a.ndim == 3: a = a.sum(axis=(0, 1))
    n, w, _ = h["bins"]["Z"]
    return (np.arange(n) + 0.5) * w, np.asarray(a, float).ravel()
