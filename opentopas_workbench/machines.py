"""matRad machine files (base data as .mat): reader, folder summary, the data behind the Machines view, and comparisons.

A machine is the MATLAB struct `machine` with `meta` (radiationMode, dataType, SAD, ...) and the struct array `data`,
one element per energy (energy, depths, Z, offset, initFocus, sigma or sigma1/sigma2/weight, LET, alpha, beta, ...).
Depths are in mm; as in matRad's dose engine the water depth is depths + offset.

Unlike the rest of the workbench this module needs numpy and scipy (MATLAB v5/v7) or h5py (v7.3). They are imported
only when a machine is read, so the workbench starts without them.

Machine files may hold clinical base data: they are read where they lie and nothing of them is written to disk or
logged; parsed machines stay in memory only.
"""
import math, os, re, sys, threading
from types import SimpleNamespace

DEPTH_SKIP = ("depths",)
_CACHE, _LOCK, _MAX = {}, threading.Lock(), 8   # parsed machines by (path, mtime, size)
_SUMMARY, _VIEW = {}, {}


def need_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        raise RuntimeError("matRad machines need numpy and scipy: python3 -m pip install --user numpy scipy")


# ------------------------------------------------------------------ reading
def _h5(f, o):
    import h5py
    np = need_numpy()
    if isinstance(o, h5py.Group):
        keys = [k for k in o.keys() if not k.startswith("#")]
        items = {k: o[k] for k in keys}
        is_array = keys and all(isinstance(v, h5py.Dataset) and v.dtype == h5py.ref_dtype for v in items.values()) \
            and o.attrs.get("MATLAB_class", b"struct") in (b"struct", "struct")
        if is_array:  # struct array: every field holds one reference per element
            refs = {k: v[()].flatten(order="F") for k, v in items.items()}
            n = len(next(iter(refs.values())))
            el = [SimpleNamespace(**{k: _h5(f, f[refs[k][i]]) for k in keys}) for i in range(n)]
            for e in el: e._fieldnames = keys
            return el[0] if n == 1 else el
        ns = SimpleNamespace(**{k: _h5(f, v) for k, v in items.items()}); ns._fieldnames = keys
        return ns
    cls = o.attrs.get("MATLAB_class", b"")
    cls = cls.decode() if isinstance(cls, bytes) else str(cls)
    if o.attrs.get("MATLAB_empty", 0): return np.array([])
    if o.dtype == h5py.ref_dtype:  # cell array
        return [_h5(f, f[r]) for r in o[()].flatten(order="F")]
    a = np.asarray(o[()])
    if cls == "char": return "".join(chr(c) for c in a.flatten(order="F") if c)
    a = a.T
    if cls == "logical": a = a.astype(bool)
    return a.item() if a.size == 1 else np.squeeze(a)


def load_struct(path, var="machine"):
    """A MATLAB struct variable with attribute access (scipy's mat_struct or SimpleNamespace); struct arrays become lists."""
    need_numpy()
    if mat_format(path) != "v7.3":
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("MATLAB v5/v7 file: install scipy (python3 -m pip install --user scipy)")
        try:
            got = sio.loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=[var])
        except NotImplementedError:  # an HDF5 file that the header check missed
            pass
        else:
            if var not in got: raise ValueError(f"no variable '{var}' in {os.path.basename(path)}")
            return got[var]
    try:
        import h5py
    except ImportError:
        raise RuntimeError("MATLAB v7.3 file: install h5py (python3 -m pip install --user h5py)")
    with h5py.File(path, "r") as f:
        if var not in f: raise ValueError(f"no variable '{var}' in {os.path.basename(path)}")
        return _h5(f, f[var])


def mat_format(path):
    """'v7.3' (HDF5), 'v7' (compressed v5 container), 'v6' (uncompressed), or 'v4' from the first bytes of the file."""
    with open(path, "rb") as fh: head = fh.read(520)
    if head.startswith(b"MATLAB 7.3") or head[512:520] == b"\x89HDF\r\n\x1a\n" or head[:8] == b"\x89HDF\r\n\x1a\n": return "v7.3"
    if len(head) >= 136 and head[:6] == b"MATLAB":
        order = "<" if head[126:128] == b"IM" else ">"
        import struct
        return "v7" if struct.unpack(order + "I", head[128:132])[0] == 15 else "v6"   # 15 = miCOMPRESSED
    return "v4"


def fields(o):
    """Field names of a loaded struct (scipy mat_struct or the h5py namespace)."""
    return list(getattr(o, "_fieldnames", None) or [])


def entries(m):
    np = need_numpy()
    d = getattr(m, "data", None)
    if d is None: raise ValueError("not a matRad machine: no field 'data'")
    return d if isinstance(d, list) else list(np.atleast_1d(d))


def zvec(x):
    """The depth dose of one entry; some machines store several Z models in a struct, then the usual one is taken."""
    np = need_numpy()
    z = x.Z
    if fields(z):
        for k in ("doubleGauss", "profileORG", "singleGauss"):
            if k in fields(z): return np.ravel(np.asarray(getattr(z, k), float))
        return np.ravel(np.asarray(getattr(z, fields(z)[0]), float))
    return np.ravel(np.asarray(z, float))


def offset(x):
    np = need_numpy()
    v = getattr(x, "offset", 0.0)
    return float(np.ravel(v)[0]) if np.size(v) else 0.0


def idd_entries(path):
    """Energy, depths, depth dose and offset of every entry, as compare_idd_matrad.py uses them."""
    np = need_numpy()
    return [{"energy": float(np.ravel(x.energy)[0]), "depths": np.ravel(np.asarray(x.depths, float)), "Z": zvec(x), "offset": offset(x)}
            for x in entries(load_struct(path, "machine"))]


# ------------------------------------------------------------------ depth dose metrics
def metrics(z, y):
    """Peak, R80, BPW80, distal fall-off D80-D20, peak-to-entrance ratio and entrance value of a depth dose (z in mm)."""
    np = need_numpy()
    i = int(np.argmax(y)); m = y[i]; n = y / m
    def cross(level, step):
        k = i
        while 0 <= k + step < len(n):
            if n[k] >= level > n[k + step]: return z[k] + (z[k + step] - z[k]) * (n[k] - level) / (n[k] - n[k + step])
            k += step
        return np.nan
    r80, r20, p80 = cross(.8, 1), cross(.2, 1), cross(.8, -1)
    k0 = max(1, int(np.searchsorted(z, 2.0)))                     # entrance: mean over the first 2 mm
    return {"peak": z[i], "r80": r80, "bpw80": r80 - p80, "dfo": r20 - r80, "pe": m / y[:k0].mean(), "z0": y[:k0].mean()}


# ------------------------------------------------------------------ cache and paths
def _key(path):
    st = os.stat(path)
    return (os.path.realpath(path), st.st_mtime_ns, st.st_size)


def machine(path):
    """The parsed machine, cached in memory by modification time (the folder view and a comparison read it again)."""
    k = _key(path)
    with _LOCK:
        if k in _CACHE: return _CACHE[k]
    m = load_struct(path, "machine")
    with _LOCK:
        for old in [o for o in _CACHE if o[0] == k[0]]: del _CACHE[old]
        while len(_CACHE) >= _MAX: del _CACHE[next(iter(_CACHE))]
        _CACHE[k] = m
    return m


def _is_wsl():
    try:
        with open("/proc/version") as fh: return "microsoft" in fh.read().lower()
    except OSError:
        return False


def local_path(s):
    """A folder as typed: ~ and %USERPROFILE% expanded; on WSL a Windows path C:\\x\\y becomes /mnt/c/x/y."""
    s = (s or "").strip().strip('"')
    if "%USERPROFILE%" in s.upper():
        s = re.sub(r"(?i)%USERPROFILE%", lambda _: windows_home() or os.path.expanduser("~"), s)
        if os.sep == "/": s = s.replace("\\", "/")
    m = re.match(r"^([A-Za-z]):[\\/]?(.*)$", s)
    if m and os.sep == "/":
        s = f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")
    return os.path.expanduser(s)


_WINHOME = {}


def windows_home():
    """The Windows user folder: USERPROFILE on Windows; on WSL asked once from cmd.exe, as a /mnt path. None elsewhere."""
    if "h" in _WINHOME: return _WINHOME["h"]
    h = None
    if os.name == "nt":
        h = os.environ.get("USERPROFILE")
    elif _is_wsl():
        v = os.environ.get("USERPROFILE", "")
        if not v:
            import subprocess
            try:
                v = subprocess.run(["cmd.exe", "/c", "echo %USERPROFILE%"], capture_output=True, text=True, timeout=10, cwd="/mnt/c").stdout.strip()
            except (OSError, subprocess.SubprocessError):
                v = ""
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", v)
        if m: h = f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/").rstrip("/")
    _WINHOME["h"] = h
    return h


def default_dir():
    """%USERPROFILE%\\matRad\\machines, where matRad keeps user machines; ~/matRad/machines outside Windows and WSL."""
    return os.path.join(windows_home() or os.path.expanduser("~"), "matRad", "machines")


# ------------------------------------------------------------------ JSON
def _clean(v):
    """numpy values to JSON; NaN and inf to None. Seven significant digits: a machine view stays near 1 MB."""
    np = need_numpy()
    if isinstance(v, dict): return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_clean(x) for x in v]
    if isinstance(v, np.ndarray): return _clean(v.tolist())
    if isinstance(v, (np.floating, float)): v = float(v); return float(f"{v:.7g}") if math.isfinite(v) else None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    return v


def _num(v):
    """A numeric array of v, or None for text, structs, and cells."""
    np = need_numpy()
    if isinstance(v, (str, bytes)) or fields(v) or isinstance(v, list): return None
    try:
        a = np.asarray(v)
    except Exception:
        return None
    if a.dtype.kind not in "biuf": return None
    return a.astype(float)


def _scalar(v):
    a = _num(v)
    return float(a.ravel()[0]) if a is not None and a.size == 1 else None


def _text(v):
    np = need_numpy()
    if isinstance(v, str): return v
    if isinstance(v, np.ndarray) and v.dtype.kind in "US" and v.size == 1: return str(v.ravel()[0])
    return None


# ------------------------------------------------------------------ summary of a folder
def summary(path):
    """One row of the folder list. The parsed machine is not kept for this: a folder may hold many large files."""
    np = need_numpy()
    k = _key(path)
    row = _SUMMARY.get(k)  # one lookup: another request may clear the dict between a test and a read
    if row is not None: return row
    m = load_struct(path, "machine")
    meta = getattr(m, "meta", None)
    get = lambda name: _text(getattr(meta, name, None)) if meta is not None else None
    data = entries(m)
    e = np.array([float(np.ravel(x.energy)[0]) for x in data])
    nd = [np.size(getattr(x, "depths", [])) for x in data]
    dmax = [float(np.max(x.depths)) for x in data if np.size(getattr(x, "depths", []))]
    sad = _scalar(getattr(meta, "SAD", None)) if meta is not None else None
    row = {"file": os.path.basename(path), "name": get("name") or get("machine") or os.path.splitext(os.path.basename(path))[0].split("_", 1)[-1],
           "radiation": get("radiationMode") or os.path.basename(path).split("_", 1)[0], "data_type": get("dataType"),
           "n": len(data), "e_min": float(e.min()) if len(e) else None, "e_max": float(e.max()) if len(e) else None,
           "depth_n_min": min(nd) if nd else None, "depth_n_max": max(nd) if nd else None, "depth_max": max(dmax) if dmax else None,
           "sad": sad, "format": mat_format(path), "size": os.path.getsize(path)}
    row = _clean(row)
    if len(_SUMMARY) > 500: _SUMMARY.clear()
    _SUMMARY[k] = row
    return row


def list_dir(folder):
    """Every .mat file of a folder (not its subfolders) with its summary or the error that stopped reading it."""
    if not os.path.isdir(folder): raise ValueError(f"not a folder: {folder}")
    out = []
    for n in sorted(os.listdir(folder), key=str.lower):
        p = os.path.join(folder, n)
        if not n.lower().endswith(".mat") or not os.path.isfile(p): continue
        try:
            out.append(summary(p))
        except Exception as e:  # one broken or unrelated .mat file must not hide the others
            out.append({"file": n, "error": f"{type(e).__name__}: {e}"})
    return out


# ------------------------------------------------------------------ one machine
def _tree(o, depth=0):
    """Every field of a struct as nodes {name, kind, shape, value or preview, children}, so unknown fields show too."""
    np = need_numpy()
    if fields(o):
        return {"kind": "struct", "children": [dict(name=k, **_tree(getattr(o, k), depth + 1)) for k in fields(o)]} if depth < 6 else {"kind": "struct"}
    if isinstance(o, list):
        return {"kind": "cell", "shape": [len(o)], "children": [dict(name=f"{{{i + 1}}}", **_tree(x, depth + 1)) for i, x in enumerate(o[:50])]}
    t = _text(o)
    if t is not None: return {"kind": "char", "value": t}
    a = _num(o)
    if a is None:
        if isinstance(o, np.ndarray) and o.dtype == object:
            return {"kind": "cell", "shape": list(o.shape), "children": [dict(name=f"{{{i + 1}}}", **_tree(x, depth + 1)) for i, x in enumerate(o.ravel()[:50])]}
        return {"kind": type(o).__name__, "value": str(o)[:200]}
    if a.size == 1: return {"kind": "number", "value": _clean(float(a.ravel()[0]))}
    node = {"kind": "array", "shape": list(a.shape)}
    if a.size:
        f = a[np.isfinite(a)]
        node.update(min=_clean(float(f.min())) if f.size else None, max=_clean(float(f.max())) if f.size else None,
                    preview=_clean(a.ravel()[:8]))
    return node


def tree(path, i=0):
    m = machine(path)
    data = entries(m)
    i = max(0, min(int(i), len(data) - 1))
    out = {"meta": _tree(getattr(m, "meta", SimpleNamespace(_fieldnames=[]))), "data": _tree(data[i]), "index": i,
           "energy": _clean(_scalar(data[i].energy))}
    other = [k for k in fields(m) if k not in ("meta", "data")]
    if other: out["other"] = {"kind": "struct", "children": [dict(name=k, **_tree(getattr(m, k), 1)) for k in other]}
    return out


def _iso_sigma(init, sad):
    """initFocus sigma at the isocenter: interpolated over initFocus.dist at the SAD (first focus if there are several)."""
    np = need_numpy()
    dist, sig = _num(getattr(init, "dist", None)), _num(getattr(init, "sigma", None))
    if dist is None or sig is None or not dist.size or sad is None: return None
    if dist.ndim > 1: dist, sig = dist[0], sig[0]
    dist, sig = np.ravel(dist), np.ravel(sig)
    if dist.size == 1: return float(sig[0])
    o = np.argsort(dist)
    return float(np.interp(sad, dist[o], sig[o]))


def _depth_label(name, col, x, meta):
    """alpha and beta of carbon machines have one column per tissue; the tissue's alphaX and betaX tell them apart."""
    ax, bx = _num(getattr(x, "alphaX", None)), _num(getattr(x, "betaX", None))
    if ax is not None and bx is not None and ax.size > col and bx.size > col:
        return f"{name} (αx {ax.ravel()[col]:g}, βx {bx.ravel()[col]:g})"
    return f"{name}[{col + 1}]"


def view(path):
    """Everything the Machines view draws: per-energy curves over depth and per-energy scalars."""
    np = need_numpy()
    ck = _key(path)
    out = _VIEW.get(ck)  # one lookup, as in summary()
    if out is not None: return out
    m = machine(path)
    meta = getattr(m, "meta", None)
    sad = _scalar(getattr(meta, "SAD", None)) if meta is not None else None
    curves, scal = [], {}
    data = entries(m)
    for j, x in enumerate(data):
        d = _num(getattr(x, "depths", None))
        d = np.ravel(d) if d is not None else np.array([])
        nd = d.size
        c = {"energy": _scalar(x.energy), "offset": offset(x), "depths": d, "Z": zvec(x) if hasattr(x, "Z") else None, "fields": {}}
        for k in fields(x):
            v = getattr(x, k)
            if k in DEPTH_SKIP or k == "Z" and not fields(v): continue
            if fields(v):   # a struct: its depth curves (Z models) and its scalars (initFocus, energySpectrum)
                for kk in fields(v):
                    a = _num(getattr(v, kk))
                    if a is None: continue
                    if a.size == 1: scal.setdefault(f"{k}.{kk}", [None] * len(data))[j] = float(a.ravel()[0])
                    elif nd and a.ndim == 1 and a.size == nd: c["fields"][f"{k}.{kk}"] = a
                continue
            a = _num(v)
            if a is None: continue
            if a.size == 1:
                scal.setdefault(k, [None] * len(data))[j] = float(a.ravel()[0])
            elif nd > 1 and a.shape[0] == nd and a.ndim <= 2:
                if a.ndim == 1: c["fields"][k] = a
                else:
                    for col in range(a.shape[1]): c["fields"][_depth_label(k, col, x, meta)] = a[:, col]
        init = getattr(x, "initFocus", None)
        if init is not None and fields(init):
            s = _iso_sigma(init, sad)
            if s is not None: scal.setdefault("initFocus.sigma@iso", [None] * len(data))[j] = s
        curves.append(c)
    scal.pop("energy", None)
    order = ["range", "peakPos", "offset", "initFocus.sigma@iso", "initFocus.SisFWHMAtIso", "initFocus.emittance"]
    keys = [k for k in order if k in scal] + sorted(k for k in scal if k not in order)
    out = _clean({"summary": summary(path), "energies": [c["energy"] for c in curves], "curves": curves,
                  "scalars": {k: scal[k] for k in keys}, "metrics": [_metrics_of(c) for c in curves]})
    if len(_VIEW) >= _MAX: _VIEW.clear()
    _VIEW[ck] = out
    return out


def _metrics_of(c, use_offset=True):
    np = need_numpy()
    if c["Z"] is None or not np.size(c["depths"]) or np.size(c["Z"]) != np.size(c["depths"]): return None
    z = np.asarray(c["depths"], float) + (c["offset"] if use_offset else 0.0)
    y = np.asarray(c["Z"], float)
    keep = z >= 0
    if keep.sum() < 3 or not np.nanmax(y[keep]) > 0: return None
    return metrics(z[keep], y[keep])


# ------------------------------------------------------------------ two machines
def compare(path_a, path_b, energies=None, norm="max", use_offset=True):
    """Machine B against machine A: for every energy of A the nearest energy of B, the R80 and peak shift (B - A),
    and for the requested energies of A both curves and the difference (B - A) in percent of the maximum of A,
    on the depth grid of A. norm 'max' scales both curves to their own maximum first, 'abs' keeps them as they are."""
    np = need_numpy()
    va, vb = view(path_a), view(path_b)
    eb = np.array([e if e is not None else np.nan for e in vb["energies"]])
    rows, curves = [], []
    want = None if energies is None else [float(e) for e in energies]
    for i, ca in enumerate(va["curves"]):
        if ca["energy"] is None or not len(eb): continue
        j = int(np.nanargmin(np.abs(eb - ca["energy"])))
        cb = vb["curves"][j]
        ma = _metrics_of(ca, use_offset); mb = _metrics_of(cb, use_offset)
        row = {"energy_a": ca["energy"], "energy_b": cb["energy"], "index_a": i, "index_b": j, "a": ma, "b": mb}
        if ma and mb: row.update(d_r80=mb["r80"] - ma["r80"], d_peak=mb["peak"] - ma["peak"], d_bpw80=mb["bpw80"] - ma["bpw80"])
        rows.append(row)
        if want is not None and any(abs(ca["energy"] - w) < 1e-6 for w in want):
            za = np.asarray(ca["depths"], float) + (ca["offset"] if use_offset else 0.0)
            zb = np.asarray(cb["depths"], float) + (cb["offset"] if use_offset else 0.0)
            ya, yb = np.asarray(ca["Z"] or [], float), np.asarray(cb["Z"] or [], float)
            if not (ya.size and yb.size): continue
            if norm == "max": ya, yb = ya / np.nanmax(ya), yb / np.nanmax(yb)
            ob = np.argsort(zb)
            ybi = np.interp(za, zb[ob], yb[ob], left=np.nan, right=np.nan)
            curves.append({"energy_a": ca["energy"], "energy_b": cb["energy"], "za": za, "ya": ya, "zb": zb, "yb": yb,
                           "diff": 100 * (ybi - ya) / np.nanmax(ya)})
    return _clean({"a": va["summary"], "b": vb["summary"], "rows": rows, "curves": curves, "norm": norm, "offset": use_offset})
