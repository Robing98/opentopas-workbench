"""Loads a MATLAB struct variable (e.g. a matRad machine) from a .mat file, v5/v7 via scipy or v7.3 (HDF5) via h5py.

Returns objects with attribute access like scipy's mat_struct; struct arrays become lists.
"""
from types import SimpleNamespace
import numpy as np


def _h5(f, o):
    import h5py
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
    import h5py
    if o.dtype == h5py.ref_dtype:  # cell array
        return [_h5(f, f[r]) for r in o[()].flatten(order="F")]
    a = np.asarray(o[()])
    if cls == "char": return "".join(chr(c) for c in a.flatten(order="F") if c)
    a = a.T
    if cls == "logical": a = a.astype(bool)
    return a.item() if a.size == 1 else np.squeeze(a)


def load_struct(path, var="machine"):
    try:
        import scipy.io as sio
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False)[var]
    except NotImplementedError:  # MATLAB v7.3
        try:
            import h5py
        except ImportError:
            raise SystemExit("MATLAB v7.3 file: install h5py (python3 -m pip install --user h5py)")
        with h5py.File(path, "r") as f:
            return _h5(f, f[var])
