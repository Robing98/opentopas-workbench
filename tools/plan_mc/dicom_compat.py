"""pydicom 2.4 and 3 side by side: Ubuntu ships 2.4, pip gives 3, and the chain must run on both.

pydicom 3 writes a conformant file with save_as(enforce_file_format=True); pydicom 2 does the same with
write_like_original=False, but needs the encoding flags on the dataset, which pydicom 3 deprecates.
"""
INSTALL_HINT = 'python3 -m pip install --user --break-system-packages "pydicom>=3"'


def check():
    """Version string of pydicom, or SystemExit with the install line when it is missing or older than 2.4."""
    try:
        import pydicom
    except ImportError:
        raise SystemExit(f"pydicom is missing: {INSTALL_HINT}")
    major, minor = (int(p) for p in pydicom.__version__.split(".")[:2])
    if (major, minor) < (2, 4):
        raise SystemExit(f"pydicom {pydicom.__version__} is too old (2.4 or newer): {INSTALL_HINT}")
    return pydicom.__version__


def save(ds, path):
    try:
        ds.save_as(path, enforce_file_format=True)
    except TypeError:   # pydicom 2: no enforce_file_format
        ts = ds.file_meta.TransferSyntaxUID
        ds.is_little_endian = ts != "1.2.840.10008.1.2.2"
        ds.is_implicit_VR = ts == "1.2.840.10008.1.2"
        ds.save_as(path, write_like_original=False)
