"""Pick MRI series from DICOM headers: 3D T1 before and after contrast, FLAIR. Reports kinds, never names."""
import os

T1_WORDS = ("t1", "mprage", "vibe", "ffe", "spgr", "bravo", "tfe")
FLAIR_WORDS = ("flair", "dark-fluid", "dark fluid", "tirm")
SKIP_WORDS = ("sub", "mip", "localizer", "scout", "survey", "adc", "diff", "dwi", "swi", "perf", "haste")
KM_WORDS = ("km", "gd", "kontrast", "+c", "post", "contrast")


def header(path):
    import pydicom
    try: return pydicom.dcmread(path, stop_before_pixels=True)
    except Exception: return None


def kind(ds):
    """'t1_km', 't1', 'flair' or None, from SeriesDescription, ProtocolName, ImageType and ContrastBolusAgent."""
    text = " ".join(str(getattr(ds, k, "") or "") for k in ("SeriesDescription", "ProtocolName")).lower()
    words = text.replace("_", " ").split()
    image_type = [str(v).upper() for v in getattr(ds, "ImageType", [])]
    # "mpr" as a whole word is a scanner reformat; inside "mprage" it is the sequence
    if any(w in text for w in SKIP_WORDS) or "mpr" in words or "MPR" in image_type or "RESAMPLED" in image_type:
        return None
    if any(w in text for w in FLAIR_WORDS): return "flair"
    if any(w in text for w in T1_WORDS):
        # some vendors leave ContrastBolusAgent empty after contrast: the series name counts too
        km = bool(str(getattr(ds, "ContrastBolusAgent", "") or "").strip()) or any(
            w in KM_WORDS or w.startswith("+c") for w in words)
        return "t1_km" if km else "t1"
    return None


def scan(folder):
    """One entry per series under `folder`: number, kind, date, slice thickness, files, suitable (<= 3 mm, 3D)."""
    first = {}
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            ds = header(os.path.join(root, f))
            if ds is None or not hasattr(ds, "SeriesInstanceUID"): continue
            uid = str(ds.SeriesInstanceUID)
            entry = first.setdefault(uid, {"ds": ds, "dir": root, "files": 0})
            entry["files"] += 1
    out = []
    for uid, e in first.items():
        ds = e["ds"]
        thick = float(getattr(ds, "SliceThickness", 0) or 0)
        frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        out.append({"series_number": int(getattr(ds, "SeriesNumber", 0) or 0), "kind": kind(ds),
                    "date": str(getattr(ds, "SeriesDate", "") or getattr(ds, "StudyDate", "") or ""),
                    "slice_thickness_mm": thick, "images": max(e["files"], frames),
                    "suitable": 0 < thick <= 3.0 and max(e["files"], frames) >= 40, "dir": e["dir"]})
    return sorted(out, key=lambda r: (r["date"], r["series_number"]))


def pick(entries):
    """Per date and kind the suitable series with the thinnest slices."""
    best = {}
    for e in entries:
        if not e["kind"] or not e["suitable"]: continue
        k = (e["date"], e["kind"])
        if k not in best or e["slice_thickness_mm"] < best[k]["slice_thickness_mm"]: best[k] = e
    return best
