"""cacheF from the viewer itself, when the aligned volumes of the case are gone.

The viewer's page (Verlauf.html) and its data folder (Verlauf-Daten) hold every aligned volume as an 8-bit mosaic of
its axial slices, on the page's grid: dim [nx, ny, nz], spacing sp, cols tiles per row, slices z by z (build_page.py
of the Timeline Viewer writes them so). Decoded, they give the cacheF that the commands viewer and pet read:

    log.json        the viewer's own copy of the cacheF log (Skripte/stand/cacheF_log.json), size and spacing of the page
    <seq>_<day>.npy uint8 [z, y, x], one per sequence and exam the page shows
    ref_t1.nii.gz   the 3D T1 of the reference day (refday), spacing sp, origin 0, unit direction
    mask.npy        the brain mask from mask_all.js (lossless): 0 outside, 1 inside

The volumes are lossy (WEBP, quality 80) and 8 bit. That is enough for the mutual-information registration onto the
CT and the PET, and they lie exactly on the grid the page shows, which is what the layers must match. A full rebuild
from the DICOM exams stays a later option.
"""
import base64
import datetime
import io
import json
import os
import re

import numpy as np

VOL = re.compile(r'MRT\.vol\("([^"]*)","([^"]*)","data:image/(\w+);base64,([^"]*)"\)')
DATA = re.compile(r"const\s+D\s*=\s*")


class CacheError(Exception):
    pass


def page_data(page):
    """The data object the page embeds (const D = {...};)."""
    with open(page, encoding="utf-8") as fh: text = fh.read()
    m = DATA.search(text)
    if not m: raise CacheError(f"{os.path.basename(page)}: keine Daten der Seite (const D) gefunden")
    try: d, _ = json.JSONDecoder().raw_decode(text, m.end())
    except ValueError as e: raise CacheError(f"{os.path.basename(page)}: Daten der Seite unlesbar ({e})")
    for k in ("dim", "sp", "tracks"):
        if k not in d: raise CacheError(f"{os.path.basename(page)}: Daten ohne {k}")
    return d


def read_js(path):
    """(track, day, image bytes) of one volume file of the data folder."""
    with open(path, encoding="utf-8") as fh: m = VOL.search(fh.read())
    if not m: raise CacheError(f"{os.path.basename(path)}: kein MRT.vol(...)")
    return m.group(1), m.group(2), base64.b64decode(m.group(4))


def unmosaic(image_bytes, dim, cols):
    """The mosaic back to a volume [z, y, x] uint8."""
    from PIL import Image
    nx, ny, nz = (int(v) for v in dim)
    m = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("L"))
    rows = -(-nz // cols)
    if m.shape != (rows * ny, cols * nx):
        raise CacheError(f"Mosaik {m.shape[1]}x{m.shape[0]} passt nicht zu dim {nx}x{ny}x{nz} mit {cols} Spalten")
    v = np.empty((nz, ny, nx), np.uint8)
    for z in range(nz):
        v[z] = m[(z // cols) * ny:(z // cols + 1) * ny, (z % cols) * nx:(z % cols + 1) * nx]
    return v


def spacing3(sp):
    return [float(sp)] * 3 if isinstance(sp, (int, float)) else [float(x) for x in sp]


def ref_day(days, refday):
    """The T1 day nearest to the reference day; the newest when the page names none."""
    if not days: raise CacheError("die Seite zeigt keine T1: ohne sie kein ref_t1.nii.gz")
    if not refday: return days[-1]
    try: r = datetime.date.fromisoformat(refday)
    except ValueError: return days[-1]
    return min(days, key=lambda d: (abs((datetime.date.fromisoformat(d) - r).days), d))


def _save_npy(path, arr):
    tmp = path + ".part"
    with open(tmp, "wb") as fh: np.save(fh, arr)
    os.replace(tmp, path)


def build(page, daten, stand_log, out, say=print):
    """Writes cacheF into `out` and returns a summary. log.json comes last: a folder without it is no cacheF, so a run
    that breaks off leaves nothing the search would take."""
    import SimpleITK as sitk
    say("## STEP read")
    d = page_data(page)
    dim, sp, cols = [int(v) for v in d["dim"]], d["sp"], int(d.get("cols") or 12)
    try:
        with open(stand_log, encoding="utf-8") as fh: log = json.load(fh)
    except (OSError, ValueError) as e:
        raise CacheError(f"{os.path.basename(stand_log)}: {e}")
    if not isinstance(log, dict): raise CacheError(f"{os.path.basename(stand_log)}: kein Objekt")
    todo = [(k, e["day"]) for k, t in sorted(d["tracks"].items()) for e in t.get("exams", [])]
    os.makedirs(out, exist_ok=True)
    say("## STEP volumes")
    done, missing, t1 = [], [], {}
    for i, (track, day) in enumerate(todo, 1):
        say(f"## VOL {i}/{len(todo)}")
        js = os.path.join(daten, f"{track}_{day}.js")
        if not os.path.isfile(js):
            missing.append(f"{track}_{day}"); continue
        _, _, img = read_js(js)
        v = unmosaic(img, dim, cols)
        _save_npy(os.path.join(out, f"{track}_{day}.npy"), v)
        done.append(f"{track} {day}")
        if track == "t1": t1[day] = v
    if not done: raise CacheError("kein Volumen der Seite im Datenordner gefunden")
    say("## STEP mask")
    mjs = os.path.join(daten, "mask_all.js")
    if os.path.isfile(mjs):
        _save_npy(os.path.join(out, "mask.npy"), (unmosaic(read_js(mjs)[2], dim, cols) > 0).astype(np.uint8))
    say("## STEP write")
    day = ref_day(sorted(t1), d.get("refday"))
    img = sitk.GetImageFromArray(t1[day])
    img.SetSpacing(spacing3(sp))
    sitk.WriteImage(img, os.path.join(out, "ref_t1.nii.gz"))
    # the page's grid wins over the copy of the log: the volumes are on the page's grid
    log.update(size=dim, spacing=sp)
    log["aus_viewer"] = {"seite": os.path.basename(page), "ref_t1": day, "verlustbehaftet": True}
    tmp = os.path.join(out, "log.json.part")
    with open(tmp, "w", encoding="utf-8") as fh: json.dump(log, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(out, "log.json"))
    return {"volumen": len(done), "fehlen": missing, "ref_t1": day, "maske": os.path.isfile(os.path.join(out, "mask.npy"))}
