"""Structures of an RT Structure Set as masks on an axial dose grid, numpy only (no matplotlib or scikit-image).

Several contours of one structure in one plane combine by even-odd, so a hole drawn as an inner contour stays empty.
Each dose slice inside the structure's z extent takes the contour plane nearest to it, so contour planes and dose
slices need not have the same spacing. Structures without any
contour on the grid are left out. A structure counts as a target when its RTROIInterpretedType is PTV, CTV or GTV, or,
without a type, when its name starts with one of them.

The external contour (the body) is the structure of type EXTERNAL, else one named as in EXTERNAL_NAMES (see external).
"""
import re
import unicodedata

import numpy as np

TARGET_TYPES = ("PTV", "CTV", "GTV")
# names of the body contour without the type EXTERNAL, compared after normalise(); a digit may follow ("Body 2")
EXTERNAL_NAMES = ("body", "external", "outline", "koerper", "aussenkontur", "koerperkontur")


def inside(px, py, poly):
    """Even-odd test of points (px, py) against a polygon [[x, y], ...] (closed implicitly)."""
    poly = np.asarray(poly, float)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    res = np.zeros(px.shape, bool)
    for a, b, c, d in zip(x1, y1, x2, y2):
        if b == d: continue
        cross = (py >= min(b, d)) & (py < max(b, d))
        xc = a + (py - b) * (c - a) / (d - b)
        res ^= cross & (px < xc)
    return res


def normalise(name):
    """Lower case, umlauts and sharp s spelled out, everything but letters and digits dropped: "Außenkontur" ->
    "aussenkontur", "KÖRPER_1" -> "koerper1". Umlauts written as a letter plus a combining mark count, and so do UTF-8
    names that pydicom decoded as Latin-1 because the file declares no SpecificCharacterSet ("KÃ¶rper")."""
    s = str(name)
    try: s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError): pass
    s = unicodedata.normalize("NFC", s).lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")): s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]", "", s)


def is_external_name(name):
    n = normalise(name)
    return any(re.fullmatch(re.escape(e) + r"\d*", n) for e in EXTERNAL_NAMES)


def external(typed):
    """The body of a structure set read with read(..., types=True): (name, "Typ EXTERNAL" or "Name", mask), or None.
    The first structure of type EXTERNAL wins; without one, the first whose name is a body name (any type but a
    target type, so a PTV named "Body" is never the body)."""
    for name, _, mask, roi_type in typed:
        if roi_type == "EXTERNAL": return name, "Typ EXTERNAL", mask
    for name, _, mask, roi_type in typed:
        if roi_type not in TARGET_TYPES and is_external_name(name): return name, "Name", mask
    return None


def read(path, shape, origin, spacing, types=False):
    """[(name, "TARGET", "OAR" or "EXTERNAL", mask [z, y, x])] of the structure set at `path` on the grid (origin,
    spacing x y z). The body (see external) is "EXTERNAL", not an organ. `types`: each entry also carries the
    RTROIInterpretedType in upper case ("" without one)."""
    import pydicom
    ds = pydicom.dcmread(path)
    names = {int(r.ROINumber): str(r.ROIName) for r in getattr(ds, "StructureSetROISequence", [])}
    kinds = {int(o.ReferencedROINumber): str(getattr(o, "RTROIInterpretedType", "") or "").upper()
             for o in getattr(ds, "RTROIObservationsSequence", [])}
    nz, ny, nx = shape
    gx = origin[0] + spacing[0] * np.arange(nx)
    gy = origin[1] + spacing[1] * np.arange(ny)
    PX, PY = np.meshgrid(gx, gy)
    out = []
    gz = origin[2] + spacing[2] * np.arange(nz)
    for rc in getattr(ds, "ROIContourSequence", []):
        num = int(rc.ReferencedROINumber)
        # even-odd only within one contour plane: two planes that round to the same dose slice must not cancel
        planes = {}
        for c in getattr(rc, "ContourSequence", []):
            pts = np.asarray(c.ContourData, float).reshape(-1, 3)
            if len(pts) < 3: continue
            z = round(float(pts[:, 2].mean()), 3)
            planes[z] = planes.get(z, np.zeros((ny, nx), bool)) ^ inside(PX, PY, pts[:, :2])
        mask = np.zeros(shape, bool)
        if planes:
            zs = np.array(sorted(planes))
            # each dose slice takes its nearest contour plane, inside the structure's extent widened by half a plane
            # spacing (half a dose slice for a single plane), so a finer dose grid gets no empty slices between planes
            half = 0.5 * (float(np.median(np.diff(zs))) if len(zs) > 1 else spacing[2])
            for k in np.nonzero((gz >= zs[0] - half - 1e-6) & (gz <= zs[-1] + half + 1e-6))[0]:
                mask[k] = planes[float(zs[np.argmin(np.abs(zs - gz[k]))])]
        if not mask.any(): continue
        name = names.get(num, f"ROI {num}")
        kind = kinds.get(num, "")
        target = kind in TARGET_TYPES or (not kind and name.upper().startswith(TARGET_TYPES))
        out.append((name, "TARGET" if target else "OAR", mask, kind))
    body = external(out)
    if body:
        # every structure of type EXTERNAL is a body outline, not an organ; only the first one is the body mask
        out = [(n, "EXTERNAL" if t == "EXTERNAL" or m is body[2] else k, m, t) for n, k, m, t in out]
    return out if types else [o[:3] for o in out]


def write(path, structures, origin, spacing, frame_uid=None):
    """Synthetic structure set for tests: structures [(name, type, mask [z, y, x])] as one rectangle per slice and
    structure (the bounding box of the mask on that slice, so give box shaped masks)."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from plan_mc import dicom_compat
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = "1.2.840.10008.5.1.4.1.1.481.3", ds.file_meta.MediaStorageSOPInstanceUID, "RTSTRUCT"
    ds.PatientName, ds.PatientID = "Synthetic^Phantom", "SYNTH"
    frame = frame_uid or generate_uid()
    rois, conts, obs = [], [], []
    for i, (name, typ, mask) in enumerate(structures, 1):
        r = Dataset(); r.ROINumber = i; r.ROIName = name; r.ReferencedFrameOfReferenceUID = frame; rois.append(r)
        o = Dataset(); o.ObservationNumber = i; o.ReferencedROINumber = i; o.RTROIInterpretedType = typ; obs.append(o)
        cs = []
        for k in np.nonzero(mask.any(axis=(1, 2)))[0]:
            yy, xx = np.nonzero(mask[k])
            # rectangle through the outer voxel edges, so the centres of the box fall inside
            x0, x1 = origin[0] + spacing[0] * (xx.min() - 0.5), origin[0] + spacing[0] * (xx.max() + 0.5)
            y0, y1 = origin[1] + spacing[1] * (yy.min() - 0.5), origin[1] + spacing[1] * (yy.max() + 0.5)
            z = origin[2] + spacing[2] * k
            c = Dataset(); c.ContourGeometricType = "CLOSED_PLANAR"; c.NumberOfContourPoints = 4
            c.ContourData = [float(v) for v in (x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z)]
            cs.append(c)
        rc = Dataset(); rc.ReferencedROINumber = i; rc.ContourSequence = Sequence(cs); conts.append(rc)
    ds.StructureSetROISequence, ds.ROIContourSequence, ds.RTROIObservationsSequence = Sequence(rois), Sequence(conts), Sequence(obs)
    dicom_compat.save(ds, path)
    return path
