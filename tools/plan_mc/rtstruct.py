"""Structures of an RT Structure Set as masks on an axial dose grid, numpy only (no matplotlib or scikit-image).

Each closed planar contour fills the voxel centres it encloses on the slice nearest to its z; several contours of one
structure on a slice combine by even-odd, so a hole drawn as an inner contour stays empty. Structures without any
contour on the grid are left out. A structure counts as a target when its RTROIInterpretedType is PTV, CTV or GTV, or,
without a type, when its name starts with one of them.
"""
import numpy as np

TARGET_TYPES = ("PTV", "CTV", "GTV")


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


def read(path, shape, origin, spacing):
    """[(name, "TARGET" or "OAR", mask [z, y, x])] of the structure set at `path` on the grid (origin, spacing x y z)."""
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
    for rc in getattr(ds, "ROIContourSequence", []):
        num = int(rc.ReferencedROINumber)
        mask = np.zeros(shape, bool)
        for c in getattr(rc, "ContourSequence", []):
            pts = np.asarray(c.ContourData, float).reshape(-1, 3)
            if len(pts) < 3: continue
            k = int(np.rint((pts[:, 2].mean() - origin[2]) / spacing[2]))
            if not 0 <= k < nz: continue
            mask[k] ^= inside(PX, PY, pts[:, :2])
        if not mask.any(): continue
        name = names.get(num, f"ROI {num}")
        kind = kinds.get(num, "")
        target = kind in TARGET_TYPES or (not kind and name.upper().startswith(TARGET_TYPES))
        out.append((name, "TARGET" if target else "OAR", mask))
    return out


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
