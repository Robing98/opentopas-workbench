"""Synthetic DICOM CT and RT Dose for public tests: slab phantoms, matRad phantoms, TPS-like dose grids.

Arrays are indexed [z, y, x] (slice, row, column), as DICOM stores them. `origin` is the centre of the first voxel
in DICOM patient coordinates (mm), `spacing` is (dx, dy, dz) in mm. Orientation is always the standard axial one
(ImageOrientationPatient 1 0 0 0 1 0): TOPAS ignores PatientPosition anyway, the plan carries it (iec.py).
"""
import os

import numpy as np

from plan_mc import dicom_compat

CT_CLASS = "1.2.840.10008.5.1.4.1.1.2"
RTDOSE_CLASS = "1.2.840.10008.5.1.4.1.1.481.2"


def _base(path, sop_class, modality, study_uid, frame_uid, series_uid):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.PatientName, ds.PatientID = "Synthetic^Phantom", "SYNTH"
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    return ds


def write_ct(folder, hu, origin, spacing, position="HFS", study_uid=None, frame_uid=None):
    """One file per slice, int16 HU with slope 1 and intercept 0. Returns (study_uid, frame_uid)."""
    from pydicom.uid import generate_uid
    hu = np.asarray(np.rint(hu), dtype=np.int16)
    nz, ny, nx = hu.shape
    os.makedirs(folder, exist_ok=True)
    study_uid, frame_uid, series = study_uid or generate_uid(), frame_uid or generate_uid(), generate_uid()
    for k in range(nz):
        p = os.path.join(folder, f"CT_{k:04d}.dcm")
        ds = _base(p, CT_CLASS, "CT", study_uid, frame_uid, series)
        ds.PatientPosition = position
        ds.SeriesNumber, ds.InstanceNumber = 1, k + 1
        ds.ImagePositionPatient = [float(origin[0]), float(origin[1]), float(origin[2] + k * spacing[2])]
        ds.SliceLocation = float(origin[2] + k * spacing[2])
        ds.SliceThickness = float(spacing[2])
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]       # row spacing (y), column spacing (x)
        ds.Rows, ds.Columns = ny, nx
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = 0, 1, "HU"
        ds.PixelData = hu[k].tobytes()
        dicom_compat.save(ds, p)
    return study_uid, frame_uid


def write_rtdose(path, dose_gy, origin, spacing, study_uid=None, frame_uid=None, plan_uid=None, dose_type="PHYSICAL",
                 summation="PLAN"):
    """Multi-frame RT Dose, uint32 with DoseGridScaling, frames along +z."""
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    from pydicom.uid import generate_uid
    d = np.asarray(dose_gy, float)
    nz, ny, nx = d.shape
    scale = max(float(d.max()), 1e-30) / (2 ** 32 - 1)
    ds = _base(path, RTDOSE_CLASS, "RTDOSE", study_uid or generate_uid(), frame_uid or generate_uid(), generate_uid())
    ds.ImagePositionPatient = [float(v) for v in origin]
    ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
    ds.SliceThickness = float(spacing[2])
    ds.GridFrameOffsetVector = [float(k * spacing[2]) for k in range(nz)]
    ds.FrameIncrementPointer = 0x3004000C
    ds.NumberOfFrames, ds.Rows, ds.Columns = nz, ny, nx
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 32, 32, 31, 0
    ds.DoseUnits, ds.DoseType, ds.DoseSummationType, ds.DoseGridScaling = "GY", dose_type, summation, f"{scale:.10g}"
    if plan_uid:
        r = Dataset(); r.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.8"; r.ReferencedSOPInstanceUID = plan_uid
        ds.ReferencedRTPlanSequence = Sequence([r])
    ds.PixelData = np.rint(d / scale).astype(np.uint32).tobytes()
    dicom_compat.save(ds, path)


def read_rtdose(path):
    """(dose in Gy [z, y, x], origin of the first voxel centre, spacing (dx, dy, dz))."""
    import pydicom
    ds = pydicom.dcmread(path)
    d = ds.pixel_array.astype(float) * float(ds.DoseGridScaling)
    if d.ndim == 2: d = d[None]
    off = np.atleast_1d(np.asarray(ds.GridFrameOffsetVector, float)).tolist()   # one frame: a single value
    dz = off[1] - off[0] if len(off) > 1 else float(getattr(ds, "SliceThickness", 0) or 0)
    if len(off) > 2 and np.ptp(np.diff(off)) > 1e-3: raise ValueError("RT Dose: uneven frame spacing")
    # TOPAS clones the grid as axial, ascending z: refuse layouts that would put the voxels elsewhere
    iop = np.asarray(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
    if not np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=1e-4): raise ValueError(f"RT Dose: orientation {iop.tolist()} not supported")
    if dz < 0: raise ValueError("RT Dose: descending GridFrameOffsetVector not supported")
    x0, y0, z0 = (float(v) for v in ds.ImagePositionPatient)
    return d, (x0, y0, z0 + off[0]), (float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), dz)


def read_ct(folder):
    """(HU [z, y, x], origin, spacing) from a folder of axial CT slices (any file the reader accepts as CT)."""
    import pydicom
    sl = []
    for n in sorted(os.listdir(folder)):
        try: ds = pydicom.dcmread(os.path.join(folder, n))
        except Exception: continue
        if getattr(ds, "Modality", "") == "CT": sl.append(ds)
    if not sl: raise ValueError("CT folder: no CT slices")
    sl.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    hu = np.stack([s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept) for s in sl])
    z = [float(s.ImagePositionPatient[2]) for s in sl]
    dz = z[1] - z[0] if len(z) > 1 else float(sl[0].SliceThickness)
    x0, y0, _ = (float(v) for v in sl[0].ImagePositionPatient)
    return hu, (x0, y0, z[0]), (float(sl[0].PixelSpacing[1]), float(sl[0].PixelSpacing[0]), dz)


def slab_phantom(shape, background=0, slabs=()):
    """HU array [z, y, x] with slabs along y (rows): slabs = [(row_from, row_to, hu), ...]."""
    hu = np.full(shape, background, dtype=np.int16)
    for a, b, v in slabs: hu[:, a:b, :] = v
    return hu
