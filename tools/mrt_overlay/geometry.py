"""Images in patient space (DICOM LPS, mm) as SimpleITK images: RT Dose, CT series, RT Structure Set masks."""
import os

import numpy as np
import SimpleITK as sitk


def from_array(arr_zyx, origin, spacing, direction=None):
    """SimpleITK image of a [z, y, x] array; origin is the centre of voxel 0, spacing (dx, dy, dz)."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr_zyx))
    img.SetOrigin([float(v) for v in origin])
    img.SetSpacing([float(v) for v in spacing])
    if direction is not None: img.SetDirection([float(v) for v in direction])
    return img


def like(arr_zyx, ref):
    """An array on the grid of `ref`."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr_zyx))
    img.CopyInformation(ref)
    return img


def read_rtdose(path, flipped=False):
    """RT Dose as a float image in Gy, with its orientation (any IOP, ascending or descending frames).

    Descending frames are reversed so the spacing stays positive; with `flipped` it returns (image, reversed), and
    an array in the file's frame order needs [::-1] on axis 0 to lie on that image."""
    import pydicom
    ds = pydicom.dcmread(path)
    d = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)
    if d.ndim == 2: d = d[None]
    off = np.atleast_1d(np.asarray(ds.GridFrameOffsetVector, float))
    if len(off) > 2 and np.ptp(np.diff(off)) > 1e-3: raise ValueError("RT Dose: uneven frame spacing")
    iop = np.asarray(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    # ImagePositionPatient is frame 0 for relative offsets (first 0) and for absolute ones (first = its z), so
    # only the offsets relative to the first frame move along the normal
    off = off - off[0]
    dz = float(off[1]) if len(off) > 1 else float(getattr(ds, "SliceThickness", 1) or 1)
    origin = np.asarray(ds.ImagePositionPatient, float)
    rev = dz < 0
    if rev:   # frames run against the normal: flip them so the spacing stays positive
        d, dz, origin = d[::-1], -dz, origin + off[-1] * normal
    spacing = (float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), dz)
    img = from_array(d.astype(np.float32), origin, spacing, np.stack([row, col, normal], 1).ravel())
    return (img, rev) if flipped else img


def write_rtdose(path, dose_zyx, origin, spacing):
    """Axial multi-frame RT Dose in Gy (uint32 with DoseGridScaling), for tests and synthetic runs."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    d = np.asarray(dose_zyx, float)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID, "RTDOSE"
    scale = max(float(d.max()), 1e-30) / (2 ** 32 - 1)
    nz, ny, nx = d.shape
    ds.ImagePositionPatient = [float(v) for v in origin]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
    ds.GridFrameOffsetVector = [float(k * spacing[2]) for k in range(nz)]
    ds.NumberOfFrames, ds.Rows, ds.Columns = nz, ny, nx
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 32, 32, 31, 0
    ds.DoseUnits, ds.DoseType, ds.DoseGridScaling = "GY", "EFFECTIVE", f"{scale:.10g}"
    ds.PixelData = np.rint(d / scale).astype(np.uint32).tobytes()
    pydicom.dcmwrite(path, ds, enforce_file_format=True)


def read_series(folder):
    """The DICOM image series in `folder` (the one with most files when there are several)."""
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(folder)
    if not ids: raise SystemExit(f"no DICOM image series in {folder}")
    files = max((reader.GetGDCMSeriesFileNames(folder, i) for i in ids), key=len)
    reader.SetFileNames(files)
    return reader.Execute()


def read_image(path):
    """A NIfTI/MHA file or a DICOM series folder."""
    return read_series(path) if os.path.isdir(path) else sitk.ReadImage(path)


def _fill_polygon(mask2d, poly_ij):
    """Even-odd fill of one polygon given in continuous (i, j) = (column, row) indices, into mask2d[row, col]."""
    ny, nx = mask2d.shape
    i, j = poly_ij[:, 0], poly_ij[:, 1]
    j0, j1 = max(0, int(np.floor(j.min()))), min(ny - 1, int(np.ceil(j.max())))
    i2, j2 = np.roll(i, -1), np.roll(j, -1)
    for r in range(j0, j1 + 1):
        cross = (j <= r) != (j2 <= r)
        if not cross.any(): continue
        xs = np.sort(i[cross] + (r - j[cross]) * (i2[cross] - i[cross]) / (j2[cross] - j[cross]))
        for a, b in zip(xs[0::2], xs[1::2]):
            # voxel centres in [a, b): half open like the rows, so the filled area equals the polygon area
            c0, c1 = max(0, int(np.ceil(a))), min(nx - 1, int(np.ceil(b)) - 1)
            if c1 >= c0: mask2d[r, c0:c1 + 1] ^= True
    return mask2d


def contour_mask(contours_mm, ref):
    """Mask on the grid of `ref` from closed planar contours, each an (n, 3) array in patient mm.

    A contour fills the slice nearest to its plane. Holes and islands follow the even-odd rule within a slice."""
    nx, ny, nz = ref.GetSize()
    out = np.zeros((nz, ny, nx), bool)
    for pts in contours_mm:
        idx = np.array([ref.TransformPhysicalPointToContinuousIndex([float(v) for v in p]) for p in pts])
        k = int(round(float(np.median(idx[:, 2]))))
        if 0 <= k < nz and len(idx) >= 3:
            layer = np.zeros((ny, nx), bool)
            out[k] ^= _fill_polygon(layer, idx[:, :2])
    return out


def read_rtstruct(path):
    """{ROI name: (RTROIInterpretedType, [contour arrays in mm])} of an RT Structure Set."""
    import pydicom
    ds = pydicom.dcmread(path)
    names = {int(r.ROINumber): str(r.ROIName) for r in ds.StructureSetROISequence}
    kinds = {int(o.ReferencedROINumber): str(getattr(o, "RTROIInterpretedType", "") or "").upper()
             for o in getattr(ds, "RTROIObservationsSequence", [])}
    out = {}
    for c in getattr(ds, "ROIContourSequence", []):
        n = int(c.ReferencedROINumber)
        polys = [np.asarray(s.ContourData, float).reshape(-1, 3) for s in getattr(c, "ContourSequence", [])
                 if str(getattr(s, "ContourGeometricType", "CLOSED_PLANAR")).upper() == "CLOSED_PLANAR"]
        out[names.get(n, str(n))] = (kinds.get(n, ""), polys)
    return out


def pick_ptv(structures, name=None):
    """The PTV to draw: the named ROI, else the one ROI of type PTV or with PTV in its name (largest if several)."""
    if name:
        if name not in structures: raise SystemExit(f"ROI {name!r} is not in the RT Structure Set")
        return name
    cand = [n for n, (k, _) in structures.items() if k == "PTV" or "ptv" in n.lower()]
    if not cand: return None
    return max(cand, key=lambda n: sum(len(p) for p in structures[n][1]))


def brain_points(mask_img, step=4):
    """Physical points of every `step`-th voxel of a mask, for displacement measures."""
    a = sitk.GetArrayViewFromImage(mask_img)
    z, y, x = np.nonzero(a[::step, ::step, ::step])
    return np.array([mask_img.TransformIndexToPhysicalPoint((int(i * step), int(j * step), int(k * step)))
                     for i, j, k in zip(x, y, z)], float)


def displacement_mm(t_a, t_b, points):
    """RMS and maximum distance between two transforms over points (mm)."""
    if len(points) == 0: return float("nan"), float("nan")
    d = np.array([np.subtract(t_a.TransformPoint(tuple(p)), t_b.TransformPoint(tuple(p))) for p in points])
    n = np.linalg.norm(d, axis=1)
    return float(np.sqrt(np.mean(n ** 2))), float(n.max())
