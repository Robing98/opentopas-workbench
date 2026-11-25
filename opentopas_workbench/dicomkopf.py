"""DICOM headers with the standard library alone: the workbench server has no pydicom.

Only what setting up a case needs: Modality, UIDs, and from an RT plan the fractions and the prescription. The reader
stops at the pixel data, so a CT or PET slice costs a few kilobytes of reading, not its whole size. It reads explicit
and implicit VR little endian, the two transfer syntaxes of every file this project meets; a big-endian or
deflated file gives None, and the setup then asks instead of guessing.

The writer exists for the demo case and the tests: synthetic files only.
"""
import struct

# (group, element) of the tags this module knows by name
TAGS = {
    "TransferSyntaxUID": (0x0002, 0x0010),
    "SOPClassUID": (0x0008, 0x0016),
    "SOPInstanceUID": (0x0008, 0x0018),
    "StudyDate": (0x0008, 0x0020),
    "SeriesDate": (0x0008, 0x0021),
    "Modality": (0x0008, 0x0060),
    "SeriesDescription": (0x0008, 0x103E),
    "ReferencedSOPInstanceUID": (0x0008, 0x1155),
    "PatientID": (0x0010, 0x0020),
    "SliceThickness": (0x0018, 0x0050),
    "StudyInstanceUID": (0x0020, 0x000D),
    "SeriesInstanceUID": (0x0020, 0x000E),
    "InstanceNumber": (0x0020, 0x0013),
    "ImagePositionPatient": (0x0020, 0x0032),
    "FrameOfReferenceUID": (0x0020, 0x0052),
    "Rows": (0x0028, 0x0010),
    "Columns": (0x0028, 0x0011),
    "PixelSpacing": (0x0028, 0x0030),
    "RTPlanLabel": (0x300A, 0x0002),
    "DoseReferenceSequence": (0x300A, 0x0010),
    "DoseReferenceType": (0x300A, 0x0020),
    "DeliveryMaximumDose": (0x300A, 0x0023),
    "TargetPrescriptionDose": (0x300A, 0x0026),
    "FractionGroupSequence": (0x300A, 0x0070),
    "NumberOfFractionsPlanned": (0x300A, 0x0078),
    "RadiationType": (0x300A, 0x00C6),
    "IonBeamSequence": (0x300A, 0x03A2),
    "ReferencedFrameOfReferenceSequence": (0x3006, 0x0010),
    "StructureSetROISequence": (0x3006, 0x0020),
    "ROIName": (0x3006, 0x0026),
    "ReferencedStructureSetSequence": (0x300C, 0x0060),
}
NAMES = {v: k for k, v in TAGS.items()}
PIXEL = (0x7FE0, 0x0010)
EXPLICIT_LE, IMPLICIT_LE = "1.2.840.10008.1.2.1", "1.2.840.10008.1.2"
LONG_VR = {b"OB", b"OW", b"OF", b"OD", b"OL", b"OV", b"SQ", b"UC", b"UR", b"UT", b"UN"}
SEQ_IMPLICIT = {v for k, v in TAGS.items() if k.endswith("Sequence")}   # implicit VR carries no "SQ": known by tag
UNDEFINED = 0xFFFFFFFF
ON_ERROR = None                  # called with the path of a file that exists but cannot be read (a search counts them)


def _value(vr, raw):
    if vr in (b"US",): return struct.unpack("<H", raw[:2])[0] if len(raw) >= 2 else None
    if vr in (b"UL",): return struct.unpack("<I", raw[:4])[0] if len(raw) >= 4 else None
    if vr in (b"FD",): return struct.unpack("<d", raw[:8])[0] if len(raw) >= 8 else None
    if vr in (b"FL",): return struct.unpack("<f", raw[:4])[0] if len(raw) >= 4 else None
    s = raw.decode("latin-1").strip("\x00 ")
    if vr in (b"DS", b"IS"):
        parts = [p.strip() for p in s.split("\\") if p.strip()]
        try: nums = [float(p) if vr == b"DS" else int(p) for p in parts]
        except ValueError: return s
        return nums[0] if len(nums) == 1 else nums
    return s


class _Reader:
    def __init__(self, buf, explicit):
        self.b, self.explicit = buf, explicit

    def element(self, i):
        """(tag, vr, length, start of value) at i, or None past the end."""
        b = self.b
        if i + 8 > len(b): return None
        g, e = struct.unpack_from("<HH", b, i)
        tag = (g, e)
        if g == 0xFFFE:   # item, item end, sequence end: never explicit
            return tag, None, struct.unpack_from("<I", b, i + 4)[0], i + 8
        if self.explicit or g == 0x0002:
            vr = b[i + 4:i + 6]
            if vr in LONG_VR:
                if i + 12 > len(b): return None
                return tag, vr, struct.unpack_from("<I", b, i + 8)[0], i + 12
            return tag, vr, struct.unpack_from("<H", b, i + 6)[0], i + 8
        vr = b"SQ" if tag in SEQ_IMPLICIT else b"UN"
        return tag, vr, struct.unpack_from("<I", b, i + 4)[0], i + 8

    def dataset(self, i, end):
        """{name: value} from i to end; stops at the pixel data or an item end."""
        out = {}
        while i < end:
            el = self.element(i)
            if el is None: break
            tag, vr, n, j = el
            if tag == PIXEL: break
            if tag == (0xFFFE, 0xE00D): return out, j
            if n == UNDEFINED and vr == b"UN": vr = b"SQ"   # an unknown sequence of undefined length (implicit VR)
            if vr == b"SQ":
                items, j = self.sequence(j, n)
                if tag in NAMES: out[NAMES[tag]] = items
                i = j; continue
            if n == UNDEFINED: break      # an encapsulated value we do not need: nothing after it is read
            raw = self.b[j:j + n]
            if tag in NAMES:
                kind = vr if self.explicit or tag[0] == 0x0002 else _implicit_vr(tag)
                out[NAMES[tag]] = _value(kind, raw)
            i = j + n
        return out, i

    def sequence(self, i, n):
        items, end = [], (len(self.b) if n == UNDEFINED else i + n)
        while i < end:
            el = self.element(i)
            if el is None: break
            tag, _, ln, j = el
            if tag == (0xFFFE, 0xE0DD): return items, j
            if tag != (0xFFFE, 0xE000): break
            ds, k = self.dataset(j, len(self.b) if ln == UNDEFINED else j + ln)
            items.append(ds)
            i = k if ln == UNDEFINED else j + ln
        return items, end if n != UNDEFINED else i


def _implicit_vr(tag):
    """The VR of a known tag in an implicit file, from the standard's dictionary."""
    name = NAMES.get(tag, "")
    if name in ("Rows", "Columns"): return b"US"
    if name in ("NumberOfFractionsPlanned", "InstanceNumber"): return b"IS"
    if name in ("DeliveryMaximumDose", "TargetPrescriptionDose", "PixelSpacing", "ImagePositionPatient",
                "SliceThickness"): return b"DS"
    return b"LO"


def read(path, limit=1 << 20):
    """The known tags of a DICOM file as a dict, or None for a file that is not DICOM or not readable here.
    Reads at most `limit` bytes: an RT plan with many beams fits, a slice stops at its pixels anyway."""
    try:
        with open(path, "rb") as fh: buf = fh.read(limit)
    except OSError:
        if ON_ERROR: ON_ERROR(path)
        return None
    if len(buf) < 132 or buf[128:132] != b"DICM": return None
    meta, i = _meta(buf)
    ts = meta.get("TransferSyntaxUID", EXPLICIT_LE)
    if ts not in (EXPLICIT_LE, IMPLICIT_LE): return None
    ds, _ = _Reader(buf, ts == EXPLICIT_LE).dataset(i, len(buf))
    ds.update(meta)
    return ds


def _meta(buf):
    """The file meta group (always explicit little endian) and where the data set starts."""
    r, i, meta = _Reader(buf, True), 132, {}
    while i + 8 <= len(buf) and struct.unpack_from("<H", buf, i)[0] == 0x0002:
        tag, vr, n, j = r.element(i)
        if tag in NAMES: meta[NAMES[tag]] = _value(vr, buf[j:j + n])
        i = j + n
    return meta, i


def plan_facts(ds):
    """From an RT (Ion) Plan: fractions, prescription in Gy (RBE for ions), particle and the referenced structure set.
    The prescription is TargetPrescriptionDose of the target dose reference, else its DeliveryMaximumDose; None when the
    plan names neither, so the setup asks instead of taking a number from elsewhere."""
    if not ds or ds.get("Modality") != "RTPLAN": return None
    fx = None
    for g in ds.get("FractionGroupSequence") or []:
        if isinstance(g.get("NumberOfFractionsPlanned"), int): fx = g["NumberOfFractionsPlanned"]; break
    rx = None
    refs = ds.get("DoseReferenceSequence") or []
    for want in ("TARGET", None):
        for r in refs:
            if want and r.get("DoseReferenceType") != want: continue
            v = r.get("TargetPrescriptionDose")
            if isinstance(v, float) and v > 0: rx = v; break
        if rx: break
    if rx is None:
        for r in refs:
            v = r.get("DeliveryMaximumDose")
            if isinstance(v, float) and v > 0: rx = v; break
    particle = None
    for b in ds.get("IonBeamSequence") or []:
        if b.get("RadiationType"): particle = b["RadiationType"].lower(); break
    rs = None
    for r in ds.get("ReferencedStructureSetSequence") or []:
        rs = r.get("ReferencedSOPInstanceUID") or rs
    return {"fraktionen": fx, "verschreibung_gy_rbe": rx, "teilchen": particle, "rtstruct_uid": rs,
            "frame_of_reference_uid": ds.get("FrameOfReferenceUID")}


def frame_uid(ds):
    """The frame of reference of a data set: its own FrameOfReferenceUID (CT, MR, PT, RT plan, RT dose), else the first
    one an RT structure set names in ReferencedFrameOfReferenceSequence."""
    if not ds: return None
    if ds.get("FrameOfReferenceUID"): return ds["FrameOfReferenceUID"]
    for r in ds.get("ReferencedFrameOfReferenceSequence") or []:
        if r.get("FrameOfReferenceUID"): return r["FrameOfReferenceUID"]
    return None


# ------------------------------------------------------------------ writer (synthetic files only)
def _el(tag, vr, value):
    g, e = TAGS[tag] if isinstance(tag, str) else tag
    if vr == b"SQ":
        body = b"".join(struct.pack("<HHI", 0xFFFE, 0xE000, len(it)) + it for it in (_ds(x) for x in value))
        return struct.pack("<HH", g, e) + b"SQ\x00\x00" + struct.pack("<I", len(body)) + body
    if isinstance(value, bytes): raw = value
    elif vr == b"US": raw = struct.pack("<H", value)
    else:
        raw = ("\\".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value)
               if isinstance(value, (list, tuple)) else (f"{value:g}" if isinstance(value, float) else str(value))).encode()
        if len(raw) % 2: raw += b"\x00" if vr == b"UI" else b" "
    if vr in LONG_VR:
        return struct.pack("<HH", g, e) + vr + b"\x00\x00" + struct.pack("<I", len(raw)) + raw
    return struct.pack("<HH", g, e) + vr + struct.pack("<H", len(raw)) + raw


VR = {"Modality": b"CS", "DoseReferenceType": b"CS", "RadiationType": b"CS", "Rows": b"US", "Columns": b"US",
      "NumberOfFractionsPlanned": b"IS", "InstanceNumber": b"IS", "StudyDate": b"DA", "SeriesDate": b"DA",
      "DeliveryMaximumDose": b"DS", "TargetPrescriptionDose": b"DS", "PixelSpacing": b"DS",
      "ImagePositionPatient": b"DS", "SliceThickness": b"DS", "SeriesDescription": b"LO", "PatientID": b"LO",
      "RTPlanLabel": b"SH", "ROIName": b"LO"}


def _ds(d):
    out = []
    for name in sorted(d, key=lambda n: TAGS[n]):
        v = d[name]
        vr = b"SQ" if name.endswith("Sequence") else VR.get(name, b"UI")
        out.append(_el(name, vr, v))
    return b"".join(out)


def write(path, d, pixels=b""):
    """A synthetic explicit-VR-little-endian file from {name: value}; sequences as lists of dicts."""
    meta = _el("TransferSyntaxUID", b"UI", EXPLICIT_LE)
    data = _ds(d)
    if pixels: data += struct.pack("<HH", *PIXEL) + b"OW\x00\x00" + struct.pack("<I", len(pixels)) + pixels
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 128 + b"DICM" + meta + data)
