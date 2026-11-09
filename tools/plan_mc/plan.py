"""Plan model for the spot-by-spot TOPAS recalculation: fields, energy layers, spots.

Readers: DICOM RT Ion Plan (pydicom) and a synthetic RT Ion Plan writer for tests on public data.
Spot positions are in the IEC gantry system at the isocenter plane, in mm (DICOM ScanSpotPositionMap).
Spot weights stay in the unit of the plan (MU or NP for DICOM, particles for matRad); the beam model converts MU to
particles.
"""
from dataclasses import dataclass, field
import re
from typing import List, Optional


@dataclass
class Layer:
    energy: float                      # nominal energy [MeV]
    x: List[float]                     # spot positions at the isocenter plane, IEC gantry X [mm]
    y: List[float]                     # IEC gantry Y [mm]
    w: List[float]                     # spot weights (MU or particles, see Field.unit)
    tune_id: str = ""
    spot_size: Optional[tuple] = None  # (FWHM x, FWHM y) at the isocenter as stated in the plan [mm], information only


@dataclass
class RangeShifter:
    rs_id: str
    wet: float = 0.0                   # water equivalent thickness [mm] as stated in the plan
    iso_distance: Optional[float] = 0.0  # isocenter to range shifter distance [mm] (downstream face, DICOM
                                         # definition); None when the plan does not state it


@dataclass
class Field:
    name: str
    number: int
    gantry: float                      # IEC gantry angle [deg]
    couch: float                       # IEC patient support angle [deg]
    iso: tuple                         # isocenter in DICOM patient coordinates [mm]
    layers: List[Layer]
    unit: str = "MU"                   # "MU", "NP" (DICOM particles), "particles" (matRad) or "relative"
    snout_position: Optional[float] = None
    range_shifter: Optional[RangeShifter] = None
    radiation: str = "PROTON"

    @property
    def n_spots(self): return sum(len(l.w) for l in self.layers)

    @property
    def total_weight(self): return sum(sum(l.w) for l in self.layers)


@dataclass
class Plan:
    label: str
    patient_position: str              # HFS, HFP, FFS, FFP, ...
    fields: List[Field] = field(default_factory=list)
    frame_of_reference: str = ""
    fractions: int = 1


def _f(v, default=None):
    try: return float(v)
    except (TypeError, ValueError): return default


def _floats(ds, name):
    """Values of a DICOM attribute as floats, or None when it is absent or empty (type 2 attributes may be empty)."""
    v = [_f(x) for x in _as_list(getattr(ds, name, None))]
    return v if v and None not in v else None


# PrimaryDosimeterUnit of DICOM: MU or NP (number of particles). The writer converts MU (and "relative", weights
# without a BeamMeterset) to particles with the beam model; every other unit is taken as particles.
UNITS = ("MU", "NP")


def _fractions(ds, beam_numbers):
    """Fractions of the plan: the fraction groups that reference the treatment beams must agree.

    A plan may carry a second group for setup beams without NumberOfFractionsPlanned; that group says nothing.
    """
    seen = {}
    for fg in getattr(ds, "FractionGroupSequence", []):
        refs = {int(rb.ReferencedBeamNumber) for rb in getattr(fg, "ReferencedBeamSequence", [])}
        n = _f(getattr(fg, "NumberOfFractionsPlanned", None))
        if refs & beam_numbers and n:
            seen[int(n)] = seen.get(int(n), set()) | (refs & beam_numbers)
    if len(seen) > 1:
        raise ValueError("treatment beams sit in fraction groups with different fraction counts ("
                         + ", ".join(f"{n} fractions: beams {sorted(b)}" for n, b in sorted(seen.items()))
                         + "): recalculate each group on its own")
    return next(iter(seen), 1)


def read_rtplan(path, include_setup=False):
    """Reads a DICOM RT Ion Plan. Spot weights become meterset units: BeamMeterset * weight /
    FinalCumulativeMetersetWeight, in the beam's PrimaryDosimeterUnit (MU or NP).

    Setup beams (TreatmentDeliveryType SETUP) are skipped unless include_setup. Layers are built from every control point
    with nonzero spot weights; the closing control point of a layer carries zero weights in the usual export and adds
    nothing. Empty type 2 attributes (SnoutPosition, IsocenterPosition, ScanningSpotSize, range shifter settings) count
    as missing. A range shifter without IsocenterToRangeShifterDistance gets iso_distance None: the caller decides.
    """
    import pydicom
    ds = pydicom.dcmread(path)
    if ds.SOPClassUID != "1.2.840.10008.5.1.4.1.1.481.8":
        raise ValueError(f"not an RT Ion Plan: SOP class {ds.SOPClassUID}")
    meterset = {}
    for fg in getattr(ds, "FractionGroupSequence", []):
        for rb in getattr(fg, "ReferencedBeamSequence", []):
            m = _f(getattr(rb, "BeamMeterset", None))
            if m is not None: meterset[int(rb.ReferencedBeamNumber)] = m
    pos = ""
    for ps in getattr(ds, "PatientSetupSequence", []):
        pos = str(getattr(ps, "PatientPosition", "") or pos)
    plan = Plan(label=str(getattr(ds, "RTPlanLabel", "") or "plan"), patient_position=pos or "HFS",
                frame_of_reference=str(getattr(ds, "FrameOfReferenceUID", "")))
    for b in ds.IonBeamSequence:
        if str(getattr(b, "TreatmentDeliveryType", "TREATMENT")).upper() != "TREATMENT" and not include_setup:
            continue
        if str(getattr(b, "ScanMode", "MODULATED")).upper() not in ("MODULATED", "MODULATED_SPEC"):
            raise ValueError(f"beam {b.BeamName}: scan mode {b.ScanMode} is not pencil beam scanning")
        cps = b.IonControlPointSequence
        cp0 = cps[0]
        rs = None
        if int(getattr(b, "NumberOfRangeShifters", 0) or 0) > 0:
            rsd = b.RangeShifterSequence[0]
            rs = RangeShifter(str(rsd.RangeShifterID), iso_distance=None)
            for s in getattr(cp0, "RangeShifterSettingsSequence", []):
                if str(getattr(s, "RangeShifterSetting", "IN")).upper() == "OUT": rs = None; break
                rs.wet = _f(getattr(s, "RangeShifterWaterEquivalentThickness", None), 0.0)
                rs.iso_distance = _f(getattr(s, "IsocenterToRangeShifterDistance", None))
        final = _f(getattr(b, "FinalCumulativeMetersetWeight", None)) or sum(
            sum(float(w) for w in _as_list(getattr(cp, "ScanSpotMetersetWeights", []))) for cp in cps)
        mu = meterset.get(int(b.BeamNumber))
        scale = (mu / final) if (mu and final) else 1.0
        unit = str(getattr(b, "PrimaryDosimeterUnit", "") or "MU").upper()
        if unit not in UNITS: raise ValueError(f"beam {b.BeamName}: PrimaryDosimeterUnit {unit} not supported")
        layers, energy, gantry, couch, iso = [], None, None, None, None
        for cp in cps:
            energy = _f(getattr(cp, "NominalBeamEnergy", None), energy)
            gantry = _f(getattr(cp, "GantryAngle", None), gantry)
            couch = _f(getattr(cp, "PatientSupportAngle", None), couch)
            iso = tuple(_floats(cp, "IsocenterPosition") or iso or ()) or None
            w = [float(v) for v in _as_list(getattr(cp, "ScanSpotMetersetWeights", []))]
            if not any(w): continue
            m = [float(v) for v in _as_list(cp.ScanSpotPositionMap)]
            keep = [i for i, wi in enumerate(w) if wi > 0]
            size = _floats(cp, "ScanningSpotSize")
            layers.append(Layer(energy, [m[2 * i] for i in keep], [m[2 * i + 1] for i in keep], [w[i] * scale for i in keep],
                                str(getattr(cp, "ScanSpotTuneID", "")), tuple(size) if size and len(size) == 2 else None))
        if iso is None or len(iso) != 3: raise ValueError(f"beam {b.BeamName}: no isocenter")
        snout = _f(getattr(cp0, "SnoutPosition", None))
        plan.fields.append(Field(str(getattr(b, "BeamName", b.BeamNumber)), int(b.BeamNumber), gantry or 0.0, couch or 0.0,
                                 iso, layers, unit if mu else "relative", snout, rs, str(getattr(b, "RadiationType", "PROTON"))))
    if not plan.fields: raise ValueError("no treatment beams in the plan")
    plan.fractions = _fractions(ds, {f.number for f in plan.fields})
    return plan


def safe_label(ds_or_path):
    """The plan label for reports, or "Plan" when it may carry the patient's name (any part of PatientName in it)."""
    import pydicom
    ds = ds_or_path if not isinstance(ds_or_path, str) else pydicom.dcmread(ds_or_path, stop_before_pixels=True)
    label = str(getattr(ds, "RTPlanLabel", "") or "").strip()
    parts = [p for p in re.split(r"[\\^\s,]+", str(getattr(ds, "PatientName", "") or "").lower()) if len(p) > 1]
    if not label or "^" in label or any(p in label.lower() for p in parts): return "Plan"
    return label


def _as_list(v):
    if v is None: return []
    try: return list(v)
    except TypeError: return [v]


def write_rtplan(plan, path, mu_per_field=None, study_uid=None, frame_uid=None, setup_beams=0):
    """Writes a minimal but valid RT Ion Plan for tests and public phantoms. Weights are meterset units of the field
    (PrimaryDosimeterUnit NP for a field with unit "NP" or "particles", else MU); a layer's spot_size becomes
    ScanningSpotSize.

    Each layer becomes two control points (spots, then zero weights), as common TPS exports do. setup_beams adds that
    many setup beams (meterset 0) in a second fraction group without NumberOfFractionsPlanned, as a TPS export can.
    """
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.8"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTPLAN"
    ds.PatientName = "Synthetic^Phantom"
    ds.PatientID = "SYNTH"
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = frame_uid or plan.frame_of_reference or generate_uid()
    ds.RTPlanLabel = plan.label[:16]
    ps = Dataset(); ps.PatientPosition = plan.patient_position; ps.PatientSetupNumber = 1
    ds.PatientSetupSequence = Sequence([ps])
    fg = Dataset(); fg.FractionGroupNumber = 1; fg.NumberOfFractionsPlanned = plan.fractions
    fg.NumberOfBeams = len(plan.fields); refs = []
    beams = []
    for f in plan.fields:
        total = f.total_weight
        rb = Dataset(); rb.ReferencedBeamNumber = f.number
        rb.BeamMeterset = (mu_per_field or {}).get(f.number, total); refs.append(rb)
        b = Dataset()
        b.BeamNumber = f.number; b.BeamName = f.name[:16]; b.BeamType = "STATIC"; b.RadiationType = f.radiation
        b.TreatmentDeliveryType = "TREATMENT"; b.ScanMode = "MODULATED"
        b.PrimaryDosimeterUnit = "NP" if f.unit in ("NP", "particles") else "MU"
        b.VirtualSourceAxisDistances = [7549.0, 8219.0]
        b.FinalCumulativeMetersetWeight = total
        b.NumberOfRangeShifters = 1 if f.range_shifter else 0
        if f.range_shifter:
            r = Dataset(); r.RangeShifterNumber = 1; r.RangeShifterID = f.range_shifter.rs_id; r.RangeShifterType = "BINARY"
            b.RangeShifterSequence = Sequence([r])
        cps, cum, idx = [], 0.0, 0
        for li, l in enumerate(f.layers):
            for closing in (False, True):
                cp = Dataset(); cp.ControlPointIndex = idx; idx += 1
                cp.NominalBeamEnergy = l.energy
                cp.ScanSpotTuneID = l.tune_id or "1.0"
                cp.NumberOfScanSpotPositions = len(l.w)
                cp.ScanSpotPositionMap = [v for xy in zip(l.x, l.y) for v in xy]
                cp.ScanSpotMetersetWeights = [0.0] * len(l.w) if closing else list(l.w)
                cp.CumulativeMetersetWeight = cum
                if l.spot_size: cp.ScanningSpotSize = [float(v) for v in l.spot_size]
                if not closing: cum += sum(l.w)
                if idx == 1:
                    cp.GantryAngle = f.gantry; cp.GantryRotationDirection = "NONE"
                    cp.PatientSupportAngle = f.couch; cp.PatientSupportRotationDirection = "NONE"
                    cp.IsocenterPosition = list(f.iso)
                    if f.snout_position is not None: cp.SnoutPosition = f.snout_position
                    if f.range_shifter:
                        s = Dataset(); s.RangeShifterSetting = "IN"; s.ReferencedRangeShifterNumber = 1
                        s.IsocenterToRangeShifterDistance = f.range_shifter.iso_distance
                        s.RangeShifterWaterEquivalentThickness = f.range_shifter.wet
                        cp.RangeShifterSettingsSequence = Sequence([s])
                cps.append(cp)
        b.NumberOfControlPoints = len(cps)
        b.IonControlPointSequence = Sequence(cps)
        beams.append(b)
    fg.ReferencedBeamSequence = Sequence(refs)
    groups = [fg]
    if setup_beams:
        sg = Dataset(); sg.FractionGroupNumber = 2; sg.NumberOfBeams = setup_beams; srefs = []
        first = max(f.number for f in plan.fields) + 1
        for n in range(first, first + setup_beams):
            rb = Dataset(); rb.ReferencedBeamNumber = n; rb.BeamMeterset = 0.0; srefs.append(rb)
            b = Dataset(); b.BeamNumber = n; b.BeamName = f"SETUP{n}"; b.BeamType = "STATIC"; b.RadiationType = "PROTON"
            b.TreatmentDeliveryType = "SETUP"; b.ScanMode = "NONE"; b.PrimaryDosimeterUnit = "MU"
            b.NumberOfRangeShifters = 0; b.NumberOfControlPoints = 1
            cp = Dataset(); cp.ControlPointIndex = 0; cp.GantryAngle = 0.0; cp.PatientSupportAngle = 0.0
            cp.IsocenterPosition = list(plan.fields[0].iso)
            b.IonControlPointSequence = Sequence([cp])
            beams.append(b)
        sg.ReferencedBeamSequence = Sequence(srefs)
        groups.append(sg)
    ds.FractionGroupSequence = Sequence(groups)
    ds.IonBeamSequence = Sequence(beams)
    from plan_mc import dicom_compat
    dicom_compat.save(ds, path)
    return path
