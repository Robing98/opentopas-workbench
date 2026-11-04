"""Plan model for the spot-by-spot TOPAS recalculation: fields, energy layers, spots.

Readers: DICOM RT Ion Plan (pydicom) and a synthetic RT Ion Plan writer for tests on public data.
Spot positions are in the IEC gantry system at the isocenter plane, in mm (DICOM ScanSpotPositionMap).
Spot weights stay in the unit of the plan (MU for DICOM, particles for matRad); the beam model converts MU to particles.
"""
from dataclasses import dataclass, field
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
    iso_distance: float = 0.0          # isocenter to range shifter distance [mm] (downstream face, DICOM definition)


@dataclass
class Field:
    name: str
    number: int
    gantry: float                      # IEC gantry angle [deg]
    couch: float                       # IEC patient support angle [deg]
    iso: tuple                         # isocenter in DICOM patient coordinates [mm]
    layers: List[Layer]
    unit: str = "MU"                   # "MU" or "particles"
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


def read_rtplan(path, include_setup=False):
    """Reads a DICOM RT Ion Plan. Spot weights become MU: BeamMeterset * weight / FinalCumulativeMetersetWeight.

    Setup beams (TreatmentDeliveryType SETUP) are skipped unless include_setup. Layers are built from every control point
    with nonzero spot weights; the closing control point of a layer carries zero weights in the usual export and adds
    nothing.
    """
    import pydicom
    ds = pydicom.dcmread(path)
    if ds.SOPClassUID != "1.2.840.10008.5.1.4.1.1.481.8":
        raise ValueError(f"not an RT Ion Plan: SOP class {ds.SOPClassUID}")
    meterset = {}
    fractions = 1
    for fg in getattr(ds, "FractionGroupSequence", []):
        fractions = int(getattr(fg, "NumberOfFractionsPlanned", 1) or 1)
        for rb in getattr(fg, "ReferencedBeamSequence", []):
            meterset[int(rb.ReferencedBeamNumber)] = _f(getattr(rb, "BeamMeterset", None))
    pos = ""
    for ps in getattr(ds, "PatientSetupSequence", []):
        pos = str(getattr(ps, "PatientPosition", "") or pos)
    plan = Plan(label=str(getattr(ds, "RTPlanLabel", "plan")), patient_position=pos or "HFS",
                frame_of_reference=str(getattr(ds, "FrameOfReferenceUID", "")), fractions=fractions)
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
            rs = RangeShifter(str(rsd.RangeShifterID))
            for s in getattr(cp0, "RangeShifterSettingsSequence", []):
                if str(getattr(s, "RangeShifterSetting", "IN")).upper() == "OUT": rs = None; break
                rs.wet = _f(getattr(s, "RangeShifterWaterEquivalentThickness", None), 0.0)
                rs.iso_distance = _f(getattr(s, "IsocenterToRangeShifterDistance", None), 0.0)
        final = _f(getattr(b, "FinalCumulativeMetersetWeight", None)) or sum(
            sum(float(w) for w in _as_list(getattr(cp, "ScanSpotMetersetWeights", []))) for cp in cps)
        mu = meterset.get(int(b.BeamNumber))
        scale = (mu / final) if (mu and final) else 1.0
        layers, energy, gantry, couch, iso = [], None, None, None, None
        for cp in cps:
            energy = _f(getattr(cp, "NominalBeamEnergy", None), energy)
            gantry = _f(getattr(cp, "GantryAngle", None), gantry)
            couch = _f(getattr(cp, "PatientSupportAngle", None), couch)
            if hasattr(cp, "IsocenterPosition"): iso = tuple(float(v) for v in cp.IsocenterPosition)
            w = [float(v) for v in _as_list(getattr(cp, "ScanSpotMetersetWeights", []))]
            if not any(w): continue
            m = [float(v) for v in _as_list(cp.ScanSpotPositionMap)]
            keep = [i for i, wi in enumerate(w) if wi > 0]
            size = tuple(float(v) for v in cp.ScanningSpotSize) if hasattr(cp, "ScanningSpotSize") else None
            layers.append(Layer(energy, [m[2 * i] for i in keep], [m[2 * i + 1] for i in keep], [w[i] * scale for i in keep],
                                str(getattr(cp, "ScanSpotTuneID", "")), size))
        if iso is None: raise ValueError(f"beam {b.BeamName}: no isocenter")
        snout = None
        if hasattr(cp0, "SnoutPosition"): snout = float(cp0.SnoutPosition)
        plan.fields.append(Field(str(getattr(b, "BeamName", b.BeamNumber)), int(b.BeamNumber), gantry or 0.0, couch or 0.0,
                                 iso, layers, "MU" if mu else "relative", snout, rs, str(getattr(b, "RadiationType", "PROTON"))))
    if not plan.fields: raise ValueError("no treatment beams in the plan")
    return plan


def _as_list(v):
    if v is None: return []
    try: return list(v)
    except TypeError: return [v]


def write_rtplan(plan, path, mu_per_field=None, study_uid=None, frame_uid=None):
    """Writes a minimal but valid RT Ion Plan for tests and public phantoms. Weights of the plan are taken as MU.

    Each layer becomes two control points (spots, then zero weights), as common TPS exports do.
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
        b.TreatmentDeliveryType = "TREATMENT"; b.ScanMode = "MODULATED"; b.PrimaryDosimeterUnit = "MU"
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
    ds.FractionGroupSequence = Sequence([fg])
    ds.IonBeamSequence = Sequence(beams)
    ds.save_as(path, enforce_file_format=True)
    return path
