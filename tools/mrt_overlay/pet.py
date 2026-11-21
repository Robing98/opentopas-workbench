"""PET layer for the MRI viewer: a PET DICOM series registered onto a 3D T1 of the viewer grid, as a ratio volume.

Display only, no reading of the images. The PET header is not trusted for position: older scanners write
ImagePositionPatient from 0,0,0 with a unit orientation, so the volume is read as LPS with the header's orientation
and an origin at 0 and placed by registration alone. Without Bq/ml, decay correction, weight and dose there is no
SUV; the layer shows the ratio to the median of a reference region instead.

The registration is rigid (Euler3D, Mattes MI) with the T1 fixed, so its transform maps viewer grid points into the
PET and sitk.Resample(pet, grid, transform) puts the PET onto the grid. A rigid transform cannot mirror, so the same
search with the PET flipped left to right is the side check: the unflipped PET must fit clearly better. The same
holds for the slice order: the slices are sorted by the z of ImagePositionPatient, and the search with the slice
order reversed is the head to foot check.

Progress goes to stdout as "## STEP name" and "## START i/n" lines, which the workbench tab reads.
"""
import datetime
import glob
import json
import os

import numpy as np
import SimpleITK as sitk

from . import geometry as geo, layer as ly

BINS = 40
START_DEG = (-12.0, 0.0, 12.0)         # starts: rotations about x and z, each of these, from the centres of mass
LEVELS = ((4, 4.0), (2, 2.0), (1, 1.0))   # shrink factor, smoothing sigma in mm
MASK_DILATE_MM = 5.0                   # the metric looks at the brain and a little around it
MIRROR_MARGIN = 0.01                   # unflipped MI must be at least 1 % better than flipped, else the run stops
                                       # (phantom with a one-sided cavity: 3.7 % in either direction)
RATIO_MAX = 4.0                        # viewer byte = ratio x 255 / 4: steps of 0.016 up to 4
STRIATUM = (11, 12, 50, 51)            # SynthSeg: caudate and putamen, left and right
HINWEIS = ("Nur Anzeige, keine Befundung. Die Werte sind ein Verhältnis zur Referenzregion. "
           "Deutung bei Nuklearmedizin und Neuroradiologie.")


# ---------- reading ----------
def _num(ds, name, default=None):
    v = ds.get(name, None)
    try: return float(v) if v not in (None, "") else default
    except (TypeError, ValueError): return default


def read_pet(folder):
    """(image, header facts) of the largest PT series in `folder` (searched recursively).

    Slices are sorted along the slice normal by ImagePositionPatient, by InstanceNumber when all positions are the
    same. Each slice keeps its own RescaleSlope and RescaleIntercept."""
    import pydicom
    series = {}
    for p in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True)):
        if not os.path.isfile(p): continue
        try: ds = pydicom.dcmread(p, stop_before_pixels=True)
        except Exception: continue
        if str(ds.get("Modality", "")).upper() != "PT" or "Rows" not in ds: continue
        series.setdefault(str(ds.get("SeriesInstanceUID", "")), []).append((p, ds))
    if not series: raise SystemExit(f"no PET (Modality PT) slices in {folder}")
    files = max(series.values(), key=len)
    first = files[0][1]
    iop = np.asarray(first.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    pos = np.array([np.dot(np.asarray(ds.get("ImagePositionPatient", [0, 0, 0]), float), normal) for _, ds in files])
    if np.ptp(pos) > 1e-6:
        order, how = np.argsort(pos, kind="stable"), "ImagePositionPatient"
    else:
        order, how = np.argsort([int(ds.get("InstanceNumber") or 0) for _, ds in files], kind="stable"), "InstanceNumber"
    files = [files[i] for i in order]
    gaps = np.diff(np.sort(pos))
    if np.ptp(pos) > 1e-6:
        if np.ptp(gaps) > 0.01 * max(float(np.median(gaps)), 1e-6): raise SystemExit("PET: uneven slice spacing")
        dz = float(np.median(gaps))
    else:
        dz = _num(first, "SpacingBetweenSlices") or _num(first, "SliceThickness") or 1.0
    vol = []
    for p, _ in files:
        ds = pydicom.dcmread(p)
        vol.append(ds.pixel_array.astype(np.float64) * _num(ds, "RescaleSlope", 1.0) + _num(ds, "RescaleIntercept", 0.0))
    arr = np.stack(vol).astype(np.float32)
    ps = first.PixelSpacing
    img = geo.from_array(arr, (0.0, 0.0, 0.0), (float(ps[1]), float(ps[0]), dz), np.stack([row, col, normal], 1).ravel())
    rp = first.get("RadiopharmaceuticalInformationSequence", None)
    tracer = str(rp[0].get("Radiopharmaceutical", "") or "").strip() if rp else ""
    day = next((str(first.get(k, "")) for k in ("AcquisitionDate", "SeriesDate", "StudyDate") if first.get(k, "")), "")
    facts = {"day": f"{day[:4]}-{day[4:6]}-{day[6:8]}" if len(day) == 8 else "",
             "tracer": tracer or "unbekannt", "units": str(first.get("Units", "") or "").upper(),
             "decay_correction": str(first.get("DecayCorrection", "") or "").upper(),
             "slices": len(files), "sorted_by": how,
             "spacing_mm": [round(float(ps[1]), 3), round(float(ps[0]), 3), round(dz, 3)]}
    return img, facts


# ---------- registration ----------
def _metric_mask(mask_img):
    r = [int(np.ceil(MASK_DILATE_MM / s)) for s in mask_img.GetSpacing()]
    return sitk.BinaryDilate(sitk.Cast(mask_img, sitk.sitkUInt8), r)


def _full_mi(fixed, moving, mask, t):
    e = sitk.ImageRegistrationMethod()
    e.SetMetricAsMattesMutualInformation(BINS)
    e.SetMetricFixedMask(mask)
    e.SetInterpolator(sitk.sitkLinear)
    e.SetInitialTransform(t)
    return float(e.MetricEvaluate(fixed, moving))


def _search(fixed, moving, mask, start, seed, levels, iterations):
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(BINS)
    r.SetMetricFixedMask(mask)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.15, seed)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=iterations,
                                               relaxationFactor=0.6, gradientMagnitudeTolerance=1e-8)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([s for s, _ in levels])
    r.SetSmoothingSigmasPerLevel([g for _, g in levels])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(start, inPlace=False)
    t = r.Execute(fixed, moving)
    if isinstance(t, sitk.CompositeTransform): t = t.GetNthTransform(t.GetNumberOfTransforms() - 1)
    return sitk.Euler3DTransform(t)


def _progress(line):
    print(line, flush=True)


def rigid(fixed, moving, mask, levels=LEVELS, iterations=200, start_deg=START_DEG, count=None):
    """Best of the starts by fully sampled MI. Returns (transform grid -> PET, MI, [MI per start]).
    `count` is [done, total] over all searches of a run, for the "## START" progress lines."""
    f, m = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
    mm = _metric_mask(mask)
    init = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(
        f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS))
    best, values = None, []
    for i, (rx, rz) in enumerate((a, b) for a in start_deg for b in start_deg):
        s = sitk.Euler3DTransform(init)
        s.SetRotation(np.radians(rx), 0.0, np.radians(rz))
        try: t = _search(f, m, mm, s, i + 1, levels, iterations)
        except RuntimeError:
            if count is not None:
                count[0] += 1; _progress(f"## START {count[0]}/{count[1]}")
            continue
        v = _full_mi(f, m, mm, t)
        values.append(round(v, 5))
        if count is not None:
            count[0] += 1; _progress(f"## START {count[0]}/{count[1]}")
        if best is None or v < best[1]: best = (t, v)
    if best is None: raise SystemExit("PET registration failed from every start")
    return best[0], best[1], values


def mirror(img):
    """The image with its columns reversed on the same geometry. sitk.Flip would not do: it also flips the direction
    cosines, so the flipped image shows the same anatomy at the same physical place."""
    return geo.like(sitk.GetArrayFromImage(img)[:, :, ::-1], img)


def upside_down(img):
    """The image with its slice order reversed on the same geometry: head and foot swapped, as a wrong slice sort
    would give. Like a mirror, no rigid transform can undo it."""
    return geo.like(sitk.GetArrayFromImage(img)[::-1], img)


def _gain(v, other):
    return (other - v) / abs(v) if v else 0.0


def side_check(fixed, moving, mask, margin=MIRROR_MARGIN, **kw):
    """Register the PET as read, with its slice order reversed, and flipped left to right. Returns (transform,
    quality). Stops the run when the PET as read does not fit clearly better than either variant: then head and foot
    (the slice order) or left and right of the PET are not settled. Head to foot comes first, since a PET standing
    on its head fails the side check too and the message would name the wrong cause."""
    n = len(kw.get("start_deg", START_DEG)) ** 2
    count = [0, 3 * n]
    _progress("## STEP register")
    t, v, starts = rigid(fixed, moving, mask, count=count, **kw)
    _progress("## STEP headfoot")
    _, vz, _ = rigid(fixed, upside_down(moving), mask, count=count, **kw)
    gz = _gain(v, vz)
    if gz < margin:
        raise SystemExit(f"Kopf-Fuß-Prüfung nicht bestanden: das PET mit umgekehrter Schichtfolge passt fast so gut "
                         f"oder besser (MI {v:.4f} gegen kopfstehend {vz:.4f}, Abstand {gz:.1%}, nötig {margin:.0%}). "
                         "Die Schichtreihenfolge ist nicht sicher; der Lauf schreibt nichts.")
    _progress("## STEP side")
    _, vm, _ = rigid(fixed, mirror(moving), mask, count=count, **kw)
    gain = _gain(v, vm)
    q = {"metric": "Mattes MI, 40 Bins, voll abgetastet (niedriger ist besser)", "mi": round(v, 5),
         "mi_gespiegelt": round(vm, 5), "abstand_relativ": round(gain, 4), "mi_kopfstehend": round(vz, 5),
         "abstand_kopfstehend_relativ": round(gz, 4), "grenze_relativ": margin,
         "starts": len(starts), "mi_je_start": starts}
    if gain < margin:
        raise SystemExit(f"Seitenprüfung nicht bestanden: das gespiegelte PET passt fast so gut oder besser "
                         f"(MI {v:.4f} gegen gespiegelt {vm:.4f}, Abstand {gain:.1%}, nötig {margin:.0%}). "
                         "Links und rechts sind nicht sicher; der Lauf schreibt nichts.")
    return t, q


# ---------- reference region and ratio ----------
def reference_mask(kind, brain, grid, labels=None, side=None):
    """Mask of the reference region on the grid: 'brain' (median of the brain mask), 'striatum' (SynthSeg caudate
    and putamen), 'contralateral' (the hemisphere opposite to `side`, split at the brain's centre left to right)."""
    if kind == "brain": return brain, "Median im Hirn (Hirnmaske des Viewers)"
    if kind == "striatum":
        if labels is None: raise SystemExit("--ref striatum needs --labels (SynthSeg of the viewer's reference T1)")
        lab = ly.resample(sitk.ReadImage(labels), grid, sitk.Transform(), label=True)
        m = np.isin(lab, STRIATUM)
        if not m.any(): raise SystemExit("the labels hold no caudate or putamen")
        return m, "Median im Striatum (Nucleus caudatus und Putamen, SynthSeg)"
    if kind == "contralateral":
        if side not in ("links", "rechts"): raise SystemExit("--ref contralateral needs --side links or rechts")
        z, y, x = np.nonzero(brain)
        d = np.array(grid.GetDirection()).reshape(3, 3)
        lps_x = (np.stack([x, y, z], 1) * np.array(grid.GetSpacing())) @ d.T[:, 0] + grid.GetOrigin()[0]
        mid = float(np.median(lps_x))
        # LPS: x grows towards the patient's left; the reference is the side without the named finding
        keep = lps_x < mid if side == "links" else lps_x > mid
        m = np.zeros(brain.shape, bool)
        m[z[keep], y[keep], x[keep]] = True
        other = "rechte" if side == "links" else "linke"
        return m, f"Median in der {other}n Hirnhälfte (Gegenseite zu {side})"
    raise SystemExit(f"unknown reference region {kind!r}")


def ratio_volume(pet_on_grid, ref_mask, covered=None):
    # grid voxels outside the PET field of view resample to 0; counted, they would lower the median and raise every ratio
    if covered is not None:
        ref_mask = ref_mask & covered
        if not ref_mask.any(): raise SystemExit("the PET does not cover the reference region")
    ref = float(np.median(pet_on_grid[ref_mask]))
    if not ref > 0: raise SystemExit("the PET is zero in the reference region: registration or region is wrong")
    return pet_on_grid / ref, ref


def to_byte(ratio):
    return np.clip(np.rint(ratio * 255.0 / RATIO_MAX), 0, 255).astype(np.uint8)


# ---------- control sheet ----------
SHEET_PX = 300                          # height of one panel
SHEET_FROM = 1.0                        # ratio where the colour starts: below it the T1 shows alone
SHEET_FULL = 0.6                        # the colour reaches full strength this far above SHEET_FROM


def _lut():
    """Black body colours for ratio SHEET_FROM .. RATIO_MAX: red, yellow, white."""
    x = np.linspace(1 / 3, 1, 256)
    return (np.stack([np.clip(3 * x, 0, 1), np.clip(3 * x - 1, 0, 1), np.clip(3 * x - 2, 0, 1)], 1) * 255).astype(np.uint8)


def _panel(t1, ratio, spacing_hw, flip_rows):
    """One RGB panel: T1 grey, ratio in colour over it from SHEET_FROM up, rows flipped so that cranial is at the top."""
    from PIL import Image
    if flip_rows: t1, ratio = t1[::-1], ratio[::-1]
    lo, hi = np.percentile(t1[t1 > 0], (1, 99.5)) if (t1 > 0).any() else (0.0, 1.0)
    g = (np.clip((t1 - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(g[..., None], 3, -1).astype(np.float32)
    # the colour fades in above SHEET_FROM, so that the brain at the reference level keeps its T1 grey
    a = (np.clip((ratio - SHEET_FROM) / SHEET_FULL, 0, 1) * 0.7)[..., None]
    k = np.clip((ratio - SHEET_FROM) / (RATIO_MAX - SHEET_FROM) * 255, 0, 255).astype(int)
    rgb = (1 - a) * rgb + a * _lut()[k]
    im = Image.fromarray(rgb.astype(np.uint8))
    h, w = t1.shape
    scale = SHEET_PX / (h * spacing_hw[0])
    return im.resize((max(1, round(w * spacing_hw[1] * scale)), SHEET_PX), Image.BILINEAR)


def control_sheet(t1, ratio, brain, grid, path, title=""):
    """PNG with the PET on the T1 through the centre of the brain: axial, coronal, sagittal, the T1 alone above and
    with the PET below, R and L on the panels. It is for checking the fit and the side by eye, not for reading.

    Arrays are [z, y, x] on `grid`. They are turned to LPS first (the viewer's ref_t1 may be stored as RAS): then
    column 0 is the patient's right, row 0 anterior, z grows cranially, and the axial and coronal panels show the
    patient's right on the image's left (radiological view)."""
    from PIL import Image, ImageDraw
    lps = lambda a: sitk.DICOMOrient(geo.like(np.asarray(a, np.float32), grid), "LPS")
    t1_img = lps(t1)
    spacing = t1_img.GetSpacing()
    t1, ratio, brain = (sitk.GetArrayFromImage(i) for i in (t1_img, lps(ratio), lps(brain)))
    brain = brain > 0.5
    z, y, x = (int(round(c)) for c in np.mean(np.nonzero(brain), 1))
    sx, sy, sz = spacing
    cuts = [(t1[z], ratio[z], (sy, sx), False, ("R", "L")),           # axial: rows anterior..posterior
            (t1[:, y], ratio[:, y], (sz, sx), True, ("R", "L")),      # coronal
            (t1[:, :, x], ratio[:, :, x], (sz, sy), True, ("A", "P"))]  # sagittal
    zero = np.zeros_like
    tops = [_panel(a, zero(b), hw, f) for a, b, hw, f, _ in cuts]
    bots = [_panel(a, b, hw, f) for a, b, hw, f, _ in cuts]
    gap, head = 6, 24
    width = sum(p.width for p in tops) + gap * (len(tops) + 1)
    sheet = Image.new("RGB", (width, head + 2 * SHEET_PX + 3 * gap), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 6), title, fill=(230, 230, 230))
    left = gap
    for top, bot, (*_, marks) in zip(tops, bots, cuts):
        for row, p in ((0, top), (1, bot)):
            yy = head + gap + row * (SHEET_PX + gap)
            sheet.paste(p, (left, yy))
            d.text((left + 4, yy + SHEET_PX // 2), marks[0], fill=(255, 255, 0))
            d.text((left + p.width - 12, yy + SHEET_PX // 2), marks[1], fill=(255, 255, 0))
        left += top.width + gap
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sheet.save(path)
    return path


# ---------- the run ----------
def nearest_t1(log, cache, day, fixed_day=None):
    """The 3D T1 exam of the viewer cache nearest in time to `day` (or the one named), as (day, path)."""
    cands = sorted(k.split(" ", 1)[1] for k, e in log.items()
                   if k.startswith("t1 ") and isinstance(e, dict) and e.get("kind", "3D") == "3D"
                   and os.path.exists(os.path.join(cache, f"t1_{k.split(' ', 1)[1]}.npy")))
    if fixed_day:
        if fixed_day not in cands: raise SystemExit(f"no 3D T1 of {fixed_day} in the viewer cache")
        cands = [fixed_day]
    if not cands: raise SystemExit("no 3D T1 volume in the viewer cache")
    if not day: return cands[-1], os.path.join(cache, f"t1_{cands[-1]}.npy")
    t = datetime.date.fromisoformat(day)
    best = min(cands, key=lambda d: (abs((datetime.date.fromisoformat(d) - t).days), d))
    return best, os.path.join(cache, f"t1_{best}.npy")


def run(pet_dir, cache, viewer_data, out=None, day=None, ref="brain", labels=None, side=None, fixed_day=None,
        weight_kg=None, activity_mbq=None, margin=MIRROR_MARGIN, **kw):
    """Read, register with the side check, write pet_<day>.js and the entry of pet.json. Returns that entry."""
    _progress("## STEP read")
    pet, facts = read_pet(pet_dir)
    day = day or facts["day"]
    if weight_kg or activity_mbq:       # checked before the long registration
        if not (weight_kg and activity_mbq): raise SystemExit("SUV needs both --weight-kg and --activity-mbq")
        if facts["units"] != "BQML":
            raise SystemExit(f"SUV braucht Werte in Bq/ml; diese Serie hat Units {facts['units'] or 'leer'}.")
    if not day: raise SystemExit("the PET header has no date: give --day YYYY-MM-DD")
    log = json.load(open(os.path.join(cache, "log.json"), encoding="utf-8"))
    grid = sitk.ReadImage(os.path.join(cache, "ref_t1.nii.gz"))
    t1_day, t1_path = nearest_t1(log, cache, day, fixed_day)
    fixed = geo.like(np.load(t1_path).astype(np.float32), grid)
    brain = np.load(os.path.join(cache, "mask.npy")).astype(bool)
    t, q = side_check(fixed, pet, geo.like(brain.astype(np.uint8), grid), margin=margin, **kw)
    on_grid = sitk.GetArrayFromImage(sitk.Resample(pet, grid, t, sitk.sitkLinear, 0.0, sitk.sitkFloat32))
    ones = sitk.Image(pet.GetSize(), sitk.sitkUInt8) + 1
    ones.CopyInformation(pet)
    covered = sitk.GetArrayFromImage(sitk.Resample(ones, grid, t, sitk.sitkNearestNeighbor, 0)).astype(bool)
    _progress("## STEP write")
    region, region_name = reference_mask(ref, brain, grid, labels, side)
    ratio, ref_value = ratio_volume(on_grid, region, covered)
    # SUV = C [Bq/ml] x weight [g] / activity [Bq]; per unit of ratio, C = ratio x reference value
    suv = ref_value * weight_kg * 1000.0 / (activity_mbq * 1e6) if weight_kg else None
    entry = {"day": day, "tracer": facts["tracer"], "reference": region_name, "fixed_t1": t1_day,
             "registration": q, "units": facts["units"], "decay_correction": facts["decay_correction"],
             "sorted_by": facts["sorted_by"], "spacing_mm": facts["spacing_mm"], "suv_per_ratio": suv,
             "hinweis": "kein SUV" if suv is None else "SUV aus Gewicht und Aktivität von Hand, ohne Prüfung der Zerfallskorrektur"}
    if out:
        sheet = control_sheet(sitk.GetArrayFromImage(fixed), ratio, brain, grid,
                              os.path.join(out, f"pet_kontrolle_{day}.png"),
                              # the default font of PIL has no umlauts
                              f"PET {day} auf T1 {t1_day}: Farbe ab {SHEET_FROM:g} x Referenz. R = rechts")
    os.makedirs(viewer_data, exist_ok=True)
    open(os.path.join(viewer_data, f"pet_{day}.js"), "w").write(ly.montage_js(to_byte(ratio), "pet", day))
    meta_path = os.path.join(viewer_data, "pet.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    scans = {s["day"]: s for s in meta.get("scans", [])}
    scans[day] = entry
    meta = {"ratio_max": RATIO_MAX, "per_byte": RATIO_MAX / 255.0, "hinweis": HINWEIS,
            "scans": [scans[d] for d in sorted(scans)]}
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if out:
        os.makedirs(out, exist_ok=True)
        sitk.WriteImage(geo.like(ratio.astype(np.float32), grid), os.path.join(out, f"pet_ratio_{day}.nii.gz"))
        sitk.WriteTransform(t, os.path.join(out, f"grid_to_pet_{day}.tfm"))
        entry["sheet"] = sheet
        json.dump(entry, open(os.path.join(out, f"pet_{day}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return entry
