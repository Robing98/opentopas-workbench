"""Ventricles from a SynthSeg label map and the 4 mm margin around them, inside the brain only."""
import os
import shlex
import shutil
import subprocess

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from . import geometry as geo

# FreeSurfer labels that SynthSeg writes: lateral and inferior lateral ventricles, 3rd and 4th ventricle
VENTRICLES = (4, 5, 43, 44, 14, 15)
CSF = 24                   # extracerebral CSF: not brain
CROP_MM = 192              # SynthSeg works on a 1 mm grid, so this is its crop in voxels
THREADS = 2
MARGIN_MM = 4.0            # Bahn et al. 2020: lesion risk about threefold within 4 mm of the ventricles


def run_synthseg(t1_path, out_path, command=None):
    """Segment a T1 with SynthSeg. `command` (or $SYNTHSEG) is the call before the arguments, e.g. `mri_synthseg`
    or `python /opt/SynthSeg/scripts/commands/SynthSeg_predict.py`; both take --i, --o and --robust."""
    cmd = command or os.environ.get("SYNTHSEG") or ("mri_synthseg" if shutil.which("mri_synthseg") else None)
    if not cmd: raise SystemExit("SynthSeg not found: set SYNTHSEG to its command or pass --synthseg")
    # The robust model on a whole field of view needs about 11 GB and was killed in a 15 GB WSL; a centred
    # 192 mm cube needs 9.5 GB and gave the same ventricles on RIRE patient 001 (Dice 0.991 against 160 mm).
    args = shlex.split(cmd) + ["--i", t1_path, "--o", out_path, "--robust", "--cpu", "--crop", str(CROP_MM),
                               "--threads", str(THREADS)]
    subprocess.run(args, check=True)
    return out_path


def margin(labels_img, mm=MARGIN_MM):
    """(ventricles, margin) as uint8 images on the label grid. The margin is brain within `mm` of a ventricle,
    measured in mm in the label map's own voxel spacing, outside the ventricles."""
    lab = sitk.GetArrayFromImage(labels_img)
    vent = np.isin(lab, VENTRICLES)
    brain = (lab > 0) & (lab != CSF) & ~vent
    if not vent.any(): raise SystemExit("no ventricle labels in the segmentation")
    sx, sy, sz = labels_img.GetSpacing()
    dist = ndi.distance_transform_edt(~vent, sampling=(sz, sy, sx))
    rim = brain & (dist <= mm)
    return geo.like(vent.astype(np.uint8), labels_img), geo.like(rim.astype(np.uint8), labels_img)
