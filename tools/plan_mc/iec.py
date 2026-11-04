"""IEC 61217 geometry in numpy, independent of TOPAS: where a spot ray runs in DICOM patient coordinates.

Fixed system f: Xf to the right of the gantry seen from the foot of the couch, Yf towards the gantry, Zf up.
Gantry angle g turns the source from +Zf towards +Xf; couch angle c turns the patient support counterclockwise seen
from above. For HFS, DICOM x (patient left) is Xf, DICOM z (head) is Yf, DICOM y (posterior) is -Zf.
Used to check the TOPAS placement (checks/alignment.py) and to project doses along beams.
"""
import numpy as np

# DICOM patient axes -> patient support system (columns: images of DICOM x, y, z), per patient position
POSITION = {
    "HFS": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),
    "FFS": np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float),
    "HFP": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float),
    "FFP": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
}


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def gantry_axes(gantry_deg):
    """Unit vectors Xg, Yg, Zg of the gantry system in the fixed system (Zg points to the source)."""
    r = ry(np.radians(gantry_deg))
    return r[:, 0], r[:, 1], r[:, 2]


def fixed_to_dicom(p_f, iso, couch_deg, position="HFS"):
    """Point(s) in the fixed system (origin at the isocenter) to DICOM patient coordinates."""
    m = rz(np.radians(couch_deg)) @ POSITION[position.upper()]     # DICOM offset -> fixed
    return np.asarray(iso, float) + np.linalg.solve(m, np.asarray(p_f, float).T).T


def spot_ray(x_iso, y_iso, gantry, couch, iso, vsad=(7549.0, 8219.0), position="HFS"):
    """(point at the isocenter plane, unit direction) of a spot ray in DICOM coordinates.

    The ray comes from the virtual source of each axis; for the direction the two angles are combined as
    (tan ax, tan ay, -1) in the gantry system, which is what the TOPAS writer aims at.
    """
    xg, yg, zg = gantry_axes(gantry)
    p_f = x_iso * xg + y_iso * yg
    d_f = (x_iso / vsad[0]) * xg + (y_iso / vsad[1]) * yg - zg
    p = fixed_to_dicom(p_f, iso, couch, position)
    d = fixed_to_dicom(d_f, iso, couch, position) - np.asarray(iso, float)
    return p, d / np.linalg.norm(d)
