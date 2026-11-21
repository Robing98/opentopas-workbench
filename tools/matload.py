"""Loads a MATLAB struct variable (e.g. a matRad machine) from a .mat file, v5/v7 via scipy or v7.3 (HDF5) via h5py.

Returns objects with attribute access like scipy's mat_struct; struct arrays become lists. The reader lives in
opentopas_workbench/machines.py, which the Machines view of the workbench uses too; this module keeps the tools' import.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opentopas_workbench.machines import load_struct  # noqa: E402,F401
