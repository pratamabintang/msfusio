# Modified by Mzero #20240123
# simplified version of mamba 
# kHasZ = False # delete implementation for z
# kIsVariableC = True; kIsVariableB = True # delete implementation for B, C not variable
# kIsComplex = False # delete implementation for complex_t

import torch

import os
import sys

_SELECTIVE_SCAN_CORE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "VMamba",
        "kernels",
        "selective_scan",
    )
)

if _SELECTIVE_SCAN_CORE_DIR not in sys.path:
    sys.path.insert(0, _SELECTIVE_SCAN_CORE_DIR)

if os.name == "nt":
    _CONDA_BIN = os.path.join(
        os.environ.get("CONDA_PREFIX", ""),
        "bin",
    )

    _TORCH_LIB = os.path.join(
        os.environ.get("CONDA_PREFIX", ""),
        "Lib",
        "site-packages",
        "torch",
        "lib",
    )

    if os.path.isdir(_CONDA_BIN):
        os.add_dll_directory(_CONDA_BIN)

    if os.path.isdir(_TORCH_LIB):
        os.add_dll_directory(_TORCH_LIB)

from .selective_scan_interface import SelectiveScanFn, selective_scan_fn, selective_scan_ref
