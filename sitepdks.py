"""
# Site PDKs

Using the conda-installed SkyWater 130nm PDK package 
"""

import os
from pathlib import Path

CONDA_PREFIX = os.environ.get("CONDA_PREFIX", None)
if CONDA_PREFIX is None or not Path(CONDA_PREFIX).exists():
    raise RuntimeError("Cannot located CONDA_PATH, cannot find PDK contents")

model_lib = Path(CONDA_PREFIX) / "share/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice"

if not model_lib.exists():
    raise RuntimeError(f"Sky130 model files at {model_lib} not found")


# Import the sky130 Hdl21 PDK package, and set up its installation
import sky130

sky130.install = sky130.Install(model_lib=model_lib)
