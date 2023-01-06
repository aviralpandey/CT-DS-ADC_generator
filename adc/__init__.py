"""
ADC Package
"""

from pathlib import Path

# Set some common file paths, relative to the root of the repository
_repo = Path(__file__).parent.parent
data_dir = _repo / "data"
scratch = _repo / "scratch"
