"""Fetch the demo dataset used by the other example scripts.

Run this once before the other demos to download the Nelson (unit_397) OG1
NetCDF file into examples/data/OG1. The file is hosted on the BODC BIO-Carbon
deployment catalogue; see https://noc.ac.uk/projects/bio-carbon for context.
"""

import os
from pathlib import Path

import requests

DATA_URL = (
    "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001/"
    "Nelson_20240528/Nelson_646_R.nc"
)

# Work from the repo root so the relative paths below resolve the same way no
# matter where the script was started from. Safe to re-run.
_config = "examples/configs/example_config_nelson.yaml"
if not Path(_config).exists() and Path("../..", _config).exists():
    os.chdir("../..")

input_dir = Path("examples/data/OG1")
input_file = input_dir / "Nelson_646_R.nc"

input_dir.mkdir(parents=True, exist_ok=True)

if input_file.exists():
    print(f"Example file already present at {input_file.resolve()}")
else:
    print("Downloading example file...")
    response = requests.get(DATA_URL)
    if response.status_code == 200:
        with open(input_file, "wb") as f:
            f.write(response.content)
        print(f"Example file downloaded and written to {input_dir.resolve()}")
    else:
        print(f"File download failed (HTTP {response.status_code})")
