# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of demo OG1 datasets shared by ``examples/python/get_demo_file.py``
and the dashboard's own download-on-demand fallback (``dashboard/app.py``).

Each entry maps a short name to (download URL, output filename, TIME window to
keep or ``None`` for the whole file). Files are hosted on the BODC deployment
catalogue; see https://noc.ac.uk/projects/bio-carbon for context. Every key
``k`` here corresponds to a shipped ``dashboard/configs/demo_{k}.yaml``.
"""

DEMOS = {
    # --- Bio-Carbon ---
    "nelson": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001/"
        "Nelson_20240528/Nelson_646_R.nc",
        "Nelson_646_R.nc",
        None,
    ),
    "churchill": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery/"
        "Churchill_20240528/Churchill_647.nc",
        "Churchill_647.nc",
        ("2024-08-01", "2024-09-01"),
    ),
    "alr": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery/"
        "ALR_4_20240609/ALR_4_649.nc",
        "ALR_4_649.nc",
        None,
    ),
    "doombar": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery/"
        "Doombar_20240528/Doombar_648.nc",
        "Doombar_648.nc",
        None,
    ),
    "cabot": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery/"
        "Cabot_20240528/Cabot_645.nc",
        "Cabot_645.nc",
        None,
    ),
    # --- Custard ---
    "custard_churchill": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery/"
        "Churchill_20181204/Churchill_501.nc",
        "Churchill_501.nc",
        None,
    ),
    "bellamite": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery/"
        "Bellamite_20191206/Bellamite_538.nc",
        "Bellamite_538.nc",
        None,
    ),
    # --- Rebels ---
    "growler": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery/"
        "Growler_20250323/Growler_677.nc",
        "Growler_677.nc",
        None,
    ),
    "stella": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001/"
        "Stella_20260403/Stella_713_R.nc",
        "Stella_713_R.nc",
        None,
    ),
}

#: Where demo files are downloaded to, relative to the repo root.
DEMO_DATA_DIR = "examples/data/OG1"
