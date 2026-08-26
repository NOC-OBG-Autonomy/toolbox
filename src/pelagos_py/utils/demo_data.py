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

"""Registry of demo OG1 datasets used by the dashboard's download-on-demand
config picker (``dashboard/app.py``).

Each glider deployment can have up to two entries in ``DEMOS``: an NRT
(near-real-time, "_R" filename) and a Delayed/recovered-mode one, keyed
``"<base>_nrt"`` / ``"<base>_delayed"`` -- only the modes actually hosted for
that deployment are present. Every entry carries its download URL, output
filename, an optional TIME window to keep (``None`` for the whole file),
which shipped template config to derive its dashboard config from
("default" -- the standard glider template -- or "alr" for the ALR
platform's config), and the display label shown in the picker. Files are
hosted on the BODC deployment catalogue; see
https://noc.ac.uk/projects/bio-carbon for context. ``MISSIONS`` groups the
keys by deployment campaign, in picker display order.
"""

from dataclasses import dataclass

_MODE_LABELS = {"nrt": "NRT", "delayed": "Full"}


@dataclass(frozen=True)
class DemoEntry:
    url: str
    filename: str
    window: tuple[str, str] | None
    template: str  # "default" or "alr" -- which shipped config to derive from
    label: str  # glider display name, shared by its nrt/delayed variants
    mode: str  # "nrt" or "delayed"

    @property
    def display_label(self) -> str:
        """Glider name plus its mode, e.g. "Nelson (NRT)" -- distinguishes
        the two variants of the same glider in the picker."""
        return f"{self.label} ({_MODE_LABELS[self.mode]})"


_OG1_NRT = "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001"
_OG1_DELAYED = "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery"
_GLIDER_DATA = "https://linkedsystems.uk/erddap/files/Public_Glider_Data_0711"


def _variants(
    base: str, folder: str, template: str, label: str,
    *, nrt: str | None = None, delayed: str | None = None, window=None,
) -> dict:
    """Build the DEMOS entries for one glider deployment -- an "nrt" one if
    ``nrt`` (its "_R" filename) is given, a "delayed" one if ``delayed`` is,
    or both. Missing modes simply aren't hosted for that deployment."""
    entries = {}
    if nrt:
        entries[f"{base}_nrt"] = DemoEntry(
            f"{_OG1_NRT}/{folder}/{nrt}", nrt, None, template, label, "nrt",
        )
    if delayed:
        entries[f"{base}_delayed"] = DemoEntry(
            f"{_OG1_DELAYED}/{folder}/{delayed}", delayed, window, template, label, "delayed",
        )
    return entries


DEMOS = {
    # --- Bio-Carbon ---
    **_variants("nelson", "Nelson_20240528", "default", "Nelson",
                nrt="Nelson_646_R.nc", delayed="Nelson_646.nc"),
    **_variants("doombar", "Doombar_20240528", "default", "Doombar",
                nrt="Doombar_648_R.nc", delayed="Doombar_648.nc"),
    **_variants("churchill", "Churchill_20240528", "default", "Churchill",
                nrt="Churchill_647_R.nc", delayed="Churchill_647.nc",
                window=("2024-08-01", "2024-09-01")),
    **_variants("alr4", "ALR_4_20240609", "alr", "ALR 4",
                nrt="ALR_4_649_R.nc", delayed="ALR_4_649.nc"),
    **_variants("alr6", "ALR_6_20240611", "alr", "ALR 6",
                nrt="ALR_6_650_R.nc", delayed="ALR_6_650.nc"),
    **_variants("cabot", "Cabot_20240528", "default", "Cabot",
                nrt="Cabot_645_R.nc", delayed="Cabot_645.nc"),
    # --- Custard 1 ---
    **_variants("custard1_churchill", "Churchill_20181204", "default", "Churchill",
                nrt="Churchill_501_R.nc", delayed="Churchill_501.nc"),
    **_variants("pancake", "Pancake_20181209", "default", "Pancake",
                nrt="Pancake_502_R.nc"),  # no delayed-mode file hosted
    "custard1_doombar_nrt": DemoEntry(
        f"{_GLIDER_DATA}/Doombar_20181204/Doombar_503_R.nc",
        "Doombar_503_R.nc", None, "default", "Doombar", "nrt",
    ),  # only hosted on the raw glider-data store, not the OG1 one
    # --- Custard 2 ---
    **_variants("bellamite", "Bellamite_20191206", "default", "Bellamite",
                nrt="Bellamite_538_R.nc", delayed="Bellamite_538.nc"),
    **_variants("custard2_zephyr", "Zephyr_20191206", "default", "Zephyr",
                delayed="Zephyr_539.nc"),  # no NRT file hosted
    # --- ReBELS ---
    **_variants("rebels_zephyr", "Zephyr_20250323", "default", "Zephyr",
                nrt="Zephyr_675_R.nc", delayed="Zephyr_675.nc"),
    **_variants("omg1", "OMG-1_20250324", "default", "OMG-1",
                nrt="OMG-1_676_R.nc"),  # no delayed-mode file hosted
    **_variants("9ja", "9JA_20250812", "default", "9JA",
                nrt="9JA_699_R.nc", delayed="9JA_699.nc"),
    **_variants("growler", "Growler_20250323", "default", "Growler",
                nrt="Growler_677_R.nc", delayed="Growler_677.nc"),
    **_variants("rebels_stella", "Stella_20250323", "default", "Stella",
                nrt="Stella_678_R.nc", delayed="Stella_678.nc"),
    # --- ReBELS 2 ---
    **_variants("stella2026", "Stella_20260403", "default", "Stella",
                nrt="Stella_713_R.nc"),  # no delayed-mode file hosted (yet)
}

#: Picker display groups, in order: mission name -> base glider keys.
_MISSION_BASE_KEYS = {
    "Bio-Carbon": ["nelson", "doombar", "churchill", "alr4", "alr6", "cabot"],
    "Custard 1": ["custard1_churchill", "pancake", "custard1_doombar"],
    "Custard 2": ["bellamite", "custard2_zephyr"],
    "ReBELS": ["rebels_zephyr", "omg1", "9ja", "growler", "rebels_stella"],
    "ReBELS 2": ["stella2026"],
}

#: mission name -> ordered list of DEMOS keys (nrt then delayed per glider,
#: whichever modes actually exist for it).
MISSIONS = {
    mission: [
        f"{base}_{mode}"
        for base in bases
        for mode in ("nrt", "delayed")
        if f"{base}_{mode}" in DEMOS
    ]
    for mission, bases in _MISSION_BASE_KEYS.items()
}

#: Where demo files are downloaded to, relative to the repo root.
DEMO_DATA_DIR = "examples/data/OG1"
