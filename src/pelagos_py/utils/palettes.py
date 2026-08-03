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

"""Shared colour palettes for pelagos_py plots.

Single source of truth for the per-variable colourmaps used across the toolbox
(report cross-sections, step diagnostics, ...). Change a palette here and every
plot that pulls from it updates. Each entry is a list of hex stops ordered low
value -> high value (not reversed); ``get_cmap`` builds a matplotlib
``LinearSegmentedColormap`` from one.
"""

#: Per-variable sequential palettes, keyed by lower-case variable name. Stops run
#: low value -> high value.
SEQUENTIAL = {
    "temperature": [
        "#1b1c6e", "#365292", "#5286b7", "#8db4c4", "#dbe5cd",
        "#f1d8b4", "#d9997e", "#c05e4c", "#a0372b", "#811910",
    ],
    "salinity": [
        "#f9e8b1", "#f1c38f", "#e8a074", "#db7c5f", "#cb5c58",
        "#b2425c", "#943061", "#732460", "#511c53", "#321340",
    ],
    "density": [
        "#e6f1f7", "#c4d8e8", "#a6bed9", "#8ba4c9", "#7489b8", "#636c9f",
        "#595388", "#55406e", "#4e3055", "#3f2040", "#2e1226",
    ],
    "oxygen": [
        "#eaf85a", "#f5d848", "#fbba3e", "#fb9e3e", "#f4844c", "#e4725f",
        "#ca6774", "#af5f82", "#95568a", "#7c4d90", "#634198", "#4735a0",
        "#203487", "#0a2f59", "#313335",
    ],
    "chlorophyll": [
        "#182548", "#2c5398", "#4a88a4", "#87b8b5", "#dae4da",
        "#e6d992", "#a8ab3e", "#56872e", "#285932", "#1b2617",
    ],
    "backscatter": [
        "#cccccc", "#737373", "#000000", "#1335f5", "#3f8df7",
        "#67dffb", "#a1fc4e", "#f8d748", "#ef8733", "#ea3323",
    ],
    # cmocean "solar" sampled to hex (dark maroon -> bright yellow), for PAR.
    "solar": [
        "#331318", "#4d1c1a", "#6a2517", "#862f13", "#9c3d13", "#ad5017",
        "#bb641c", "#c67c24", "#cf9730", "#d4b445", "#d3d55e",
    ],
}

#: Dataset variable names -> :data:`SEQUENTIAL` palette key, so a step can pick
#: the right palette straight from the variable it is colouring.
VARIABLE_PALETTES = {
    "TEMP": "temperature",
    "CONS_TEMP": "temperature",
    "PSAL": "salinity",
    "PRAC_SALINITY": "salinity",
    "ABS_SALINITY": "salinity",
    "DENSITY": "density",
    "SIGMA0": "density",
    "DOXY": "oxygen",
    "CHLA": "chlorophyll",
    "CHLA_ADJUSTED": "chlorophyll",
    "BBP700": "backscatter",
    "BBP700_ADJUSTED": "backscatter",
}


def get_cmap(name):
    """Build a matplotlib colormap from a named :data:`SEQUENTIAL` palette.

    Parameters
    ----------
    name : str
        Palette key (case-insensitive), e.g. ``"chlorophyll"``.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    from matplotlib.colors import LinearSegmentedColormap

    key = name.lower()
    if key not in SEQUENTIAL:
        raise KeyError(
            f"No palette named '{name}'. Available: {', '.join(SEQUENTIAL)}."
        )
    return LinearSegmentedColormap.from_list(f"pelagos_{key}", SEQUENTIAL[key])


def cmap_for_variable(variable, default=None):
    """Return the colormap for a dataset variable name, or ``default``.

    Looks the variable up in :data:`VARIABLE_PALETTES` (case-insensitive) and
    builds the matching palette; variables with no mapping fall back to
    ``default`` (e.g. a matplotlib colormap name).

    Parameters
    ----------
    variable : str
        Dataset variable name, e.g. ``"DENSITY"`` or ``"CHLA_ADJUSTED"``.
    default : optional
        Returned when the variable has no palette. Default ``None``.
    """
    key = VARIABLE_PALETTES.get(str(variable).upper())
    if key is None:
        return default
    return get_cmap(key)
