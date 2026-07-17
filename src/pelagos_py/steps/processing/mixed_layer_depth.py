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

"""Pipeline step for calculating the mixed layer depth (MLD) of each profile."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag
import pelagos_py.utils.palettes as palettes

#### Custom imports ####
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Maps the user-facing ``method`` choice to the dataset variable it keys off.
METHOD_VARIABLES = {"density": "DENSITY", "temp": "TEMP"}


@register_step
class MixedLayerDepthStep(BaseStep, QCHandlingMixin):
    """
    Calculate the mixed layer depth (MLD) of each profile.

    The MLD is found per profile by a threshold method: starting from a
    near-surface reference point, the first depth at which the chosen variable
    departs from its reference value by more than a threshold marks the base of
    the mixed layer. Because a depth (in metres) is wanted, the step keys off
    ``DEPTH`` and requires that both ``PROFILE_NUMBER`` and ``DEPTH`` have already
    been derived.

    Two derived variables are written, both on the ``N_MEASUREMENTS`` dimension:

    - ``MLD`` — the mixed layer depth of the profile each measurement belongs to,
      in the same positive-down convention as ``DEPTH`` (e.g. ``25.0`` m). It is
      ``NaN`` for measurements not in a profile, or in a profile for which no MLD
      could be found.
    - ``MLD_BOOL`` — ``0`` where the measurement is above the MLD (shallower),
      ``1`` where it is at or below the MLD (deeper), and ``NaN`` where the MLD is
      undefined for that measurement.

    Samples whose flags fall in ``calculation_flag_filter`` (by default
    probably-bad (3), bad (4) and missing (9); see ``qc_handling_settings``) take no
    part in the search above, so a bad density value cannot place the MLD. Both
    derived variables are still written for every measurement in the profile.

    Parameters
    ----------
    method : str, optional
        Variable the threshold is applied to: ``"density"`` (``DENSITY``) or
        ``"temp"`` (``TEMP``). Default ``"auto"`` uses ``DENSITY`` if present,
        otherwise falls back to ``TEMP``. An explicit choice whose variable is
        missing halts the pipeline.
    reference_depth : float, optional
        Near-surface reference depth (positive down). The reference value is taken
        at the shallowest measurement at or below this depth. Default ``10``.
    density_threshold : float, optional
        Density departure (kg/m3) from the reference marking the MLD, used when the
        method resolves to density. Default ``0.03``.
    temp_threshold : float, optional
        Temperature departure (degC) from the reference marking the MLD, used when
        the method resolves to temperature. Default ``0.2``.

    Examples
    --------
    .. code-block:: yaml

        steps:
          - name: Mixed Layer Depth
            parameters:
              method: density
              reference_depth: 10
              density_threshold: 0.03
              temp_threshold: 0.2
            diagnostics: true
    """

    step_name = "Mixed Layer Depth"
    required_variables = ["PROFILE_NUMBER", "DEPTH"]
    provided_variables = ["MLD", "MLD_BOOL"]

    parameter_schema = {
        "method": {
            "type": str,
            "default": "auto",
            "options": ["auto", "density", "temp"],
            "description": "Variable the threshold keys off: 'density', 'temp', or 'auto' (density, else temp).",
        },
        "reference_depth": {
            "type": [int, float],
            "default": 10,
            "description": "Near-surface reference depth (positive down).",
        },
        "density_threshold": {
            "type": [int, float],
            "default": 0.03,
            "description": "Density departure (kg/m3) from the reference marking the MLD.",
        },
        "temp_threshold": {
            "type": [int, float],
            "default": 0.2,
            "description": "Temperature departure (degC) from the reference marking the MLD.",
        },
    }

    def run(self):
        self.filter_qc()

        # Resolve which variable the threshold keys off, honouring the density ->
        # temp fallback when method is "auto".
        self.threshold_variable, self.threshold = self._resolve_method()
        self.log(
            f"Calculating MLD from {self.threshold_variable} "
            f"(threshold {self.threshold}, reference depth {self.reference_depth})..."
        )

        profile_number = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values
        threshold_values = self.data[self.threshold_variable].values

        # Flagged samples must not influence where the MLD sits, so the search below
        # runs on NaN-masked copies. MLD/MLD_BOOL are still assigned to every sample
        # in the profile, using the unmasked depth.
        usable = self.calculation_mask(
            ["PROFILE_NUMBER", "DEPTH", self.threshold_variable]
        )
        search_depth = np.where(usable, depth, np.nan)
        search_values = np.where(usable, threshold_values, np.nan)

        mld = np.full(profile_number.shape, np.nan)
        mld_bool = np.full(profile_number.shape, np.nan)

        profile_numbers = np.unique(profile_number[~np.isnan(profile_number)])
        for pn in tqdm(
            profile_numbers, colour="green", desc="\033[97mProgress\033[0m", unit="prof"
        ):
            indices = np.where(profile_number == pn)[0]
            profile_mld = self._profile_mld(
                search_depth[indices], search_values[indices]
            )
            if not np.isfinite(profile_mld):
                continue

            profile_depth = depth[indices]
            mld[indices] = profile_mld
            # Above the MLD (shallower) -> 0, at/below (deeper) -> 1, NaN depth -> NaN.
            flags = np.where(profile_depth >= profile_mld, 1.0, 0.0)
            flags[np.isnan(profile_depth)] = np.nan
            mld_bool[indices] = flags

        self.data["MLD"] = (("N_MEASUREMENTS",), mld)
        self.data["MLD"].attrs = {
            "long_name": "Mixed layer depth of the profile (positive down, matching DEPTH). NaN where undefined.",
            "units": "m",
            "standard_name": "MLD",
        }

        self.data["MLD_BOOL"] = (("N_MEASUREMENTS",), mld_bool)
        self.data["MLD_BOOL"].attrs = {
            "long_name": "Mixed layer flag: 0 above MLD, 1 at/below MLD, NaN where undefined.",
            "units": "None",
            "standard_name": "MLD_BOOL",
            "valid_min": 0,
            "valid_max": 1,
            "flag_values": "0, 1",
            "flag_meanings": "above_mld below_mld",
        }

        self.reconstruct_data()
        self.update_qc()

        qc_parents = [
            f"{var}_QC"
            for var in ("PROFILE_NUMBER", "DEPTH", self.threshold_variable)
        ]
        self.generate_qc({"MLD_QC": qc_parents, "MLD_BOOL_QC": qc_parents})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def _resolve_method(self):
        """Return ``(threshold_variable, threshold)`` for the configured method.

        ``"auto"`` prefers ``DENSITY`` and falls back to ``TEMP``. An explicit
        method whose variable is absent halts the pipeline.
        """
        method = str(self.method).lower()

        if method == "auto":
            for candidate in ("density", "temp"):
                if METHOD_VARIABLES[candidate] in self.data.data_vars:
                    method = candidate
                    break
            else:
                self.halt(
                    "Method 'auto' needs DENSITY or TEMP in the dataset, but neither "
                    "is present. Derive one beforehand (e.g. Derive CTD)."
                )

        if method not in METHOD_VARIABLES:
            self.halt(
                f"Unknown MLD method '{self.method}'. Choose 'auto', 'density', or 'temp'."
            )

        variable = METHOD_VARIABLES[method]
        if variable not in self.data.data_vars:
            self.halt(
                f"Method '{method}' requires {variable} in the dataset, but it is "
                f"not present. Derive it beforehand (e.g. Derive CTD)."
            )

        threshold = (
            self.density_threshold if method == "density" else self.temp_threshold
        )
        return variable, threshold

    def _profile_mld(self, depth, values):
        """Return the MLD (positive-down metres) for one profile, or ``NaN``.

        Starting from the shallowest measurement at or below ``reference_depth``,
        the MLD is the shallowest depth whose value departs from that reference by
        at least ``threshold``.
        """
        # Restrict to valid points at or below the reference depth.
        below_reference = depth >= self.reference_depth
        valid = below_reference & ~np.isnan(depth) & ~np.isnan(values)
        depth = depth[valid]
        values = values[valid]
        if depth.size == 0:
            return np.nan

        # Reference point: the shallowest remaining measurement (smallest DEPTH).
        # If it is deeper than twice the reference depth there is no data near the
        # surface to anchor to, so no MLD can be found.
        reference_index = np.argmin(depth)
        if depth[reference_index] > 2 * self.reference_depth:
            return np.nan
        reference_value = values[reference_index]

        # Scan from the surface downward for the first threshold crossing.
        order = np.argsort(depth)
        depth = depth[order]
        values = values[order]
        exceeded = np.where(np.abs(values - reference_value) >= np.abs(self.threshold))[0]
        if exceeded.size == 0:
            return np.nan
        return float(depth[exceeded[0]])

    def generate_diagnostics(self):
        """Plot the depth time series coloured by the threshold variable, with a
        red line marking the MLD across each profile's span."""
        matplotlib.use("tkagg")

        profile_number = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values
        threshold_values = self.data[self.threshold_variable].values
        mld = self.data["MLD"].values
        # TIME is not required by this step; fall back to measurement index.
        x = (
            self.data["TIME"].values
            if "TIME" in self.data
            else np.arange(depth.size)
        )

        fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

        valid = ~np.isnan(depth) & ~np.isnan(threshold_values)
        cmap = palettes.cmap_for_variable(self.threshold_variable, default="viridis")
        scatter = ax.scatter(
            x[valid], depth[valid], c=threshold_values[valid], s=2, cmap=cmap
        )
        colourbar = fig.colorbar(scatter, ax=ax)
        colourbar.set_label(self.threshold_variable)

        # Draw the MLD as a red line spanning each profile's time extent.
        mld_label = "MLD"
        for pn in np.unique(profile_number[~np.isnan(profile_number)]):
            indices = np.where(profile_number == pn)[0]
            finite_mld = mld[indices][np.isfinite(mld[indices])]
            profile_x = x[indices]
            if finite_mld.size == 0 or profile_x.size == 0:
                continue
            ax.plot(
                [np.min(profile_x), np.max(profile_x)],
                [finite_mld[0], finite_mld[0]],
                c="red",
                lw=1,
                label=mld_label,
            )
            mld_label = None  # only label the first line

        ax.set_xlabel("TIME" if "TIME" in self.data else "Measurement")
        ax.set_ylabel("DEPTH")
        ax.invert_yaxis()  # positive-down depth: surface at the top
        ax.set_title(f"Mixed Layer Depth ({self.threshold_variable})")
        if mld_label is None:
            ax.legend(loc="lower right")
        fig.tight_layout()
        plt.show(block=True)
