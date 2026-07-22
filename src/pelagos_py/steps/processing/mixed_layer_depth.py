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
import gsw
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Maps the user-facing ``method`` choice to the dataset variables it reads. Density
# is derived here as potential density rather than read from DENSITY: that variable
# is in-situ, and its pressure term alone crosses the threshold within a few metres
# of the reference depth, so it cannot place a mixed layer.
METHOD_INPUTS = {"density": ["ABS_SALINITY", "CONS_TEMP"], "temp": ["TEMP"]}

# Name each method's values go by in logs and on plots.
METHOD_LABELS = {"density": "SIGMA0", "temp": "TEMP"}


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
    part in the search above, so a bad value cannot place the MLD; for the density
    method this keys off the flags of the inputs density is derived from. Both
    derived variables are still written for every measurement in the profile.

    The density method uses potential density (sigma0, referenced to 0 dbar) derived
    within this step, not the dataset's ``DENSITY``: that variable is in-situ, and its
    pressure term alone crosses a typical threshold within a few metres of the
    reference depth. The derived sigma0 is not written to the dataset.

    Parameters
    ----------
    method : str, optional
        Values the threshold is applied to: ``"density"`` (potential density,
        derived here from ``ABS_SALINITY`` and ``CONS_TEMP``) or ``"temp"``
        (``TEMP``). Default ``"auto"`` uses density if its inputs are present,
        otherwise falls back to temp. An explicit choice whose inputs are missing
        halts the pipeline.
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

        # Resolve which values the threshold keys off, honouring the density ->
        # temp fallback when method is "auto".
        self.resolved_method, self.threshold = self._resolve_method()
        self.log(
            f"Calculating MLD from {METHOD_LABELS[self.resolved_method]} "
            f"(threshold {self.threshold}, reference depth {self.reference_depth})..."
        )

        depth = self.data["DEPTH"].values
        mld = self._compute_mld(self.resolved_method, self.threshold, progress=True)

        # Above the MLD (shallower) -> 0, at/below (deeper) -> 1, NaN where either
        # the depth or the profile's MLD is undefined.
        mld_bool = np.where(depth >= mld, 1.0, 0.0)
        mld_bool[np.isnan(depth) | np.isnan(mld)] = np.nan

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
            for var in ["PROFILE_NUMBER", "DEPTH"] + METHOD_INPUTS[self.resolved_method]
        ]
        self.generate_qc({"MLD_QC": qc_parents, "MLD_BOOL_QC": qc_parents})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def _resolve_method(self):
        """Return ``(method, threshold)`` for the configured method.

        ``"auto"`` prefers density and falls back to temp. An explicit method whose
        inputs are absent halts the pipeline.
        """
        method = str(self.method).lower()

        if method == "auto":
            for candidate in ("density", "temp"):
                if self._method_available(candidate):
                    method = candidate
                    break
            else:
                self.halt(
                    "Method 'auto' needs either ABS_SALINITY and CONS_TEMP (density) or "
                    "TEMP in the dataset, but neither set is present. Derive them "
                    "beforehand (e.g. Derive CTD)."
                )

        if method not in METHOD_INPUTS:
            self.halt(
                f"Unknown MLD method '{self.method}'. Choose 'auto', 'density', or 'temp'."
            )

        if not self._method_available(method):
            missing = [
                var for var in METHOD_INPUTS[method] if var not in self.data.data_vars
            ]
            self.halt(
                f"Method '{method}' requires {', '.join(METHOD_INPUTS[method])} in the "
                f"dataset, but {', '.join(missing)} is not present. Derive it beforehand "
                f"(e.g. Derive CTD)."
            )

        threshold = (
            self.density_threshold if method == "density" else self.temp_threshold
        )
        return method, threshold

    def _method_available(self, method):
        """True where every dataset variable a method reads is present."""
        return all(var in self.data.data_vars for var in METHOD_INPUTS[method])

    def _method_values(self, method):
        """Return the values a method applies its threshold to.

        Density is potential density (sigma0, referenced to 0 dbar) derived here from
        ABS_SALINITY and CONS_TEMP; it is used only to place the MLD and is not
        written to the dataset.
        """
        if method == "temp":
            return self.data["TEMP"].values
        return gsw.sigma0(
            self.data["ABS_SALINITY"].values, self.data["CONS_TEMP"].values
        )

    def _compute_mld(self, method, threshold, progress=False):
        """Return the per-measurement MLD array for one method.

        Every measurement in a profile carries that profile's MLD; measurements
        outside a profile, or in a profile with no MLD, are ``NaN``.
        """
        profile_number = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values

        # Flagged samples must not influence where the MLD sits, so the search below
        # runs on NaN-masked copies. For density this gates on the flags of the inputs
        # sigma0 is derived from. MLD/MLD_BOOL are still assigned to every sample in
        # the profile, using the unmasked depth.
        usable = self.calculation_mask(
            ["PROFILE_NUMBER", "DEPTH"] + METHOD_INPUTS[method]
        )
        search_depth = np.where(usable, depth, np.nan)
        search_values = np.where(usable, self._method_values(method), np.nan)

        mld = np.full(profile_number.shape, np.nan)
        profile_numbers = np.unique(profile_number[~np.isnan(profile_number)])
        if progress:
            profile_numbers = self.log_progress(profile_numbers, desc="", unit="prof")

        for pn in profile_numbers:
            indices = np.where(profile_number == pn)[0]
            profile_mld = self._profile_mld(
                search_depth[indices], search_values[indices], threshold
            )
            if np.isfinite(profile_mld):
                mld[indices] = profile_mld
        return mld

    def _profile_mld(self, depth, values, threshold):
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
        exceeded = np.where(np.abs(values - reference_value) >= np.abs(threshold))[0]
        if exceeded.size == 0:
            return np.nan
        return float(depth[exceeded[0]])

    # Panels and MLD lines are drawn for each of these that is present in the data.
    DIAGNOSTIC_METHODS = (("temp", "red"), ("density", "black"))
    DIAGNOSTIC_MAX_DEPTH = 200

    def _profile_spans(self, profile_number, x):
        """Return ``(indices, [x_start, x_end])`` per profile, for drawing MLD lines.

        PROFILE_NUMBER is extended over the transects either side of a profile's
        ascent/descent core, so the line is spanned over the core alone (marked by
        PROFILE_DIRECTION +-1) where that variable is available — spanning the whole
        profile would stretch it across its transect legs.
        """
        direction = (
            self.data["PROFILE_DIRECTION"].values
            if "PROFILE_DIRECTION" in self.data
            else None
        )

        spans = []
        for pn in np.unique(profile_number[~np.isnan(profile_number)]):
            indices = np.where(profile_number == pn)[0]
            span_indices = indices
            if direction is not None:
                core = indices[np.isin(direction[indices], (-1, 1))]
                if core.size:
                    span_indices = core
            span_x = x[span_indices]
            if span_x.size == 0:
                continue
            spans.append((indices, [np.min(span_x), np.max(span_x)]))
        return spans

    def generate_diagnostics(self):
        """Plot the top 200 m of the depth time series, one panel per available
        threshold variable, each overlaid with the MLD from every method."""
        matplotlib.use("tkagg")

        profile_number = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values
        # TIME is not required by this step; fall back to measurement index.
        x = (
            self.data["TIME"].values
            if "TIME" in self.data
            else np.arange(depth.size)
        )

        profile_spans = self._profile_spans(profile_number, x)

        # Both MLDs are drawn on every panel so the unselected method can be compared
        # against the selected one; only the selected one is stored in the dataset.
        methods = [
            (method, colour)
            for method, colour in self.DIAGNOSTIC_METHODS
            if self._method_available(method)
        ]
        mlds = []
        for method, colour in methods:
            threshold = (
                self.density_threshold if method == "density" else self.temp_threshold
            )
            mld = (
                self.data["MLD"].values
                if method == self.resolved_method
                else self._compute_mld(method, threshold)
            )
            mlds.append((f"MLD from {method}", colour, mld))

        fig, axes = plt.subplots(
            len(methods), 1, figsize=(14, 4 * len(methods)), dpi=150, sharex=True,
            squeeze=False,
        )
        for ax, (method, _) in zip(axes[:, 0], methods):
            panel_label = METHOD_LABELS[method]
            values = self._method_values(method)

            # Flagged samples are kept off the scatter so they cannot stretch the
            # colourbar, as are points below the plotted depth range.
            usable = self.calculation_mask(["DEPTH"] + METHOD_INPUTS[method])
            valid = (
                usable
                & ~np.isnan(depth)
                & ~np.isnan(values)
                & (depth <= self.DIAGNOSTIC_MAX_DEPTH)
            )
            cmap = palettes.cmap_for_variable(panel_label, default="viridis")
            scatter = ax.scatter(x[valid], depth[valid], c=values[valid], s=2, cmap=cmap)
            colourbar = fig.colorbar(scatter, ax=ax)
            colourbar.set_label(panel_label)

            # Draw each MLD as a line spanning every profile's span.
            for label, colour, mld in mlds:
                for indices, span in profile_spans:
                    finite_mld = mld[indices][np.isfinite(mld[indices])]
                    if finite_mld.size == 0:
                        continue
                    ax.plot(
                        span,
                        [finite_mld[0], finite_mld[0]],
                        c=colour,
                        lw=1,
                        label=label,
                    )
                    label = None  # only label the first line of each MLD

            ax.set_ylabel("DEPTH")
            # Positive-down depth: surface at the top, clipped to the shallow range.
            ax.set_ylim(self.DIAGNOSTIC_MAX_DEPTH, 0)
            ax.set_title(f"Mixed Layer Depth ({panel_label})")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="lower right")

        axes[-1, 0].set_xlabel("TIME" if "TIME" in self.data else "Measurement")
        fig.tight_layout()
        plt.show(block=True)
