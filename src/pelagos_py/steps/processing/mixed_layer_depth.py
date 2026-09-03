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

# Dataset variables each method reads; density is derived here as potential density.
METHOD_INPUTS = {"density": ["ABS_SALINITY", "CONS_TEMP"], "temp": ["TEMP"]}

# Name each method's values go by in logs and on plots.
METHOD_LABELS = {"density": "SIGMA0", "temp": "TEMP"}


@register_step
class MixedLayerDepthStep(BaseStep, QCHandlingMixin):
    """
    Calculate the mixed layer depth (MLD) of each profile.

    The MLD is found per profile by a threshold method: from a near-surface
    reference point, the first depth at which the chosen variable departs from its
    reference value by more than a threshold marks the base of the mixed layer.
    ``PROFILE_NUMBER`` and ``DEPTH`` must already be derived.

    Two derived variables are written on the ``N_MEASUREMENTS`` dimension: ``MLD``,
    the profile's mixed layer depth (positive down, matching ``DEPTH``; ``NaN`` where
    undefined), and ``MLD_BOOL``, ``0`` above the MLD, ``1`` at/below it, ``NaN``
    where undefined. Samples whose flags fall in ``calculation_flag_filter`` take no
    part in the search, so a bad value cannot place the MLD; both variables are still
    written for every measurement in the profile.

    The density method uses potential density (sigma0, referenced to 0 dbar) derived
    here from ``ABS_SALINITY`` and ``CONS_TEMP``, not the dataset's in-situ
    ``DENSITY`` whose pressure term alone crosses a typical threshold within a few
    metres of the surface. The derived sigma0 is not written to the dataset.

    Parameters
    ----------
    method : str, optional
        ``"density"``, ``"temp"``, or ``"auto"`` (default; density if its inputs are
        present, else temp). An explicit choice whose inputs are missing halts the
        pipeline.
    reference_depth : float, optional
        Near-surface reference depth (positive down). Default ``10``.
    density_threshold : float, optional
        Density departure (kg/m3) marking the MLD. Default ``0.03``.
    temp_threshold : float, optional
        Temperature departure (degC) marking the MLD. Default ``0.2``.

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
    # ABS_SALINITY/CONS_TEMP/TEMP cover both "auto" method resolution and the
    # diagnostics panel, which computes MLD from every available method, not
    # just the configured one; TIME/PROFILE_DIRECTION are read only if present.
    optional_variables = [
        "ABS_SALINITY", "CONS_TEMP", "TEMP", "TIME", "PROFILE_DIRECTION",
    ]
    uses_data_subset = True

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

        self.resolved_method, self.threshold = self._resolve_method()
        self.log(
            f"Calculating MLD from {METHOD_LABELS[self.resolved_method]} "
            f"(threshold {self.threshold}, reference depth {self.reference_depth})..."
        )

        depth = self.data["DEPTH"].values
        mld = self._compute_mld(self.resolved_method, self.threshold, progress=True)

        # Above MLD -> 0, at/below -> 1, NaN where depth or the profile's MLD is undefined.
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

        self.context["data"].update(self.data)
        return self.context

    def _resolve_method(self):
        # Returns (method, threshold); "auto" prefers density, falls back to temp.
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
        return all(var in self.data.data_vars for var in METHOD_INPUTS[method])

    def _method_values(self, method):
        # Density is potential density (sigma0) derived here, only to place the MLD.
        if method == "temp":
            return self.data["TEMP"].values
        return gsw.sigma0(
            self.data["ABS_SALINITY"].values, self.data["CONS_TEMP"].values
        )

    def _compute_mld(self, method, threshold, progress=False):
        profile_number = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values

        # Flagged samples must not influence the MLD, so the search runs on NaN-masked
        # copies; MLD/MLD_BOOL are still assigned to every sample using unmasked depth.
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
        # Restrict to valid points at or below the reference depth.
        below_reference = depth >= self.reference_depth
        valid = below_reference & ~np.isnan(depth) & ~np.isnan(values)
        depth = depth[valid]
        values = values[valid]
        if depth.size == 0:
            return np.nan

        # Reference point: shallowest remaining measurement; if deeper than twice the
        # reference depth there is nothing near the surface to anchor to.
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
        # (indices, [x_start, x_end]) per profile; span the core (PROFILE_DIRECTION
        # +-1) where available so MLD lines don't stretch across transect legs.
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

        # Both MLDs are drawn on every panel for comparison; only the selected one is stored.
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

            # Flagged samples and points below the plotted range are kept off the scatter.
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
