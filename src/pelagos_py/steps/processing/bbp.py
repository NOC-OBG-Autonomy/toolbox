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

"""Pipeline steps for deriving particulate backscatter (BBP) from beta and isolating BBP spikes."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
from pelagos_py.utils.processing_utils import *
import pelagos_py.utils.diagnostics as diag

#### Custom imports ####
import re
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glidertools as gt
from pelagos_py.utils import fig_spec


@register_step
class BBPFromBeta(BaseStep, QCHandlingMixin):

    step_name = "BBP from Beta"
    required_variables = ["TIME", "DEPTH", "TEMP", "PRAC_SALINITY"]
    provided_variables = []
    # PROFILE_NUMBER is read in run() (data_subset) but not declared required above
    # (pre-existing; left as-is to avoid changing pipeline-validation behaviour).
    optional_variables = ["PROFILE_NUMBER"]
    variable_parameters = ["apply_to", "output_as"]
    # apply_to has a fallback chain (see _resolve_beta_var) resolved by the
    # step itself at run time, so the generic variable_parameters check can't
    # tell whether it's actually missing.
    variable_parameters_optional = ("apply_to",)
    uses_data_subset = True

    parameter_schema = {
        "apply_to": {
            "type": str,
            "default": "BETA_BACKSCATTERING700",
            "description": (
                "Name of the beta backscatter variable to convert. If not found, "
                "falls back to any other BETA_BACKSCATTERING<wavelength> variable "
                "(closest to 700nm if several are present), then to a "
                "BBP<wavelength> variable."
            ),
        },
        "output_as": {
            "type": str,
            "default": "BBP700",
            "description": "Name for the output variable added to the dataset.",
        },
        "theta": {
            "type": float,
            "default": 124,
            "description": "Effective optical backscatter scattering angle (degrees).",
        },
        "xfactor": {
            "type": float,
            "default": 1.076,
            "description": "Chi factor scaling particulate scattering to total backscatter.",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "BBP from Beta"
              parameters:
                apply_to: "BETA_BACKSCATTERING700"
                output_as: "BBP700"
                theta: 124
                xfactor: 1.076
              diagnostics: false

        Returns
        -------

        """
        self.filter_qc()

        self.beta_var = self._resolve_beta_var()

        # Get the required variables
        self.data_subset = self.data[
            ["TIME", "PROFILE_NUMBER", "DEPTH", "TEMP", "PRAC_SALINITY", self.beta_var]
        ]

        # Gaps in TEMP/PRAC_SALINITY are left as NaN: BBP is not derived there and is
        # flagged missing (9) below. Add an Interpolate Data step first for gap-free BBP.

        # Apply the correction
        bbp_corrected = gt.flo_functions.flo_bback_total(
            self.data_subset[self.beta_var],
            self.data_subset["TEMP"],
            self.data_subset["PRAC_SALINITY"],
            self.theta,
            700,
            self.xfactor,
        )

        # Stitch back into the data
        self.data[self.output_as] = bbp_corrected
        self.data[self.output_as].attrs["units"] = "m-1"
        self.data[self.output_as].attrs["long_name"] = "Total particulate backscatter"
        self.data[self.output_as].attrs["standard_name"] = self.output_as

        self.reconstruct_data()
        self.update_qc()

        # Generate QC if a new variable is added. Otherwise warn the user that input is being overwritten.
        if self.beta_var != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.beta_var}_QC"]})
        else:
            self.log_warn(
                f"'apply_to' and 'output_as' are the same. This will cause {self.beta_var} to be overwritten."
            )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"].update(self.data)
        return self.context

    def _resolve_beta_var(self):
        """Resolve the beta backscatter variable, walking down a fallback chain
        when `apply_to` isn't present: any other BETA_BACKSCATTERING<wavelength>
        variable (closest to 700nm if several), then a BBP<wavelength> variable.
        """
        full_vars = self.context["data"].data_vars
        if self.apply_to in full_vars:
            return self._pull_into_subset(self.apply_to)

        fallback = self._closest_wavelength_var(full_vars, "BETA_BACKSCATTERING", exclude={self.apply_to})
        if fallback:
            self.log_warn(f"'{self.apply_to}' not found; using '{fallback}' instead.")
            return self._pull_into_subset(fallback)

        # TODO: current glider files from BODC mistakenly label raw beta backscatter
        # as BBP<wavelength> (a derived-variable name); remove this fallback once
        # BODC fixes the mislabelling upstream.
        fallback = self._closest_wavelength_var(full_vars, "BBP")
        if fallback:
            self.log_warn(f"No BETA_BACKSCATTERING* variable found; using mislabelled '{fallback}' instead.")
            return self._pull_into_subset(fallback)

        raise ValueError(
            f"'{self.apply_to}' not found, and no BETA_BACKSCATTERING* or BBP<wavelength> "
            "variable is present to fall back to."
        )

    def _pull_into_subset(self, name):
        # variable_parameters subsetting (QCHandlingMixin.__init__) only knows the
        # configured apply_to, so a fallback name resolved here may not be in
        # self.data yet.
        if name not in self.data:
            self.data[name] = self.context["data"][name]
            if f"{name}_QC" in self.context["data"]:
                self.data[f"{name}_QC"] = self.context["data"][f"{name}_QC"]
        return name

    @staticmethod
    def _closest_wavelength_var(names, prefix, exclude=()):
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        candidates = [
            (abs(int(match.group(1)) - 700), name)
            for name in names
            if name not in exclude and (match := pattern.match(name))
        ]
        return min(candidates)[1] if candidates else None

    def generate_diagnostics(self):
        mpl.use("tkagg")

        # Clean both datasets
        beta_clean = remove_outliers(self.data_subset[self.beta_var])
        bbp_clean = remove_outliers(self.data[self.output_as])

        # Plot
        plt.figure(figsize=(10, 6))
        plt.boxplot(
            [beta_clean, bbp_clean],
            vert=True,
            patch_artist=True,
            labels=["Beta", "BBP"],
        )

        plt.title("Beta vs BBP")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show(block=True)


@register_step
class IsolateBBPSpikes(BaseStep, QCHandlingMixin):

    step_name = "Isolate BBP Spikes"
    required_variables = ["TIME"]
    provided_variables = []
    variable_parameters = ["apply_to"]
    uses_data_subset = True

    parameter_schema = {
        "apply_to": {
            "type": str,
            "default": "BBP700",
            "description": "Name of the variable to filter.",
        },
        "window_size": {
            "type": int,
            "default": 50,
            "description": "Median/minmax filter window size in samples.",
        },
        "method": {
            "type": str,
            "default": "median",
            "description": "Filter method used to determine the baseline.",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Isolate BBP Spikes"
              parameters:
                apply_to: "BBP700"
                window_size: 50
                method: "median"
              diagnostics: false

        Returns
        -------

        """
        self.filter_qc()

        # Flagged samples are left out of the despike so they cannot drag their
        # neighbours' rolling baseline; they get no baseline and are flagged missing (9)
        # below. Add an Interpolate Data step first for a gap-free baseline.
        usable = self.data[self.apply_to].where(
            self.calculation_mask([self.apply_to])
        )

        self.baseline, self.spikes = gt.cleaning.despike(
            usable, self.window_size, spike_method=self.method
        )

        self.data[f"{self.apply_to}_BASELINE"] = self.baseline
        self.data[f"{self.apply_to}_SPIKES"] = self.spikes

        self.reconstruct_data()
        self.update_qc()

        # Generate QC if a new variable is added. Otherwise warn the user that input is being overwritten.
        self.generate_qc(
            {
                f"{self.apply_to}_BASELINE_QC": [f"{self.apply_to}_QC"],
                f"{self.apply_to}_SPIKES_QC": [f"{self.apply_to}_QC"],
            }
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"].update(self.data)
        return self.context

    def generate_diagnostics(self):
        mpl.use("tkagg")

        raw = self.data[self.apply_to]
        time = self.data["TIME"]

        fig, axes = fig_spec.new_fig(nrows=2, sharex=True, height_ratios=(2, 1))
        ax1, ax2 = axes[0][0], axes[1][0]

        # Panel 1: raw and baseline time series.
        ax1.plot(time[~np.isnan(raw)], raw[~np.isnan(raw)],
                 ls="--", color=fig_spec.FLAGGED, label="Raw")
        ax1.plot(time[~np.isnan(self.baseline)], self.baseline[~np.isnan(self.baseline)],
                 color=fig_spec.CATEGORY[1], alpha=fig_spec.ALPHA, label="Baseline")

        # Panel 2: isolated spike points.
        fig_spec.points(ax2, time[~np.isnan(self.spikes)], self.spikes[~np.isnan(self.spikes)],
                        color=fig_spec.CATEGORY[2], label="Spikes")

        ylabel = fig_spec.axis_label(self.apply_to, self.data[self.apply_to].attrs.get("units"))
        for ax in (ax1, ax2):
            fig_spec.date_axis(ax, which="x")
            fig_spec.legend(ax)
        fig_spec.style_axes(ax1, ylabel=ylabel)
        fig_spec.style_axes(ax2, xlabel="Time", ylabel=ylabel)

        fig_spec.finish(fig, suptitle=f"{self.apply_to}: Baseline Timeseries & Spikes")
        plt.show(block=True)
