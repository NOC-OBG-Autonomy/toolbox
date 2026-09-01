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
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glidertools as gt


@register_step
class BBPFromBeta(BaseStep, QCHandlingMixin):

    step_name = "BBP from Beta"
    required_variables = ["TIME", "DEPTH", "TEMP", "PRAC_SALINITY"]
    provided_variables = []

    parameter_schema = {
        "apply_to": {
            "type": str,
            "default": "BBP700",
            "description": "Name of the variable to convert.",
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
                apply_to: "BBP700"
                output_as: "BBP700"
                theta: 124
                xfactor: 1.076
              diagnostics: false

        Returns
        -------

        """
        self.filter_qc()

        # Get the required variables
        self.data_subset = self.data[
            ["TIME", "PROFILE_NUMBER", "DEPTH", "TEMP", "PRAC_SALINITY", self.apply_to]
        ]

        # Interp DEPTH, TEMP and PRAC_SALINITY
        for var in ["DEPTH", "TEMP", "PRAC_SALINITY"]:
            self.data_subset[var][:] = interpolate_nans(
                self.data_subset[var], self.data_subset["TIME"]
            )

        # Apply the correction.
        #
        # NOTE: glidertools' flo_bback_total is deliberately NOT used here. It returns
        # the TOTAL backscattering coefficient (particulate + seawater), because it adds
        # the seawater backscattering term bsw/2 back in before returning. That is not
        # what BBP700 is meant to hold: BBP is the PARTICULATE backscattering
        # coefficient, following the BGC-Argo definition
        #
        #     BBP = 2 * pi * chi * (beta(theta) - beta_sw(theta))
        #
        # so only the seawater volume scattering function at the measurement angle is
        # subtracted, and no seawater backscattering is added back.
        betasw, _bsw = gt.flo_functions.flo_zhang_scatter_coeffs(
            self.data_subset["TEMP"],
            self.data_subset["PRAC_SALINITY"],
            self.theta,
            700,
        )
        bbp_corrected = (
            self.xfactor * 2.0 * np.pi * (self.data_subset[self.apply_to] - betasw)
        )

        # Stitch back into the data
        self.data[self.output_as] = bbp_corrected

        self.reconstruct_data()
        self.update_qc()

        # Generate QC if a new variable is added. Otherwise warn the user that input is being overwritten.
        if self.apply_to != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.apply_to}_QC"]})
        else:
            self.log_warn(
                f"'apply_to' and 'output_as' are the same. This will cause {self.apply_to} to be overwritten."
            )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        mpl.use("tkagg")

        # Clean both datasets
        beta_clean = remove_outliers(self.data_subset[self.apply_to])
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

        self.baseline, self.spikes = gt.cleaning.despike(
            self.data[self.apply_to], self.window_size, spike_method=self.method
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

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        mpl.use("tkagg")

        raw = self.data[self.apply_to]

        # Plot
        fig, axs = plt.subplots(
            nrows=2, figsize=(10, 6), height_ratios=(2, 1), sharex=True
        )

        # Plot original and baseline time series
        axs[0].plot(
            self.data["TIME"][~np.isnan(raw)],
            raw[~np.isnan(raw)],
            ls="--",
            c="gray",
            label="Raw",
        )
        axs[0].plot(
            self.data["TIME"][~np.isnan(self.baseline)],
            self.baseline[~np.isnan(self.baseline)],
            c="b",
            alpha=0.5,
            label="Baseline",
        )

        # Plot spike points
        axs[1].plot(
            self.data["TIME"][~np.isnan(self.spikes)],
            self.spikes[~np.isnan(self.spikes)],
            marker="o",
            c="r",
            label="Spikes",
        )

        for ax in axs:
            ax.legend(loc="upper right")

        ax.set(
            xlabel="Time",
            ylabel=self.apply_to,
            title=f"{self.apply_to}: Baseline Timeseries & Spikes",
        )

        fig.tight_layout()
        plt.show(block=True)
