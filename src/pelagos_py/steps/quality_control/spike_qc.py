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

"""QC test for flagging using spike/despike detection methods."""

#### Mandatory imports ####
import numpy as np
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib
from pelagos_py.utils import fig_spec


@register_qc
class spike_qc(BaseQC):
    """
    Target Variable: Any
    Flag Number: 4 (bad)
    Variables Flagged: Any
    Checks for spiking in the data using rolling median values compared against the
    median average deviation (MAD).

    EXAMPLE
    -------
    ::

        - name: "Apply QC"
          parameters:
            qc_settings: {
                "spike test": {
                  "variables": {"PRES": 2, "LATITUDE": 1},
                  "also_flag": {"PRES": ["CNDC", "TEMP"], "LATITUDE": ["LONGITUDE"]},
                  "plot": ["PRES", "LATITUDE"]
                  "window_size": 10,
                }
            }
          diagnostics: true
    """

    qc_name = "spike qc"

    # Samples already carrying these flags don't inform the rolling median / std
    # baseline (they'd bias detection); they keep their existing flag.
    IGNORE_FLAGS = [3, 4, 9]

    # Specify if test target variable is user-defined (if True, __init__ has to be redefined)
    dynamic = True

    parameter_schema = {
        "variables": {
            "type": dict,
            "required": True,
            "description": "Mapping of variable -> spike sensitivity, e.g. {'PRES': 2}.",
        },
        "also_flag": {
            "type": dict,
            "default": {},
            "description": "Propagate a variable's flags onto other variables.",
        },
        "plot": {
            "type": list,
            "default": [],
            "description": "Variables to plot in diagnostics.",
        },
        "window_size": {
            "type": int,
            "default": 50,
            "description": "Rolling-median window size used for spike detection.",
        },
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.required_variables = list(set(self.variables.keys())) + ["PROFILE_NUMBER"]
        self.qc_outputs = list(
            set(f"{var}_QC" for var in self.required_variables)
            | set(f"{var}_QC" for var in sum(self.also_flag.values(), []))
        )

    def return_qc(self):
        # Subset the data, keeping any existing _QC so already-bad samples can be
        # excluded from the baseline below.
        qc_cols = [f"{v}_QC" for v in self.variables if f"{v}_QC" in self.data]
        self.data = self.data[self.required_variables + qc_cols]

        # Generate the variable-specific flags
        for var, sensitivity in self.variables.items():
            spike_qc = np.full(len(self.data[var]), 0)

            # Apply the checks across individual profiles
            profile_numbers = np.unique(
                self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS")
            )
            for profile_number in self.log_progress(
                profile_numbers,
                desc=f"[{var}]",
                unit="prof",
            ):
                # Subset the data
                profile = self.data.where(
                    self.data["PROFILE_NUMBER"] == profile_number, drop=True
                )

                # Usable = not NaN and not already flagged bad/missing; only these
                # inform the baseline, so prior-bad spikes can't bias detection.
                usable = ~profile[var].isnull()
                if f"{var}_QC" in profile:
                    usable = usable & ~profile[f"{var}_QC"].isin(self.IGNORE_FLAGS)

                var_data = profile[var].where(usable, drop=True)
                if len(var_data) < self.window_size:
                    continue

                # Calculate the residules from the running median of the data
                rolling_median = (
                    var_data.to_pandas()
                    .rolling(window=self.window_size, center=True)
                    .median()
                    .to_numpy()
                )
                residules = var_data.to_numpy() - rolling_median

                # Define the residule threshold
                threshold = np.nanstd(residules) * sensitivity

                # Apply the threshold to residules to get the flags
                spike_flags = np.where((np.abs(residules) > threshold), 4, 1)

                # NaNs are missing (9); excluded/usable points start good (1).
                # Excluded points stay 1 so Apply QC's combinatrix keeps their
                # existing (worse) flag; usable points take their spike result.
                profile_flags = np.where(profile[var].isnull().to_numpy(), 9, 1)
                profile_flags[np.where(usable.to_numpy())] = spike_flags

                # Stitch the QC results back into the QC container
                profile_indices = np.where(
                    self.data["PROFILE_NUMBER"] == profile_number
                )
                spike_qc[profile_indices] = profile_flags

            # Add the flags to the data
            self.data[f"{var}_QC"] = (["N_MEASUREMENTS"], spike_qc)

            # Broadcast the QC found for var into variables specified by "also_flag"
            if extra_vars := self.also_flag.get(var):
                for extra_var in extra_vars:
                    self.data[f"{extra_var}_QC"] = self.data[f"{var}_QC"]

        # Select just the flags
        self.flags = self.data[
            [var_qc for var_qc in self.data.data_vars if "_QC" in var_qc]
        ]

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")

        # If not plots were specified
        if len(self.plot) == 0:
            print(
                f"WARNING: In '{self.qc_name}', diagnostics were called but no variables were specified for plotting."
            )
            return

        # Plot the QC output
        fig, axes = fig_spec.new_fig(nrows=len(self.plot), sharex=True)
        for ax, var in zip(axes[:, 0], self.plot):
            # Check that the user specified var exists in the test set
            if f"{var}_QC" not in self.qc_outputs:
                print(
                    f"WARNING: Cannot plot {var}_QC as it was not included in this test."
                )
                continue

            fig_spec.flag_points(
                ax, self.data["N_MEASUREMENTS"], self.data[var], self.data[f"{var}_QC"]
            )
            ylabel = fig_spec.axis_label(var, self.data[var].attrs.get("units"))
            fig_spec.style_axes(ax, title=f"{var} Spike Test", xlabel="Index", ylabel=ylabel)
            fig_spec.legend(ax, title="Flags")

        fig_spec.finish(fig)
        plt.show(block=True)
