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

"""QC test to flag entire glider profiles based on number of bad flags."""

#### Mandatory imports ####
import numpy as np
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib
from pelagos_py.utils import fig_spec


@register_qc
class flag_full_profile(BaseQC):
    """
    Flag an entire profile once it accumulates too many bad measurements.

    | **Target variable:** user-defined (dynamic)
    | **Variables flagged:** the same user-defined variable(s)
    | **Flag applied:** 4 (bad)

    For each variable in ``check_vars``, the number of bad (4) flags in its ``_QC``
    column is counted per profile (grouped by ``PROFILE_NUMBER``, as produced by
    :doc:`Find Profiles <../processing/find_profiles/index>`). If a profile's count
    reaches the variable's threshold, **every** measurement of that variable in the
    profile is set to bad (4) — the rationale being that a profile riddled with bad
    points is untrustworthy as a whole.

    Each variable is evaluated independently against its own threshold, and only the
    listed variables' ``_QC`` columns are modified.

    Parameters
    ----------
    check_vars : dict
        Mapping of variable name to an integer threshold, e.g.
        ``{"PRES": 10, "CHLA": 20}``. A profile with at least that many bad (4) flags
        in ``<variable>_QC`` has all of that variable's points flagged bad. Required;
        each listed variable and its ``_QC`` column must be present in the dataset.

    Examples
    --------
    Flag any profile that contains 10 or more bad pressure points (and, separately,
    20 or more bad chlorophyll points):

    .. code-block:: yaml

        - name: "Apply QC"
          parameters:
            qc_settings:
              flag full profile:
                check_vars:
                  PRES: 10
                  CHLA: 20
          diagnostics: true  # plot each variable vs index, coloured by flag
    """

    qc_name = "flag full profile"
    required_variables = ["PROFILE_NUMBER"]
    provided_variables = []

    # Specify if test target variable is user-defined (if True, __init__ has to be redefined)
    dynamic = True

    parameter_schema = {
        "check_vars": {
            "type": dict,
            "required": True,
            "description": (
                "Mapping of variable -> bad-flag-count threshold, e.g. {'PRES': 10}. "
                "A profile reaching the threshold has all that variable's points flagged bad."
            ),
        },
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        # Variables required/flagged are derived from the configured check_vars.
        self.required_variables = (
            list(self.check_vars.keys())
            + [f"{k}_QC" for k in self.check_vars.keys()]
            + ["PROFILE_NUMBER"]
        )

    def return_qc(self):
        # TODO: Add support for flagging if threshold is a mix of 3 (questionable) and 4 (definitely bad) flags
        # Subset the data
        self.data = self.data[self.required_variables]

        for var, threshold in self.check_vars.items():
            flag_counts = (
                (self.data[f"{var}_QC"] == 4).groupby(self.data["PROFILE_NUMBER"]).sum()
            )  # Default to flag 4 (definitely bad)
            bad_profiles = flag_counts.where(flag_counts >= threshold, drop=True)[
                "PROFILE_NUMBER"
            ]
            self.data[f"{var}_QC"] = xr.where(
                self.data[f"PROFILE_NUMBER"].isin(bad_profiles),
                4,
                self.data[f"{var}_QC"],
            )

        # Select just the flags
        self.flags = self.data[
            [var_qc for var_qc in self.data.data_vars if "_QC" in var_qc]
        ]

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")

        # Plot the QC output
        n_plots = len(self.check_vars.keys())
        fig, axes = fig_spec.new_fig(nrows=n_plots, sharex=True)
        for ax, var in zip(axes[:, 0], self.check_vars.keys()):
            fig_spec.flag_points(
                ax, self.data["N_MEASUREMENTS"], self.data[var], self.data[f"{var}_QC"]
            )
            ylabel = fig_spec.axis_label(var, self.data[var].attrs.get("units"))
            fig_spec.style_axes(ax, title=f"{var} Flag Full Profile", xlabel="Index", ylabel=ylabel)
            fig_spec.legend(ax, title="Flags")

        fig_spec.finish(fig)
        plt.show(block=True)
