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

"""QC tests for assessing validity of a glider profile, based on different definitions of successful data."""

#### Mandatory imports ####
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import matplotlib.pyplot as plt
import polars as pl
import xarray as xr
import matplotlib
from pelagos_py.utils import fig_spec


@register_qc
class valid_profile_qc(BaseQC):
    """
    Flag whole profiles that are too short or never reach a target depth range.

    | **Target variable:** ``PROFILE_NUMBER``
    | **Variables flagged:** ``PROFILE_NUMBER``
    | **Flags applied:** 1 (good), 3 (probably bad), 4 (bad), 9 (missing)

    Each profile (a group of measurements sharing a ``PROFILE_NUMBER``, as produced
    by :doc:`Find Profiles <../processing/find_profiles/index>`) is assessed as a
    whole and every row in that profile receives the same ``PROFILE_NUMBER_QC``:

    - **9 (missing)** — the row has no profile (``PROFILE_NUMBER`` is NaN, e.g.
      surfacing rows or data gaps).
    - **4 (bad)** — the profile contains fewer than ``profile_length`` measurements.
    - **3 (probably bad)** — the profile is long enough but has no measurement whose
      ``DEPTH`` falls inside ``depth_range``.
    - **1 (good)** — the profile passes both checks.

    Only ``PROFILE_NUMBER_QC`` is written; the underlying data is never modified.

    Parameters
    ----------
    profile_length : int, optional
        Minimum number of measurements a profile must contain to be kept. Profiles
        shorter than this are flagged bad (4). Default ``100``.
    depth_range : tuple of float, optional
        ``(min, max)`` depth window (in the same units/sign convention as ``DEPTH``,
        i.e. positive downward) that a profile must reach into. A profile with no data
        inside this window is flagged probably bad (3). Default ``(0, 1000)``.

    Examples
    --------
    The check works with its defaults, so the minimal configuration sets no
    parameters:

    .. code-block:: yaml

        - name: "Apply QC"
          parameters:
            qc_settings:
              valid profile qc: {}

    Both parameters may be tuned — here profiles must be at least 50 points long and
    contain data somewhere between 1000 m depth and the surface:

    .. code-block:: yaml

        - name: "Apply QC"
          parameters:
            qc_settings:
              valid profile qc:
                profile_length: 50
                depth_range: [0, 1000]
          diagnostics: true  # plot DEPTH vs index, coloured by the resulting flag
    """

    qc_name = "valid profile qc"
    parameter_schema = {
        "profile_length": {
            "type": int,
            "default": 100,
            "description": "Minimum number of measurements a profile must contain to be kept.",
        },
        "depth_range": {
            "type": list,
            "default": (0, 1000),
            "description": "(min, max) depth window a profile must reach into.",
        },
    }
    required_variables = ["PROFILE_NUMBER", "DEPTH"]
    qc_outputs = ["PROFILE_NUMBER_QC"]

    def return_qc(self):
        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(), nan_to_null=False
        )

        # Check profiles are of a given length
        profile_lengths = self.df.group_by("PROFILE_NUMBER").agg(
            pl.len().alias("count")
        )
        self.df = self.df.join(profile_lengths, on="PROFILE_NUMBER", how="left")

        # Find profiles that have no data between the sepcified depth ranges
        profile_ranges = self.df.group_by("PROFILE_NUMBER").agg(
            (pl.col("DEPTH").is_between(*self.depth_range).any()).alias(
                "in_depth_range"
            )
        )
        self.df = self.df.join(profile_ranges, on="PROFILE_NUMBER", how="left")

        self.df = self.df.with_columns(
            pl.when(pl.col("PROFILE_NUMBER").is_nan())
            .then(9)
            .when(pl.col("count") < self.profile_length)
            .then(4)
            .when(pl.col("in_depth_range").not_())
            .then(3)
            .otherwise(1)
            .alias("PROFILE_NUMBER_QC")
        )

        # Convert back to xarray
        flags = self.df.select(pl.col("^.*_QC$"))
        self.flags = xr.Dataset(
            data_vars={
                col: ("N_MEASUREMENTS", flags[col].to_numpy()) for col in flags.columns
            },
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        df = self.df.with_row_index()
        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]
        fig_spec.flag_points(ax, df["index"], df["DEPTH"], df["PROFILE_NUMBER_QC"])
        fig_spec.style_axes(ax, xlabel="Index", ylabel="Pressure")
        fig_spec.legend(ax, title="Flags")
        fig_spec.finish(fig, suptitle="Valid Profile Test")
        plt.show(block=True)
