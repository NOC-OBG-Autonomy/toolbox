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

"""QC test to identify impossible dates in TIME variable."""

#### Mandatory imports ####
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import polars as pl
import xarray as xr
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from pelagos_py.utils import fig_spec


@register_qc
class impossible_date_qc(BaseQC):
    """
    Target Variable: TIME
    Flag Number: 4 (bad data)
    Variables Flagged: TIME
    Checks that the datetime of each point is valid.
    """

    qc_name = "impossible date qc"
    parameter_schema = {}
    required_variables = ["TIME"]
    qc_outputs = ["TIME_QC"]

    def return_qc(self):
        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(), nan_to_null=False
        )

        # Check if any of the datetime stamps fall outside 1985 and the current datetime
        # TODO: Add optional bounds via parameters (such as known deployment dates, for example)
        self.df = self.df.with_columns(
            pl.when(pl.col("TIME").is_null())
            .then(9)
            .when(
                (
                    (pl.col("TIME") > datetime(1985, 1, 1))
                    & (pl.col("TIME") < datetime.now())
                )
            )
            .then(1)
            .otherwise(4)
            .alias("TIME_QC")
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
        fig_spec.flag_points(ax, df["index"], df["TIME"], df["TIME_QC"])
        fig_spec.date_axis(ax, which="y")
        fig_spec.style_axes(ax, xlabel="Index", ylabel="TIME")
        fig_spec.legend(ax, title="Flags")
        fig_spec.finish(fig, suptitle="Impossible Date Test")
        plt.show(block=True)
