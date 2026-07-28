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

"""QC test to identify impossible speeds in glider data."""

#### Mandatory imports ####
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import matplotlib.pyplot as plt
import polars as pl
import xarray as xr
import numpy as np
import matplotlib
from pelagos_py.utils import fig_spec


@register_qc
class impossible_speed_qc(BaseQC):
    """
    Target Variable: TIME, LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: TIME, LATITUDE, LONGITUDE
    Checks that the the gliders horizontal speed stays below 3m/s
    """

    qc_name = "impossible speed qc"
    parameter_schema = {}
    required_variables = ["TIME", "LATITUDE", "LONGITUDE"]
    qc_outputs = ["TIME_QC", "LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(), nan_to_null=False
        )

        self.df = self.df.with_columns(
            (pl.col("TIME").diff().cast(pl.Float64) * 1e-9).alias("dt")
        )
        for label in ["LATITUDE", "LONGITUDE"]:
            self.df = self.df.with_columns(
                pl.col(label)
                .replace([np.inf, -np.inf, np.nan], None)
                .interpolate_by("TIME")
                .diff()
                .alias(f"delta_{label}")
            )
            self.df = self.df.with_columns(
                (pl.col(f"delta_{label}") / pl.col("dt")).alias(f"{label}_speed")
            )
        # Define absolute speed
        self.df = self.df.with_columns(
            (
                (pl.col("LATITUDE_speed") ** 2 + pl.col("LONGITUDE_speed") ** 2) ** 0.5
            ).alias("absolute_speed")
        )

        # TODO: Does this need a flag for potentially bad data for cases where speed is inf?
        self.df = self.df.with_columns(
            (
                (pl.col("absolute_speed") < 3)  #  Speed threshold
                & pl.col("absolute_speed").is_not_null()
                & pl.col("absolute_speed").is_finite()
            ).alias("speed_is_valid")
        )

        for label in ["LATITUDE", "LONGITUDE", "TIME"]:
            self.df = self.df.with_columns(
                pl.when(pl.col("speed_is_valid"))
                .then(1)
                .otherwise(4)
                .alias(f"{label}_QC")
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
        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]
        fig_spec.flag_points(
            ax, self.df["TIME"], self.df["absolute_speed"], self.df["LATITUDE_QC"]
        )
        fig_spec.date_axis(ax, which="x")
        ax.set_ylim(0, 4)
        ax.axhline(3, ls="--", c="k")
        fig_spec.style_axes(
            ax, xlabel="Time", ylabel="Absolute Horizontal Speed (m/s)"
        )
        fig_spec.legend(ax, title="Flags")
        fig_spec.finish(fig, suptitle="Impossible Speed Test")
        plt.show(block=True)
