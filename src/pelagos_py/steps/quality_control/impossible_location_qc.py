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

"""QC test to identify impossible locations in LATITUDE and LONGITUDE variables."""

#### Mandatory imports ####
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import polars as pl
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from pelagos_py.utils import fig_spec


@register_qc
class impossible_location_qc(BaseQC):
    """
    Target Variable: LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: LATITUDE, LONGITUDE
    Checks that the latitude and longitude are valid.
    """

    qc_name = "impossible location qc"
    parameter_schema = {}
    required_variables = ["LATITUDE", "LONGITUDE"]
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(), nan_to_null=False
        )

        # Check LAT/LONG exist within expected bounds
        # TODO: Add optional bounds via parameters (such as Southern Hemisphere, for example)
        for label, bounds in zip(["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]):
            self.df = self.df.with_columns(
                pl.when(pl.col(label).is_nan())
                .then(9)
                .when((pl.col(label) > bounds[0]) & (pl.col(label) < bounds[1]))
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
        df = self.df.with_row_index()
        fig, axes = fig_spec.new_fig(nrows=2, sharex=True)

        for ax, var, bounds in zip(
            axes[:, 0], ["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]
        ):
            fig_spec.flag_points(ax, df["index"], df[var], df[f"{var}_QC"])
            ylabel = fig_spec.axis_label(var, self.data[var].attrs.get("units"))
            fig_spec.style_axes(ax, xlabel="Index", ylabel=ylabel)
            fig_spec.legend(ax, title="Flags")
            for bound in bounds:
                ax.axhline(bound, ls="--", c="k")

        fig_spec.finish(fig, suptitle="Impossible Location Test")
        plt.show(block=True)
