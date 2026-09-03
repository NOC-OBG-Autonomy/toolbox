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

"""QC test that identifies glider positions not located on land and flags accordingly."""

#### Mandatory imports ####
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
from geodatasets import get_path
import matplotlib.pyplot as plt
import shapely as sh
import polars as pl
import xarray as xr
import matplotlib
import geopandas
from pelagos_py.utils import fig_spec


@register_qc
class position_on_land_qc(BaseQC):
    """
    Target Variable: LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: LATITUDE, LONGITUDE
    Checks that the measurement location is not on land.
    """

    qc_name = "position on land qc"
    parameter_schema = {}
    required_variables = ["LATITUDE", "LONGITUDE"]
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(), nan_to_null=False
        )

        # Concat the polygons into a MultiPolygon object
        self.world = geopandas.read_file(get_path("naturalearth.land"))
        land_polygons = sh.ops.unary_union(self.world.geometry)

        # Check if lat, long coords fall within the area of the land polygons
        self.df = self.df.with_columns(
            pl.when(pl.col("LONGITUDE").is_nan() | pl.col("LATITUDE").is_nan())
            .then(9)
            .otherwise(
                pl.struct("LONGITUDE", "LATITUDE")
                .map_batches(
                    lambda x: sh.contains_xy(
                        land_polygons,
                        x.struct.field("LONGITUDE").to_numpy(),
                        x.struct.field("LATITUDE").to_numpy(),
                    )
                    * 4
                )
                .replace({0: 1})
            )
            .alias("LONGITUDE_QC")
        )
        # Add the flags to LATITUDE as well.
        self.df = self.df.with_columns(pl.col("LONGITUDE_QC").alias("LATITUDE_QC"))

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

        # Plot land boundaries
        self.world.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.3)

        fig_spec.flag_points(
            ax, self.df["LONGITUDE"], self.df["LATITUDE"], self.df["LATITUDE_QC"]
        )
        fig_spec.style_axes(
            ax,
            xlabel=fig_spec.axis_label("LONGITUDE", self.data["LONGITUDE"].attrs.get("units")),
            ylabel=fig_spec.axis_label("LATITUDE", self.data["LATITUDE"].attrs.get("units")),
        )
        fig_spec.legend(ax, title="Flags")
        fig_spec.finish(fig, suptitle="Position On Land Test")
        plt.show(block=True)
