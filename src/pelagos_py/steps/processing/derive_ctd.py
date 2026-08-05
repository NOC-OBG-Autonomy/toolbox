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

"""Pipeline step for deriving CTD variables (salinity, density, depth) using the GSW toolbox."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag

#### Custom imports ####
import polars as pl
import numpy as np
import gsw
import matplotlib
import matplotlib.pyplot as plt
from pelagos_py.utils import fig_spec
from pelagos_py.utils.processing_utils import cndc_scale_factor


@register_step
class DeriveCTDVariables(BaseStep, QCHandlingMixin):
    """
    A processing step class for deriving oceanographic variables from CTD data.

    TEOS-10 implementation provided through Gibbs SeaWater (GSW) Oceanographic Toolbox functions.
    This step requires that "TIME", "LATITUDE", "LONGITUDE", "CNDC", "PRES" and "TEMP" are present 
    in the dataset variables.

    Parameters
    ----------
    to_derive : list
        list of variables to derive
        The following variables are supported:
            - "DEPTH"
            - "PRAC_SALINITY" (practical salinity)
            - "ABS_SALINITY" (absolute salinity)
            - "CONS_TEMP" (conservative temperature)
            - "DENSITY

    Examples
    --------
    Example usage in a pipeline configuration:

    .. code-block:: yaml

        steps:
          - name: "Derive CTD"
            parameters:
                to_derive: [
                    DEPTH,
                    PRAC_SALINITY,
                    ABS_SALINITY,
                    CONS_TEMP,
                    DENSITY
                ]
    """

    step_name = "Derive CTD"
    required_variables = ["TIME", "LATITUDE", "LONGITUDE", "CNDC", "PRES", "TEMP"]
    provided_variables = [
        "DEPTH",
        "PRAC_SALINITY",
        "ABS_SALINITY",
        "CONS_TEMP",
        "DENSITY",
    ]

    parameter_schema = {
        "to_derive": {
            "type": list,
            "required": True,
            "options": [
                "DEPTH",
                "PRAC_SALINITY",
                "ABS_SALINITY",
                "CONS_TEMP",
                "DENSITY",
            ],
            "description": "Subset of CTD variables to derive and add to the dataset.",
        },
    }

    def run(self):
        self.log(f"Processing CTD...")

        self.filter_qc()

        # Convert xarray Dataset to Polars DataFrame for efficient numerical processing
        # Extract only the variables needed for GSW calculations
        base_columns = ["TIME", "LATITUDE", "LONGITUDE", "CNDC", "PRES", "TEMP"]
        # Pull in any already-derived variable too, so derivations can be split across
        # two Derive CTD steps (e.g. to correct PRAC_SALINITY in between).
        derived_columns = [
            var
            for var in self.provided_variables
            if var in self.data and var not in base_columns
        ]
        df = pl.from_pandas(
            self.data[base_columns + derived_columns].to_dataframe(),
            nan_to_null=False,
        )

        # gsw wants conductivity in mS/cm; scale from the units attribute (S/m assumed if unset)
        cndc_factor = cndc_scale_factor(self.data["CNDC"].attrs.get("units"))

        # Define GSW (Gibbs SeaWater) function calls for deriving oceanographic variables
        # Each tuple contains: (output_variable_name, gsw_function, [required_input_variables])
        gsw_function_calls = (
            # gsw.z_from_p returns TEOS-10 height (negative down); negate for OG1 positive-down depth
            ("DEPTH", lambda p, lat: -gsw.z_from_p(p, lat), ["PRES", "LATITUDE"]),
            (
                "PRAC_SALINITY",
                lambda c, t, p: gsw.SP_from_C(c * cndc_factor, t, p),
                ["CNDC", "TEMP", "PRES"],
            ),
            (
                "ABS_SALINITY",
                gsw.SA_from_SP,
                ["PRAC_SALINITY", "PRES", "LONGITUDE", "LATITUDE"],
            ),
            ("CONS_TEMP", gsw.CT_from_t, ["ABS_SALINITY", "TEMP", "PRES"]),
            ("DENSITY", gsw.rho, ["ABS_SALINITY", "CONS_TEMP", "PRES"]),
        )

        # Define metadata for each derived variable following CF conventions
        variable_metadata = {
            "DEPTH": {
                "long_name": (
                    "Depth below surface of the water body by unknown instrument "
                    "and correction to zero at sea level using unspecified algorithm."
                ),
                "units": "metres",
                "standard_name": "depth",
                "valid_min": 0.0,
                "valid_max": 10000.0,
                "positive": "down",
                "ancillary_variables": "DEPTH_QC",
                "depth_vocabulary": "https://vocab.nerc.ac.uk/collection/OG1/current/DEPTH/",
            },
            "PRAC_SALINITY": {
                "long_name": "Practical salinity",
                "units": "1",
                "standard_name": "PRAC_SALINITY",
                "valid_min": 2,  # Extremely fresh water
                "valid_max": 42,  # Hypersaline conditions
            },
            "ABS_SALINITY": {
                "long_name": "Absolute salinity",
                "units": "g/kg",
                "standard_name": "ABS_SALINITY",
                "valid_min": 0,  # Pure water
                "valid_max": 1000,  # Pure salt (theoretical maximum)
            },
            "CONS_TEMP": {
                "long_name": "Conservative temperature",
                "units": "degC",
                "standard_name": "CONS_TEMP",
                "valid_min": -2,  # Freezing point of seawater
                "valid_max": 102,  # Boiling point of seawater
            },
            "DENSITY": {
                "long_name": "Density",
                "units": "kg/m3",
                "standard_name": "DENSITY",
                "valid_min": 900,  # Warm, low salinity surface water
                "valid_max": 1100,  # Cold, high salinity bottom water
            },
        }

        # Process each GSW function call to derive new variables
        for var_name, func, args in gsw_function_calls:
            if var_name not in self.to_derive:
                continue

            self.log(f"Deriving {var_name}...")

            # Validate that all required inputs exist for this specific calculation
            # (e.g. an intermediate like PRAC_SALINITY may not have been derived)
            missing_args = [arg for arg in args if arg not in df.columns]
            if missing_args:
                self.log(
                    f"Warning: Missing required variables {missing_args} for {var_name}. Skipping."
                )
                continue

            # Apply the GSW function to pure numpy arrays
            input_arrays = [df[arg].to_numpy() for arg in args]
            derived_values = func(*input_arrays)

            df = df.with_columns(pl.Series(var_name, derived_values))

            # Add the derived variable to the xarray Dataset with CF-compliant metadata
            self.data[var_name] = (("N_MEASUREMENTS",), derived_values)
            self.data[var_name].attrs = variable_metadata[var_name]

            # Safely generate QC by only passing source columns that actually exist
            source_qcs = [f"{arg}_QC" for arg in args if f"{arg}_QC" in self.data]
            if source_qcs:
                self.generate_qc({f"{var_name}_QC": source_qcs})

        # Show diagnostic plots if diagnostics are enabled
        if self.diagnostics:
            self.plot_diagnostics()

        self.reconstruct_data()
        self.update_qc()

        # Update the context with the enhanced dataset
        self.context["data"] = self.data
        return self.context

    def plot_diagnostics(self):
        if "TIME" not in self.data:
            return

        # Combine physical inputs and derived outputs, filtering for what actually exists
        target_variables = ["PRES", "CNDC", "TEMP"] + self.provided_variables
        plot_vars = [var for var in target_variables if var in self.data]

        if not plot_vars:
            return

        matplotlib.use("tkagg")
        fig, axes = fig_spec.new_fig(nrows=len(plot_vars), sharex=True)
        time_data = self.data["TIME"].values

        for i, (ax, var_name) in enumerate(zip(axes[:, 0], plot_vars)):
            data_vals = self.data[var_name].values

            # Colour by QC flag where a QC column exists, else a single series.
            qc_col = f"{var_name}_QC"
            if qc_col in self.data:
                fig_spec.flag_points(ax, time_data, data_vals, self.data[qc_col].values)
                fig_spec.legend(ax, title="Flags")
            else:
                fig_spec.points(ax, time_data, data_vals, color=fig_spec.CATEGORY[0])

            fig_spec.date_axis(ax, which="x")
            ylabel = fig_spec.axis_label(var_name, self.data[var_name].attrs.get("units"))
            xlabel = "Time" if i == len(plot_vars) - 1 else None
            fig_spec.style_axes(ax, xlabel=xlabel, ylabel=ylabel)

            # Invert y-axis for pressure/depth so the ocean surface is at the top of the plot
            if var_name in ("PRES", "DEPTH"):
                ax.invert_yaxis()

        fig_spec.finish(fig, suptitle=f"{self.step_name} Diagnostics")
        plt.show(block=True)
