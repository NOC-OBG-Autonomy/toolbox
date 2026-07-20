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

# Diagnostic plot settings
PLOT_SIZE = (10, 8)  # Widened slightly to accommodate the external legend
PLOT_COLOURS = ["#00b894", "#0984e3", "#d63031", "#fdcb6e", "#6c5ce7", "#e84393", "#00cec9", "#e17055"]
FLAGGED_COLOUR = "#b2bec3"  # Grey for flagged/bad data
MARKER_SIZE = 1
PLOT_ALPHA = 0.6


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
        # A derived variable an earlier Derive CTD step already added is a valid
        # input here, so pull it in too - otherwise splitting the derivations
        # across two steps (e.g. to correct PRAC_SALINITY in between) would make
        # the second step's inputs look missing.
        derived_columns = [
            var
            for var in self.provided_variables
            if var in self.data and var not in base_columns
        ]
        df = pl.from_pandas(
            self.data[base_columns + derived_columns].to_dataframe(),
            nan_to_null=False,
        )

        # Define GSW (Gibbs SeaWater) function calls for deriving oceanographic variables
        # Each tuple contains: (output_variable_name, gsw_function, [required_input_variables])
        gsw_function_calls = (
            # gsw.z_from_p returns TEOS-10 height (negative down); negate for OG1 positive-down depth
            ("DEPTH", lambda p, lat: -gsw.z_from_p(p, lat), ["PRES", "LATITUDE"]),
            ("PRAC_SALINITY", gsw.SP_from_C, ["CNDC", "TEMP", "PRES"]),
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
        n_vars = len(plot_vars)

        fig, axes = plt.subplots(n_vars, 1, sharex=True, figsize=PLOT_SIZE, dpi=150)

        if n_vars == 1:
            axes = [axes]

        time_data = self.data["TIME"].values

        for i, var_name in enumerate(plot_vars):
            ax = axes[i]
            colour = PLOT_COLOURS[i % len(PLOT_COLOURS)]
            data_vals = self.data[var_name].values

            # Extract units and format cleanly (ignore "1" or missing units)
            units = str(self.data[var_name].attrs.get("units", "")).strip()
            if units in ["1", "unitless", "unknown", "None", ""]:
                unit_str = ""
            else:
                unit_str = f"\n[{units}]"

            # Determine QC status if the QC column exists
            qc_col = f"{var_name}_QC"
            if qc_col in self.data:
                qc_vals = self.data[qc_col].values
                # Treat 0 (No QC), 1, 2, 5, 8 as "Good" points
                good_mask = np.isin(qc_vals, [0, 1, 2, 5, 8])
                bad_mask = ~good_mask & ~np.isnan(data_vals)
                good_plot_mask = good_mask & ~np.isnan(data_vals)

                # Plot bad data first so it sits beneath good data
                if np.any(bad_mask):
                    ax.plot(
                        time_data[bad_mask],
                        data_vals[bad_mask],
                        ls="",
                        marker="o",
                        markersize=MARKER_SIZE,
                        alpha=PLOT_ALPHA,
                        c=FLAGGED_COLOUR,
                        zorder=1,
                    )

                # Plot good data on top
                if np.any(good_plot_mask):
                    ax.plot(
                        time_data[good_plot_mask],
                        data_vals[good_plot_mask],
                        ls="",
                        marker="o",
                        markersize=MARKER_SIZE,
                        alpha=PLOT_ALPHA,
                        c=colour,
                        zorder=2,
                    )

                # We calculate stats only on the good data for a cleaner representation
                stats_data = data_vals[good_plot_mask]
            else:
                # Fallback if no QC column exists
                ax.plot(
                    time_data,
                    data_vals,
                    ls="",
                    marker="o",
                    markersize=MARKER_SIZE,
                    alpha=PLOT_ALPHA,
                    c=colour,
                    zorder=2,
                )
                stats_data = data_vals[~np.isnan(data_vals)]

            # Calculate robust statistics
            if len(stats_data) > 0:
                v_min = np.nanmin(stats_data)
                v_max = np.nanmax(stats_data)
                v_mean = np.nanmean(stats_data)
                v_std = np.nanstd(stats_data)

                # Add formatted statistical legend outside the plot area
                stat_text = f"Min: {v_min:.3f}\nMax: {v_max:.3f}\nMean: {v_mean:.3f}\nStd: {v_std:.3f}"
                ax.plot([], [], ls="", label=stat_text)
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    fontsize=6,
                    framealpha=0.9,
                    fancybox=True,
                )

            ax.set_ylabel(f"{var_name}{unit_str}", fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=7)

            # Invert y-axis for pressure/depth so the ocean surface is at the top of the plot
            if var_name in ("PRES", "DEPTH"):
                ax.invert_yaxis()

        axes[-1].set_xlabel("Time", fontsize=8)
        fig.suptitle(f"{self.step_name} Diagnostics", fontsize=10, fontweight="bold")

        # Adjust layout to leave room on the right for the external legends
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        plt.show(block=True)
