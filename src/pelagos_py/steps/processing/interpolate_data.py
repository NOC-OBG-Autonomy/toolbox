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

"""Class definition for deriving CTD variables."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag

#### Custom imports ####
import polars as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pelagos_py.utils import fig_spec


@register_step
class InterpolateVariables(BaseStep, QCHandlingMixin):
    """
    A processing step for interpolating data.

    This class processes data to interpolate missing values and fill gaps in
    variables using time-based interpolation. It supports quality control
    handling and optional diagnostic visualization.

    Inherits from BaseStep and processes data stored in the context dictionary.

    Parameters
    ----------
    name : str
        Name identifier for this step instance.
    parameters : dict, optional
        Configuration parameters for the interpolation step.
    diagnostics : bool, optional
        Whether to generate diagnostic visualizations. Default is False.
    context : dict, optional
        Processing context dictionary.

    Attributes
    ----------
    step_name : str
        Identifier for this processing step. Set to "Interpolate Data".

    Examples
    --------
    Example config usage::

        - name: "Interpolate Data"
          parameters:
            max_interp_time: 5.0
            qc_handling_settings: {
              flag_filter_settings: {
                "PRES": [3, 4, 9],
                "LATITUDE": [3, 4, 9],
                "LONGITUDE": [3, 4, 9]
              },
              reconstruction_behaviour: "replace",
              flag_mapping: { 3: 8, 4: 8, 9: 8 }
            }
          diagnostics: false
    """

    step_name = "Interpolate Data"
    required_variables = ["TIME"]
    provided_variables = []
    uses_data_subset = True

    # Variables to interpolate are driven entirely by the framework
    # ``qc_handling_settings`` (flag_filter_settings).
    parameter_schema = {
        "max_interp_time": {
            "type": [float, bool, str],
            "default": 5.0,
            "description": (
                "Maximum time (minutes) from the nearest surrounding non-interpolated "
                "point that a gap will be filled across, so interpolation only fills "
                "small gaps rather than whole missing profiles. 0/False/off disables "
                "the limit (fills gaps of any size)."
            ),
        },
    }

    def run(self):
        """
        Execute the interpolation workflow.

        This method performs the following steps:

        1. Filters data based on quality control flags
        2. Converts xarray data to a Polars DataFrame
        3. Interpolates missing values using time as the reference dimension
        4. QC and data reconstruction based on user specification
        5. Updates QC flags for interpolated values
        6. Generates diagnostic plots if enabled

        Returns
        -------
        dict
            The updated context dictionary containing the interpolated dataset
            under the "data" key.
        """
        self.log(f"Interpolating variables...")

        self.filter_qc()

        max_interp_seconds = self._max_interp_seconds()

        # Convert to polars dataframe
        self.df = pl.from_pandas(
            self.data[list(self.filter_settings.keys() | {"TIME"})].to_dataframe(),
            nan_to_null=False,
        )
        self.unprocessed_df = (
            self.df.clone()
        )  # Making a copy for plotting change in diagnostics

        # Interpolate
        self.df = self.df.with_columns(
            pl.col(var)
            .replace({np.nan: None})
            .interpolate_by("TIME")
            .replace({None: np.nan})
            for var in self.filter_settings.keys()
        )

        time = self.df["TIME"].to_numpy()
        for var in self.filter_settings.keys():
            interpolated = self.df[var].to_numpy().copy()
            if max_interp_seconds:
                was_nan = self.unprocessed_df[var].is_nan().to_numpy()
                self._limit_gap_fill(time, interpolated, was_nan, max_interp_seconds)
            self.df = self.df.with_columns(pl.Series(var, interpolated))
            self.data[var][:] = interpolated

        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            self.generate_diagnostics()

        # Update the context with the enhanced dataset
        self.context["data"].update(self.data)
        return self.context

    def _max_interp_seconds(self):
        """Resolve ``max_interp_time`` (minutes) to seconds, or None if disabled."""
        value = self.max_interp_time
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("off", "false", "no", "0", "0.0"):
                return None
            value = float(text)
        if not value:
            return None
        return float(value) * 60

    @staticmethod
    def _limit_gap_fill(time, interpolated, was_nan, max_seconds):
        """Revert interior interpolated points back to NaN if their surrounding
        gap (between the nearest non-interpolated points either side) exceeds
        max_seconds. Leading/trailing NaNs are untouched (interpolate_by never
        fills them)."""
        valid_idx = np.flatnonzero(~was_nan)
        if valid_idx.size == 0:
            return

        prev_valid = np.full(was_nan.shape, -1)
        prev_valid[valid_idx] = valid_idx
        np.maximum.accumulate(prev_valid, out=prev_valid)

        next_valid = np.full(was_nan.shape, was_nan.size)
        next_valid[valid_idx] = valid_idx
        next_valid[::-1] = np.minimum.accumulate(next_valid[::-1])

        interior = was_nan & (prev_valid >= 0) & (next_valid < was_nan.size)
        gap_seconds = (time[next_valid[interior]] - time[prev_valid[interior]]) / np.timedelta64(1, "s")
        too_far = np.flatnonzero(interior)[gap_seconds > max_seconds]
        interpolated[too_far] = np.nan

    def generate_diagnostics(self):
        """
        Generate diagnostic plots comparing original and interpolated data.

        Creates a side-by-side comparison visualization showing the first
        variable in filter_settings before and after interpolation.

        This method uses the Tkinter backend for interactive display.

        Returns
        -------
        None
        """

        matplotlib.use("tkagg")
        fig, axes = fig_spec.new_fig(nrows=2, sharex=True, sharey=True)

        plot_var = list(self.filter_settings.keys())[0]
        titles = ["Original", "Interpolated"]
        for ax, data, title in zip(axes[:, 0], [self.unprocessed_df, self.df], titles):
            ax.plot(data[plot_var], color=fig_spec.CATEGORY[1])
            fig_spec.style_axes(ax, title=title, ylabel=plot_var)

        fig_spec.finish(fig, suptitle=f"Interpolation: {plot_var}")
        plt.show(block=True)
