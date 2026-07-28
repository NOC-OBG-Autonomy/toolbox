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

    # Takes no step-specific parameters; variables to interpolate are driven entirely
    # by the framework ``qc_handling_settings`` (flag_filter_settings).
    parameter_schema = {}

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

        for var in self.filter_settings.keys():
            self.data[var][:] = self.df[var].to_numpy()

        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            self.generate_diagnostics()

        # Update the context with the enhanced dataset
        self.context["data"] = self.data
        return self.context

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
