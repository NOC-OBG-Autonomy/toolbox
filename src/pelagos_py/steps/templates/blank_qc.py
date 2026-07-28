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

"""Example QC test template, using parts of impossible_date_test as a skeleton."""

#### Mandatory imports ####
from pelagos_py.steps.base_qc import BaseQC

#### Custom imports ####
# any additional imports required for the test go here
import matplotlib
import matplotlib.pyplot as plt
from pelagos_py.utils import fig_spec  # shared diagnostic-plot style


# @register_qc  # Uncomment when implementing
class blank_qc(BaseQC):
    """
    Example Docstring:
    Target Variable: TIME
    Flag Number: 4 (bad data)
    Variables Flagged: TIME
    Checks that the datetime of each point is valid.
    """

    qc_name = ""
    parameter_schema = {}
    required_variables = []
    provided_variables = []
    qc_outputs = []

    def return_qc(self):
        # Access the data with self.data
        # self.flags should be an xarray Dataset with data_vars that hold the "{variable}_QC" columns produced by the test
        return self.flags

    def plot_diagnostics(self):
        # Style every diagnostic with the shared fig_spec helpers so plots stay
        # uniform and dashboard-interactive. Example: one QC-flag panel.
        matplotlib.use("tkagg")
        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]
        # fig_spec.flag_points(ax, self.data["N_MEASUREMENTS"], self.data[var],
        #                      self.data[f"{var}_QC"])
        fig_spec.style_axes(ax, xlabel="Index", ylabel="")
        fig_spec.legend(ax, title="Flags")
        fig_spec.finish(fig, suptitle=self.qc_name)
        plt.show(block=True)
