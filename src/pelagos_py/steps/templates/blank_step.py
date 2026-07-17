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

"""Example step template. Copy and populate this example, which will inherit additional functionality from BaseStep."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag

#### Custom imports ####


# @register_step  # Uncomment when implementing
class BlankStep(BaseStep, QCHandlingMixin):

    step_name = "Blank Step"
    required_variables = []
    provided_variables = []

    # Declare every parameter here (see pelagos_py.utils.parameter_spec). Omit
    # "default" (or set "required": True) for parameters with no sensible preset.
    # Resolved values are available as attributes, e.g. self.my_threshold.
    parameter_schema = {
        # "my_threshold": {
        #     "type": float,
        #     "default": 1.0,
        #     "description": "What this controls.",
        # },
    }

    def run(self):
        self.filter_qc()

        # EXAMPLE: self.data["C"] = self.data["A"] * self.data["B"]

        # If the step DERIVES anything from more than one sample (a rolling window,
        # a fit, a mean/median/max, a peak), flagged samples must not feed it. Use
        # self.calculation_mask() to exclude them from the derivation while still
        # writing the result to every sample. A pure per-point formula (as above)
        # needs none of this - generate_qc already propagates its inputs' flags.
        # EXAMPLE: usable = self.calculation_mask(["A", "B"])
        #          reference = np.nanmedian(self.data["A"].values[usable])
        #          self.data["C"] = self.data["A"] - reference

        self.reconstruct_data()
        self.update_qc()

        # If a new variable was added, use self.generate_qc()
        # EXAMPLE: self.generate_qc({"C_QC": ["A_QC", "B_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass
