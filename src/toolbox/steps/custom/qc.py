"""Class definition for quality control steps."""

#### Mandatory imports ####
from ..base_step import BaseStep
import utils.diagnostics as diag


class SalinityQCStep(BaseStep):
    def run(self):
        print(f"[SalinityQC] Running QC with threshold {self.parameters['threshold']}")


class TemperatureQCStep(BaseStep):
    def run(self):
        print(
            f"[TemperatureQC] Running QC with threshold {self.parameters['threshold']}"
        )
