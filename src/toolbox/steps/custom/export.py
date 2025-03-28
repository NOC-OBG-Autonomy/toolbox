"""Class definition for exporting data steps."""

#### Mandatory imports ####
from ..base_step import BaseStep
import utils.diagnostics as diag


class ExportStep(BaseStep):
    step_name = "Data Export"

    def run(self):
        print(
            f"[Export] Exporting data in {self.parameters['export_format']} format to {self.parameters['output_path']}"
        )
