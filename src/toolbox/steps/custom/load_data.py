"""Class definition for loading data steps."""

#### Mandatory imports ####
from ..base_step import BaseStep
import utils.diagnostics as diag

#### Custom imports ####
import xarray as xr


class LoadOG1(BaseStep):
    step_name = "Load OG1"

    def run(self):
        source = self.parameters["file_path"]
        print(f"[LoadData] Loading {source} OG1")
        # load data from xarray
        self.data = xr.open_dataset(source)

        # Generate diagnostics if enabled
        if self.diagnostics:
            self.generate_diagnostics()

        # Continue with the rest of the step logic...

    def generate_diagnostics(self):
        print(f"[LoadData] Generating diagnostics...")
        # Print summary stats
        diag.generate_summary_stats(self.data)
