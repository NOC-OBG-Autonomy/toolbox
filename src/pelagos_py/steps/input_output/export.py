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

"""Class definition for exporting data steps."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
import pelagos_py.utils.diagnostics as diag
import json


@register_step
class ExportStep(BaseStep):
    """
    Exports the the data output by the previous step. 

    Parameters
    ----------
    output_path : str
        Path and file name to output to. The file extension should be included.
    export_format : str
        Either "netcdf", "csv", "hdf5" or "parquet". Defaults to "netcdf".
    compression : int
        zlib compression level for netcdf/hdf5 output: 0 turns compression off,
        1 (fastest) to 9 (smallest). Defaults to 4. Ignored for csv/parquet.

    Examples
    --------
    Example usage in a pipeline configuration:

    .. code-block:: yaml

        steps:
          - name: Data Export
            parameters:
                output_path: "save/my/data/here.nc"
                export_format: "netcdf"
                compression: 4
    """

    step_name = "Data Export"

    parameter_schema = {
        "export_format": {
            "type": str,
            "default": "netcdf",
            "options": ["csv", "netcdf", "hdf5", "parquet"],
            "description": "Export format compatible with xarray (csv/netcdf/hdf5/parquet).",
        },
        "output_path": {
            "type": str,
            "required": True,
            "description": "Path the dataset will be exported to.",
        },
        "compression": {
            "type": int,
            "default": 4,
            "min": 0,
            "max": 9,
            "description": "zlib level for netcdf/hdf5 (0 = off, 1-9). Ignored for csv/parquet.",
        },
    }

    def run(self):
        self.log(
            f"Exporting data in {self.parameters['export_format']} format to {self.parameters['output_path']}"
        )

        # Check if the data is in the context
        self.check_data()
        data = self.context["data"]
        # Add exiting notes on QC history if available TODO: Move earlier to individual QC steps on each data variable attribute
        if "qc_history" in self.context:
            self.log(f"QC history found in context.")
            data.attrs["delayed_qc_history"] = json.dumps(self.context["qc_history"])

        export_format = self.parameters["export_format"]
        output_path = self.parameters["output_path"]
        compression = self.parameters["compression"]

        # Validate the export format  # TODO: have all of these file types been tested?
        if export_format not in ["csv", "netcdf", "hdf5", "parquet"]:
            raise ValueError(
                f"Unsupported export format: {export_format}. Supported formats are: csv, netcdf, hdf5, parquet."
            )
        if not output_path:
            raise ValueError("Output path must be specified for data export.")
        # Ensure the output path is a string
        if not isinstance(output_path, str):
            raise ValueError("Output path must be a string.")

        # Build zlib encoding for netcdf/hdf5 (compression=0 disables it)
        if compression:
            if compression > 9 or compression < 0:
                raise ValueError(
                    f"Unsupported compression level: {compression}. Please specify compression from 0-9."
                )
            encoding = {
                var: {"zlib": True, "complevel": compression} for var in data.data_vars
            }
        else:
            encoding = None

        # Export data based on the specified format
        if export_format == "csv":
            data.to_dataframe().to_csv(output_path, index=False)
        elif export_format == "netcdf":
            data.to_netcdf(output_path, engine="netcdf4", encoding=encoding)
        elif export_format == "hdf5":
            data.to_netcdf(output_path, engine="h5netcdf", encoding=encoding)
        elif export_format == "parquet":
            data.to_dataframe().to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        self.log(f"Data exported successfully to {output_path}")
        return self.context

    def generate_diagnostics(self):
        """
        Generate diagnostics for the export step.
        """
        self.log(f"Generating diagnostics for {self.step_name}")
        diag.generate_diagnostics(self.context, self.step_name)
        self.log(f"Diagnostics generated successfully.")
