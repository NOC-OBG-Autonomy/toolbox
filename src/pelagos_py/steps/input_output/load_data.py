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

"""Class definition for loading data steps."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
import pelagos_py.utils.diagnostics as diag

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

MIN_YEAR_FILTER = "1990-01-01"


@register_step
class LoadOG1(BaseStep):
    """
    Loads NetCDF files from ``file_path``. 
    
    If ``filter_bad_times`` is set then measurements made
    outside of the time range specified by ``data_start`` and ``data_end`` are removed before 
    further processing.

    Time values are expected to be stored under the "TIME" variable name in the NetCDF file 
    (corresponding to OG1 format). If "TIME" is not monotonically increasing, or has missing values
    (NaT) then this will raise an error.

    Parameters
    ----------
    filter_bad_time : bool, optional
        If True (default), removes all timestamps outside the expected time window.
    data_start : str or np.datetime64, optional
        The minimum valid timestamp for the data. If not provided, the filter defaults 
        to the DEPLOYMENT_TIME found in the dataset, or 1990-01-01T00:00:00 if no 
        deployment time is found.
    data_end : str or np.datetime64, optional
        The maximum valid timestamp for the data. If not provided, it defaults to 
        the current system time when the pipeline is run.

    Examples
    --------
    Example usage in a pipeline configuration:

    .. code-block:: yaml

        steps:
          - name: Load OG1
            parameters:
              file_path: "/path/to/your/dataset.nc"
              filter_bad_time: false
              data_start: "2023-05-01T00:00:00"
              data_end: "2024-05-01T00:00:00"
    """

    step_name = "Load OG1"
    required_variables = []
    provided_variables = ["TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "CNDC"]

    parameter_schema = {
        "file_path": {
            "type": str,
            "required": True,
            "description": "Path to the OG1 data file."
        },
        "filter_bad_time": {
            "type": bool,
            "default": True,
            "description": "If True, removes all timestamps outside the expected time window."
        },
        "data_start": {
            "type": str,
            "default": None,
            "description": "Minimum valid timestamp (e.g. '2023-05-01T00:00:00'). Defaults to deployment time or 1990."
        },
        "data_end": {
            "type": str,
            "default": None,
            "description": "Maximum valid timestamp. Defaults to current system time."
        }
    }

    def run(self):
        # Fail cleanly if the file is missing, rather than surfacing a wrapped
        # xarray/netCDF4 traceback.
        if not Path(self.file_path).is_file():
            self.halt(
                f"Could not find data file '{self.file_path}'. "
                "Check the 'file_path' parameter points to an existing file."
            )

        # load data from xarray
        self.data = xr.open_dataset(self.file_path)
        self.log(f"Loaded data from {self.file_path}")

        self.data.load()

        # Parameters are resolved from parameter_schema in BaseStep.__init__,
        # so every declared attribute is guaranteed to be set.
        filter_bad_time = self.filter_bad_time
        data_start = self.data_start
        data_end = self.data_end

        # Filter data to the specified time window
        if filter_bad_time and (
            "TIME" in self.data.variables or "TIME" in self.data.coords
        ):
            orig_len = len(self.data["TIME"])
            time_array = self.data["TIME"]

            start_val = (
                np.datetime64(data_start)
                if data_start
                else np.datetime64(MIN_YEAR_FILTER)
            )
            end_val = (
                np.datetime64(data_end)
                if data_end
                else np.datetime64(pd.Timestamp.now())
            )

            valid_mask = time_array >= start_val
            valid_mask &= time_array <= end_val

            if not data_start and "DEPLOYMENT_TIME" in self.data.variables:
                deploy_time = pd.to_datetime(self.data["DEPLOYMENT_TIME"].values)
                if isinstance(deploy_time, pd.DatetimeIndex):
                    deploy_time = deploy_time[0]
                valid_mask &= time_array >= np.datetime64(deploy_time)

            time_dim = self.data["TIME"].dims[0]
            self.data = self.data.isel({time_dim: valid_mask.values})
            new_len = len(self.data["TIME"])

            if new_len < orig_len:
                self.log_warn(
                    f"Removed {orig_len - new_len} records containing invalid or pre-deployment timestamps."
                )

        if "TIME" in self.data.coords:
            self.data = self.data.reset_coords("TIME", drop=False)
            self.data = self.data.reset_coords("LATITUDE", drop=False)
            self.data = self.data.reset_coords("LONGITUDE", drop=False)

        if "TIME" not in self.data.data_vars:
            raise ValueError(
                "\n'TIME' could not be found in the dataset. Pipelines cannot be run without this variable.\n"
                "If TIME is listed under another name, please rename it to conform to the OG1 format."
            )

        # Check that the "TIME" variable is monotonic and has no missing values
        if np.any(np.isnat(self.data["TIME"].values)):
            raise ValueError(
                "\n'TIME' has NaT values. Pipelines cannot be run without a continuous monotonic time coordinate.\n"
                "Please remove these values (and their concurrent measurements) from the input."
            )

        if not np.all(np.diff(self.data["TIME"]) >= 0):
            self.log_warn(
                "'TIME' is not monotonically increasing. This may cause fatal issues in processing. "
                "Please check the quality of your input data."
            )
            
        # Make these available to other steps (e.g. Format Checker, which reads the
        # original file from disk; the report uses filename_core for naming).
        self.context["global_parameters"]["filename_core"] = Path(self.file_path).stem
        self.context["global_parameters"]["source_file"] = self.file_path

        self.normalise_qc_flags()

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def normalise_qc_flags(self):
        """
        Coerce QC flag variables to integers, mapping NaN to 0 (no QC).

        Raw OG1 files store QC flags as float with NaN. Downstream steps use
        flags as integer array indices / dict keys, so normalise them once at
        load rather than having each step guard against float NaN flags. NaN
        means no QC has been applied yet, which is flag 0 (not 9/missing).
        """
        qc_vars = [v for v in self.data.data_vars if v.endswith("_QC")]
        normalised = []
        for var in qc_vars:
            flags = self.data[var]
            if flags.dtype.kind != "f":
                continue
            # int8 is plenty for flags 0-9 and keeps QC arrays small.
            self.data[var] = flags.fillna(0).astype(np.int8)
            normalised.append(var)

        if normalised:
            self.log_warn(
                f"{len(normalised)} QC flag variable(s) were stored as float with "
                "NaN values; normalised to integer (NaN flags mapped to 0/no QC). "
                f"Affected: {', '.join(sorted(normalised))}."
            )

    def generate_diagnostics(self):
        """
        Print a structural summary of the loaded dataset.

        Called automatically at the end of :meth:`run` when ``diagnostics`` is
        enabled. Delegates to
        :func:`pelagos_py.utils.diagnostics.generate_info`, which prints the
        dataset's dimensions, variables and global attributes (via
        :meth:`xarray.Dataset.info`) to stdout — a quick check that the data
        was loaded as expected.
        """
        self.log_generating_diagnostics()
        diag.generate_info(self.data)