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

"""Class definition for quality control steps."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
import pelagos_py.utils.diagnostics as diag
from pelagos_py.steps import QC_CLASSES

#### Custom imports ####
import polars as pl
import xarray as xr
import numpy as np
import json


@register_step
class ApplyQC(BaseStep):
    """
    Step to apply quality control tests to the dataset.

    Inherits properties from BaseStep (see base_step.py).
    """

    step_name = "Apply QC"

    parameter_schema = {
        "qc_settings": {
            "type": dict,
            "required": True,
            "description": (
                "Mapping of QC test name -> that test's settings. Each test is run "
                "in order and merges its flags into the dataset's _QC columns."
            ),
        },
    }

    def organise_flags(self, new_flags):
        """
        Method for taking in new flags (new_flags) and cross checking against existing flags (self.flag_store), including upgrading flags when necessary, following ARGO flagging standards.
        See Wong et al. 2025 pp. 106 (http://dx.doi.org/10.13155/33951) and Mancini et al. 2021 pp. 43-44 for additional ARGO flag definitions.

        Combinatrix logic:
        0: No QC performed, the initial flag.
        1: Good data. No adjustment needed.
        2: Probably good data.
        3: Probably bad data that are potentially correctable.
        4: Bad data that are not correctable.
        5: Value changed.
        6, 7: Not used.
        8: Estimated by interpolation, extrapolation, or other algorithm.
        9: Missing value.

        The combinatrix defines flagging priority when merging in new flags. The flag value itself acts as a kind of index.
        As an example, if an existing flag is 2 (probably good data) and a new flag is 4 (bad data), the resulting flag will be 4.
        2 (probably good data) + 4 (bad data) -> 4 (bad data)
        3 (probably bad data) + 5 (value changed) -> 3 (probably bad data)

        parameters
        ----------
        new_flags : xarray.Dataset
            Dataset containing new QC flag variables to be merged into the existing flag store.
        """

        # Define combinatrix for handling flag upgrade behaviour
        qc_combinatrix = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 1, 2, 3, 4, 5, 1, 1, 8, 9],
                [2, 2, 2, 3, 4, 5, 2, 2, 8, 9],
                [3, 3, 3, 3, 4, 3, 3, 3, 3, 9],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 9],
                [5, 5, 5, 3, 4, 5, 5, 5, 8, 9],
                [6, 1, 2, 3, 4, 5, 6, 6, 8, 9],
                [7, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [8, 8, 8, 3, 4, 8, 8, 8, 8, 9],
                [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            ]
        )

        # Update existing flag columns
        flag_columns_to_update = set(new_flags.data_vars) & set(
            self.flag_store.data_vars
        )
        for column_name in flag_columns_to_update:
            self.flag_store[column_name][:] = qc_combinatrix[
                self.flag_store[column_name], new_flags[column_name]
            ]

        # Add new QC flag columns if they dont already exist
        flag_columns_to_add = set(new_flags.data_vars) - set(self.flag_store.data_vars)
        if len(flag_columns_to_add) > 0:
            for column_name in flag_columns_to_add:
                self.flag_store[column_name] = new_flags[column_name]

    def run(self):
        """
        Run the Apply QC step.

        raises
        ------
        KeyError
            If no QC operations are specified, if requested QC tests are invalid, or esssential variables are missing.
        ValueError
            If no data is found in context.
        """
        # Defining the order of operations
        if len(self.qc_settings.keys()) == 0:
            raise KeyError(
                "[Apply QC] No QC operations were specified in an ApplyQC step."
            )
        else:
            #   Check requested QC tests against valid tests
            invalid_requests = set(self.qc_settings.keys()) - set(QC_CLASSES.keys())
            if invalid_requests:
                raise KeyError(
                    f"[Apply QC] The following requested QC tests could not be found: {invalid_requests}"
                )
        queued_qc = [QC_CLASSES.get(key) for key in self.qc_settings.keys()]

        # Check if the data is in the context
        self.check_data()
        data = self.context["data"].copy(deep=True)

        # Try and fetch the qc history from context and update it
        qc_history = self.context.setdefault("qc_history", {})

        # Collect all of the required varible names and qc outputs for each test
        all_required_variables = set({})
        test_qc_outputs_cols = set({})
        for test in queued_qc:
            if hasattr(test, "dynamic"):
                # Initialise the test to check its dynamic attributes
                test_instance = test(None, **self.qc_settings[test.qc_name])
                all_required_variables.update(test_instance.required_variables)
                test_qc_outputs_cols.update(test_instance.qc_outputs)
                del test_instance
            else:
                all_required_variables.update(test.required_variables)
                test_qc_outputs_cols.update(test.qc_outputs)
            #   Check that the required variables for the test are in the dataset.
            #   Use data.variables (data vars + coordinates), not data.keys() (data
            #   vars only), so a required variable stored as a coordinate (e.g. TIME,
            #   LATITUDE, LONGITUDE) is not falsely reported as missing.
            present = set(data.variables)
            if not set(all_required_variables).issubset(present):
                raise KeyError(
                    f"[Apply QC] The data is missing variables: ({set(all_required_variables) - present}) which are required for running QC '{test.qc_name}'."
                    f" Make sure that the variables are present in the data, or remove tests from the order."
                )
        # Convert data to polars for fast processing
        # Fetch existing flags from the data and create a place to store them
        existing_flags = [
            flag_col for flag_col in data.data_vars if flag_col in test_qc_outputs_cols
        ]
        self.flag_store = xr.Dataset(coords={"N_MEASUREMENTS": data["N_MEASUREMENTS"]})
        if len(existing_flags) > 0:
            self.log(f"Found existing flags columns {set(existing_flags)} in data.")
            self.flag_store = data[existing_flags].fillna(9).astype(int)

        other_existing_qc = set(
            [var for var in data.data_vars if var.endswith("_QC")]
        ) - set(test_qc_outputs_cols)
        if any(other_existing_qc):
            # File-only: this set can be large and repeats every QC step.
            self.logger.info(
                "[%s] %s pre-existing QC column(s) left untouched by this step: %s",
                self.name,
                len(other_existing_qc),
                ", ".join(sorted(other_existing_qc)),
                extra={"console": False},
            )

        # Initialize the missing flag columns
        mia_qc = test_qc_outputs_cols - set(data.data_vars)
        base = [var[:-3] for var in mia_qc]
        if not set(base).issubset(
            set(data.keys())
        ):  #   Confirm that the required QC columns exist
            raise KeyError(
                f"[Apply QC] The data is missing: ({set(base) - set(data.keys())}), which is/are defined in the config as a variable to flag or use during one of the tests."
                f" Double check the configuration file and make sure all variable parameters (like 'also flag' [CHLA]) are present in the data."
            )
        data_subset = data[base]
        masks = xr.where(data_subset.isnull(), 9, 0).astype(int)
        masks = masks.rename({var: f"{var}_QC" for var in base})
        self.flag_store.update(masks)

        # Run through all of the QC steps and add the flags to flag_store
        for qc_qc_name, qc_test_params in self.qc_settings.items():
            # Create an instance of this test step
            self.log(
                f"Applying: {qc_qc_name}"
            )  # print(f"[Apply QC] Applying: {qc_qc_name}")
            qc_test_instance = QC_CLASSES[qc_qc_name](data, **qc_test_params)
            returned_flags = (
                qc_test_instance.return_qc()
            )  #   Runs the test, returns the flags
            self.organise_flags(returned_flags)

            # Update QC history
            for flagged_var in returned_flags.data_vars:
                #   Track percent of flags no longer 0 (following ARGO convention)
                var_flags = returned_flags[flagged_var]
                percent_flagged = (var_flags.to_numpy() != 0).sum() / len(var_flags)
                if percent_flagged == 0:
                    self.log_warn(
                        f"All flags for {flagged_var} remain 0 after {qc_qc_name}"
                    )
                # else: #   TODO: Add 'verbose' log option if needed. Might not need to happen at this point.
                #     self.log(f"{percent_flagged*100:.2f}% of {flagged_var} points accounted for by {qc_qc_name}")
                qc_history.setdefault(flagged_var, []).append(
                    (qc_qc_name, percent_flagged)
                )

                # Write additional QC details to _QC variable attributes
                # TODO: Find where columns are initialized, or just run on non-QC'd datasets
                parent_attrs = data[flagged_var[:-3]].attrs
                attrs = self.flag_store[flagged_var].attrs
                attrs["quality_control_conventions"] = "Argo standard flags"
                attrs["valid_min"] = 0
                attrs["valid_max"] = 9
                attrs["flag_values"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                attrs["flag_meanings"] = (
                    "NO_QC, GOOD, PROB_GOOD, PROB_BAD, BAD, VALUE_CHANGED, NOT_USED, NOT_USED, ESTIMATED, MISSING"
                )
                attrs["long_name"] = f"{parent_attrs['long_name']} quality flag"
                attrs["standard_name"] = f"{parent_attrs['standard_name']}_flag"
                attr_test = qc_qc_name.replace(" ", "_").lower()
                attrs[f"{attr_test}_flag_cts"] = json.dumps(
                    {i: int(np.sum(var_flags.to_numpy() == i)) for i in range(10)}
                )
                attrs[f"{attr_test}_stats"] = json.dumps(
                    var_flags.to_series().describe().round(5).to_dict()
                )
                attrs[f"{attr_test}_params"] = json.dumps(qc_test_params)
                # Can get indices of 3/4 with np.where(var_flags.to_numpy() == 3)[0] for future reference

            # Diagnostic plotting. Never let a diagnostic-only error abort QC;
            # this matters when the report writer force-enables diagnostics to
            # capture plots for every test.
            if self.diagnostics:
                try:
                    qc_test_instance.plot_diagnostics()
                except Exception as exc:  # noqa: BLE001 - diagnostics must not be fatal
                    self.log_warn(
                        f"Diagnostic plotting failed for QC test '{qc_qc_name}': {exc}"
                    )

            # Once finished, remove the test instance from memory
            del qc_test_instance

        # Append the flags from self.flag_store to the xarray data and push back into context
        for flag_column in self.flag_store.data_vars:
            if (self.flag_store[flag_column] == 0).all():
                self.log_warn(
                    f"{flag_column} is all 0 after running all QC steps. Check intended QC variables and test requirements."
                )
            elif (self.flag_store[flag_column] == 0).any():
                n_zero = int((self.flag_store[flag_column] == 0).sum())
                self.log_warn(
                    f"{flag_column} (length={len(self.flag_store[flag_column])}) has {n_zero} zero QC values following all QC steps."
                )

            data[flag_column] = (
                ("N_MEASUREMENTS",),
                self.flag_store[flag_column].to_numpy(),
            )
            data[flag_column].attrs = self.flag_store[flag_column].attrs.copy()
        self.context["data"] = data
        self.context["qc_history"] = qc_history

        return self.context
