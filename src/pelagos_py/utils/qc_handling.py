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

"""Class definition to handle quality control bulk operations."""

import numpy as np
import xarray as xr

#: Flags a step's calculations ignore unless ``calculation_flag_filter`` says
#: otherwise: probably-bad (3), bad (4) and missing (9). Distinct from
#: ``flag_filter_settings``, which removes samples from processing entirely —
#: these samples are still corrected, they just don't inform the correction.
DEFAULT_CALCULATION_FLAGS = [3, 4, 9]


class QCHandlingMixin:
    def __init__(self):
        # fetch user inputs
        qc_settings = self.parameters.get("qc_handling_settings") or {}
        self.filter_settings = qc_settings.get("flag_filter_settings") or {}
        self.behaviour = qc_settings.get("reconstruction_behaviour") or "reinsert"

        calculation_flags = qc_settings.get("calculation_flag_filter")
        self.calculation_flag_filter = (
            list(DEFAULT_CALCULATION_FLAGS)
            if calculation_flags is None
            else list(calculation_flags)
        )

        self.flag_mapping = {flag: flag for flag in list(range(10))}
        if user_mappings := qc_settings.get("flag_mapping"):
            self.flag_mapping.update(user_mappings)

        # Validate that data exists in the processing context (logs + STOPs the
        # pipeline if absent; see BaseStep.check_data).
        self.check_data()
        self.data = self.context["data"].copy(deep=True)

        # Pristine "before" snapshot for reconstruct_data/update_qc. Those only
        # read back the filter_settings variables (and their _QC), so copy just
        # those rather than the whole dataset. Steps needing a broader snapshot
        # (e.g. Salinity/Chla diagnostics) replace this in their own run().
        snapshot_vars = [
            name
            for var in self.filter_settings
            for name in (var, f"{var}_QC")
            if name in self.data
        ]
        self.data_copy = self.data[snapshot_vars].copy(deep=True)

        # Check that the variables are present for filter execusion
        missing_variables = []
        for var in self.filter_settings:
            if var not in self.data or f"{var}_QC" not in self.data:
                self.log(
                    f"One or both of {var}/{var}_QC are missing from the dataset. They will be skipped."
                )
                missing_variables.append(var)
        for missing in missing_variables:
            self.filter_settings.pop(missing)

        # Continue method resolution order
        super().__init__()

    def print_qc_settings(self):
        self.log(
            "\n--------------------\n"
            f"Filter settings: {self.filter_settings}\n"
            f"Reconstruction behaviour: {self.behaviour}\n"
            f"Flag mappings: {self.flag_mapping}\n"
            "--------------------"
        )

    def filter_qc(self):
        """
        NaN-out data based on bad QC flags
        """
        for var, flags_to_nan in self.filter_settings.items():
            # find all positions where bad flags are present
            mask = ~self.data[f"{var}_QC"].isin(flags_to_nan)

            # nan-out the bad flagged data
            self.data[var] = self.data[var].where(mask, np.nan)

    def calculation_mask(self, variables):
        """
        Boolean mask of the samples a step may compute from.

        A sample is usable only where *every* listed variable carries a flag outside
        ``calculation_flag_filter`` (by default probably-bad (3), bad (4) and missing
        (9)). Unlike :meth:`filter_qc`, this does not touch ``self.data`` — the caller
        applies it to whatever it computes its statistics from, so the excluded samples
        still receive the step's correction, they just don't inform it. A variable with
        no ``_QC`` companion contributes nothing to the mask.

        parameters
        ----------
        variables : list of str
            Variables whose flags gate the calculation, e.g. the inputs it reads.

        returns
        -------
        numpy.ndarray
            Boolean array over N_MEASUREMENTS; True where the sample is usable.
        """
        mask = np.ones(self.data.sizes["N_MEASUREMENTS"], dtype=bool)
        if not self.calculation_flag_filter:
            return mask

        ungated = []
        for var in variables:
            if f"{var}_QC" not in self.data:
                ungated.append(var)
                continue
            mask &= ~np.isin(self.data[f"{var}_QC"].values, self.calculation_flag_filter)

        if ungated:
            self.log(
                f"No QC found for {ungated}; their values cannot be excluded from "
                "this step's calculations."
            )
        n_excluded = int((~mask).sum())
        if n_excluded:
            self.log(
                f"Excluding {n_excluded} of {mask.size} samples flagged "
                f"{self.calculation_flag_filter} in {list(variables)} from this "
                "step's calculations (they are still corrected)."
            )
        return mask

    def reconstruct_data(self):
        """
        Reconstruct data by replacing flagged values with original values.

        raises
        ------
        KeyError
            If the specified behaviour is not specified in this method.
        """
        if self.behaviour == "replace":
            pass

        elif self.behaviour == "reinsert":
            for var, flags_to_nan in self.filter_settings.items():
                # Find all of the postitions where there was bad data
                mask = self.data[f"{var}_QC"].isin(flags_to_nan)

                # Where there was a bad flag, reinsert the original values back into the data
                self.data[var] = xr.where(mask, self.data_copy[var], self.data[var])

        else:
            raise KeyError(f"Behaviour '{self.behaviour}' is not recgnised.")

    def update_qc(self):
        """
        Update QC flags based on changes in data values
        """
        for var in self.filter_settings.keys():
            # Find all values that haven't changed during processing
            is_same = self.data[var] == self.data_copy[var]
            both_nan = np.logical_and(
                self.data[var].isnull(), self.data_copy[var].isnull()
            )  # required because nan == nan is False
            mask = is_same | both_nan

            # Remap flags per flag_mapping. Flags with no remap rule (incl. NaN)
            # pass through unchanged. Done with xr.where rather than a vectorized
            # dict lookup so NaN flags survive (and the dtype is preserved).
            updated_flags = self.data[f"{var}_QC"].copy()
            for old_flag, new_flag in self.flag_mapping.items():
                if old_flag == new_flag:
                    continue
                updated_flags = xr.where(
                    self.data[f"{var}_QC"] == old_flag, new_flag, updated_flags
                )

            # Where data has changed, replace the old flag with the updated flag
            self.data[f"{var}_QC"] = xr.where(
                mask, self.data_copy[f"{var}_QC"], updated_flags
            )

    def generate_qc(self, qc_constituents: dict):
        """
        Generate QC flags for child variables based on parent variables' QC flags.

        parameters
        ----------
        qc_constituents : dict
            A dictionary mapping child QC variable names to lists of parent QC variable names.
        """
        # Unpack the parent qc
        for qc_child, qc_parents in qc_constituents.items():
            # Check the child exists
            if qc_child[:-3] not in self.data:
                self.log(
                    f"Trying to assign QC to a variable ({qc_child[:-3]}) which is not present in the dataset. Skipping..."
                )
                continue

            # Check parents are present
            if not set(qc_parents).issubset(set(self.data.data_vars)):
                self.log(
                    f"{qc_child} is missing one or multiple of ({qc_parents}) in the dataset. Skipping..."
                )
                continue

            # Assign the child the first parents QC
            self.data[qc_child] = self.data[qc_parents[0]].copy(deep=True)

            # If there is more than 1 parent, then itteratively upgrade the QC
            if len(qc_parents) > 1:
                # Define a combinatrix for flag upgrading priority
                qc_combinatrix = np.array(
                    [
                        [0, 0, 0, 3, 4, 0, 0, 0, 0, 9],
                        [0, 1, 2, 3, 4, 5, 1, 1, 8, 9],
                        [0, 2, 2, 3, 4, 5, 2, 2, 8, 9],
                        [3, 3, 3, 3, 4, 3, 3, 3, 3, 9],
                        [4, 4, 4, 4, 4, 4, 4, 4, 4, 9],
                        [0, 5, 5, 3, 4, 5, 5, 5, 8, 9],
                        [0, 1, 2, 3, 4, 5, 6, 6, 8, 9],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [0, 8, 8, 3, 4, 8, 8, 8, 8, 9],
                        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                    ]
                )

                for qc_parent in qc_parents[1:]:
                    self.data[qc_child][:] = qc_combinatrix[
                        self.data[qc_child], self.data[qc_parent]
                    ]

            # Flag nans as missing values
            is_nan = np.isnan(self.data[f"{qc_child[:-3]}"])
            self.data[f"{qc_child}"] = xr.where(is_nan, 9, self.data[f"{qc_child}"])

        # Check for any new columns that are missing QC
        all_var_names = {
            var
            for var in self.data.data_vars
            if var.isupper() and ("_QC" not in var) and (var not in self.data.dims)
        }
        all_qc_names = {var[:-3] for var in self.data.data_vars if "_QC" in var}
        missing_qc = all_var_names - all_qc_names

        if len(missing_qc) > 0:
            self.log(
                f"The following variables are missing QC: {missing_qc}. Assigning unchecked (0) QC flags."
            )
            data_subset = self.data[list(missing_qc)]
            flags = (
                xr.where(data_subset.isnull(), 9, 0)
                .astype(int)
                .rename({var: f"{var}_QC" for var in missing_qc})
            )
            self.data.update(flags)
