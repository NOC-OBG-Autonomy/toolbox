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

# Flags a step's calculations ignore unless ``calculation_flag_filter`` says
# otherwise: probably-bad (3), bad (4) and missing (9). Unlike
# ``flag_filter_settings``, these samples are still corrected, they just don't
# inform the correction.
DEFAULT_CALCULATION_FLAGS = [3, 4, 9]


class QCHandlingMixin:
    def __init__(self):
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

        # Logs + STOPs the pipeline if data is absent (see BaseStep.check_data).
        self.check_data()
        full_data = self.context["data"]

        if getattr(self, "uses_data_subset", False):
            # Opt-in: deep-copying the whole dataset is wasteful when a step only
            # reads required_variables/provided_variables/optional_variables (the
            # last for anything read conditionally, e.g. diagnostics-only vars),
            # their _QC companions, and whatever the config's flag_filter_settings
            # names. Steps write back via ``self.context["data"].update(self.data)``
            # so nothing outside the subset is ever dropped. Not yet the default:
            # steps declaring param-driven variable names (e.g. ``self.apply_to``)
            # need auditing before they can safely opt in.
            subset_names = set(getattr(self, "required_variables", []))
            subset_names.update(getattr(self, "provided_variables", []))
            subset_names.update(getattr(self, "optional_variables", []))
            # variable_parameters: names of *parameters* (already resolved onto
            # self, see BaseStep.__init__) whose value is itself a variable name
            # (or list of them), e.g. par_var="DOWNWELLING_PAR". Config-driven, so
            # can't be listed statically like optional_variables.
            for attr in getattr(self, "variable_parameters", []):
                value = getattr(self, attr, None)
                if value is None:
                    continue
                values = [value] if isinstance(value, str) else value
                subset_names.update(values)
                # Several steps use the OG1 "prefer an existing _ADJUSTED variant"
                # convention (e.g. apply_to="CHLA" but CHLA_ADJUSTED is read/used if
                # present). Harmless to include speculatively: filtered out below if
                # it doesn't exist in the full dataset.
                subset_names.update(f"{v}_ADJUSTED" for v in values)
            subset_names.update(f"{var}_QC" for var in list(subset_names))
            subset_names.update(
                name
                for var in self.filter_settings
                for name in (var, f"{var}_QC")
            )
            subset_vars = [name for name in subset_names if name in full_data.variables]
            self.data = full_data[subset_vars].copy(deep=True)
        else:
            self.data = full_data.copy(deep=True)

        # "Before" snapshot for reconstruct_data/update_qc, which only read back
        # the filter_settings variables (and their _QC). Steps needing a broader
        # snapshot (e.g. Salinity/Chla diagnostics) replace this in their run().
        snapshot_vars = [
            name
            for var in self.filter_settings
            for name in (var, f"{var}_QC")
            if name in self.data
        ]
        self.data_copy = self.data[snapshot_vars].copy(deep=True)

        # Drop filter_settings variables whose data or _QC is missing.
        missing_variables = []
        for var in self.filter_settings:
            if var not in self.data or f"{var}_QC" not in self.data:
                self.log(
                    f"One or both of {var}/{var}_QC are missing from the dataset. They will be skipped."
                )
                missing_variables.append(var)
        for missing in missing_variables:
            self.filter_settings.pop(missing)

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
        """NaN-out data based on bad QC flags."""
        for var, flags_to_nan in self.filter_settings.items():
            mask = ~self.data[f"{var}_QC"].isin(flags_to_nan)
            self.data[var] = self.data[var].where(mask, np.nan)

    def calculation_mask(self, variables):
        """
        Boolean mask over N_MEASUREMENTS of the samples a step may compute from.

        True only where *every* listed variable carries a flag outside
        ``calculation_flag_filter``. Unlike :meth:`filter_qc` this doesn't touch
        ``self.data``, so excluded samples are still corrected, they just don't
        inform the correction. A variable with no ``_QC`` contributes nothing.

        parameters
        ----------
        variables : list of str
            Variables whose flags gate the calculation.
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
            # Off the console, where it would repeat once per variable-set.
            self.log(
                f"Excluding {n_excluded} of {mask.size} samples flagged "
                f"{self.calculation_flag_filter} in {list(variables)} from this "
                "step's calculations (they are still corrected).",
                console=False,
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
                mask = self.data[f"{var}_QC"].isin(flags_to_nan)
                self.data[var] = xr.where(mask, self.data_copy[var], self.data[var])

        else:
            raise KeyError(f"Behaviour '{self.behaviour}' is not recgnised.")

    def update_qc(self):
        """Update QC flags based on changes in data values."""
        for var in self.filter_settings.keys():
            is_same = self.data[var] == self.data_copy[var]
            both_nan = np.logical_and(
                self.data[var].isnull(), self.data_copy[var].isnull()
            )  # required because nan == nan is False
            mask = is_same | both_nan

            # Remap per flag_mapping; unmapped flags (incl. NaN) pass through.
            # xr.where rather than a dict lookup so NaN flags and dtype survive.
            updated_flags = self.data[f"{var}_QC"].copy()
            for old_flag, new_flag in self.flag_mapping.items():
                if old_flag == new_flag:
                    continue
                updated_flags = xr.where(
                    self.data[f"{var}_QC"] == old_flag, new_flag, updated_flags
                )

            # Where data has changed, apply the updated flag.
            self.data[f"{var}_QC"] = xr.where(
                mask, self.data_copy[f"{var}_QC"], updated_flags
            )

    def generate_qc(self, qc_constituents: dict):
        """
        Generate QC flags for child variables based on parent variables' QC flags.

        parameters
        ----------
        qc_constituents : dict
            Maps child QC variable names to lists of parent QC variable names.
        """
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

            # Assign the child the first parent's QC, then upgrade per parent.
            self.data[qc_child] = self.data[qc_parents[0]].copy(deep=True)

            if len(qc_parents) > 1:
                # Combinatrix defining flag-upgrade priority.
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

        # Assign unchecked QC to any new variables that still lack it.
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
