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
"""This module defines the base class for QC tests and a registry for QC test classes."""

import logging

from pelagos_py.utils import parameter_spec

REGISTERED_QC = {}
"""Registry of explicitly registered QC test classes."""

flag_cols = {
    0: "gray",
    1: "blue",
    2: "lightblue",
    3: "orange",
    4: "red",
    5: "gray",
    6: "gray",
    7: "gray",
    8: "cyan",
    9: "black",
}
"""Map of QC flag values to colors for diagnostics plotting."""


def register_qc(cls):
    """Decorator to mark QC tests that can be accessed by the ApplyQC step."""
    qc_name = getattr(cls, "qc_name", None)
    if qc_name is None:
        raise ValueError(
            f"QC test {cls.__name__} is missing required 'qc_name' attribute."
        )
    REGISTERED_QC[qc_name] = cls
    return cls


class BaseQC:
    """
    Initializes a base class for quality control, to be further tweaked when inherited.

    Follow the docstring format below when creating new QC tests.

    Target Variable: "Any" or a specific variable names (see impossible_location_test.py)
    Flag Number: "Any" or a specific ARGO flag number
    Variables Flagged: "Any" or specific variable names, possibly external to the target variable (see valid_profile_test.py)
    Your description follows here.

    Target Variable:
    Flag Number:
    Variables Flagged:

    """

    qc_name = None
    parameter_schema = {}
    required_variables = []
    qc_outputs = []

    def __init__(self, data, **kwargs):
        # data may be None when a test is instantiated to introspect its
        # required/provided variables from its parameters.
        # Shallow copy: a new Dataset container per test (cheap - no array copy)
        # so a test adding *_QC variables can't leak them into other tests sharing
        # the same underlying data. Deep copy was the main QC RAM ratchet.
        self.data = data.copy(deep=False) if data is not None else None

        # Connect to the main pipeline logging hierarchy
        self.logger = logging.getLogger(f"pelagos_py.pipeline.qc.{self.qc_name.replace(' ', '_')}")

        # Resolve parameters against the schema: applies defaults, enforces required
        # parameters, and rejects unknown ones. Resolved values become attributes.
        resolved = parameter_spec.resolve(
            self.parameter_schema, kwargs, label=self.qc_name
        )
        for k, v in resolved.items():
            setattr(self, k, v)

        self.flags = None

    @classmethod
    def describe_parameters(cls):
        """Return a JSON-serialisable description of this QC check's parameters.

        See :func:`pelagos_py.utils.parameter_spec.describe`.
        """
        return parameter_spec.describe(cls.parameter_schema or {})

    def log(self, message, console=True):
        """Log an info-level message with the QC name prefix. ``console=False`` keeps it in the log file only."""
        self.logger.info("[%s] %s", self.qc_name, message, extra={"console": console})

    def log_warn(self, message):
        """Log a warning-level message with the QC name prefix."""
        self.logger.warning("[%s] %s", self.qc_name, message)

    def log_progress(self, iterable=None, *, desc, total=None, unit="it", leave=False):
        """Wrap a countable loop as a standard pipeline progress bar (see :meth:`BaseStep.log_progress`)."""
        from pelagos_py.utils.console import progress_bar

        return progress_bar(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            leave=leave,
            logger=self.logger,
            step_name=self.qc_name,
        )

    def return_qc(self):
        """Representative of QC processing, to be overridden by subclasses.

        Returns
        -------
        flags : array-like
            Output QC flags for the data specific to the test.
        """
        self.flags = None  # replace with processing of some kind
        return self.flags

    def plot_diagnostics(self):
        """Representative of diagnostic plotting (optional)."""
        # Any relevant diagnostic is generated or written out here
        pass