"""Tests the shared QC handling mixin (src/pelagos_py/utils/qc_handling.py).

Covers ``calculation_mask``, which marks the samples a step may compute from.
Unlike ``filter_qc`` it never touches the dataset: excluded samples still receive
the step's correction, they just do not inform it. The steps that use it
(Deep Correction, Mixed Layer Depth, CHLA Quenching) exercise it end-to-end in
their own test modules.
"""

#   Test module import
from pelagos_py.utils import qc_handling

import numpy as np
import xarray as xr

QCHandlingMixin = qc_handling.QCHandlingMixin
DEFAULT_CALCULATION_FLAGS = qc_handling.DEFAULT_CALCULATION_FLAGS


def make_step(flags=None, calculation_flag_filter=None):
    """Bare mixin instance holding a dataset with the given per-variable flags.

    ``flags`` maps a variable name to its QC flags; the variable's values are
    irrelevant here, only the flags gate the mask.
    """
    flags = flags or {}
    n = len(next(iter(flags.values()))) if flags else 0
    data = {}
    for var, var_flags in flags.items():
        data[var] = ("N_MEASUREMENTS", np.zeros(n))
        if var_flags is not None:
            data[f"{var}_QC"] = ("N_MEASUREMENTS", np.asarray(var_flags, dtype=int))

    step = QCHandlingMixin.__new__(QCHandlingMixin)
    step.data = xr.Dataset(data)
    step.calculation_flag_filter = (
        list(DEFAULT_CALCULATION_FLAGS)
        if calculation_flag_filter is None
        else list(calculation_flag_filter)
    )
    step.log = lambda message: None
    return step


def test_defaults_exclude_probably_bad_bad_and_missing():
    """3, 4 and 9 are excluded by default; every other flag is usable."""
    step = make_step({"CHLA": [0, 1, 2, 3, 4, 5, 8, 9]})
    assert step.calculation_mask(["CHLA"]).tolist() == [
        True, True, True, False, False, True, True, False
    ]


def test_a_sample_needs_every_listed_variable_to_be_usable():
    """The mask is the AND across variables, so one bad input excludes the sample."""
    step = make_step({"CHLA": [1, 1, 4, 1], "DEPTH": [1, 4, 1, 1]})
    assert step.calculation_mask(["CHLA", "DEPTH"]).tolist() == [
        True, False, False, True
    ]
    # Gating on one variable alone ignores the other's flags.
    assert step.calculation_mask(["CHLA"]).tolist() == [True, True, False, True]


def test_variable_without_qc_does_not_gate():
    """A variable with no _QC companion cannot exclude anything."""
    step = make_step({"CHLA": [1, 4, 1], "BBP700": None})
    assert step.calculation_mask(["CHLA", "BBP700"]).tolist() == [True, False, True]


def test_empty_filter_makes_every_sample_usable():
    """An empty calculation_flag_filter opts out of the exclusion entirely."""
    step = make_step({"CHLA": [3, 4, 9, 1]}, calculation_flag_filter=[])
    assert step.calculation_mask(["CHLA"]).all()


def test_filter_is_configurable():
    """Only the listed flags are excluded, so a 3 stays usable when not listed."""
    step = make_step({"CHLA": [3, 4, 9, 1]}, calculation_flag_filter=[4, 9])
    assert step.calculation_mask(["CHLA"]).tolist() == [True, False, False, True]


def test_mask_does_not_mutate_the_data():
    """calculation_mask is read-only: excluded samples survive for correction."""
    step = make_step({"CHLA": [1, 4, 1]})
    before = step.data["CHLA"].values.copy()
    step.calculation_mask(["CHLA"])
    assert np.array_equal(step.data["CHLA"].values, before)
