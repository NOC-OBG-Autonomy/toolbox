"""Behavioural tests for the unified 'range qc' check
(src/pelagos_py/steps/quality_control/range_qc.py)."""

import numpy as np
import xarray as xr
import pytest

from pelagos_py.steps.quality_control.range_qc import range_qc


def make_data(**variables):
    """Build a minimal QC dataset from {var: 1-D array} pairs."""
    n = len(next(iter(variables.values())))
    data_vars = {k: ("N_MEASUREMENTS", np.asarray(v, dtype=float)) for k, v in variables.items()}
    return xr.Dataset(data_vars, coords={"N_MEASUREMENTS": np.arange(n)})


def test_ascending_band_flags_values_outside_it():
    """Ascending [low, high] is a good band: data below low or above high gets the flag."""
    data = make_data(TEMP=[-5.0, 10.0, 50.0])  # below, inside, above
    qc = range_qc(data, variable_ranges={"TEMP": {4: [-2.5, 40]}})

    flags = qc.return_qc()["TEMP_QC"].values
    assert list(flags) == [4, 1, 4]  # ends bad, middle good


def test_descending_band_flags_values_inside_it():
    """Descending [high, low] is an impossible band: only data strictly within it is flagged."""
    data = make_data(PRES=[-10.0, -3.0, 0.0])  # outside, inside, outside
    qc = range_qc(data, variable_ranges={"PRES": {3: [-2.4, -5]}})  # descending

    flags = qc.return_qc()["PRES_QC"].values
    assert list(flags) == [1, 3, 1]  # only the middle (in the impossible band) flagged


def test_infinite_descending_bound():
    """A descending [-5, -inf] flags everything below -5 (impossible band)."""
    data = make_data(PRES=[-10.0, -3.0])  # below -5, above -5
    qc = range_qc(data, variable_ranges={"PRES": {4: [-5, float("-inf")]}})  # descending

    assert list(qc.return_qc()["PRES_QC"].values) == [4, 1]


def test_single_scalar_flags_exact_matches():
    """A single scalar (not a pair) flags exact-value matches, e.g. a fill value."""
    data = make_data(PRES=[0.0, -3.0, 0.0, -1.5])
    qc = range_qc(data, variable_ranges={"PRES": {4: 0.0}})  # flag exactly 0.0 as bad

    flags = qc.return_qc()["PRES_QC"].values
    assert list(flags) == [4, 1, 4, 1]


def test_scalar_and_range_entries_combine():
    """A variable can mix a scalar fill-value flag with band ranges."""
    data = make_data(PRES=[0.0, -3.0, -10.0])  # fill value, inside impossible band, deep-bad
    qc = range_qc(
        data,
        variable_ranges={"PRES": {9: 0.0, 3: [-2.4, -5], 4: [-5, float("-inf")]}},
    )

    flags = qc.return_qc()["PRES_QC"].values
    assert list(flags) == [9, 3, 4]


def test_most_severe_flag_wins_on_overlap():
    """When ranges overlap, the more severe flag is applied."""
    data = make_data(TEMP=[100.0])  # far outside both good bands
    qc = range_qc(data, variable_ranges={"TEMP": {3: [0, 30], 4: [-2.5, 40]}})

    assert qc.return_qc()["TEMP_QC"].values[0] == 4


def test_ascending_and_descending_mixed_per_variable():
    """Inside/outside can be mixed in one test purely via bound order."""
    data = make_data(PRES=[-3.0, 100.0], TEMP=[10.0, 100.0])
    qc = range_qc(
        data,
        variable_ranges={
            "PRES": {4: [-2.4, -5]},  # descending -> impossible band
            "TEMP": {4: [-2.5, 40]},  # ascending  -> good band
        },
    )

    flags = qc.return_qc()
    assert list(flags["PRES_QC"].values) == [4, 1]  # inside the impossible band flagged
    assert list(flags["TEMP_QC"].values) == [1, 4]  # outside the good band flagged


def test_outside_keyword_flags_values_outside_band():
    """An explicit 'outside' keyword flags data outside the band, regardless of bound order."""
    data = make_data(TEMP=[-5.0, 10.0, 50.0])  # below, inside, above
    qc = range_qc(data, variable_ranges={"TEMP": {4: [-2.5, 40, "outside"]}})

    assert list(qc.return_qc()["TEMP_QC"].values) == [4, 1, 4]


def test_inside_keyword_flags_values_inside_band():
    """An explicit 'inside' keyword flags data within the band, regardless of bound order."""
    data = make_data(PRES=[-10.0, -3.0, 0.0])  # outside, inside, outside
    qc = range_qc(data, variable_ranges={"PRES": {3: [-5, -2.4, "inside"]}})  # ascending bounds

    assert list(qc.return_qc()["PRES_QC"].values) == [1, 3, 1]


def test_keyword_overrides_bound_order():
    """An explicit keyword wins over the ascending/descending fallback."""
    # Ascending bounds would mean 'outside' by the fallback, but 'inside' is forced.
    data = make_data(TEMP=[5.0, 20.0, 50.0])  # below, inside, above
    qc = range_qc(data, variable_ranges={"TEMP": {4: [10, 30, "inside"]}})

    assert list(qc.return_qc()["TEMP_QC"].values) == [1, 4, 1]


@pytest.mark.parametrize("kw", ["outside", "OUT", "o", "O", "Outside"])
def test_outside_keyword_aliases(kw):
    data = make_data(TEMP=[-5.0, 10.0, 50.0])
    qc = range_qc(data, variable_ranges={"TEMP": {4: [-2.5, 40, kw]}})
    assert list(qc.return_qc()["TEMP_QC"].values) == [4, 1, 4]


@pytest.mark.parametrize("kw", ["inside", "IN", "i", "I", "Inside"])
def test_inside_keyword_aliases(kw):
    data = make_data(PRES=[-10.0, -3.0, 0.0])
    qc = range_qc(data, variable_ranges={"PRES": {3: [-5, -2.4, kw]}})
    assert list(qc.return_qc()["PRES_QC"].values) == [1, 3, 1]


def test_multiple_bands_for_one_flag():
    """A flag can list several bands; a point is flagged if it falls in any of them."""
    data = make_data(X=[2.5, 5.0, 0.05, 20.0])  # inside [2,3]; clean; outside [0.1,10]; outside
    qc = range_qc(
        data,
        variable_ranges={"X": {4: [[2, 3, "inside"], [0.1, 10, "outside"]]}},
    )

    assert list(qc.return_qc()["X_QC"].values) == [4, 1, 4, 4]


def test_unknown_keyword_raises():
    data = make_data(TEMP=[10.0])
    qc = range_qc(data, variable_ranges={"TEMP": {4: [0, 30, "nonsense"]}})
    with pytest.raises(ValueError, match="Unknown range keyword"):
        qc.return_qc()


def test_also_flag_propagates_to_companions():
    """A bad CNDC point cross-flags its companions; good points are left good."""
    data = make_data(
        CNDC=[1.0, 30.0],   # first is out of range (bad), second good
        TEMP=[10.0, 10.0],  # both in range on their own
        PRES=[-3.0, -3.0],
    )
    qc = range_qc(
        data,
        variable_ranges={"CNDC": {4: [2, 45]}, "TEMP": {4: [-2.5, 40]}},
        also_flag={"CNDC": ["TEMP", "PRES"]},
    )

    flags = qc.return_qc()
    assert list(flags["CNDC_QC"].values) == [4, 1]
    # TEMP is tested (good) but inherits CNDC's bad flag at the bad point.
    assert list(flags["TEMP_QC"].values) == [4, 1]
    # PRES is not tested itself, so it mirrors CNDC's flags.
    assert list(flags["PRES_QC"].values) == [4, 1]


def test_also_flag_uses_argo_combinatrix_not_naive_overwrite():
    """Propagation merges via the Argo matrix: an existing bad (4) is never downgraded
    by a propagated probably-bad (3), but a probably-bad (3) is upgraded to bad (4)."""
    data = make_data(
        PRES=[-3.0, -10.0],   # impossible band: -3 -> 3 (prob bad); -10 (< -5) -> 4 (bad)
        TEMP=[100.0, 35.0],   # good band: 100 -> 4 (bad); 35 (outside [0,30]) -> 3 (prob bad)
    )
    qc = range_qc(
        data,
        variable_ranges={
            "PRES": {3: [-2.4, -5], 4: [-5, float("-inf")]},  # descending -> impossible
            "TEMP": {3: [0, 30], 4: [-2.5, 40]},              # ascending  -> good
        },
        also_flag={"PRES": ["TEMP"]},
    )

    flags = qc.return_qc()
    # Point 0: TEMP is already 4, PRES propagates 3 -> stays 4 (a naive overwrite would
    # have downgraded it to 3). Point 1: TEMP is 3, PRES propagates 4 -> upgraded to 4.
    assert list(flags["TEMP_QC"].values) == [4, 4]


def test_qc_outputs_include_companions():
    """Dynamic I/O resolution lists tested variables and propagation targets."""
    qc = range_qc(
        None,
        variable_ranges={"CNDC": {4: [2, 45]}},
        also_flag={"CNDC": ["PRES", "TEMP"]},
    )
    assert qc.required_variables == ["CNDC"]
    assert set(qc.qc_outputs) == {"CNDC_QC", "PRES_QC", "TEMP_QC"}


def test_flag_instead_flags_companion_without_flagging_the_source():
    """CHLA is flagged by DEPTH's bands (3 in the top 5 m, 2 below) and DEPTH is left
    out of the outputs entirely."""
    data = make_data(DEPTH=[2.0, 50.0], CHLA=[0.5, 0.5])
    qc = range_qc(
        data,
        variable_ranges={"DEPTH": {3: [0, 5, "inside"], 2: [0, 5, "outside"]}},
        flag_instead={"DEPTH": ["CHLA"]},
    )

    flags = qc.return_qc()
    assert list(flags["CHLA_QC"].values) == [3, 2]
    assert "DEPTH_QC" not in flags
    assert qc.qc_outputs == ["CHLA_QC"]


def test_flag_instead_merges_rather_than_overwrites_a_tested_companion():
    """A propagated probably-good (2) cannot downgrade a companion's own bad (4)."""
    data = make_data(DEPTH=[50.0, 50.0], CHLA=[0.5, 200.0])  # second CHLA is out of range
    qc = range_qc(
        data,
        variable_ranges={
            "DEPTH": {3: [0, 5, "inside"], 2: [0, 5, "outside"]},
            "CHLA": {4: [-0.2, 100, "outside"]},
        },
        flag_instead={"DEPTH": ["CHLA"]},
    )

    assert list(qc.return_qc()["CHLA_QC"].values) == [2, 4]


def test_flag_instead_source_without_ranges_raises():
    with pytest.raises(ValueError, match="no entry in variable_ranges"):
        range_qc(
            None,
            variable_ranges={"CHLA": {4: [0, 100]}},
            flag_instead={"DEPTH": ["CHLA"]},
        )


def test_test_depth_range_adds_depth_requirement_and_limits_checks():
    """A depth window adds DEPTH as a requirement and leaves out-of-window points unchecked (0)."""
    qc = range_qc(
        None,
        variable_ranges={"TEMP": {4: [-2.5, 40]}},
        test_depth_range=[-100, 0],
    )
    assert "DEPTH" in qc.required_variables

    data = make_data(TEMP=[100.0, 100.0], DEPTH=[-50.0, -500.0])  # in-window, out-of-window
    qc = range_qc(
        data,
        variable_ranges={"TEMP": {4: [-2.5, 40]}},
        test_depth_range=[-100, 0],
    )
    flags = qc.return_qc()["TEMP_QC"].values
    assert flags[0] == 4  # checked and flagged
    assert flags[1] == 0  # outside depth window -> left unchecked


def test_missing_required_param_raises():
    with pytest.raises(ValueError, match="variable_ranges"):
        range_qc(None, also_flag={})


def test_unknown_param_raises():
    """mode/plot were removed; supplying them is now an unknown-parameter error."""
    with pytest.raises(ValueError, match="mode"):
        range_qc(None, variable_ranges={"TEMP": {4: [0, 30]}}, mode="inside")
