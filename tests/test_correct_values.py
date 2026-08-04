"""Tests the step 'Correct Values' (src/pelagos_py/steps/input_output/correct_values.py)."""

#   Test module import
from pelagos_py.steps.input_output import correct_values

import numpy as np
import xarray as xr
import pytest

CorrectValues = correct_values.CorrectValues


def make_context(values, var="CNDC", units=None, times=None):
    """Wrap a 1-D array in a minimal pipeline context, like the pipeline passes in."""
    ds = xr.Dataset({var: ("N_MEASUREMENTS", np.asarray(values, dtype=float))})
    if units is not None:
        ds[var].attrs["units"] = units
    if times is not None:
        ds["TIME"] = ("N_MEASUREMENTS", np.asarray(times, dtype="datetime64[ns]"))
    return {"data": ds, "global_parameters": {}}


def make_step(parameters, context):
    return CorrectValues(name="correct_values", parameters=parameters, context=context)


def test_scales_when_outside_expected_range():
    """CNDC in S/m (median ~3.5) is outside [20, 45], so the x10 correction is applied."""
    ctx = make_context(np.full(20, 3.5), units="S/m")
    step = make_step(
        {"target_variable": "CNDC", "slope": 10.0, "expected_range": [20, 45], "corrected_units": "mS/cm"},
        ctx,
    )

    out = step.run()

    assert np.allclose(out["data"]["CNDC"].values, 35.0)
    assert out["data"]["CNDC"].attrs["units"] == "mS/cm"
    assert step.applied is True


def test_skips_when_inside_expected_range():
    """Data already in mS/cm (median 35, inside [20, 45]) is left untouched."""
    ctx = make_context(np.full(20, 35.0), units="mS/cm")
    step = make_step(
        {"target_variable": "CNDC", "slope": 10.0, "expected_range": [20, 45], "corrected_units": "mS/cm"},
        ctx,
    )

    out = step.run()

    assert np.allclose(out["data"]["CNDC"].values, 35.0)
    assert step.applied is False


def test_always_applies_without_expected_range():
    """With no expected_range the affine correction (slope + intercept) is always applied."""
    ctx = make_context(np.full(20, 2.0), var="X")
    step = make_step({"target_variable": "X", "slope": 2.0, "intercept": 1.0}, ctx)

    out = step.run()

    assert np.allclose(out["data"]["X"].values, 5.0)  # 2*2 + 1
    assert step.applied is True


def test_units_untouched_when_not_specified():
    """Omitting corrected_units leaves the existing units attribute as it was."""
    ctx = make_context(np.full(20, 3.5), units="S/m")
    step = make_step({"target_variable": "CNDC", "slope": 10.0}, ctx)

    out = step.run()

    assert out["data"]["CNDC"].attrs["units"] == "S/m"


def test_nans_are_preserved():
    """NaNs propagate through the correction rather than becoming real numbers."""
    vals = np.array([3.5, np.nan, 3.5, 3.5])
    ctx = make_context(vals)
    step = make_step({"target_variable": "CNDC", "slope": 10.0}, ctx)

    out = step.run()
    result = out["data"]["CNDC"].values

    assert np.isnan(result[1])
    assert np.allclose(result[[0, 2, 3]], 35.0)


def test_missing_target_variable_raises():
    """A target_variable absent from the dataset raises a clear error."""
    ctx = make_context(np.full(5, 3.5), var="CNDC")
    step = make_step({"target_variable": "TEMP", "slope": 10.0}, ctx)

    with pytest.raises(ValueError, match="not found in dataset"):
        step.run()


def test_output_as_renames_leaving_source_intact():
    ctx = make_context(np.full(5, 51.2), var="LATITUDE_GPS")
    step = make_step({"target_variable": "LATITUDE_GPS", "output_as": "LATITUDE", "slope": 1.0}, ctx)

    out = step.run()

    assert np.allclose(out["data"]["LATITUDE"].values, 51.2)
    assert "LATITUDE_GPS" in out["data"]  # source left in place (copy, not move)


def test_output_as_scales_into_new_variable():
    ctx = make_context(np.full(5, 3.5), var="CNDC")
    step = make_step({"target_variable": "CNDC", "output_as": "CNDC_SCALED", "slope": 10.0}, ctx)

    out = step.run()

    assert np.allclose(out["data"]["CNDC_SCALED"].values, 35.0)
    assert np.allclose(out["data"]["CNDC"].values, 3.5)  # source unchanged


def test_time_window_limits_correction():
    times = np.array(["2024-08-01", "2024-08-02", "2024-08-03", "2024-08-04"], dtype="datetime64[ns]")
    ctx = make_context(np.full(4, 2.0), var="X", times=times)
    step = make_step(
        {"target_variable": "X", "slope": 10.0, "time_start": "2024-08-03"}, ctx
    )

    out = step.run()
    result = out["data"]["X"].values

    assert np.allclose(result[:2], 2.0)   # before window: untouched
    assert np.allclose(result[2:], 20.0)  # in window: scaled


def test_append_description_adds_to_existing_comment():
    ctx = make_context(np.full(3, 1.0), var="X")
    ctx["data"]["X"].attrs["comment"] = "original"
    step = make_step({"target_variable": "X", "append_description": "rescaled x10", "slope": 10.0}, ctx)

    out = step.run()

    assert out["data"]["X"].attrs["comment"] == "original rescaled x10"


def test_overwrite_description_replaces_comment():
    ctx = make_context(np.full(3, 1.0), var="X")
    ctx["data"]["X"].attrs["comment"] = "original"
    step = make_step({"target_variable": "X", "overwrite_description": "new note", "slope": 1.0}, ctx)

    out = step.run()

    assert out["data"]["X"].attrs["comment"] == "new note"


def test_both_descriptions_raises():
    ctx = make_context(np.full(3, 1.0), var="X")
    step = make_step(
        {"target_variable": "X", "append_description": "a", "overwrite_description": "b"}, ctx
    )

    with pytest.raises(ValueError, match="only one of"):
        step.run()
