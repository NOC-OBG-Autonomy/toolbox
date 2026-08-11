"""Tests the step 'Shift Oxygen To CTD' (src/pelagos_py/steps/processing/oxygen.py)."""

from pelagos_py.steps.processing import oxygen

import numpy as np
import xarray as xr
import pytest

ShiftOxygenToCTD = oxygen.ShiftOxygenToCTD


def make_context(n=50, dt_s=1.0, with_pitch=True, with_gradient=True):
    times = np.datetime64("2024-01-01T00:00:00") + (np.arange(n) * dt_s * 1e9).astype(
        "timedelta64[ns]"
    )
    epoch_s = times.astype("datetime64[ns]").astype("int64") / 1e9

    ds = xr.Dataset(
        {
            "TIME": ("N_MEASUREMENTS", times),
            # Identity signal (value == its own epoch time) makes a correct shift easy to check.
            "UNCAL_PHASE_DOXY": ("N_MEASUREMENTS", epoch_s.copy()),
            "UNCAL_PHASE_DOXY_QC": ("N_MEASUREMENTS", np.zeros(n, dtype=int)),
            "PROFILE_NUMBER": (
                "N_MEASUREMENTS",
                np.repeat(np.arange(n // 10 + 1), 10)[:n].astype(float),
            ),
            "PROFILE_DIRECTION": ("N_MEASUREMENTS", np.resize([1.0, -1.0], n)),
        }
    )
    if with_pitch:
        ds["GLIDER_PITCH"] = ("N_MEASUREMENTS", np.full(n, np.deg2rad(30)))
    if with_gradient:
        ds["PROFILE_GRADIENT"] = ("N_MEASUREMENTS", np.full(n, 0.1))

    return {"data": ds, "global_parameters": {}}


def make_step(parameters, context, diagnostics=False):
    return ShiftOxygenToCTD(
        name="Shift Oxygen To CTD", parameters=parameters, diagnostics=diagnostics, context=context
    )


def test_constant_lag_shifts_values_forward_in_time():
    ctx = make_context(n=50, dt_s=1.0)
    step = make_step({"shift_vars": ["UNCAL_PHASE_DOXY"], "lag_seconds": 3.0}, ctx)

    out = step.run()
    data = out["data"]

    epoch_s = data["TIME"].values.astype("datetime64[ns]").astype("int64") / 1e9
    expected = epoch_s + 3.0

    # Interior points land exactly on source samples (identity signal); the last few
    # query points fall past the end of the source data and become NaN.
    assert np.allclose(data["UNCAL_PHASE_DOXY_SHIFTED"].values[:-3], expected[:-3])
    assert np.isnan(data["UNCAL_PHASE_DOXY_SHIFTED"].values[-3:]).all()
    assert "UNCAL_PHASE_DOXY_SHIFTED_QC" in data


def test_missing_pitch_raises_when_no_constant_lag_given():
    ctx = make_context(with_pitch=False)
    step = make_step({"shift_vars": ["UNCAL_PHASE_DOXY"]}, ctx)

    with pytest.raises(KeyError):
        step.run()


def test_custom_pitch_name_is_used_for_dynamic_lag():
    ctx = make_context(with_pitch=False)
    n = ctx["data"].sizes["N_MEASUREMENTS"]
    ctx["data"]["MY_PITCH"] = ("N_MEASUREMENTS", np.full(n, np.deg2rad(30)))

    step = make_step({"shift_vars": ["UNCAL_PHASE_DOXY"], "pitch_name": "MY_PITCH"}, ctx)
    out = step.run()

    assert "UNCAL_PHASE_DOXY_SHIFTED" in out["data"]


def test_missing_shift_var_raises():
    ctx = make_context()
    step = make_step({"shift_vars": ["DOES_NOT_EXIST"], "lag_seconds": 1.0}, ctx)

    with pytest.raises(KeyError):
        step.run()
