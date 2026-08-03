"""Tests the step 'Deep Correction' (src/pelagos_py/steps/processing/deep_correction.py)."""

from pelagos_py.steps.processing import deep_correction

import logging
import numpy as np
import xarray as xr
import pytest

DeepCorrection = deep_correction.deep_correction
MIN_DEEP_THRESHOLD = deep_correction.MIN_DEEP_THRESHOLD


def make_context(
    dark=0.7, n_profiles=6, max_pres=1000.0, bloom=3.0, seed=0, var="CHLA", extra=None
):
    # Each profile is a near-surface bloom on a constant deep `dark` offset, so the correct dark value is known.
    rng = np.random.default_rng(seed)
    pid, pres, vals = [], [], []
    for p in range(n_profiles):
        d = np.linspace(0, max_pres, 200)
        signal = bloom * np.exp(-((d - 30) ** 2) / 500)
        v = dark + signal + rng.normal(0, 0.01, d.size)
        pid += [p] * d.size
        pres += list(d)
        vals += list(v)

    n = len(pres)
    ds = xr.Dataset(
        {
            "PRES": ("N_MEASUREMENTS", np.array(pres)),
            var: ("N_MEASUREMENTS", np.array(vals)),
            f"{var}_QC": ("N_MEASUREMENTS", np.zeros(n, dtype=int)),
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.array(pid, dtype=float)),
        }
    )
    if extra:
        for name, values in extra.items():
            ds[name] = ("N_MEASUREMENTS", np.asarray(values, dtype=float))
    return {"data": ds, "global_parameters": {}}


def make_step(parameters, context, diagnostics=False):
    return DeepCorrection(
        name="Deep Correction",
        parameters=parameters,
        diagnostics=diagnostics,
        context=context,
    )


def test_computes_dark_value_and_creates_adjusted():
    ctx = make_context(dark=0.7)
    step = make_step({"apply_to": "CHLA"}, ctx)

    out = step.run()
    data = out["data"]

    assert step.dark_value == pytest.approx(0.7, abs=0.1)
    assert "CHLA_ADJUSTED" in data
    assert "CHLA_ADJUSTED_QC" in data
    # Deep, signal-free values should straddle zero after correction.
    deep = data["CHLA_ADJUSTED"].where(data["PRES"] > 950)
    assert float(deep.median()) == pytest.approx(0.0, abs=0.1)
    # The correction is a constant offset of the whole record.
    assert np.allclose(
        data["CHLA_ADJUSTED"].values, data["CHLA"].values - step.dark_value, equal_nan=True
    )


def _corrupt_deep(data, profiles):
    # Wreck the deep, signal-free region of `profiles`, flag it bad (4), return the corrupted mask.
    bad = np.isin(data["PROFILE_NUMBER"].values, profiles) & (data["PRES"].values > 950)
    assert bad.any()
    data["CHLA"].values[bad] = -20.0
    data["CHLA_QC"].values[bad] = 4
    return bad


def test_flagged_values_are_excluded_from_the_dark_value_but_still_corrected():
    ctx = make_context(dark=0.7)
    data = ctx["data"]

    # The dark value is a median of per-profile minima, so the corruption must outnumber the clean profiles to be observable.
    bad = _corrupt_deep(data, profiles=(0, 1, 2, 3))

    step = make_step({"apply_to": "CHLA"}, ctx)
    out = step.run()

    # The dark value comes from the two clean profiles; the -20 values would otherwise supply four minima and take over the median.
    assert step.dark_value == pytest.approx(0.7, abs=0.1)
    # Every sample is still corrected, flagged ones included.
    assert np.allclose(
        out["data"]["CHLA_ADJUSTED"].values,
        out["data"]["CHLA"].values - step.dark_value,
        equal_nan=True,
    )
    assert np.isfinite(out["data"]["CHLA_ADJUSTED"].values[bad]).all()


def test_calculation_flag_filter_can_be_overridden():
    ctx = make_context(dark=0.7)
    _corrupt_deep(ctx["data"], profiles=(0, 1, 2, 3))

    step = make_step(
        {"apply_to": "CHLA", "qc_handling_settings": {"calculation_flag_filter": []}},
        ctx,
    )
    step.run()

    # With nothing excluded the corrupted profiles supply four -20 minima, which take over the median and wreck the dark value.
    assert step.dark_value == pytest.approx(-20.0, abs=0.5)


def test_uses_config_dark_value_directly():
    ctx = make_context(dark=0.7)
    step = make_step({"apply_to": "CHLA", "dark_value": 0.5}, ctx)

    out = step.run()
    data = out["data"]

    assert step.dark_value == 0.5
    assert step._profile_diagnostics == {}
    assert np.allclose(
        data["CHLA_ADJUSTED"].values, data["CHLA"].values - 0.5, equal_nan=True
    )


def test_existing_adjusted_is_corrected_in_place():
    ctx = make_context(dark=0.7)
    # Seed an existing adjusted variable (+ its QC) to be picked up.
    ctx["data"]["CHLA_ADJUSTED"] = ctx["data"]["CHLA"].copy()
    ctx["data"]["CHLA_ADJUSTED_QC"] = ctx["data"]["CHLA_QC"].copy()

    step = make_step({"apply_to": "CHLA"}, ctx)
    out = step.run()

    assert step.apply_to == "CHLA_ADJUSTED"
    assert step.output_as == "CHLA_ADJUSTED"
    deep = out["data"]["CHLA_ADJUSTED"].where(out["data"]["PRES"] > 950)
    assert float(deep.median()) == pytest.approx(0.0, abs=0.1)


def test_configurable_depth_var():
    ctx = make_context(dark=0.7)
    # Mirror PRES into a DEPTH variable so the deep threshold can key off it.
    ctx["data"]["DEPTH"] = ctx["data"]["PRES"].copy()

    step = make_step({"apply_to": "CHLA", "depth_var": "DEPTH"}, ctx)
    step.run()

    assert step.dark_value == pytest.approx(0.7, abs=0.1)


def test_warns_when_threshold_too_shallow(caplog):
    ctx = make_context(dark=0.7)
    step = make_step({"apply_to": "CHLA", "depth_threshold": 200}, ctx)

    with caplog.at_level(logging.WARNING):
        step.run()

    assert 200 < MIN_DEEP_THRESHOLD
    assert "shallower" in caplog.text


def test_no_warning_at_default_threshold(caplog):
    ctx = make_context(dark=0.7)
    step = make_step({"apply_to": "CHLA"}, ctx)

    with caplog.at_level(logging.WARNING):
        step.run()

    assert "shallower" not in caplog.text


def test_raises_when_no_profiles_reach_threshold():
    ctx = make_context(dark=0.7, max_pres=500.0)  # never exceeds the 950 default
    step = make_step({"apply_to": "CHLA"}, ctx)

    with pytest.raises(ValueError, match="No profiles reach past the depth threshold"):
        step.run()


def test_raises_when_apply_to_missing():
    ctx = make_context(dark=0.7)
    step = make_step({"apply_to": "TEMP"}, ctx)

    with pytest.raises(KeyError, match="does not exist in the data"):
        step.run()


def test_sparse_deep_profile_is_skipped_not_selected():
    ctx = make_context(dark=0.7, n_profiles=6)
    ds = ctx["data"]
    # Blank profile 0's deep values (still reaches past the threshold but has no data there) so it must be skipped for a later profile.
    deep_of_zero = (ds["PROFILE_NUMBER"] == 0) & (ds["PRES"] > 950)
    ds["CHLA"] = ds["CHLA"].where(~deep_of_zero)

    step = make_step({"apply_to": "CHLA"}, ctx)
    step.run()

    assert 0 not in step._profile_diagnostics
    assert len(step._profile_diagnostics) == 5
    assert step.dark_value == pytest.approx(0.7, abs=0.1)


def test_raises_when_too_few_valid_deep_points():
    ctx = make_context(dark=0.7)
    # min_valid_points impossibly high => no profile qualifies.
    step = make_step({"apply_to": "CHLA", "min_valid_points": 10_000}, ctx)

    with pytest.raises(ValueError, match="valid deep points"):
        step.run()
