"""Tests the step 'Salinity Adjustment' (src/pelagos_py/steps/processing/salinity.py).

Focused on QC handling: flagged samples must not inform the estimate or anchor the
interpolants, while still receiving the corrections themselves.
"""

from pelagos_py.steps.processing.salinity import AdjustSalinity

import numpy as np
import pytest
import xarray as xr

#: Sized so profiles span over an hour with > 3 * filter_window_size samples, which correct_ct_lag requires.
SAMPLES_PER_PROFILE = 1200
SAMPLE_SECONDS = 4.0
FILTER_WINDOW = 21

#: The flagged stretch used throughout, and the good samples either side of it.
BAD_BLOCK = slice(400, 460)
GOOD = np.r_[0:400, 460:SAMPLES_PER_PROFILE]


def make_context(bad_block=None, n_profiles=2):
    # Two smooth CTD profiles, optionally with a wild, QC-flagged TEMP stretch.
    t0 = np.datetime64("2024-01-01T00:00:00")
    times, prof, temp, cndc, pres = [], [], [], [], []
    for p in range(n_profiles):
        elapsed = np.arange(SAMPLES_PER_PROFILE) * SAMPLE_SECONDS
        times.append(t0 + (p * 6000 + elapsed).astype("timedelta64[s]"))
        prof.append(np.full(SAMPLES_PER_PROFILE, float(p)))
        depth = np.linspace(0, 200, SAMPLES_PER_PROFILE)
        pres.append(depth)
        temp.append(15 - 8 * (depth / 200))  # a smooth thermocline
        cndc.append(35 + 0.5 * np.sin(depth / 20))

    ds = xr.Dataset(
        {
            "TIME": ("N_MEASUREMENTS", np.concatenate(times)),
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.concatenate(prof)),
            "PRES": ("N_MEASUREMENTS", np.concatenate(pres)),
            "TEMP": ("N_MEASUREMENTS", np.concatenate(temp)),
            "CNDC": ("N_MEASUREMENTS", np.concatenate(cndc)),
        }
    )
    n = ds.sizes["N_MEASUREMENTS"]
    ds["CNDC"].attrs["units"] = "mS/cm"  # fixture values are mS/cm magnitude
    for var in ("TEMP", "CNDC", "PRES"):
        ds[f"{var}_QC"] = ("N_MEASUREMENTS", np.ones(n, dtype=int))

    if bad_block is not None:
        ds["TEMP"].values[bad_block] = 99.0  # a wild reading...
        ds["TEMP_QC"].values[bad_block] = 4  # ...that QC already knows is bad

    return {"data": ds, "global_parameters": {}}


def run_step(context, parameters=None):
    step = AdjustSalinity(
        name="Salinity Adjustment",
        parameters={"filter_window_size": FILTER_WINDOW, **(parameters or {})},
        diagnostics=False,
        context=context,
    )
    return step.run()["data"]


def test_flagged_stretch_does_not_affect_its_neighbours_correction():
    clean = run_step(make_context())["TEMP"].values
    flagged = run_step(make_context(bad_block=BAD_BLOCK))["TEMP"].values

    assert np.allclose(clean[GOOD], flagged[GOOD], atol=1e-9)


def test_flagged_samples_are_still_corrected():
    flagged = run_step(make_context(bad_block=BAD_BLOCK))["TEMP"].values

    assert np.isfinite(flagged[BAD_BLOCK]).all()


def test_without_the_filter_the_flagged_stretch_poisons_its_neighbours():
    # Counterpart: an empty calculation_flag_filter lets the bad stretch anchor the interpolants, which the default prevents above.
    clean = run_step(make_context())["TEMP"].values
    unmasked = run_step(
        make_context(bad_block=BAD_BLOCK),
        {"qc_handling_settings": {"calculation_flag_filter": []}},
    )["TEMP"].values

    # The good samples are dragged well away from their true correction.
    assert np.nanmax(np.abs(clean[GOOD] - unmasked[GOOD])) > 1.0


def test_clean_data_is_unaffected_by_the_filter():
    default = run_step(make_context())["TEMP"].values
    disabled = run_step(
        make_context(), {"qc_handling_settings": {"calculation_flag_filter": []}}
    )["TEMP"].values

    assert np.allclose(default, disabled, atol=1e-12, equal_nan=True)
