"""Tests the steps 'BBP from Beta' and 'Isolate BBP Spikes' (src/pelagos_py/steps/processing/bbp.py)."""

#   Test module import
from pelagos_py.steps.processing import bbp

import numpy as np
import pytest
import xarray as xr

BBPFromBeta = bbp.BBPFromBeta
IsolateBBPSpikes = bbp.IsolateBBPSpikes


def make_beta_context(temp=None, depth=None, flags=None, n=8):
    """Minimal context for BBP from Beta, with all inputs good unless overridden."""
    ones = np.ones(n)
    data = {
        "TIME": (
            "N_MEASUREMENTS",
            np.arange(n).astype("datetime64[s]").astype("datetime64[ns]"),
        ),
        "PROFILE_NUMBER": ("N_MEASUREMENTS", np.zeros(n)),
        "DEPTH": ("N_MEASUREMENTS", ones * 10 if depth is None else np.asarray(depth, dtype=float)),
        "TEMP": ("N_MEASUREMENTS", ones * 10 if temp is None else np.asarray(temp, dtype=float)),
        "PRAC_SALINITY": ("N_MEASUREMENTS", ones * 35),
        "BBP700": ("N_MEASUREMENTS", ones * 2e-4),
    }
    ds = xr.Dataset(data)
    for var in ("BBP700", "DEPTH", "TEMP", "PRAC_SALINITY"):
        ds[f"{var}_QC"] = ("N_MEASUREMENTS", np.zeros(n, dtype=int))
    for var, var_flags in (flags or {}).items():
        ds[f"{var}_QC"] = ("N_MEASUREMENTS", np.asarray(var_flags, dtype=int))
    return {"data": ds, "global_parameters": {}}


def make_beta_step(context, parameters=None):
    return BBPFromBeta(
        name="BBP from Beta",
        parameters={"apply_to": "BBP700", "output_as": "BBP700_OUT", **(parameters or {})},
        diagnostics=False,
        context=context,
    )


# --- BBP from Beta ----------------------------------------------------------


def test_gaps_in_inputs_are_left_alone_and_bbp_is_not_derived_there():
    """The step must not gap-fill its inputs: filling is Interpolate Data's job.

    A NaN TEMP simply yields no BBP at that sample, flagged missing (9).
    """
    ctx = make_beta_context(temp=[10, 10, np.nan, 10, 10, 10, 10, 10])
    out = make_beta_step(ctx).run()["data"]

    # The input keeps its gap - the step used to silently interpolate it away.
    assert np.isnan(out["TEMP"].values[2])
    assert np.isnan(out["BBP700_OUT"].values[2])
    assert out["BBP700_OUT_QC"].values[2] == 9
    # Every other sample is derived normally.
    assert np.isfinite(out["BBP700_OUT"].values[[0, 1, 3, 4, 5, 6, 7]]).all()


def test_depth_is_never_modified():
    """DEPTH is not an input to the BBP formula, so the step must leave it untouched."""
    depth = [1, 2, np.nan, 4, 5, 6, 7, 8]
    ctx = make_beta_context(depth=depth)
    before = ctx["data"]["DEPTH"].values.copy()

    out = make_beta_step(ctx).run()["data"]

    assert np.array_equal(out["DEPTH"].values, before, equal_nan=True)


# --- Isolate BBP Spikes -----------------------------------------------------


def make_spikes_context(values, flags=None, n=None):
    n = len(values)
    ds = xr.Dataset(
        {
            "TIME": (
                "N_MEASUREMENTS",
                np.arange(n).astype("datetime64[s]").astype("datetime64[ns]"),
            ),
            "BBP700": ("N_MEASUREMENTS", np.asarray(values, dtype=float)),
            "BBP700_QC": (
                "N_MEASUREMENTS",
                np.zeros(n, dtype=int) if flags is None else np.asarray(flags, dtype=int),
            ),
        }
    )
    return {"data": ds, "global_parameters": {}}


def make_spikes_step(context, window_size=5):
    return IsolateBBPSpikes(
        name="Isolate BBP Spikes",
        parameters={"apply_to": "BBP700", "window_size": window_size, "method": "median"},
        diagnostics=False,
        context=context,
    )


def _bad_stretch_with_one_good_sample():
    """A bad stretch (e.g. biofouling) with one good reading surviving inside it.

    This is the case a rolling median is *not* robust to: the lone good sample's
    window is mostly bad, so the median reports the bad level for it. An isolated
    outlier, by contrast, is already rejected by the median with or without a mask.

    Returns (values, flags, good_idx).
    """
    n = 40
    values = np.full(n, 99.0)
    values[:15] = 1.0
    values[25:] = 1.0
    values[20] = 1.0  # the lone good reading inside the bad stretch

    flags = np.zeros(n, dtype=int)
    flags[15:25] = 4
    flags[20] = 1
    return values, flags, 20


def test_flagged_stretch_cannot_drag_a_good_samples_baseline():
    """A good reading inside a flagged bad stretch must get its own baseline, not
    the bad stretch's."""
    values, flags, good = _bad_stretch_with_one_good_sample()

    out = make_spikes_step(make_spikes_context(values, flags)).run()["data"]
    baseline = out["BBP700_BASELINE"].values

    # The lone good sample's baseline reflects its own level, not the bad stretch.
    assert baseline[good] == pytest.approx(1.0)
    # No baseline is derived across the flagged stretch, which is flagged missing.
    assert np.isnan(baseline[15:20]).all()
    assert out["BBP700_BASELINE_QC"].values[16] == 9


def test_unflagged_stretch_does_drag_the_baseline():
    """Counterpart: unflagged, the bad stretch takes over the good sample's rolling
    median entirely - which is what the calculation mask prevents above."""
    values, _, good = _bad_stretch_with_one_good_sample()

    out = make_spikes_step(make_spikes_context(values)).run()["data"]

    assert out["BBP700_BASELINE"].values[good] == pytest.approx(99.0)
