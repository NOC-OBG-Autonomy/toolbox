"""Tests the step 'Interpolate PAR' (src/pelagos_py/steps/processing/interpolate_par.py).

Covers the two per-profile scalars it derives from PAR — the euphotic depth
(ZEU) and the iPAR isolume depth (Z_IPAR) — and the in-time interpolation that
optionally fills them onto profiles without usable PAR.
"""

from pelagos_py.steps.processing import interpolate_par as ipar

import numpy as np
import pytest

estimate_euphotic_depth = ipar.estimate_euphotic_depth
depth_of_ipar = ipar.depth_of_ipar
InterpolatePAR = ipar.InterpolatePAR


# --- estimate_euphotic_depth -----------------------------------------------


def test_estimate_euphotic_depth_recovers_1pct_level():
    """A clean exponential PAR profile yields Zeu = ln(100) / Kd."""
    z = np.arange(0, 60, 2.0)
    par = 100 * np.exp(-0.1 * z)  # Kd = 0.1 -> Zeu = 46.05 m
    assert estimate_euphotic_depth(par, z) == pytest.approx(46.05, abs=0.1)


def test_estimate_euphotic_depth_invalid_inputs_return_nan():
    z = np.arange(0, 60, 2.0)
    assert np.isnan(estimate_euphotic_depth(np.full(z.size, np.nan), z))  # no data
    assert np.isnan(estimate_euphotic_depth(np.full(z.size, 50.0), z))  # flat -> no slope


# --- depth_of_ipar ----------------------------------------------------------


def test_depth_of_ipar_interpolates_and_clamps():
    z = np.array([0, 10, 20, 30.0])
    assert depth_of_ipar(z, np.array([100, 40, 10, 2.0]), 15) == pytest.approx(
        18.333, abs=1e-2
    )
    # Whole profile brighter than the level -> deepest sample; darker -> surface.
    assert depth_of_ipar(z, np.array([100, 90, 80, 70.0]), 15) == 30.0
    assert depth_of_ipar(z, np.array([10, 8, 5, 2.0]), 15) == 0.0


def test_depth_of_ipar_honours_the_level():
    z = np.array([0, 10, 20, 30.0])
    par = np.array([100, 40, 10, 2.0])
    # A brighter isolume sits shallower than a dimmer one on the same profile.
    assert depth_of_ipar(z, par, 40) < depth_of_ipar(z, par, 15)


# --- interpolation ----------------------------------------------------------


def test_interpolate_scalar_fills_interior_and_leaves_ends_nan():
    profiles = np.array([1, 2, 3, 4])
    prof_tsec = {1: 0.0, 2: 10.0, 3: 20.0, 4: 30.0}
    # Computed on profiles 2 and 3 only; 1 (before) and 4 (after) have no anchor.
    calc = {1: np.nan, 2: 40.0, 3: 60.0, 4: np.nan}
    out = InterpolatePAR._interpolate_scalar(calc, profiles, prof_tsec)
    assert np.isnan(out[1])  # no extrapolation before the first computed profile
    assert out[2] == pytest.approx(40.0)
    assert out[3] == pytest.approx(60.0)
    assert np.isnan(out[4])  # no extrapolation past the last computed profile


def test_interpolate_scalar_interpolates_a_gap_in_the_middle():
    profiles = np.array([1, 2, 3])
    prof_tsec = {1: 0.0, 2: 5.0, 3: 10.0}
    calc = {1: 20.0, 2: np.nan, 3: 40.0}  # midpoint in time -> midpoint in value
    out = InterpolatePAR._interpolate_scalar(calc, profiles, prof_tsec)
    assert out[2] == pytest.approx(30.0)


def test_interpolate_scalar_needs_two_anchors():
    profiles = np.array([1, 2])
    prof_tsec = {1: 0.0, 2: 10.0}
    calc = {1: 40.0, 2: np.nan}  # only one computed profile -> nothing to fill
    out = InterpolatePAR._interpolate_scalar(calc, profiles, prof_tsec)
    assert np.isnan(out[2])
