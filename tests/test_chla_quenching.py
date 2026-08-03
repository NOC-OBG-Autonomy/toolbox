"""Tests the step 'CHLA Quenching' (src/pelagos_py/steps/processing/chla_quenching.py).

Covers the newly implemented backscatter- and euphotic-depth-based NPQ methods
and their helpers. The per-profile correction methods are exercised directly
(the solar-elevation lookup is stubbed) so no real data or pvlib call is needed.
"""

#   Test module import
from pelagos_py.steps.processing import chla_quenching

import numpy as np
import xarray as xr
import pytest

Quenching = chla_quenching.chla_quenching_correction


def make_profile(
    chlf, depth, bbp=None, ipar=None, mld=None, zeu=None, z_ipar=None,
    profile_number=101.0, calc_mask=None,
):
    """Single-profile dataset in the step's positive-down DEPTH convention.

    ``run()`` adds a ``*__FOR_CALC`` copy of each input (the same values with
    QC-flagged samples NaN'd out) that the methods derive their quantities from, so
    the profiles here carry them too. ``calc_mask`` is a boolean array marking the
    samples usable for calculation; by default every sample is usable, so the calc
    copies match the raw inputs.

    The euphotic depth (``ZEU``) and iPAR isolume depth (``Z_IPAR``) are now
    per-profile scalars supplied by the 'Interpolate PAR' step rather than derived
    from PAR here, so they are broadcast across the profile (``NaN`` when absent).
    """
    n = len(chlf)
    data = {
        "PROFILE_NUMBER": ("N_MEASUREMENTS", np.full(n, profile_number)),
        "DEPTH": ("N_MEASUREMENTS", np.asarray(depth, dtype=float)),
        "CHLA": ("N_MEASUREMENTS", np.asarray(chlf, dtype=float)),
        "ZEU": ("N_MEASUREMENTS", np.full(n, np.nan if zeu is None else float(zeu))),
        "Z_IPAR": ("N_MEASUREMENTS", np.full(n, np.nan if z_ipar is None else float(z_ipar))),
    }
    if bbp is not None:
        data["BBP700"] = ("N_MEASUREMENTS", np.asarray(bbp, dtype=float))
    if ipar is not None:
        data["DOWNWELLING_PAR"] = ("N_MEASUREMENTS", np.asarray(ipar, dtype=float))
    if mld is not None:
        data["MLD"] = ("N_MEASUREMENTS", np.full(n, float(mld)))

    usable = np.ones(n, dtype=bool) if calc_mask is None else np.asarray(calc_mask, dtype=bool)
    for var in ("CHLA", "BBP700", "DOWNWELLING_PAR"):
        if var in data:
            values = np.asarray(data[var][1], dtype=float)
            data[f"{var}{chla_quenching.CALC_SUFFIX}"] = (
                "N_MEASUREMENTS",
                np.where(usable, values, np.nan),
            )
    return xr.Dataset(data)


def make_step(sun_angle=40.0, hybrid=True):
    """Bare step instance with the sun-elevation lookup stubbed to ``sun_angle``."""
    step = Quenching.__new__(Quenching)
    step.apply_to = "CHLA"
    step.bbp_var = "BBP700"
    step.par_var = "DOWNWELLING_PAR"
    step.hybrid = hybrid
    step._sun_elevation = lambda profile: sun_angle
    return step


# The euphotic depth these profiles imply (Kd=0.12 -> Zeu = 4.605/0.12 ~ 38.4 m);
# it is now supplied as the ZEU scalar rather than derived from PAR in the step.
_ZEU_38 = 38.4


# --- Biermann 2015 ---------------------------------------------------------


def test_biermann_lifts_shallow_to_max_below_zeu():
    z_pos = np.array([2, 6, 10, 15, 20, 30, 45, 60, 80.0])
    chlf = np.array([0.4, 0.5, 0.7, 0.9, 1.0, 0.8, 0.4, 0.2, 0.1])
    ipar = 200 * np.exp(-0.12 * z_pos)  # Zeu ~ 38 m
    prof = make_profile(chlf, z_pos, ipar=ipar, zeu=_ZEU_38)

    out = make_step().apply_biermann2015_quenching_correction(prof)

    # Reference is the max fluorescence within the euphotic zone (1.0 at 20 m);
    # everything shallower than that quenching depth is set to it, deeper is
    # untouched.
    assert np.all(out[:5] == pytest.approx(1.0))
    assert out[5:].tolist() == chlf[5:].tolist()


def test_biermann_flagged_spike_cannot_set_the_reference_but_is_still_corrected():
    """A flagged sample must not become the reference maximum, yet must still be
    lifted to it like any other shallow sample."""
    z_pos = np.array([2, 6, 10, 15, 20, 30, 45, 60, 80.0])
    chlf = np.array([0.4, 9.9, 0.7, 0.9, 1.0, 0.8, 0.4, 0.2, 0.1])
    ipar = 200 * np.exp(-0.12 * z_pos)  # Zeu ~ 38 m

    # The 9.9 spike at 6 m is flagged, so it is unusable for calculation.
    usable = np.ones(z_pos.size, dtype=bool)
    usable[1] = False
    prof = make_profile(chlf, z_pos, ipar=ipar, zeu=_ZEU_38, calc_mask=usable)

    out = make_step().apply_biermann2015_quenching_correction(prof)

    # The reference is the largest *usable* value in the euphotic zone (1.0 at
    # 20 m), not the flagged 9.9 spike.
    assert np.all(out[:5] == pytest.approx(1.0))
    # ...including the flagged sample itself, which is corrected down to it.
    assert out[1] == pytest.approx(1.0)
    assert out[5:].tolist() == chlf[5:].tolist()


def test_biermann_unflagged_spike_does_set_the_reference():
    """Counterpart to the test above: with nothing flagged the spike takes over,
    which is what the calculation mask is preventing."""
    z_pos = np.array([2, 6, 10, 15, 20, 30, 45, 60, 80.0])
    chlf = np.array([0.4, 9.9, 0.7, 0.9, 1.0, 0.8, 0.4, 0.2, 0.1])
    ipar = 200 * np.exp(-0.12 * z_pos)
    prof = make_profile(chlf, z_pos, ipar=ipar, zeu=_ZEU_38)

    out = make_step().apply_biermann2015_quenching_correction(prof)

    assert out[0] == pytest.approx(9.9)


def test_biermann_no_par_signal_returns_unchanged():
    z_pos = np.array([2, 6, 10, 15, 20, 30.0])
    chlf = np.array([0.4, 0.5, 0.7, 0.9, 1.0, 0.8])
    prof = make_profile(chlf, z_pos, ipar=np.full(z_pos.size, np.nan))
    out = make_step().apply_biermann2015_quenching_correction(prof)
    assert np.array_equal(out, chlf)


# --- Xing 2018 (+ Terrats 2020 hybrid) -------------------------------------


def _bbp_profile():
    z_pos = np.array([2, 6, 10, 15, 20, 30, 45, 60, 80.0])
    chlf = np.array([0.4, 0.5, 0.7, 0.9, 1.0, 0.8, 0.4, 0.2, 0.1])
    bbp = np.array([2, 2, 2, 2, 2, 1.5, 1, 0.6, 0.4]) * 1e-3
    return z_pos, chlf, bbp


@pytest.mark.parametrize("hybrid", [True, False])
def test_xing2018_resets_npq_layer_to_bbp_times_rmax(hybrid):
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)  # iPAR=15 ~ 21.6 m, above MLD=25 -> deep
    prof = make_profile(chlf, z_pos, bbp=bbp, ipar=ipar, mld=25.0, z_ipar=21.6)

    out = make_step(hybrid=hybrid).apply_xing2018_quenching_correction(prof)

    # Deep mixing, so both hybrid settings take the Xing 2018 branch. Within the
    # NPQ layer bbp is constant and R_max hits the 1.0 peak, so the suppressed
    # near-surface points are lifted to 1.0; deeper points unchanged.
    assert np.all(out[:5] == pytest.approx(1.0))
    assert out[5:].tolist() == chlf[5:].tolist()


def _shallow_mixing_profile():
    """Shallow mixing: iPAR=15 deep (~80 m) with a shallow MLD of 10 m."""
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 800 * np.exp(-0.05 * z_pos)  # iPAR=15 ~ 79.5 m, below MLD=10 -> shallow
    return chlf, make_profile(chlf, z_pos, bbp=bbp, ipar=ipar, mld=10.0, z_ipar=79.5)


def test_xing2018_hybrid_shallow_mixing_runs_and_never_reduces():
    chlf, prof = _shallow_mixing_profile()

    out = make_step(hybrid=True).apply_xing2018_quenching_correction(prof)

    assert out.shape == chlf.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= chlf - 1e-12)  # correction never reduces fluorescence


def test_xing2018_hybrid_off_uses_xing_branch_when_mixing_is_shallow():
    """With hybrid off, a shallow-mixing profile still gets the Xing 2018 layer."""
    chlf, prof = _shallow_mixing_profile()

    plain = make_step(hybrid=False).apply_xing2018_quenching_correction(prof)
    hybrid = make_step(hybrid=True).apply_xing2018_quenching_correction(prof)

    # Xing's NPQ layer is 0 to min(MLD, iPAR=15 depth) = the 10 m MLD, over which
    # bbp is constant, so the layer is flattened to its own deepest (least
    # quenched) value and nothing below the MLD moves.
    assert plain[:3] == pytest.approx(0.7)
    assert plain[3:].tolist() == chlf[3:].tolist()
    # The hybrid instead de-quenches below the MLD, so the two disagree there -
    # which is the whole point of the parameter.
    assert not np.allclose(plain, hybrid)


@pytest.mark.parametrize("hybrid", [True, False])
def test_backscatter_methods_are_noop_at_night(hybrid):
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)
    prof = make_profile(chlf, z_pos, bbp=bbp, ipar=ipar, mld=25.0, z_ipar=21.6)
    step = make_step(sun_angle=-5.0, hybrid=hybrid)
    out = step.apply_xing2018_quenching_correction(prof)
    assert np.array_equal(out, chlf, equal_nan=True)


# --- Hemsley 2015 ----------------------------------------------------------


def test_hemsley_replaces_euphotic_zone_with_bbp_estimate():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)  # Kd=0.12 -> Zeu ~ 38.4 m
    prof = make_profile(chlf, z_pos, bbp=bbp, ipar=ipar, zeu=_ZEU_38)

    step = make_step()
    step._hemsley_regression = {"slope": 100.0, "intercept": 0.1}
    out = step.apply_hemsley2015_quenching_correction(prof)

    # Over the euphotic zone (z <= ~38 m) fluorescence becomes m*bbp + c;
    # deeper points are untouched.
    assert out[:6] == pytest.approx(100.0 * bbp[:6] + 0.1)
    assert out[6:].tolist() == chlf[6:].tolist()


def test_hemsley_no_regression_returns_unchanged():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)
    prof = make_profile(chlf, z_pos, bbp=bbp, ipar=ipar)
    step = make_step()
    step._hemsley_regression = None
    assert np.array_equal(step.apply_hemsley2015_quenching_correction(prof), chlf)


# --- Thomalla 2017 ---------------------------------------------------------


def test_bin_night_averages_ratio_per_depth_bin():
    z = np.array([0.5, 1.5, 1.6, 2.5])
    fl = np.array([1.0, 2.0, 4.0, 6.0])
    bbp = np.full(4, 1e-3)
    ref = Quenching._bin_night(z, fl, bbp)
    assert ref["z"].tolist() == [0.5, 1.5, 2.5]
    # Middle bin averages fl (3.0) before taking the ratio 3.0 / 1e-3.
    assert ref["ratio"] == pytest.approx([1000.0, 3000.0, 6000.0])


def test_quenching_depth_picks_steepest_gradient_point():
    z = np.array([2, 6, 10, 15, 20, 30.0])
    fl_day = np.array([0.4, 0.5, 0.7, 0.9, 1.0, 0.75])
    fl_night = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.75])  # D returns to 0 by depth
    assert Quenching._quenching_depth(z, fl_day, fl_night, zeu=38.0) == 15.0


def test_thomalla_corrects_above_quenching_depth_and_only_raises():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)  # Zeu ~ 38 m
    prof = make_profile(chlf, z_pos, bbp=bbp, ipar=ipar, zeu=_ZEU_38)

    step = make_step()
    # Night reference: constant fl:bbp ratio so corrected = 500*bbp (=1.0 at
    # surface), and a mean night fluorescence that is unquenched near surface.
    step._night_refs = [{"z": z_pos, "fl": 500.0 * bbp, "ratio": np.full(z_pos.size, 500.0)}]
    step._thomalla_day_night = {101: 0}

    out = step.apply_thomalla2017_quenching_correction(prof)

    # QD resolves to 15 m; surface points are lifted to 500*bbp (1.0) where that
    # exceeds the quenched value, and everything deeper is left unchanged.
    assert out[:4] == pytest.approx(1.0)
    assert out[4:].tolist() == chlf[4:].tolist()


def test_thomalla_unmapped_profile_returns_unchanged():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)
    prof = make_profile(chlf, z_pos, bbp=bbp, ipar=ipar)
    step = make_step()
    step._night_refs = []
    step._thomalla_day_night = {}  # this profile has no paired night
    assert np.array_equal(step.apply_thomalla2017_quenching_correction(prof), chlf)


# --- Sackmann 2008 / Swart 2015 -------------------------------------------


def test_sackmann_resets_layer_to_bbp_times_max_ratio_in_mld():
    z_pos, chlf, bbp = _bbp_profile()
    # Max fl:bbp within the MLD (25 m) is at 20 m (0.4/2e-3 = 500); the layer
    # from the surface to there is reset to bbp*500 (=1.0), deeper is untouched.
    prof = make_profile(chlf, z_pos, bbp=bbp, mld=25.0)

    out = make_step().apply_sackmann2008_quenching_correction(prof)

    assert np.all(out[:5] == pytest.approx(1.0))
    assert out[5:].tolist() == chlf[5:].tolist()


def test_swart_uses_euphotic_zone_window():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)  # Kd=0.12 -> Zeu ~ 38 m
    # Zeu (~38 m) reaches the 30 m point, whose fl:bbp (0.8/1.5e-3 = 533) is the
    # window max; the surface-to-30 m layer is reset to bbp*533, deeper untouched.
    prof = make_profile(chlf, z_pos, bbp=bbp, ipar=ipar, zeu=_ZEU_38)

    out = make_step().apply_swart2015_quenching_correction(prof)

    assert np.all(out[:6] == pytest.approx(bbp[:6] * (0.8 / 1.5e-3)))
    assert out[6:].tolist() == chlf[6:].tolist()


def test_sackmann_never_reduces_fluorescence():
    z_pos, chlf, bbp = _bbp_profile()
    prof = make_profile(chlf, z_pos, bbp=bbp, mld=25.0)
    out = make_step().apply_sackmann2008_quenching_correction(prof)
    assert np.all(out >= chlf - 1e-12)


def test_max_ratio_methods_are_noop_at_night():
    z_pos, chlf, bbp = _bbp_profile()
    prof = make_profile(chlf, z_pos, bbp=bbp, mld=25.0)
    out = make_step(sun_angle=-5.0).apply_sackmann2008_quenching_correction(prof)
    assert np.array_equal(out, chlf, equal_nan=True)


def test_swart_no_par_signal_returns_unchanged():
    z_pos, chlf, bbp = _bbp_profile()
    prof = make_profile(chlf, z_pos, bbp=bbp, ipar=np.full(z_pos.size, np.nan))
    out = make_step().apply_swart2015_quenching_correction(prof)
    assert np.array_equal(out, chlf)


# --- global-disable helpers -------------------------------------------------
# Three profiles; cast 1 & 3 carry PAR, cast 2 (middle) has none.
_DEPTH = np.array([0, 10, 20, 30.0] * 3)
_PNUM = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3.0])
_PAR = np.array([100, 50, 25, 12, np.nan, np.nan, np.nan, np.nan, 200, 100, 50, 25.0])


def _disable_step(scalar_name=None, scalar_values=None):
    """Bare step whose ``self.data`` holds the three-cast fixture above."""
    step = Quenching.__new__(Quenching)
    step.par_var = "DOWNWELLING_PAR"
    step.method = "biermann2015"
    data = {
        "DOWNWELLING_PAR": ("N_MEASUREMENTS", _PAR),
        "DEPTH": ("N_MEASUREMENTS", _DEPTH),
        "PROFILE_NUMBER": ("N_MEASUREMENTS", _PNUM),
    }
    if scalar_name is not None:
        data[scalar_name] = ("N_MEASUREMENTS", np.asarray(scalar_values, dtype=float))
    step.data = xr.Dataset(data)
    return step


def test_count_profiles_without_full_par_counts_the_parless_daytime_cast():
    step = _disable_step()
    assert step._count_profiles_without_full_par([1, 2, 3]) == 1  # cast 2 only
    assert step._count_profiles_without_full_par([1, 3]) == 0  # night cast excluded


def test_count_profiles_without_full_par_all_missing_when_no_par_variable():
    step = _disable_step()
    del step.data["DOWNWELLING_PAR"]
    assert step._count_profiles_without_full_par([1, 2, 3]) == 3


def test_require_scalar_on_days_halts_when_a_daytime_profile_lacks_the_scalar():
    # ZEU present on casts 1 & 3, NaN on cast 2.
    zeu = np.where(_PNUM == 2, np.nan, 40.0)
    step = _disable_step("ZEU", zeu)
    step.halt = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    with pytest.raises(RuntimeError, match="interpolate_zeu"):
        step._require_scalar_on_days("ZEU", "interpolate_zeu", [1, 2, 3])
    # ...but not when the gap is a night profile that no method corrects.
    step._require_scalar_on_days("ZEU", "interpolate_zeu", [1, 3])
