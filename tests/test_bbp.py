"""Behavioural tests for the 'BBP from Beta' step
(src/pelagos_py/steps/processing/bbp.py)."""

import numpy as np
import xarray as xr
import glidertools as gt

from pelagos_py.steps.processing.bbp import BBPFromBeta

THETA = 124.0
XFACTOR = 1.076
WAVELENGTH = 700


def make_beta_context(beta, temp=10.0, psal=35.0):
    """Build a minimal context for BBP from Beta from a 1-D beta array."""
    beta = np.asarray(beta, dtype=float)
    n = len(beta)
    ds = xr.Dataset(
        {
            "TIME": (
                "N_MEASUREMENTS",
                np.arange(n).astype("datetime64[s]").astype("datetime64[ns]"),
            ),
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.ones(n, dtype=int)),
            "DEPTH": ("N_MEASUREMENTS", np.linspace(1.0, float(n), n)),
            "TEMP": ("N_MEASUREMENTS", np.full(n, float(temp))),
            "PRAC_SALINITY": ("N_MEASUREMENTS", np.full(n, float(psal))),
            "BETA700": ("N_MEASUREMENTS", beta),
            "BETA700_QC": ("N_MEASUREMENTS", np.ones(n, dtype=int)),
        }
    )
    return {"data": ds, "global_parameters": {}}


def run_step(ctx):
    step = BBPFromBeta(
        name="BBP from Beta",
        parameters={
            "apply_to": "BETA700",
            "output_as": "BBP700",
            "theta": THETA,
            "xfactor": XFACTOR,
        },
        context=ctx,
    )
    return step.run()["data"]


def test_pure_seawater_beta_gives_zero_bbp():
    """The headline invariant: water with no particles must give BBP == 0.

    BBP is the PARTICULATE backscattering coefficient. If beta is exactly the
    theoretical seawater-only volume scattering function, there are no particles,
    so BBP must be zero. Regression test for the bug where the step returned the
    TOTAL coefficient (particulate + seawater) because glidertools'
    flo_bback_total adds the seawater backscattering term bsw/2 back in, which
    left a spurious offset of ~3.2e-4 m-1 here instead of 0.
    """
    temp, psal = 10.0, 35.0
    betasw, _bsw = gt.flo_functions.flo_zhang_scatter_coeffs(
        np.array([temp]), np.array([psal]), THETA, WAVELENGTH
    )

    out = run_step(make_beta_context([betasw[0]] * 3, temp=temp, psal=psal))

    assert np.allclose(out["BBP700"].values, 0.0, atol=1e-12)


def test_bbp_matches_the_bgc_argo_definition():
    """BBP = 2 * pi * chi * (beta - beta_sw), with no seawater term added back."""
    temp, psal = 10.0, 35.0
    betasw, _bsw = gt.flo_functions.flo_zhang_scatter_coeffs(
        np.array([temp]), np.array([psal]), THETA, WAVELENGTH
    )
    beta = betasw[0] + np.array([1.0e-5, 5.0e-5, 2.0e-4])
    expected = XFACTOR * 2.0 * np.pi * (beta - betasw[0])

    out = run_step(make_beta_context(beta, temp=temp, psal=psal))

    assert np.allclose(out["BBP700"].values, expected, rtol=1e-10)


def test_no_seawater_backscatter_offset_remains():
    """Guard the specific regression: output must not be high by bsw/2.

    flo_bback_total returns bbackp + bsw/2. This asserts we are not returning
    that quantity, which at 700 nm would inflate every value by ~3.2e-4 m-1 --
    more than the entire particulate signal in deep or oligotrophic water.
    """
    temp, psal = 10.0, 35.0
    betasw, bsw = gt.flo_functions.flo_zhang_scatter_coeffs(
        np.array([temp]), np.array([psal]), THETA, WAVELENGTH
    )
    beta = np.array([betasw[0] + 1.0e-5])

    out = run_step(make_beta_context(beta, temp=temp, psal=psal))
    total = gt.flo_functions.flo_bback_total(
        beta, np.array([temp]), np.array([psal]), THETA, WAVELENGTH, XFACTOR
    )

    # The step must return the particulate coefficient, not the total one.
    assert not np.allclose(out["BBP700"].values, total)
    assert np.allclose(out["BBP700"].values, total - bsw[0] / 2, rtol=1e-10)
