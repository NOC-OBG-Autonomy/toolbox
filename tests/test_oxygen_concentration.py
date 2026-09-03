"""Tests the step 'Derive Oxygen Concentration' (src/pelagos_py/steps/processing/oxygen.py)."""

from pelagos_py.steps.processing import oxygen

import numpy as np
import xarray as xr

DeriveOxygenConcentration = oxygen.DeriveOxygenConcentration


def make_context(temp, phase):
    n = len(temp)
    ds = xr.Dataset(
        {
            "TEMP_DOXY": ("N_MEASUREMENTS", np.asarray(temp, dtype=float)),
            "TEMP_DOXY_QC": ("N_MEASUREMENTS", np.zeros(n, dtype=int)),
            "CAL_PHASE_DOXY": ("N_MEASUREMENTS", np.asarray(phase, dtype=float)),
            "CAL_PHASE_DOXY_QC": ("N_MEASUREMENTS", np.zeros(n, dtype=int)),
        }
    )
    return {"data": ds, "global_parameters": {}}


def make_step(parameters, context, diagnostics=False):
    return DeriveOxygenConcentration(
        name="Derive Oxygen Concentration", parameters=parameters, diagnostics=diagnostics, context=context
    )


def test_svu_matches_stern_volmer_uchida_equation():
    """The 'SVU' method must reproduce (P0/Pc - 1) / K_SV directly from SVUFoilCoef0-6."""
    c0, c1, c2, c3, c4, c5, c6 = 2.67915e-03, 1.12428e-04, 2.29021e-06, 1.43603e02, -2.08012e-01, -3.59303e01, 2.82344e00
    temp = [10.0, 15.0, 20.0]
    phase = [25.0, 27.0, 29.0]

    ctx = make_context(temp, phase)
    step = make_step(
        {
            "method": "SVU",
            "temperature_name": "TEMP_DOXY",
            "svu_coefficients": [c0, c1, c2, c3, c4, c5, c6],
        },
        ctx,
    )
    out = step.run()

    T = np.asarray(temp)
    P = np.asarray(phase)
    K_SV = c0 + c1 * T + c2 * T**2
    P0 = c3 + c4 * T
    Pc = c5 + c6 * P
    expected = (P0 / Pc - 1.0) / K_SV

    assert np.allclose(out["data"]["MOLAR_DOXY"].values, expected)
