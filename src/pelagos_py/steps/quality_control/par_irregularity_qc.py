# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
QC tests to identify irregularities in PAR profiles based on La Forgia & Organelli (2025).
* Shapiro–Wilk test
* Day and night sequences
"""

#### Mandatory imports ####
from IPython.core.pylabtools import figsize
from pelagos_py.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import shapiro
from scipy.interpolate import interp1d
from datetime import datetime
import warnings
import xarray as xr
import pandas as pd
import pvlib


# Functions written and provided by Thomas Ryan-Keogh based off of (https://doi.org/10.1002/lom3.10701)
def calculate_solar_elevation(latitude, longitude, datetime):
    """
    Calculate the solar elevation angle for given geographic coordinates and timestamps.

    Parameters
    ----------
    latitude : array-like or float
        Latitude(s) of the observation point(s) in decimal degrees. Positive north.
    longitude : array-like or float
        Longitude(s) of the observation point(s) in decimal degrees. Positive east.
    datetime : array-like or pandas.DatetimeIndex
        Datetime(s) of observation. Can be any format convertible by `pandas.to_datetime`.

    Returns
    -------
    np.ndarray
        Solar elevation angle(s) in degrees, corresponding to each input coordinate/time.

    Notes
    -----
    - The calculation uses `pvlib.solarposition.get_solarposition` with times localized to UTC.
    - Intended for use with xarray datasets where each profile has a single latitude,
      longitude, and datetime (e.g., one value per `N_PROF`).
    """
    # Ensure datetime is timezone-aware (UTC)
    time_utc = pd.to_datetime(datetime).tz_localize("UTC")

    # Compute solar position
    solar_position = pvlib.solarposition.get_solarposition(
        time_utc, latitude, longitude
    )

    return solar_position["elevation"].values


def qc_par_flagging(pres, par, sun_elev, nei_par=3e-2):
    """
    Real-time quality control (RT-QC) for PAR profiles
    following La Forgia & Organelli (2025, L&O Methods, 23:526–542).

    The algorithm flags each PAR measurement according to the
    statistical shape of the irradiance profile and solar elevation.
    It is designed for irregularly spaced glider or float data.

    Parameters
    ----------
    pres : array-like
        Pressure or depth (dbar or m), increasing with depth.
    par : array-like
        Downwelling PAR (µmol photons m⁻² s⁻¹).
    sun_elev : float
        Solar elevation (degrees). >0 = day, ≤0 = night.
    nei_par : float, optional
        Noise Equivalent Irradiance (µmol photons m⁻² s⁻¹).
        Default = 3×10⁻² (Jutard et al. 2021).

    Returns
    -------
    flags : ndarray of int
        QC flags (1=good, 2=probably good, 3=probably bad,
        4=bad, 9=missing) for each data point.
    profile_flag : int
        Summary QC flag for the entire profile (1–4).
    pa : float
        Critical pressure separating lit vs transition region.
        NaN if undefined.

    Notes
    -----
    Implements the steps in La Forgia & Organelli (2025):

      Preliminary tests:
        • Missing data → 9
        • PAR < −NEI → 3
        • PAR < 0 or > 2500 → 4
        • Determine day/night from solar elevation

      Daytime sequence:
        • Interpolate PAR on 0.1 dbar grid (only where finite)
        • Apply Shapiro–Wilk normality test on successive tail segments
        • Last null p-value → P_A (lit/transition boundary)
        • Fit 4th-order polynomial below P_A to locate P_C
          (transition → 2, deeper → 3)

      Nighttime sequence:
        • If all p-values >0 → 2 (probably good)
        • Else compute P_A and compare mean PAR above/below it

    Reference
    ---------
    La Forgia, G. & Organelli, E. (2025).
      Real-time quality assessment for Biogeochemical Argo radiometric profiles.
      *Limnology & Oceanography: Methods*, 23, 526–542.
      https://doi.org/10.1002/lom3.10701
    """
    pres = np.asarray(pres, dtype=float)
    par = np.asarray(par, dtype=float)
    N = len(par)
    flags = np.full(N, 1, int)

    # ───────────────────────────────────────────────
    # 1. Preliminary checks
    # ───────────────────────────────────────────────
    flags[np.isnan(par)] = 9
    flags[par < -nei_par] = 3
    flags[(par < 0) | (par > 2500)] = 4

    # determine mode
    mode = "day" if sun_elev > 0 else "night"

    # valid subset for interpolation
    mask = np.isfinite(pres) & np.isfinite(par)
    pres_v, par_v = pres[mask], par[mask]

    # must have ≥3 valid points for interpolation and Shapiro–Wilk
    if pres_v.size < 3:
        flags = np.where(np.isfinite(par), 3, 9)
        return flags.astype(int), 3, np.nan

    # sort by pressure and remove duplicates
    order = np.argsort(pres_v)
    pres_v, par_v = pres_v[order], par_v[order]
    pres_v, unique_idx = np.unique(pres_v, return_index=True)
    par_v = par_v[unique_idx]

    # ───────────────────────────────────────────────
    # 2. Interpolation on 0.1 dbar grid
    # (La Forgia & Organelli 2025)
    # ───────────────────────────────────────────────
    p_min, p_max = np.nanmin(pres_v), np.nanmax(pres_v)
    if not np.isfinite(p_min) or not np.isfinite(p_max) or (p_max - p_min) < 0.2:
        flags = np.where(np.isfinite(par), 3, 9)
        return flags.astype(int), 3, np.nan

    # ---- Prepare interpolated profile (adaptive resolution) ----
    p_min, p_max = np.nanmin(pres_v), np.nanmax(pres_v)

    # attempt 0.1 dbar resolution (as per the paper)
    pres_i = np.arange(p_min, p_max + 0.05, 0.1)

    # if N > 5000, resample at 0.2 dbar instead
    if pres_i.size > 5000:
        pres_i = np.arange(p_min, p_max + 0.1, 0.2)

    # if somehow still > 5000 (very deep clear profiles), fallback to 0.25 dbar
    if pres_i.size > 5000:
        pres_i = np.arange(p_min, p_max + 0.125, 0.25)

    # now perform interpolation
    f = interp1d(pres_v, par_v, kind="linear", bounds_error=False, fill_value=np.nan)
    par_i = f(pres_i)

    # ───────────────────────────────────────────────
    # 3. Shapiro–Wilk test on successive tails
    # ───────────────────────────────────────────────
    pvals = np.full_like(pres_i, np.nan, dtype=float)
    for i in range(pres_i.size):
        seg = par_i[i:]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 3:
            try:
                # TODO: Dev. verbosity to disable warning ignoring. Also below...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, pvals[i] = shapiro(seg)
            except Exception:
                pvals[i] = np.nan

    # Determine P_A = last depth where p ≈ 0 (non-normal)
    null_mask = np.isfinite(pvals) & (pvals <= 1e-4)
    pa = pres_i[np.where(null_mask)[0][-1]] if np.any(null_mask) else np.nan

    # ───────────────────────────────────────────────
    # 4. DAYTIME tests
    # ───────────────────────────────────────────────
    if mode == "day":
        if np.isfinite(pa):
            flags[pres < pa] = 1  # lit region

        # Transition / deep region polynomial fit
        trans_mask = pres >= (pa if np.isfinite(pa) else np.nanmin(pres))
        if np.sum(trans_mask) >= 6:
            z, y = pres[trans_mask], par[trans_mask]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = np.polyfit(z, y, deg=4)
            fit = np.polyval(coeffs, z)
            # find first relative minimum (derivative zero → upward curvature)
            grad = np.gradient(fit)
            minima_idx = np.where(np.diff(np.sign(grad)) > 0)[0]
            if minima_idx.size > 0:
                pc = z[minima_idx[0]]
                flags[(pres >= pa) & (pres < pc)] = 2
                flags[pres >= pc] = 3
            else:
                flags[pres >= pa] = 2

    # ───────────────────────────────────────────────
    # 5. NIGHTTIME tests
    # ───────────────────────────────────────────────
    else:
        if np.all(pvals > 0):  # fully normal
            flags[:] = 2
        elif np.isfinite(pa):
            mu_d = np.nanmean(par[pres >= pa])
            mu_u = np.nanmean(par[pres < pa])
            if mu_u >= mu_d:
                flags[pres < pa] = 2
            else:
                flags[pres < pa] = 3
            flags[pres >= pa] = 2

    # ───────────────────────────────────────────────
    # 6. Profile-level summary flag
    # ───────────────────────────────────────────────
    n_good = np.sum(np.isin(flags, [1, 2]))
    n_bad = np.sum(np.isin(flags, [4, 9]))
    n_prob_bad = np.sum(flags == 3)
    profile_flag = (
        4
        if n_bad >= n_good + n_prob_bad
        else (
            1
            if n_good / N >= 0.25
            else 2 if np.sum(flags == 2) >= np.sum(flags == 3) else 3
        )
    )

    return flags.astype(int), profile_flag, pa


@register_qc
class par_irregularity_qc(BaseQC):
    """
    Wrapper for qc_par_flagging, defining solar_elevation if it is not provided.
    """

    qc_name = "PAR irregularity qc"
    parameter_schema = {
        "noise_equivalent_estimate": {
            "type": float,
            "default": 3e-2,
            "description": "Sensor noise-equivalent estimate used when flagging PAR irregularities.",
        },
        "plot_profiles": {
            "type": list,
            "default": [],
            "description": "Profile numbers to plot in diagnostics.",
        },
    }
    required_variables = [
        "LATITUDE",
        "LONGITUDE",
        "TIME",
        "PRES",
        "DOWNWELLING_PAR",
        "PROFILE_NUMBER",
    ]
    qc_outputs = ["DOWNWELLING_PAR_QC"]

    def return_qc(self):
        # Subset the data
        self.data = self.data[self.required_variables]

        # Make an unchecked (0) QC container for PAR QC
        par_qc = np.full(len(self.data["DOWNWELLING_PAR"]), 0)

        # Apply the checks across individual profiles
        profile_numbers = np.unique(
            self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS")
        )
        for profile_number in self.log_progress(profile_numbers, desc="", unit="profile"):
            # Subset the data
            profile = self.data.where(
                self.data["PROFILE_NUMBER"] == profile_number, drop=True
            )

            # Find the solar elevation
            solar_elevation = calculate_solar_elevation(
                profile["LATITUDE"][0].values,
                profile["LONGITUDE"][0].values,
                profile["TIME"][0].values,
            )

            # Apply the QC opperation
            profile_element_qc, _, _ = qc_par_flagging(
                profile["PRES"],
                profile["DOWNWELLING_PAR"],
                solar_elevation,
                self.noise_equivalent_estimate,
            )

            # Stitch the QC results back into the QC container
            profile_element_indices = np.where(
                self.data["PROFILE_NUMBER"] == profile_number
            )
            par_qc[profile_element_indices] = profile_element_qc

        # any remaining flags that are 0 (unchecked) are updated to 1 (good)
        par_qc[par_qc == 0] = 1

        # Collect the flags
        self.data["DOWNWELLING_PAR_QC"] = (["N_MEASUREMENTS"], par_qc)
        self.flags = self.data["DOWNWELLING_PAR_QC"].to_dataset()

        return self.flags

    def plot_diagnostics(self):
        mpl.use("tkagg")

        if len(self.plot_profiles) == 0:
            self.log("To see diagnostics, please specify the plot_profiles setting.")
            return

        nrows = int(np.ceil(len(self.plot_profiles) / 3))
        fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(12, nrows * 6), dpi=200)

        for profile_number, ax in zip(self.plot_profiles, axs.flatten()):
            # Select the profile data
            profile = self.data.where(
                self.data["PROFILE_NUMBER"] == profile_number, drop=True
            ).dropna(dim="N_MEASUREMENTS", subset=["DOWNWELLING_PAR", "PRES"])

            if len(profile["DOWNWELLING_PAR"]) == 0:
                ax.legend(
                    title=f"Prof. {profile_number} (data missing)", loc="upper right"
                )
                continue

            for flag in range(10):
                # Get the data for each flag and check it isn't empty
                plot_data = profile.where(
                    profile["DOWNWELLING_PAR_QC"] == flag, drop=True
                )
                if len(plot_data["DOWNWELLING_PAR"]) == 0:
                    continue

                ax.plot(
                    plot_data["DOWNWELLING_PAR"],
                    plot_data["PRES"],
                    c=flag_cols[flag],
                    ls="",
                    marker="o",
                    label=f"{flag}",
                )

            ax.set(
                xlabel="PAR",
                ylabel="PRES",
                ylim=(np.nanmax(profile["PRES"]), 0),
            )

            ax.legend(title=f"Prof. {profile_number} flags", loc="upper right")

        fig.suptitle("PAR irregularity test")
        fig.tight_layout()
        plt.show(block=True)
