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

"""Pipeline step for adjusting and deriving salinity from conductivity, temperature and pressure."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag

#### Custom imports ####
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from scipy import interpolate
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
import gsw


def running_average_nan(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Symmetric running-average mean that ignores NaNs. ``window_size`` must be odd.

    :meta private:
    """

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd for symmetry.")

    pad_size = window_size // 2  # Symmetric padding
    padded = np.pad(arr, pad_size, mode="reflect")  # Edge handling

    kernel = np.ones(window_size)
    # Compute weighted sums while ignoring NaNs
    sum_vals = np.convolve(np.nan_to_num(padded), kernel, mode="valid")
    count_vals = np.convolve(~np.isnan(padded), kernel, mode="valid")

    # Compute the moving average, handling NaNs properly and preventing uninitialised memory warnings
    avg = np.divide(
        sum_vals,
        count_vals,
        out=np.full_like(sum_vals, np.nan, dtype=float),
        where=(count_vals != 0),
    )

    return avg


def compute_optimal_lag(
    profile_data, filter_window_size, time_col, return_cost_data=False
):
    """
    Find the optimal conductivity-temperature time lag (seconds) for one profile.

    Trials lags from -2 s to +2 s in 0.1 s steps and returns the lag that minimises
    the standard deviation of (salinity - running-average salinity), i.e. the lag
    that suppresses salinity spiking. When ``return_cost_data`` is True a second
    dict of intermediate arrays is also returned for diagnostics. See
    :meth:`AdjustSalinity.correct_ct_lag` for the full method and references.

    :meta private:
    """

    # remove any rows where conductivity is bad (nan)
    profile_data = profile_data[[time_col, "CNDC", "PRES", "TEMP"]].dropna(
        dim="N_MEASUREMENTS", subset=["CNDC"]
    )

    if len(profile_data[time_col]) == 0:
        if return_cost_data:
            return np.nan, None
        return np.nan

    # Find the elapsed time in seconds from the start of the profile
    t0 = profile_data[time_col].values[0]
    profile_data["ELAPSED_TIME[s]"] = (profile_data[time_col] - t0).dt.total_seconds()

    # Creates a callable function that predicts what CNDC would be at any given time
    conductivity_from_time = interpolate.interp1d(
        profile_data["ELAPSED_TIME[s]"].values,
        profile_data["CNDC"].values,
        bounds_error=False,
    )

    # Specify the range time lags that the optimum will be found from. Column indexes are: (lag value, lag score)
    time_lags = np.array([np.linspace(-2, 2, 41), np.full(41, np.nan)]).T

    saved_psal = {} if return_cost_data else None

    # For each lag find its score and add it to the time_lags array
    for i, lag in enumerate(time_lags[:, 0].copy()):
        # Apply the time shift
        time_shifted_conductivity = conductivity_from_time(
            profile_data["ELAPSED_TIME[s]"] + lag
        )

        # Scale if necessary (handles conductivity supplied in S/m rather than mS/cm)
        cndc_scaled = (
            time_shifted_conductivity * 10
            if np.nanmax(time_shifted_conductivity) < 10
            else time_shifted_conductivity
        )

        # Derive salinity with the time shifted CNDC (spiking will be minimized when CNDC and TEMP are aligned)
        PSAL = gsw.conversions.SP_from_C(
            cndc_scaled, profile_data["TEMP"], profile_data["PRES"]
        )

        # Smooth the salinity profile (to remove spiking)
        PSAL_Smooth = running_average_nan(PSAL, filter_window_size)

        # Subtracting the raw and smoothed salinity gives an indication of "spikiness".
        PSAL_Diff = PSAL - PSAL_Smooth

        # More spiky data will have higher standard deviation - which is used to score the effectiveness of the applied lag
        time_lags[i, 1] = np.nanstd(PSAL_Diff)

        if return_cost_data:
            saved_psal[lag] = (PSAL, PSAL_Smooth)

    # return the time lag which has the lowest score (standard deviation)
    best_score_index = np.argmin(time_lags[:, 1])
    best_lag = time_lags[best_score_index, 0]

    if return_cost_data:
        zero_idx = int(np.argmin(np.abs(time_lags[:, 0])))
        zero_lag = time_lags[zero_idx, 0]
        p_best, p_smooth_best = saved_psal[best_lag]
        p_zero, p_smooth_zero = saved_psal[zero_lag]

        cost_data = {
            "lags": time_lags[:, 0],
            "costs": time_lags[:, 1],
            "best_lag": best_lag,
            "zero_lag": zero_lag,
            "elapsed_time": profile_data["ELAPSED_TIME[s]"].values,
            "resid_zero": p_zero - p_smooth_zero,
            "resid_best": p_best - p_smooth_best,
        }
        return best_lag, cost_data

    return best_lag


@register_step
class AdjustSalinity(BaseStep, QCHandlingMixin):
    """
    Corrects conductivity- and temperature-related sensor errors so that salinity
    can be derived cleanly from a glider CTD.

    Two corrections are applied in sequence to the dataset:

    - **Conductivity-temperature lag (C-T lag).** Conductivity and temperature are
      measured by separate sensors with different response times, so the two records
      are slightly misaligned and produce salinity spikes at sharp gradients.
      :meth:`correct_ct_lag` estimates the optimal time shift between ``CNDC`` and
      ``TEMP`` from a sample of profiles and applies the median shift to the whole
      dataset, following Woo (2019) [3]_.
    - **Thermal-mass (thermal lag) error.** The conductivity cell stores and releases
      heat, so the temperature of the water inside it lags the ambient temperature.
      :meth:`correct_thermal_lag` reconstructs the in-cell temperature with the
      recursive filter and fixed coefficients of Morison et al. (1994) [1]_.

    The thermal-mass coefficients (``alpha``/``tau``) are taken directly from
    Morison et al. (1994) and are not re-optimised in T/S space, as done by
    Morison et al. (1994) or Garau et al. (2011) [2]_. They are appropriate for a
    pumped Sea-Bird CT sail at the conductivity-cell flow rate reported by
    Woo (2019); unpumped CTDs would require the flow rate to be derived from the
    glider's velocity through the water (e.g. from pitch or a flight model).

    Parameters
    ----------
    filter_window_size : int, optional
        Length, in samples, of the running-average filter used when searching for
        the optimal C-T lag. Must be odd. Default ``21``.

    Examples
    --------
    Example usage in a pipeline configuration:

    .. code-block:: yaml

        steps:
          - name: "ADJ: Salinity"
            parameters:
              filter_window_size: 21
            diagnostics: false

    Samples whose flags fall in ``calculation_flag_filter`` (by default probably-bad
    (3), bad (4) and missing (9); see ``qc_handling_settings``) inform neither the
    optimal-lag estimate nor the interpolants the two corrections are built from, so a
    bad conductivity or temperature cannot bend its neighbours' correction. Both
    corrections are still evaluated at every sample.

    References
    ----------
    .. [1] Morison, J., Andersen, R., Larson, N., D'Asaro, E., & Boyd, T. (1994).
       The correction for thermal-lag effects in Sea-Bird CTD data. *Journal of
       Atmospheric and Oceanic Technology*, 11(4), 1151-1164.
    .. [2] Garau, B., Ruiz, S., Zhang, W. G., Pascual, A., Heslop, E., Kerfoot, J.,
       & Tintoré, J. (2011). Thermal lag correction on Slocum CTD glider data.
       *Journal of Atmospheric and Oceanic Technology*, 28(9), 1065-1071.
    .. [3] Woo, L. M. (2019). Delayed Mode QA/QC Best Practice Manual Version 2.0.
       Integrated Marine Observing System. DOI: 10.26198/5c997b5fdc9bd
       (http://dx.doi.org/10.26198/5c997b5fdc9bd).
    """

    step_name = "Salinity Adjustment"
    required_variables = ["TIME", "PROFILE_NUMBER", "CNDC", "TEMP", "PRES"]
    provided_variables = []

    parameter_schema = {
        "filter_window_size": {
            "type": int,
            "default": 21,
            "description": "Running-average filter size used when computing optimal time lags.",
        },
    }

    def run(self):
        self.log(f"Running adjustment...")
        # TODO: TIME_CTD checking

        # Required for plotting later
        self.data_copy = self.data.copy(deep=True)

        # Check if TIME_CTD exists
        self.time_col = "TIME_CTD"
        if self.time_col not in self.data:
            self.log("TIME_CTD cound not be found. Defaulting to TIME instead.")
            self.time_col = "TIME"

        # Filter user-specified flags
        self.filter_qc()

        # Flagged samples must not anchor the interpolants or inform the lag estimate
        # below; both corrections are still evaluated at every sample. Each input is
        # gated on its own flags, so a lag estimate combining them drops a sample if
        # any one is flagged.
        self._usable_ct = self.calculation_mask(["CNDC", "TEMP", "PRES"])
        self._usable_cndc = self.calculation_mask(["CNDC"])
        self._usable_temp = self.calculation_mask(["TEMP"])

        # Correct conductivity-temperature response time misalignment (C-T Lag)
        self.correct_ct_lag()

        # Correct thermal mass error
        self.correct_thermal_lag()

        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def correct_ct_lag(self):
        """
        Align conductivity to temperature to suppress salinity spikes.

        Conductivity and temperature are measured by separate sensors whose physical
        offset and differing response times leave the two records slightly out of
        phase, producing salinity spikes at sharp gradients.

        For a random sample of up to 100 qualifying profiles, the optimal time shift
        of ``CNDC`` relative to ``TEMP`` is found by minimising the standard deviation
        of (salinity - running-average salinity) — the approach used by RBR's
        pyRSKtools — rather than comparing downcast against upcast as originally
        described by Woo (2019). The median of the per-profile lags is then applied
        to ``CNDC`` across the whole dataset.

        Trial lags run from -2 s to +2 s in 0.1 s steps. Only profiles longer than
        one hour with more than ``3 * filter_window_size`` samples contribute to the
        median.

        Notes
        -----
        Operates in place on ``self.data``. ``self.ct_lag_median`` is stored for the
        diagnostics dashboard.
        """
        profile_numbers = np.unique(
            self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS").values
        )

        # Making a place to store intermediate products. Column dimensions: (profile number, time lag)
        self.per_profile_optimal_lag = np.full((len(profile_numbers), 2), np.nan)
        self._ct_cost_data = None

        prof_arr = self.data["PROFILE_NUMBER"].values

        # Randomly permute to ensure uniform sampling across the dataset
        indices = np.random.permutation(len(profile_numbers))

        processed_count = 0
        max_profiles = 100
        filter_size = self.filter_window_size

        # Only a random sample of up to ``max_profiles`` profiles is processed to estimate
        # the median lag. Cheaply pre-scan (no salinity/interpolation, just per-profile time
        # span and count) to find how many profiles qualify, so the bar total matches what
        # will actually be processed and reaches 100%.
        time_arr = self.data[self.time_col].values
        finite = ~pd.isnull(time_arr) & ~pd.isnull(prof_arr)
        grouped_times = pd.Series(time_arr[finite]).groupby(prof_arr[finite])
        durations = grouped_times.max() - grouped_times.min()
        counts = grouped_times.count()
        qualifying = (durations >= pd.Timedelta(hours=1)) & (counts > 3 * filter_size)
        n_to_process = min(max_profiles, int(qualifying.sum()))

        pbar = tqdm(
            total=n_to_process,
            colour="green",
            desc="\033[97mCT Lag Progress\033[0m",
            unit="prof",
        )

        # Loop through all good profiles and store the optimal C-T lag for each.
        for i in indices:
            if processed_count >= max_profiles:
                break

            profile_number = profile_numbers[i]
            prof_indices = np.where(prof_arr == profile_number)[0]

            if len(prof_indices) == 0:
                continue

            profile = self.data.isel(N_MEASUREMENTS=prof_indices)

            # The lag is derived entirely from these three, so flagged samples are
            # NaN'd out of the estimate. The lag itself is still applied to the whole
            # record below.
            usable = self._usable_ct[prof_indices]
            profile = profile.assign(
                **{
                    var: (
                        profile[var].dims,
                        np.where(usable, profile[var].values, np.nan),
                    )
                    for var in ("CNDC", "TEMP", "PRES")
                }
            )

            valid_times = profile[self.time_col].dropna(dim="N_MEASUREMENTS")

            if len(valid_times) > 0:
                duration = valid_times.values[-1] - valid_times.values[0]

                if duration >= np.timedelta64(1, "h") and len(valid_times) > 3 * filter_size:
                    if getattr(self, "diagnostics", False) and self._ct_cost_data is None:
                        optimal_lag, cost_data = compute_optimal_lag(
                            profile, filter_size, self.time_col, return_cost_data=True
                        )
                        self._ct_cost_data = cost_data
                    else:
                        optimal_lag = compute_optimal_lag(
                            profile, filter_size, self.time_col
                        )

                    self.per_profile_optimal_lag[i, :] = [profile_number, optimal_lag]
                    processed_count += 1
                    pbar.update(1)

        pbar.close()

        # Apply shifts
        valid_data_mask = (
            self.data["CNDC"].notnull() & self.data[self.time_col].notnull()
        )
        if not np.any(valid_data_mask):
            self.log("No valid CNDC data found. Skipping CT lag correction.")
            return

        lags = self.per_profile_optimal_lag[
            ~np.isnan(self.per_profile_optimal_lag[:, 1]), 1
        ]
        self.ct_lag_median = np.median(lags) if len(lags) > 0 else 0.0

        data_subset = self.data[[self.time_col, "CNDC"]].where(valid_data_mask, drop=True)

        # Find the elapsed time in seconds
        t0 = data_subset[self.time_col].values[0]
        data_subset["ELAPSED_TIME[s]"] = (
            data_subset[self.time_col] - t0
        ).dt.total_seconds()

        # Only usable samples anchor the interpolant, so a flagged conductivity cannot
        # bend the time-shifted value of its neighbours. It is still evaluated at every
        # valid sample below, so flagged ones are shifted like any other.
        anchors = self._usable_cndc[np.where(valid_data_mask)[0]]
        if anchors.sum() < 2:
            self.log(
                "Fewer than two usable CNDC samples for the CT lag shift "
                f"(flags {self.calculation_flag_filter} excluded). Skipping."
            )
            return

        CNDC_from_TIME = interpolate.interp1d(
            data_subset["ELAPSED_TIME[s]"].values[anchors],
            data_subset["CNDC"].values[anchors],
            bounds_error=False,
        )
        shifted_time = data_subset["ELAPSED_TIME[s]"].values + self.ct_lag_median

        data_subset["CNDC"].values = CNDC_from_TIME(shifted_time)

        # Reinsert the time-shifted data back into self.data
        self.data["CNDC"][valid_data_mask] = data_subset["CNDC"]

    def correct_thermal_lag(self):
        """
        Correct the thermal-mass error in temperature.

        The conductivity cell stores and releases heat, so the temperature of the
        water inside it lags the ambient temperature and biases the derived salinity.
        This reconstructs the in-cell temperature for each profile using the recursive
        filter of Morison et al. (1994) (their eq. 5), which does not require the
        sensitivity of temperature to conductivity (their eq. 2).

        The amplitude (``alpha``) and time-constant (``tau``) coefficients are the
        fixed values of Morison et al. (1994), scaled by the conductivity-cell flow
        rate reported by Woo (2019); they are not re-optimised in T/S space (cf.
        Garau et al., 2011). Temperature is resampled to 1 Hz for the filter and
        interpolated back onto the original sampling.

        Notes
        -----
        Operates in place on ``self.data``.
        """
        corrected_temp_array = np.full(len(self.data["TEMP"]), np.nan)
        profile_numbers = np.unique(
            self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS").values
        )

        self.filter_params = {}
        self._thermal_scatter_data = None

        for prof in tqdm(
            profile_numbers,
            colour="blue",
            desc="\033[97mThermal Lag Progress\033[0m",
            unit="prof",
        ):

            mask = self.data["PROFILE_NUMBER"] == prof
            nan_mask = self.data["TEMP"].isnull() | ~mask
            data_subset = self.data[[self.time_col, "TEMP", "PRES"]].where(
                ~nan_mask, drop=True
            )
            indices = np.where(~nan_mask)[0]

            if len(data_subset[self.time_col]) < 5:
                continue

            # Only usable samples anchor the interpolant the filter runs on, so a
            # flagged temperature cannot bend the correction of its neighbours. Every
            # sample in the profile is still corrected below.
            anchors = self._usable_temp[indices]
            if anchors.sum() < 2:
                continue

            # Find the elapsed time in seconds
            t0 = data_subset[self.time_col].values[0]
            data_subset["ELAPSED_TIME[s]"] = (
                data_subset[self.time_col] - t0
            ).dt.total_seconds()

            # Define a function that can estimate TEMP at any time point
            TEMP_from_TIME = interpolate.interp1d(
                data_subset["ELAPSED_TIME[s]"].values[anchors],
                data_subset["TEMP"].values[anchors],
                bounds_error=False,
                fill_value="extrapolate",
            )

            # Resample the data onto a 1Hz sample rate timeseries
            TIME_1Hz_sampling = np.arange(0, data_subset["ELAPSED_TIME[s]"].values[-1], 1)
            if len(TIME_1Hz_sampling) < 2:
                continue
            TEMP_1Hz_sampling = TEMP_from_TIME(TIME_1Hz_sampling)
            n_resamples = len(TEMP_1Hz_sampling)

            # Set up the recursive filter defined in "CTD dynamic performance and corrections through gradients"
            # Tau and alpha are the fixed coefficients of Morison94 for unpumped cell.
            # alpha: initial amplitude of the temperature error for a unit step change in ambient temperature [without unit].
            alpha_offset = 0.0135
            alpha_slope = 0.0264
            # tau = beta^-1: time constant of the error, the e-folding time of the temperature error [s].
            tau_offset = 7.1499
            tau_slope = 2.7858
            # flow_rate: The flow rate in the conductivity cell from Woo (2019).
            flow_rate = 0.4867

            tau = tau_offset + tau_slope / np.sqrt(flow_rate)
            alpha = alpha_offset + alpha_slope / flow_rate

            self.filter_params = {"alpha": alpha, "tau": tau}

            # Set the filter coefficients
            nyquist_frequency = (
                1 / 2
            )  # Nyquist frequency for 1 Hz sampling (= sample frequency / 2)
            a = 4 * nyquist_frequency * alpha * tau / (1 + 4 * nyquist_frequency * tau)
            b = 1 - (2 * a / alpha)

            # Apply the filter
            TEMP_correction = np.full(n_resamples, 0.0)
            for i in range(1, n_resamples):
                TEMP_correction[i] = -b * TEMP_correction[i - 1] + a * (
                    TEMP_1Hz_sampling[i] - TEMP_1Hz_sampling[i - 1]
                )
            corrected_TEMP_1Hz_sampling = TEMP_1Hz_sampling - TEMP_correction

            # Resample the TEMP back onto the original time sampling
            corrected_TEMP_from_TIME = interpolate.interp1d(
                TIME_1Hz_sampling,
                corrected_TEMP_1Hz_sampling,
                bounds_error=False,
                fill_value="extrapolate",
            )
            data_subset["TEMP"][:] = corrected_TEMP_from_TIME(
                data_subset["ELAPSED_TIME[s]"]
            )

            # Store adjusted data
            corrected_temp_array[indices] = data_subset["TEMP"].values

            if (
                getattr(self, "diagnostics", False)
                and self._thermal_scatter_data is None
                and TIME_1Hz_sampling[-1] >= 3600
                and np.nanmax(TEMP_1Hz_sampling) - np.nanmin(TEMP_1Hz_sampling) >= 1.0
            ):
                self._thermal_scatter_data = {
                    "dT_dt": np.gradient(TEMP_1Hz_sampling, TIME_1Hz_sampling),
                    "correction": TEMP_correction,
                }

        # Reinsert the corrected data back into self.data
        final_temp = np.where(
            np.isnan(corrected_temp_array), self.data["TEMP"].values, corrected_temp_array
        )
        self.data["TEMP"][:] = final_temp

    def generate_diagnostics(self):
        """
        Displays a comprehensive diagnostics dashboard detailing applied adjustments
        to conductivity and temperature, along with overall impacts on the dataset.
        """
        mpl.use("tkagg")

        # --- Friendly Configuration Variables ---
        FIG_SIZE = (12, 7)
        DPI = 120

        # Colours
        COLOUR_CORR_T = "darkred"
        COLOUR_CORR_C = "darkblue"
        COLOUR_BEST = "darkorange"
        COLOUR_SMOOTH = "dimgrey"
        COLOUR_SCATTER = "tab:purple"
        COLOUR_COMBINED = "teal"

        # Text Styles
        TITLE_SIZE = 9
        LABEL_SIZE = 8

        # --- Data Preparation ---
        prof_arr = self.data["PROFILE_NUMBER"].values
        unique_profs = np.unique(prof_arr[~pd.isnull(prof_arr)])

        plot_qc_mask = xr.ones_like(self.data_copy["PROFILE_NUMBER"], dtype=bool)
        for var in ["TEMP", "CNDC", "PRES", "DEPTH", self.time_col]:
            qc_col = f"{var}_QC"
            if qc_col in self.data_copy.data_vars:
                plot_qc_mask = plot_qc_mask & ~self.data_copy[qc_col].isin([3, 4, 9])

        valid_lags = self.per_profile_optimal_lag[
            ~np.isnan(self.per_profile_optimal_lag[:, 1])
        ]
        processed_profs = valid_lags[:, 0]

        if len(processed_profs) > 0:
            sample_prof = processed_profs[len(processed_profs) // 2]
        else:
            sample_prof = unique_profs[0] if len(unique_profs) > 0 else np.nan

        # --- Main Figure Setup ---
        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, constrained_layout=True)
        gs = fig.add_gridspec(2, 3)

        ax_lag = fig.add_subplot(gs[0, 0:2])
        ax_cost = fig.add_subplot(gs[0, 2])
        ax_scatter = fig.add_subplot(gs[1, 0])
        ax_sal = fig.add_subplot(gs[1, 1])
        ax_diff = fig.add_subplot(gs[1, 2])

        # (1) Row 1, Col 1-2: Applied Lag Distribution over Profile Index
        ax_lag.axhline(0, color="black", linestyle="-", lw=1.2, alpha=0.8, zorder=1)

        profs_subset = self.per_profile_optimal_lag[:, 0]
        lags_subset = self.per_profile_optimal_lag[:, 1]
        valid_indices = ~np.isnan(lags_subset)

        if np.any(valid_indices):
            profs_plot = profs_subset[valid_indices]
            lags_plot = lags_subset[valid_indices]

            label_text = f"Combined (median: {self.ct_lag_median:.2f}s)"
            ax_lag.scatter(
                profs_plot,
                lags_plot,
                c=COLOUR_COMBINED,
                s=12,
                alpha=0.6,
                label=label_text,
                zorder=2,
            )
            ax_lag.axhline(
                self.ct_lag_median, color=COLOUR_COMBINED, linestyle="--", lw=1.5, zorder=3
            )

        ax_lag.set_title("Dataset Lag Distribution by Profile", fontsize=TITLE_SIZE)
        ax_lag.set_xlabel("Profile Number", fontsize=LABEL_SIZE)
        ax_lag.set_ylabel("Optimal Lag (s)", fontsize=LABEL_SIZE)
        ax_lag.tick_params(axis="both", labelsize=LABEL_SIZE)
        ax_lag.grid(True, alpha=0.2)
        ax_lag.legend(fontsize=7)

        # (2) Row 1, Col 3: CT Lag Cost Curve
        if self._ct_cost_data:
            c = self._ct_cost_data
            ax_cost.plot(c["lags"], c["costs"], "o-", color=COLOUR_SMOOTH, lw=1, ms=3)
            ax_cost.axvline(
                c["best_lag"], color=COLOUR_BEST, ls="--", label=f"Best: {c['best_lag']:.2f}s"
            )
            ax_cost.set_xlabel("Trial Lag (s)", fontsize=LABEL_SIZE)
            ax_cost.set_ylabel("std(PSAL - smooth)", fontsize=LABEL_SIZE)
            ax_cost.set_title(
                f"Optimal CT Lag Search (Profile {sample_prof:.0f})", fontsize=TITLE_SIZE
            )
            ax_cost.tick_params(axis="both", labelsize=LABEL_SIZE)
            ax_cost.legend(fontsize=7)
            ax_cost.grid(True, alpha=0.2)

        # (3) Row 2, Col 1: Thermal Scatter & Parameters Legend
        if self._thermal_scatter_data:
            ts = self._thermal_scatter_data
            finite = np.isfinite(ts["correction"]) & np.isfinite(ts["dT_dt"])
            ax_scatter.scatter(
                ts["dT_dt"][finite],
                ts["correction"][finite],
                s=4,
                alpha=0.3,
                color=COLOUR_SCATTER,
            )
            ax_scatter.set_xlabel("dT/dt (°C/s)", fontsize=LABEL_SIZE)
            ax_scatter.set_ylabel("Corr Amplitude (°C)", fontsize=LABEL_SIZE)
            ax_scatter.set_title(
                f"Thermal Mass Verification (Profile {sample_prof:.0f})", fontsize=TITLE_SIZE
            )
            ax_scatter.tick_params(axis="both", labelsize=LABEL_SIZE)
            ax_scatter.grid(True, alpha=0.2)

            alpha_val = self.filter_params.get("alpha", np.nan)
            tau_val = self.filter_params.get("tau", np.nan)
            param_text = f"Flow Velocity: ~0.49 m/s\nAlpha (α): {alpha_val:.4f}\nTau (τ): {tau_val:.2f} s"
            ax_scatter.text(
                0.05,
                0.95,
                param_text,
                transform=ax_scatter.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#ccc"),
            )

        # (4) Row 2, Col 2: Combined Salinity Profiles
        mask_range = self.data_copy["PROFILE_NUMBER"].isin(processed_profs)
        uncorr = self.data_copy.where(mask_range & plot_qc_mask, drop=True)
        corr = self.data.where(mask_range & plot_qc_mask, drop=True)

        if len(uncorr["DEPTH"].dropna(dim="N_MEASUREMENTS")) > 0:
            c_raw = uncorr["CNDC"].values
            c_new = corr["CNDC"].values
            c_raw = c_raw * 10 if np.nanmax(c_raw) < 10 else c_raw
            c_new = c_new * 10 if np.nanmax(c_new) < 10 else c_new

            p_raw = gsw.conversions.SP_from_C(
                c_raw, uncorr["TEMP"].values, uncorr["PRES"].values
            )
            p_new = gsw.conversions.SP_from_C(
                c_new, corr["TEMP"].values, uncorr["PRES"].values
            )

            ax_sal.plot(
                p_raw,
                uncorr["DEPTH"].values,
                c="grey",
                ls="",
                marker=".",
                ms=1,
                alpha=0.3,
                label="Raw",
            )

            sal_legend = [
                mlines.Line2D(
                    [], [], color="grey", marker=".", ls="", markersize=4, label="Raw (All)"
                )
            ]

            ax_sal.plot(
                p_new,
                uncorr["DEPTH"].values,
                c=COLOUR_COMBINED,
                ls="",
                marker=".",
                ms=1.5,
                alpha=0.7,
            )
            sal_legend.append(
                mlines.Line2D(
                    [],
                    [],
                    color=COLOUR_COMBINED,
                    marker=".",
                    ls="",
                    markersize=4,
                    label="Corr Combined",
                )
            )

            ax_sal.set_title("Combined Result", fontsize=TITLE_SIZE)
            ax_sal.set_xlabel("Practical Salinity", fontsize=LABEL_SIZE)
            ax_sal.set_ylabel("Depth (m)", fontsize=LABEL_SIZE)
            ax_sal.tick_params(axis="both", labelsize=LABEL_SIZE)

            y_min, y_max = ax_sal.get_ylim()
            if abs(y_max) < abs(y_min):
                ax_sal.set_ylim(y_min, y_max)
            else:
                ax_sal.set_ylim(y_max, y_min)

            ax_sal.grid(True, alpha=0.2)
            ax_sal.legend(handles=sal_legend, fontsize=7, loc="lower right")

        # (5) Row 2, Col 3: Dataset Adjustments Diff Plot
        t_all = self.data_copy[self.time_col].values

        valid_t = (
            ~np.isnat(t_all)
            & ~np.isnan(self.data_copy["TEMP"].values)
            & ~np.isnan(self.data_copy["CNDC"].values)
            & plot_qc_mask.values
        )
        sub_step = max(1, np.sum(valid_t) // 50000)

        t_valid = t_all[valid_t][::sub_step]
        if len(t_valid) > 0:
            elapsed_days = (t_valid - t_valid[0]) / np.timedelta64(1, "D")

            temp_raw_all = self.data_copy["TEMP"].values[valid_t][::sub_step]
            temp_corr_all = self.data["TEMP"].values[valid_t][::sub_step]
            cndc_raw_all = self.data_copy["CNDC"].values[valid_t][::sub_step]
            cndc_corr_all = self.data["CNDC"].values[valid_t][::sub_step]

            cndc_raw_all = cndc_raw_all * 10 if np.nanmax(cndc_raw_all) < 10 else cndc_raw_all
            cndc_corr_all = (
                cndc_corr_all * 10 if np.nanmax(cndc_corr_all) < 10 else cndc_corr_all
            )

            temp_diff = temp_corr_all - temp_raw_all
            cndc_diff = cndc_corr_all - cndc_raw_all

            ax_diff_c = ax_diff.twinx()

            ax_diff.plot(
                elapsed_days,
                temp_diff,
                color=COLOUR_CORR_T,
                marker=".",
                ls="",
                ms=1,
                alpha=0.4,
                label="Temp Diff",
            )
            ax_diff_c.plot(
                elapsed_days,
                cndc_diff,
                color=COLOUR_CORR_C,
                marker=".",
                ls="",
                ms=1,
                alpha=0.4,
                label="CNDC Diff",
            )

            ax_diff.set_xlabel("Elapsed Time (Days)", fontsize=LABEL_SIZE)
            ax_diff.set_ylabel("Temp Difference (°C)", fontsize=LABEL_SIZE)
            ax_diff_c.set_ylabel("CNDC Difference (mS/cm)", fontsize=LABEL_SIZE)
            ax_diff.set_title("Dataset-Wide Adjustments (Corr - Raw)", fontsize=TITLE_SIZE)
            ax_diff.tick_params(axis="both", labelsize=LABEL_SIZE)
            ax_diff_c.tick_params(axis="y", labelsize=LABEL_SIZE)

            # Scale both axes symmetrically about zero so the two 0-lines coincide
            t_absmax = np.nanmax(np.abs(temp_diff))
            c_absmax = np.nanmax(np.abs(cndc_diff))
            if np.isfinite(t_absmax) and t_absmax > 0:
                ax_diff.set_ylim(-t_absmax * 1.05, t_absmax * 1.05)
            if np.isfinite(c_absmax) and c_absmax > 0:
                ax_diff_c.set_ylim(-c_absmax * 1.05, c_absmax * 1.05)

            lines1, labels1 = ax_diff.get_legend_handles_labels()
            lines2, labels2 = ax_diff_c.get_legend_handles_labels()

            leg_handles = []
            for line in lines1 + lines2:
                leg_handles.append(
                    mlines.Line2D([], [], color=line.get_color(), marker=".", ls="", markersize=6)
                )

            ax_diff.legend(leg_handles, labels1 + labels2, loc="best", fontsize=7)
            ax_diff.grid(True, alpha=0.2)

        # Final Render
        fig.suptitle("Salinity Adjustment Diagnostics Dashboard", fontsize=11, fontweight="bold")
        plt.show(block=True)
