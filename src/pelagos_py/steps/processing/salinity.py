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
from pelagos_py.utils.processing_utils import cndc_scale_factor

#### Custom imports ####
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
import xarray as xr
import pandas as pd
import numpy as np
import gsw


def running_average_nan(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Symmetric running-average mean that ignores NaNs. ``window_size`` must be odd.

    :meta private:
    """

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd for symmetry.")

    pad_size = window_size // 2
    padded = np.pad(arr, pad_size, mode="reflect")

    kernel = np.ones(window_size)
    sum_vals = np.convolve(np.nan_to_num(padded), kernel, mode="valid")
    count_vals = np.convolve(~np.isnan(padded), kernel, mode="valid")

    # NaN out empty windows to avoid uninitialised-memory warnings
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
    """Find the optimal CNDC-TEMP time lag (seconds) for one profile by minimising
    std(salinity - running-average salinity). With ``return_cost_data`` also returns
    a dict of intermediate arrays for diagnostics.

    :meta private:
    """

    profile_data = profile_data[[time_col, "CNDC", "PRES", "TEMP"]].dropna(
        dim="N_MEASUREMENTS", subset=["CNDC"]
    )

    if len(profile_data[time_col]) == 0:
        if return_cost_data:
            return np.nan, None
        return np.nan

    # Elapsed time in seconds from the start of the profile
    t0 = profile_data[time_col].values[0]
    profile_data["ELAPSED_TIME[s]"] = (profile_data[time_col] - t0).dt.total_seconds()

    # gsw wants conductivity in mS/cm; scale from the units attribute (S/m assumed if unset)
    cndc_factor = cndc_scale_factor(profile_data["CNDC"].attrs.get("units"))

    # Callable predicting CNDC at any given time
    conductivity_from_time = interpolate.interp1d(
        profile_data["ELAPSED_TIME[s]"].values,
        profile_data["CNDC"].values,
        bounds_error=False,
    )

    # Trial lags; columns are (lag value, lag score)
    time_lags = np.array([np.linspace(-2, 2, 41), np.full(41, np.nan)]).T

    saved_psal = {} if return_cost_data else None

    for i, lag in enumerate(time_lags[:, 0].copy()):
        time_shifted_conductivity = conductivity_from_time(
            profile_data["ELAPSED_TIME[s]"] + lag
        )

        PSAL = gsw.conversions.SP_from_C(
            time_shifted_conductivity * cndc_factor,
            profile_data["TEMP"],
            profile_data["PRES"],
        )

        PSAL_Smooth = running_average_nan(PSAL, filter_window_size)

        # std of (raw - smoothed) scores spikiness; spikier data scores higher
        PSAL_Diff = PSAL - PSAL_Smooth
        time_lags[i, 1] = np.nanstd(PSAL_Diff)

        if return_cost_data:
            saved_psal[lag] = (PSAL, PSAL_Smooth)

    # Lag with the lowest score
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

    Two corrections are applied in sequence:

    - **Conductivity-temperature lag (C-T lag).** ``CNDC`` and ``TEMP`` are measured
      by separate sensors with different response times, so the records are slightly
      misaligned and spike salinity at sharp gradients. :meth:`correct_ct_lag`
      estimates the optimal ``CNDC``/``TEMP`` time shift from a sample of profiles and
      applies the median shift to the whole dataset, following Woo (2019) [3]_.
    - **Thermal-mass (thermal lag) error.** The conductivity cell stores and releases
      heat, so the in-cell water temperature lags the ambient temperature.
      :meth:`correct_thermal_lag` reconstructs the in-cell temperature with the
      recursive filter and fixed coefficients of Morison et al. (1994) [1]_.

    The thermal-mass coefficients (``alpha``/``tau``) are taken directly from
    Morison et al. (1994), not re-optimised in T/S space (cf. Garau et al., 2011
    [2]_). They suit a pumped Sea-Bird CT sail at the flow rate reported by
    Woo (2019); unpumped CTDs would need the flow rate derived from the glider's
    velocity through the water.

    Samples flagged in ``calculation_flag_filter`` (by default probably-bad (3),
    bad (4), missing (9)) neither inform the lag estimate nor anchor the correction
    interpolants, but both corrections are still evaluated at every sample.

    Parameters
    ----------
    filter_window_size : int, optional
        Length, in samples, of the running-average filter used when searching for
        the optimal C-T lag. Must be odd. Default ``21``.

    Examples
    --------
    .. code-block:: yaml

        steps:
          - name: "ADJ: Salinity"
            parameters:
              filter_window_size: 21
            diagnostics: false

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
    # Read only if present: TIME_CTD is an optional fallback for TIME (see run());
    # PROFILE_DIRECTION/DEPTH are only used to build the diagnostics QC mask.
    optional_variables = ["TIME_CTD", "PROFILE_DIRECTION", "DEPTH"]
    uses_data_subset = True

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

        # Per-input usable masks: flagged samples anchor no interpolant and inform no
        # lag estimate; the combined mask drops a sample if any one input is flagged.
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

        # self.data is a subset of context["data"] (see QCHandlingMixin); merge
        # rather than replace so variables outside the subset aren't dropped.
        self.context["data"].update(self.data)
        return self.context

    def correct_ct_lag(self):
        """Align conductivity to temperature to suppress salinity spikes.

        For a random sample of up to 100 qualifying profiles (longer than one hour
        with more than ``3 * filter_window_size`` samples), the optimal ``CNDC``
        time shift is found by minimising std(salinity - running-average salinity)
        over trial lags of -2 s to +2 s in 0.1 s steps. The median per-profile lag is
        applied to ``CNDC`` across the whole dataset. Operates in place on
        ``self.data``; ``self.ct_lag_median`` is stored for diagnostics.
        """
        profile_numbers = np.unique(
            self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS").values
        )

        # Intermediate products; columns are (profile number, time lag)
        self.per_profile_optimal_lag = np.full((len(profile_numbers), 2), np.nan)
        self._ct_cost_data = None

        prof_arr = self.data["PROFILE_NUMBER"].values

        # Randomly permute for uniform sampling across the dataset
        indices = np.random.permutation(len(profile_numbers))

        processed_count = 0
        max_profiles = 100
        filter_size = self.filter_window_size

        # Cheap pre-scan (time span + count, no interpolation) so the progress bar
        # total matches the number of qualifying profiles actually processed.
        time_arr = self.data[self.time_col].values
        finite = ~pd.isnull(time_arr) & ~pd.isnull(prof_arr)
        grouped_times = pd.Series(time_arr[finite]).groupby(prof_arr[finite])
        durations = grouped_times.max() - grouped_times.min()
        counts = grouped_times.count()
        qualifying = (durations >= pd.Timedelta(hours=1)) & (counts > 3 * filter_size)
        n_to_process = min(max_profiles, int(qualifying.sum()))

        pbar = self.log_progress(total=n_to_process, desc="CT Lag", unit="prof")

        # Loop through all good profiles and store the optimal C-T lag for each.
        for i in indices:
            if processed_count >= max_profiles:
                break

            profile_number = profile_numbers[i]
            prof_indices = np.where(prof_arr == profile_number)[0]

            if len(prof_indices) == 0:
                continue

            profile = self.data.isel(N_MEASUREMENTS=prof_indices)

            # NaN out flagged samples so they do not affect the lag estimate
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

        # Only usable samples anchor the interpolant; it is still evaluated everywhere
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
        """Correct the thermal-mass error in temperature.

        Reconstructs the in-cell temperature per profile using the recursive filter
        of Morison et al. (1994) (their eq. 5). The ``alpha``/``tau`` coefficients are
        the fixed Morison et al. (1994) values scaled by the flow rate reported by
        Woo (2019). Temperature is resampled to 1 Hz for the filter and interpolated
        back onto the original sampling. Operates in place on ``self.data``.
        """
        corrected_temp_array = np.full(len(self.data["TEMP"]), np.nan)
        prof_arr = self.data["PROFILE_NUMBER"].values
        profile_numbers = np.unique(
            self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS").values
        )

        self.filter_params = {}
        self._thermal_scatter_data = None

        for prof in self.log_progress(profile_numbers, desc="Thermal Lag", unit="prof"):

            # Restrict to this profile's rows first (like correct_ct_lag above),
            # so the NaN-mask/where below runs on a per-profile slice instead of
            # rebuilding a full-dataset-sized copy on every iteration.
            prof_indices = np.where(prof_arr == prof)[0]
            if len(prof_indices) == 0:
                continue
            profile = self.data[[self.time_col, "TEMP", "PRES"]].isel(
                N_MEASUREMENTS=prof_indices
            )
            nan_mask = profile["TEMP"].isnull()
            data_subset = profile.where(~nan_mask, drop=True)
            indices = prof_indices[~nan_mask.values]

            if len(data_subset[self.time_col]) < 5:
                continue

            # Only usable samples anchor the interpolant; every sample is still corrected
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
        # Dashboard of applied CNDC/TEMP adjustments and their dataset-wide impact
        mpl.use("tkagg")

        # --- Friendly Configuration Variables ---
        FIG_SIZE = (12, 7)
        DPI = 120

        # Colours
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

        # (4)+(5) Row 2, Col 2-3: up/down-cast salinity, raw (left, all samples) vs
        # QC-clean + adjusted (right). ~20 mid-mission profiles sharing one sal/pres range.
        COLOUR_DOWN = "grey"
        COLOUR_UP = "tab:blue"

        def psal_from(ds):
            c = ds["CNDC"].values * cndc_scale_factor(ds["CNDC"].attrs.get("units"))
            return gsw.conversions.SP_from_C(c, ds["TEMP"].values, ds["PRES"].values)

        has_direction = "PROFILE_DIRECTION" in self.data
        if len(unique_profs) > 0 and has_direction:
            # Contiguous ~20-profile block from the middle of the deployment
            p1 = int(np.nanmedian(unique_profs))
            p2 = p1 + int(min(len(unique_profs) / 2, 20))
            in_subset = (prof_arr > p1) & (prof_arr < p2)
            subset_mask = in_subset & plot_qc_mask.values

            dir_arr = self.data["PROFILE_DIRECTION"].values
            pres = self.data_copy["PRES"].values
            psal_raw = psal_from(self.data_copy)
            psal_corr = psal_from(self.data)

            # Log sample counts and depth/salinity scales so a small correction on a
            # deep axis stays diagnosable
            _sub = subset_mask & np.isin(dir_arr, [1.0, -1.0])
            _dsal = np.abs(psal_corr[_sub] - psal_raw[_sub])
            _p = pres[_sub][np.isfinite(pres[_sub])]
            _sr = psal_raw[_sub]
            _dropped = int((in_subset & ~plot_qc_mask.values & np.isfinite(psal_raw)).sum())
            self.log(
                f"Salinity profile panels: {int(_sub.sum())} samples shown, "
                f"{_dropped} hidden by QC flags (3/4/9); "
                f"PRES {np.nanmin(_p):.0f}-{np.nanmax(_p):.0f} dbar; "
                f"raw PSAL spread {np.nanmax(_sr) - np.nanmin(_sr):.3f}; "
                f"correction |Δpsal| median {np.nanmedian(_dsal):.4f}, "
                f"max {np.nanmax(_dsal):.4f}"
            )

            def draw_profiles(ax, psal, mask, title):
                for direction, colour, lbl in (
                    (1.0, COLOUR_DOWN, "Downcast"),
                    (-1.0, COLOUR_UP, "Upcast"),
                ):
                    sel = mask & (dir_arr == direction)
                    first = True
                    for pn in np.unique(prof_arr[sel]):
                        idx = np.where(sel & (prof_arr == pn))[0]
                        if len(idx) == 0:
                            continue
                        # Drop NaN samples so points connect into a profile line
                        x = psal[idx]
                        y = pres[idx]
                        finite = np.isfinite(x) & np.isfinite(y)
                        if not finite.any():
                            continue
                        ax.plot(
                            x[finite],
                            y[finite],
                            color=colour,
                            lw=0.7,
                            alpha=0.7,
                            label=lbl if first else None,
                        )
                        first = False
                ax.set_title(title, fontsize=TITLE_SIZE)
                ax.set_xlabel("Practical Salinity", fontsize=LABEL_SIZE)
                ax.tick_params(axis="both", labelsize=LABEL_SIZE)
                ax.grid(True, alpha=0.2)
                ax.legend(fontsize=7, loc="lower right")

            draw_profiles(ax_sal, psal_raw, in_subset, "Raw salinity")
            draw_profiles(ax_diff, psal_corr, subset_mask, "QC + adjusted")
            ax_sal.set_ylabel("Pressure (dbar)", fontsize=LABEL_SIZE)

            # Shared pressure-down ranges framed on QC-clean salinity; raw spikes clip
            raw_sub = in_subset & np.isin(dir_arr, [1.0, -1.0])
            clean_sal = psal_corr[subset_mask & np.isin(dir_arr, [1.0, -1.0])]
            clean_sal = clean_sal[np.isfinite(clean_sal)]
            if clean_sal.size:
                s_lo, s_hi = clean_sal.min(), clean_sal.max()
                pad = 0.02 * (s_hi - s_lo) if s_hi > s_lo else 0.02
                for ax in (ax_sal, ax_diff):
                    ax.set_xlim(s_lo - pad, s_hi + pad)
            p_finite = pres[raw_sub][np.isfinite(pres[raw_sub])]
            if len(p_finite) > 0:
                pmin, pmax = p_finite.min(), p_finite.max()
                ppad = 0.02 * (pmax - pmin) if pmax > pmin else 1.0
                for ax in (ax_sal, ax_diff):
                    ax.set_ylim(pmax + ppad, pmin - ppad)

        # Final Render
        fig.suptitle("Salinity Adjustment Diagnostics Dashboard", fontsize=11, fontweight="bold")
        plt.show(block=True)
