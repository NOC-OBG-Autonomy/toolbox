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

"""Pipeline steps for processing dissolved-oxygen optode data (uncalibrated phase and optode temperature)."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag

#### Custom imports ####
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pelagos_py.utils import fig_spec
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt


def check_config(self, expected_params):
    """Runtime checks beyond the parameter schema.

    Parameter *presence* and defaults are handled by ``parameter_schema``; this
    additionally (a) catches method-dependent parameters left unset (``None``) and
    (b) verifies that any ``*_name`` parameter points at a variable that actually
    exists in the dataset.
    """
    for param in expected_params:
        if getattr(self, param, None) is None:
            raise KeyError(f"[{self.step_name}] '{param}' is missing from config")
        if "_name" in param:
            if getattr(self, param) not in self.data.data_vars:
                raise KeyError(
                    f"[{self.step_name}] {getattr(self, param)} could not be found in the data"
                )


def _plot_section(data, var, pressure_var, step_name):
    # Single panel section: TIME vs pressure_var (inverted), coloured by var's value.
    # Continuous colour + colorbar isn't reproducible by the dashboard's WebGL viewer (PNG-only).
    if var not in data or pressure_var not in data:
        return

    matplotlib.use("tkagg")
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]

    time, pres, values = data["TIME"].values, data[pressure_var].values, data[var].values
    finite = ~pd.isnull(time) & np.isfinite(pres) & np.isfinite(values)

    sc = ax.scatter(
        time[finite], pres[finite], c=values[finite], cmap="viridis",
        s=fig_spec.MARKER, alpha=fig_spec.ALPHA, rasterized=finite.sum() > fig_spec.RASTER_ABOVE,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(fig_spec.axis_label(var, data[var].attrs.get("units")), fontsize=fig_spec.FS_LABEL)
    cbar.ax.tick_params(labelsize=fig_spec.FS_TICK)

    fig_spec.date_axis(ax, which="x")
    ylabel = fig_spec.axis_label(pressure_var, data[pressure_var].attrs.get("units"))
    fig_spec.style_axes(ax, ylabel=ylabel)
    ax.invert_yaxis()

    fig_spec.finish(fig, suptitle=f"{step_name} Diagnostics")
    plt.show(block=True)


def _qc_good_mask(data, var):
    # True where var isn't flagged bad (4); no QC var means nothing to exclude.
    qc_name = f"{var}_QC"
    if qc_name not in data:
        return np.ones(data[var].shape, dtype=bool)
    return data[qc_name].values != 4


def _plot_diff(data, raw_var, corrected_var, pressure_var, step_name):
    # Two panels: raw+corrected overlaid on TIME, and TIME vs pressure_var (inverted)
    # coloured by (corrected - raw) - where in the water column the correction bites.
    if raw_var not in data or corrected_var not in data:
        return

    matplotlib.use("tkagg")
    fig, axes = fig_spec.new_fig(nrows=2, sharex=True)
    ax0, ax1 = axes[0][0], axes[1][0]

    # Reserve the same width on ax0 as the colorbar takes on ax1, so the shared TIME axis lines up.
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="3%", pad=0.15)
    cax0.axis("off")

    good = _qc_good_mask(data, raw_var) & _qc_good_mask(data, corrected_var)
    time = data["TIME"].values
    fig_spec.points(ax0, time[good], data[raw_var].values[good], color=fig_spec.FLAGGED, label=raw_var)
    fig_spec.points(ax0, time[good], data[corrected_var].values[good], color=fig_spec.CATEGORY[1], label=corrected_var)
    fig_spec.style_axes(ax0, ylabel=fig_spec.axis_label(corrected_var, data[corrected_var].attrs.get("units")))
    # An outside (bbox_to_anchor) legend would need more width than the colorbar spacer
    # reserves above, re-breaking the TIME-axis alignment with ax1 - so keep it inside.
    ax0.legend(fontsize=fig_spec.FS_LEGEND, loc="upper right", framealpha=0.9, markerscale=2)

    diff = data[corrected_var].values - data[raw_var].values
    if pressure_var in data:
        pres = data[pressure_var].values
        finite = np.isfinite(diff) & np.isfinite(pres) & good
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="3%", pad=0.15)
        sc = ax1.scatter(
            time[finite], pres[finite], c=diff[finite], cmap="viridis",
            s=fig_spec.MARKER, alpha=fig_spec.ALPHA, rasterized=True,
        )
        cbar = fig.colorbar(sc, cax=cax1)
        cbar.set_label(f"{corrected_var} - {raw_var}", fontsize=fig_spec.FS_LABEL)
        cbar.ax.tick_params(labelsize=fig_spec.FS_TICK)
        fig_spec.date_axis(ax1, which="x")
        fig_spec.style_axes(ax1, xlabel="TIME", ylabel=fig_spec.axis_label(pressure_var, data[pressure_var].attrs.get("units")))
        ax1.invert_yaxis()
    else:
        fig_spec.points(ax1, time[good], diff[good], color=fig_spec.CATEGORY[0])
        ax1.axhline(0, color="grey", alpha=0.7, zorder=0, linewidth=0.8)
        fig_spec.date_axis(ax1, which="x")
        fig_spec.style_axes(ax1, xlabel="TIME", ylabel=f"{corrected_var} - {raw_var}")

    fig_spec.finish(fig, suptitle=f"{step_name} Diagnostics")
    plt.show(block=True)


@register_step
class DeriveUncalibratedPhase(BaseStep, QCHandlingMixin):

    step_name = "Derive Uncalibrated Phase"

    parameter_schema = {
        "blue_phase_name": {
            "type": str,
            "required": True,
            "description": "Name of the blue-phase variable in the dataset.",
        },
        "red_phase_name": {
            "type": str,
            "default": None,
            "description": "Optional red-phase variable; subtracted from blue phase when given.",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Derive Uncalibrated Phase"
              parameters:
                #  <MANDATORY>
                blue_phase_name: "BPHASE_DOXY"
                # <OPTIONAL>
                red_phase_name: "RPHASE_DOXY"
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check blue_phase_name is present
        check_config(self, ("blue_phase_name",))

        # Check if the output already exists
        if "UNCAL_PHASE_DOXY" in self.data.data_vars:
            self.log_warn("UNCAL_PHASE_DOXY already exists in the data. Overwriting...")

        # Calculate Uncalibrated phase and specify what QC will be derived from
        qc_parents = [f"{self.blue_phase_name}_QC"]
        if self.red_phase_name is not None:
            check_config(self, ("red_phase_name",))
            self.data["UNCAL_PHASE_DOXY"] = (
                self.data[self.blue_phase_name] - self.data[self.red_phase_name]
            )
            qc_parents.append(f"{self.red_phase_name}_QC")
        else:
            self.data["UNCAL_PHASE_DOXY"] = self.data[self.blue_phase_name]

        self.data["UNCAL_PHASE_DOXY"].attrs["units"] = self.data[self.blue_phase_name].attrs.get(
            "units", "degree"
        )
        self.data["UNCAL_PHASE_DOXY"].attrs["long_name"] = "Uncalibrated oxygen optode phase"
        self.data["UNCAL_PHASE_DOXY"].attrs["standard_name"] = "UNCAL_PHASE_DOXY"

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc({"UNCAL_PHASE_DOXY_QC": qc_parents})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        _plot_section(self.data, "UNCAL_PHASE_DOXY", "PRES", self.step_name)


@register_step
class DeriveOptodeTemperature(BaseStep, QCHandlingMixin):

    step_name = "Derive Optode Temperature"

    parameter_schema = {
        "temp_voltage_name": {
            "type": str,
            "required": True,
            "description": "Name of the optode temperature-voltage variable in the dataset.",
        },
        "calib_coefficients": {
            "type": list,
            "required": True,
            "description": "Polynomial calibration coefficients (at least two).",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Derive Optode Temperature"
              parameters:
                temp_voltage_name: "TEMP_VOLTAGE_DOXY"
                calib_coefficients: [0, 1, 0, 0, 0, 0]
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check the optode temperature voltage and calibration coefficients are present
        check_config(self, ("temp_voltage_name", "calib_coefficients"))

        # Check there are at least two coefficients for the polynomial. Fill in missing values.
        if len(self.calib_coefficients) < 2:
            raise ValueError(
                f"[{self.step_name}] At least two calibration coefficients are required."
            )
        coeffs = [0] * 6
        for i in range(len(self.calib_coefficients)):
            coeffs[i] = self.calib_coefficients[i]

        # Check if the output already exists
        if "TEMP_DOXY" in self.data.data_vars:
            self.log_warn("TEMP_DOXY already exists in the data. Overwriting...")

        # Calculate temp_doxy
        temp_doxy = 0
        for i, coeff in enumerate(coeffs):
            temp_doxy += coeff[i] * self.data[self.temp_voltage_name] ** i
        self.data["TEMP_DOXY"] = temp_doxy
        self.data["TEMP_DOXY"].attrs["units"] = "degree_Celsius"
        self.data["TEMP_DOXY"].attrs["long_name"] = "Oxygen optode temperature"
        self.data["TEMP_DOXY"].attrs["standard_name"] = "TEMP_DOXY"

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc({"TEMP_DOXY_QC": [f"{self.temp_voltage_name}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        _plot_section(self.data, "TEMP_DOXY", "PRES", self.step_name)


@register_step
class PhasePressureCorrection(BaseStep, QCHandlingMixin):

    step_name = "Phase Pressure Correction"

    parameter_schema = {
        "optode_pressure_name": {
            "type": str,
            "required": True,
            "description": "Name of the pressure variable used for the correction.",
        },
        "correction_coefficient": {
            "type": float,
            "required": True,
            "description": "Pressure correction coefficient.",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Phase Pressure Correction"
              parameters:
                optode_pressure_name: "PRES"
                correction_coefficient: 0.1
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check the optode pressure and correction coefficient are present and that UNCAL_PHASE_DOXY is in the data
        check_config(self, ("optode_pressure_name", "correction_coefficient"))
        if "UNCAL_PHASE_DOXY" not in self.data.data_vars:
            raise KeyError(
                f"[{self.step_name}] UNCAL_PHASE_DOXY required but is missing from the data"
            )

        # Apply the correction
        self.data["UNCAL_PHASE_DOXY_PCORR"] = (
            self.data["UNCAL_PHASE_DOXY"]
            + 0.001 * self.correction_coefficient * self.data[self.optode_pressure_name]
        )
        self.data["UNCAL_PHASE_DOXY_PCORR"].attrs["units"] = self.data["UNCAL_PHASE_DOXY"].attrs.get(
            "units", "degree"
        )
        self.data["UNCAL_PHASE_DOXY_PCORR"].attrs["long_name"] = "Pressure-corrected uncalibrated oxygen optode phase"
        self.data["UNCAL_PHASE_DOXY_PCORR"].attrs["standard_name"] = "UNCAL_PHASE_DOXY_PCORR"

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {
                "UNCAL_PHASE_DOXY_PCORR_QC": [
                    f"{self.optode_pressure_name}_QC",
                    "UNCAL_PHASE_DOXY_QC",
                ]
            }
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        _plot_diff(
            self.data,
            "UNCAL_PHASE_DOXY",
            "UNCAL_PHASE_DOXY_PCORR",
            self.optode_pressure_name,
            self.step_name,
        )


def _plot_shift_diff(data, raw_var, shifted_var, pressure_var, step_name, lag_label):
    # Single panel: TIME vs pressure_var (inverted), coloured by (shifted - raw); the
    # raw/shifted overlay isn't useful here since a good shift looks almost identical to raw.
    if raw_var not in data or shifted_var not in data or pressure_var not in data:
        return

    matplotlib.use("tkagg")
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]

    good = _qc_good_mask(data, raw_var) & _qc_good_mask(data, shifted_var)
    time, pres = data["TIME"].values, data[pressure_var].values
    diff = data[shifted_var].values - data[raw_var].values
    finite = np.isfinite(diff) & np.isfinite(pres) & good

    sc = ax.scatter(
        time[finite], pres[finite], c=diff[finite], cmap="viridis",
        s=fig_spec.MARKER, alpha=fig_spec.ALPHA, rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f"{shifted_var} - {raw_var}", fontsize=fig_spec.FS_LABEL)
    cbar.ax.tick_params(labelsize=fig_spec.FS_TICK)

    ax.plot([], [], ls="", label=f"lag: {lag_label}")
    ax.legend(fontsize=fig_spec.FS_LEGEND, loc="upper right", framealpha=0.9)

    fig_spec.date_axis(ax, which="x")
    ylabel = fig_spec.axis_label(pressure_var, data[pressure_var].attrs.get("units"))
    fig_spec.style_axes(ax, ylabel=ylabel)
    ax.invert_yaxis()

    fig_spec.finish(fig, suptitle=f"{step_name} Diagnostics")
    plt.show(block=True)


@register_step
class ShiftOxygenToCTD(BaseStep, QCHandlingMixin):

    step_name = "Shift Oxygen To CTD"

    parameter_schema = {
        "shift_vars": {
            "type": list,
            "required": True,
            "description": "Names of optode variables (e.g. phase, optode temperature) to time-shift onto the CTD sampling grid.",
        },
        "lag_seconds": {
            "type": float,
            "default": None,
            "description": "Constant geometric time lag (s), the water travel time between optode and CTD. If not set, the lag is instead derived per profile from pitch and dive rate.",
        },
        "pitch_name": {
            "type": str,
            "default": "GLIDER_PITCH",
            "description": "Name of the glider pitch variable (radians), used to derive a per-profile lag when 'lag_seconds' is not set.",
        },
        "distance_cm": {
            "type": float,
            "default": 90.0,
            "description": "Distance between the CTD and oxygen optode along the glider body, in centimetres. Used when deriving a per-profile lag.",
        },
        "cast_id_var": {
            "type": str,
            "default": "PROFILE_DIRECTION",
            "options": ["PROFILE_DIRECTION", "SCI_PHASE"],
            "description": "Variable distinguishing up- and downcasts, used when deriving a per-profile lag.",
        },
    }

    def _derive_profile_lag(self):
        # Per-profile geometric time lag (s) from mean pitch and dive rate, following Woo & Gourcuff (2023).
        valid_casts = {"PROFILE_DIRECTION": (-1, 1), "SCI_PHASE": (1, 2)}[self.cast_id_var]

        df = pd.DataFrame(
            {
                "PROFILE_NUMBER": self.data["PROFILE_NUMBER"].values,
                "CAST": self.data[self.cast_id_var].values,
                "PITCH": self.data[self.pitch_name].values,
                "GRADIENT": self.data["PROFILE_GRADIENT"].values,
            }
        ).dropna(subset=["PROFILE_NUMBER", "CAST"])
        df = df.loc[df["CAST"].isin(valid_casts)]

        per_cast = df.groupby(["CAST", "PROFILE_NUMBER"])[["PITCH", "GRADIENT"]].mean().reset_index()
        velocity = per_cast["GRADIENT"] / np.sin(per_cast["PITCH"])
        per_cast["LAG"] = (self.distance_cm / 100) / velocity

        # Low-pass filter (per cast direction) needs a handful of profiles to be meaningful.
        lag_lookup = {}
        for _, group in per_cast.groupby("CAST"):
            group = group.sort_values("PROFILE_NUMBER")
            filled = group["LAG"].ffill().bfill()
            if filled.notna().sum() >= 7:
                b, a = butter(N=3, Wn=1 / 30, btype="low", fs=1)
                filled = pd.Series(filtfilt(b, a, filled.to_numpy()), index=filled.index)
            lag_lookup.update(dict(zip(group["PROFILE_NUMBER"], filled)))

        return lag_lookup

    def _shift_onto_grid(self, var, lag_lookup):
        epoch_s = self.data["TIME"].values.astype("datetime64[ns]").astype("int64") / 1e9
        values = self.data[var].values.astype(float)
        valid = ~np.isnan(values) & ~np.isnan(epoch_s)

        if valid.sum() < 2:
            self.log_warn(f"'{var}' has fewer than two valid samples; cannot shift.")
            return np.full(epoch_s.shape, np.nan)

        src_time = epoch_s[valid]
        src_values = values[valid]
        order = np.argsort(src_time)
        src_time, src_values = src_time[order], src_values[order]

        if lag_lookup is None:
            lag = self.lag_seconds
        else:
            lag = np.array([lag_lookup.get(p, np.nan) for p in self.data["PROFILE_NUMBER"].values])

        # Value at CTD time t is the optode measurement of the same water parcel, taken 'lag' seconds later.
        query_time = epoch_s + lag
        shifted = np.interp(query_time, src_time, src_values, left=np.nan, right=np.nan)
        shifted[np.isnan(query_time)] = np.nan

        return shifted

    def run(self):
        """
        Example
        -------
        ::

            - name: "Shift Oxygen To CTD"
              parameters:
                # <MANDATORY>
                shift_vars: ["UNCAL_PHASE_DOXY", "TEMP_DOXY"]
                # <OPTIONAL>
                lag_seconds: null
                pitch_name: "GLIDER_PITCH"
                distance_cm: 90
                cast_id_var: "PROFILE_DIRECTION"
              diagnostics: false

        Geometrically time-shifts optode measurements (e.g. phase, optode
        temperature) onto the CTD's dense sampling grid, correcting for the
        water travel time between the two sensors' positions on the glider
        (Woo & Gourcuff, 2023). This is the reverse of the more common
        approach of cloning CTD data onto the sparse optode grid: it produces
        one shifted oxygen value per ``N_MEASUREMENTS`` row, named
        ``<variable>_SHIFTED``.

        If ``lag_seconds`` is not set, the lag is instead derived per profile
        from mean pitch and dive rate. If ``pitch_name`` cannot be found (or
        is empty), a warning is logged and the step fails, since no lag is
        then available; set ``lag_seconds`` for a constant lag instead.

        Returns
        -------

        """

        self.filter_qc()

        check_config(self, ("shift_vars", "distance_cm", "cast_id_var"))
        for var in self.shift_vars:
            if var not in self.data.data_vars:
                raise KeyError(f"[{self.step_name}] '{var}' is missing from the data")
        if "PROFILE_NUMBER" not in self.data.data_vars:
            raise KeyError(f"[{self.step_name}] PROFILE_NUMBER required but is missing from the data")

        lag_lookup = None
        if self.lag_seconds is not None:
            self.log(f"Using constant lag of {self.lag_seconds} s.")
            self._lag_label = f"{self.lag_seconds:.3f}s (constant)"
        else:
            if self.pitch_name not in self.data.data_vars or bool(
                self.data[self.pitch_name].isnull().all()
            ):
                self.log_warn(
                    f"'{self.pitch_name}' not found or empty; cannot derive a per-profile lag. "
                    "Set 'lag_seconds' for a constant lag instead."
                )
                raise KeyError(
                    f"[{self.step_name}] No lag available: '{self.pitch_name}' missing and 'lag_seconds' not set."
                )
            if "PROFILE_GRADIENT" not in self.data.data_vars:
                raise KeyError(
                    f"[{self.step_name}] PROFILE_GRADIENT required but is missing from the data"
                )
            if self.cast_id_var not in self.data.data_vars:
                raise KeyError(f"[{self.step_name}] {self.cast_id_var} required but is missing from the data")

            lag_lookup = self._derive_profile_lag()
            mean_lag = np.nanmean(list(lag_lookup.values())) if lag_lookup else np.nan
            self._lag_label = f"per-profile (mean {mean_lag:.3f}s)" if np.isfinite(mean_lag) else "per-profile"

        for var in self.shift_vars:
            out_name = f"{var}_SHIFTED"
            if out_name in self.data.data_vars:
                self.log_warn(f"{out_name} already exists in the data. Overwriting...")
            self.data[out_name] = (("N_MEASUREMENTS",), self._shift_onto_grid(var, lag_lookup))
            self.data[out_name].attrs["units"] = self.data[var].attrs.get("units")
            parent_long_name = self.data[var].attrs.get("long_name", var)
            self.data[out_name].attrs["long_name"] = f"{parent_long_name}, geometrically shifted onto the CTD grid"
            self.data[out_name].attrs["standard_name"] = out_name

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc({f"{var}_SHIFTED_QC": [f"{var}_QC"] for var in self.shift_vars})

        # The shift evaluates every row via interpolation, so a valid shifted value is
        # never really its parent's original flag (e.g. still-missing 9) - it's "changed".
        for var in self.shift_vars:
            qc_name = f"{var}_SHIFTED_QC"
            has_value = self.data[f"{var}_SHIFTED"].notnull()
            self.data[qc_name] = xr.where(has_value, 5, self.data[qc_name])

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        for var in self.shift_vars:
            _plot_shift_diff(self.data, var, f"{var}_SHIFTED", "PRES", self.step_name, self._lag_label)


@register_step
class DeriveCalibratedPhase(BaseStep, QCHandlingMixin):

    step_name = "Derive Calibrated Phase"

    parameter_schema = {
        "uncalibrated_phase_name": {
            "type": str,
            "required": True,
            "description": "Name of the uncalibrated-phase variable in the dataset.",
        },
        "calib_coefficients": {
            "type": list,
            "required": True,
            "description": "Polynomial calibration coefficients (at least two).",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Derive Calibrated Phase"
              parameters:
                uncalibrated_phase_name: "UNCAL_PHASE_DOXY"
                calib_coefficients: [0, 1, 0, 0]
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check the config satisfies requirements
        check_config(self, ("uncalibrated_phase_name", "calib_coefficients"))

        # Check there are at least two coefficients for the polynomial. Fill in missing values.
        if len(self.calib_coefficients) < 2:
            raise ValueError(
                f"[{self.step_name}] At least two calibration coefficients are required."
            )
        coeffs = [0] * 4
        for i in range(len(self.calib_coefficients)):
            coeffs[i] = self.calib_coefficients[i]

        # Check if the output already exists
        if "CAL_PHASE_DOXY" in self.data.data_vars:
            self.log_warn("CAL_PHASE_DOXY already exists in the data. Overwriting...")

        # Calculate cal_phase_doxy
        cal_phase_doxy = 0
        for i, coeff in enumerate(coeffs):
            cal_phase_doxy += coeff * self.data[self.uncalibrated_phase_name] ** i
        self.data["CAL_PHASE_DOXY"] = cal_phase_doxy
        self.data["CAL_PHASE_DOXY"].attrs["units"] = self.data[self.uncalibrated_phase_name].attrs.get(
            "units", "degree"
        )
        self.data["CAL_PHASE_DOXY"].attrs["long_name"] = "Calibrated oxygen optode phase"
        self.data["CAL_PHASE_DOXY"].attrs["standard_name"] = "CAL_PHASE_DOXY"

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc({"CAL_PHASE_DOXY_QC": [f"{self.uncalibrated_phase_name}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        _plot_section(self.data, "CAL_PHASE_DOXY", "PRES", self.step_name)


@register_step
class DeriveOxygenConcentration(BaseStep, QCHandlingMixin):

    step_name = "Derive Oxygen Concentration"

    parameter_schema = {
        "method": {
            "type": str,
            "required": True,
            "options": ["poly", "SVU"],
            "description": "Conversion method: 'poly' or 'SVU'.",
        },
        "temperature_name": {
            "type": str,
            "required": True,
            "description": "Name of the temperature variable in the dataset.",
        },
        "calib_coefficient_matrix": {
            "type": list,
            "default": None,
            "description": "Calibration coefficient matrix, shape (5, 4). Required by the 'poly' method.",
        },
        "svu_coefficients": {
            "type": list,
            "default": None,
            "description": (
                "Manufacturer SVUFoilCoef0-6, required by the 'SVU' method: "
                "K_SV = c0 + c1*T + c2*T^2, P0 = c3 + c4*T, Pc = c5 + c6*CAL_PHASE_DOXY."
            ),
        },
    }

    def func_poly(self):
        # Check the calibration matrix has the right shape
        if np.shape(self.calib_coefficient_matrix) != (5, 4):
            raise ValueError(
                f"[{self.step_name}] Calib coefficient matrix must be of shape (5, 4) for method 'poly'."
            )

        # Build the internal coefficient matrix
        coeffs_matrix = np.full((5, 4), 0)
        for i, row in enumerate(self.calib_coefficient_matrix):
            coeffs_matrix[i, :] = row

        # Apply the conversion
        poly_temp = np.array(
            [self.data[self.temperature_name].values ** i for i in range(4)]
        )[np.newaxis, :, :]
        molar_doxy = (
            (poly_temp * coeffs_matrix[:, :, np.newaxis]).sum(axis=1)
            * np.array([self.data["CAL_PHASE_DOXY"].values ** i for i in range(5)])
        ).sum(axis=0)

        return molar_doxy

    def func_SVU(self):
        if len(self.svu_coefficients) != 7:
            raise ValueError(
                f"[{self.step_name}] 'svu_coefficients' must have 7 values (SVUFoilCoef0-6) for method 'SVU'."
            )
        c0, c1, c2, c3, c4, c5, c6 = self.svu_coefficients

        T = self.data[self.temperature_name].values
        phase = self.data["CAL_PHASE_DOXY"].values

        # Stern-Volmer-Uchida equation (Aanderaa manufacturer form)
        K_SV = c0 + c1 * T + c2 * T**2
        P0 = c3 + c4 * T
        Pc = c5 + c6 * phase

        return (P0 / Pc - 1.0) / K_SV

    def run(self):
        """
        Example
        -------
        ::

            - name: "Derive Oxygen Concentration"
              parameters:
                # <MANDATORY>
                method: "SVU"
                # <METHOD DEPENDENT>
                # The following params are for "SVU" method
                temperature_name: "TEMP_DOXY"
                svu_coefficients: [c0, c1, c2, c3, c4, c5, c6]  # manufacturer SVUFoilCoef0-6
              diagnostics: false

        ``method: "poly"`` instead expects ``calib_coefficient_matrix`` (shape
        (5, 4)); ``method: "SVU"`` expects ``svu_coefficients``, the 7
        manufacturer SVUFoilCoef values (Stern-Volmer-Uchida equation).

        Returns
        -------

        """

        self.filter_qc()

        methods = {
            "poly": (self.func_poly, ("temperature_name", "calib_coefficient_matrix")),
            "SVU": (self.func_SVU, ("temperature_name", "svu_coefficients")),
        }

        # Check the specified method
        check_config(self, ("method",))
        if self.method not in methods.keys():
            raise ValueError(f"[{self.step_name}] Unknown method '{self.method}'")

        # Unpack the method args and functions
        func, args = methods[self.method]

        # Check the config satisfies requirements
        check_config(self, args)

        # Check if the output already exists
        if "MOLAR_DOXY" in self.data.data_vars:
            self.log_warn("MOLAR_DOXY already exists in the data. Overwriting...")

        self.data["MOLAR_DOXY"] = (("N_MEASUREMENTS",), func())
        self.data["MOLAR_DOXY"].attrs["units"] = "micromole/l"
        self.data["MOLAR_DOXY"].attrs["long_name"] = "Molar dissolved oxygen concentration"
        self.data["MOLAR_DOXY"].attrs["standard_name"] = "MOLAR_DOXY"

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {"MOLAR_DOXY_QC": ["CAL_PHASE_DOXY_QC", f"{self.temperature_name}_QC"]}
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        _plot_section(self.data, "MOLAR_DOXY", "PRES", self.step_name)


@register_step
class MolarDOXYSalinityCorrection(BaseStep, QCHandlingMixin):

    step_name = "Molar DOXY Salinity Correction"

    parameter_schema = {
        "salinity_name": {
            "type": str,
            "required": True,
            "description": "Name of the salinity variable in the dataset.",
        },
        "temperature_name": {
            "type": str,
            "required": True,
            "description": "Name of the temperature variable in the dataset.",
        },
        "reference_salinity": {
            "type": float,
            "default": 0,
            "description": "Reference salinity the correction is computed relative to.",
        },
    }

    def oxy_solubility_salinity_correction(self):
        # Get data
        T = self.data[self.temperature_name]
        S = self.data[self.salinity_name]

        # Coefficients (Garcia & Gordon 1992 – Benson & Krause refit)
        B0 = -6.24523e-3
        B1 = -7.37614e-3
        B2 = -1.03410e-2
        B3 = -8.17083e-3
        C0 = -4.88682e-7

        # Scaled temperature term Ts
        Ts = np.log((298.15 - T) / (273.15 + T))

        # SCorr computation
        salinity_correction_factor = np.exp(
            (S - self.reference_salinity) * (B0 + B1 * Ts + B2 * (Ts**2) + B3 * (Ts**3))
            + C0 * ((S**2) - (self.reference_salinity**2))
        )

        return salinity_correction_factor

    def water_vapour_partial_pressure(self, reference_salinity=None):
        # Get data
        T = self.data[self.temperature_name]
        if reference_salinity is None:
            S = self.data[self.salinity_name]
        else:
            S = reference_salinity

        # Convert degrees C to Kelvin
        T = T + 273.15

        # Constants from polynomial equation 10 in Weiss&Price, 1980.
        A = 24.4543
        B = -67.4509
        C = -4.8489
        D = -0.000544

        # Equation 10 in Weiss&Price, 1980
        vapour_partial_pressure = 1013.25 * np.exp(
            A + B * (100 / T) + C * np.log(T / 100) + D * S
        )

        return vapour_partial_pressure

    def run(self):
        """
        Example
        -------
        ::

            - name: "Molar DOXY Salinity Correction"
              parameters:
                # <MANDATORY>
                salinity_name: "PRAC_SALINITY"
                temperature_name: "TEMP"
                # <OPTIONAL>
                reference_salinity: 0
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check the requred variable names are specified
        check_config(self, ("salinity_name", "temperature_name"))
        if "MOLAR_DOXY" not in self.data.data_vars:
            raise KeyError(
                f"[{self.step_name}] MOLAR_DOXY required but is missing from the data"
            )

        # Calculate factor with partial pressure of water vapour, following Weiss & PRice (1980)
        A = 1013.25 - self.water_vapour_partial_pressure(
            reference_salinity=self.reference_salinity
        )
        B = 1013.25 - self.water_vapour_partial_pressure()

        S_Corr = self.oxy_solubility_salinity_correction()

        MOLAR_DOXY_PSAL = (A / B) * S_Corr * self.data["MOLAR_DOXY"]

        # Apply the correction
        self.data["MOLAR_DOXY_PSAL"] = MOLAR_DOXY_PSAL
        self.data["MOLAR_DOXY_PSAL"].attrs["units"] = self.data["MOLAR_DOXY"].attrs.get(
            "units", "micromole/l"
        )
        self.data["MOLAR_DOXY_PSAL"].attrs["long_name"] = "Salinity-corrected molar dissolved oxygen concentration"
        self.data["MOLAR_DOXY_PSAL"].attrs["standard_name"] = "MOLAR_DOXY_PSAL"

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {
                "MOLAR_DOXY_PSAL_QC": [
                    f"{self.salinity_name}_QC",
                    f"{self.temperature_name}_QC",
                    "MOLAR_DOXY_QC",
                ]
            }
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        _plot_diff(self.data, "MOLAR_DOXY", "MOLAR_DOXY_PSAL", "PRES", self.step_name)


@register_step
class MolarDOXYPressureCorrection(BaseStep, QCHandlingMixin):

    step_name = "Molar DOXY Pressure Correction"

    parameter_schema = {
        "pressure_name": {
            "type": str,
            "required": True,
            "description": "Name of the pressure variable in the dataset.",
        },
        "temperature_name": {
            "type": str,
            "required": True,
            "description": "Name of the temperature variable in the dataset.",
        },
        "molar_doxy_name": {
            "type": str,
            "required": True,
            "description": "Name of the molar oxygen variable to correct.",
        },
        "uncalibrated_phase_correction_applied": {
            "type": bool,
            "required": True,
            "description": "Whether the uncalibrated-phase pressure correction was already applied (selects coefficients).",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "Molar DOXY Pressure Correction"
              parameters:
                # <MANDATORY>
                pressure_name: "PRES"
                temperature_name: "TEMP"
                molar_doxy_name: "MOLAR_DOXY_PSAL"
                uncalibrated_phase_correction_applied: true
              diagnostics: false

        Returns
        -------

        """

        self.filter_qc()

        # Check the required variable names are supplied
        check_config(
            self,
            (
                "pressure_name",
                "temperature_name",
                "molar_doxy_name",
                "uncalibrated_phase_correction_applied",
            ),
        )

        # Set the correction coefficients
        if self.uncalibrated_phase_correction_applied:
            C1, C2 = 0.00022, 0.0419
        else:
            C1, C2 = 0.00025, 0.0328

        MOLAR_DOXY_PSAL_PRES = self.data[self.molar_doxy_name] * (
            1.0
            + (
                (C1 * self.data[self.temperature_name] + C2)
                * self.data[self.pressure_name]
            )
            / 1000
        )

        # Apply the correction
        self.data["MOLAR_DOXY_PSAL_PRES"] = MOLAR_DOXY_PSAL_PRES
        self.data["MOLAR_DOXY_PSAL_PRES"].attrs["units"] = self.data[self.molar_doxy_name].attrs.get(
            "units", "micromole/l"
        )
        self.data["MOLAR_DOXY_PSAL_PRES"].attrs["long_name"] = (
            "Salinity- and pressure-corrected molar dissolved oxygen concentration"
        )
        self.data["MOLAR_DOXY_PSAL_PRES"].attrs["standard_name"] = "MOLAR_DOXY_PSAL_PRES"

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {
                "MOLAR_DOXY_PSAL_PRES_QC": [
                    f"{self.pressure_name}_QC",
                    f"{self.temperature_name}_QC",
                    f"{self.molar_doxy_name}_QC",
                ]
            }
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        _plot_diff(
            self.data,
            self.molar_doxy_name,
            "MOLAR_DOXY_PSAL_PRES",
            self.pressure_name,
            self.step_name,
        )
