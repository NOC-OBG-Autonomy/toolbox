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

"""Reconstruct PAR on casts where it was not measured, from casts where it was.

Some gliders only log downwelling PAR on one cast direction (e.g. upcasts), so
the opposite direction and any gaps carry no irradiance. Interpolating PAR
directly through time fails because the sun moves a long way between casts: a
straight line through the gap is really trying to guess the diurnal cycle, not
the water.

This step avoids that by splitting each PAR reading into the two factors that
behave differently in time::

    PAR(z, t) = surface_irradiance(t) x attenuation(z, t)

The attenuation term is a property of the water (its diffuse attenuation, Kd),
which varies slowly, so it is safe to interpolate. The surface term is
dominated by solar geometry, which is fast but deterministic, so it is computed
rather than interpolated. Concretely, for each source cast the step forms two
slowly-varying quantities and interpolates them across time:

- ``shape(z)  = log10(PAR(z) / PAR(z_ref))`` -- the attenuation structure
  relative to a fixed reference depth (surface brightness divides out).
- ``trans     = log10(PAR(z_ref) / cos(sza))`` -- the atmospheric/cloud
  transmission at the reference depth (solar geometry divides out).

At every target sample the reconstruction is then::

    PAR = 10 ** (trans_hat + shape_hat) * cos(sza)

where ``cos(sza)`` is evaluated exactly at the target's own time and position.
Only water and cloud are interpolated; the sun is supplied by astronomy. See the
validation in ``examples`` for the skill of this reconstruction against withheld
measurements.

Reconstructed samples are written into the target variable only where it has no
usable measurement, and their QC flag is set to interpolated (8) so measured and
reconstructed values stay distinguishable downstream.
"""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
from pelagos_py.utils.palettes import get_cmap

#### Custom imports ####
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cos_solar_zenith(times, lat, lon):
    """Cosine of the solar zenith angle (NOAA low-precision solar equations).

    ``times`` is a datetime64 array; ``lat``/``lon`` are degree arrays of the
    same length. Negative results mean the sun is below the horizon. Samples
    with a missing time or position return NaN.

    :meta private:
    """
    t = pd.DatetimeIndex(times)
    valid = ~np.asarray(t.isna()) & np.isfinite(lat) & np.isfinite(lon)

    hour = np.asarray(t.hour) + np.asarray(t.minute) / 60 + np.asarray(t.second) / 3600
    doy = np.asarray(t.dayofyear)
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g) + 0.001480 * np.sin(3 * g)
    )
    true_solar_min = hour * 60 + eqtime + 4 * np.asarray(lon, dtype=float)
    ha = np.deg2rad(true_solar_min / 4 - 180)
    latr = np.deg2rad(np.asarray(lat, dtype=float))
    cos_sza = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(ha)

    return np.where(valid, cos_sza, np.nan)


@register_step
class InterpolatePAR(BaseStep, QCHandlingMixin):
    """
    Reconstruct downwelling PAR on casts/gaps where it was not measured.

    PAR is split into a slowly-varying water attenuation term and a
    deterministic solar-geometry term; only the water (and cloud) part is
    interpolated in time, while the sun's position is computed exactly at each
    target sample. This lets PAR measured on (say) upcasts fill the opposite
    direction without the interpolation collapsing into a guess of the diurnal
    cycle. See the module docstring for the method.

    Reconstructed values are written into ``target_variable`` only where it
    currently lacks a usable measurement, and flagged interpolated (8).

    Parameters
    ----------
    target_variable : str, optional
        PAR variable to reconstruct in place. Default ``"DOWNWELLING_PAR"``.
    depth_variable : str, optional
        Vertical coordinate (metres, positive down) used to grid the
        attenuation structure. Default ``"DEPTH"``.
    reference_depth : float, optional
        Depth (m) at which surface brightness is divided out to form the
        attenuation shape and transmission. Default ``15.0``. This is a purely
        numerical anchor and is unrelated to any PAR isolume such as the
        Terrats et al. (2020) ``z_iPAR`` quenching depth, which is an
        irradiance level rather than a fixed depth -- do not assume the two
        should track each other.
    max_depth : float, optional
        Deepest depth (m) reconstructed. PAR below this is at the sensor noise
        floor in typical glider data. Default ``100.0``.
    depth_bin_size : float, optional
        Depth-grid resolution (m) for the per-cast attenuation profile.
        Default ``1.0``.
    min_cos_sza : float, optional
        Minimum ``cos(solar zenith angle)`` for a sample to be treated as
        daytime. Below this the sun is too low to reconstruct against and the
        target is left unfilled. Default ``0.05``.
    daylight_min_par : float, optional
        Minimum PAR (umol/m2/s) at ``reference_depth`` for a source cast to
        count as a daytime cast usable for reconstruction. Default ``1.0``.
    min_valid_bins : int, optional
        Minimum number of populated depth bins for a source cast to be used.
        Default ``30``.

    Examples
    --------
    Example usage in a pipeline configuration:

    .. code-block:: yaml

        steps:
          - name: Interpolate PAR
            parameters:
              target_variable: DOWNWELLING_PAR
              reference_depth: 15.0
              max_depth: 100.0
            diagnostics: false

    Which samples are usable *sources* is governed by ``qc_handling_settings``
    (``calculation_flag_filter``, default probably-bad (3)/bad (4)/missing (9)),
    exactly as in the other processing steps.
    """

    step_name = "Interpolate PAR"
    # DOWNWELLING_PAR (and the target/depth variables generally) are file-native
    # and validated at run time in run(), not listed here — matching CHLA
    # Quenching. The pipeline's pre-run check only understands standard and
    # step-produced variables, so listing a file-native one falsely fails it.
    required_variables = ["PROFILE_NUMBER", "TIME", "DEPTH", "LATITUDE", "LONGITUDE"]
    provided_variables = []

    parameter_schema = {
        "target_variable": {
            "type": str,
            "default": "DOWNWELLING_PAR",
            "description": "PAR variable reconstructed in place.",
        },
        "depth_variable": {
            "type": str,
            "default": "DEPTH",
            "description": "Vertical coordinate (m, positive down) used to grid attenuation.",
        },
        "reference_depth": {
            "type": float,
            "default": 15.0,
            "description": "Anchor depth (m) where surface brightness is normalised out.",
            "min": 0.0,
            "unit": "m",
        },
        "max_depth": {
            "type": float,
            "default": 100.0,
            "description": "Deepest depth (m) reconstructed.",
            "min": 0.0,
            "unit": "m",
        },
        "depth_bin_size": {
            "type": float,
            "default": 1.0,
            "description": "Depth-grid resolution (m) for the per-cast profile.",
            "min": 0.1,
            "unit": "m",
        },
        "min_cos_sza": {
            "type": float,
            "default": 0.05,
            "description": "Minimum cos(solar zenith angle) treated as daytime.",
        },
        "daylight_min_par": {
            "type": float,
            "default": 1.0,
            "description": "Minimum PAR at reference_depth for a usable source cast.",
            "unit": "umol/m2/s",
        },
        "min_valid_bins": {
            "type": int,
            "default": 30,
            "description": "Minimum populated depth bins for a usable source cast.",
        },
    }

    def run(self):
        self.log("Reconstructing PAR from Kd-normalised + clear-sky interpolation...")

        par_var = self.target_variable
        qc_var = f"{par_var}_QC"
        depth_var = self.depth_variable
        for var in (par_var, depth_var, "TIME", "PROFILE_NUMBER", "LATITUDE", "LONGITUDE"):
            if var not in self.data:
                self.halt(
                    f"'{var}' is required by Interpolate PAR but is missing from the "
                    "dataset. Run the steps that provide it first."
                )

        # A sample is a usable *source* when its PAR flag is acceptable
        # (calculation_flag_filter) and it carries a finite, positive, daytime
        # reading within the reconstruction depth range.
        usable = self.calculation_mask([par_var])

        time = self.data["TIME"].values
        depth = self.data[depth_var].values.astype(float)
        prof = self.data["PROFILE_NUMBER"].values
        par = self.data[par_var].values.astype(float)
        lat = self.data["LATITUDE"].values.astype(float)
        lon = self.data["LONGITUDE"].values.astype(float)

        cos_sza = cos_solar_zenith(time, lat, lon)
        self._cos_sza = cos_sza
        time_idx = pd.DatetimeIndex(time)
        tsec = time_idx.asi8.astype(float) / 1e9
        tsec[np.asarray(time_idx.isna())] = np.nan

        in_range = np.isfinite(depth) & (depth >= 0) & (depth <= self.max_depth)
        source = (
            usable
            & in_range
            & np.isfinite(par)
            & (par > 0)
            & (cos_sza > self.min_cos_sza)
            & np.isfinite(tsec)
        )

        edges = np.arange(0.0, self.max_depth + self.depth_bin_size, self.depth_bin_size)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ref_bin = int(np.argmin(np.abs(centres - self.reference_depth)))
        self._depth_centres = centres

        cast_time, shape, trans = self._build_source_casts(
            prof, depth, par, tsec, source, edges, centres, ref_bin
        )
        if len(cast_time) < 2:
            self.log(
                "Fewer than two usable source casts found; no PAR reconstructed. "
                "Check QC flags, daylight_min_par and the depth range."
            )
            self._interp_mask = np.zeros(par.shape, dtype=bool)
            self.context["data"] = self.data
            return self.context

        # Targets: in-range daytime samples that lack a usable measurement and
        # fall within the time span the sources bracket (no extrapolation).
        measured = usable & np.isfinite(par)
        target = (
            in_range
            & ~measured
            & (cos_sza > self.min_cos_sza)
            & np.isfinite(tsec)
            & (tsec >= cast_time[0])
            & (tsec <= cast_time[-1])
        )

        recon = self._reconstruct(
            target, tsec, depth, cos_sza, edges, centres, cast_time, shape, trans
        )

        filled = target & np.isfinite(recon) & (recon > 0)
        par_out = par.copy()
        par_out[filled] = recon[filled]
        self.data[par_var].values[:] = par_out

        if qc_var in self.data:
            qc_out = self.data[qc_var].values.copy()
            qc_out[filled] = 8
            self.data[qc_var].values[:] = qc_out
        else:
            self.log(f"'{qc_var}' not present; reconstructed points are not flagged 8.")

        self._interp_mask = filled
        self.log(
            f"Reconstructed PAR at {int(filled.sum())} samples across "
            f"{len(cast_time)} source casts (flagged interpolated=8)."
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def _build_source_casts(self, prof, depth, par, tsec, source, edges, centres, ref_bin):
        """Grid each source cast's PAR to depth, returning shape/trans per cast.

        Returns ``(cast_time, shape, trans)`` sorted ascending in time:
        ``cast_time`` (seconds), ``shape[n_cast, n_bin]`` = log10(PAR/PAR_ref),
        ``trans[n_cast]`` = log10(PAR_ref / cos_sza) at the reference depth.

        :meta private:
        """
        df = pd.DataFrame(
            {
                "prof": prof,
                "depth": depth,
                "par": par,
                "tsec": tsec,
                "cos": self._cos_sza,
            }
        )[source]
        df["bin"] = np.clip(np.digitize(df["depth"], edges) - 1, 0, len(centres) - 1)

        cast_time, shape_rows, trans_rows = [], [], []
        for _, cast in df.groupby("prof"):
            col = np.full(len(centres), np.nan)
            binned = cast.groupby("bin")["par"].mean()
            col[binned.index.to_numpy()] = binned.to_numpy()

            pos = np.isfinite(col) & (col > 0)
            if pos.sum() < self.min_valid_bins:
                continue
            # Log-linear fill of interior gaps only; never extrapolate past the
            # sampled span (that would invent data).
            idx = np.flatnonzero(pos)
            span = np.arange(idx[0], idx[-1] + 1)
            col[span] = 10 ** np.interp(
                centres[span], centres[idx], np.log10(col[idx])
            )

            ref = col[ref_bin]
            if not np.isfinite(ref) or ref <= self.daylight_min_par:
                continue
            cos_ref = np.clip(cast["cos"].median(), self.min_cos_sza, None)

            cast_time.append(cast["tsec"].min())
            shape_rows.append(np.log10(col / ref))
            trans_rows.append(np.log10(ref / cos_ref))

        if not cast_time:
            return np.array([]), np.empty((0, len(centres))), np.array([])

        order = np.argsort(cast_time)
        return (
            np.asarray(cast_time)[order],
            np.asarray(shape_rows)[order],
            np.asarray(trans_rows)[order],
        )

    def _reconstruct(self, target, tsec, depth, cos_sza, edges, centres, cast_time, shape, trans):
        """Rebuild PAR at every ``target`` sample. Returns a full-length array.

        :meta private:
        """
        recon = np.full(tsec.shape, np.nan)
        idx = np.flatnonzero(target)
        if idx.size == 0:
            return recon

        t_t = tsec[idx]
        # Transmission depends only on time -- one interpolation for all targets.
        trans_hat = np.interp(t_t, cast_time, trans)

        tbin = np.clip(np.digitize(depth[idx], edges) - 1, 0, len(centres) - 1)
        shape_hat = np.full(idx.shape, np.nan)
        for b in np.unique(tbin):
            valid = np.isfinite(shape[:, b])
            if valid.sum() < 2:
                continue
            here = tbin == b
            ct, sh = cast_time[valid], shape[valid, b]
            vals = np.interp(t_t[here], ct, sh)
            # np.interp clamps beyond the ends; null those so a bin sampled over
            # only part of the record does not flat-extrapolate its shape.
            vals[(t_t[here] < ct[0]) | (t_t[here] > ct[-1])] = np.nan
            shape_hat[here] = vals

        recon[idx] = 10 ** (trans_hat + shape_hat) * cos_sza[idx]
        return recon

    def generate_diagnostics(self):
        """Three aligned depth-time panels: measured, reconstructed, sun-normalised.

        Top: PAR before interpolation (grey where the water column was sampled
        but PAR is absent). Middle: PAR after interpolation (measured +
        reconstructed). Bottom: the same field divided by ``cos(sza)`` -- with
        the sun removed the measured and reconstructed casts should merge with
        no banding, which is the visual check that the reconstruction is
        sunlight-independent.
        """
        mpl.use("tkagg")

        par_var = self.target_variable
        cmap = get_cmap("oxygen").reversed()  # dark -> bright: reads as a light field
        par_floor = 0.1

        depth = self.data[self.depth_variable].values.astype(float)
        par = self.data[par_var].values.astype(float)
        time = self.data["TIME"].values
        cos = np.clip(self._cos_sza, self.min_cos_sza, None)
        interp = self._interp_mask
        measured_now = np.isfinite(par) & ~interp

        in_range = np.isfinite(depth) & (depth >= 0) & (depth <= self.max_depth)
        finite_t = ~pd.isnull(time)

        # Cap plotted points for responsiveness (as the salinity dashboard does).
        def thin(mask, cap=60000):
            sel = np.flatnonzero(mask)
            if sel.size > cap:
                sel = sel[:: int(np.ceil(sel.size / cap))]
            return sel

        fig, axes = plt.subplots(
            3, 1, figsize=(12, 8), dpi=120, sharex=True, sharey=True,
            constrained_layout=True,
        )
        vmin, vmax = np.log10(par_floor), None
        present = par[measured_now & in_range & (par > par_floor)]
        vmax = np.log10(np.nanpercentile(present, 99)) if present.size else 3.0

        # (1) Before interpolation: measured PAR, grey where sampled but no PAR.
        gap = thin(in_range & finite_t & ~np.isfinite(par))
        axes[0].scatter(time[gap], depth[gap], s=2, c="0.8", lw=0, label="no PAR")
        m = thin(measured_now & in_range & (par > par_floor))
        sc = axes[0].scatter(
            time[m], depth[m], s=4, c=np.log10(par[m]), cmap=cmap, vmin=vmin, vmax=vmax, lw=0
        )
        axes[0].set_title("PAR before interpolation (grey = water column with no PAR)", fontsize=9)
        axes[0].legend(loc="lower right", fontsize=7, framealpha=0.9, markerscale=3)

        # (2) After interpolation: measured + reconstructed combined.
        a = thin(np.isfinite(par) & in_range & (par > par_floor))
        axes[1].scatter(
            time[a], depth[a], s=4, c=np.log10(par[a]), cmap=cmap, vmin=vmin, vmax=vmax, lw=0
        )
        axes[1].set_title("PAR after interpolation (measured + reconstructed)", fontsize=9)

        # (3) Sun-normalised: PAR / cos(sza). No banding across the seams => the
        #     reconstruction is sunlight-independent.
        norm = par / cos
        b = thin(np.isfinite(norm) & in_range & (norm > par_floor))
        nvmax = np.log10(np.nanpercentile(norm[b], 99)) if b.size else vmax
        sc_n = axes[2].scatter(
            time[b], depth[b], s=4, c=np.log10(norm[b]), cmap=cmap,
            vmin=np.log10(par_floor), vmax=nvmax, lw=0,
        )
        axes[2].set_title(
            "PAR / cos(SZA) after interpolation (sunlight-independent; seams should vanish)",
            fontsize=9,
        )

        axes[0].set_ylim(self.max_depth, 0)
        for ax in axes:
            ax.set_ylabel("DEPTH (m)", fontsize=8)
            ax.tick_params(labelsize=7)
            for spine in ax.spines.values():
                spine.set(color="0.85")
        axes[2].set_xlabel("TIME", fontsize=8)

        fig.colorbar(sc, ax=axes[:2], pad=0.008, label="log10 PAR (umol/m2/s)")
        fig.colorbar(sc_n, ax=axes[2], pad=0.008, label="log10 PAR/cos(SZA)")
        fig.suptitle("Interpolate PAR diagnostics", fontsize=11, fontweight="bold")
        plt.show(block=True)
