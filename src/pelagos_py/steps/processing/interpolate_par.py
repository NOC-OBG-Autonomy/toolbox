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

"""Derive per-profile light depths (euphotic depth and an iPAR isolume) from PAR.

Computes two per-profile depth scalars from the downwelling PAR profile for
downstream quenching corrections, optionally interpolating each in time onto
profiles without usable PAR. See :class:`InterpolatePAR` for details.
"""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin

#### Custom imports ####
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pelagos_py.utils import fig_spec


def estimate_euphotic_depth(par, depth):
    """Euphotic depth (1% light) from a log-linear Beer-Lambert fit of PAR vs depth.

    Returns metres positive down, or ``NaN`` when the fit is invalid (fewer than
    4 valid points, non-physical slope, or beyond the ~186 m clear-water limit).
    """
    from scipy.stats import linregress

    par = np.asarray(par, dtype=float)
    depth = np.asarray(depth, dtype=float)

    # Only finite, positive PAR at finite depths can be log-fitted.
    mask = np.isfinite(par) & (par > 0) & np.isfinite(depth)
    if np.sum(mask) < 4:
        return np.nan

    z = depth[mask]
    y = np.log(par[mask])
    if z[0] > z[-1]:  # regression wants increasing depth
        z = z[::-1]
        y = y[::-1]

    slope = linregress(z, y).slope
    # Reject non-physical attenuation (too clear or too turbid).
    if not np.isfinite(slope) or slope >= -0.005 or slope <= -1.0:
        return np.nan

    zeu = 4.605 / (-slope)
    return zeu if zeu <= 186 else np.nan


def depth_of_ipar(depth, ipar, level):
    """Depth (positive-down m) where downwelling iPAR crosses ``level``, or NaN.

    Interpolates on the irregular profile grid; clamps to the deepest /
    shallowest sample when ``level`` lies outside the observed PAR range.
    """
    depth = np.asarray(depth, dtype=float)
    ipar = np.asarray(ipar, dtype=float)
    valid = np.isfinite(depth) & np.isfinite(ipar)
    if np.sum(valid) < 2:
        return np.nan
    zi = depth[valid]
    pi = ipar[valid]
    order = np.argsort(zi)  # surface -> deep
    zi = zi[order]
    pi = pi[order]
    if level <= np.min(pi):  # whole profile brighter than level -> deepest sample
        return float(zi[-1])
    if level >= np.max(pi):  # whole profile darker than level -> surface
        return 0.0
    # PAR decreases with depth; reverse so np.interp sees increasing x.
    return float(np.interp(level, pi[::-1], zi[::-1]))


@register_step
class InterpolatePAR(BaseStep, QCHandlingMixin):
    """Derive the euphotic depth ``ZEU`` and the iPAR isolume depth ``Z_IPAR``.

    Both are per-profile scalars found from the downwelling PAR profile and
    broadcast across each profile's measurements. Profiles without usable PAR
    are optionally filled by interpolating each scalar in time (``interpolate_*``);
    otherwise they stay ``NaN``.

    Parameters
    ----------
    par_var : str, optional
        Downwelling PAR variable read per profile. Default ``"DOWNWELLING_PAR"``.
    depth_variable : str, optional
        Vertical coordinate (m, positive down). Default ``"DEPTH"``.
    ipar_level : float, optional
        Irradiance level (umol m-2 s-1) whose crossing depth becomes ``Z_IPAR``.
        Default ``15.0``.
    compute_zeu, compute_ipar : bool, optional
        Whether to produce ``ZEU`` / ``Z_IPAR`` at all. Both default ``True``.
    interpolate_zeu, interpolate_ipar : bool, optional
        When ``True`` (default) the scalar is interpolated in time onto every
        profile; when ``False`` only profiles with usable PAR carry it.

    Examples
    --------
    .. code-block:: yaml

        steps:
          - name: Interpolate PAR
            parameters:
              par_var: DOWNWELLING_PAR
              ipar_level: 15.0
              interpolate_zeu: true
              interpolate_ipar: true
            diagnostics: false

    Which samples are usable is governed by ``qc_handling_settings``
    (``calculation_flag_filter``), exactly as in the other processing steps.
    """

    step_name = "Interpolate PAR"
    # DOWNWELLING_PAR is file-native, so it is validated in run() rather than by
    # the pipeline pre-run check (which only knows standard/step-produced vars).
    required_variables = ["PROFILE_NUMBER", "TIME", "DEPTH"]
    provided_variables = ["ZEU", "Z_IPAR"]

    parameter_schema = {
        "par_var": {
            "type": str,
            "default": "DOWNWELLING_PAR",
            "description": "Downwelling PAR variable read per profile.",
        },
        "depth_variable": {
            "type": str,
            "default": "DEPTH",
            "description": "Vertical coordinate (m, positive down).",
        },
        "ipar_level": {
            "type": float,
            "default": 15.0,
            "description": "Irradiance level whose crossing depth becomes Z_IPAR.",
            "min": 0.0,
            "unit": "umol/m2/s",
        },
        "compute_zeu": {
            "type": bool,
            "default": True,
            "description": "Produce the euphotic depth ZEU.",
        },
        "compute_ipar": {
            "type": bool,
            "default": True,
            "description": "Produce the iPAR isolume depth Z_IPAR.",
        },
        "interpolate_zeu": {
            "type": bool,
            "default": True,
            "description": (
                "Interpolate ZEU in time onto profiles without usable PAR "
                "(off: leave those NaN)."
            ),
        },
        "interpolate_ipar": {
            "type": bool,
            "default": True,
            "description": (
                "Interpolate Z_IPAR in time onto profiles without usable PAR "
                "(off: leave those NaN)."
            ),
        },
    }

    def run(self):
        self.log("Deriving PAR light depths (ZEU / Z_IPAR)...")

        par_var = self.par_var
        depth_var = self.depth_variable
        for var in (par_var, depth_var, "PROFILE_NUMBER", "TIME"):
            if var not in self.data:
                self.halt(
                    f"'{var}' is required by Interpolate PAR but is missing from the "
                    "dataset. Run the steps that provide it first."
                )
        if not (self.compute_zeu or self.compute_ipar):
            self.log("Neither ZEU nor Z_IPAR requested; nothing to do.")
            self.context["data"] = self.data
            return self.context

        # Flagged PAR/depth samples must not inform a scalar, so NaN them out.
        usable = self.calculation_mask([par_var, depth_var])
        par = np.where(usable, np.asarray(self.data[par_var].values, dtype=float), np.nan)
        depth = np.asarray(self.data[depth_var].values, dtype=float)
        prof = self.data["PROFILE_NUMBER"].values
        tidx = pd.DatetimeIndex(self.data["TIME"].values)
        tsec = tidx.asi8.astype(float) / 1e9
        tsec[np.asarray(tidx.isna())] = np.nan

        profiles = np.unique(prof[np.isfinite(prof.astype(float))])

        prof_tsec = {}
        zeu_calc, zipar_calc = {}, {}
        for pn in profiles:
            sel = prof == pn
            z, p, t = depth[sel], par[sel], tsec[sel]
            prof_tsec[pn] = np.nanmedian(t) if np.any(np.isfinite(t)) else np.nan
            if self.compute_zeu:
                zeu_calc[pn] = estimate_euphotic_depth(p, z)
            if self.compute_ipar:
                zipar_calc[pn] = depth_of_ipar(z, p, self.ipar_level)

        self._diag = {"prof_tsec": prof_tsec, "profiles": profiles}

        if self.compute_zeu:
            self._emit_scalar(
                "ZEU", zeu_calc, prof, profiles, prof_tsec, self.interpolate_zeu,
                long_name="Euphotic depth (1% light, positive down). NaN where undefined.",
            )
        if self.compute_ipar:
            self._emit_scalar(
                "Z_IPAR", zipar_calc, prof, profiles, prof_tsec, self.interpolate_ipar,
                long_name=(
                    f"Depth where downwelling iPAR crosses {self.ipar_level:g} "
                    "umol/m2/s (positive down). NaN where undefined."
                ),
                extra_attrs={"ipar_level": float(self.ipar_level)},
            )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def _emit_scalar(
        self, name, calc, prof, profiles, prof_tsec, interpolate, *, long_name,
        extra_attrs=None,
    ):
        # Interpolate (optionally), broadcast to N_MEASUREMENTS and write `name`.
        final = self._interpolate_scalar(calc, profiles, prof_tsec) if interpolate else dict(calc)

        n_calc = sum(np.isfinite(v) for v in calc.values())
        n_final = sum(np.isfinite(v) for v in final.values())
        if interpolate:
            self.log(
                f"{name}: computed on {n_calc}/{len(profiles)} profiles, "
                f"interpolated to {n_final}."
            )
        else:
            self.log(f"{name}: computed on {n_calc}/{len(profiles)} profiles (no interpolation).")

        broadcast = np.full(prof.shape, np.nan)
        for pn in profiles:
            broadcast[prof == pn] = final.get(pn, np.nan)

        self.data[name] = (("N_MEASUREMENTS",), broadcast)
        attrs = {"long_name": long_name, "units": "m", "standard_name": name}
        if extra_attrs:
            attrs.update(extra_attrs)
        self.data[name].attrs = attrs

        # Kept for the diagnostics: which profiles were computed vs interpolated.
        self._diag[name] = {"calc": calc, "final": final}

    @staticmethod
    def _interpolate_scalar(calc, profiles, prof_tsec):
        # Fill NaN profiles by interpolating in time; no extrapolation beyond the
        # span bracketed by computed profiles (those stay NaN).
        order = sorted(profiles, key=lambda p: (prof_tsec[p] if np.isfinite(prof_tsec[p]) else np.inf))
        times = np.array([prof_tsec[p] for p in order], dtype=float)
        vals = np.array([calc.get(p, np.nan) for p in order], dtype=float)

        anchor = np.isfinite(times) & np.isfinite(vals)
        if anchor.sum() < 2:
            return dict(calc)  # not enough to interpolate from

        interp = np.interp(times, times[anchor], vals[anchor])
        lo, hi = times[anchor][0], times[anchor][-1]
        interp[~np.isfinite(times) | (times < lo) | (times > hi)] = np.nan
        return {p: float(interp[i]) for i, p in enumerate(order)}

    def generate_diagnostics(self):
        # One panel per scalar: computed profiles vs interpolated fills over time.
        matplotlib.use("tkagg")

        names = [n for n in ("Z_IPAR", "ZEU") if n in self._diag]
        if not names:
            return
        prof_tsec = self._diag["prof_tsec"]
        profiles = self._diag["profiles"]

        fig, axes = fig_spec.new_fig(nrows=len(names), sharex=True)
        for ax, name in zip(axes[:, 0], names):
            calc = self._diag[name]["calc"]
            final = self._diag[name]["final"]

            def series(pred):
                pns = [p for p in profiles if pred(p)]
                t = pd.to_datetime([prof_tsec[p] * 1e9 for p in pns])
                y = [final.get(p, np.nan) for p in pns]
                return np.asarray(t), np.asarray(y, dtype=float)

            ct, cy = series(lambda p: np.isfinite(calc.get(p, np.nan)))
            it, iy = series(
                lambda p: not np.isfinite(calc.get(p, np.nan))
                and np.isfinite(final.get(p, np.nan))
            )
            fig_spec.points(ax, ct, cy, color=fig_spec.CATEGORY[1], label="computed")
            if it.size:
                fig_spec.points(ax, it, iy, color=fig_spec.CATEGORY[3], label="interpolated")

            fig_spec.style_axes(ax, title=name, ylabel=fig_spec.axis_label(name, "m"))
            fig_spec.date_axis(ax)
            ax.invert_yaxis()  # positive-down depth: shallower at the top
            fig_spec.legend(ax)

        fig_spec.finish(fig, suptitle="PAR light depths")
        plt.show(block=True)
