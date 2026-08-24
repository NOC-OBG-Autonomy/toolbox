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

"""Pipeline step for correcting chlorophyll-a fluorescence for non-photochemical quenching."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag
import pelagos_py.utils.palettes as palettes

#### Custom imports ####
import xarray as xr
import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import linregress

CALC_SUFFIX = "__FOR_CALC"  # suffix of the QC-masked calculation-only copies; see run().

# Backscatter variables tried, in order, when 'bbp_var' is absent (despiked
# baseline preferred, raw BBP as fallbacks); the step halts if none are present.
BBP_VAR_FALLBACKS = ["BBP700_BASELINE", "BBP700", "BBP532_BASELINE", "BBP532"]

# Night-reference tuning ('hemsley2015'/'thomalla2018').
NIGHT_REF_BIN_METRES = 1.0  # depth bin (m) for averaging nighttime profiles.
HEMSLEY_REGRESSION_DEPTH = 60.0  # top-of-water depth (m) the Hemsley regression is fit over.

# Warn once if a backscatter-ratio correction lifts CHLA past this multiple of
# the profile's own max (usually a near-zero bbp inflating the CHLA/bbp ratio).
CORRECTION_WARN_FACTOR = 5.0

# Diagnostics-only tuning for the method-comparison figure: plot appearance and
# how day/night profiles are paired and scored. None of these affect the correction.
COMPARE_BIN_METRES = 5.0  # depth bin (m) for pairing day/night median fluorescence.
COMPARE_SURFACE_LIMIT_METRES = 50.0  # only bins this shallow are scored (where methods differ).
MAX_COMPARE_PROFILES = 200  # cap on day profiles run through every method.
MIDDAY_MIDNIGHT_WINDOW_HOURS = 1.5  # solar-time half-window (h) around noon/midnight for the regression.
TIMESERIES_DEPTH_LIMIT = 300.0  # max depth (m) shown in the timeseries section.
TIMESERIES_DEPTH_MIN = 50.0  # min depth (m) the dynamic section window shrinks to.
SECTION_MARKER_SIZE = 1.5  # scatter marker size for the section plots.
SECTION_MAX_POINTS = 100_000  # cap on points drawn in the bottom section panel.

# Colour/label per category in the bottom section panel, in draw order
# (later entries plot on top).
SECTION_CATEGORY_STYLE = [
    ("uncorrected", "Uncorrected", "#3b6fb5"),
    ("corrected", "Corrected", "#e6cf8b"),
]


def check_chl_variables(self, allowed_requests):
    user_request = self.apply_to
    if user_request not in self.data.data_vars:
        raise KeyError(f"The variable {user_request} does not exist in the data.")
    if user_request not in allowed_requests:
        raise KeyError(
            f"The variable {user_request} is not permitted for [{self.step_name}]"
        )

    if f"{user_request}_ADJUSTED" in self.data.data_vars:
        self.log_warn(
            f"User requested processing on {user_request} but {user_request}_ADJUSTED already exists. Using {user_request}_ADJUSTED..."
        )
        user_request = f"{user_request}_ADJUSTED"

    output_as = user_request + ("_ADJUSTED" if "_ADJUSTED" not in user_request else "")

    self.log(f"Processing {user_request}...")
    return user_request, output_as


@register_step
class chla_quenching_correction(BaseStep, QCHandlingMixin):
    """Correct non-photochemical quenching of chlorophyll fluorescence.

    Samples whose flags fall in ``calculation_flag_filter`` (by default probably-bad
    (3), bad (4) and missing (9); see ``qc_handling_settings``) inform none of the
    quantities the methods derive — quenching depth, night fl:bbp references,
    in-mixed-layer maxima, euphotic depth. They are still corrected like any other
    sample; each method just reads a NaN-masked copy of its inputs wherever it
    derives a quantity, so flagging the unstable top few metres keeps it out of the
    references while it still gets corrected.

    The mixed layer depth is read from the ``MLD`` variable, produced by a
    preceding Mixed Layer Depth step.

    Example
    -------
    ::

        - name: "CHLA Quenching"
          parameters:
            method: "xing2012"
            apply_to: "CHLA"
          diagnostics: true
    """

    step_name = "CHLA Quenching"
    # MLD, backscatter and PAR are only needed by some methods, so they are
    # checked at run time against the selected method rather than required here.
    required_variables = ["PROFILE_NUMBER", "TIME", "DEPTH", "LATITUDE", "LONGITUDE"]
    provided_variables = []
    # MLD/ZEU/Z_IPAR and every backscatter fallback are read whenever present:
    # the diagnostics method-comparison panel (_draw_method_comparison) re-runs
    # every method, not just the configured one, so all of their inputs must
    # stay available even when diagnostics is force-enabled after __init__ (see
    # report capture). Likewise the four CHLA-family names cover check_chl_
    # variables()'s "_ADJUSTED variant already exists" branch.
    optional_variables = ["MLD", "ZEU", "Z_IPAR"] + BBP_VAR_FALLBACKS + [
        "CHLA", "CHLA_ADJUSTED", "CHLA_FLUORESCENCE", "CHLA_FLUORESCENCE_ADJUSTED",
    ]
    variable_parameters = ["bbp_var", "par_var", "apply_to"]
    uses_data_subset = True

    # methods whose correction needs the solar-elevation angle (so the per-profile
    # sun inputs are only computed when one of them is selected)
    methods_requiring_sun = {
        "xing2012",
        "biermann2015",
        "xing2018",
        "hemsley2015",
        "thomalla2018",
        "swart2015",
        "sackmann2008",
    }

    parameter_schema = {
        "method": {
            "type": str,
            "default": "xing2012",
            # in order of publication (also the comparison panel order)
            "options": [
                "sackmann2008",
                "xing2012",
                "biermann2015",
                "hemsley2015",
                "swart2015",
                "thomalla2018",
                "xing2018",
            ],
            "description": (
                "Quenching correction method. Implemented: 'xing2012' (MLD-based), "
                "'biermann2015' (euphotic-depth-based, needs PAR), 'xing2018' "
                "(backscatter-based, needs BBP + PAR + MLD; see 'hybrid'), "
                "'hemsley2015' (global night fluorescence-bbp regression, needs BBP + PAR), "
                "'thomalla2018' (per-night fl:bbp ratio profile, needs BBP + PAR), "
                "'sackmann2008' (max fl:bbp ratio within the MLD, needs BBP + MLD) and "
                "'swart2015' (max fl:bbp ratio within the euphotic zone, needs BBP + PAR)."
            ),
        },
        "hybrid": {
            "type": bool,
            "default": True,
            "description": (
                "Only used when 'method' is 'xing2018'. When true (the default) the "
                "Terrats et al. (2020) X18_S08 hybrid is applied: deep-mixing "
                "profiles (iPAR=15 depth <= MLD) are corrected with the Xing (2018) "
                "S08+ method, while shallow-mixing profiles (iPAR=15 depth > MLD) "
                "instead use the XB18 sigmoid below the MLD and a uniform "
                "'bbp x R_MLD' above it. When false, the Xing (2018) S08+ method is "
                "applied to every profile regardless of the mixing regime."
            ),
        },
        "apply_to": {
            "type": str,
            "default": "CHLA",
            "description": "Name of the variable to apply the correction to.",
        },
        "bbp_var": {
            "type": str,
            "default": "BBP700_BASELINE",
            "description": (
                "Backscatter variable used by 'xing2018'/'thomalla2018'/"
                "'hemsley2015'/'sackmann2008'/'swart2015'. Defaults to the despiked "
                "'BBP700_BASELINE'; if that is "
                "absent the step falls back through "
                f"{BBP_VAR_FALLBACKS}, and halts if none are present."
            ),
        },
        "par_var": {
            "type": str,
            "default": "DOWNWELLING_PAR",
            "description": (
                "Downwelling PAR variable. The euphotic depth (ZEU) and iPAR "
                "isolume depth (Z_IPAR) are read from a preceding 'PAR Light "
                "Depths' step, not derived here; this variable is used only by the "
                "Terrats 2020 hybrid ('xing2018', hybrid=true), whose XB18 sigmoid "
                "reads the raw PAR profile at depth."
            ),
        },
        "day_min_elevation": {
            "type": float,
            "default": 0.0,
            "description": (
                "Solar elevation (degrees) above which a profile counts as "
                "daytime; the quenching correction is applied only to profiles "
                "above this. Default 0.0 (sun on the horizon); raise it (e.g. 5) "
                "to skip low-sun profiles near sunrise/sunset."
            ),
        },
        "night_max_elevation": {
            "type": float,
            "default": 0.0,
            "description": (
                "Solar elevation (degrees) below which a profile counts as "
                "nighttime; only these profiles inform the night "
                "fluorescence:bbp references used by 'hemsley2015'/'thomalla2018'. "
                "Default 0.0; lower it (e.g. -5) to build references only from "
                "profiles well after dusk. Profiles between night_max_elevation "
                "and day_min_elevation (twilight) are neither corrected nor used "
                "as references."
            ),
        },
        "max_photic_depth": {
            "type": float,
            "default": 100.0,
            "description": (
                "Depth (m) bounding the 'thomalla2018' quenching-depth search "
                "(the photic layer). Matches glidertools' max_photic_depth; unlike "
                "the euphotic depth ZEU it needs no PAR, so Thomalla runs on gliders "
                "without a PAR sensor."
            ),
        },
    }

    # ==================================================================
    # Step entry point
    # ==================================================================
    def run(self):
        self.filter_qc()

        # kept uncorrected for the diagnostics plots
        self.data_copy = self.data.copy(deep=True)

        self.apply_to, self.output_as = check_chl_variables(
            self,
            ["CHLA", "CHLA_ADJUSTED", "CHLA_FLUORESCENCE", "CHLA_FLUORESCENCE_ADJUSTED"],
        )
        if self.apply_to != self.output_as:
            self.data[self.output_as] = self.data[self.apply_to]

        method_key = self.method.lower()
        methods = {
            "xing2012": self.apply_xing2012_quenching_correction,
            "biermann2015": self.apply_biermann2015_quenching_correction,
            "hemsley2015": self.apply_hemsley2015_quenching_correction,
            "xing2018": self.apply_xing2018_quenching_correction,
            "thomalla2018": self.apply_thomalla2018_quenching_correction,
            "swart2015": self.apply_swart2015_quenching_correction,
            "sackmann2008": self.apply_sackmann2008_quenching_correction,
        }
        if method_key not in methods:
            raise KeyError(
                f"Method '{self.method}' is not supported. "
                f"Choose from: {', '.join(methods)}"
            )
        method_function = methods[method_key]

        # Each method needs a different subset of auxiliary variables; check the
        # chosen one's are present up front. ZEU (euphotic depth) and Z_IPAR
        # (isolume depth) come from a preceding 'Interpolate PAR' step.
        needs_mld = method_key in ("xing2012", "xing2018", "sackmann2008")
        needs_zeu = method_key in ("biermann2015", "hemsley2015", "swart2015")
        needs_ipar = method_key in ("xing2018",)
        needs_bbp = method_key in (
            "xing2018", "hemsley2015", "thomalla2018", "sackmann2008", "swart2015",
        )
        missing = []
        if needs_mld and "MLD" not in self.data.data_vars:
            missing.append("MLD (add a Mixed Layer Depth step beforehand)")
        if needs_zeu and "ZEU" not in self.data.data_vars:
            missing.append("ZEU (add a 'Interpolate PAR' step with compute_zeu beforehand)")
        if needs_ipar and "Z_IPAR" not in self.data.data_vars:
            missing.append("Z_IPAR (add a 'Interpolate PAR' step with compute_ipar beforehand)")
        if needs_bbp:
            resolved_bbp = self._resolve_bbp_var()
            if resolved_bbp is None:
                missing.append(
                    f"backscatter variable not found (looked for '{self.bbp_var}' and "
                    f"fallbacks {BBP_VAR_FALLBACKS}; set 'bbp_var' or add a BBP step "
                    "beforehand, e.g. 'BBP from Beta' + 'Isolate BBP Spikes')"
                )
            else:
                self.bbp_var = resolved_bbp
        if missing:
            self.halt(f"Method '{self.method}' requires: " + "; ".join(missing) + ".")

        # Per-profile median surface fix (time/lat/lon) for the solar-angle lookup.
        if method_key in self.methods_requiring_sun:
            self.sun_args = (
                self.data[["PROFILE_NUMBER", "TIME", "DEPTH", "LATITUDE", "LONGITUDE"]]
                .to_pandas()
                .dropna()
            )

            # median over the 50 shallowest samples of each profile
            self.sun_args = (
                self.sun_args.groupby("PROFILE_NUMBER", group_keys=True)
                .apply(lambda x: x.nsmallest(50, "DEPTH"), include_groups=False)
                .groupby(level="PROFILE_NUMBER")
                .agg({var: "median" for var in ["TIME", "LATITUDE", "LONGITUDE"]})
            )

        # Scalar-driven / hybrid methods are all-or-nothing over daytime profiles,
        # so gate on what every daytime profile carries (night profiles are skipped
        # by every method). Skip the solar pass entirely for the other methods.
        self._effective_hybrid = self.hybrid
        gate = needs_zeu or needs_ipar or (method_key == "xing2018" and self.hybrid)
        if gate and hasattr(self, "sun_args"):
            day_pns = [
                int(p)
                for p in self.sun_args.index
                if self._sun_elevation_for(int(p)) > self.day_min_elevation
            ]
            # halt (not silently skip) if a required per-profile scalar is missing
            if needs_zeu:
                self._require_scalar_on_days("ZEU", "interpolate_zeu", day_pns)
            if needs_ipar:
                self._require_scalar_on_days("Z_IPAR", "interpolate_ipar", day_pns)
            # The Terrats 2020 hybrid needs the raw PAR profile at depth (the XB18
            # sigmoid). If any daytime profile lacks it, disable the hybrid for the
            # whole run and fall back to pure Xing 2018 S08+ (needs only Z_IPAR).
            if method_key == "xing2018" and self.hybrid:
                n_missing = self._count_profiles_without_full_par(day_pns)
                if n_missing:
                    self._effective_hybrid = False
                    self.log_warn(
                        f"{n_missing} daytime profile(s) lack a full {self.par_var} "
                        f"profile; disabling the Terrats 2020 hybrid and applying pure "
                        f"Xing 2018 S08+ to all profiles. Measure PAR on every cast (or "
                        f"add a PAR-filling step) to enable the hybrid."
                    )

        # The night-reference methods build their references once, up front, from
        # the whole uncorrected dataset (the per-profile loop sees one profile at a
        # time). With diagnostics on, build both so they can be scored in the
        # comparison panel even when neither is the configured method.
        build_refs = {method_key} & {"hemsley2015", "thomalla2018"}
        if (
            self.diagnostics
            and hasattr(self, "sun_args")
            and self.bbp_var in self.data.data_vars
            and "ZEU" in self.data.data_vars
        ):
            build_refs |= {"hemsley2015", "thomalla2018"}
        for ref_method in build_refs:
            self._build_night_references(ref_method, quiet=self.diagnostics)

        # Subset to just the variables the chosen method needs; par_var is only
        # needed by the effective Terrats hybrid (raw PAR at depth).
        subset_vars = ["PROFILE_NUMBER", "DEPTH", self.apply_to]
        if needs_mld:
            subset_vars.append("MLD")
        if needs_bbp:
            subset_vars.append(self.bbp_var)
        if needs_zeu:
            subset_vars.append("ZEU")
        if needs_ipar:
            subset_vars.append("Z_IPAR")
        if method_key == "xing2018" and self._effective_hybrid:
            subset_vars.append(self.par_var)
        data_subset = self.data[subset_vars]

        # QC-masked copies of each input (flagged samples NaN'd). Every quantity the
        # methods *derive* reads these, so a flagged sample never informs a
        # correction, while the raw variables stay the base the correction writes
        # onto. (ZEU / Z_IPAR are already QC-masked at source.)
        calc_vars = [self.apply_to]
        if needs_bbp:
            calc_vars.append(self.bbp_var)
        # diagnostics re-run every method, so build the backscatter copy regardless
        if self.diagnostics:
            calc_vars += [
                var
                for var in (self.bbp_var,)
                if var in self.data_copy.data_vars and var not in calc_vars
            ]

        for var in calc_vars:
            usable = self.calculation_mask(["PROFILE_NUMBER", "DEPTH", var])
            calc = np.where(
                usable, np.asarray(self.data_copy[var].values, dtype=float), np.nan
            )
            # data_copy carries them too, for the diagnostics re-runs
            self.data_copy[f"{var}{CALC_SUFFIX}"] = (self.data_copy[var].dims, calc)
            if var in data_subset.data_vars:
                data_subset[f"{var}{CALC_SUFFIX}"] = (data_subset[var].dims, calc)

        # Correct one profile at a time, stitching each result back into the full data.
        profile_numbers = np.unique(
            data_subset["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS")
        )
        for profile_number in self.log_progress(profile_numbers, desc="", unit="prof"):
            profile = data_subset.where(
                data_subset["PROFILE_NUMBER"] == profile_number, drop=True
            )
            corrected_chla = method_function(profile)
            profile_indices = np.where(self.data["PROFILE_NUMBER"] == profile_number)
            self.data[self.output_as][profile_indices] = corrected_chla

        if method_key == "thomalla2018":
            counts = getattr(self, "_thomalla_debug", {})
            total = counts.get("total", 0)
            no_qd = counts.get("no_qd", 0)
            if no_qd:
                reasons = {
                    k[len("no_qd:"):]: v for k, v in counts.items() if k.startswith("no_qd:")
                }
                breakdown = ", ".join(
                    f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])
                )
                ref_no_surface = counts.get("ref_no_surface", 0)
                extra = (
                    f" ({ref_no_surface} of their night reference(s) never reach <=5 m depth.)"
                    if ref_no_surface
                    else ""
                )
                self.log_warn(
                    f"Thomalla 2018: {no_qd}/{total} daytime profile(s) could not be "
                    f"corrected (no quenching depth resolved): {breakdown}.{extra}"
                )

        self.reconstruct_data()
        self.update_qc()

        # a new _ADJUSTED output variable needs its own QC, copied from the source
        if self.apply_to != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.apply_to}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"].update(self.data)
        return self.context

    # ==================================================================
    # Shared helpers - inputs and per-profile quantities used by the methods below
    # ==================================================================
    def _calc_values(self, profile, var):
        # QC-masked copy of var: read when *deriving* a quantity, not when correcting.
        return np.asarray(profile[f"{var}{CALC_SUFFIX}"].values, dtype=float)

    @staticmethod
    def _profile_scalar(profile, name):
        # ZEU/Z_IPAR/MLD are one value per profile broadcast across it; take the first.
        vals = np.asarray(profile[name].values, dtype=float)
        finite = vals[np.isfinite(vals)]
        return float(finite[0]) if finite.size else np.nan

    def _require_scalar_on_days(self, name, toggle, day_pns):
        # Scalar-driven methods are all-or-nothing; halt if any daytime profile
        # lacks name, pointing at the 'Interpolate PAR' toggle that would fill it.
        pnum = self.data["PROFILE_NUMBER"].values
        vals = np.asarray(self.data[name].values, dtype=float)
        missing = [pn for pn in day_pns if not np.any(np.isfinite(vals[pnum == pn]))]
        if missing:
            self.halt(
                f"Method '{self.method}' needs {name} on every daytime profile, but "
                f"{len(missing)} lack it. Enable '{toggle}' in the 'Interpolate PAR' "
                f"step (or measure PAR on every cast) so {name} covers all profiles."
            )

    def _count_profiles_without_full_par(self, day_pns):
        # The XB18 sigmoid needs at least MIN_PTS finite positive PAR points (a Kd
        # fit); count daytime profiles that fall short and so can't run the hybrid.
        MIN_PTS = 4
        if self.par_var not in self.data.data_vars:
            return len(day_pns)
        pnum = self.data["PROFILE_NUMBER"].values
        par = np.asarray(self.data[self.par_var].values, dtype=float)
        depth = np.asarray(self.data["DEPTH"].values, dtype=float)
        missing = 0
        for pn in day_pns:
            sel = pnum == pn
            z, p = depth[sel], par[sel]
            if np.count_nonzero(np.isfinite(z) & np.isfinite(p) & (p > 0)) < MIN_PTS:
                missing += 1
        return missing

    def _thomalla_debug_count(self, key):
        # Per-profile counters behind the thomalla2018 QD-failure warning
        # (run()): how many daytime profiles corrected vs. why the rest didn't.
        counts = getattr(self, "_thomalla_debug", None)
        if counts is None:
            counts = self._thomalla_debug = {}
        counts[key] = counts.get(key, 0) + 1

    def _resolve_bbp_var(self):
        # Configured bbp_var if present, else the first available fallback (logged);
        # None if none are present, so the caller can halt.
        candidates = [self.bbp_var] + [b for b in BBP_VAR_FALLBACKS if b != self.bbp_var]
        for name in candidates:
            if name in self.data.data_vars:
                if name != self.bbp_var:
                    self.log(
                        f"Backscatter variable '{self.bbp_var}' not found; "
                        f"using fallback '{name}'."
                    )
                return name
        return None

    def _warn_if_correction_blows_up(self, chlf, chl_corr):
        # Warn once per run if a bbp-ratio method lifts CHLA implausibly far above
        # the input (a near-zero bbp inflating the ratio). Judged on the output, not
        # the ratio; suppressed during diagnostics so only the real correction warns.
        if getattr(self, "_blowup_warned", False) or getattr(self, "_suppress_warn", False):
            return
        chlf = np.asarray(chlf, dtype=float)
        chl_corr = np.asarray(chl_corr, dtype=float)
        ref = np.nanmax(chlf) if np.any(np.isfinite(chlf)) else np.nan
        peak = np.nanmax(chl_corr) if np.any(np.isfinite(chl_corr)) else np.nan
        if not (np.isfinite(ref) and np.isfinite(peak)) or ref <= 0:
            return
        if peak <= CORRECTION_WARN_FACTOR * ref:
            return
        self._blowup_warned = True
        self.log_warn(
            f"This correction lifted {self.apply_to} to {peak:.1f} - {peak / ref:.0f}x "
            f"the profile's own maximum of {ref:.1f}. That usually means a near-zero "
            f"{self.bbp_var} inflated the {self.apply_to}/bbp ratio and blew up the "
            f"correction. Check {self.bbp_var} is cleaned (despiked / dark-offset "
            f"corrected) before this step - e.g. run an 'Isolate BBP Spikes' step and "
            f"set 'bbp_var' to its baseline."
        )

    def _sun_elevation(self, profile):
        return self._sun_elevation_for(int(profile["PROFILE_NUMBER"][0]))

    def _sun_elevation_for(self, profile_number):
        # Solar elevation (deg) from the profile's median surface fix, memoised
        # per profile (diagnostics run every method over many profiles).
        cache = getattr(self, "_sun_cache", None)
        if cache is None:
            cache = self._sun_cache = {}
        if profile_number not in cache:
            time, lat, long = self.sun_args.loc[profile_number].to_numpy()
            time_utc = pd.to_datetime(time).tz_localize("UTC")
            solar_position = pvlib.solarposition.get_solarposition(time_utc, lat, long)
            cache[profile_number] = float(solar_position["elevation"].values[0])
        return cache[profile_number]

    def _hours_from_solar_noon(self, profile_number):
        # Hours from nearest solar noon, in [0, 12]: 0 = solar noon (max quenching),
        # 12 = solar midnight. Uses the equation of time + longitude to track the
        # sun rather than the clock. Memoised per profile.
        cache = getattr(self, "_solar_noon_cache", None)
        if cache is None:
            cache = self._solar_noon_cache = {}
        if profile_number not in cache:
            time, lat, long = self.sun_args.loc[profile_number].to_numpy()
            time_utc = pd.to_datetime(time).tz_localize("UTC")
            solpos = pvlib.solarposition.get_solarposition(time_utc, lat, long)
            eot = float(solpos["equation_of_time"].values[0])  # minutes
            utc_hours = time_utc.hour + time_utc.minute / 60 + time_utc.second / 3600
            # Local apparent solar time (hours): UTC + longitude offset + EoT.
            solar_hours = (utc_hours + long / 15.0 + eot / 60.0) % 24.0
            cache[profile_number] = abs(solar_hours - 12.0)
        return cache[profile_number]

    def _build_night_references(self, method_key, quiet=False):
        # Build the nighttime references, once, before the per-profile loop:
        #   hemsley2015  -> one global night fl-vs-bbp regression (self._hemsley_regression)
        #   thomalla2018 -> per-night depth-binned fl:bbp profiles (self._night_refs),
        #                   each day profile mapped to its most recent preceding night
        #                   (self._thomalla_day_night); earliest days use the next night.
        pns = [int(p) for p in self.sun_args.index]
        elev = {pn: self._sun_elevation_for(pn) for pn in pns}
        times = {pn: pd.to_datetime(self.sun_args.loc[pn, "TIME"]).value for pn in pns}
        pns_time = sorted(pns, key=lambda p: times[p])
        is_night = {pn: elev[pn] < self.night_max_elevation for pn in pns}

        pnum = self.data["PROFILE_NUMBER"].values
        z_all = np.asarray(self.data["DEPTH"].values, dtype=float)
        fl_all = np.asarray(self.data[self.apply_to].values, dtype=float)
        bbp_all = np.asarray(self.data[self.bbp_var].values, dtype=float)

        # The references are pure calculation, so flagged samples are dropped from
        # them entirely (day profiles are still corrected against the result).
        usable = self.calculation_mask(["DEPTH", self.apply_to, self.bbp_var])
        fl_all = np.where(usable, fl_all, np.nan)
        bbp_all = np.where(usable, bbp_all, np.nan)

        if method_key == "hemsley2015":
            night_pns = [pn for pn in pns if is_night[pn]]
            mask = np.isin(pnum, night_pns)
            z, f, b = z_all[mask], fl_all[mask], bbp_all[mask]
            sel = (
                np.isfinite(z)
                & (z >= 0)
                & (z <= HEMSLEY_REGRESSION_DEPTH)
                & np.isfinite(f)
                & np.isfinite(b)
            )
            if np.sum(sel) < 5 or np.ptp(b[sel]) == 0:
                self._hemsley_regression = None
                self.log(
                    "Hemsley 2015: too few valid nighttime fluorescence/backscatter "
                    "points to fit a regression; day profiles will be left uncorrected."
                )
                return
            fit = linregress(b[sel], f[sel])
            self._hemsley_regression = {
                "slope": float(fit.slope),
                "intercept": float(fit.intercept),
                "r2": float(fit.rvalue ** 2),
                "n": int(np.sum(sel)),
                # Raw fitted points, kept so the diagnostics can scatter them.
                "bbp": b[sel],
                "fl": f[sel],
            }
            if not quiet:
                self.log(
                    f"Hemsley 2015: night regression Chl = {fit.slope:.4g}*bbp "
                    f"+ {fit.intercept:.4g} (r2={fit.rvalue ** 2:.2f}, n={int(np.sum(sel))})."
                )
            return

        # thomalla2018: group consecutive nighttime profiles into nights.
        nights_members, current = [], []
        for pn in pns_time:
            if is_night[pn]:
                current.append(pn)
            elif current:
                nights_members.append(current)
                current = []
        if current:
            nights_members.append(current)

        night_refs = []
        raw_shallow, ref_shallow = [], []  # debug: sampled vs. post-QC shallowest depth
        for members in nights_members:
            mask = np.isin(pnum, members)
            z_raw = z_all[mask]
            z_raw_finite = z_raw[np.isfinite(z_raw)]
            if z_raw_finite.size:
                raw_shallow.append(float(np.min(z_raw_finite)))
            ref = self._bin_night(z_all[mask], fl_all[mask], bbp_all[mask])
            if ref is None:
                continue
            ref["time"] = float(np.median([times[pn] for pn in members]))
            night_refs.append(ref)
            ref_shallow.append(float(np.min(ref["z"])))
        if not quiet and raw_shallow:
            self.log(
                f"Thomalla 2018 debug: nightly shallowest sampled DEPTH "
                f"median={np.median(raw_shallow):.1f}m (min={min(raw_shallow):.1f}m); "
                f"shallowest usable (post-QC, binned) fl/bbp DEPTH "
                f"median={np.median(ref_shallow) if ref_shallow else np.nan:.1f}m "
                f"(min={min(ref_shallow) if ref_shallow else np.nan:.1f}m)."
            )

        day_night = {}
        if night_refs:
            night_times = [ref["time"] for ref in night_refs]
            for pn in (p for p in pns if elev[p] > self.day_min_elevation):
                dt = times[pn]
                preceding = [i for i, nt in enumerate(night_times) if nt <= dt]
                if preceding:
                    day_night[pn] = max(preceding, key=lambda i: night_times[i])
                else:  # earliest day profiles: no preceding night -> use the next one
                    day_night[pn] = min(
                        range(len(night_times)), key=lambda i: night_times[i]
                    )

        self._night_refs = night_refs
        self._thomalla_day_night = day_night
        if not quiet:
            self.log(
                f"Thomalla 2018: built {len(night_refs)} nighttime fl:bbp reference "
                f"profile(s) covering {len(day_night)} day profile(s)."
            )

    @staticmethod
    def _bin_night(z, fl, bbp):
        # Depth-binned {z, mean fl, fl:bbp ratio} for a night (ascending z), or
        # None if no bin has both finite mean fl and positive mean bbp.
        z = np.asarray(z, dtype=float)
        fl = np.asarray(fl, dtype=float)
        bbp = np.asarray(bbp, dtype=float)
        valid = np.isfinite(z) & (z >= 0)
        if not np.any(valid):
            return None
        keys = np.floor(z[valid] / NIGHT_REF_BIN_METRES).astype(int)
        fl_v, bbp_v = fl[valid], bbp[valid]

        centres, mean_fl, ratio = [], [], []
        for k in np.unique(keys):
            in_bin = keys == k
            f = np.nanmean(fl_v[in_bin]) if np.any(np.isfinite(fl_v[in_bin])) else np.nan
            b = np.nanmean(bbp_v[in_bin]) if np.any(np.isfinite(bbp_v[in_bin])) else np.nan
            if not (np.isfinite(f) and np.isfinite(b) and b > 0):
                continue
            centres.append((k + 0.5) * NIGHT_REF_BIN_METRES)
            mean_fl.append(f)
            ratio.append(f / b)
        if not centres:
            return None
        centres = np.asarray(centres, dtype=float)
        order = np.argsort(centres)  # np.interp needs increasing x
        return {
            "z": centres[order],
            "fl": np.asarray(mean_fl, dtype=float)[order],
            "ratio": np.asarray(ratio, dtype=float)[order],
        }


    # ==================================================================
    # Correction methods (public apply_*_quenching_correction), in order of
    # publication. Each takes a single-profile dataset and returns its corrected
    # fluorescence array, followed by the private helpers it uses.
    # ==================================================================
    def apply_sackmann2008_quenching_correction(self, profile):
        """Sackmann et al. (2008, *Biogeosciences*, 5:2839) NPQ correction.

        Within the mixed layer the maximum fluorescence-to-backscatter ratio
        ``R_max = max(F_Chl / b_bp)`` is taken as the non-quenched reference, and
        fluorescence from the surface to the depth of ``R_max`` is reset to
        ``b_bp x R_max`` (needs MLD + backscatter).
        """
        return self._apply_max_ratio_correction(profile, window="mld")

    def _apply_max_ratio_correction(self, profile, window):
        # Shared max fl:bbp-ratio correction (Sackmann window='mld' / Swart
        # window='zeu'): reset F to bbp x R_max from the surface to the depth of the
        # max ratio. Returns chlf unchanged if it can't correct; never lowers F.
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        chlf_calc = self._calc_values(profile, self.apply_to)
        bbp_calc = self._calc_values(profile, self.bbp_var)
        N = len(chlf)

        sun_angle = self._sun_elevation(profile)
        if (
            sun_angle <= self.day_min_elevation
            or N == 0
            or len(bbp) != N
            or np.all(np.isnan(chlf))
            or np.all(np.isnan(bbp))
        ):
            return chlf

        # Search window: mixed layer (Sackmann) or euphotic zone (Swart).
        if window == "mld":
            finite_mld = np.asarray(profile["MLD"].values, dtype=float)
            finite_mld = finite_mld[np.isfinite(finite_mld)]
            z_win = float(finite_mld[0]) if finite_mld.size else np.nan
        else:
            z_win = self._profile_scalar(profile, "ZEU")
        if not np.isfinite(z_win) or z_win <= 0:
            return chlf

        # R_max is a derived reference, so it is found from the calculation copies.
        within = (depth <= z_win) & np.isfinite(depth)
        fratio = np.divide(
            chlf_calc, bbp_calc, out=np.full_like(chlf_calc, np.nan), where=(bbp_calc != 0)
        )
        fratio_within = np.where(within, fratio, np.nan)
        if np.all(np.isnan(fratio_within)):
            return chlf

        idx_rmax = np.nanargmax(fratio_within)
        r_max = fratio[idx_rmax]
        rmax_depth = float(depth[idx_rmax])

        chl_corr = np.copy(chlf)
        fill = (depth <= rmax_depth) & np.isfinite(bbp) & (~np.isnan(chlf))
        chl_corr[fill] = bbp[fill] * r_max

        # never let the correction reduce fluorescence (fmax ignores NaNs)
        result = np.fmax(chlf, chl_corr)
        self._warn_if_correction_blows_up(chlf, result)
        return result

    def apply_xing2012_quenching_correction(self, profile):
        """Xing et al. (2012, *JGR–Oceans*, 117:C01019) NPQ correction.

        The maximum fluorescence within the mixed-layer depth (MLD) is taken as
        the non-quenched reference; all shallower values are lifted to it.
        Applied only in daytime (needs MLD).
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        N = len(chlf)

        mld_values = np.asarray(profile["MLD"].values, dtype=float)
        finite_mld = mld_values[np.isfinite(mld_values)]
        mld = float(finite_mld[0]) if finite_mld.size else np.nan

        profile_number = int(profile["PROFILE_NUMBER"][0])
        time, lat, long = self.sun_args.loc[profile_number].to_numpy()
        time_utc = pd.to_datetime(time).tz_localize("UTC")
        solar_position = pvlib.solarposition.get_solarposition(time_utc, lat, long)
        sun_angle = solar_position["elevation"].values
        if (
            sun_angle <= self.day_min_elevation
            or N == 0
            or len(depth) != N
            or not np.isfinite(mld)
            or mld <= 0
            or np.all(np.isnan(chlf))
        ):
            return chlf

        within_mld = depth <= mld
        if not np.any(within_mld):
            return chlf

        # reference max is derived, so flagged samples cannot supply it
        chlf_mld = np.where(within_mld, self._calc_values(profile, self.apply_to), np.nan)
        if np.all(np.isnan(chlf_mld)):
            return chlf
        idx_max, chlf_max = np.nanargmax(chlf_mld), np.nanmax(chlf_mld)
        chlf_max_depth = float(depth[idx_max])

        # flatten everything shallower than the reference up to that max
        chl_corr = np.copy(chlf)
        chl_corr[(depth <= chlf_max_depth) & (~np.isnan(chlf))] = chlf_max

        return chl_corr

    def apply_biermann2015_quenching_correction(self, profile):
        """Biermann et al. (2015, *Ocean Science*, 11:83-91) NPQ correction.

        Like Xing 2012 but the reference is the maximum fluorescence within the
        euphotic zone (surface to ZEU, the 1% light level) rather than the mixed
        layer. Applied only in daytime (needs ZEU).
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        N = len(chlf)

        sun_angle = self._sun_elevation(profile)
        zeu = self._profile_scalar(profile, "ZEU")

        if (
            sun_angle <= self.day_min_elevation
            or N == 0
            or len(depth) != N
            or not np.isfinite(zeu)
            or zeu <= 0
            or np.all(np.isnan(chlf))
        ):
            return chlf

        # reference max is derived, so flagged samples cannot supply it
        chlf_calc = self._calc_values(profile, self.apply_to)
        within_zeu = depth <= zeu
        chlf_within = np.where(within_zeu, chlf_calc, np.nan)
        if np.all(np.isnan(chlf_within)):
            return chlf

        idx_max = np.nanargmax(chlf_within)
        z_qd = depth[idx_max]
        f_max = chlf_calc[idx_max]

        chl_corr = np.copy(chlf)
        chl_corr[(depth <= z_qd) & (~np.isnan(chlf))] = f_max

        return chl_corr

    def apply_hemsley2015_quenching_correction(self, profile):
        """Hemsley et al. (2015, *Biogeosciences*, 12:7093) NPQ correction.

        One global regression of nighttime fluorescence against backscatter,
        ``Chl = m*bbp + c``, is fit once for the whole deployment, then applied to
        every daytime profile over the euphotic zone ``0 <= z <= ZEU`` (needs
        backscatter + ZEU).
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        N = len(chlf)

        regression = getattr(self, "_hemsley_regression", None)
        sun_angle = self._sun_elevation(profile)
        zeu = self._profile_scalar(profile, "ZEU")
        if (
            sun_angle <= self.day_min_elevation
            or regression is None
            or N == 0
            or len(bbp) != N
            or not np.isfinite(zeu)
            or zeu <= 0
            or np.all(np.isnan(chlf))
        ):
            return chlf

        m, c = regression["slope"], regression["intercept"]
        chl_corr = np.copy(chlf)
        # Replace fluorescence over the euphotic zone with the bbp-based estimate.
        fill = (depth >= 0) & (depth <= zeu) & np.isfinite(bbp) & (~np.isnan(chlf))
        chl_corr[fill] = m * bbp[fill] + c

        return chl_corr

    def apply_swart2015_quenching_correction(self, profile):
        """Swart et al. (2015, *J. Plankton Res.*, 37:635) NPQ correction.

        Same scheme as Sackmann 2008, but the maximum fluorescence-to-backscatter
        ratio is sought within the euphotic zone (surface to ZEU) rather than the
        mixed layer (needs backscatter + ZEU).
        """
        return self._apply_max_ratio_correction(profile, window="zeu")

    def apply_thomalla2018_quenching_correction(self, profile):
        """Thomalla et al. (2018, *L&O: Methods*, 16:132) NPQ correction.

        Each daytime profile is corrected against its most recent preceding
        night's mean fl:bbp ratio profile. Above the quenching depth QD,
        fluorescence is reset to ``(Fl_NT/bbp_NT) * bbp_DT``, kept only where that
        raises it. QD comes from the night-minus-day fluorescence difference
        within the photic layer (shallower than ``max_photic_depth``, following
        glidertools; needs backscatter, no PAR).
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        N = len(chlf)

        profile_number = int(profile["PROFILE_NUMBER"][0])
        day_night = getattr(self, "_thomalla_day_night", {})
        ref_idx = day_night.get(profile_number)
        sun_angle = self._sun_elevation(profile)
        if (
            sun_angle <= self.day_min_elevation
            or ref_idx is None
            or N == 0
            or len(bbp) != N
            or np.all(np.isnan(chlf))
        ):
            return chlf

        ref = self._night_refs[ref_idx]
        self._thomalla_debug_count("total")
        if float(np.min(ref["z"])) > 5.0:
            self._thomalla_debug_count("ref_no_surface")

        # Night fl:bbp ratio and mean fluorescence interpolated onto day depths;
        # NaN outside the night's sampled depth range rather than clamping to the
        # nearest endpoint, so unsampled depths can't manufacture a night-day diff.
        ratio_at_z = np.interp(depth, ref["z"], ref["ratio"], left=np.nan, right=np.nan)
        fl_night_at_z = np.interp(depth, ref["z"], ref["fl"], left=np.nan, right=np.nan)

        # QD is derived, so the quenched top of the profile cannot set it if flagged.
        qd, reason = self._quenching_depth(
            depth, self._calc_values(profile, self.apply_to), fl_night_at_z,
            self.max_photic_depth,
        )
        if not np.isfinite(qd):
            self._thomalla_debug_count("no_qd")
            self._thomalla_debug_count(f"no_qd:{reason}")
            return chlf
        self._thomalla_debug_count("corrected")

        corrected = ratio_at_z * bbp
        chl_corr = np.copy(chlf)
        # Correct from the surface to the quenching depth, keeping the result
        # only where it raises the (quenched) daytime fluorescence.
        fill = (
            (depth >= 0)
            & (depth <= qd)
            & np.isfinite(bbp)
            & (~np.isnan(chlf))
            & np.isfinite(corrected)
            & (corrected > chlf)
        )
        chl_corr[fill] = corrected[fill]

        self._warn_if_correction_blows_up(chlf, chl_corr)
        return chl_corr

    @staticmethod
    def _quenching_depth(z, fl_day, fl_night, max_photic_depth):
        # Quenching depth QD (positive-down m), Thomalla 2018: within the photic
        # layer (0 to max_photic_depth), restricted to night > day (fl_diff > 0,
        # the quenching signal), the difference D(z) is anchored at its near-surface
        # max (top 5 m); QD is the deeper point of steepest gradient down to one of
        # the five smallest |D| or a zero crossing. Returns (qd, reason); qd is NaN
        # if unresolvable, including when there is no near-surface observation to
        # anchor on. reason is surfaced to the caller for the per-run QD-failure
        # warning (run()).
        z = np.asarray(z, dtype=float)
        D = np.asarray(fl_night, dtype=float) - np.asarray(fl_day, dtype=float)
        mask_all = np.isfinite(z) & np.isfinite(D) & (z >= 0) & (z <= max_photic_depth)
        mask = mask_all & (D > 0)
        if np.sum(mask) < 3:
            # distinguish missing day/night coverage from a genuine lack of a
            # night > day (quenching) signal in the data that is there
            reason = (
                "too_few_valid_points" if np.sum(mask_all) < 3 else "no_positive_diff_signal"
            )
            return np.nan, reason
        zz, DD = z[mask], D[mask]
        order = np.argsort(zz)  # surface -> deep
        zz, DD = zz[order], DD[order]

        # Anchor at the largest difference near the surface (top 5 m); without one
        # there, the quenching layer can't be resolved.
        top = zz <= 5
        if not np.any(top):
            return np.nan, "no_positive_diff_within_5m"
        anchor = np.argmax(np.where(top, DD, -np.inf))
        z_a, D_a = zz[anchor], DD[anchor]

        # Candidates: five smallest |D| deeper than the anchor, plus zero crossings.
        candidates = set()
        for i in np.argsort(np.abs(DD)):
            if zz[i] > z_a:
                candidates.add(int(i))
            if len(candidates) >= 5:
                break
        for i in range(len(DD) - 1):
            crossing = DD[i] == 0 or (DD[i] > 0) != (DD[i + 1] > 0)
            if crossing and zz[i + 1] > z_a:
                candidates.add(i + 1)
        if not candidates:
            return np.nan, "no_candidates_deeper_than_anchor"

        best_qd, best_gradient = np.nan, -np.inf
        for i in candidates:
            gradient = abs(D_a - DD[i]) / (zz[i] - z_a)
            if gradient > best_gradient:
                best_gradient, best_qd = gradient, float(zz[i])
        return best_qd, "ok"

    def apply_xing2018_quenching_correction(self, profile):
        """Xing et al. (2018, *Optics Express*, 26:24734) S08+ NPQ correction,
        optionally extended by the Terrats et al. (2020, *GRL*) X18_S08 hybrid.

        The mixing regime compares the iPAR=15 isolume depth against the MLD.

        ``hybrid=False`` (pure Xing 2018)
            Every daytime profile gets S08+: within the NPQ layer (surface to the
            shallower of MLD and the iPAR=15 depth) the fl:bbp ratio is maximised
            and fluorescence is reset to ``b_bp x R_max``.

        ``hybrid=True`` (Terrats 2020, the default)
            Deep-mixing profiles (iPAR=15 depth <= MLD) get S08+ as above.
            Shallow-mixing profiles instead get the XB18 sigmoid below the MLD and
            ``b_bp x R_MLD`` above it.

        The correction never lowers fluorescence (needs MLD + backscatter + PAR).
        """
        return self._apply_xing_terrats(
            profile, hybrid=getattr(self, "_effective_hybrid", self.hybrid)
        )

    def _apply_xing_terrats(self, profile, hybrid):
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        chlf_calc = self._calc_values(profile, self.apply_to)
        bbp_calc = self._calc_values(profile, self.bbp_var)
        N = len(chlf)

        sun_angle = self._sun_elevation(profile)
        if (
            sun_angle <= self.day_min_elevation
            or N == 0
            or len(bbp) != N
            or np.all(np.isnan(chlf))
            or np.all(np.isnan(bbp))
        ):
            return chlf

        # MLD for this profile (one value, broadcast across its measurements).
        finite_mld = np.asarray(profile["MLD"].values, dtype=float)
        finite_mld = finite_mld[np.isfinite(finite_mld)]
        mld = float(finite_mld[0]) if finite_mld.size else np.nan

        # iPAR isolume depth for this profile, read from the Interpolate PAR step.
        z_ipar = self._profile_scalar(profile, "Z_IPAR")

        # Shallow mixing: light penetrates below the mixed layer.
        shallow = hybrid and np.isfinite(z_ipar) and np.isfinite(mld) and z_ipar > mld

        if not shallow:
            # --- Deep-mixing S08+ (Xing 2018) --------------------------------
            if not (np.isfinite(mld) and np.isfinite(z_ipar)):
                return chlf
            # NPQ layer: shallower than the shallower of MLD and the isolume depth.
            z_ref = min(mld, z_ipar)
            npq_layer = (depth <= z_ref) & np.isfinite(depth)
            # R_max is a derived reference, so it comes from the calculation copies.
            fratio = np.divide(
                chlf_calc, bbp_calc, out=np.full_like(chlf_calc, np.nan), where=(bbp_calc != 0)
            )
            fratio_layer = np.where(npq_layer, fratio, np.nan)
            if np.all(np.isnan(fratio_layer)):
                return chlf
            idx_rmax = np.nanargmax(fratio_layer)
            r_max = fratio[idx_rmax]

            chl_corr = np.copy(chlf)
            fill = npq_layer & np.isfinite(bbp) & (~np.isnan(chlf))
            chl_corr[fill] = bbp[fill] * r_max

        else:
            # --- Shallow-mixing X18_S08 hybrid (Terrats 2020) ----------------
            # This branch alone needs the raw PAR profile at depth (the sigmoid);
            # the run-level check disables the hybrid unless every daytime profile
            # carries it, so par_var is present here.
            ipar = np.asarray(profile[self.par_var].values, dtype=float)
            if not np.isfinite(mld) or len(ipar) != N or np.all(np.isnan(ipar)):
                return chlf
            r, ipar_mid, e = 0.092, 261.0, 2.2  # XB18 sigmoid parameters.

            chl_corr = np.copy(chlf)
            below = (depth > mld) & np.isfinite(depth)
            # Sigmoid de-quenching below the MLD; clip PAR away from zero first.
            ipar_safe = np.clip(ipar[below], 1e-3, None)
            s = r + (1 - r) / (1 + (ipar_safe / ipar_mid) ** e)
            s = np.clip(s, r, 1.0)
            chl_corr[below] = chlf[below] / s

            # Ratio at the shallowest valid point just below the MLD. This is a
            # derived reference, so a flagged sample cannot supply it.
            below_idx = np.where(below)[0]
            order = below_idx[np.argsort(depth[below_idx])]
            r_mld = np.nan
            for k in order:
                if (
                    np.isfinite(chl_corr[k])
                    and np.isfinite(chlf_calc[k])
                    and np.isfinite(bbp_calc[k])
                    and bbp_calc[k] > 0
                ):
                    r_mld = chl_corr[k] / bbp_calc[k]
                    break
            if not np.isfinite(r_mld):
                return chlf

            above = (depth <= mld) & np.isfinite(bbp) & (~np.isnan(chlf))
            chl_corr[above] = bbp[above] * r_mld

        # never let the correction reduce fluorescence (fmax ignores NaNs)
        result = np.fmax(chlf, chl_corr)
        self._warn_if_correction_blows_up(chlf, result)
        return result

    # ==================================================================
    # Diagnostics
    # ==================================================================
    # comparison-panel titles per method
    _METHOD_LABELS = {
        "none": "No correction",
        "xing2012": "Xing 2012",
        "biermann2015": "Biermann 2015",
        "xing2018": "Xing 2018",
        "hemsley2015": "Hemsley 2015",
        "thomalla2018": "Thomalla 2018",
        "swart2015": "Swart 2015",
        "sackmann2008": "Sackmann 2008",
    }

    # plain-language caption per method, shown beneath the example profile
    _METHOD_DESCRIPTIONS = {
        "xing2012": (
            "Quenching depth (QD) is the depth of max CHLA in range 0 – MLD.\n"
            "Correction sets all CHLA values above QD to that max CHLA."
        ),
        "biermann2015": (
            "Reference is the max CHLA within the euphotic zone (0 – Zeu).\n"
            "Correction lifts all CHLA above that depth to the reference."
        ),
        "xing2018": (
            "NPQ layer = 0 to the shallower of MLD and the iPAR=15 depth.\n"
            "R_max = max CHLA/bbp there; CHLA is reset to bbp × R_max."
        ),
        "hemsley2015": (
            "One global night CHLA–bbp regression is fit for the deployment.\n"
            "Daytime CHLA over the euphotic zone is set to slope × bbp + intercept."
        ),
        "thomalla2018": (
            "Each day profile uses its preceding night's CHLA:bbp ratio.\n"
            "Above QD, CHLA is set to (night CHLA:bbp) × day bbp where it raises it."
        ),
        "sackmann2008": (
            "R_max = max CHLA/bbp within the mixed layer (0 – MLD).\n"
            "From the surface to that depth, CHLA is reset to bbp × R_max."
        ),
        "swart2015": (
            "R_max = max CHLA/bbp within the euphotic zone (0 – Zeu).\n"
            "From the surface to that depth, CHLA is reset to bbp × R_max."
        ),
    }

    # appended to the 'xing2018' caption (its correction also depends on 'hybrid')
    _HYBRID_NOTES = {
        True: (
            "\nhybrid on: shallow mixing (iPAR=15 deeper than MLD) instead uses\n"
            "the XB18 sigmoid below MLD and bbp × R_MLD above it (Terrats 2020)."
        ),
        False: "\nhybrid off: every profile uses the Xing 2018 layer above.",
    }

    def generate_diagnostics(self):
        # One figure: left = method-comparison scatter grid (every method scored
        # against night profiles), top right = original/corrected depth-time
        # sections, bottom right = an example day profile for the configured method.
        mpl.use("tkagg")

        # panels re-run every method, so suppress the blow-up warning here
        self._suppress_warn = True

        if not hasattr(self, "sun_args"):
            self.log("Solar inputs unavailable; cannot build quenching diagnostics.")
            return

        fig = plt.figure(figsize=(21, 12), dpi=120)
        # tight outer margins so the panels fill the figure
        outer = fig.add_gridspec(
            1, 2, width_ratios=[1.4, 2.2], wspace=0.14,
            left=0.045, right=0.965, top=0.93, bottom=0.055,
        )

        self._draw_method_comparison(fig, outer[0, 0])
        right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.3)
        self._draw_timeseries(fig, right[0, 0])
        self._draw_example_profiles(fig, right[1, 0])

        fig.suptitle(
            f"CHLA Quenching diagnostics — method: {self.method}  "
            f"({self.apply_to} -> {self.output_as})",
            fontsize=13,
            fontweight="bold",
        )
        plt.show(block=True)

    # --- Left column: method comparison -------------------------------

    def _draw_method_comparison(self, fig, subspec):
        pairs = self._day_night_pairs()
        if not pairs:
            ax = fig.add_subplot(subspec)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No day/night profile pairs\navailable for method comparison.",
                ha="center",
                va="center",
                fontsize=9,
            )
            return

        # par_var is only needed when the Terrats hybrid is (effectively) on
        xing_needs = {"MLD", self.bbp_var, "Z_IPAR"}
        if getattr(self, "_effective_hybrid", self.hybrid):
            xing_needs = xing_needs | {self.par_var}
        implemented = {
            "xing2012": (self.apply_xing2012_quenching_correction, {"MLD"}),
            "biermann2015": (
                self.apply_biermann2015_quenching_correction,
                {"ZEU"},
            ),
            "xing2018": (
                self.apply_xing2018_quenching_correction,
                xing_needs,
            ),
            "sackmann2008": (
                self.apply_sackmann2008_quenching_correction,
                {"MLD", self.bbp_var},
            ),
            "swart2015": (
                self.apply_swart2015_quenching_correction,
                {self.bbp_var, "ZEU"},
            ),
        }
        # night-reference methods are only scorable once their references are built
        if getattr(self, "_hemsley_regression", None) is not None:
            implemented["hemsley2015"] = (
                self.apply_hemsley2015_quenching_correction,
                {self.bbp_var, "ZEU"},
            )
        if getattr(self, "_thomalla_day_night", None):
            implemented["thomalla2018"] = (
                self.apply_thomalla2018_quenching_correction,
                {self.bbp_var},
            )
        have = set(self.data_copy.data_vars)
        runnable = {k: fn for k, (fn, needs) in implemented.items() if needs <= have}

        day_pns = [d for d, _ in pairs]
        night_pns = [n for _, n in pairs]
        subsets = self._profile_subsets(set(day_pns) | set(night_pns))
        night_dv = self._raw_dv(night_pns, subsets)

        # 'none' = no-correction baseline, then each runnable method
        results = {"none": self._score(self._raw_dv(day_pns, subsets), night_dv, pairs)}
        for key, fn in runnable.items():
            day_dv = self._run_method_over(fn, day_pns, subsets)
            results[key] = self._score(day_dv, night_dv, pairs)

        # two-column grid: baseline + every method (un-runnable ones as
        # placeholders); +1 cell reserves space for the sample-size box
        panels = ["none"] + self.parameter_schema["method"]["options"]
        ncols = 2
        nrows = -(-(len(panels) + 1) // ncols)
        gl = subspec.subgridspec(nrows, ncols, hspace=0.55, wspace=0.34)

        # x-label only on the lowest scatter panel of each column
        cells = []
        bottom_scatter = {}
        for i, key in enumerate(panels):
            row, col = divmod(i, ncols)
            is_placeholder = key not in implemented and key != "none"
            has_scatter = (not is_placeholder) and bool(results.get(key))
            cells.append((row, col, key, is_placeholder))
            if has_scatter:
                bottom_scatter[col] = max(bottom_scatter.get(col, -1), row)

        for row, col, key, is_placeholder in cells:
            ax = fig.add_subplot(gl[row, col])
            placeholder = (
                f"{self._METHOD_LABELS.get(key, key)}\n(not implemented)"
                if is_placeholder
                else None
            )
            self._draw_scatter_panel(
                ax,
                self._METHOD_LABELS.get(key, key),
                results.get(key),
                show_xlabel=(row == bottom_scatter.get(col)),
                show_ylabel=(col == 0),
                placeholder=placeholder,
            )

        # spare cell shows the shared sample size (same n behind every panel)
        n_values = [r["n"] for r in results.values() if r]
        if n_values:
            spare = fig.add_subplot(gl[divmod(len(panels), ncols)])
            spare.axis("off")
            spare.text(
                0.5,
                0.5,
                f"n = {max(n_values)}\npaired day/night\nsurface depth-bin medians",
                ha="center",
                va="center",
                transform=spare.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.7", alpha=0.9),
            )

    def _day_night_pairs(self):
        # (midday, midnight) profile pairs nearest in time, Thomalla 2018 Fig. 4
        # style: only profiles within MIDDAY_MIDNIGHT_WINDOW_HOURS of solar
        # noon/midnight (the worst-case quenching extremes), capped at
        # MAX_COMPARE_PROFILES for speed.
        pns = [int(pn) for pn in self.sun_args.index]
        times = {
            pn: pd.to_datetime(self.sun_args.loc[pn, "TIME"]).value for pn in pns
        }
        from_noon = {pn: self._hours_from_solar_noon(pn) for pn in pns}
        window = MIDDAY_MIDNIGHT_WINDOW_HOURS
        midday = [pn for pn in pns if from_noon[pn] <= window]
        midnight = [pn for pn in pns if from_noon[pn] >= 12.0 - window]
        if not midday or not midnight:
            self.log(
                "Too few midday/midnight profiles for the method comparison; "
                f"widen MIDDAY_MIDNIGHT_WINDOW_HOURS (currently {window:g} h)."
            )
            return []

        midnight_times = np.array([times[pn] for pn in midnight])
        pairs = [
            (d, midnight[int(np.argmin(np.abs(midnight_times - times[d])))])
            for d in midday
        ]
        if len(pairs) > MAX_COMPARE_PROFILES:
            keep = np.linspace(0, len(pairs) - 1, MAX_COMPARE_PROFILES).astype(int)
            pairs = [pairs[i] for i in keep]
            self.log(
                f"Method comparison capped to {MAX_COMPARE_PROFILES} midday profiles "
                f"(of {len(midday)}) for speed."
            )
        return pairs

    def _profile_subsets(self, profile_numbers):
        # {pn: single-profile subset of data_copy}
        pnum = self.data_copy["PROFILE_NUMBER"].values
        subsets = {}
        for pn in profile_numbers:
            idx = np.where(pnum == pn)[0]
            if idx.size:
                subsets[pn] = self.data_copy.isel(N_MEASUREMENTS=idx)
        return subsets

    def _raw_dv(self, profile_numbers, subsets):
        # {pn: (depth, uncorrected apply_to)}
        out = {}
        for pn in profile_numbers:
            s = subsets.get(pn)
            if s is not None:
                out[pn] = (s["DEPTH"].values, s[self.apply_to].values)
        return out

    def _run_method_over(self, method_fn, profile_numbers, subsets):
        # {pn: (depth, corrected fluorescence)} from running method_fn
        out = {}
        for pn in profile_numbers:
            s = subsets.get(pn)
            if s is None:
                continue
            try:
                corrected = method_fn(s)
            except Exception:  # a single bad profile shouldn't sink the panel
                continue
            out[pn] = (s["DEPTH"].values, np.asarray(corrected, dtype=float))
        return out

    def _score(self, day_dv, night_dv, pairs):
        # Fit stats for corrected-day vs night fluorescence across pairs. Only the
        # surface bins are scored (where quenching lives); deeper bins, where day
        # already matches night, would only dilute the metric.
        max_key = int(np.floor(COMPARE_SURFACE_LIMIT_METRES / COMPARE_BIN_METRES))
        xs, ys = [], []
        for day_pn, night_pn in pairs:
            day = day_dv.get(day_pn)
            night = night_dv.get(night_pn)
            if day is None or night is None:
                continue
            day_bins = self._bin_medians(day[0], day[1])
            night_bins = self._bin_medians(night[0], night[1])
            for k in day_bins.keys() & night_bins.keys():
                if k > max_key:  # deeper than the surface window -> skip
                    continue
                xs.append(night_bins[k])
                ys.append(day_bins[k])
        return self._fit_stats(xs, ys)

    @staticmethod
    def _bin_medians(depth, values):
        # median values per COMPARE_BIN_METRES depth bin, keyed by bin
        depth = np.asarray(depth, dtype=float)
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(depth) & np.isfinite(values)
        if not np.any(mask):
            return {}
        keys = np.floor(depth[mask] / COMPARE_BIN_METRES).astype(int)
        vals = values[mask]
        return {int(k): float(np.nanmedian(vals[keys == k])) for k in np.unique(keys)}

    @staticmethod
    def _fit_stats(xs, ys):
        # regression + agreement stats of ys (day) against xs (night)
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 2 or np.ptp(x) == 0:
            return None
        fit = linregress(x, y)
        resid = y - x
        bias = float(np.mean(resid))
        mean_night = float(np.mean(x))
        return {
            "x": x,
            "y": y,
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "r2": float(fit.rvalue ** 2),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "bias": bias,
            # Bias relative to the mean night fluorescence, as a percentage.
            "bias_pct": 100.0 * bias / mean_night if mean_night != 0 else np.nan,
            "n": int(x.size),
        }

    def _draw_scatter_panel(
        self, ax, label, stats,
        show_xlabel=True, show_ylabel=True, placeholder=None,
    ):
        if placeholder is not None:
            ax.text(
                0.5,
                0.5,
                placeholder,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="0.5",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set(color="0.85")
            return

        if not stats:
            ax.text(
                0.5,
                0.5,
                f"{label}\n(insufficient data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return

        x, y = stats["x"], stats["y"]
        ax.scatter(x, y, s=7, c="#3b7dd8", alpha=0.35, edgecolors="none")
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * ((hi - lo) or 1.0)
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, ls="--", c="0.5", lw=1)  # 1:1 line
        line_x = np.array(lims)
        ax.plot(
            line_x, stats["slope"] * line_x + stats["intercept"], c="#d1495b", lw=1.6
        )  # regression fit
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        sign = "+" if stats["intercept"] >= 0 else "-"
        legend = (
            f"y={stats['slope']:.2f}x {sign} {abs(stats['intercept']):.2f}\n"
            f"RMSE={stats['rmse']:.3f}\n"
            f"Bias={stats['bias_pct']:+.1f}%\n"
            f"R$^2$={stats['r2']:.2f}"
        )
        # Stats in the top-left corner, above the 1:1 line where the scatter is
        # sparse; the wider panels keep them clear of the data.
        ax.text(
            0.03,
            0.97,
            legend,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.5,
            family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
        )
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=6)
        if show_xlabel:
            ax.set_xlabel("night F", fontsize=6.5)
        if show_ylabel:
            ax.set_ylabel("day F (corr)", fontsize=6.5)

    # --- Bottom right: example profile --------------------------------

    def _draw_example_profiles(self, fig, subspec):
        # The example day profile, with a caption beneath it.
        gm = subspec.subgridspec(2, 1, height_ratios=[1.0, 0.16], hspace=0.4)
        day_pn, _ = self._example_profiles()

        self._draw_profile_change(
            fig.add_subplot(gm[0, 0]),
            day_pn,
            f"Example day profile (#{day_pn})",
        )
        self._draw_method_description(fig.add_subplot(gm[1, 0]))

    def _draw_method_description(self, ax):
        ax.axis("off")
        text = self._METHOD_DESCRIPTIONS.get(
            self.method.lower(), "No description available for this method."
        )
        # xing2018's correction also depends on 'hybrid', so note which way it's set
        if self.method.lower() == "xing2018":
            text += self._HYBRID_NOTES[bool(self.hybrid)]
        ax.text(
            0.5,
            0.98,
            text,
            ha="center",
            va="top",
            fontsize=7.5,
            style="italic",
            color="0.35",
            linespacing=1.3,
            transform=ax.transAxes,
        )

    def _example_profiles(self):
        # day profile = the one the configured method changed most; night profile
        # = the nearest in time to it
        pnum = self.data["PROFILE_NUMBER"].values
        change = np.abs(
            self.data[self.output_as].values - self.data_copy[self.apply_to].values
        )
        change[~np.isfinite(change)] = 0.0

        elev = {int(pn): self._sun_elevation_for(int(pn)) for pn in self.sun_args.index}
        total_change = {}
        for pn in np.unique(pnum[np.isfinite(pnum)]):
            total_change[int(pn)] = float(np.nansum(change[np.where(pnum == pn)[0]]))

        day_candidates = [p for p in total_change if elev.get(p, 0) > self.day_min_elevation]
        pool = day_candidates or list(total_change)
        day_pn = max(pool, key=lambda p: total_change[p])

        night_candidates = [p for p in total_change if elev.get(p, 0) < self.night_max_elevation]
        if night_candidates:
            day_t = pd.to_datetime(self.sun_args.loc[day_pn, "TIME"]).value
            night_pn = min(
                night_candidates,
                key=lambda p: abs(pd.to_datetime(self.sun_args.loc[p, "TIME"]).value - day_t),
            )
        else:
            night_pn = day_pn
        return day_pn, night_pn

    def _draw_profile_change(self, ax, profile_number, title):
        idx = np.where(self.data["PROFILE_NUMBER"].values == profile_number)[0]
        depth = self.data["DEPTH"].values[idx]
        orig = self.data_copy[self.apply_to].values[idx]
        corr = self.data[self.output_as].values[idx]

        valid = np.isfinite(depth) & np.isfinite(orig)
        depth, orig, corr = depth[valid], orig[valid], corr[valid]
        changed = np.isfinite(corr) & (np.abs(corr - orig) > 1e-9)

        # Faint connectors show how far each corrected point moved.
        for o, c, d in zip(orig[changed], corr[changed], depth[changed]):
            ax.plot([o, c], [d, d], c="0.85", lw=0.6, zorder=1)
        ax.scatter(
            orig[~changed], depth[~changed], s=16, c="#1f9e89", label="Unchanged", zorder=2
        )
        ax.scatter(
            orig[changed], depth[changed], s=16, c="0.6", label="Original (quenched)", zorder=2
        )
        ax.scatter(
            corr[changed], depth[changed], s=16, c="#d1495b", label="Corrected", zorder=3
        )
        ax.set_xlabel(self.apply_to, fontsize=8)
        ax.set_ylabel("DEPTH", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
        # cap at 200 m and invert so the surface (where quenching acts) is at top
        bottom, top = ax.get_ylim()
        ax.set_ylim(min(top, 200.0), bottom)

    # --- Right column: depth-time sections ----------------------------

    def _draw_timeseries(self, fig, subspec):
        # dedicated thin colourbar column so ax1/ax2 (with colourbars) keep the
        # same width as ax3 (without) and their time axes stay aligned
        gr = subspec.subgridspec(
            3, 2, width_ratios=[1, 0.02], wspace=0.015, hspace=0.35
        )
        time = self.data["TIME"].values
        depth = self.data["DEPTH"].values
        orig = self.data_copy[self.apply_to].values
        corr = self.data[self.output_as].values

        # keep only valid time/depth points with a real CHLA value in the top
        # window (NaN-CHLA rows, e.g. CTD-only samples, would just bury the data)
        finite = (
            ~pd.isnull(time)
            & np.isfinite(depth)
            & np.isfinite(orig)
            & (depth <= TIMESERIES_DEPTH_LIMIT)
        )
        time, depth = time[finite], depth[finite]
        orig, corr = orig[finite], corr[finite]

        vmin, vmax = self._robust_vlim(orig, corr)

        # zoom so the deepest corrected point sits ~two-thirds down (1.5x deepest
        # QD), bounded to [TIMESERIES_DEPTH_MIN, TIMESERIES_DEPTH_LIMIT]
        line_qd = self._section_quenching_depths()
        finite_qd = line_qd[np.isfinite(line_qd)]
        if finite_qd.size:
            depth_limit = min(
                TIMESERIES_DEPTH_LIMIT,
                max(TIMESERIES_DEPTH_MIN, 1.5 * float(np.max(finite_qd))),
            )
        else:
            depth_limit = TIMESERIES_DEPTH_LIMIT

        chla_cmap = palettes.get_cmap("chlorophyll")
        ax1 = fig.add_subplot(gr[0, 0])
        sc1 = ax1.scatter(
            time, depth, c=orig, cmap=chla_cmap, vmin=vmin, vmax=vmax,
            s=SECTION_MARKER_SIZE, rasterized=True,
        )
        ax1.set_title(f"Original fluorescence (top {depth_limit:.0f} m)", fontsize=9)

        ax2 = fig.add_subplot(gr[1, 0], sharex=ax1, sharey=ax1)
        sc2 = ax2.scatter(
            time, depth, c=corr, cmap=chla_cmap, vmin=vmin, vmax=vmax,
            s=SECTION_MARKER_SIZE, rasterized=True,
        )
        ax2.set_title(f"Quenching-corrected fluorescence (top {depth_limit:.0f} m)", fontsize=9)

        # colour every section point by whether the correction touched it
        ax3 = fig.add_subplot(gr[2, 0], sharex=ax1, sharey=ax1)
        cat_time, cat_depth, cat_key = self._section_point_categories(
            depth_limit, SECTION_MAX_POINTS
        )
        for z, (key, label, color) in enumerate(SECTION_CATEGORY_STYLE):
            sel = cat_key == key
            n = int(np.count_nonzero(sel))
            if not n:
                continue
            ax3.scatter(
                cat_time[sel], cat_depth[sel], c=color,
                s=SECTION_MARKER_SIZE, rasterized=True, zorder=2 + z,
                label=f"{label} ({n})",
            )
        ax3.set_title("Quenching layer — points by correction status", fontsize=9)
        ax3.legend(fontsize=6.0, loc="lower right", framealpha=0.9, markerscale=8)

        ax1.set_ylim(depth_limit, 0)  # positive-down: surface at top

        # Colourbars go in the reserved column; ax3's cell is blanked so its
        # plot width still matches the two above it.
        for sc, row in ((sc1, 0), (sc2, 1)):
            cax = fig.add_subplot(gr[row, 1])
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label(self.apply_to, fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        fig.add_subplot(gr[2, 1]).axis("off")
        for ax in (ax1, ax2):
            plt.setp(ax.get_xticklabels(), visible=False)
        for ax in (ax1, ax2, ax3):
            ax.set_ylabel("DEPTH", fontsize=8)
            ax.tick_params(labelsize=7)
        ax3.set_xlabel("TIME", fontsize=8)
        plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")

    def _section_quenching_depths(self):
        # per-profile quenching depth (deepest changed point), used to zoom the
        # sections; NaN where the correction touched nothing
        pnum = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values
        orig = self.data_copy[self.apply_to].values
        corr = self.data[self.output_as].values
        changed = np.isfinite(corr) & np.isfinite(orig) & (np.abs(corr - orig) > 1e-9)

        qd = []
        for pn in self.sun_args.index:
            idx = np.where(pnum == pn)[0]
            if idx.size == 0:
                continue
            in_profile = changed[idx]
            qd.append(float(np.max(depth[idx][in_profile])) if np.any(in_profile) else np.nan)
        return np.asarray(qd)

    def _section_point_categories(self, depth_limit, max_points):
        # (time, depth, cat) for the bottom section panel: in-profile points with a
        # real CHLA value, within depth_limit, subsampled to max_points. cat is
        # 'corrected' where the value changed, else 'uncorrected'.
        pnum = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values
        time = self.data["TIME"].values
        orig = self.data_copy[self.apply_to].values
        corr = self.data[self.output_as].values
        changed = np.isfinite(corr) & np.isfinite(orig) & (np.abs(corr - orig) > 1e-9)

        cat = np.where(changed, "corrected", "uncorrected")

        plot_mask = (
            ~pd.isnull(time)
            & np.isfinite(depth)
            & np.isfinite(orig)
            & np.isfinite(pnum)
            & (depth <= depth_limit)
        )
        idx = np.where(plot_mask)[0]
        if idx.size > max_points:
            keep = np.linspace(0, idx.size - 1, max_points).astype(int)
            idx = idx[keep]
            self.log(
                f"Section debug panel: subsampled {int(plot_mask.sum())} points "
                f"to {max_points} for plotting."
            )
        return time[idx], depth[idx], cat[idx]

    @staticmethod
    def _robust_vlim(*arrays):
        # 2nd-98th percentile colour limits across the given arrays
        stacked = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
        stacked = stacked[np.isfinite(stacked)]
        if stacked.size == 0:
            return None, None
        return float(np.percentile(stacked, 2)), float(np.percentile(stacked, 98))
