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

"""Generic deep-value (dark offset) correction for profile data.

Estimates a constant baseline (``dark_value``) from the deepest measurements of
the first few deep profiles and subtracts it from the whole record. Written for
chlorophyll fluorescence but works for any variable with a deep signal-free
region (the offset and thresholds are all parameters)."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin

#### Custom imports ####
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pelagos_py.utils.palettes as palettes

# A depth_threshold shallower than this warns the user.
MIN_DEEP_THRESHOLD = 300

# Full-record diagnostic panel is downsampled to this many points.
MAX_DIAGNOSTIC_POINTS = 10_000


@register_step
class deep_correction(BaseStep, QCHandlingMixin):
    """
    Subtract a deep dark offset from a profile variable.

    Estimates a roughly-constant baseline (the *dark value*) from the deepest
    measurements and subtracts it from the whole record, writing the result to
    an ``_ADJUSTED`` companion variable. Written for chlorophyll but works for
    any variable with a deep, signal-free region.

    The dark value is estimated as follows:

    1. Find profiles whose ``depth_var`` reaches past ``depth_threshold``
       (deeper = larger, e.g. ``PRES > 950``) that also carry enough data:
       at least ``min_profile_points`` finite points overall and
       ``min_valid_points`` below the threshold. Take the first ``n_profiles``
       that qualify.
    2. Smooth each profile with a ``smoothing_window``-point rolling median.
    3. Keep the deep points (below the threshold) that are finite and below
       ``max_valid_value`` (guards against non-dark readings).
    4. Take each profile's minimum deep value, provided it has at least
       ``min_valid_points`` valid deep points.
    5. Use the median of those per-profile minima as the dark value.

    A ``dark_value`` supplied via config skips this estimation and is subtracted
    directly. Samples whose flags fall in ``calculation_flag_filter`` are left
    out of the estimate but still corrected — they just cannot bias it.

    Examples
    --------
    A minimal config only needs the variable to correct; every other parameter
    has a sensible default::

        - name: "Deep Correction"
          parameters:
            apply_to: "CHLA"
          diagnostics: true
    """

    step_name = "Deep Correction"
    required_variables = ["PROFILE_NUMBER"]
    provided_variables = []
    variable_parameters = ["apply_to", "depth_var"]
    uses_data_subset = True

    parameter_schema = {
        "apply_to": {
            "type": str,
            "default": "CHLA",
            "description": "Name of the variable to apply the correction to.",
        },
        "dark_value": {
            "type": float,
            "default": None,
            "description": "Dark offset to subtract; if null it is computed from the data.",
        },
        "depth_var": {
            "type": str,
            "default": "PRES",
            "description": "Vertical-coordinate variable used to find deep data (deeper = larger).",
        },
        "depth_threshold": {
            "type": float,
            "default": 950,
            "description": "Only data where depth_var exceeds this is used to compute the dark value.",
            "unit": "dbar",
        },
        "n_profiles": {
            "type": int,
            "default": 5,
            "description": "Number of deep profiles (in order of occurrence) to use.",
        },
        "smoothing_window": {
            "type": int,
            "default": 5,
            "description": "Rolling-median window (points) applied along each profile before sampling.",
        },
        "max_valid_value": {
            "type": float,
            "default": 5,
            "description": "Deep values at or above this are ignored (guards against non-dark readings).",
        },
        "min_profile_points": {
            "type": int,
            "default": 20,
            "description": "A profile must have at least this many finite points overall to be selected.",
        },
        "min_valid_points": {
            "type": int,
            "default": 3,
            "description": "A profile must have at least this many valid deep points to be selected and contribute.",
        },
    }

    def run(self):
        self.filter_qc()

        # Resolve the input/output variable names (OG1 _ADJUSTED convention).
        self.apply_to, self.output_as = self.resolve_variables()

        self.compute_dark_value()
        self.log(
            f"Dark correction value for {self.apply_to} = {self.dark_value:.6f}"
        )
        self.apply_dark_correction()

        self.reconstruct_data()
        self.update_qc()

        # Adding an _ADJUSTED variable needs its own QC.
        if self.apply_to != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.apply_to}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"].update(self.data)
        return self.context

    def resolve_variables(self):
        # OG1 convention: prefer an existing _ADJUSTED variable as input, else
        # produce one.
        apply_to = self.apply_to
        if apply_to not in self.data.data_vars:
            raise KeyError(
                f"[{self.step_name}] The variable {apply_to} does not exist in the data."
            )

        if not apply_to.endswith("_ADJUSTED") and f"{apply_to}_ADJUSTED" in self.data.data_vars:
            self.log(
                f"User requested processing on {apply_to} but {apply_to}_ADJUSTED "
                f"already exists. Using {apply_to}_ADJUSTED...",
                console=False,
            )
            apply_to = f"{apply_to}_ADJUSTED"

        output_as = apply_to if apply_to.endswith("_ADJUSTED") else f"{apply_to}_ADJUSTED"

        self.log(f"Processing {apply_to}...", console=False)
        return apply_to, output_as

    def compute_dark_value(self):
        # Per-profile diagnostics for plotting; empty when dark_value is given.
        self._profile_diagnostics = {}

        if self.dark_value is not None:
            self.log(f"Using dark value from config: {self.dark_value}", console=False)
            return

        var, depth = self.apply_to, self.depth_var
        self.log(
            f"Computing dark value from the first {self.n_profiles} profiles "
            f"reaching {depth} > {self.depth_threshold}.",
            console=False,
        )

        if self.depth_threshold < MIN_DEEP_THRESHOLD:
            self.log_warn(
                f"depth_threshold ({self.depth_threshold}) is shallower than "
                f"{MIN_DEEP_THRESHOLD} {depth}; this may not be deep enough for a "
                "signal-free dark region and could bias the dark value."
            )

        missing_vars = {"PROFILE_NUMBER", depth, var} - set(self.data.data_vars)
        if missing_vars:
            raise KeyError(
                f"[{self.step_name}] {missing_vars} could not be found in the data."
            )

        df = self.data[["PROFILE_NUMBER", depth, var]].to_pandas()

        # Flagged samples must not inform the dark value, but are still corrected
        # by it, so drop them here rather than from self.data.
        df.loc[~self.calculation_mask(["PROFILE_NUMBER", depth, var]), var] = np.nan

        df = df.dropna(subset=["PROFILE_NUMBER", depth])

        # Profiles reaching past the threshold, in order of occurrence; coverage
        # is checked below before a candidate spends an n_profiles slot.
        deep_reach = df.groupby("PROFILE_NUMBER")[depth].max()
        candidates = deep_reach[deep_reach > self.depth_threshold].index.to_numpy()
        if len(candidates) == 0:
            self.halt(
                "No profiles reach past the depth threshold. "
                "Try adjusting 'depth_threshold' or 'depth_var'."
            )

        minima = []
        skipped = []
        for profile_number in candidates:
            if len(self._profile_diagnostics) >= self.n_profiles:
                break
            profile = df[df["PROFILE_NUMBER"] == profile_number].sort_values(depth)

            # Skip sparse/empty deep profiles.
            n_points = int(profile[var].notna().sum())
            below = profile[profile[depth] > self.depth_threshold]
            n_deep_points = int(below[var].notna().sum())
            if n_points < self.min_profile_points or n_deep_points < self.min_valid_points:
                skipped.append(int(profile_number))
                continue

            # Smooth along the full profile, then sample the deep region.
            smoothed = (
                profile[var]
                .rolling(self.smoothing_window, min_periods=1, center=True)
                .median()
            )
            deep_mask = (
                (profile[depth] > self.depth_threshold)
                & np.isfinite(smoothed)
                & (smoothed < self.max_valid_value)
            )

            record = {
                "depth": profile[depth].to_numpy(),
                "raw": profile[var].to_numpy(),
                "smoothed": smoothed.to_numpy(),
                "deep_mask": deep_mask.to_numpy(),
            }
            self._profile_diagnostics[profile_number] = record

            deep_vals = smoothed[deep_mask]
            if int(deep_vals.notnull().sum()) >= self.min_valid_points:
                idxmin = deep_vals.idxmin()
                record["min_value"] = float(deep_vals.loc[idxmin])
                record["min_depth"] = float(profile.loc[idxmin, depth])
                minima.append(record["min_value"])

        if skipped:
            self.log(
                f"Skipped {len(skipped)} deep profile(s) with too few points "
                f"(need >= {self.min_profile_points} total, "
                f">= {self.min_valid_points} below {self.depth_threshold} {depth}): "
                f"{skipped}",
                console=False,
            )

        if len(minima) == 0:
            raise ValueError(
                f"[{self.step_name}] No profiles had at least {self.min_valid_points} "
                "valid deep points. Try adjusting the parameters."
            )

        self.dark_value = float(np.nanmedian(minima))
        self.log(
            f"Computed dark value: {self.dark_value:.6f} "
            f"(median of {len(minima)} profile minimums)",
            console=False,
        )
        self.log(
            f"Min values range: {np.min(minima):.6f} to {np.max(minima):.6f}",
            console=False,
        )

    def apply_dark_correction(self):
        self.data[self.output_as] = xr.DataArray(
            self.data[self.apply_to] - self.dark_value,
            dims=self.data[self.apply_to].dims,
            coords=self.data[self.apply_to].coords,
        )

        if hasattr(self.data[self.apply_to], "attrs"):
            self.data[self.output_as].attrs = self.data[self.apply_to].attrs.copy()
        self.data[self.output_as].attrs[
            "comment"
        ] = f"{self.apply_to} with dark value correction (dark_value={self.dark_value:.6f})"
        self.data[self.output_as].attrs["dark_value"] = self.dark_value

    def generate_diagnostics(self):
        # Two-panel figure: deep profiles used for the estimate (left) and the
        # whole record raw vs corrected (right).
        mpl.use("tkagg")

        if not self._profile_diagnostics:
            self.log(
                "No profile diagnostics to plot (dark value was supplied via config)."
            )
            return

        fig, (ax_prof, ax_corr) = plt.subplots(
            1, 2, figsize=(9, 5.5), dpi=150, sharey=True
        )

        cmap = mpl.cm.viridis
        n = len(self._profile_diagnostics)
        colours = [cmap(i / max(n - 1, 1)) for i in range(n)]

        # --- Left: the deep profiles used, so biofouled/invalid ones stand out.
        for colour, (profile_number, rec) in zip(
            colours, self._profile_diagnostics.items()
        ):
            ax_prof.plot(rec["raw"], rec["depth"], c=colour, alpha=0.25, lw=0.8)
            ax_prof.plot(
                rec["smoothed"],
                rec["depth"],
                c=colour,
                lw=1.4,
                label=f"Prof. {int(profile_number)}",
            )
            if "min_value" in rec:
                ax_prof.scatter(
                    rec["min_value"],
                    rec["min_depth"],
                    c=[colour],
                    edgecolors="k",
                    zorder=5,
                    s=30,
                )

        ax_prof.axhline(
            self.depth_threshold, ls="--", c="grey", lw=1, label="Depth threshold"
        )
        ax_prof.axvline(0, ls=":", c="0.4", lw=1, label="Zero")
        ax_prof.axvline(
            self.dark_value, ls="--", c="r", lw=1, label=f"Dark value ({self.dark_value:.4g})"
        )
        ax_prof.invert_yaxis()  # deeper (larger depth_var) at the bottom
        ax_prof.set_xlabel(self.apply_to, fontsize=8)
        ax_prof.set_ylabel(self.depth_var, fontsize=8)
        ax_prof.set_title("Deep profiles used for dark estimate", fontsize=9)
        ax_prof.tick_params(labelsize=7)
        ax_prof.legend(fontsize=6, loc="upper left", framealpha=0.9)

        self._add_deep_inset(ax_prof, colours)

        # --- Right: the whole record, raw (grey) vs corrected (coloured).
        self._plot_full_record(fig, ax_corr)

        fig.suptitle(f"Deep Correction — {self.apply_to}", fontsize=11)
        fig.tight_layout()
        plt.show(block=True)

    def _plot_full_record(self, fig, ax_corr):
        # Whole record vs depth: raw in grey, corrected coloured; downsampled.
        depth = np.asarray(self.data[self.depth_var].values).ravel()
        raw = np.asarray(self.data[self.apply_to].values).ravel()
        corr = np.asarray(self.data[self.output_as].values).ravel()

        finite = np.isfinite(depth) & np.isfinite(raw) & np.isfinite(corr)
        depth, raw, corr = depth[finite], raw[finite], corr[finite]

        if depth.size > MAX_DIAGNOSTIC_POINTS:
            rng = np.random.default_rng(0)
            keep = rng.choice(depth.size, MAX_DIAGNOSTIC_POINTS, replace=False)
            depth, raw, corr = depth[keep], raw[keep], corr[keep]
            n_label = f"{MAX_DIAGNOSTIC_POINTS:,} of {int(finite.sum()):,} pts"
        else:
            n_label = f"{depth.size:,} pts"

        ax_corr.scatter(raw, depth, c="0.7", s=4, alpha=0.4, label=f"Raw ({n_label})")
        cmap = palettes.cmap_for_variable(self.output_as, default="viridis")
        sc = ax_corr.scatter(
            corr, depth, c=corr, cmap=cmap, s=4, alpha=0.6, label="Corrected"
        )
        ax_corr.axvline(0, ls="--", c="k", lw=1, label="Corrected baseline")

        cbar = fig.colorbar(sc, ax=ax_corr, pad=0.02)
        cbar.set_label(self.output_as, fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        ax_corr.set_xlabel(self.output_as, fontsize=8)
        ax_corr.set_title("Full record: raw (grey) vs corrected", fontsize=9)
        ax_corr.tick_params(labelsize=7)
        ax_corr.legend(fontsize=6, loc="upper right", framealpha=0.9)

    def _add_deep_inset(self, ax_prof, colours):
        # Inset zoomed onto the deep (below-threshold) points driving the estimate.
        axins = ax_prof.inset_axes([0.56, 0.05, 0.4, 0.4])

        deep_depths, deep_vals = [], []
        for colour, rec in zip(colours, self._profile_diagnostics.values()):
            mask = rec["deep_mask"]
            depth = rec["depth"][mask]
            vals = rec["smoothed"][mask]
            axins.plot(vals, depth, c=colour, marker="o", ls="", ms=3)
            deep_depths.append(depth)
            deep_vals.append(vals)
            if "min_value" in rec:
                axins.scatter(
                    rec["min_value"],
                    rec["min_depth"],
                    c=[colour],
                    edgecolors="k",
                    zorder=5,
                    s=45,
                )
        axins.axvline(self.dark_value, ls="--", c="r")

        # Frame on the deep points plus the dark value, with a 10% margin.
        depth_cat = np.concatenate(deep_depths) if deep_depths else np.array([])
        vals_cat = np.concatenate(deep_vals) if deep_vals else np.array([])
        if depth_cat.size and np.isfinite(vals_cat).any():
            xlo = min(np.nanmin(vals_cat), self.dark_value)
            xhi = max(np.nanmax(vals_cat), self.dark_value)
            dlo, dhi = np.nanmin(depth_cat), np.nanmax(depth_cat)
            xlo, xhi = self._pad_range(xlo, xhi)
            dlo, dhi = self._pad_range(dlo, dhi)
            axins.set_xlim(xlo, xhi)
            axins.set_ylim(dhi, dlo)  # inverted: deeper at the bottom

        axins.tick_params(labelsize=6)
        axins.set_title("Below threshold (zoom)", fontsize=7)

    @staticmethod
    def _pad_range(lo, hi, frac=0.1):
        # Widen a [lo, hi] range by a fraction of its span on each side.
        span = hi - lo
        pad = (span if span > 0 else abs(hi) or 1.0) * frac
        return lo - pad, hi + pad
