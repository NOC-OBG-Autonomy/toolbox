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

"""Pipeline step for applying a linear correction (slope/intercept) to a variable.

Generic enough for sensor cross-calibration (slope + intercept), unit conversions,
or any other affine rescaling. An optional
``expected_range`` makes the correction self-detecting: it is applied only when
the data looks like it still needs it, so the same config keeps working even after
upstream input files are fixed.
"""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step

#### Custom imports ####
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pelagos_py.utils import fig_spec


@register_step
class CorrectValues(BaseStep):
    """
    Apply an affine correction ``corrected = slope * value + intercept`` to a variable.

    The correction is conditional when ``expected_range`` is given: the median of the
    valid (non-NaN) data is compared against ``[min, max]``, and the correction is
    applied only when the median falls *outside* that range (i.e. the data still looks
    uncorrected). When ``expected_range`` is omitted the correction is always applied.

    This makes a config robust to upstream fixes: a unit conversion targeting an
    ``expected_range`` quietly skips files that already arrive in the right units.
    (CNDC does not need this: Derive CTD reads the units attribute and converts
    S/m -> mS/cm for gsw itself.)

    Parameters
    ----------
    target_variable : str
        Name of the variable to correct (e.g. ``"CNDC"``).
    output_as : str, optional
        Name to write the result under. Defaults to ``target_variable`` (in-place).
        Set it to a new name to copy/rename, e.g. ``LATITUDE_GPS`` -> ``LATITUDE``
        with ``slope: 1`` leaves the values unchanged and just exposes the new name.
    slope : float, optional
        Multiplicative factor (default ``1.0``). For a x10 unit conversion, set ``10``.
    intercept : float, optional
        Additive offset applied after scaling (default ``0.0``). Use for alignment.
    expected_range : list, optional
        ``[min, max]`` for the *corrected* variable. The correction is applied only
        when the data's median falls outside this range. If omitted, the correction
        is always applied.
    time_start, time_end : str, optional
        Restrict the correction to a TIME window (e.g. ``"2024-07-01T00:00:00"``).
        Points outside the window keep their raw value. Either bound may be omitted.
    corrected_units : str, optional
        Units string written to the output variable's attributes after a correction
        is applied (e.g. ``"mS/cm"``). Left untouched if omitted or if no correction runs.
    append_description, overwrite_description : str, optional
        Note written to the output variable's ``comment`` attribute (e.g.
        ``"Renamed from LATITUDE_GPS"``). ``append_description`` adds to any existing
        comment; ``overwrite_description`` replaces it. Set at most one.

    Examples
    --------
    Example usage in a pipeline configuration:

    .. code-block:: yaml

        steps:
          - name: Correct Values
            parameters:
              target_variable: TEMP
              slope: 1.001
              intercept: -0.05
              expected_range: [-2.5, 40]
            diagnostics: false
    """

    step_name = "Correct Values"
    required_variables = []
    provided_variables = []

    parameter_schema = {
        "target_variable": {
            "type": str,
            "required": True,
            "description": "Name of the variable to correct (e.g. 'CNDC').",
        },
        "output_as": {
            "type": str,
            "default": None,
            "description": "Name to write the result under (default: target_variable, "
                           "i.e. in place). Set to a new name to copy/rename.",
        },
        "slope": {
            "type": float,
            "default": 1.0,
            "description": "Multiplicative factor (corrected = slope * value + intercept). "
                           "For a simple x10 unit conversion, set 10.",
        },
        "intercept": {
            "type": float,
            "default": 0.0,
            "description": "Additive offset applied after scaling (corrected = slope * value + intercept). "
                           "Use for sensor alignment.",
        },
        "expected_range": {
            "type": list,
            "default": None,
            "description": "Optional [min, max] for the corrected variable. The correction is applied "
                           "only when the data's median falls OUTSIDE this range. If omitted, the "
                           "correction is always applied.",
        },
        "corrected_units": {
            "type": str,
            "default": None,
            "description": "Optional units string written to the output variable's attributes "
                           "after a correction is applied (e.g. 'mS/cm').",
        },
        "time_start": {
            "type": str,
            "default": None,
            "description": "Optional ISO timestamp; only points at/after this TIME are corrected.",
        },
        "time_end": {
            "type": str,
            "default": None,
            "description": "Optional ISO timestamp; only points at/before this TIME are corrected.",
        },
        "append_description": {
            "type": str,
            "default": None,
            "description": "Optional note appended to the output variable's 'comment' attribute.",
        },
        "overwrite_description": {
            "type": str,
            "default": None,
            "description": "Optional note that replaces the output variable's 'comment' attribute.",
        },
    }

    def run(self):
        self.check_data()
        self.data = self.context["data"]

        var = self.target_variable
        if var not in self.data:
            raise ValueError(
                f"[{self.name}] target_variable '{var}' not found in dataset. "
                f"Available variables: {list(self.data.data_vars)}."
            )

        out = self.output_as or var
        if self.append_description is not None and self.overwrite_description is not None:
            raise ValueError(
                f"[{self.name}] set only one of 'append_description' / 'overwrite_description'."
            )

        vals = self.data[var].values.astype(float)
        self._raw_data = vals.copy()
        self.applied = False

        valid_mask = ~np.isnan(vals)
        if not np.any(valid_mask):
            self.log_warn(f"'{var}' has no valid (non-NaN) values; nothing to correct.")
            self.context["data"] = self.data
            return self.context

        # Restrict the correction to a TIME window when requested.
        window = np.ones(vals.shape, dtype=bool)
        if self.time_start is not None or self.time_end is not None:
            if "TIME" not in self.data:
                raise ValueError(
                    f"[{self.name}] time_start/time_end need a 'TIME' variable in the data."
                )
            times = self.data["TIME"].values
            if self.time_start is not None:
                window &= times >= np.datetime64(self.time_start)
            if self.time_end is not None:
                window &= times <= np.datetime64(self.time_end)

        # Decide whether the correction is needed, judging by the windowed data.
        do_scale = True
        if self.expected_range is not None:
            lo, hi = float(self.expected_range[0]), float(self.expected_range[1])
            sample = vals[valid_mask & window]
            median_val = float(np.nanmedian(sample)) if sample.size else np.nan
            if np.isfinite(median_val) and lo <= median_val <= hi:
                self.log(
                    f"'{var}' median ({median_val:.4g}) is within expected range "
                    f"[{lo}, {hi}]; skipping correction."
                )
                do_scale = False
            elif np.isfinite(median_val):
                self.log(
                    f"'{var}' median ({median_val:.4g}) is outside expected range "
                    f"[{lo}, {hi}]; applying correction."
                )

        # Nothing changes when there is no scaling, no rename and no comment to set.
        no_description = self.append_description is None and self.overwrite_description is None
        if not do_scale and out == var and no_description:
            self.context["data"] = self.data
            return self.context

        # Apply the affine correction within the window (NaNs propagate harmlessly).
        corrected = vals.copy()
        if do_scale:
            corrected[window] = self.slope * vals[window] + self.intercept
            self.applied = True

        # Write to output_as (a copy/rename when it differs from target_variable).
        self.data[out] = self.data[var].copy(data=corrected)

        if self.applied:
            self.log(
                f"Applied correction to '{out}': corrected = {self.slope} * value + {self.intercept}."
            )
        if out != var:
            self.log(f"Wrote '{var}' to '{out}'.")

        if self.corrected_units is not None:
            self.data[out].attrs["units"] = self.corrected_units
            self.log(f"Set '{out}' units to '{self.corrected_units}'.")

        if self.overwrite_description is not None:
            self.data[out].attrs["comment"] = self.overwrite_description
        elif self.append_description is not None:
            existing = self.data[out].attrs.get("comment", "")
            self.data[out].attrs["comment"] = (
                f"{existing} {self.append_description}".strip() if existing else self.append_description
            )

        if self.diagnostics:
            self.plot_diagnostics()

        self.context["data"] = self.data
        return self.context

    def plot_diagnostics(self):
        if not self.applied:
            return

        var = self.target_variable
        corrected = self.data[var].values

        # Plot against TIME if available, otherwise against sample index.
        if "TIME" in self.data:
            x = self.data["TIME"].values
            xlabel = "Time"
        else:
            x = np.arange(len(corrected))
            xlabel = "Sample index"

        matplotlib.use("tkagg")
        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]

        fig_spec.points(ax, x, self._raw_data, color=fig_spec.FLAGGED, label="Raw")
        fig_spec.points(ax, x, corrected, color=fig_spec.CATEGORY[1], label="Corrected")

        if self.expected_range is not None:
            lo, hi = float(self.expected_range[0]), float(self.expected_range[1])
            ax.axhline(hi, color="black", linestyle="--", alpha=0.6, linewidth=1, label=f"Max ({hi})")
            ax.axhline(lo, color="black", linestyle="--", alpha=0.6, linewidth=1, label=f"Min ({lo})")

        if xlabel == "Time":
            fig_spec.date_axis(ax, which="x")
        ylabel = fig_spec.axis_label(var, self.data[var].attrs.get("units"))
        fig_spec.style_axes(ax, xlabel=xlabel, ylabel=ylabel)
        fig_spec.legend(ax)

        fig_spec.finish(
            fig,
            suptitle=f"Value Correction: {var}\n"
            f"(corrected = {self.slope} * value + {self.intercept})",
        )
        plt.show(block=True)
