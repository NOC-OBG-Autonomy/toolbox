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

Generic enough for unit conversions (e.g. CNDC S/m -> mS/cm is a simple x10),
sensor alignment (slope + intercept), or any other affine rescaling. An optional
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

    This makes a config robust to upstream fixes: e.g. a CNDC unit conversion
    (``slope: 10``) targeting ``expected_range: [20, 45]`` will scale S/m data into
    mS/cm, but quietly skip files that already arrive in mS/cm.

    Parameters
    ----------
    target_variable : str
        Name of the variable to correct (e.g. ``"CNDC"``).
    slope : float, optional
        Multiplicative factor (default ``1.0``). For a x10 unit conversion, set ``10``.
    intercept : float, optional
        Additive offset applied after scaling (default ``0.0``). Use for alignment.
    expected_range : list, optional
        ``[min, max]`` for the *corrected* variable. The correction is applied only
        when the data's median falls outside this range. If omitted, the correction
        is always applied.
    corrected_units : str, optional
        Units string written to the variable's attributes after a correction is
        applied (e.g. ``"mS/cm"``). Left untouched if omitted or if no correction runs.

    Examples
    --------
    Example usage in a pipeline configuration:

    .. code-block:: yaml

        steps:
          - name: Correct Values
            parameters:
              target_variable: CNDC
              slope: 10.0
              intercept: 0.0
              expected_range: [20, 45]
              corrected_units: mS/cm
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
            "description": "Optional units string written to the variable's attributes after a "
                           "correction is applied (e.g. 'mS/cm').",
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

        vals = self.data[var].values.astype(float)
        self._raw_data = vals.copy()
        self.applied = False

        valid_mask = ~np.isnan(vals)
        if not np.any(valid_mask):
            self.log_warn(f"'{var}' has no valid (non-NaN) values; nothing to correct.")
            self.context["data"] = self.data
            return self.context

        # Decide whether the correction is needed.
        if self.expected_range is not None:
            lo, hi = float(self.expected_range[0]), float(self.expected_range[1])
            median_val = float(np.nanmedian(vals[valid_mask]))
            if lo <= median_val <= hi:
                self.log(
                    f"'{var}' median ({median_val:.4g}) is within expected range "
                    f"[{lo}, {hi}]; skipping correction."
                )
                self.context["data"] = self.data
                return self.context
            self.log(
                f"'{var}' median ({median_val:.4g}) is outside expected range "
                f"[{lo}, {hi}]; applying correction."
            )

        # Apply the affine correction (NaNs propagate harmlessly through arithmetic).
        corrected = self.slope * vals + self.intercept
        self.data[var].values = corrected
        self.applied = True
        self.log(
            f"Applied correction to '{var}': corrected = {self.slope} * value + {self.intercept}."
        )

        if self.corrected_units is not None:
            self.data[var].attrs["units"] = self.corrected_units
            self.log(f"Set '{var}' units to '{self.corrected_units}'.")

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
