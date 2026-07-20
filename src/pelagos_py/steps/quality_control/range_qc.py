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

"""Unified range QC test.

Replaces the old ``gross range qc`` (flag values *outside* a good band) and
``impossible range qc`` (flag values *inside* an impossible band) with a single
test that does both. Each range carries an explicit ``inside``/``outside`` keyword
that chooses the behaviour: ``outside`` flags data outside the band (a good band),
``inside`` flags data within it (an impossible band). When no keyword is given the
bound *order* is used as a fallback (ascending ``[low, high]`` -> outside, descending
``[high, low]`` -> inside). A flag may list several ranges so the same flag can cover
more than one band. The test can also propagate a variable's flags onto companion
variables (e.g. flag PRES and TEMP bad whenever CNDC is bad) either alongside the
source's own flags (``also_flag``) or in place of them (``flag_instead``, e.g. flag
CHLA by DEPTH without flagging DEPTH itself), limit the checks to a DEPTH window, and
plot every flagged variable when diagnostics are enabled.
"""

#### Mandatory imports ####
import numpy as np
from pelagos_py.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr


# Argo flag-merge matrix used when propagating an "also_flag" flag onto a companion:
# the result of merging an existing flag (row) with a new one (column) is
# QC_COMBINATRIX[existing, new]. This is the same logic Apply QC uses to merge each
# test's flags into the dataset, copied here so a single test's own cross-flagging is
# consistent with it (e.g. a companion already flagged bad (4) is never downgraded to
# probably-bad (3) by propagation). See ApplyQC.organise_flags; this could later move
# to a shared utility.
QC_COMBINATRIX = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 1, 2, 3, 4, 5, 1, 1, 8, 9],
        [2, 2, 2, 3, 4, 5, 2, 2, 8, 9],
        [3, 3, 3, 3, 4, 3, 3, 3, 3, 9],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 9],
        [5, 5, 5, 3, 4, 5, 5, 5, 8, 9],
        [6, 1, 2, 3, 4, 5, 6, 6, 8, 9],
        [7, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [8, 8, 8, 3, 4, 8, 8, 8, 8, 9],
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    ]
)

FLAG_MEANINGS = {
    0: "no QC",
    1: "good",
    2: "prob good",
    3: "prob bad",
    4: "bad",
    5: "changed",
    8: "interp",
    9: "missing",
}


@register_qc
class range_qc(BaseQC):
    """
    Flag measurements by value range. A single, configurable replacement for the
    old separate "gross range" and "impossible range" tests.

    Each ``{flag: bounds}`` entry describes one or more bands. A band is given as a
    ``[low, high]`` pair with an ``inside``/``outside`` keyword saying which side to
    flag:

    - **``[low, high, "outside"]``** — a *good* band; data ``< low`` or ``> high``
      gets the flag. (The old "gross range" behaviour.) Accepts ``outside``, ``out``
      or ``o`` (any capitalisation).
    - **``[low, high, "inside"]``** — an *impossible* band; data between the two bounds,
      inclusive of the bounds themselves, gets the flag. (The old "impossible range"
      behaviour.) Accepts ``inside``, ``in`` or ``i`` (any capitalisation).
    - **A single scalar** ``value`` — flags exact matches (``data == value``). Handy
      for fill/filler values, e.g. ``4: 0.0`` to flag a pressure of exactly ``0`` as
      bad.

    The same flag can cover several bands by giving a *list of bands*, e.g.
    ``4: [[2, 3, "inside"], [0.1, 10, "outside"]]`` — a point is flagged if it falls in
    *any* of them.

    If the keyword is omitted the bound *order* is used as a fallback: an ascending
    ``[low, high]`` means ``outside`` (a good band) and a descending ``[high, low]``
    means ``inside`` (an impossible band). An explicit keyword always wins over the
    order, so write ``inside``/``outside`` when in doubt.

    Within a variable, entries are applied most-severe-flag-first, so on overlap the
    worse flag wins. Anything checked but not flagged is marked good (1).

    Target Variable: Any
    Flag Number: Any (user-defined)
    Variables Flagged: Any (the tested variables, minus any ``flag_instead`` sources,
    plus any ``also_flag``/``flag_instead`` companions)

    EXAMPLE
    -------
    ::

        - name: "Apply QC"
          parameters:
            qc_settings:
              range qc:
                variable_ranges:
                  PRES:
                    3: [-2.4, -5, inside]   # impossible band: flag data INSIDE it
                    4: [-5, -.inf, inside]
                    9: 0.0                  # single scalar -> flag the exact fill value 0.0
                  TEMP:
                    3: [0, 30, outside]     # good band: flag data OUTSIDE it
                    4: [-2.5, 40, outside]
                  CNDC:
                    # one flag, two bands: flag bad both inside [2, 3] and outside [0.1, 10]
                    4: [[2, 3, inside], [0.1, 10, outside]]
                also_flag:
                  CNDC: [PRES, TEMP]    # CNDC's flags propagate onto PRES & TEMP (worst wins)
                test_depth_range: [0, 100]    # OPTIONAL: only check this DEPTH window
          diagnostics: true             # plots every flagged variable, coloured by flag

    Flagging one variable by another's range (``flag_instead``) — here CHLA is judged
    by DEPTH (probably-bad in the quenched top 5 m, probably-good below it) while DEPTH
    itself is left unflagged::

        - name: "Apply QC"
          parameters:
            qc_settings:
              range qc:
                variable_ranges:
                  DEPTH:
                    3: [0, 5, inside]     # top 5 m -> probably bad (correctable)
                    2: [0, 5, outside]    # deeper   -> probably good
                flag_instead:
                  DEPTH: [CHLA]           # DEPTH's flags go onto CHLA; no DEPTH_QC written
    """

    qc_name = "range qc"

    # Target variables are user-defined, so __init__ is redefined to resolve the
    # test's required/provided variables from the parameters.
    dynamic = True

    parameter_schema = {
        "variable_ranges": {
            "type": dict,
            "required": True,
            "description": "Per-variable {flag: band} ranges. A band is [low, high, 'inside'|'outside'] "
                           "('outside' flags data outside it, 'inside' flags data within it); the keyword "
                           "may be omitted, in which case an ascending pair means outside and a descending "
                           "pair means inside. A flag may give a list of bands to cover several ranges.",
        },
        "also_flag": {
            "type": dict,
            "default": {},
            "description": "Propagate a variable's flags onto companion variables, e.g. "
                           "{'CNDC': ['PRES', 'TEMP']}. Merged with the Argo matrix so the worst "
                           "flag wins.",
        },
        "flag_instead": {
            "type": dict,
            "default": {},
            "description": "Like also_flag, but the source variable's own flags are not written, e.g. "
                           "{'DEPTH': ['CHLA']} flags CHLA by DEPTH's ranges while leaving DEPTH "
                           "unflagged.",
        },
        "test_depth_range": {
            "type": list,
            "default": None,
            "description": "Optional [min, max] DEPTH window; checks apply only to samples within it.",
        },
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

        if self.also_flag is None:
            self.also_flag = {}
        if self.flag_instead is None:
            self.flag_instead = {}

        self.tested_variables = list(self.variable_ranges.keys())

        # flag_instead sources are only a means of flagging their companions, so they
        # must define ranges but never appear in the outputs.
        for var in self.flag_instead:
            if var not in self.variable_ranges:
                raise ValueError(
                    f"[{self.qc_name}] flag_instead source {var!r} has no entry in "
                    f"variable_ranges; give it ranges or remove it."
                )

        # Flags become QC values that Apply QC (and this class's also_flag
        # propagation) merges via the Argo 10x10 matrix, so they must be integer
        # indices 0-9. Validate up front to fail with a clear config error rather
        # than an IndexError deep in return_qc.
        for var, meta in self.variable_ranges.items():
            for flag in meta:
                if isinstance(flag, bool) or not isinstance(flag, int) or not (0 <= flag <= 9):
                    raise ValueError(
                        f"[{self.qc_name}] invalid QC flag {flag!r} for variable "
                        f"{var!r}; expected an Argo QC flag 0-9."
                    )

        self.required_variables = self.tested_variables.copy()
        if self.test_depth_range is not None:
            self.required_variables.append("DEPTH")

        # Outputs are the tested variables plus any companions they propagate onto,
        # less the flag_instead sources (tested only to flag their companions).
        self.qc_outputs = list(
            (
                {f"{var}_QC" for var in self.tested_variables}
                | {f"{var}_QC" for var in sum(self.also_flag.values(), [])}
                | {f"{var}_QC" for var in sum(self.flag_instead.values(), [])}
            )
            - {f"{var}_QC" for var in self.flag_instead}
        )

    # Keyword aliases (any capitalisation) that force a band's behaviour.
    _OUTSIDE_KEYWORDS = {"outside", "out", "o"}
    _INSIDE_KEYWORDS = {"inside", "in", "i"}

    @classmethod
    def _iter_bands(cls, bounds):
        """Yield each individual band in a flag's configured ``bounds`` entry.

        A flag can carry a single band or a *list* of bands. A list of bands is one
        whose elements are themselves lists/tuples (e.g. ``[[2, 3], [0.1, 10]]``);
        anything else (a scalar, or a single ``[low, high(, kw)]`` band) is treated as
        one band and yielded as-is.
        """
        if (
            isinstance(bounds, (list, tuple))
            and bounds
            and all(isinstance(b, (list, tuple)) for b in bounds)
        ):
            yield from bounds
        else:
            yield bounds

    @classmethod
    def _band_hit(cls, vals, band):
        """Return a boolean mask of the values a single configured ``band`` flags.

        - A single scalar flags exact matches (e.g. a fill value such as ``0``).
        - ``[low, high, 'outside']`` is a good band: values outside it are flagged.
        - ``[low, high, 'inside']`` is an impossible band: values between the bounds,
          inclusive of the bounds themselves, are flagged.
        - When no keyword is given the order decides: an ascending ``[low, high]`` is a
          good band (flag outside), a descending ``[high, low]`` an impossible band
          (flag inside).

        An explicit keyword always wins over the bound order. NaNs compare ``False``
        throughout, so missing values are never flagged here.
        """
        if not isinstance(band, (list, tuple)):
            return vals == band  # exact-match a single value

        mode = None  # None -> fall back to bound order
        nums = list(band)
        if nums and isinstance(nums[-1], str):
            kw = nums[-1].strip().lower()
            if kw in cls._OUTSIDE_KEYWORDS:
                mode = "outside"
            elif kw in cls._INSIDE_KEYWORDS:
                mode = "inside"
            else:
                raise ValueError(
                    f"Unknown range keyword {nums[-1]!r}; expected one of "
                    f"{sorted(cls._OUTSIDE_KEYWORDS | cls._INSIDE_KEYWORDS)}."
                )
            nums = nums[:-1]

        if len(nums) != 2:
            raise ValueError(
                f"Invalid range band {band!r}; expected a scalar, [low, high], "
                f"or [low, high, keyword]."
            )
        a, b = nums
        if mode is None:
            mode = "outside" if a <= b else "inside"

        low, high = (a, b) if a <= b else (b, a)
        if mode == "outside":
            return (vals < low) | (vals > high)  # good band -> flag outside (bounds good)
        return (vals >= low) & (vals <= high)  # impossible band -> flag inside (bounds incl.)

    def return_qc(self):
        n = len(self.data["N_MEASUREMENTS"])

        # Restrict checks to a DEPTH window if requested; otherwise check everything.
        if self.test_depth_range is not None:
            depth = self.data["DEPTH"].values
            low, high = self.test_depth_range
            depth_mask = (depth >= low) & (depth <= high)
        else:
            depth_mask = np.ones(n, dtype=bool)

        qc_arrays = {}
        for var in self.tested_variables:
            vals = self.data[var].values
            qc = np.zeros(n, dtype=int)

            # Most-severe flag first so it wins where ranges overlap.
            for flag in sorted(self.variable_ranges[var], reverse=True):
                hit = np.zeros(n, dtype=bool)
                for band in self._iter_bands(self.variable_ranges[var][flag]):
                    hit |= self._band_hit(vals, band)
                qc[hit & depth_mask & (qc == 0)] = flag

            # Anything checked but unflagged is good.
            qc[(qc == 0) & depth_mask] = 1
            qc_arrays[var] = qc

        # Propagate flags onto companions using the Argo merge matrix, so a companion
        # keeps its own flag wherever merging the propagated one does not upgrade it
        # (e.g. an existing bad (4) is never downgraded by a propagated probably-bad (3)).
        # Companions that are not themselves tested start from "no QC" (0), so they
        # mirror the source's flags; Apply QC then merges the result with their existing
        # flags.
        for mapping in (self.also_flag, self.flag_instead):
            for var, companions in mapping.items():
                src = qc_arrays.get(var)
                if src is None:
                    continue
                for companion in companions:
                    base = qc_arrays.get(companion, np.zeros(n, dtype=int))
                    qc_arrays[companion] = QC_COMBINATRIX[base, src]

        # Drop the flag_instead sources now their flags have been propagated.
        for var in self.flag_instead:
            qc_arrays.pop(var, None)

        self.flags = xr.Dataset(coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]})
        for var, qc in qc_arrays.items():
            self.flags[f"{var}_QC"] = (("N_MEASUREMENTS",), qc)

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")

        # Auto-plot every variable this test flagged: the tested variables first
        # (they get range lines), then any companions it propagated onto.
        # (flag_instead sources drop out below: they have no flags in self.flags.)
        plot_order = list(self.tested_variables)
        for companion in sum(self.also_flag.values(), []) + sum(self.flag_instead.values(), []):
            if companion not in plot_order:
                plot_order.append(companion)
        plot_vars = [
            var for var in plot_order
            if var in self.data and f"{var}_QC" in self.flags
        ]
        if not plot_vars:
            return

        # Use TIME on the x-axis when available, otherwise the measurement index.
        if "TIME" in self.data:
            x = self.data["TIME"].values
            xlabel = "Time"
        else:
            x = self.data["N_MEASUREMENTS"].values
            xlabel = "Index"

        n_vars = len(plot_vars)
        fig, axes = plt.subplots(
            n_vars, 1, sharex=True, figsize=(10, 2.6 * n_vars + 1.5), dpi=150
        )
        if n_vars == 1:
            axes = [axes]

        for ax, var in zip(axes, plot_vars):
            vals = self.data[var].values
            qc = self.flags[f"{var}_QC"].values

            # Plot points coloured by flag (drawn low-to-high so worse flags sit on top).
            for flag in range(10):
                mask = qc == flag
                if not np.any(mask):
                    continue
                ax.plot(
                    x[mask], vals[mask], ls="", marker="o", markersize=2.5,
                    alpha=0.8, color=flag_cols[flag],
                    label=f"{flag} ({FLAG_MEANINGS.get(flag, 'n/a')})", zorder=flag,
                )

            # Range boundaries for variables that define their own ranges (a single
            # scalar is drawn as one line).
            if var in self.variable_ranges:
                for flag, bounds in self.variable_ranges[var].items():
                    for band in self._iter_bands(bounds):
                        band_list = band if isinstance(band, (list, tuple)) else [band]
                        for bound in band_list:
                            # Skip the inside/outside keyword; draw only numeric bounds.
                            if isinstance(bound, str) or not np.isfinite(bound):
                                continue
                            ax.axhline(
                                bound, ls="--", lw=1, alpha=0.6,
                                color=flag_cols.get(flag, "k"),
                            )

            ax.set_ylabel(var, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=8)
            if var == "PRES":
                ax.invert_yaxis()
            ax.legend(
                title="Flag", loc="center left", bbox_to_anchor=(1.01, 0.5),
                fontsize=7, framealpha=0.9, fancybox=True,
            )

        axes[-1].set_xlabel(xlabel, fontsize=8)
        fig.suptitle("Range QC", fontsize=10, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.83, 1])
        plt.show(block=True)
