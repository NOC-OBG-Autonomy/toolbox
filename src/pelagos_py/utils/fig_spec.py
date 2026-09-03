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

"""Single source of truth for the look of every ``plot_diagnostics``: a 16:9
figure, one font/margin scale, one flag/category palette, one date format. Keeps
PNGs uniform and within what the dashboard's interactive (WebGL) renderer can
reproduce.

WebGL-safe rules:
  * Draw data with ``ax.plot(..., ls="", marker="o")`` (preferred, faster) or
    ``ax.scatter``; both round-trip. ``axhline``/``axvline`` are fine.
  * Colour by DISCRETE category (one call per category). Continuous colour +
    ``fig.colorbar`` adds a second axes the serialiser can't read and drops the
    WHOLE figure to PNG-only (as with MLD/salinity/bbp).
  * PNG-only (not reproducible), but still style them: hist, bar, fill_between,
    pcolormesh, contourf, imshow, fig.colorbar, twinx, cartopy/polar, LineCollection.
  * Multi-panel: build with plt.subplots; the interactive view stacks axes
    vertically in creation order with a shared x-axis, so use sharex=True.
"""

import matplotlib.dates as mdates
import numpy as np

# Geometry: single panel is 16:9; multi-panel keeps width, grows height by ROW_H/row.
FIG_W, FIG_H, DPI = 10, 5.625, 130
ROW_H = 3.0  # inches per panel for multi-panel (nrows > 1) figures

FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 12, 11, 9, 8, 8

MARKER = 4            # markersize for plot() point series
ALPHA = 0.7
RASTER_ABOVE = 5000   # rasterize dense point layers above this many points

# Colours (hex, one mapping used everywhere).
FLAG_COLOURS = {
    0: "#9aa5ad", 1: "#1f6fd6", 2: "#7fb2e5", 3: "#e8912b", 4: "#d6392f",
    5: "#9aa5ad", 6: "#9aa5ad", 7: "#9aa5ad", 8: "#17b6c4", 9: "#111111",
}
FLAGGED = "#b2bec3"
CATEGORY = ["#00b894", "#0984e3", "#d63031", "#fdcb6e", "#6c5ce7",
            "#e84393", "#00cec9", "#e17055"]

# QC flag meanings, used to label flag legend entries as "1 (good)".
FLAG_MEANINGS = {
    0: "no QC", 1: "good", 2: "prob good", 3: "prob bad", 4: "bad",
    5: "changed", 8: "interp", 9: "missing",
}


def flag_label(flag):
    """Legend label for a QC flag: '1 (good)' when the meaning is known, else '6'."""
    meaning = FLAG_MEANINGS.get(flag)
    return f"{flag} ({meaning})" if meaning else str(flag)


def new_fig(nrows=1, ncols=1, sharex=False, sharey=False, height_ratios=None):
    """A standard figure + axes at the standard width/dpi. Axes always 2D: axes[r][c].

    Single panel is 16:9; multi-panel grows height by ``ROW_H`` per row.
    ``height_ratios`` (e.g. ``(3, 1)``) sets unequal row heights via gridspec.
    """
    import matplotlib.pyplot as plt
    height = FIG_H if nrows == 1 else ROW_H * nrows
    gridspec_kw = {"height_ratios": height_ratios} if height_ratios is not None else None
    return plt.subplots(nrows, ncols, figsize=(FIG_W, height), dpi=DPI,
                        sharex=sharex, sharey=sharey, squeeze=False,
                        gridspec_kw=gridspec_kw)


def style_axes(ax, *, title=None, xlabel=None, ylabel=None):
    """The common per-axes look."""
    if title:
        ax.set_title(title, fontsize=FS_TITLE)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def axis_label(var, units=None):
    """'TEMP [degC]' when units are meaningful, else just the name."""
    u = (units or "").strip()
    return var if u in ("", "1", "unitless", "unknown", "none", "None") else f"{var} [{u}]"


def date_axis(ax, which="x"):
    """Consistent date formatting on the chosen axis (x or y)."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    loc = mdates.AutoDateLocator()
    axis.set_major_locator(loc)
    axis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def points(ax, x, y, *, color, label=None, size=MARKER, alpha=ALPHA):
    """Standard point series: fast plot() markers, WebGL-safe, one legend entry."""
    x = np.asarray(x)
    ax.plot(x, np.asarray(y), ls="", marker="o", markersize=size,
            markeredgewidth=0, color=color, alpha=alpha, label=label,
            rasterized=x.size > RASTER_ABOVE)


def flag_points(ax, x, y, flags):
    """y vs x coloured by QC flag: one series per present flag (0..9)."""
    flags, x, y = np.asarray(flags), np.asarray(x), np.asarray(y)
    for f in range(10):
        m = flags == f
        if m.any():
            points(ax, x[m], y[m], color=FLAG_COLOURS[f], label=flag_label(f))


def legend(ax, *, title=None):
    """Compact legend just outside the axes on the right."""
    ax.legend(title=title, fontsize=FS_LEGEND, title_fontsize=FS_LEGEND,
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.9, markerscale=2)


def finish(fig, suptitle=None):
    """Suptitle + tight layout, the same for every figure."""
    if suptitle:
        fig.suptitle(suptitle, fontsize=FS_SUPTITLE, fontweight="bold")
    fig.tight_layout()
