"""Serialise a matplotlib figure to a plot spec the browser can redraw.

Diagnostic plots reach the dashboard as PNGs, which cannot be zoomed into. This
module walks a finished ``Figure`` and, when every artist on it is one this
understands (lines and scatter points), emits the underlying x/y arrays plus the
axes' labels and limits. The browser redraws that with plotly, so the user gets
box-zoom, pan, hover and legend toggling on the real data.

It is deliberately partial. Anything it does not recognise -- images,
pcolormesh, bar/histogram patches, map projections -- makes the whole figure
``None`` and the dashboard keeps showing the PNG it already saved. That way a
plot is never rendered *wrongly*: it is either faithful or it is the image.

Dashboard-only: nothing in pelagos_py imports this, and a pipeline run outside
the dashboard never touches it.
"""

import math

import matplotlib.dates as mdates
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex

# Point budget for a whole figure, shared out between its traces (see
# _point_cap). A full-resolution glider record is millions of points per
# variable, and a QC plot draws one trace per flag, so without a cap the spec
# would dwarf the PNG it sits beside. 150k points is ~3 MB of JSON and still far
# more than the few thousand pixel columns it gets drawn into.
POINT_BUDGET = 150000

# No trace is thinned below this, however many share the figure.
MIN_POINTS = 5000

# Per-request cap for a zoomed-in range query (see app.py's figdata route).
ZOOM_POINT_CAP = 100000


def _point_cap(fig):
    """Per-trace point cap: the figure's budget split across all its traces."""
    traces = sum(len(ax.get_lines()) + len(ax.collections) for ax in fig.axes)
    return max(MIN_POINTS, POINT_BUDGET // max(1, traces))


def _hex(color, default="#1f77b4"):
    """A ``#rrggbb`` string for any matplotlib colour spec."""
    try:
        return to_hex(color)
    except Exception:  # noqa: BLE001 - a colour is cosmetic, never fatal
        return default


def _alpha(color, fallback):
    """The alpha of an RGBA colour, else the artist's own alpha (else 1)."""
    try:
        if not isinstance(color, str) and len(color) == 4:
            return float(color[3])
    except TypeError:
        pass
    return 1.0 if fallback is None else float(fallback)


def _is_datetime_axis(axis):
    """Whether this axis holds matplotlib date numbers rather than plain floats."""
    return isinstance(axis.get_major_locator(), mdates.DateLocator) or isinstance(
        axis.get_major_formatter(), mdates.DateFormatter
    )


def _coord_floats(raw):
    """A trace coordinate as matplotlib date numbers, and whether it was a date.

    A step may hand ``plot`` either matplotlib date numbers (which come back as
    ordinary floats around 19000) or raw ``datetime64``/``datetime`` objects,
    letting matplotlib convert them for display. In the latter case
    ``get_data`` returns the originals, and a plain ``float`` cast would give
    nanoseconds-since-epoch -- off the date scale by a factor of 1e11, so every
    point lands outside the axis and the plot looks empty. Convert those with
    ``date2num`` so they share the scale of the axis limits and the date axis
    formatting downstream. Applies to either axis: a step may plot value-vs-time
    with the time on x *or* y (an index-vs-TIME QC plot does the latter).
    """
    arr = np.asarray(raw)
    if np.issubdtype(arr.dtype, np.datetime64):
        return np.asarray(mdates.date2num(arr), dtype=float), True
    if arr.dtype == object and arr.size:
        try:  # datetime.datetime / pandas.Timestamp objects
            return np.asarray(mdates.date2num(arr), dtype=float), True
        except (TypeError, ValueError):
            pass
    return np.asarray(arr, dtype=float), False


def _is_monotonic(x):
    """Whether ``x`` (ignoring NaNs) is non-decreasing -- the only case a
    range query or bucket decimation can treat index order as x order."""
    finite = x[np.isfinite(x)]
    return len(finite) < 2 or bool(np.all(np.diff(finite) >= 0))


def _decimate_indices(x, y, cap):
    """Indices thinning a trace to ``cap`` points, keeping bucket extremes.

    Plain striding would drop exactly the features these diagnostics exist to
    show -- a one-sample spike survives only if it is the min or max of its
    bucket. So for a monotonic x (index or time, which is almost all of them)
    each bucket contributes its y-min and y-max at their own x. For unordered x
    there is no meaningful bucket, so fall back to striding. Returns
    ``(keep_idx, thinned)``; ``keep_idx`` indexes ``x``/``y`` and anything else
    of the same length (e.g. a per-point colour array).
    """
    n = len(x)
    if n <= cap:
        return np.arange(n), False
    if not _is_monotonic(x):
        step = int(math.ceil(n / cap))
        return np.arange(0, n, step), True

    buckets = cap // 2
    edges = np.linspace(0, n, buckets + 1).astype(int)
    keep = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        chunk = y[lo:hi]
        valid = np.flatnonzero(np.isfinite(chunk))
        if len(valid) == 0:
            keep.append(lo)
            continue
        keep.append(lo + int(valid[np.argmin(chunk[valid])]))
        keep.append(lo + int(valid[np.argmax(chunk[valid])]))
    return np.unique(np.asarray(keep, dtype=int)), True


def _decimate(x, y, cap):
    """Thin a trace to ``cap`` points; see ``_decimate_indices``."""
    keep, thinned = _decimate_indices(x, y, cap)
    return x[keep], y[keep], thinned


def _values(arr, is_date):
    """A JSON-safe list: ISO strings for a date axis, else floats with NaN->None."""
    arr = np.asarray(arr, dtype=float)
    if is_date:
        out = []
        for v in arr:
            if not np.isfinite(v):
                out.append(None)
                continue
            try:
                out.append(mdates.num2date(v).isoformat())
            except Exception:  # noqa: BLE001 - an unconvertible date drops out
                out.append(None)
        return out
    # round() keeps the payload compact; 6 significant decimals is far finer
    # than any pixel the browser will draw this at.
    return [None if not np.isfinite(v) else round(float(v), 6) for v in arr]


def _ref_line(line, ax, x_date, y_date):
    """An axhline/axvline as a spanning reference line, or None.

    ``axhline``/``axvline`` draw a Line2D with one coordinate in *axes* space
    ([0, 1]) and the other in data space, so their raw x/y are not data points
    -- read as such they land at 0..1 and vanish off the real axis. Matplotlib
    marks them by the blended transform they use, which is what we test here.
    Emitted separately so the browser can draw a full-width/height line at the
    data value instead (range bounds, min/max limits, correction levels).
    """
    try:
        is_h = line.get_transform() == ax.get_yaxis_transform()
        is_v = line.get_transform() == ax.get_xaxis_transform()
    except Exception:  # noqa: BLE001 - not a ref line if the transform is odd
        return None
    if not (is_h or is_v):
        return None
    base = {
        "color": _hex(line.get_color()),
        "dash": {"--": "dash", "-.": "dashdot", ":": "dot"}.get(line.get_linestyle(), "solid"),
        "opacity": _alpha(line.get_color(), line.get_alpha()),
        "width": float(line.get_linewidth()),
    }
    if is_h:
        val = float(line.get_ydata()[0])
        return {"axis": "y", "value": _values(np.asarray([val]), y_date)[0], **base}
    val = float(line.get_xdata()[0])
    return {"axis": "x", "value": _values(np.asarray([val]), x_date)[0], **base}


def _epoch_ms(ordinal):
    """A matplotlib date-ordinal array (days since the Unix epoch, mpl>=3.3) as
    epoch milliseconds -- what the browser's ``Date`` and the figdata wire
    format use, sidestepping the ordinal/epoch conversion client-side."""
    return np.asarray(ordinal, dtype=np.float64) * 86400000.0


def _fullres(x, y, x_date, y_date, cap, color_rgba=None):
    """Full-resolution capture for a trace the caller is about to thin, or
    ``None`` if it isn't eligible.

    Only worth keeping when the trace actually gets thinned (nothing gained by
    re-fetching a trace whose spec already holds every point) and its x is
    monotonic (a range query cannot mean anything otherwise -- see
    ``_decimate_indices``). ``color_rgba`` is the full-length per-point RGBA
    float array for a scatter trace, or ``None`` for a line / uncoloured
    scatter.
    """
    if len(x) <= cap or not _is_monotonic(x):
        return None
    return {
        "x": _epoch_ms(x) if x_date else np.asarray(x, dtype=np.float64),
        "y": _epoch_ms(y) if y_date else np.asarray(y, dtype=np.float64),
        "color": None if color_rgba is None
        else np.clip(np.asarray(color_rgba) * 255, 0, 255).astype(np.uint8),
    }


def _line_trace(line, x_date, y_date, cap):
    """A trace dict for a Line2D, or None if it holds nothing to draw."""
    raw_x, raw_y = line.get_data()
    x, x_is_date = _coord_floats(raw_x)
    y, y_is_date = _coord_floats(raw_y)
    x_date = x_date or x_is_date
    y_date = y_date or y_is_date
    # A trace with nothing finite to draw is dropped: matplotlib draws nothing
    # for it either, and an all-null trace makes plotly's WebGL path complain.
    if len(x) == 0 or not np.any(np.isfinite(y)):
        return None, None
    fullres = _fullres(x, y, x_date, y_date, cap)
    x, y, thinned = _decimate(x, y, cap)

    ls = line.get_linestyle()
    marker = line.get_marker()
    has_line = ls not in ("None", " ", "", None)
    has_marker = marker not in ("None", " ", "", None)
    if not has_line and not has_marker:
        has_line = True  # nothing declared: matplotlib's own default is a line

    color = line.get_color()
    return {
        "x": _values(x, x_date),
        "y": _values(y, y_date),
        "mode": ("lines" if has_line else "") + ("+markers" if has_line and has_marker else ("markers" if has_marker else "")),
        "label": line.get_label(),
        "color": _hex(color),
        "opacity": _alpha(color, line.get_alpha()),
        "width": float(line.get_linewidth()),
        "dash": {"--": "dash", "-.": "dashdot", ":": "dot"}.get(ls, "solid"),
        "size": float(line.get_markersize()),
        "thinned": thinned,
        "lod": fullres is not None,
    }, fullres


def _scatter_trace(coll, x_date, y_date, cap):
    """A trace dict for a PathCollection (ax.scatter), or None if empty.

    Per-point colours are kept as a list so a scatter coloured by value (QC
    flags, say) still looks right; a single colour collapses to one string.
    """
    offsets = np.asarray(coll.get_offsets())
    if offsets.size == 0:
        return None, None
    x, x_is_date = _coord_floats(offsets[:, 0])
    y, y_is_date = _coord_floats(offsets[:, 1])
    x_date = x_date or x_is_date
    y_date = y_date or y_is_date
    if not np.any(np.isfinite(y)):
        return None, None

    # A scatter given c=<array> only resolves its colours at draw time, and the
    # figure reaching us has never been drawn (Agg, saved not shown).
    try:
        coll.update_scalarmappable()
    except Exception:  # noqa: BLE001 - fall back to whatever colours it has
        pass

    faces = coll.get_facecolors()
    per_point_faces = faces if len(faces) == len(x) else None
    if len(faces) == 0:
        color, opacity = "#1f77b4", 1.0
    elif len(faces) == 1:
        color, opacity = _hex(faces[0]), _alpha(faces[0], coll.get_alpha())
    elif per_point_faces is not None and len(x) <= cap:
        color = [_hex(c) for c in faces]
        opacity = _alpha(faces[0], coll.get_alpha())
    else:
        # Too many to ship per point (or a length mismatch): use the first.
        color, opacity = _hex(faces[0]), _alpha(faces[0], coll.get_alpha())

    fullres = _fullres(x, y, x_date, y_date, cap, color_rgba=per_point_faces)
    x, y, thinned = _decimate(x, y, cap)
    if thinned and isinstance(color, list):
        color = color[0]  # the per-point mapping no longer lines up

    sizes = coll.get_sizes()
    return {
        "x": _values(x, x_date),
        "y": _values(y, y_date),
        "mode": "markers",
        "label": coll.get_label(),
        "color": color,
        "opacity": opacity,
        # PathCollection sizes are points^2; plotly wants a diameter in points.
        "size": float(np.sqrt(sizes[0])) if len(sizes) else 6.0,
        "thinned": thinned,
        "lod": fullres is not None,
    }, fullres


def _unsupported(ax):
    """Why this axes cannot be redrawn faithfully, or None if it can be."""
    if ax.images:
        return "image"
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            return type(coll).__name__
    # Bars, histograms and filled regions live in patches; the axes' own
    # background Rectangle is not in this list, so anything here is real content.
    if len(ax.patches) > 0:
        return "patches"
    # Anything non-rectilinear (a cartopy map, a polar plot) needs a projection
    # the browser side does not implement.
    if ax.name != "rectilinear":
        return f"projection:{ax.name}"
    return None


def _shared_x(fig):
    """Indices of axes that share an x-axis with an earlier one, as a group map.

    Returned as ``{axes index: group id}`` so the browser can link the zoom of
    stacked panels the step drew with ``sharex=True``.
    """
    groups, out = [], {}
    for i, ax in enumerate(fig.axes):
        try:
            siblings = set(ax.get_shared_x_axes().get_siblings(ax))
        except Exception:  # noqa: BLE001 - no sharing info: treat as independent
            siblings = {ax}
        for gid, members in enumerate(groups):
            if members & siblings:
                members.add(ax)
                out[i] = gid
                break
        else:
            groups.append(set(siblings))
            out[i] = len(groups) - 1
    return out


def _grid_cell(ax):
    """Normalised ``[left, top, width, height]`` of this axes' grid cell, top-down.

    Read from the axes' gridspec so the browser can reproduce the panel layout
    -- unequal row heights (``height_ratios``) and multi-row/column spans
    included -- instead of stacking every panel at equal height. Fractions of
    the figure, origin top-left (the browser's own origin). ``None`` if the axes
    was not placed on a gridspec (e.g. ``add_axes``), where the browser falls
    back to an even vertical stack.
    """
    try:
        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()
        nrows, ncols = gs.get_geometry()
        hr = gs.get_height_ratios() or [1] * nrows
        wr = gs.get_width_ratios() or [1] * ncols
        htot, wtot = float(sum(hr)), float(sum(wr))
        r0, r1 = ss.rowspan.start, ss.rowspan.stop
        c0, c1 = ss.colspan.start, ss.colspan.stop
        return [
            round(sum(wr[:c0]) / wtot, 6), round(sum(hr[:r0]) / htot, 6),
            round(sum(wr[c0:c1]) / wtot, 6), round(sum(hr[r0:r1]) / htot, 6),
        ]
    except Exception:  # noqa: BLE001 - no gridspec: browser stacks evenly instead
        return None


def _panel(ax, x_date, y_date, cap):
    """The spec for one axes and its full-res captures: ``(panel, fullres)``.

    ``fullres`` maps a trace's index within ``panel["traces"]`` to its
    ``_fullres`` capture, for traces eligible for a zoomed-in range query.
    """
    traces = []
    fullres = {}
    reflines = []
    for line in ax.get_lines():
        ref = _ref_line(line, ax, x_date, y_date)
        if ref is not None:
            reflines.append(ref)
            continue
        trace, trace_fullres = _line_trace(line, x_date, y_date, cap)
        if trace is not None:
            if trace_fullres is not None:
                fullres[len(traces)] = trace_fullres
            traces.append(trace)
    for coll in ax.collections:
        trace, trace_fullres = _scatter_trace(coll, x_date, y_date, cap)
        if trace is not None:
            if trace_fullres is not None:
                fullres[len(traces)] = trace_fullres
            traces.append(trace)

    legend = ax.get_legend()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    return {
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xdate": x_date,
        "ydate": y_date,
        "xlim": _values(np.asarray(xlim), x_date),
        "ylim": _values(np.asarray(ylim), y_date) if y_date
        else [round(float(v), 6) for v in ylim],
        # Depth axes are inverted; plotly needs the range in drawing order.
        "y_inverted": bool(ylim[0] > ylim[1]),
        "xscale": ax.get_xscale(),
        "yscale": ax.get_yscale(),
        "legend": legend is not None,
        "legend_title": legend.get_title().get_text() if legend is not None else "",
        "cell": _grid_cell(ax),
        "traces": traces,
        "reflines": reflines,
    }, fullres


def serialise(fig):
    """``(spec, reason, fullres)`` for ``fig``: the plot spec, or None and why
    not, plus any full-resolution captures for zoomed-in range queries.

    All-or-nothing on the spec, on purpose: a figure that is half interactive
    and half silently missing its data would be worse than the PNG. ``reason``
    names the artist that stopped it, so the run log can say which plots
    stayed PNG-only and what would have to be added here to change that.

    ``fullres`` maps ``(panel_index, trace_index)`` to a ``_fullres`` capture
    for every trace marked ``"lod": true`` in the spec -- empty when the spec
    itself is ``None`` or nothing needed thinning.
    """
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if not axes:
        return None, "no axes", {}
    for ax in axes:
        blocker = _unsupported(ax)
        if blocker is not None:
            return None, blocker, {}

    shared = _shared_x(fig)
    cap = _point_cap(fig)
    panels = []
    fullres = {}
    for i, ax in enumerate(axes):
        panel, panel_fullres = _panel(
            ax, _is_datetime_axis(ax.xaxis), _is_datetime_axis(ax.yaxis), cap
        )
        if not panel["traces"]:
            return None, "empty axes", {}
        panel["share_x"] = shared.get(i, i)
        panels.append(panel)
        for trace_idx, capture in panel_fullres.items():
            fullres[(i, trace_idx)] = capture

    suptitle = ""
    if fig._suptitle is not None:
        suptitle = fig._suptitle.get_text()
    width, height = fig.get_size_inches()
    return {
        "suptitle": suptitle,
        "aspect": float(height) / float(width) if width else 0.75,
        "panels": panels,
        "thinned": any(t["thinned"] for p in panels for t in p["traces"]),
    }, "", fullres
