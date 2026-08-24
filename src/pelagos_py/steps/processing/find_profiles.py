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

"""Class definition for finding vertical and horizontal profiles in depth data."""

from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pelagos_py.utils import fig_spec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNKNOWN = 0
ASCENT = 1
DESCENT = 2
SURFACING = 3
INFLECTION = 5
PROPELLED = 6
TRANSITION = 7

PHASE_COLOURS = {
    UNKNOWN: "#9ca3af",
    ASCENT: "#22c55e",
    DESCENT: "#3b82f6",
    SURFACING: "#f97316",
    INFLECTION: "#06b6d4",
    PROPELLED: "#ef4444",
    TRANSITION: "#eab308",
}

PHASE_NAMES = {
    UNKNOWN: "0 Unknown",
    ASCENT: "1 Ascent",
    DESCENT: "2 Descent",
    SURFACING: "3 Surfacing",
    INFLECTION: "5 Inflection",
    PROPELLED: "6 Propelled",
    TRANSITION: "7 Transition",
}

DERIVED_COLUMNS = ["SCI_PHASE", "PROFILE_NUMBER", "PROFILE_DIRECTION", "CYCLE", "GRADIENT"]

# ---------------------------------------------------------------------------
# Core Processing Logic
# ---------------------------------------------------------------------------


def _compute_chunk_id(time_seconds, gap_threshold_seconds):
    # Real data gaps split the record into disconnected chunks - velocity is
    # never computed, and no run ever allowed, across one.
    return np.concatenate((
        [0], np.cumsum(np.diff(time_seconds) > gap_threshold_seconds)
    )).astype(np.int32)


def _gradient_per_chunk(values, time_seconds, chunk_id):
    # np.gradient over the whole record would bridge real data gaps (e.g. surface
    # comms windows, or an upcast-only glider whose data just stops mid-ascent),
    # diluting the slope right at the edge of a gap. Compute it chunk-by-chunk.
    result = np.zeros(len(values))
    for cid in np.unique(chunk_id):
        idx = np.flatnonzero(chunk_id == cid)
        if idx.size >= 2:
            result[idx] = np.gradient(values[idx], time_seconds[idx])
    return result


def _smoothed_velocity(depth, time, time_seconds, chunk_id, window):
    # Smooth depth, differentiate per-chunk, then smooth the resulting velocity
    # itself (median, to despike) - both rolling passes are time-windowed so
    # they stay meaningful under irregular sampling.
    depth_series = pd.Series(depth, index=pd.DatetimeIndex(time))
    smoothed_depth = depth_series.rolling(window, center=True, min_periods=1).mean().to_numpy()
    velocity = _gradient_per_chunk(smoothed_depth, time_seconds, chunk_id)
    return (
        pd.Series(velocity, index=depth_series.index)
        .rolling(window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )


def _runs_by_chunk(mask, chunk_id):
    # Yields (start, end) for each maximal run of True in `mask`, additionally
    # split wherever chunk_id changes inside it, so a run never bridges a gap.
    n = len(mask)
    if n == 0 or not mask.any():
        return
    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = (mask[1:] != mask[:-1]) | (chunk_id[1:] != chunk_id[:-1])
    edges = np.flatnonzero(boundary)
    edges = np.append(edges, n)
    for s, e in zip(edges[:-1], edges[1:]):
        if mask[s]:
            yield int(s), int(e)


def _classify_ascent_descent(smoothed_velocity, time_seconds, chunk_id, velocity_threshold, min_duration_seconds):
    # Threshold velocity into raw ascent/descent, then run-length merge, dropping
    # runs too short to trust (sensor noise) or that straddle a chunk boundary.
    n = len(smoothed_velocity)
    raw_phase = np.zeros(n, dtype=np.int8)
    raw_phase[smoothed_velocity > velocity_threshold] = DESCENT
    raw_phase[smoothed_velocity < -velocity_threshold] = ASCENT

    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (raw_phase[1:] != raw_phase[:-1]) | (chunk_id[1:] != chunk_id[:-1])
    run_starts = np.flatnonzero(change)
    run_ends = np.append(run_starts[1:], n)
    run_values = raw_phase[run_starts]
    run_durations = time_seconds[run_ends - 1] - time_seconds[run_starts]
    keep = (run_values != UNKNOWN) & (run_durations >= min_duration_seconds)

    return np.repeat(np.where(keep, run_values, UNKNOWN), run_ends - run_starts).astype(np.int8)


def _classify_propelled_surfacing(phase, depth, time_seconds, chunk_id,
                                   surfacing_depth_threshold, min_duration_seconds,
                                   min_transect_duration_seconds):
    # Applied only to what ascent/descent left unknown. A flat, undulating stretch
    # under propulsion (ALR-style) is propelled, gated by a much longer minimum
    # duration than surfacing so a turnaround isn't mistaken for one - a turn also
    # sits near-zero velocity briefly, but only for seconds, not minutes.
    for rs, re in _runs_by_chunk(phase == UNKNOWN, chunk_id):
        duration = time_seconds[re - 1] - time_seconds[rs]
        if np.median(depth[rs:re]) <= surfacing_depth_threshold:
            if duration >= min_duration_seconds:
                phase[rs:re] = SURFACING
        elif duration >= min_transect_duration_seconds:
            phase[rs:re] = PROPELLED


def _classify_inflection(phase, depth, chunk_id, surfacing_depth_threshold):
    # The single apex of a turn between a descent and an ascent (or vice versa,
    # for a mid-water W-cast). Only the one deepest/shallowest sample is marked,
    # unless that apex itself is shallow, in which case it's surfacing instead.
    # The run's leading edge may be missing entirely (record starts mid-turn,
    # e.g. no descent ever sampled) as long as the trailing edge confirms the
    # turn; a missing trailing edge is genuinely ambiguous and always skipped.
    n = len(phase)
    for s, e in _runs_by_chunk(phase == UNKNOWN, chunk_id):
        if e == n or chunk_id[e] != chunk_id[e - 1]:
            continue
        start_gap = s == 0 or chunk_id[s - 1] != chunk_id[s]
        before = None if start_gap else phase[s - 1]
        after = phase[e]
        lo = s if start_gap else s - 1

        if after == ASCENT and before in (DESCENT, None):
            idx = lo + np.argmax(depth[lo:e + 1])
        elif after == DESCENT and before in (ASCENT, None):
            idx = lo + np.argmin(depth[lo:e + 1])
        else:
            continue
        phase[idx] = SURFACING if depth[idx] <= surfacing_depth_threshold else INFLECTION


def _classify_transition(phase, depth, chunk_id, surfacing_depth_threshold):
    # Whatever's still unknown immediately either side of a turn - the shoulder
    # between an inflection/surfacing point and the ascent/descent it leads into
    # or out of. Shallow, it's surfacing instead, same backstop as above. Same
    # leading/trailing asymmetry as the inflection pass: the shoulder heading
    # into a turn can have nothing before it at all (the turn point itself
    # confirms it), but the shoulder coming out always needs a real ascent/descent
    # after it, or it's left unknown.
    n = len(phase)
    for s, e in _runs_by_chunk(phase == UNKNOWN, chunk_id):
        if e == n or chunk_id[e] != chunk_id[e - 1]:
            continue
        start_gap = s == 0 or chunk_id[s - 1] != chunk_id[s]
        before = None if start_gap else phase[s - 1]
        after = phase[e]
        turn_to_core = before in (SURFACING, INFLECTION) and after in (ASCENT, DESCENT)
        core_to_turn = after in (SURFACING, INFLECTION) and before in (ASCENT, DESCENT, None)
        if not (turn_to_core or core_to_turn):
            continue
        phase[s:e] = SURFACING if np.median(depth[s:e]) <= surfacing_depth_threshold else TRANSITION


def _assign_profile_and_cycle(phase, chunk_id):
    # Each ascent/descent run is its own profile - no pairing required, so an
    # upcast with no downcast is still a valid, numbered profile. A profile also
    # claims its adjacent transition shoulders, and - only on its leading edge -
    # the single bottom inflection point that marks where it started. It never
    # reaches past surfacing/propelled/a top inflection: those always belong to
    # whatever comes after them, so a bottom turn is never claimed by both the
    # descent before it and the ascent after.
    #
    # Cycle: a new one starts as soon as a descent begins (from the same
    # extended point PROFILE_NUMBER gives it), running up to but not including
    # the start of the next descent - so it carries through the bottom
    # inflection, the ascent, its trailing transition, and surfacing, all as one
    # cycle. An ascent with nothing directly adjacent before its own extended
    # start (upcast-only) starts a fresh cycle the same way a descent would -
    # which is also what makes a mostly-propelled platform start a new cycle
    # each time it actually goes underwater.
    n = len(phase)
    core_mask = (phase == ASCENT) | (phase == DESCENT)
    padded = np.concatenate(([False], core_mask, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])  # exclusive

    profile_num = np.full(n, np.nan)
    cycle = np.ones(n, dtype=np.int32)
    prev_hi = None
    current_cycle = 1
    for k, (s, e) in enumerate(zip(starts, ends), start=1):
        lo = s
        while lo > 0 and phase[lo - 1] == TRANSITION and chunk_id[lo - 1] == chunk_id[lo]:
            lo -= 1
        if phase[s] == ASCENT and lo > 0 and phase[lo - 1] == INFLECTION and chunk_id[lo - 1] == chunk_id[lo]:
            lo -= 1

        hi = e
        while hi < n and phase[hi] == TRANSITION and chunk_id[hi] == chunk_id[hi - 1]:
            hi += 1

        profile_num[lo:hi] = k

        is_new_cycle = prev_hi is None or phase[s] == DESCENT or lo != prev_hi
        if is_new_cycle and prev_hi is not None:
            current_cycle += 1
        cycle[lo:] = current_cycle
        prev_hi = hi

    return profile_num, cycle


def find_profiles(
    df_raw,
    depth_col="PRES",
    smoothing_window_seconds=30,
    velocity_threshold=0.033,
    min_duration_seconds=60,
    gap_threshold_minutes=5,
    surfacing_depth_threshold=2.0,
    min_transect_duration_seconds=300,
):
    df = df_raw.dropna(subset=["TIME", depth_col]).sort_values("TIME")

    if df.empty:
        df_raw["SCI_PHASE"] = UNKNOWN
        df_raw["PROFILE_NUMBER"] = np.nan
        df_raw["PROFILE_DIRECTION"] = np.nan
        df_raw["CYCLE"] = 1
        df_raw["GRADIENT"] = np.nan
        return df_raw

    time_seconds = df["TIME"].to_numpy().astype("int64") / 1e9
    depth = df[depth_col].to_numpy(dtype=float)

    chunk_id = _compute_chunk_id(time_seconds, gap_threshold_minutes * 60)
    smoothed_velocity = _smoothed_velocity(
        depth, df["TIME"].to_numpy(), time_seconds, chunk_id, f"{smoothing_window_seconds}s"
    )

    phase = _classify_ascent_descent(
        smoothed_velocity, time_seconds, chunk_id, velocity_threshold, min_duration_seconds
    )
    _classify_propelled_surfacing(
        phase, depth, time_seconds, chunk_id,
        surfacing_depth_threshold, min_duration_seconds, min_transect_duration_seconds,
    )
    _classify_inflection(phase, depth, chunk_id, surfacing_depth_threshold)
    _classify_transition(phase, depth, chunk_id, surfacing_depth_threshold)

    direction = np.full(len(phase), np.nan)
    direction[phase == ASCENT] = -1
    direction[phase == DESCENT] = 1
    direction[(phase == SURFACING) | (phase == PROPELLED)] = 0

    profile_num, cycle = _assign_profile_and_cycle(phase, chunk_id)

    result = pd.DataFrame(
        {
            "SCI_PHASE": phase,
            "PROFILE_NUMBER": profile_num,
            "PROFILE_DIRECTION": direction,
            "CYCLE": cycle,
            "GRADIENT": smoothed_velocity,
        },
        index=df.index,
    )

    out = df_raw.copy()
    out[DERIVED_COLUMNS] = result.reindex(out.index)
    out["SCI_PHASE"] = out["SCI_PHASE"].fillna(UNKNOWN).astype(int)
    out["CYCLE"] = out["CYCLE"].ffill().fillna(1).astype(int)
    return out


@register_step
class FindProfilesStep(BaseStep, QCHandlingMixin):
    """
    Identifies and classifies vertical and horizontal profiles from depth-time data.

    The step smooths the depth (or pressure) record, derives vertical velocity from
    it, and uses that to label every measurement with a scientific phase. From those
    phases it then derives a continuous ``PROFILE_NUMBER``, a ``PROFILE_DIRECTION``,
    a ``CYCLE`` count, and ``PROFILE_GRADIENT`` (the smoothed vertical velocity used
    for classification, in depth units/s).

    **All parameters are optional** — every parameter has a sensible default, so the
    step runs unchanged on a typical OG1 dataset with no configuration at all (see the
    first example below).

    Phase definitions
    -----------------
    Phases follow the OceanGliders OG1 ``phase`` vocabulary [1]_ and are written to
    ``SCI_PHASE``. This step's scope:

    - **0 – unknown**: the default until reclassified below, and where it stays if
      nothing below applies (including a passively-drifting "parking" stretch,
      which this step does not attempt to detect).
    - **1 – ascent** / **2 – descent**: vertical velocity beyond ``velocity_threshold``,
      sustained for at least ``min_duration_seconds``. This pass runs first and always
      wins — nothing later ever overrides an ascent/descent classification.
    - **3 – surfacing**: whatever's left unknown near the surface (run median depth
      at or below ``surfacing_depth_threshold``) once ascent/descent is settled —
      covers both a flat/undulating stretch at the surface and the shoulder or apex
      of a turn that happens to be shallow.
    - **5 – inflection**: the single deepest/shallowest sample of a non-surface turn
      between a descent and an ascent (or vice versa, for a mid-water turn).
    - **6 – propelled**: a flat, undulating stretch under propulsion away from the
      surface, gated by the longer ``min_transect_duration_seconds`` so a turnaround
      (also briefly near-zero velocity) isn't mistaken for one.
    - **7 – transition**: whatever's still unknown on the shoulder of a turn, between
      an inflection/surfacing point and the ascent/descent either side of it.

    A real data gap longer than ``gap_threshold_minutes`` splits the record into
    disconnected chunks: velocity is never computed, and no run ever allowed, across
    one — so e.g. an upcast whose data simply stops just before surfacing, with no
    matching downcast sampled, is still its own valid profile rather than being
    bridged into a false transition.

    Parameters
    ----------
    depth_column : str, optional
        Depth or pressure column used for the analysis. Default ``"PRES"``.
    smoothing_window_seconds : int, optional
        Rolling-mean window (seconds) applied to depth before differentiating, to
        keep sensor noise from flipping the sign of the velocity. Default ``30``.
    velocity_threshold : float, optional
        Vertical velocity (depth units/s) a smoothed sample must exceed to count as
        moving; below this, in either direction, it's unknown. Default ``0.033``.
    min_duration_seconds : int, optional
        A run of consistent ascent/descent (or of surfacing) shorter than this is
        too brief to trust and is folded back to unknown. Default ``60``.
    gap_threshold_minutes : int, optional
        A time gap longer than this splits the record into disconnected chunks.
        Default ``5``.
    surfacing_depth_threshold : float, optional
        Depth below which a propelled stretch, or the apex/shoulder of a turn, is
        classified surfacing instead — judged on run median depth. Default ``2.0``.
    min_transect_duration_seconds : int, optional
        Minimum duration for an unknown run to be classified propelled. Deliberately
        much longer than ``min_duration_seconds``, since a turnaround also sits
        near-zero velocity for a few seconds but never for minutes. Default ``300``.

    Examples
    --------
    The defaults are designed to work out of the box, so the simplest valid
    configuration sets no parameters at all:

    .. code-block:: yaml

        steps:
          - name: Find Profiles

    To tune the classification, any subset of parameters may be supplied. The block
    below lists every parameter set to its default value:

    .. code-block:: yaml

        steps:
          - name: Find Profiles
            parameters:
              depth_column: "PRES"
              smoothing_window_seconds: 30
              velocity_threshold: 0.033
              min_duration_seconds: 60
              gap_threshold_minutes: 5
              surfacing_depth_threshold: 2.0
              min_transect_duration_seconds: 300

    References
    ----------
    .. [1] OceanGliders Community, OG1 format user manual — ``phase`` vocabulary
       collection.
       https://github.com/OceanGlidersCommunity/OG-format-user-manual/blob/main/vocabularyCollection/phase.md
    """

    step_name = "Find Profiles"
    required_variables = ["TIME"]
    provided_variables = ["PROFILE_NUMBER", "PROFILE_DIRECTION", "PROFILE_GRADIENT", "CYCLE", "SCI_PHASE"]
    variable_parameters = ["depth_column"]
    uses_data_subset = True

    parameter_schema = {
        "depth_column": {
            "type": str,
            "default": "PRES",
            "description": "Depth or pressure column name. Defaults to PRES."
        },
        "smoothing_window_seconds": {
            "type": int,
            "default": 30,
            "description": "Rolling-mean window (seconds) applied to depth before differentiating."
        },
        "velocity_threshold": {
            "type": float,
            "default": 0.033,
            "description": "Vertical velocity (depth units/s) to trigger ascent/descent classification."
        },
        "min_duration_seconds": {
            "type": int,
            "default": 60,
            "description": "Minimum seconds for an ascent/descent/surfacing run to be trusted."
        },
        "gap_threshold_minutes": {
            "type": int,
            "default": 5,
            "description": "Time gap (minutes) that splits the record into disconnected chunks."
        },
        "surfacing_depth_threshold": {
            "type": float,
            "default": 2.0,
            "description": "Depth below which a propelled/turn run is classified surfacing instead."
        },
        "min_transect_duration_seconds": {
            "type": int,
            "default": 300,
            "description": "Minimum duration for an unknown run to be classified propelled."
        },
    }

    def run(self):
        self.log("Attempting to designate profile numbers, cycles, directions, and phases")
        self.check_data()
        self.filter_qc()

        # Parameters are resolved from parameter_schema in BaseStep.__init__,
        # so every declared attribute is guaranteed to be set.
        depth_col = self.depth_column
        if depth_col not in self.data.variables:
            raise ValueError(f"Specified depth column '{depth_col}' not found in the dataset.")

        cols_to_extract = ["TIME", depth_col]
        df_raw = self.data[cols_to_extract].to_dataframe().reset_index()

        # Flagged samples must not shape the depth smoothing/velocity, but every
        # sample is still labelled: results are merged back on TIME, untouched here.
        # Interpolated (flag 8) depth is excluded too, on top of the default
        # calculation_mask (3/4/9): a linearly-interpolated ramp across a real gap
        # (e.g. surface comms) would otherwise be read as genuine depth movement.
        calc_mask = self.calculation_mask(["TIME", depth_col])
        depth_qc_var = f"{depth_col}_QC"
        if depth_qc_var in self.data:
            calc_mask &= self.data[depth_qc_var].values != 8
        df_raw.loc[~calc_mask, depth_col] = np.nan

        df_final = find_profiles(
            df_raw, depth_col,
            smoothing_window_seconds=self.smoothing_window_seconds,
            velocity_threshold=self.velocity_threshold,
            min_duration_seconds=self.min_duration_seconds,
            gap_threshold_minutes=self.gap_threshold_minutes,
            surfacing_depth_threshold=self.surfacing_depth_threshold,
            min_transect_duration_seconds=self.min_transect_duration_seconds,
        )

        if self.diagnostics:
            self.generate_diagnostics(df_final, depth_col)

        self.data["PROFILE_NUMBER"] = (("N_MEASUREMENTS",), df_final["PROFILE_NUMBER"].to_numpy())
        self.data.PROFILE_NUMBER.attrs = {
            "long_name": "Derived profile number. NaN indicates no profile.",
            "units": "None",
            "standard_name": "Profile Number",
            "valid_min": 1,
            "valid_max": np.inf,
        }

        self.data["PROFILE_DIRECTION"] = (("N_MEASUREMENTS",), df_final["PROFILE_DIRECTION"].to_numpy())
        self.data.PROFILE_DIRECTION.attrs = {
            "long_name": "Profile direction: -1 ascent, 1 descent, 0 transect (surfacing/propelled), NaN otherwise.",
            "units": "None",
            "standard_name": "Profile Direction",
            "valid_min": -1,
            "valid_max": 1,
        }

        self.data["PROFILE_GRADIENT"] = (("N_MEASUREMENTS",), df_final["GRADIENT"].to_numpy())
        self.data.PROFILE_GRADIENT.attrs = {
            "long_name": "Smoothed vertical velocity used for phase classification",
            "units": "m/s",
        }

        self.data["CYCLE"] = (("N_MEASUREMENTS",), df_final["CYCLE"].to_numpy())
        self.data.CYCLE.attrs = {
            "long_name": "Continuous cycle number derived from surfacing points",
            "units": "None",
            "standard_name": "Cycle Number",
            "valid_min": 1,
            "valid_max": np.inf,
        }

        self.data["SCI_PHASE"] = (("N_MEASUREMENTS",), df_final["SCI_PHASE"].to_numpy())
        self.data.SCI_PHASE.attrs = {
            "long_name": "Scientific Phase Classification",
            "units": "None",
            "valid_min": 0,
            "valid_max": 7,
            "flag_values": "0, 1, 2, 3, 4, 5, 6, 7",
            "flag_meanings": "unknown ascent descent surfacing parking inflection propelled transition"
        }

        self.generate_qc({
            "PROFILE_NUMBER_QC": ["TIME_QC", f"{depth_col}_QC"],
            "PROFILE_DIRECTION_QC": ["TIME_QC", f"{depth_col}_QC"],
            "PROFILE_GRADIENT_QC": ["TIME_QC", f"{depth_col}_QC"],
            "CYCLE_QC": ["TIME_QC", f"{depth_col}_QC"],
            "SCI_PHASE_QC": ["TIME_QC", f"{depth_col}_QC"]
        })

        self.context["data"].update(self.data)
        return self.context

    def generate_diagnostics(self, mapped_df, depth_col):
        # Two panels: depth coloured by scientific phase, and profile/cycle numbering.
        matplotlib.use("tkagg")
        fig, axes = fig_spec.new_fig(nrows=2, sharex=True, height_ratios=(3, 1))
        ax1, ax2 = axes[0][0], axes[1][0]

        # Panel 1: high-resolution phase mapping, one series per scientific phase.
        has_data = mapped_df[depth_col].notna()
        for p_val in range(8):
            mask = (mapped_df["SCI_PHASE"] == p_val) & has_data
            n_points = int(mask.sum())
            t_data = mapped_df["TIME"][mask] if n_points else []
            depth_data = mapped_df[depth_col][mask] if n_points else []
            lbl = f"{PHASE_NAMES.get(p_val, f'Phase {p_val}')} (n={n_points})"
            fig_spec.points(ax1, t_data, depth_data,
                            color=PHASE_COLOURS.get(p_val, "black"), label=lbl)

        ax1.invert_yaxis()
        fig_spec.style_axes(ax1, title="High Resolution Phase Mapping", ylabel="Pressure/Depth")
        fig_spec.legend(ax1)

        # Panel 2: derived profile & cycle numbering.
        fig_spec.points(ax2, mapped_df["TIME"], mapped_df["PROFILE_NUMBER"],
                        color=fig_spec.CATEGORY[1], label="Profile Number")
        fig_spec.points(ax2, mapped_df["TIME"], mapped_df["CYCLE"],
                        color=fig_spec.CATEGORY[2], label="Cycle Number")
        fig_spec.date_axis(ax2, which="x")
        fig_spec.style_axes(ax2, xlabel="Time", ylabel="ID / Cycle")
        fig_spec.legend(ax2)

        fig_spec.finish(fig, suptitle="Find Profiles Diagnostics")
        plt.show(block=True)
