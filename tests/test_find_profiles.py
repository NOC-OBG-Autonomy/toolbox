import numpy as np
import pandas as pd

from pelagos_py.steps.processing.find_profiles import find_profiles

# Default parameters, mirroring FindProfilesStep.parameter_schema. Individual
# tests override only what they need via **overrides.
DEFAULT_PARAMS = dict(
    smoothing_window_seconds=30,
    velocity_threshold=0.033,
    min_duration_seconds=60,
    gap_threshold_minutes=5,
    surfacing_depth_threshold=2.0,
    min_transect_duration_seconds=300,
)


def _make_leg(min_depth, max_depth, n, descending):
    # Raised-cosine, not a straight line: real dives ease to a near-zero
    # velocity at both the top and bottom turn, which is what the algorithm
    # relies on to find the inflection point and split consecutive profiles.
    shape = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    if descending:
        return min_depth + (max_depth - min_depth) * shape
    return max_depth - (max_depth - min_depth) * shape


def make_dive_dataframe(
    n_cycles=3, leg_minutes=10, sample_seconds=10, min_depth=1.0, max_depth=120.0, start=None
):
    """Build a clean dive record (descent/ascent legs) for profiling.

    Returns a DataFrame shaped like the one ``FindProfilesStep`` feeds to
    ``find_profiles``: an ``N_MEASUREMENTS`` index column, ``TIME`` and a depth
    column (here ``PRES``).
    """
    leg_samples = int(leg_minutes * 60 / sample_seconds)
    down = _make_leg(min_depth, max_depth, leg_samples, descending=True)
    up = _make_leg(min_depth, max_depth, leg_samples, descending=False)

    depth = np.concatenate([leg for _ in range(n_cycles) for leg in (down, up)])
    n = len(depth)
    # Nanosecond resolution to match real OG1 TIME (the step converts TIME to
    # epoch seconds assuming ns, so a coarser dtype would distort velocities).
    offsets = (np.arange(n) * sample_seconds * 1_000_000_000).astype("timedelta64[ns]")
    times = (start or np.datetime64("2024-01-01T00:00:00", "ns")) + offsets

    return pd.DataFrame(
        {"N_MEASUREMENTS": np.arange(n), "TIME": times, "PRES": depth}
    )


def run(df, **overrides):
    params = {**DEFAULT_PARAMS, **overrides}
    return find_profiles(df, "PRES", **params)


def test_output_columns_and_length():
    """The step preserves row count and adds the expected derived columns."""
    df = make_dive_dataframe()
    result = run(df)

    expected = {"PROFILE_NUMBER", "PROFILE_DIRECTION", "GRADIENT", "CYCLE", "SCI_PHASE"}
    assert expected.issubset(result.columns)
    assert len(result) == len(df)


def test_ascent_and_descent_detected():
    """Clear up/down legs are classified as ascent (1) and descent (2), and the
    derived direction is consistent with the phase (+1 descent, -1 ascent)."""
    result = run(make_dive_dataframe())
    phases = result["SCI_PHASE"]

    assert (phases == 1).any(), "expected some ascent samples"
    assert (phases == 2).any(), "expected some descent samples"

    descent = result.loc[phases == 2, "PROFILE_DIRECTION"]
    ascent = result.loc[phases == 1, "PROFILE_DIRECTION"]
    assert (descent == 1).all()
    assert (ascent == -1).all()


def test_multiple_profiles_numbered():
    """Several dive cycles produce several distinct, positive profile numbers."""
    result = run(make_dive_dataframe(n_cycles=3))
    profile_numbers = result["PROFILE_NUMBER"].dropna()

    assert profile_numbers.nunique() >= 2
    assert (profile_numbers >= 1).all()


def test_empty_input_returns_defaults():
    """An empty input still yields the derived columns without raising."""
    empty = pd.DataFrame({"N_MEASUREMENTS": [], "TIME": [], "PRES": []})
    result = run(empty)

    for col in ("PROFILE_NUMBER", "PROFILE_DIRECTION", "GRADIENT", "CYCLE", "SCI_PHASE"):
        assert col in result.columns


def test_gap_ending_before_surface_stays_ascent():
    """An upcast-only leg that stops mid-ascent at steady velocity (e.g. surface
    comms cut PRES off before reaching the true surface - not a natural turn, so
    no deceleration), followed by a real data gap, must not have its tail forced
    into a fabricated surfacing/transition phase - it should stay genuine ascent
    right up to the gap. The next leg should start its own cycle/profile rather
    than being bridged across the gap."""
    n, sample_seconds = 60, 10
    offsets = (np.arange(n) * sample_seconds * 1_000_000_000).astype("timedelta64[ns]")
    times = np.datetime64("2024-01-01T00:00:00", "ns") + offsets
    depth = np.linspace(100.0, 2.5, n)  # steady ascent, cut off abruptly - no easing
    first = pd.DataFrame({"N_MEASUREMENTS": np.arange(n), "TIME": times, "PRES": depth})

    gap_start = first["TIME"].iloc[-1] + pd.Timedelta(minutes=15)
    second = make_dive_dataframe(n_cycles=1, min_depth=2.5, start=gap_start.to_numpy())
    second["N_MEASUREMENTS"] += len(first)
    df = pd.concat([first, second], ignore_index=True)

    result = run(df)

    tail = result.iloc[len(first) - 5 : len(first)]
    assert (tail["SCI_PHASE"] == 1).all(), "ascent tail before the gap was overwritten"

    profile_numbers = result["PROFILE_NUMBER"].dropna()
    assert profile_numbers.nunique() >= 2
    assert result["CYCLE"].nunique() >= 2
