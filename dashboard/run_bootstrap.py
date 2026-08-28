"""Subprocess entry point the dashboard uses to run a pipeline.

Run as::

    python run_bootstrap.py <config_path> <figure_dir>

The dashboard runs the pipeline in a headless subprocess, so a step's
``plt.show()`` diagnostic popup is a silent no-op and the user never sees the
plot. This launcher redirects ``plt.show`` to *save* every open figure into
``figure_dir`` and print a marker line the dashboard picks off the log stream::

    __PELAGOS_FIG__ <filename>\t<caption>\t<spec filename or "">

so the browser can display the plot inline (see run.js). Alongside the PNG it
also tries to write a JSON plot spec (see fig_spec.py) holding the figure's
underlying x/y data, which the browser redraws with plotly so the plot can be
zoomed and panned. Figures that cannot be serialised faithfully just get an
empty spec field and stay PNG-only. It also forces the Agg
backend and neutralises backend switches, so no step can grab a GUI backend
(e.g. ``matplotlib.use("tkagg")``) that would crash in this displayless process.

This is deliberately dashboard-only glue: it changes nothing in pelagos_py, it
just wraps how the plots are surfaced. Only steps that actually call
``plt.show`` (i.e. ``diagnostics: true``) produce plots here.

A step whose diagnostics are log-only (e.g. Load Data, Export — they print a
summary instead of plotting) draws no figure, so it gets a ``__PELAGOS_LOG__``
marker instead::

    __PELAGOS_LOG__ <step index>\t<step name>\t<QC test or "">\t<base64 text>

so the dashboard can show that text where a plot would otherwise go. See
``_patch_diagnostics_capture``.
"""

import base64
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend setup)

# Force the Agg backend module to load *now*, while switch_backend still works.
# Neutralising it below (see next comment) before this would leave the backend
# uninitialised, so the first plt.figure() would crash.
plt.switch_backend("Agg")

# Neutralise backend switches: this subprocess has no display, so any step that
# tries to grab a GUI backend (e.g. matplotlib.use("tkagg")) must stay on
# headless Agg instead of crashing. Safe now that Agg is already initialised.
matplotlib.use = lambda *args, **kwargs: None
plt.switch_backend = lambda *args, **kwargs: None

# Imported after the backend is settled (it pulls in matplotlib itself). Python
# puts this script's directory on sys.path, so the dashboard-local module
# resolves even though the run's cwd is the repo root.
import fig_spec  # noqa: E402

FIG_DIR = sys.argv[2]
_saved = {"n": 0}
_orig_show = plt.show

# Live memory readout for the dashboard's RAM meter. RSS is this process's
# resident set (the runner and every step share this one process). The transient
# spike a step makes usually happens *inside* run() and is freed before the step
# returns, so sampling only at step boundaries would miss it and mis-attribute
# the peak. A background thread polls RSS a few times a second, tracks the max
# *during* each step, and remembers which step the run's overall peak fell in --
# so the meter can point at the step that actually blew RAM up. ``data`` is the
# xarray dataset's own byte size, separating genuine dataset growth from
# transient per-step overhead.
_mem = {
    "label": "startup",   # step the sampler currently attributes RSS to
    "step_start": 0.0,    # RSS (MB) entering the current step
    "step_peak": 0.0,     # max RSS (MB) seen during the current step
    "run_peak": 0.0,      # max RSS (MB) over the whole run
    "run_peak_label": "", # the step run_peak occurred during
}
_mem_lock = threading.Lock()


def _rss_mb():
    """This process's resident set in MB, or None if psutil is unavailable."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:  # noqa: BLE001 - the meter is a bonus, never fatal
        return None


def _mem_sampler():
    """Poll RSS ~2.5x/sec, recording the per-step and whole-run peaks."""
    while True:
        rss = _rss_mb()
        if rss is not None:
            with _mem_lock:
                if rss > _mem["step_peak"]:
                    _mem["step_peak"] = rss
                if rss > _mem["run_peak"]:
                    _mem["run_peak"] = rss
                    _mem["run_peak_label"] = _mem["label"]
        time.sleep(0.4)


def _mem_begin(label):
    """Attribute subsequent samples to ``label``; record the entry RSS."""
    rss = _rss_mb() or 0.0
    with _mem_lock:
        _mem["label"] = label
        _mem["step_start"] = rss
        _mem["step_peak"] = rss


def _emit_mem(context):
    """Print a ``__PELAGOS_MEM__`` marker for the step that just finished.

    Fields (tab-separated; labels may contain spaces but not tabs):
      settle RSS · run peak · dataset MB · step label · in-step peak · peak step
      · step-start RSS
    The last lets the meter show each step's *own* growth (peak - start),
    isolating what a step added from the ratcheted baseline it sat on.
    Best-effort: no psutil -> no marker, never fatal.
    """
    rss_mb = _rss_mb()
    if rss_mb is None:
        return
    with _mem_lock:
        step_peak = max(_mem["step_peak"], rss_mb)
        step_start = _mem["step_start"]
        if rss_mb > _mem["run_peak"]:
            _mem["run_peak"] = rss_mb
            _mem["run_peak_label"] = _mem["label"]
        run_peak = _mem["run_peak"]
        run_peak_label = _mem["run_peak_label"]
        label = _mem["label"]
    data_mb = ""
    try:
        data = (context or {}).get("data")
        nbytes = getattr(data, "nbytes", None)
        if nbytes is not None:
            data_mb = f"{nbytes / 1024 ** 2:.1f}"
    except Exception:  # noqa: BLE001 - dataset size is optional detail
        data_mb = ""
    print(
        f"__PELAGOS_MEM__ {rss_mb:.1f}\t{run_peak:.1f}\t{data_mb}\t{label}"
        f"\t{step_peak:.1f}\t{run_peak_label}\t{step_start:.1f}",
        flush=True,
    )


if _rss_mb() is not None:
    threading.Thread(target=_mem_sampler, daemon=True).start()


def _caption(fig) -> str:
    """Best-effort human label for a figure: its suptitle, else first axes title."""
    try:
        text = ""
        if fig._suptitle and fig._suptitle.get_text():
            text = fig._suptitle.get_text()
        else:
            for ax in fig.axes:
                if ax.get_title():
                    text = ax.get_title()
                    break
        # The __PELAGOS_FIG__ record is one tab-separated line; a multi-line
        # title (e.g. a correction formula on its own line) would otherwise
        # split the record and hide the spec field, forcing PNG-only.
        return " ".join(text.split())
    except Exception:  # noqa: BLE001 - a caption is cosmetic, never fatal
        return ""


def _write_fullres(fullres, stem: str):
    """Write full-resolution trace captures for zoomed-in range queries.

    One ``.npz`` per figure, entries named ``"<panel>_<trace>_x"`` etc. so
    app.py's figdata route can pick a single trace back out by index without
    parsing the whole archive's structure. Best-effort and silent: a zoomed
    trace just won't offer extra detail if this fails.
    """
    if not fullres:
        return
    try:
        import numpy as np

        arrays = {}
        for (panel_idx, trace_idx), capture in fullres.items():
            key = f"{panel_idx}_{trace_idx}"
            arrays[f"{key}_x"] = capture["x"]
            arrays[f"{key}_y"] = capture["y"]
            if capture["color"] is not None:
                arrays[f"{key}_color"] = capture["color"]
        np.savez(os.path.join(FIG_DIR, stem + "_full.npz"), **arrays)
    except Exception:  # noqa: BLE001 - a bonus feature, never fatal
        pass


def _capture_spec(fig, stem: str):
    """Write the figure's interactive plot spec: ``(filename, reason)``.

    Serialising is best-effort in every sense: a figure fig_spec cannot
    represent faithfully yields no file (and anything unexpected is swallowed),
    leaving the dashboard with the PNG it has already saved. ``reason`` says
    what stopped it, so the log tells the user which plots are PNG-only. Any
    traces eligible for a zoomed-in range query also get a full-resolution
    ``.npz`` sidecar (see ``_write_fullres``).
    """
    try:
        spec, reason, fullres = fig_spec.serialise(fig)
        if spec is None:
            return "", reason
        name = stem + ".json"
        with open(os.path.join(FIG_DIR, name), "w") as handle:
            json.dump(spec, handle)
        _write_fullres(fullres, stem)
        return name, ""
    except Exception as exc:  # noqa: BLE001 - a bonus feature, never fatal
        return "", f"{type(exc).__name__}"


def _capture_show(*args, **kwargs):
    """Save every open figure to FIG_DIR and announce it, instead of displaying."""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        _saved["n"] += 1
        stem = f"fig_{_saved['n']:03d}"
        fname = stem + ".png"
        try:
            fig.savefig(os.path.join(FIG_DIR, fname), dpi=130, bbox_inches="tight")
            spec, reason = _capture_spec(fig, stem)
            # Tab-separated so the dashboard can split the fields apart; a
            # caption may contain spaces but not tabs/newlines.
            print(f"__PELAGOS_FIG__ {fname}\t{_caption(fig)}\t{spec}\t{reason}",
                  flush=True)
        except Exception:  # noqa: BLE001 - a capture failure must never be fatal
            pass
        finally:
            plt.close(fig)


plt.show = _capture_show


class _Tee:
    """A writable stream that mirrors everything to two underlying streams."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, s):
        self._primary.write(s)
        self._secondary.write(s)
        return len(s)

    def flush(self):
        self._primary.flush()

    def __getattr__(self, name):
        return getattr(self._primary, name)


# Text captured from a step's diagnostics call when it drew no figure, so it
# can be shown as a "log" in the dashboard instead of a plot. ``None`` means
# "not capturing" (the fast path, for every non-pausable step); a list means
# the current step is pausable and its diagnostics text is being collected.
_diag_capture = {"chunks": None}


def _patch_diagnostics_capture():
    """Make every step's diagnostics call capture its printed text.

    Piggybacks on ``BaseStep._wrap_diagnostics_timing``, which already wraps
    ``generate_diagnostics``/``plot_diagnostics`` per-instance for every step.
    When the wrapped call draws no new figure (a load/export-style step that
    only prints a summary), its stdout is stashed in ``_diag_capture`` so
    ``_emit_diag_log`` can announce it as a ``__PELAGOS_LOG__`` marker once the
    step finishes — the dashboard then shows that text in place of a plot.
    """
    from pelagos_py.steps.base_step import BaseStep

    orig_wrap = BaseStep._wrap_diagnostics_timing

    def wrap_with_capture(self):
        orig_wrap(self)
        for attr in ("generate_diagnostics", "plot_diagnostics"):
            method = getattr(self, attr, None)
            if not callable(method):
                continue

            def captured(*args, _method=method, **kwargs):
                if _diag_capture["chunks"] is None:
                    return _method(*args, **kwargs)
                fig_before = _saved["n"]
                buf = io.StringIO()
                with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
                    result = _method(*args, **kwargs)
                if _saved["n"] == fig_before and buf.getvalue().strip():
                    _diag_capture["chunks"].append(buf.getvalue())
                return result

            setattr(self, attr, captured)

    BaseStep._wrap_diagnostics_timing = wrap_with_capture


def _begin_diag_capture(pausable):
    _diag_capture["chunks"] = [] if pausable else None


def _emit_diag_log(idx, name, test):
    """Print a ``__PELAGOS_LOG__`` marker for text captured since ``_begin_diag_capture``.

    No-op unless the step actually printed diagnostics text and drew no
    figure. Fields (tab-separated): step index, step name, QC test (or empty),
    base64-encoded text.
    """
    chunks = _diag_capture["chunks"]
    _diag_capture["chunks"] = None
    if not chunks:
        return
    text = "\n".join(chunks).strip()
    if not text:
        return
    payload = base64.b64encode(text.encode()).decode()
    print(f"__PELAGOS_LOG__ {idx}\t{name}\t{test or ''}\t{payload}", flush=True)


def _emit_report(context, since):
    """Announce a PDF report a step just wrote, for the dashboard's Report tab.

    Prints ``__PELAGOS_REPORT__ <abspath>\t<filename>``. Best-effort: looks under
    the run's ``out_directory`` for the newest ``.pdf`` touched since the report
    step began, so it works for both the Python and Sphinx report steps without
    hardcoding their filename logic. Nothing found -> no marker.
    """
    try:
        gp = (context or {}).get("global_parameters") or {}
        base = Path(gp.get("out_directory") or "./")
        if not base.is_absolute():
            base = Path.cwd() / base
        newest, newest_mtime = None, since
        for pdf in base.rglob("*.pdf"):
            try:
                mtime = pdf.stat().st_mtime
            except OSError:
                continue
            if mtime >= newest_mtime:
                newest, newest_mtime = pdf, mtime
        if newest is not None:
            print(f"__PELAGOS_REPORT__ {newest.resolve()}\t{newest.name}", flush=True)
    except Exception:  # noqa: BLE001 - surfacing the report is a bonus, never fatal
        pass


def _snapshot(context):
    """Copy the pipeline context so a re-run can start from the pre-step state.

    The dataset under ``data`` is copied deeply because steps mutate it in place;
    re-running a step on already-mutated data would give the wrong result. Other
    context values are shared (cheap and not destructively mutated in a way that
    matters for a single-step re-run).
    """
    if context is None:
        return None
    snap = dict(context)
    data = snap.get("data")
    if data is not None and hasattr(data, "copy"):
        try:
            snap["data"] = data.copy(deep=True)
        except TypeError:
            snap["data"] = data.copy()
    return snap


def _qc_tests(step_config):
    """The ``qc_settings`` mapping of a QC container step, or ``None``."""
    settings = (step_config.get("parameters") or {}).get("qc_settings")
    return settings if isinstance(settings, dict) and settings else None


def _pausable(step_config, test):
    """Whether the run should stop after this unit for the user to look at it."""
    step_diag = bool(step_config.get("diagnostics"))
    if test is None:
        return step_diag
    settings = _qc_tests(step_config) or {}
    return bool((settings.get(test) or {}).get("diagnostics", step_diag))


def _expand(step_config):
    """Split a QC step into one execution per test, as ``(config, test)`` pairs.

    Apply QC runs every test it is given in a single call, so the dashboard
    could only ever pause once the whole batch was done — all the plots at once,
    and a re-run form covering every test. Running each test as its own Apply QC
    step (exactly what a config could spell out by hand) makes each one a unit
    the user can inspect and re-run on its own.

    Only done when the step would pause anyway, so ordinary runs are unaffected.
    """
    tests = _qc_tests(step_config)
    if not tests or len(tests) < 2:
        return [(step_config, None)]
    if not any(_pausable(step_config, name) for name in tests):
        return [(step_config, None)]
    units = []
    for name, settings in tests.items():
        sub = dict(step_config)
        sub["parameters"] = dict(step_config["parameters"], qc_settings={name: settings})
        units.append((sub, name))
    return units


def _read_command():
    """Block until the dashboard sends a control line on stdin.

    Protocol (one line):
      ``continue``            -> proceed to the next step
      ``rerun <json params>`` -> re-run the just-paused step with new parameters
      ``stop``                -> abort the run
    EOF (control pipe closed) is treated as ``continue`` so a dropped channel
    can never hang the run forever.
    """
    line = sys.stdin.readline()
    if not line:
        return ("continue", None)
    line = line.rstrip("\n")
    if line.startswith("rerun "):
        try:
            return ("rerun", json.loads(line[len("rerun "):]))
        except Exception:  # noqa: BLE001 - a malformed command just continues
            return ("continue", None)
    if line == "stop":
        return ("stop", None)
    return ("continue", None)


def main():
    config_path = sys.argv[1]
    report_present = False
    try:
        from pelagos_py.pipeline import REPORT_STEP_NAME, SEVERE, STOP, Pipeline
        from pelagos_py.utils.valid_config_check import check_pipeline_variables

        _patch_diagnostics_capture()
        pipeline = Pipeline(config_path=config_path)

        # Mirror Pipeline.run()'s pre-flight validation.
        try:
            check_pipeline_variables(pipeline.steps, pipeline.logger)
        except ValueError:
            pipeline.logger.log(
                STOP,
                "Pipeline stopped before execution. "
                "Resolve the validation error above and re-run.",
            )
            sys.exit(1)

        # Mirror Pipeline.run()'s report-capture setup: when a report step is
        # present, execute_step() force-captures every step's diagnostic plots
        # (regardless of that step's own diagnostics setting) so the report can
        # embed them. Since the dashboard drives execute_step() itself (below)
        # rather than calling pipeline.run(), it must set this up too - without
        # it, self._capture_diagnostics stays False and the report is written
        # with no "Step diagnostics" section at all.
        report_present = any(s["name"] == REPORT_STEP_NAME for s in pipeline.steps)
        if report_present:
            pipeline._capture_diagnostics = True
            pipeline._captured_figures = []
            pipeline._capture_dir = tempfile.mkdtemp(prefix="pelagos_report_diag_")

        # Drive the steps ourselves (same loop as run()) so we can pause after a
        # diagnostics step: its plot has streamed to the Plots tab, and the user
        # can inspect it, tweak that step's params in the dashboard, and re-run
        # just that step before continuing.
        context = pipeline._context
        # A QC step becomes several units, one per test; everything else is a
        # single unit. `idx` stays the step's index in the config either way, so
        # figures and the re-run form still line up with the builder card.
        units = [
            (idx, sub, test)
            for idx, step_config in enumerate(pipeline.steps)
            for sub, test in _expand(step_config)
        ]
        for idx, step_config, test in units:
            pausable = _pausable(step_config, test)
            # Snapshot the pre-step state only when we might re-run this step.
            # For a split QC step that is the state before *this test*, so a
            # re-run replays one test rather than the whole batch.
            snapshot = _snapshot(context) if pausable else None
            label = step_config.get("name", "") + (f"\t{test}" if test else "")
            # Announce the step *before* it runs so the dashboard can attribute
            # the figures it emits. The pipeline's own "Executing:" log line is
            # file-only (extra={"console": False}), so it never reaches here.
            print(f"__PELAGOS_STEP__ {idx}\t{label}", flush=True)
            mem_label = step_config.get("name", "") + (f" · {test}" if test else "")
            _mem_begin(mem_label)
            report_since = time.time()
            _begin_diag_capture(pausable)
            try:
                context = pipeline.execute_step(step_config, context)
            except (RuntimeError, SystemExit):
                _diag_capture["chunks"] = None
                # Mirror Pipeline.run()'s continue_on_step_fail handling, which
                # this loop otherwise bypasses by driving execute_step() itself.
                if not pipeline.global_parameters.get("continue_on_step_fail", True):
                    pipeline.logger.log(
                        STOP, "Pipeline stopped at step '%s'.", label
                    )
                    sys.exit(1)
                # The fatal-error log from execute_step() already carries the
                # detail; this just marks the step skipped.
                pipeline.logger.log(SEVERE, "Step '%s' failed and was skipped.", label)
                continue
            _emit_diag_log(idx, step_config.get("name", ""), test)
            _emit_mem(context)
            # A report step drops a PDF under out_directory; surface it so the
            # dashboard can offer to open it once the run reaches it.
            if "report" in (step_config.get("name") or "").lower():
                _emit_report(context, report_since - 2)
            if not pausable:
                continue
            while True:
                print(f"__PELAGOS_PAUSE__ {idx}\t{label}", flush=True)
                action, params = _read_command()
                if action == "continue":
                    break
                if action == "stop":
                    print("Pipeline stopped.", flush=True)
                    sys.exit(130)
                if action == "rerun":
                    print(f"__PELAGOS_RERUN__ {idx}", flush=True)
                    print(f"__PELAGOS_STEP__ {idx}\t{label}", flush=True)
                    rerun_config = dict(step_config)
                    if params is not None:
                        # A split QC unit only ever re-runs its own test, even if
                        # the browser sent the whole step's settings.
                        if test is not None and isinstance(params.get("qc_settings"), dict):
                            params = dict(
                                params,
                                qc_settings={
                                    test: params["qc_settings"].get(
                                        test, (_qc_tests(step_config) or {}).get(test, {})
                                    )
                                },
                            )
                        rerun_config["parameters"] = params
                    # The other half of the round trip: what the pipeline was
                    # actually handed, next to what the browser said it sent.
                    print(
                        f"Re-running with parameters: {rerun_config.get('parameters')}",
                        flush=True,
                    )
                    # Fresh copy each re-run so repeated re-runs all start clean.
                    _mem_begin(mem_label)
                    _begin_diag_capture(True)  # only a pausable unit can be re-run
                    context = pipeline.execute_step(rerun_config, _snapshot(snapshot))
                    _emit_diag_log(idx, step_config.get("name", ""), test)
                    _emit_mem(context)
        pipeline._context = context
    except KeyboardInterrupt:
        # The Stop button sends SIGINT (works while paused on stdin too); exit
        # cleanly instead of dumping a traceback from wherever it landed.
        print("Pipeline stopped.", flush=True)
        sys.exit(130)
    finally:
        if report_present:
            # Figures have been embedded by the report writer by now.
            shutil.rmtree(pipeline._capture_dir, ignore_errors=True)
            pipeline._capture_diagnostics = False


if __name__ == "__main__":
    main()
