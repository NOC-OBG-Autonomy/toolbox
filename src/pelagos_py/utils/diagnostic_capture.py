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

"""Capture step diagnostic figures for the PDF report.

Steps draw their diagnostics with matplotlib and display them with
``plt.show()`` when ``diagnostics`` is enabled. To include those plots in the
report regardless of a step's own ``diagnostics`` setting, the pipeline
force-enables diagnostics and runs each step inside :func:`capture_figures`,
which redirects ``plt.show`` so the figures are written to disk instead of being
displayed. The saved paths are then embedded by the report writer.
"""

import contextlib
import functools
import os

import matplotlib
import matplotlib.pyplot as plt


@contextlib.contextmanager
def force_headless_backend():
    """Force the headless Agg backend for the whole report run.

    Report figures are saved to disk, never displayed, so no GUI backend is
    needed. Steps' diagnostics commonly call ``matplotlib.use("tkagg")`` to
    force the interactive Tk backend; under report capture that is both
    pointless and dangerous - instantiating a Tk canvas off the main thread
    (e.g. when driven from an IDE/GUI) can hard-crash the process on Windows.

    The backend is switched to Agg exactly once here and ``matplotlib.use()`` is
    neutralised for the duration, so every step's backend switch becomes a
    harmless no-op and the backend is never toggled mid-run (repeated
    ``use(..., force=True)`` re-initialises matplotlib's compiled backend, which
    can trigger a fatal delay-load crash on Windows). Both are restored on exit.
    """
    original_use = matplotlib.use
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    matplotlib.use = lambda *args, **kwargs: None
    try:
        yield
    finally:
        matplotlib.use = original_use
        #   Best-effort restore: never let backend restoration break the run.
        try:
            matplotlib.use(original_backend, force=True)
        except Exception:  # noqa: BLE001 - backend restore must never be fatal
            pass


def _save_open_figures(
    outdir: str, step_name: str, step_index: int, images: list, close: bool = True
) -> None:
    """Save every open matplotlib figure to ``outdir`` for the report.

    Each saved path is appended to ``images``. Saving failures are swallowed so
    capturing a diagnostic can never interrupt the pipeline. When ``close`` is
    ``True`` each figure is closed after saving so figures are neither displayed
    nor leaked between steps; when ``False`` the figures are left open so they
    can still be shown interactively afterwards (used for steps the user
    explicitly enabled diagnostics on).

    ``step_index`` (the step's pipeline position) is in the filename so repeated
    step names (e.g. several 'Apply QC' steps) don't overwrite each other.
    """
    safe = f"{step_index:02d}_" + step_name.replace(os.sep, "_").replace(" ", "_")
    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = os.path.join(outdir, f"{safe}_{len(images) + 1}.png")
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            images.append(path)
        except Exception:  # noqa: BLE001 - a capture failure must never be fatal
            pass
        finally:
            if close:
                plt.close(fig)


@contextlib.contextmanager
def capture_figures(
    outdir: str,
    step_name: str,
    step_index: int,
    images: list,
    suppress_text: bool = False,
    interactive: bool = False,
):
    """Redirect ``plt.show`` so diagnostic figures are saved for the report.

    Within the context, every ``plt.show`` call (and any figures still open when
    the context exits) is written to ``outdir`` and its path recorded in
    ``images``.

    Parameters
    ----------
    outdir : str
        Directory the figures are written into.
    step_name : str
        Name of the step producing the figures (used to name the files).
    step_index : int
        The step's pipeline position, used to keep repeated step names apart.
    images : list
        List that captured image paths are appended to, in the order produced.
    suppress_text : bool, optional
        When ``True``, stdout is silenced for the duration of the context. Used
        when diagnostics are force-enabled only to capture figures for the
        report: a step whose diagnostics are textual (``print`` rather than a
        plot, e.g. the dataset summary from Load Data) would otherwise dump that
        text to the console even though the user never enabled diagnostics.
    interactive : bool, optional
        When ``False`` (default) figures are saved and closed silently - no
        window is shown. This is used for steps whose diagnostics were
        force-enabled only to harvest figures for the report, and keeps the run
        headless (Agg) and Windows-safe.

        When ``True`` (the user explicitly set ``diagnostics: true`` on this
        step) the figure is saved to the report *and* shown in a blocking popup,
        so the pipeline pauses on that step exactly as it would outside a report
        run. This temporarily switches to the interactive Tk backend for the
        duration of the step and switches back to headless Agg afterwards; the
        GUI backend is the reason this path is opt-in only.
    """
    original_show = plt.show

    if interactive:
        #   Switch this step to the interactive Tk backend so its figure can be
        #   displayed. ``matplotlib.use`` is neutralised for the run, so switch
        #   via pyplot directly.
        #
        #   NOTE: deliberately NOT wrapped in try/except. If the Tk backend
        #   cannot be loaded (e.g. the known Windows crash) this must fail loudly
        #   so the underlying problem is visible and can be fixed. Do NOT add a
        #   fallback that degrades to save-only / headless here: a silent
        #   fallback hides whether the backend is actually working and makes the
        #   bug look "fixed" when it is not.
        plt.switch_backend("tkagg")

    def _capture_show(*args, **kwargs):
        if interactive:
            #   Save into the report first (leaving the figures open), then show
            #   them in a blocking window so the user sees the plot and the
            #   pipeline pauses. Close afterwards so figures don't leak.
            _save_open_figures(outdir, step_name, step_index, images, close=False)
            try:
                original_show(*args, **{**kwargs, "block": True})
            finally:
                for num in plt.get_fignums():
                    plt.close(num)
        else:
            _save_open_figures(outdir, step_name, step_index, images, close=True)

    plt.show = _capture_show

    try:
        with contextlib.ExitStack() as stack:
            if suppress_text:
                devnull = stack.enter_context(open(os.devnull, "w"))
                stack.enter_context(contextlib.redirect_stdout(devnull))
            yield
    finally:
        plt.show = original_show
        #   Catch any figures a diagnostic left open without calling show()
        #   (headless: never display these, just save them).
        _save_open_figures(outdir, step_name, step_index, images, close=True)
        if interactive:
            #   Return to the headless Agg backend forced for the rest of the
            #   run, so only opt-in steps ever touch the GUI backend.
            try:
                plt.switch_backend("agg")
            except Exception:  # noqa: BLE001 - never let a backend switch be fatal
                pass


def make_diagnostics_safe(step) -> None:
    """Wrap a step's diagnostic methods so a failure never aborts the step.

    Used when diagnostics are force-enabled to capture plots for the report: a
    user who did not opt into diagnostics should never have the pipeline fail
    because of a diagnostic-only error. The error is logged as a warning and the
    step's core processing continues.
    """
    for attr in ("generate_diagnostics", "plot_diagnostics"):
        method = getattr(step, attr, None)
        if not callable(method):
            continue

        @functools.wraps(method)
        def safe(*args, _method=method, **kwargs):
            try:
                return _method(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - diagnostics must never be fatal
                step.logger.warning(
                    "[%s] Diagnostic generation for the report failed: %s",
                    step.name,
                    exc,
                )
                return None

        setattr(step, attr, safe)
