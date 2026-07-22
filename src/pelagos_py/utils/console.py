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

"""Console presentation for pipeline runs.

The console mirrors the shape of the log file and report: every line is
``time  step-name  detail``. The only difference is colour (info grey, warnings
yellow, errors/STOP red) and that countable loops are shown as a live progress
bar rather than a static line. The verbose ``LEVEL - logger`` fields stay in the
log *file* only.

Building blocks:

* :class:`ConsoleFormatter` / :func:`make_console_handler` — the console handler.
  It reshapes each record to ``time  name  message`` (dropping the redundant
  ``[Step Name]`` tag and the ``LEVEL - logger`` noise), colours by severity, and
  writes through :func:`tqdm.write` so log lines and a live bar coexist.
* :func:`progress_bar` — the uniform progress-bar style used for step-internal
  loops and pipeline start-up.
"""

import logging
import re
import sys
import time

from tqdm import tqdm

from pelagos_py.utils.log_levels import _supports_color

_RESET = "\033[0m"
_RED = "\033[31m"
_YELLOW = "\033[33m"

#: Width the step/process name is padded to, so the detail columns line up.
_NAME_WIDTH = 20

#: Uniform tqdm bar for every countable loop (step internals + start-up). Kept in
#: one place so progress looks the same everywhere. ``{desc}`` already carries the
#: ``time  name`` prefix (see :func:`_bar_prefix`).
BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

#: Leading ``[Step Name] `` tag steps prepend via ``BaseStep.log``; the name is
#: lifted into its own column so the tag itself is dropped from the message.
_TAG = re.compile(r"^\[([^\]]*)\]\s*")


def _console_enabled(stream=None):
    """Whether live console decoration (colour, in-place bars) is safe here.

    False when output is redirected to a file/pipe or colour is disabled, in
    which case tqdm auto-disables and colour is skipped.
    """
    return _supports_color(stream if stream is not None else sys.stderr)


def _display_name(record):
    """Short step/process name for a record that carries no ``[...]`` tag."""
    name = record.name
    for prefix in ("pelagos_py.pipeline.step.", "pelagos_py.pipeline.qc."):
        if name.startswith(prefix):
            return name[len(prefix):].replace("_", " ")
    if name.startswith("pelagos_py.pipeline.discovery"):
        return "discovery"
    if name == "pelagos_py.pipeline":
        return "pipeline"
    return name.rsplit(".", 1)[-1]


def _bar_prefix(step_name):
    """The ``time  name`` prefix a progress bar shares with log lines."""
    stamp = time.strftime("%H:%M:%S")
    if step_name:
        return f"{stamp}  {step_name:<{_NAME_WIDTH}}  "
    return f"{stamp}  "


class ConsoleFormatter(logging.Formatter):
    """Reshape a record to ``time  name  message`` and colour it by severity.

    The step/process name is taken from the record's ``[...]`` tag when present,
    otherwise derived from the logger name. Colour: INFO grey, WARNING yellow,
    ERROR/STOP red. Pass ``extra={"raw": True}`` to emit a line verbatim.
    """

    def __init__(self, *args, stream=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_color = _console_enabled(stream)

    def format(self, record):
        if getattr(record, "raw", False):
            return record.getMessage()

        message = record.getMessage()
        tag = _TAG.match(message)
        if tag:
            name = tag.group(1)
            message = message[tag.end():]
        else:
            name = _display_name(record)

        stamp = self.formatTime(record, "%H:%M:%S")
        line = f"{stamp}  {name:<{_NAME_WIDTH}}  {message}"

        if not self._use_color:
            return line
        # Info stays the terminal's default (white); only warnings/errors colour.
        if record.levelno >= logging.ERROR:
            return f"{_RED}{line}{_RESET}"
        if record.levelno >= logging.WARNING:
            return f"{_YELLOW}{line}{_RESET}"
        return line


class _TqdmHandler(logging.StreamHandler):
    """Stream handler that emits through :func:`tqdm.write`.

    Routing log output via ``tqdm.write`` keeps records from corrupting an active
    progress bar (the bar is cleared, the line is written, the bar is redrawn).
    """

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:  # pragma: no cover - matches logging.Handler contract
            self.handleError(record)


class _ConsoleOnlyFilter(logging.Filter):
    """Drop records tagged ``extra={"console": False}``.

    Lets noisy book-keeping (per-module discovery, per-step assembly, ``Executing:``)
    reach the log file while the console shows the progress bars instead.
    """

    def filter(self, record):
        return getattr(record, "console", True)


def make_console_handler(level=logging.INFO):
    """Build the pipeline's console handler: compact, coloured, tqdm-aware."""
    handler = _TqdmHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(ConsoleFormatter(stream=sys.stderr))
    handler.addFilter(_ConsoleOnlyFilter())
    return handler


class _PhaseBar(tqdm):
    """A countable loop shown as a bar, with a matching one-line file summary.

    The bar's ``desc`` carries the same ``time  name`` prefix as the log lines, so
    a live loop reads like the rest of the console. On close it logs a summary to
    the file only (the console already showed the bar), so the file and report
    record the same phases.
    """

    def __init__(
        self, *args, logger=None, step_name=None, summary_unit=None, summary_label=None, **kwargs
    ):
        self._logger = logger
        self._step_name = step_name
        self._summary_unit = summary_unit
        self._summary_label = summary_label
        self._summary_start = time.time()
        self._summarised = False
        # Counted here rather than read from tqdm's ``n`` so the file summary is
        # right even on a redirected run, where the bar is disabled and never
        # advances its own counter.
        self._count = 0
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for obj in super().__iter__():
            self._count += 1
            yield obj

    def update(self, n=1):
        self._count += n
        return super().update(n)

    def close(self):
        super().close()
        if self._summarised:
            return
        self._summarised = True
        if self._logger is not None and self._count:
            self._logger.info(
                "[%s] %s: %s %s (%.1fs)",
                self._step_name or "",
                self._summary_label or "progress",
                self._count,
                self._summary_unit or self.unit,
                time.time() - self._summary_start,
                extra={"console": False},
            )


def progress_bar(
    iterable=None,
    *,
    desc="",
    total=None,
    unit="it",
    leave=False,
    logger=None,
    step_name=None,
):
    """A tqdm bar in the pipeline's standard style.

    Use for every countable loop — step internals and pipeline start-up alike —
    so progress looks identical everywhere. When ``step_name`` is given the bar
    gets the same ``time  name`` prefix as the log lines; when ``logger`` is given
    it also writes a one-line summary to the file. Auto-disables off a terminal.
    """
    display = _bar_prefix(step_name) + desc
    return _PhaseBar(
        iterable,
        desc=display,
        total=total,
        unit=unit,
        leave=leave,
        bar_format=BAR_FORMAT,
        colour="white",
        disable=not _console_enabled(),
        logger=logger,
        step_name=step_name,
        summary_unit=unit,
        summary_label=desc,
    )
