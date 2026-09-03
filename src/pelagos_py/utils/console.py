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

Every line is ``time  step-name  detail`` (colour by severity: warnings yellow,
SEVERE amber, errors/STOP red), with countable loops shown as a live progress bar. The console
handler reshapes each record and writes through :func:`tqdm.write` so log lines
and a live bar coexist; :func:`progress_bar` is the uniform bar style.
"""

import logging
import re
import sys
import time

from tqdm import tqdm

from pelagos_py.utils.log_levels import SEVERE, _supports_color

_RESET = "\033[0m"
_RED = "\033[31m"
_AMBER = "\033[38;5;202m"
_YELLOW = "\033[33m"

# Width the step/process name is padded to, so the detail columns line up.
_NAME_WIDTH = 20

# Uniform tqdm bar for every countable loop; ``{desc}`` carries the time/name prefix.
BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

# Leading ``[Step Name] `` tag steps prepend via ``BaseStep.log``; lifted into its own column.
_TAG = re.compile(r"^\[([^\]]*)\]\s*")


def _console_enabled(stream=None):
    # False when output is redirected or colour is disabled (tqdm/colour off).
    return _supports_color(stream if stream is not None else sys.stderr)


def _display_name(record):
    # Short step/process name for a record with no ``[...]`` tag.
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
    # The ``time  name`` prefix a progress bar shares with log lines.
    stamp = time.strftime("%H:%M:%S")
    if step_name:
        return f"{stamp}  {step_name:<{_NAME_WIDTH}}  "
    return f"{stamp}  "


class ConsoleFormatter(logging.Formatter):
    """Reshape a record to ``time  name  message`` and colour it by severity.

    Name comes from the ``[...]`` tag when present, else the logger name. WARNING
    yellow, SEVERE amber, ERROR/STOP red. Pass ``extra={"raw": True}`` to emit a
    line verbatim.
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
        # Info stays the terminal default; only warnings/errors colour.
        if record.levelno >= logging.ERROR:
            return f"{_RED}{line}{_RESET}"
        if record.levelno >= SEVERE:
            return f"{_AMBER}{line}{_RESET}"
        if record.levelno >= logging.WARNING:
            return f"{_YELLOW}{line}{_RESET}"
        return line


class _TqdmHandler(logging.StreamHandler):
    """Stream handler that emits through :func:`tqdm.write` so records don't
    corrupt an active progress bar."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:  # pragma: no cover - matches logging.Handler contract
            self.handleError(record)


class _ConsoleOnlyFilter(logging.Filter):
    """Drop records tagged ``extra={"console": False}`` so noisy book-keeping
    reaches the log file but not the console."""

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
    """A countable loop shown as a bar, logging a one-line summary to the file
    (only) on close so the file and report still record the phase."""

    def __init__(
        self, *args, logger=None, step_name=None, summary_unit=None, summary_label=None, **kwargs
    ):
        self._logger = logger
        self._step_name = step_name
        self._summary_unit = summary_unit
        self._summary_label = summary_label
        self._summary_start = time.time()
        self._summarised = False
        # Counted here, not from tqdm's ``n``, so the summary is right on a
        # redirected run where the bar is disabled and never advances.
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
    """A tqdm bar in the pipeline's standard style for every countable loop.

    ``step_name`` adds the ``time  name`` prefix; ``logger`` also writes a
    one-line summary to the file. Auto-disables off a terminal.
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
