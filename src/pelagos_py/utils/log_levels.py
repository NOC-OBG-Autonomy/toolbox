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

"""Custom logging levels shared across the pipeline."""

import logging
import os
import sys

STOP = logging.ERROR + 5
"""Custom log level for a deliberate pipeline halt (e.g. invalid config).

Sits just above ERROR so it always surfaces, and is rendered as "STOP" in logs.
"""
logging.addLevelName(STOP, "STOP")


def _supports_color(stream):
    """Whether ANSI colour is safe to emit on ``stream``.

    Only colour a real interactive terminal: skip when output is redirected to
    a file/pipe (``isatty()`` is False), when NO_COLOR is set (no-color.org),
    or on a dumb terminal. FORCE_COLOR overrides all of these.
    """
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


class ColorFormatter(logging.Formatter):
    """Formatter that colours records by level on the console (STOP/ERROR red,
    WARNING yellow).

    Colour is only emitted when the target stream is an interactive terminal
    that supports it (see :func:`_supports_color`), so redirected output and
    dumb terminals stay clean plain text. Intended for stream handlers only;
    the file handler keeps a plain formatter so log files carry no escape codes.
    """

    _RESET = "\033[0m"
    _RED = "\033[31m"
    _YELLOW = "\033[33m"
    _GREY = "\033[90m"

    #: Colours a record may request explicitly via ``extra={"color": ...}`` (e.g.
    #: a step greying its own verbose output). Unknown names leave it uncoloured.
    _NAMED_COLORS = {"grey": _GREY, "gray": _GREY, "red": _RED, "yellow": _YELLOW}

    def __init__(self, *args, stream=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_color = _supports_color(stream if stream is not None else sys.stderr)

    def _color_for(self, record):
        """Return the ANSI colour for a record, or None to leave it uncoloured.

        An explicit ``color`` on the record (set via
        ``logger.info(..., extra={"color": "grey"})``) wins, so a step can mark
        its own output; otherwise colour is chosen by level (ERROR/STOP red,
        WARNING yellow).
        """
        requested = getattr(record, "color", None)
        if requested:
            return self._NAMED_COLORS.get(requested)
        if record.levelno >= logging.ERROR:
            return self._RED
        if record.levelno >= logging.WARNING:
            return self._YELLOW
        return None

    def format(self, record):
        message = super().format(record)
        color = self._color_for(record) if self._use_color else None
        if color:
            return f"{color}{message}{self._RESET}"
        return message
