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

"""Pipeline class definition to handle configuration and step execution."""

import yaml
import pandas as pd
import numpy as np
import xarray as xr
import os
import time
import logging
import datetime as _dt
import difflib
import contextlib
import shutil
import tempfile

from pelagos_py.utils.config_mirror import ConfigMirrorMixin
from pelagos_py.utils.valid_config_check import check_pipeline_variables
from pelagos_py.utils.log_levels import STOP, ColorFormatter
from pelagos_py.utils import diagnostic_capture

REPORT_STEP_NAME = "Write Data Report (Python)"
"""Name of the report step that triggers background diagnostic capture."""

from pelagos_py.steps import create_step, STEP_CLASSES, resolve_step_name

_PIPELINE_LOGGER_NAME = "pelagos_py.pipeline"
"""Global logger name for the pipeline. Used to create child loggers for steps."""


def _setup_logging(out_dir=None, log_file=None, level=logging.INFO):
    """
    Set up logging for the entire pipeline.

    Console logging is always enabled. A log file is only written when
    ``log_file`` is set to a real name; if it is omitted, empty, or a
    "none"/"null" value (including YAML's ``log_file: None``), no file is
    written and a warning is logged.

    Parameters
    ----------
    out_dir : str, optional
        Directory the log file is written into. Defaults to the current
        directory.
    log_file : str, optional
        Name of the log file. If omitted or set to a none-like value, no log
        file is written (console logging still applies).
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(_PIPELINE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Reconfigure from scratch each call so the latest pipeline's config is
    # honoured. Without this, a console-only logger configured by an earlier
    # Pipeline() in the same process would block a later Pipeline(config_path=...)
    # from ever adding its requested file handler.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # Console handler. Always added so logs reach the console regardless of
    # whether a log file is configured. Uses a color formatter so STOP/ERROR
    # lines show up red in the terminal (file handler stays plain, below).
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(
        ColorFormatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
            stream=ch.stream,
        )
    )
    logger.addHandler(ch)

    # Treat unset / explicit "none"-like values as "no log file". This catches
    # YAML's `log_file: None`, which parses to the string "None" (truthy), as
    # well as null/empty values.
    if isinstance(log_file, str) and log_file.strip().lower() in {
        "",
        "none",
        "null",
    }:
        log_file = None

    # File handler if specified
    if log_file:
        log_file = os.path.abspath(
            os.path.join(out_dir or ".", log_file)
        )  # absolute path
        os.makedirs(
            os.path.dirname(log_file) or ".", exist_ok=True
        )  #   Builds logfile directory
        fh = logging.FileHandler(log_file, "w+")  #   Init the logfile
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info(
            "Logging to file: %s", log_file
        )  #   Should not be an empty file at the end of this
    else:
        out_dir = out_dir or "."
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger.warning(
            "log_file not specified - logging to console only, no log file written."
        )

    return logger


class Pipeline(ConfigMirrorMixin):
    """
    Pipeline that manages a sequence of processing steps.

    Config-aware pipeline that can:
      - Load config YAML into private self._parameters
      - Keep global_parameters mirrored to _parameters['pipeline']
      - Build, run, and export steps as before

    Parameters
    ----------
    ConfigMirrorMixin : Class
        Class to handle configuration

    """

    def __init__(self, config_path=None, config=None):
        """
        Initialize pipeline from a config file, an in-memory config dict, or
        neither (build it up manually with add_step).

        Parameters
        ----------
        config_path : str, optional
            Path to the YAML configuration file.
        config : dict, optional
            In-memory configuration dictionary (same structure as the YAML
            file: a ``pipeline`` block and a ``steps`` list). Useful when the
            config is generated or templated in Python. Mutually exclusive with
            ``config_path``.
        """
        self.steps = []  # hierarchical step configs
        self.global_parameters = {}  # mirrors _parameters["pipeline"]
        self._context = None

        # initialise config mirror system
        self._init_config_mirror()

        if config_path is not None and config is not None:
            raise ValueError(
                "Provide either config_path or config, not both."
            )

        has_config = config_path is not None or config is not None
        if config_path is not None:
            self.load_config_from_file(config_path, mirror_keys=["pipeline"])
        elif config is not None:
            self.load_config(config, mirror_keys=["pipeline"])

        if has_config:
            # set convenience alias for user-facing access
            self.global_parameters = self._parameters.get("pipeline", {})

        # Always set up logging so users never have to configure it manually,
        # whether the pipeline is built from a config file or assembled in
        # memory. Config (and therefore log_file/out_directory) is loaded above
        # first, so it is honoured here.
        self.logger = _setup_logging(
            self.global_parameters.get("out_directory"),
            self.global_parameters.get("log_file"),
        )

        if has_config:
            # build steps from loaded config
            self.build_steps(self._parameters.get("steps", []))
            self.logger.info("Pipeline initialised")

    def build_steps(self, steps_config):
        """
        Build steps from configuration.

        Individual steps, including parameters and diagnostics, are saved to self.steps using add_step() for other functions.

        Parameters
        ----------
        steps_config : list of dict
            List of step configurations.
        """
        self.logger.info("Assembling steps to run from config.")
        for step in steps_config:
            self.add_step(
                step_name=step["name"],
                parameters=step.get("parameters", {}),
                diagnostics=step.get("diagnostics", False),
                run_immediately=False,
            )

    def add_step(
        self,
        step_name,
        parameters=None,
        diagnostics=False,
        run_immediately=False,
    ):
        """
        Dynamically adds a step and optionally runs it immediately.

        Parameters
        ----------
        step_name : str
            Name of the step to add.
        parameters : dict, optional
            Parameters for the step.
        diagnostics : bool, optional
            Whether to enable diagnostics for this step.
        run_immediately : bool, optional
            Whether to run the step immediately after adding it.

        Raises
        ------
        ValueError
            If the step name is not recognized.
        """
        # Step names are matched case-insensitively; store the canonical
        # (registered) name so downstream lookups and logging are consistent.
        canonical_name = resolve_step_name(step_name)
        if canonical_name is None:
            available_steps = list(STEP_CLASSES.keys())
            error_msg = (
                f"Step '{step_name}' is not recognised or missing @register_step."
            )

            # Look for a typo and suggest the closest match
            close_matches = difflib.get_close_matches(
                step_name, available_steps, n=1, cutoff=0.6
            )
            if close_matches:
                error_msg += f" Did you mean '{close_matches[0]}'?"
            else:
                # If no close match, show a few available options
                sample_steps = ", ".join(available_steps[:5])
                error_msg += f" Some available steps include: {sample_steps}..."

            self.logger.error(error_msg)
            raise ValueError(error_msg)
        step_name = canonical_name

        step_config = {
            "name": step_name,
            "parameters": parameters or {},
            "diagnostics": diagnostics,
        }

        self.steps.append(step_config)
        self.logger.info(f"Step '{step_name}' added successfully!")

        if run_immediately:
            self.logger.info(f"Running step '{step_name}' immediately.")
            self._context = self.execute_step(step_config, self._context)

    def execute_step(self, step_config, _context):
        """
        Executes a single step.

        Parameters
        ----------
        step_config : dict
            Configuration for the step to execute.
        _context : dict
            Current context to pass to the step.
        """
        step_context = _context.copy() if _context else {}
        step_context["global_parameters"] = self.global_parameters
        #   Expose the full run configuration (pipeline block + ordered steps)
        #   so steps such as the report writer can reproduce it. Rebuilt from
        #   the canonical step list rather than the raw file so it is available
        #   whether the pipeline was loaded from YAML or assembled in memory,
        #   and is free of comments/blank lines.
        step_context["pipeline_config"] = {
            "pipeline": dict(self.global_parameters),
            "steps": [
                {"name": s["name"], "parameters": s.get("parameters", {})}
                | ({"diagnostics": s["diagnostics"]} if s.get("diagnostics") else {})
                for s in self.steps
            ],
        }

        # When a report step is in the pipeline, expose the running list of
        # captured diagnostic figures so the report writer can embed them.
        capture = getattr(self, "_capture_diagnostics", False)
        if capture:
            step_context["captured_diagnostics"] = self._captured_figures

        # Position in the run, used to keep captured figure filenames unique
        # when a config repeats a step name (e.g. several 'Apply QC' steps).
        step_index = getattr(self, "_step_index", 0) + 1
        self._step_index = step_index

        step = create_step(step_config, step_context)
        self.logger.info(f"Executing: {step.name}")

        # The user's own diagnostics setting drives interactive display and
        # performance logging. Capture mode additionally force-enables the
        # diagnostic code path so its figures can be saved for the report,
        # without otherwise changing how the step reports performance.
        user_diagnostics = step.diagnostics
        captured_images = []
        if capture:
            step.diagnostics = True
            diagnostic_capture.make_diagnostics_safe(step)

        try:
            capture_ctx = (
                diagnostic_capture.capture_figures(
                    self._capture_dir,
                    step.name,
                    step_index,
                    captured_images,
                    # Diagnostics were force-enabled only to capture figures for
                    # the report; a step the user did not opt into must not dump
                    # its textual diagnostics (e.g. the dataset summary) to the
                    # console.
                    suppress_text=not user_diagnostics,
                    # Steps the user explicitly enabled diagnostics on still get
                    # their interactive blocking popup (in addition to being
                    # saved into the report); force-captured steps stay headless.
                    interactive=user_diagnostics,
                )
                if capture
                else contextlib.nullcontext()
            )

            if user_diagnostics:
                # Time the processing only. When a step generates diagnostics,
                # BaseStep stops the timer (report_performance) as plotting
                # begins, so a blocking plot left open isn't counted. This call
                # is the idempotent fallback for steps that produce no plot.
                step._diagnostics_start = time.time()
                step._diagnostics_reported = False
                with capture_ctx:
                    result = step.run()
                step.report_performance()
            else:
                with capture_ctx:
                    result = step.run()

            if captured_images:
                self._captured_figures.append(
                    {"step": step.name, "images": captured_images}
                )

            return result

        except Exception as e:
            self.logger.error(
                f"Fatal error encountered while executing step '{step.name}': {e}"
            )
            raise RuntimeError(f"Pipeline failed at step '{step.name}': {e}") from e

    def run_last_step(self):
        """
        Runs only the most recently added step based on the index in self.steps.
        """
        if not self.steps:
            self.logger.info("No steps to run.")
            return
        last_step = self.steps[-1]
        self.logger.info(f"Running last step: {last_step['name']}")
        self._context = self.execute_step(last_step, self._context)

    def run(self):
        """
        Runs the entire pipeline.
        """
        try:
            check_pipeline_variables(self.steps, self.logger)
        except ValueError:
            self.logger.log(
                STOP,
                "Pipeline stopped before execution. "
                "Resolve the validation error above and re-run.",
            )
            raise SystemExit(1) from None

        # If the pipeline writes a report, capture each step's diagnostic plots
        # in the background (regardless of that step's own diagnostics setting)
        # so they can be embedded in the report. This exercises every step's
        # diagnostic code, which is why it can slow the run down.
        report_present = any(s["name"] == REPORT_STEP_NAME for s in self.steps)
        self._step_index = 0
        if report_present:
            self._capture_diagnostics = True
            self._captured_figures = []
            self._capture_dir = tempfile.mkdtemp(prefix="pelagos_report_diag_")
            self.logger.warning(
                "A report step is enabled: diagnostic plots will be generated in "
                "the background for every step that produces one, so the pipeline "
                "may run more slowly than usual."
            )

        # When capturing for the report, force the headless Agg backend once for
        # the whole run (see force_headless_backend). Toggling the backend per
        # step can hard-crash on Windows, so it stays on Agg start to finish.
        backend_ctx = (
            diagnostic_capture.force_headless_backend()
            if report_present
            else contextlib.nullcontext()
        )
        try:
            with backend_ctx:
                for step in self.steps:
                    self._context = self.execute_step(step, self._context)
        finally:
            if report_present:
                #   Figures have been embedded by the report writer by now.
                shutil.rmtree(self._capture_dir, ignore_errors=True)
                self._capture_diagnostics = False

    def generate_config(self):
        """
        Generate a configuration dictionary from the current pipeline setup.

        returns
        -------
        dict
            Configuration dictionary of the current pipeline.
        """
        cfg = {
            "pipeline": self.global_parameters,
            "steps": self.steps,
        }
        # Keep private config in sync
        self._parameters.update(cfg)
        return cfg

    def export_config(self, output_path="generated_pipeline.yaml"):
        """
        Write current config to file (respects private _parameters)

        parameters
        ----------
        output_path : str
            Path to save the exported configuration YAML file.

        returns
        -------
        dict
            Configuration dictionary of the current pipeline.
        """
        cfg = self.generate_config()
        with open(output_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self.logger.info(f"Pipeline config exported → {output_path}")
        return cfg

    def save_config(self, path="pipeline_config.yaml"):
        """
        Save the canonical private config (same as manager.save_config).

        parameters
        ----------
        path : str
            Path to save the exported configuration YAML file.
        """
        # ensure _parameters is up to date
        self._parameters.update(self.generate_config())
        super().save_config(path)

    def get_data(self):
        """
        Returns data from the current pipeline context.
        """
        if self._context and "data" in self._context:
            return self._context["data"]
        return None
