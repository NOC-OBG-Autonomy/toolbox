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

"""
pelagos_py.steps
~~~~~~~~~~~~~~~~~~
This module contains the logic to dynamically discover and register step implementations,
and to instantiate them in a config-aware way.
"""

import os
import importlib
import pathlib
import time
import yaml
from pelagos_py.utils.yaml_loading import safe_load as yaml_safe_load
import logging
from pelagos_py.steps.base_step import REGISTERED_STEPS
from pelagos_py.steps.base_qc import REGISTERED_QC

# Setup logger for discovery. A console handler is attached here (rather than
# left to Pipeline._setup_logging) because discover_steps() runs at import
# time, before a Pipeline object exists to configure logging -- without this,
# the import-time module scan (which can take several seconds due to heavy
# step dependencies) is completely silent.
logger = logging.getLogger("pelagos_py.pipeline.discovery")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.propagate = False  # Avoid duplicate lines if the app configures root logging
    # Same console handler the pipeline uses; the module scan shows a progress bar.
    from pelagos_py.utils.console import make_console_handler

    logger.addHandler(make_console_handler())

# Global registries
STEP_CLASSES = {}
"""Dictionary mapping step names to their implementing classes."""
QC_CLASSES = {}
"""Dictionary mapping QC test names to their implementing classes."""


def discover_steps():
    """
    Dynamically discover and import step modules from the steps directory.
    This populates the global STEP_CLASSES and QC_CLASSES registries for use elsewhere.
    """
    from pelagos_py.utils.console import progress_bar

    base_dir = pathlib.Path(__file__).parent.resolve()
    logger.info("Scanning for step modules in %s", base_dir, extra={"console": False})

    discovery_start = time.time()
    failed_modules = []
    # Pre-collect module files so the scan can show a determinate progress bar.
    module_files = [
        py_file
        for py_file in base_dir.rglob("*.py")
        if py_file.name != "__init__.py"
    ]
    for py_file in progress_bar(module_files, desc="Discovering steps", unit="mod"):
        # Convert file path to module path
        relative_path = py_file.resolve().relative_to(base_dir)
        module_name = ".".join(
            ("pelagos_py.steps",) + relative_path.with_suffix("").parts
        )

        module_start = time.time()
        try:
            importlib.import_module(module_name)
        except Exception as e:
            logger.error("Failed to import %s: %s", module_name, e)
            failed_modules.append(module_name)
            continue
        elapsed = time.time() - module_start
        logger.info(
            "Imported step module: %s (%.2fs)",
            module_name,
            elapsed,
            extra={"console": False},
        )

    logger.info(
        "Finished importing step modules in %.2fs",
        time.time() - discovery_start,
        extra={"console": False},
    )

    # Populate global step class map. Importing the modules above is what's
    # slow (it runs each module's top-level code, including its dependency
    # imports) and is what actually registers each class (via @register_step,
    # which fires at class-definition time); this is just copying the
    # already-registered classes into the public dicts.
    STEP_CLASSES.update(REGISTERED_STEPS)
    QC_CLASSES.update(REGISTERED_QC)
    if failed_modules:
        logger.warning(
            "Registered %d steps and %d QC tests; %d module(s) failed to import: %s",
            len(STEP_CLASSES),
            len(QC_CLASSES),
            len(failed_modules),
            ", ".join(failed_modules),
        )
    else:
        logger.info(
            "Registered %d steps and %d QC tests successfully",
            len(STEP_CLASSES),
            len(QC_CLASSES),
        )


# Auto-discover steps when pelagos_py.steps is imported
discover_steps()


def resolve_step_name(step_name):
    """Resolve a step name to its canonical STEP_CLASSES key, case-insensitively (None if no match)."""
    if step_name in STEP_CLASSES:
        return step_name
    if not isinstance(step_name, str):
        return None
    lowered = {name.lower(): name for name in STEP_CLASSES}
    return lowered.get(step_name.lower())


def create_step(step_config, context=None):
    """
    Factory to create a Step instance from a dictionary or YAML file.

    Parameters
    ----------
    step_config : dict | str
        - If dict: must contain at least {'name': <step_name>}.
        - If str: path to a YAML file describing the step.
    context : dict | None
        Shared context passed through the pipeline.

    Returns
    -------
    BaseStep
        An instantiated, config-aware step.
    """
    # --- If user passed a YAML path instead of a dict ---
    if isinstance(step_config, str):
        if not os.path.exists(step_config):
            raise FileNotFoundError(f"Step config file not found: {step_config}")
        with open(step_config, "r") as f:
            step_config = yaml_safe_load(f) or {}
        if "name" not in step_config:
            raise ValueError(f"Invalid step YAML: missing 'name' key -> {step_config}")

    # --- Validate and resolve step class ---
    step_name = step_config.get("name")
    if not step_name:
        raise ValueError("Step config missing required 'name' field.")

    canonical_name = resolve_step_name(step_name)
    step_class = STEP_CLASSES.get(canonical_name) if canonical_name else None
    if not step_class:
        raise ValueError(
            f"Step '{step_name}' not recognized or missing @register_step. "
            f"Available: {list(STEP_CLASSES.keys())}"
        )
    # Use the canonical name so downstream naming ignores the user's casing.
    step_name = canonical_name

    # --- Instantiate the step ---
    parameters = step_config.get("parameters", {}) or {}
    diagnostics = bool(step_config.get("diagnostics", False))
    step = step_class(
        name=step_name,
        parameters=parameters,
        diagnostics=diagnostics,
        context=context,
    )

    # --- Synchronize config-mirroring store (if step supports it) ---
    # BaseStep inherits ConfigMirrorMixin
    if hasattr(step, "_parameters") and hasattr(step, "_reset_parameter_bridge"):
        step._parameters.update(step_config)
        # ensure mirrored keys are recognized
        step._reset_parameter_bridge(mirror_keys=["parameters", "diagnostics"])

    return step
