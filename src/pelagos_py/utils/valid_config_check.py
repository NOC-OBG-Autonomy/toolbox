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

import json
import os
import subprocess
import sys
from pathlib import Path

from pelagos_py.steps import STEP_CLASSES, QC_CLASSES
from pelagos_py.utils import parameter_spec

#: Steps that supply the pipeline's base data -- the only steps allowed to
#: provide TIME/LATITUDE/etc from nothing. Referenced by name rather than by
#: introspecting provided_variables, since that's also true of some QC-output
#: sets; these two are singled out because a pipeline needs exactly one.
LOADER_STEP_NAMES = ("Load OG1", "Generate Data")

#: Variables only a loader step (see LOADER_STEP_NAMES) can produce -- used to
#: tell "you're missing a loader step" apart from "you're missing some other
#: step" when reporting a missing-variable error.
LOADER_PROVIDED_VARIABLES = {"TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "CNDC"}

#: Shown whenever a missing variable can only come from a loader step, so the
#: fix is spelled out rather than left as a bare variable name.
_NO_LOADER_HINT = (
    "No data-loading step ('Load OG1' or 'Generate Data') provides it -- add "
    "one, normally as the first step in the pipeline."
)


def _loader_hint(missing):
    """``_NO_LOADER_HINT`` if any of `missing` can only come from a loader step."""
    return _NO_LOADER_HINT if any(v in LOADER_PROVIDED_VARIABLES for v in missing) else None


def _variable_parameter_names(step_class, parameters):
    """Resolve a step's config-driven variable-name parameters to values.

    Some steps take the *name* of the variable to work on as a parameter
    (e.g. ``apply_to: "BBP700"``) rather than a fixed name in
    ``required_variables`` -- ``variable_parameters`` (also used by
    ``QCHandlingMixin`` for data-subsetting) lists which parameters those are.
    Resolved the same way (parameter value, defaulting from the schema) so
    they're checked for availability like any other required variable.
    ``output_as`` names an output, not an input, so is always excluded; a step
    lists any other parameter that's only conditionally required (e.g. one
    tied to a particular ``method``, checked by the step itself at run time)
    in ``variable_parameters_optional`` to exclude it here too.
    """
    schema = getattr(step_class, "parameter_schema", None) or {}
    optional = getattr(step_class, "variable_parameters_optional", ())
    names = []
    for attr in getattr(step_class, "variable_parameters", []):
        if attr == "output_as" or attr in optional:
            continue
        default = (schema.get(attr) or {}).get("default")
        value = parameters.get(attr, default)
        if value is None:
            continue
        names.extend(value if isinstance(value, (list, tuple)) else [value])
    return names


def _resolve_output_as(step_class, parameters):
    """A step's effective ``output_as`` value: the configured value, or the
    schema default if the config leaves it unset (e.g. "BBP from Beta" run
    with no parameters relies on ``output_as`` defaulting to "BBP700").
    Returns a list of names (possibly empty).
    """
    schema = getattr(step_class, "parameter_schema", None) or {}
    out = parameters.get("output_as", (schema.get("output_as") or {}).get("default"))
    if not out:
        return []
    return list(out) if isinstance(out, (list, tuple)) else [out]


def _deep_correction_output(step_class, parameters):
    """"Deep Correction" always writes ``{apply_to}_ADJUSTED`` (see
    ``resolve_variables`` in deep_correction.py) -- there is no ``output_as``
    parameter to read, so mirror that naming here.
    """
    schema = getattr(step_class, "parameter_schema", None) or {}
    apply_to = parameters.get("apply_to", (schema.get("apply_to") or {}).get("default"))
    if not apply_to:
        return []
    return [apply_to if apply_to.endswith("_ADJUSTED") else f"{apply_to}_ADJUSTED"]


#: Opens `file_path` and prints its variable names, plus the subset of
#: `candidates` (argv[2], a JSON list) that are floating-point and entirely
#: NaN (e.g. a placeholder field a data centre ships with no real values), as
#: JSON. Restricted to `candidates` rather than every variable in the file
#: because loading a variable's full data to check it is far more expensive
#: than listing names, and a raw OG1 file typically has many variables no
#: step in a given pipeline ever reads. Run in a subprocess (see
#: _read_file_variables) rather than in-process: certain netCDF4/HDF5 + h5py
#: combinations segfault the whole interpreter on open in some environments,
#: and a segfault can't be caught with try/except -- it takes the caller down
#: with it. Isolating it in a short-lived subprocess means a crash there is
#: just a failed subprocess, handled like any other unreadable file.
_READ_VARIABLES_SCRIPT = (
    "import sys, json\n"
    "import numpy as np\n"
    "import xarray as xr\n"
    "candidates = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None\n"
    "with xr.open_dataset(sys.argv[1]) as ds:\n"
    "    names = list(ds.variables)\n"
    "    to_check = names if candidates is None else [v for v in candidates if v in names]\n"
    "    all_nan = [\n"
    "        v for v in to_check\n"
    "        if np.issubdtype(ds[v].dtype, np.floating) and bool(np.isnan(ds[v].values).all())\n"
    "    ]\n"
    "    print(json.dumps({'variables': names, 'all_nan': all_nan}))\n"
)

#: Single-entry (path, mtime, candidates) -> variable-names cache. The
#: dashboard's /api/validate fires on every keystroke with the file_path
#: usually unchanged, so this avoids re-spawning a Python interpreter just to
#: open the same file again while the user edits an unrelated part of the config.
_file_vars_cache = {}


def _read_file_variables(file_path, logger, candidate_vars=None, timeout=20):
    """Variable names actually present in ``file_path``, and the subset of
    ``candidate_vars`` (or of all variables, if ``None``) that are
    floating-point and entirely NaN (e.g. a placeholder field shipped with no
    real data), as ``(names, all_nan)``. Both are ``None`` if the file can't
    be read (missing, wrong format, timed out, etc).

    ``candidate_vars`` should be every variable the pipeline actually reads --
    checking only those keeps the (relatively expensive, since it loads real
    data) all-NaN check fast on files with many unused variables.

    Callers fall back to the usual "assume file-native" behaviour on
    ``None`` rather than blocking validation on a filesystem/format problem
    the run-time load will report anyway.
    """
    try:
        cache_key = (
            str(file_path), Path(file_path).stat().st_mtime,
            frozenset(candidate_vars) if candidate_vars is not None else None,
        )
    except OSError as exc:
        logger.info("Could not read '%s' to cross-check its variables: %s", file_path, exc)
        return None, None
    if cache_key in _file_vars_cache:
        return _file_vars_cache[cache_key]

    try:
        # HDF5 file locking can transiently fail (Errno -101) if the load
        # step reopens the file right after this subprocess closes it.
        env = dict(os.environ, HDF5_USE_FILE_LOCKING="FALSE")
        cmd = [sys.executable, "-c", _READ_VARIABLES_SCRIPT, str(file_path)]
        if candidate_vars is not None:
            cmd.append(json.dumps(sorted(candidate_vars)))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        if result.returncode == 0:
            payload = json.loads(result.stdout)
            file_vars, all_nan = set(payload["variables"]), set(payload["all_nan"])
        else:
            file_vars, all_nan = None, None
        if file_vars is None:
            logger.info(
                "Could not read '%s' to cross-check its variables: %s",
                file_path, (result.stderr or "").strip().splitlines()[-1:] or "unknown error",
            )
    except Exception as exc:
        logger.info("Could not read '%s' to cross-check its variables: %s", file_path, exc)
        file_vars, all_nan = None, None

    _file_vars_cache.clear()  # only the most recently checked file is worth keeping
    _file_vars_cache[cache_key] = (file_vars, all_nan)
    return file_vars, all_nan


def _raise_missing_variables(
    logger, kind, label, missing, pipeline_provided, known_derived, file_vars, file_all_nan=None
):
    """Categorize ``missing`` variables required by ``label`` (a step or QC
    test, per ``kind``) and raise the most specific error that applies:

    1. produced by some step, but later in the pipeline -- reorder.
    2. produced by no step here, but some registered step is known to derive
       it -- add that step.
    3. produced by no step and not derivable, but the actual input file
       (opened for ``file_vars``, once ``Load OG1`` has a real path) doesn't
       contain it either.
    4. as (3), but the variable is present in the file with only
       placeholder/all-NaN data (``file_all_nan``).

    Does nothing if none of these apply: the variable is then assumed to be
    file-native and left for the run-time check, same as when ``file_vars``
    is ``None`` (no file to check against yet).
    """
    out_of_order = [v for v in missing if v in pipeline_provided]
    if out_of_order:
        missing_str = ", ".join(out_of_order)
        logger.error(
            "Validation Failed: %s '%s' requires %s, but it is produced by a "
            "later step. Reorder the pipeline so the producing step runs first.",
            kind, label, missing_str,
        )
        raise ValueError(
            f"Missing variables for {kind} '{label}': {missing_str}. These "
            f"are produced later in the pipeline — reorder the steps so they "
            f"run beforehand."
        )

    not_produced = [v for v in missing if v not in pipeline_provided and v in known_derived]
    if not_produced:
        missing_str = ", ".join(not_produced)
        hint = _loader_hint(not_produced) or (
            "Add the step that derives it (e.g. 'Find Profiles' for "
            "PROFILE_NUMBER/PROFILE_DIRECTION)."
        )
        logger.error(
            "Validation Failed: %s '%s' requires %s, but no step in the "
            "pipeline produces it. %s",
            kind, label, missing_str, hint,
        )
        raise ValueError(
            f"Missing variables for {kind} '{label}': {missing_str}. No step "
            f"in the pipeline produces them. {hint}"
        )

    if file_vars is not None:
        unverified = [v for v in missing if v not in pipeline_provided and v not in known_derived]
        missing_from_file = [v for v in unverified if v not in file_vars]
        if missing_from_file:
            missing_str = ", ".join(missing_from_file)
            logger.error(
                "Validation Failed: %s '%s' requires %s, but the input file "
                "does not contain it.",
                kind, label, missing_str,
            )
            raise ValueError(
                f"Missing variables for {kind} '{label}': {missing_str}. The "
                f"input file does not contain them."
            )

        if file_all_nan:
            all_nan_present = [v for v in unverified if v in file_all_nan]
            if all_nan_present:
                missing_str = ", ".join(all_nan_present)
                logger.error(
                    "Validation Failed: %s '%s' requires %s, but the input "
                    "file only contains placeholder (all-NaN) data for it.",
                    kind, label, missing_str,
                )
                raise ValueError(
                    f"Variable(s) required for {kind} '{label}': {missing_str} "
                    f"are present in the input file but contain only NaN "
                    f"(placeholder) data."
                )


def _missing_required_params(schema, parameters):
    """Names of required schema parameters absent from the supplied config.

    ``schema`` of ``None`` (a component not yet on the parameter schema, e.g. the
    oxygen steps) is treated as "no required parameters".
    """
    if not schema:
        return []
    return [
        name
        for name, spec in schema.items()
        if parameter_spec.is_required(spec) and name not in parameters
    ]


def _unknown_params(schema, parameters, allowed_extra=()):
    """Names of supplied parameters not declared in the schema.

    Mirrors the reject-unknown behaviour of :func:`parameter_spec.resolve`, but
    runs up front so config typos are caught before any step executes. ``schema``
    of ``None`` (a component not yet on the parameter schema) skips the check; an
    empty ``{}`` schema is strict, so any supplied parameter is unknown.
    ``allowed_extra`` permits framework keys (e.g. ``qc_handling_settings``).
    """
    if schema is None:
        return []
    return [
        name
        for name in parameters
        if name not in schema and name not in allowed_extra
    ]


def _qc_test_io(qc_class, qc_params):
    """Resolve a QC test's required and provided variables from its parameters.

    Mirrors how Apply QC resolves them at run time: dynamic tests derive their
    variables from the supplied parameters (so they are instantiated with no data
    to introspect), while static tests expose them as class attributes.
    """
    if getattr(qc_class, "dynamic", False):
        # `diagnostics` is a reserved per-test flag, not a QC parameter.
        params = {k: v for k, v in (qc_params or {}).items() if k != "diagnostics"}
        probe = qc_class(None, **params)
        return list(probe.required_variables), list(probe.qc_outputs)
    return (
        list(getattr(qc_class, "required_variables", [])),
        list(getattr(qc_class, "qc_outputs", [])),
    )


def _pipeline_provided_variables(steps_list):
    """All variables any step in the pipeline produces.

    Used to tell an ordering mistake (a required variable that *is* produced, but
    by a later step) apart from a variable that is simply unknown to the schema
    because it comes straight from the input data file. Only the former is worth
    reporting up front, so QC tests that legitimately depend on file-native
    variables (e.g. DOWNWELLING_PAR) are not flagged.
    """
    provided = set()
    for step_config in steps_list:
        step_class = STEP_CLASSES.get(step_config["name"])
        if not step_class:
            continue
        parameters = step_config.get("parameters", {}) or {}
        provided.update(getattr(step_class, "provided_variables", []))
        provided.update(getattr(step_class, "qc_outputs", []))
        provided.update(parameters.get("to_derive", []))
        provided.update(parameters.get("qc_outputs", []))
        provided.update(_resolve_output_as(step_class, parameters))
        if step_config["name"] == "Deep Correction":
            provided.update(_deep_correction_output(step_class, parameters))
        if step_config["name"] == "Apply QC":
            for qc_name, qc_params in (parameters.get("qc_settings") or {}).items():
                qc_class = QC_CLASSES.get(qc_name)
                if qc_class is None:
                    continue
                try:
                    _, outputs = _qc_test_io(qc_class, qc_params)
                except Exception:
                    # Malformed parameters are reported by the per-step validation
                    # below; here we only gather outputs, so skip what we can't resolve.
                    continue
                provided.update(outputs)
    return provided


def _non_loader_provided_variables(steps_list):
    """Like :func:`_pipeline_provided_variables`, but ignores loader steps'
    own static ``provided_variables`` (TIME/LATITUDE/etc, claimed
    unconditionally regardless of the real file's contents).

    Used to check whether a base variable missing from the raw file is fixed
    by some *other* step -- e.g. a "Correct Values" step renaming
    ``LATITUDE_GPS`` to ``LATITUDE`` -- rather than just trusting the
    loader's own claim to provide it.
    """
    others = [
        s for s in steps_list
        if not (isinstance(s, dict) and s.get("name") in LOADER_STEP_NAMES)
    ]
    return _pipeline_provided_variables(others)


def _known_derived_variables():
    """Every variable name any registered step/QC test can produce.

    Unlike :func:`_pipeline_provided_variables` this is not limited to steps
    actually in the config — it is the full registry, so it distinguishes a
    variable that is *always* derived by some step (e.g. PROFILE_NUMBER, only
    ever produced by "Find Profiles") from one that could legitimately come
    straight from the input file (e.g. DOWNWELLING_PAR). Only the former is
    worth flagging when no step in *this* pipeline produces it.
    """
    known = set()
    for step_class in STEP_CLASSES.values():
        known.update(getattr(step_class, "provided_variables", []))
        known.update(getattr(step_class, "qc_outputs", []))
    for qc_class in QC_CLASSES.values():
        known.update(getattr(qc_class, "qc_outputs", []))
    return known


def _all_required_variables(steps_list):
    """Every variable name any step or QC test in `steps_list` might read.

    Used to limit the all-NaN check in :func:`_read_file_variables` to
    variables the pipeline actually consumes, rather than every variable the
    input file happens to contain (checking each one means loading its real
    data, which is comparatively slow on a file with many unused variables).
    """
    required = set(LOADER_PROVIDED_VARIABLES)
    for step_config in steps_list:
        if not isinstance(step_config, dict):
            continue
        step_class = STEP_CLASSES.get(step_config.get("name"))
        if not step_class:
            continue
        parameters = step_config.get("parameters", {}) or {}
        required.update(getattr(step_class, "required_variables", []))
        required.update(_variable_parameter_names(step_class, parameters))
        if step_config.get("name") == "Apply QC":
            for qc_name, qc_params in (parameters.get("qc_settings") or {}).items():
                qc_class = QC_CLASSES.get(qc_name)
                if qc_class is None:
                    continue
                qc_params = {k: v for k, v in (qc_params or {}).items() if k != "diagnostics"}
                try:
                    qc_required, _ = _qc_test_io(qc_class, qc_params)
                except Exception:
                    continue
                required.update(qc_required)
    return required


def check_pipeline_variables(steps_list, logger, available_vars=None):
    file_vars = None
    file_all_nan = None
    if available_vars is None:
        logger.info("Checking pipeline variable requirements...")
        # No variable is available before some step actually loads/generates
        # data -- "Load OG1" and "Generate Data" both declare TIME, LATITUDE,
        # LONGITUDE, PRES, TEMP and CNDC as their own provided_variables (added
        # to available_vars below once the loop reaches them), so a pipeline
        # missing both is correctly flagged rather than assumed to have data.
        available_vars = set()

        loader_steps = [
            (i, s["name"]) for i, s in enumerate(steps_list)
            if isinstance(s, dict) and s.get("name") in LOADER_STEP_NAMES
        ]
        if len(loader_steps) > 1:
            where = ", ".join(f"step {i + 1} ('{n}')" for i, n in loader_steps)
            logger.error(
                "Validation Failed: multiple data-loading steps found: %s. "
                "Only one step should load or generate the pipeline's base "
                "data -- remove the extra one.",
                where,
            )
            raise ValueError(
                f"Multiple data-loading steps found: {where}. Keep only one "
                f"'Load OG1' or 'Generate Data' step."
            )
        if len(loader_steps) == 1 and loader_steps[0][1] == "Load OG1":
            idx, _ = loader_steps[0]
            file_path = (steps_list[idx].get("parameters") or {}).get("file_path")
            if not file_path or not str(file_path).strip():
                logger.error(
                    "Validation Failed: 'Load OG1' has no 'file_path' set -- "
                    "this config does not include a data file. Set "
                    "'file_path' to your input NetCDF file before running."
                )
                exc = ValueError(
                    "'Load OG1' has no 'file_path' set -- this config does "
                    "not include a data file. Set 'file_path' to your input "
                    "NetCDF file before running."
                )
                exc.step_index = idx
                raise exc
            else:
                # A secondary layer of validation, on top of the registry-driven
                # checks below: once there's a real path, open it and check
                # against what it actually contains (including whether required
                # variables hold only placeholder/all-NaN data), rather than
                # only what the pipeline's steps declare.
                file_vars, file_all_nan = _read_file_variables(
                    file_path, logger, _all_required_variables(steps_list)
                )
                if file_vars is not None:
                    # A later step (e.g. "Correct Values" renaming
                    # LATITUDE_GPS -> LATITUDE) can legitimately supply a base
                    # variable the raw file stores under a different name, so
                    # only flag it if no *other* step provides it either.
                    other_provided = _non_loader_provided_variables(steps_list)
                    missing_base = sorted(
                        v for v in LOADER_PROVIDED_VARIABLES
                        if v not in file_vars and v not in other_provided
                    )
                    if missing_base:
                        missing_str = ", ".join(missing_base)
                        logger.error(
                            "Validation Failed: 'Load OG1' file '%s' does not "
                            "contain %s, which every OG1-format file is "
                            "expected to provide.",
                            file_path,
                            missing_str,
                        )
                        exc = ValueError(
                            f"'Load OG1' file '{file_path}' does not contain "
                            f"{missing_str}, which every OG1-format file is "
                            f"expected to provide."
                        )
                        exc.step_index = idx
                        raise exc

    pipeline_provided = _pipeline_provided_variables(steps_list)
    known_derived = _known_derived_variables()

    for index, step_config in enumerate(steps_list):
        try:
            step_name = step_config["name"]

            step_class = STEP_CLASSES.get(step_name)
            if not step_class:
                continue

            parameters = step_config.get("parameters", {}) or {}
            schema = getattr(step_class, "parameter_schema", None)
            allowed_extra = getattr(step_class, "framework_parameters", set())

            # Check for missing required parameters, driven by the declared schema.
            missing_params = _missing_required_params(schema, parameters)
            if missing_params:
                missing_str = ", ".join(missing_params)
                logger.error(
                    "Validation Failed: '%s' is missing required config parameters: %s.",
                    step_name,
                    missing_str,
                )
                raise ValueError(
                    f"Missing config parameters for '{step_name}': {missing_str}."
                )

            # Check for unknown parameters (config typos), driven by the same schema.
            unknown_params = _unknown_params(schema, parameters, allowed_extra)
            if unknown_params:
                unknown_str = ", ".join(unknown_params)
                valid_str = ", ".join(sorted(schema)) or "(none)"
                logger.error(
                    "Validation Failed: '%s' has unknown config parameters: %s. "
                    "Valid parameters: %s.",
                    step_name,
                    unknown_str,
                    valid_str,
                )
                raise ValueError(
                    f"Unknown config parameters for '{step_name}': {unknown_str}. "
                    f"Valid parameters: {valid_str}."
                )

            # Check for type mismatches (e.g. a bool where a float is expected).
            if schema is not None:
                bad_types = parameter_spec.type_errors(schema, parameters)
                if bad_types:
                    bad_str = "; ".join(bad_types)
                    logger.error(
                        "Validation Failed: '%s' has invalid parameter type(s): %s.",
                        step_name,
                        bad_str,
                    )
                    raise ValueError(
                        f"Invalid parameter type(s) for '{step_name}': {bad_str}."
                    )

            # Check for out-of-options values (e.g. an unknown 'method' choice).
            if schema is not None:
                bad_options = parameter_spec.option_errors(schema, parameters)
                if bad_options:
                    bad_str = "; ".join(bad_options)
                    logger.error(
                        "Validation Failed: '%s' has invalid parameter value(s): %s.",
                        step_name,
                        bad_str,
                    )
                    raise ValueError(
                        f"Invalid parameter value(s) for '{step_name}': {bad_str}."
                    )

            # Apply QC nests each test's settings under qc_settings — validate the
            # required parameters of every requested test up front. (Their variable
            # requirements are checked by Apply QC at run time, where _QC columns and
            # also_flag propagation are resolved.)
            if step_name == "Apply QC":
                for qc_name, qc_params in (parameters.get("qc_settings") or {}).items():
                    # `diagnostics` is a reserved per-test flag handled by Apply QC,
                    # not a QC test parameter — exclude it before validating.
                    qc_params = {
                        k: v for k, v in (qc_params or {}).items() if k != "diagnostics"
                    }
                    qc_class = QC_CLASSES.get(qc_name)
                    if qc_class is None:
                        continue  # Apply QC raises a clear error for unknown tests at run time
                    qc_schema = getattr(qc_class, "parameter_schema", None)
                    qc_allowed_extra = getattr(qc_class, "framework_parameters", set())
                    qc_missing = _missing_required_params(qc_schema, qc_params or {})
                    if qc_missing:
                        missing_str = ", ".join(qc_missing)
                        logger.error(
                            "Validation Failed: QC test '%s' is missing required parameters: %s.",
                            qc_name,
                            missing_str,
                        )
                        raise ValueError(
                            f"Missing config parameters for QC test '{qc_name}': {missing_str}."
                        )

                    qc_unknown = _unknown_params(qc_schema, qc_params or {}, qc_allowed_extra)
                    if qc_unknown:
                        unknown_str = ", ".join(qc_unknown)
                        valid_str = ", ".join(sorted(qc_schema)) or "(none)"
                        logger.error(
                            "Validation Failed: QC test '%s' has unknown parameters: %s. "
                            "Valid parameters: %s.",
                            qc_name,
                            unknown_str,
                            valid_str,
                        )
                        raise ValueError(
                            f"Unknown config parameters for QC test '{qc_name}': {unknown_str}. "
                            f"Valid parameters: {valid_str}."
                        )

                    if qc_schema is not None:
                        qc_bad_types = parameter_spec.type_errors(qc_schema, qc_params or {})
                        if qc_bad_types:
                            bad_str = "; ".join(qc_bad_types)
                            logger.error(
                                "Validation Failed: QC test '%s' has invalid parameter type(s): %s.",
                                qc_name,
                                bad_str,
                            )
                            raise ValueError(
                                f"Invalid parameter type(s) for QC test '{qc_name}': {bad_str}."
                            )

                    if qc_schema is not None:
                        qc_bad_options = parameter_spec.option_errors(qc_schema, qc_params or {})
                        if qc_bad_options:
                            bad_str = "; ".join(qc_bad_options)
                            logger.error(
                                "Validation Failed: QC test '%s' has invalid parameter value(s): %s.",
                                qc_name,
                                bad_str,
                            )
                            raise ValueError(
                                f"Invalid parameter value(s) for QC test '{qc_name}': {bad_str}."
                            )

                    # Resolve this test's variable requirements the same way Apply QC
                    # does at run time, then categorize any that are missing now (see
                    # _raise_missing_variables): produced later (reorder), produced by
                    # no step at all (add the step), or -- once Load OG1's file has
                    # been opened -- genuinely absent from the input file too.
                    # Otherwise assumed file-native and left for the run-time check.
                    qc_required, qc_outputs = _qc_test_io(qc_class, qc_params)
                    qc_missing = [v for v in qc_required if v not in available_vars]
                    if qc_missing:
                        _raise_missing_variables(
                            logger, "QC test", qc_name, qc_missing,
                            pipeline_provided, known_derived, file_vars, file_all_nan,
                        )

                    # Make this test's outputs available to later tests in the same
                    # Apply QC call, so a test that legitimately depends on an earlier
                    # test's output (e.g. a profile-level test needing TEMP_QC) is not
                    # falsely flagged as depending on a later step.
                    available_vars.update(qc_outputs)

            req_vars = list(getattr(step_class, "required_variables", []))
            # Config-driven variable-name parameters (e.g. `apply_to: "BBP700"`)
            # count as required too -- resolved the same way QCHandlingMixin reads
            # them at run time (see _variable_parameter_names).
            req_vars.extend(_variable_parameter_names(step_class, parameters))

            own_provided = set(getattr(step_class, "provided_variables", []))
            own_provided.update(getattr(step_class, "qc_outputs", []))
            own_provided.update(parameters.get("to_derive") or [])
            own_provided.update(parameters.get("qc_outputs") or [])
            own_provided.update(_resolve_output_as(step_class, parameters))
            if step_name == "Deep Correction":
                own_provided.update(_deep_correction_output(step_class, parameters))

            missing_vars = [req for req in req_vars if req not in available_vars]

            if missing_vars:
                # Exclude this step's own outputs from pipeline_provided: several
                # steps read and overwrite the same variable (apply_to == output_as,
                # e.g. BBP700 by default in "BBP from Beta"), which would otherwise
                # look like the variable is "produced later" by this very step.
                _raise_missing_variables(
                    logger, "step", step_name, missing_vars,
                    pipeline_provided - own_provided, known_derived, file_vars, file_all_nan,
                )

            available_vars.update(own_provided)

        except ValueError as exc:
            if not hasattr(exc, "step_index"):
                exc.step_index = index
            raise

    if steps_list:
        logger.info("Pipeline variable check successful.")

    return True
