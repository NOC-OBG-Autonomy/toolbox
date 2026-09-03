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

from pelagos_py.steps import STEP_CLASSES, QC_CLASSES
from pelagos_py.utils import parameter_spec

STANDARD_VARIABLES = {"TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "CNDC"}


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
        probe = qc_class(None, **(qc_params or {}))
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
        out = parameters.get("output_as")
        if out:
            provided.update(out if isinstance(out, (list, tuple)) else [out])
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


def check_pipeline_variables(steps_list, logger, available_vars=None):
    if available_vars is None:
        logger.info("Checking pipeline variable requirements...")
        available_vars = set(STANDARD_VARIABLES)

    pipeline_provided = _pipeline_provided_variables(steps_list)

    for step_config in steps_list:
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
                # does at run time. A required variable is only reported when it is
                # missing now *and* produced by some step in the pipeline: that means
                # it exists but is ordered after this Apply QC (e.g. PROFILE_NUMBER
                # required before "Find Profiles" runs). A variable absent from the
                # pipeline's outputs is assumed to come from the input data file and
                # is left for the run-time check, so file-native variables (e.g.
                # DOWNWELLING_PAR) are not falsely flagged.
                qc_required, qc_outputs = _qc_test_io(qc_class, qc_params)
                out_of_order = [
                    v
                    for v in qc_required
                    if v not in available_vars and v in pipeline_provided
                ]
                if out_of_order:
                    missing_str = ", ".join(out_of_order)
                    logger.error(
                        "Validation Failed: QC test '%s' requires %s, but it is "
                        "produced by a later step. Reorder the pipeline so the "
                        "producing step runs first.",
                        qc_name,
                        missing_str,
                    )
                    raise ValueError(
                        f"Missing variables for QC test '{qc_name}': {missing_str}. "
                        f"These are produced later in the pipeline — reorder the "
                        f"steps so they run beforehand."
                    )

                # Make this test's outputs available to later tests in the same
                # Apply QC call, so a test that legitimately depends on an earlier
                # test's output (e.g. a profile-level test needing TEMP_QC) is not
                # falsely flagged as depending on a later step.
                available_vars.update(qc_outputs)

        req_vars = list(getattr(step_class, "required_variables", []))
        provided_vars = getattr(step_class, "provided_variables", []) + getattr(
            step_class, "qc_outputs", []
        )

        if step_name == "Find Profiles":
            depth_default = step_class.parameter_schema["depth_column"]["default"]
            depth_col = parameters.get("depth_column", depth_default)
            if depth_col not in req_vars:
                req_vars.append(depth_col)

        missing_vars = [req for req in req_vars if req not in available_vars]

        if missing_vars:
            missing_str = ", ".join(missing_vars)
            logger.error(
                "Validation Failed: '%s' requires %s but they are not provided. Pleaes review the config.",
                step_name,
                missing_str,
            )
            raise ValueError(
                f"Missing variables for '{step_name}': {missing_str}. Please add suitable steps beforehand."
            )

        available_vars.update(provided_vars)
        available_vars.update(parameters.get("to_derive", []))
        available_vars.update(parameters.get("qc_outputs", []))
        out = parameters.get("output_as")
        if out:
            available_vars.update(out if isinstance(out, (list, tuple)) else [out])

    if steps_list:
        logger.info("Pipeline variable check successful.")

    return True
