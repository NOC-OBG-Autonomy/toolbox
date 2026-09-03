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

"""The single, canonical parameter-schema format shared by every step and QC check.

A component declares its parameters as a ``parameter_schema`` class attribute: a
dict mapping each parameter name to a small spec dict.

Example
-------
::

    parameter_schema = {
        "velocity_threshold": {
            "type": float,
            "default": 0.033,        # has a default => optional; this is the preset
            "description": "Vertical velocity (m/s) to flag an up/down cast.",
            # optional hints, consumed by the (future) dashboard:
            "min": 0.0, "max": 0.5, "step": 0.001, "unit": "m/s",
        },
        "file_path": {
            "type": str,
            "required": True,        # no sensible preset => must be supplied
            "description": "Path to the input file.",
        },
        "method": {
            "type": str,
            "default": "poly",
            "options": ["poly", "linear"],   # enumerated choices
            "description": "Fitting method.",
        },
    }

A parameter is **required** when its spec has no ``default`` key, or sets
``"required": True``. Otherwise the ``default`` is used whenever the user omits it,
so a fully-defaulted component runs with no configuration at all.
"""

from __future__ import annotations

_MISSING = object()


def is_required(spec: dict) -> bool:
    """Return whether a single parameter spec describes a required parameter."""
    if spec.get("required"):
        return True
    return "default" not in spec


def _allowed_types(spec: dict):
    """Normalise a spec's ``type`` to a tuple, or ``None`` if no type is declared."""
    t = spec.get("type")
    if t is None:
        return None
    return tuple(t) if isinstance(t, (list, tuple)) else (t,)


def matches_type(spec: dict, value) -> bool:
    """Return whether ``value`` satisfies a single parameter spec's ``type``.

    A spec with no ``type`` accepts anything. Special cases:

    - ``None`` is accepted when ``None`` is the declared default (a sentinel
      meaning e.g. "compute it", as with ``dark_value``).
    - ``bool`` is a subclass of ``int`` but is only accepted when ``bool`` is
      explicitly listed — so ``acceleration_threshold: no`` (YAML ``False``) is
      rejected for a ``float`` parameter.
    - an ``int`` is accepted where a ``float`` is expected (e.g. ``20`` for a
      threshold), since YAML has no way to force a whole-number float.
    """
    allowed = _allowed_types(spec)
    if allowed is None:
        return True
    if value is None and spec.get("default", _MISSING) is None:
        return True
    if isinstance(value, bool):
        return bool in allowed
    if float in allowed and isinstance(value, int):
        return True
    # list and tuple are interchangeable sequences: YAML yields lists, in-code
    # callers often pass tuples, and schemas use the two without distinction.
    if (list in allowed or tuple in allowed) and isinstance(value, (list, tuple)):
        return True
    return isinstance(value, allowed)


def coerce(spec: dict, value):
    """Coerce a numeric *string* to the spec's declared numeric type.

    Configs are not always parsed by pelagos: a caller may hand the pipeline a
    dict it built itself (or parsed with plain ``yaml.safe_load``), in which case
    scientific-notation numbers like ``3e-2`` arrive as strings (a YAML 1.1
    quirk). When the spec declares a numeric type and does not also accept ``str``,
    convert such a string to ``int``/``float`` so it satisfies the type check
    regardless of how the config was produced.

    Anything that is not a numeric string, or whose spec legitimately accepts a
    string, is returned unchanged.
    """
    allowed = _allowed_types(spec)
    if allowed is None or str in allowed or not isinstance(value, str):
        return value
    text = value.strip()
    if int in allowed and float not in allowed:
        try:
            return int(text)
        except ValueError:
            return value
    if float in allowed:
        try:
            return float(text)
        except ValueError:
            return value
    return value


def _expected_str(spec: dict) -> str:
    """Render a spec's declared ``type`` for an error message."""
    names = _type_name(spec.get("type"))
    return " or ".join(names) if isinstance(names, list) else str(names)


def type_errors(schema: dict, params: dict) -> list[str]:
    """Return ``"name (expected X, got Y)"`` messages for type-mismatched params.

    Only user-supplied values present in ``schema`` are checked; omitted
    parameters and undeclared keys are out of scope here (defaults are trusted,
    and unknown keys are handled separately).
    """
    return [
        f"{name} (expected {_expected_str(spec)}, got {type(params[name]).__name__} "
        f"{params[name]!r})"
        for name, spec in schema.items()
        if name in params and not matches_type(spec, coerce(spec, params[name]))
    ]


def option_errors(schema: dict, params: dict) -> list[str]:
    """Return ``"name (expected one of ..., got ...)"`` for out-of-``options`` params.

    Only supplied params whose spec declares ``options`` are checked. For a
    list-valued param ``options`` constrains each element.
    """
    errors = []
    for name, spec in schema.items():
        if name not in params or not spec.get("options"):
            continue
        options = spec["options"]
        value = coerce(spec, params[name])
        values = value if isinstance(value, (list, tuple)) else [value]
        bad = [v for v in values if v not in options]
        if bad:
            got = bad if isinstance(value, (list, tuple)) else value
            errors.append(f"{name} (expected one of {options}, got {got!r})")
    return errors


def resolve(
    schema: dict,
    params: dict,
    *,
    label: str = "component",
    allowed_extra=(),
) -> dict:
    """Resolve user-supplied ``params`` against a ``schema``.

    Every parameter declared in ``schema`` is returned: user values are passed
    through, omitted optional parameters fall back to their ``default``, and any
    omitted required parameter raises ``ValueError``.

    Unknown parameters — keys present in ``params`` but not declared in ``schema``
    nor listed in ``allowed_extra`` — raise ``ValueError`` so config typos are
    caught early. ``allowed_extra`` exists for framework/mixin keys that a handler
    other than the schema consumes (e.g. ``qc_handling_settings``).

    Parameters
    ----------
    schema : dict
        The component's ``parameter_schema``.
    params : dict
        The user-supplied parameters for this component.
    label : str, optional
        Name used in error messages (typically the step/QC name).
    allowed_extra : iterable of str, optional
        Parameter names that are permitted even though they are not in ``schema``.

    Returns
    -------
    dict
        Resolved ``{name: value}`` for every parameter in ``schema``.

    Raises
    ------
    ValueError
        If a required parameter is missing, an unknown parameter is supplied, a
        value has the wrong type, or a value declaring ``options`` is not one of
        them.
    """
    unknown = set(params) - set(schema) - set(allowed_extra)
    if unknown:
        raise ValueError(
            f"[{label}] unknown parameter(s): {', '.join(sorted(unknown))}. "
            f"Valid parameters: {', '.join(sorted(schema)) or '(none)'}."
        )

    # Coerce numeric strings (e.g. "3e-2" from a hand-parsed/plain-YAML config)
    # to their declared numeric type before any type checking.
    supplied = {name: coerce(schema[name], params[name]) for name in params if name in schema}

    resolved = {}
    missing = []
    for name, spec in schema.items():
        if name in supplied:
            resolved[name] = supplied[name]
        elif is_required(spec):
            missing.append(name)
        else:
            resolved[name] = spec.get("default")

    if missing:
        raise ValueError(
            f"[{label}] missing required parameter(s): {', '.join(sorted(missing))}"
        )

    bad_types = type_errors(schema, supplied)
    if bad_types:
        raise ValueError(
            f"[{label}] invalid parameter type(s): {'; '.join(bad_types)}"
        )

    bad_options = option_errors(schema, supplied)
    if bad_options:
        raise ValueError(
            f"[{label}] invalid parameter value(s): {'; '.join(bad_options)}"
        )

    return resolved


def _type_name(t):
    """Render a schema ``type`` (a type, or list of types) as JSON-safe string(s)."""
    if isinstance(t, (list, tuple)):
        return [_type_name(x) for x in t]
    return getattr(t, "__name__", str(t)) if t is not None else None


def describe(schema: dict) -> list[dict]:
    """Return a JSON-serialisable description of a schema.

    This can be used to interface with a dashboard (or any external tool).
    """
    described = []
    for name, spec in schema.items():
        entry = {
            "name": name,
            "type": _type_name(spec.get("type")),
            "required": is_required(spec),
            "description": spec.get("description", ""),
        }
        if not is_required(spec):
            entry["default"] = spec.get("default")
        for hint in ("min", "max", "step", "unit", "options"):
            if hint in spec:
                entry[hint] = spec[hint]
        described.append(entry)
    return described
