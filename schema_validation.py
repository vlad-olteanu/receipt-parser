#!/usr/bin/env python3
"""Minimal JSON Schema validator for the shared receipt pipeline.

Both :mod:`receipt_parse` and :mod:`receipt_categorize` validate the LLM's
JSON output against a JSON Schema so that malformed model responses are caught
and retried before being written to disk. Both scripts are stdlib-only, so
instead of a ``jsonschema`` dependency this module implements the subset of
JSON Schema that the pipeline schemas use.

Supported draft-7 keywords:

    type, enum, const, required, properties, patternProperties,
    additionalProperties, items, minItems, maxItems, uniqueItems,
    minimum, maximum, exclusiveMinimum, exclusiveMaximum,
    minLength, maxLength, pattern

Type-checking follows JSON semantics: a Python ``bool`` is *not* a number or a
string, and a non-finite float does not satisfy ``number`` or ``integer``.
"""

import math
import re

__all__ = ["SchemaError", "validate", "errors"]


class SchemaError(ValueError):
    """Raised when a value fails schema validation.

    Subclasses :class:`ValueError` so callers that already catch
    ``ValueError`` (both pipeline scripts, inside their retry loops) pick it up
    with no change. ``SchemaError.errors`` carries the full list of violations.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or []


def validate(value, schema, path="$", show=5):
    """Validate ``value`` against ``schema``.

    Returns ``value`` on success. On failure raises :class:`SchemaError` (a
    :class:`ValueError`) whose message lists up to ``show`` violations.
    """
    errs = []
    _validate(value, schema, path, errs)
    if errs:
        joined = "; ".join(errs[:show])
        more = len(errs) - show
        if more > 0:
            joined += " (+{0} more)".format(more)
        raise SchemaError(joined, errs)
    return value


def errors(value, schema, path="$"):
    """Return the list of validation errors (empty list means valid)."""
    errs = []
    _validate(value, schema, path, errs)
    return errs


def _validate(value, schema, path, out):
    if schema is True:                    # empty schema: any value is valid
        return
    if schema is False:
        out.append("{0}: schema forbids all values".format(path))
        return
    if not isinstance(schema, dict):
        raise SchemaError("{0}: invalid schema (expected a JSON object)"
                          .format(path))

    if "enum" in schema and not _in_enum(schema["enum"], value):
        out.append("{0}: {1!r} not in enum {2}".format(
            path, value, _fmt(schema["enum"])))
    if "const" in schema and not _equal(schema["const"], value):
        out.append("{0}: {1!r} is not the required const {2}".format(
            path, value, _fmt(schema["const"])))
    if "type" in schema and not _type_ok(value, schema["type"]):
        out.append("{0}: expected type {1}, got {2}".format(
            path, _fmt(schema["type"]), _type_name(value)))
        return

    if isinstance(value, dict):
        for req in schema.get("required") or []:
            if req not in value:
                out.append("{0}: missing required property '{1}'".format(
                    path, req))
        properties = schema.get("properties") or {}
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key],
                          "{0}.{1}".format(path, key), out)
        for key, item in value.items():
            if key not in properties and _match_pattern_props(schema, key):
                for pattern, sub in schema["patternProperties"].items():
                    if re.search(pattern, key):
                        _validate(item, sub, "{0}.{1}".format(path, key), out)
                        break
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties or _match_pattern_props(schema, key):
                continue
            if additional is False:
                out.append("{0}: unexpected additional property '{1}'".format(
                    path, key))
            elif isinstance(additional, dict):
                _validate(item, additional, "{0}.{1}".format(path, key), out)
    elif isinstance(value, list):
        minimum = schema.get("minItems")
        if minimum is not None and len(value) < minimum:
            out.append("{0}: array has {1} items, fewer than minItems {2}"
                       .format(path, len(value), minimum))
        maximum = schema.get("maxItems")
        if maximum is not None and len(value) > maximum:
            out.append("{0}: array has {1} items, more than maxItems {2}"
                       .format(path, len(value), maximum))
        if schema.get("uniqueItems"):
            for i, item in enumerate(value):
                if any(_equal(item, prior) for prior in value[:i]):
                    out.append("{0}[{1}]: duplicate value violates "
                               "uniqueItems".format(path, i))
                    break
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for i, item in enumerate(value):
                _validate(item, items_schema, "{0}[{1}]".format(path, i), out)
        elif isinstance(items_schema, list):
            for i, item in enumerate(value[:len(items_schema)]):
                _validate(item, items_schema[i], "{0}[{1}]".format(path, i), out)
    elif isinstance(value, str):
        minimum = schema.get("minLength")
        if minimum is not None and len(value) < minimum:
            out.append("{0}: string length {1} less than minLength {2}"
                       .format(path, len(value), minimum))
        maximum = schema.get("maxLength")
        if maximum is not None and len(value) > maximum:
            out.append("{0}: string length {1} greater than maxLength {2}"
                       .format(path, len(value), maximum))
    if "pattern" in schema and isinstance(value, str) \
            and not re.search(schema["pattern"], value):
        out.append("{0}: {1!r} does not match pattern {2!r}".format(
            path, value, schema["pattern"]))
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value):
        _check_numeric(value, schema, path, out)


def _check_numeric(value, schema, path, out):
    if "minimum" in schema and value < schema["minimum"]:
        out.append("{0}: {1} less than minimum {2}".format(
            path, value, schema["minimum"]))
    if "maximum" in schema and value > schema["maximum"]:
        out.append("{0}: {1} greater than maximum {2}".format(
            path, value, schema["maximum"]))
    if "exclusiveMinimum" in schema \
            and value <= schema["exclusiveMinimum"]:
        out.append("{0}: {1} not greater than exclusiveMinimum {2}".format(
            path, value, schema["exclusiveMinimum"]))
    if "exclusiveMaximum" in schema \
            and value >= schema["exclusiveMaximum"]:
        out.append("{0}: {1} not less than exclusiveMaximum {2}".format(
            path, value, schema["exclusiveMaximum"]))


def _type_ok(value, stype):
    for t in (stype if isinstance(stype, list) else [stype]):
        if t == "string" and isinstance(value, str):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "null" and value is None:
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "object" and isinstance(value, dict):
            return True
        if t in ("number", "integer") \
                and not isinstance(value, bool) \
                and isinstance(value, (int, float)) \
                and math.isfinite(value):
            if t == "number" or isinstance(value, int):
                return True
    return False


def _in_enum(enum_values, value):
    return any(_equal(candidate, value) for candidate in enum_values)


def _equal(a, b):
    # In JSON a boolean is distinct from numbers, so ``True == 1`` must not
    # count as a match here.
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, (dict, list)) and isinstance(b, type(a)):
        return a == b
    return a is b or a == b


def _match_pattern_props(schema, key):
    for pat in schema.get("patternProperties") or {}:
        if re.search(pat, key):
            return True
    return False


def _type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _fmt(value):
    if isinstance(value, list):
        return "|".join(_fmt(v) for v in value)
    if isinstance(value, str):
        return value
    return repr(value)
