# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Config checks for models, background models, embedders and evaluators.

Two checkpoints, one set of rules:

- Call time (``check_config_values``): only the keys the caller set. A
  required top-level field left out passes; the schema defaults, another
  layer, or a ``wrap_model`` middleware may still fill it.
- The action boundary (``validate_whole_config``, from ``Action.run``): the
  merged config, after every middleware. A required field nothing filled
  fails here.

On both, a key that is present is judged, an explicit ``None`` included; an
absent key is unset. Spellings of one field fold to the key the class
accepts. ``version`` comes from ``ModelRef.version``, which Genkit adds to
config itself, so a schema that doesn't declare it isn't judged on it. A
Pydantic class and a JSON-schema dict follow the same rules and are reported
in the same words.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from jsonschema.exceptions import SchemaError, best_match
from jsonschema.validators import validator_for
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from genkit._core._error import (
    GenkitError,
    RuntimeErrorReason,
    validation_error_details,
    validation_input_is_printable,
    validation_problem,
)
from genkit._core._model import config_field_names, fold_config_aliases

ConfigSchema = type[BaseModel] | dict[str, Any]


def check_config_values(*, config: object, schema: ConfigSchema | None, model: str) -> None:
    """Raise INVALID_ARGUMENT if a key the caller set doesn't fit ``schema``.

    A required top-level field left out is not an error yet. A nested object
    is sent whole, so a field missing inside one is.
    """
    if schema is None or not isinstance(config, Mapping):
        return
    _check(config=cast(Mapping[str, Any], config), schema=schema, model=model, report_missing=False)


def validate_whole_config(*, config: object, schema: ConfigSchema, model: str) -> BaseModel | None:
    """Raise INVALID_ARGUMENT if the config an action is about to run with doesn't fit ``schema``.

    A missing required field is an error: this runs at the action boundary,
    after every layer and middleware. Returns the validated instance for a
    class schema, ``None`` for a JSON-schema dict or a config that isn't a
    mapping (request validation reports the wrong type).
    """
    if isinstance(schema, type) and isinstance(config, schema):
        return config
    if isinstance(config, BaseModel):
        config = config_values(config)
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        return None
    return _check(config=cast(Mapping[str, Any], config), schema=schema, model=model, report_missing=True)


def check_config_json_schema(schema: dict[str, Any], *, model: str) -> None:
    """Raise at define time when a ``config_schema=`` dict isn't a valid JSON schema."""
    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as e:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'{model}: config_schema is not a valid JSON schema: {e.message}',
            cause=e,
        ) from e


def config_values(config: BaseModel) -> dict[str, Any]:
    """``config`` as plain data, nested models included, in the shape validation takes.

    ``BaseModel.model_dump`` skips GenkitModel's ``exclude_none`` default (an
    explicit ``None`` is a value too) and its fallback serializer, so a value
    of the wrong type comes back as itself and fails validation.
    """
    return BaseModel.model_dump(config, by_alias=True, round_trip=True, exclude_none=False, warnings=False)


def _check(*, config: Mapping[str, Any], schema: ConfigSchema, model: str, report_missing: bool) -> BaseModel | None:
    if isinstance(schema, dict):
        values = {k: v for k, v in config.items() if k != 'version' or k in schema.get('properties', {})}
        described = _json_schema_problems(values, schema=schema, model=model, report_missing=report_missing)
        if described:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{model}: {"; ".join(described)}',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )
        return None
    names = config_field_names(schema)
    values = {k: v for k, v in config.items() if k != 'version' or k in names}
    return _check_class(values, schema=schema, names=names, model=model, report_missing=report_missing)


def _check_class(
    values: dict[str, Any], *, schema: type[BaseModel], names: dict[str, str], model: str, report_missing: bool
) -> BaseModel | None:
    """Fold, validate, and raise one INVALID_ARGUMENT naming every problem.

    Two spellings of one setting in the same dict (``max_output_tokens`` and
    ``maxOutputTokens``) are reported as their own problem, whatever the
    class's ``extra`` setting. Folding would keep whichever came last in dict
    order, and under ``extra='forbid'`` pydantic would call the other one
    unknown; the caller didn't misspell anything, they wrote the setting
    twice. Spellings in different layers (a ModelRef's config and the call's)
    are folded before they get here, so the later layer still wins.
    """
    spellings: dict[str, list[str]] = {}
    for key in values:
        spellings.setdefault(names.get(key, key), []).append(key)
    repeated = [sorted(keys, key=lambda k: k != field) for field, keys in spellings.items() if len(keys) > 1]
    instance: BaseModel | None = None
    problems: list[ErrorDetails] = []
    cause: ValidationError | None = None
    try:
        instance = schema.model_validate(fold_config_aliases(config=values, schema=schema))
    except ValidationError as e:
        cause = e
        # Pydantic reports a required field left out as type 'missing' at a
        # one-part path.
        problems = [
            err for err in e.errors() if report_missing or not (err['type'] == 'missing' and len(err['loc']) == 1)
        ]
    if not repeated and not problems:
        return instance
    parts = [f'{_join_words(keys)} are the same setting; pass one' for keys in repeated]
    unknown = [_config_path(err['loc']) for err in problems if err['type'] == 'extra_forbidden']
    if unknown:
        keys = ', '.join(repr(key) for key in unknown)
        noun = 'key' if len(unknown) == 1 else 'keys'
        parts.append(f"unknown config {noun} {keys}; put provider-only settings in config['extra']")
    printable = cause is not None and validation_input_is_printable(cause)
    # An error with no path is a model validator; its input is the whole config.
    parts.extend(
        f'config {_config_path(err["loc"])!r}: {validation_problem(err, show_input=printable and bool(err["loc"]))}'
        for err in problems
        if err['type'] != 'extra_forbidden'
    )
    error = GenkitError(
        status='INVALID_ARGUMENT',
        message=f'{model}: {"; ".join(parts)}',
        reason=RuntimeErrorReason.INVALID_INPUT,
        details={'errors': validation_error_details(cause)} if cause is not None else None,
    )
    if cause is None:
        raise error
    # The message already names every problem. Passing cause= would append
    # Pydantic's wording of the same problems to str(error), so the cause is
    # attached after construction instead.
    error.cause = cause
    raise error from cause


def _json_schema_problems(
    values: dict[str, Any], *, schema: dict[str, Any], model: str, report_missing: bool
) -> list[str]:
    """One line per JSON-schema error in ``values``, ordered by path, worded like the class check."""
    try:
        errors = sorted(
            validator_for(schema)(schema).iter_errors(values), key=lambda err: [str(part) for part in err.absolute_path]
        )
    except Exception as e:
        # check_schema at define time can't catch everything, e.g. a $ref to
        # nowhere. That's the model author's schema, not the caller's config.
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=f"{model}: config_schema couldn't be applied: {e}",
            cause=e,
        ) from e
    described: list[str] = []
    for err in errors:
        # An optional field is anyOf [its schema, null]; report why the value
        # failed its schema, not that it matched neither branch.
        if err.context:
            err = best_match([sub for sub in err.context if sub.validator != 'type'] or err.context)
        if err.validator == 'required':
            # One error per object missing fields, on that object's path. An
            # empty path is the top level, where middleware may still fill it.
            if not report_missing and not err.absolute_path:
                continue
            present = err.instance if isinstance(err.instance, Mapping) else {}
            for field in cast(list[str], err.validator_value):
                if field not in present:
                    described.append(f'config {_config_path((*err.absolute_path, field))!r}: Field required')
            continue
        path = _config_path(tuple(err.absolute_path))
        described.append(f'config {path!r}: {err.message}' if path else f'config: {err.message}')
    return list(dict.fromkeys(described))


def _join_words(words: list[str]) -> str:
    """`a and b`, or `a, b, and c` for three or more."""
    if len(words) <= 2:
        return ' and '.join(words)
    return f'{", ".join(words[:-1])}, and {words[-1]}'


def _config_path(loc: tuple[int | str, ...]) -> str:
    return '.'.join(str(part) for part in loc)
