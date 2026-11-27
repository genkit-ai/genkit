# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The config pipeline for models, background models and embedders.

Every entry point folds, layers and checks config with these helpers:

- Folding (``config_field_names``): every spelling of a field becomes the
  one key the class accepts.
- Layering (``layer_call_config``, ``overlay_config``, ``config_defaults``):
  ``schema defaults < ref.version < ref.config < call``, last layer wins.

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

import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from jsonschema.exceptions import SchemaError, best_match
from jsonschema.validators import validator_for
from pydantic import AliasChoices, BaseModel, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import ErrorDetails

from genkit._core._error import (
    ErrorReason,
    GenkitError,
    validation_error_details,
    validation_input_is_printable,
    validation_problem,
)

ConfigSchema = type[BaseModel] | dict[str, Any]


def config_key(schema: type[BaseModel], name: str) -> str:
    """The key ``schema`` accepts for field ``name`` when validating a dict.

    The field name when the class validates by name; otherwise its
    validation alias or alias. Folding to this key keeps an alias-only class
    (alias_generator without populate_by_name) from losing the value.
    """
    if schema.model_config.get('validate_by_name') or schema.model_config.get('populate_by_name'):
        return name
    field = schema.model_fields[name]
    accepted = field.validation_alias
    if isinstance(accepted, str):
        return accepted
    if isinstance(accepted, AliasChoices):
        for choice in accepted.choices:
            if isinstance(choice, str):
                return choice
    return field.alias or name


@functools.cache
def config_field_names(schema: type[BaseModel]) -> Mapping[str, str]:
    """Map each field name and alias to the one key the class accepts (config_key). Cached per class."""
    names: dict[str, str] = {}
    for name, field in schema.model_fields.items():
        key = config_key(schema, name)
        names[name] = key
        if field.alias:
            names[field.alias] = key
        accepted = field.validation_alias
        if isinstance(accepted, str):
            names[accepted] = key
        elif isinstance(accepted, AliasChoices):
            for choice in accepted.choices:
                if isinstance(choice, str):
                    names[choice] = key
    return MappingProxyType(names)


def fold_config_aliases(*, config: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
    """Rewrite each spelling of a field to the key the class accepts. Unknown keys stay as written."""
    names = config_field_names(schema)
    return {names.get(key, key): value for key, value in config.items()}


def overlay_config(*, layers: list[dict[str, Any]], schema: type[BaseModel] | None) -> dict[str, Any]:
    """Fold each layer, last layer wins.

    ``maxOutputTokens`` and ``max_output_tokens`` are the same slot. Keys
    the schema does not know pass through. With no schema, keys stay as
    written. An explicit ``None`` is a value: it wins over lower layers and
    goes on to validation.
    """
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(fold_config_aliases(config=layer, schema=schema) if schema is not None else layer)
    return merged


@functools.cache
def _default_fields(schema: type[BaseModel]) -> tuple[tuple[str, str, FieldInfo], ...]:
    """Fields whose default may be non-None, as (accepted key, name, field). Cached per class.

    Required fields and factories that need validated data have no
    standalone default. A plain ``None`` default is skipped here.
    """
    return tuple(
        (config_key(schema, name), name, field)
        for name, field in schema.model_fields.items()
        if not (field.is_required() or field.default_factory_takes_validated_data)
        and not (field.default_factory is None and field.default is None)
    )


@functools.cache
def _json_adapter(schema: type[BaseModel], name: str) -> TypeAdapter[Any]:
    return TypeAdapter(schema.model_fields[name].annotation)


def config_defaults(schema: type[BaseModel]) -> dict[str, Any]:
    """The non-None defaults a config class declares, keyed by the key it accepts.

    A ``None`` default means "not set" on every built-in plugin config, so it
    contributes nothing. Values are JSON-shaped, what a Dev UI caller would
    send (a nested model is a dict, an Enum its value), and fresh per call,
    so a fn that mutates a default list doesn't change the next call's.
    """
    defaults: dict[str, Any] = {}
    for key, name, field in _default_fields(schema):
        value = field.get_default(call_default_factory=True)
        if value is None:
            continue
        if not isinstance(value, (str, int, float, bool)):
            value = _json_adapter(schema, name).dump_python(value, mode='json', warnings=False)
        defaults[key] = value
    return defaults


def copy_containers(value: object) -> object:
    """Copy dicts, lists, tuples and sets all the way down; keep leaf objects."""
    if isinstance(value, dict):
        return {key: copy_containers(item) for key, item in cast(dict[Any, Any], value).items()}
    if isinstance(value, list):
        return [copy_containers(item) for item in cast(list[Any], value)]
    if isinstance(value, tuple):
        return tuple(copy_containers(item) for item in cast(tuple[Any, ...], value))
    if isinstance(value, set):
        return {copy_containers(item) for item in cast(set[Any], value)}
    return value


def normalize_config(*, config: object) -> dict[str, Any]:
    """Dump a config object or dict. Does not fold or merge.

    Pydantic dumps the Python field names, including explicit ``None``.
    Dict keys stay as written; nested dicts, lists, tuples and sets are
    copied. Fields marked ``exclude=True`` are copied back.
    """
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        dumped = config.model_dump(exclude_unset=True, exclude_none=False, by_alias=False)
        # a plugin can keep a client-only setting out of JSON with exclude=True;
        # copy it back so the setting the caller passed still reaches the plugin.
        for name in config.model_fields_set:
            if name not in dumped:
                dumped[name] = getattr(config, name)
        return dumped
    if isinstance(config, Mapping):
        # Containers are copied all the way down, so a fn that edits a nested
        # list can't change the caller's ref for the next call. Leaf objects
        # (a client, a lock) are kept as they are.
        return cast(dict[str, Any], copy_containers(dict(cast(Mapping[str, Any], config))))
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f'config is {type(config).__name__}, expected Mapping or BaseModel.',
        reason=ErrorReason.INVALID_INPUT,
    )


def layer_call_config(
    *,
    call: object,
    version: str | None = None,
    ref_config: object = None,
    schema: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """``ref.version < ref.config < call``: the merge generate and embed share.

    Each layer is the fields it set (normalize_config, deep-copied), folded
    to the keys ``schema`` accepts. An explicit ``None`` is a value: it wins
    over lower layers and goes to validation.
    """
    layers: list[dict[str, Any]] = []
    if version is not None:
        layers.append({'version': version})
    if ref_config is not None:
        layers.append(normalize_config(config=ref_config))
    layers.append(normalize_config(config=call))
    return overlay_config(layers=layers, schema=schema)


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
    """Fold, validate, and raise one INVALID_ARGUMENT naming every problem.

    Two spellings of one setting in the same dict (``max_output_tokens`` and
    ``maxOutputTokens``) are reported as their own problem, whatever the
    class's ``extra`` setting. Folding would keep whichever came last in dict
    order, and under ``extra='forbid'`` pydantic would call the other one
    unknown; the caller didn't misspell anything, they wrote the setting
    twice. Spellings in different layers (a ModelRef's config and the call's)
    are folded before they get here, so the later layer still wins.
    """
    if isinstance(schema, dict):
        values = {k: v for k, v in config.items() if k != 'version' or k in schema.get('properties', {})}
        described = _json_schema_problems(values, schema=schema, model=model, report_missing=report_missing)
        if described:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{model}: {"; ".join(described)}',
                reason=ErrorReason.INVALID_INPUT,
            )
        return None
    names = config_field_names(schema)
    values = {k: v for k, v in config.items() if k != 'version' or k in names}
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
        reason=ErrorReason.INVALID_INPUT,
        details={'errors': validation_error_details(cause)} if cause is not None else None,
    )
    if cause is None:
        raise error
    # The message already names every problem; passing cause= would append
    # Pydantic's wording of the same problems to str(error) and the Dev UI.
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
