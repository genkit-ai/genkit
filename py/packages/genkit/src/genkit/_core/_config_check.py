# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Model config checks shared by regular and background models.

Before the turn, only the values a call sets are judged; a required field
left out may still be filled by middleware. Inside the model action (or a
background model's start action), the whole config is judged. A Pydantic
class and a JSON-schema dict follow the same rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from jsonschema.exceptions import SchemaError, best_match
from jsonschema.validators import validator_for
from pydantic import AliasChoices, BaseModel, ValidationError

from genkit._core._error import GenkitError, RuntimeErrorReason


def config_field_names(schema: type[BaseModel]) -> dict[str, str]:
    """Map each field name and alias to the Python field name."""
    names: dict[str, str] = {}
    for name, field in schema.model_fields.items():
        names[name] = name
        if field.alias:
            names[field.alias] = name
        accepted = field.validation_alias
        if isinstance(accepted, str):
            names[accepted] = name
        elif isinstance(accepted, AliasChoices):
            for choice in accepted.choices:
                if isinstance(choice, str):
                    names[choice] = name
    return names


def check_request_config(request: object, *, schema: type[BaseModel] | dict[str, Any], model: str) -> None:
    """Raise if a model request's config doesn't fit ``schema``, missing fields included.

    The model action and a background model's start action both run this
    first, after every middleware. A function with an unannotated request
    param gets the raw dict on a direct run, so either shape is read.
    """
    if isinstance(request, Mapping):
        config = cast(Mapping[str, Any], request).get('config')
    else:
        config = getattr(request, 'config', None)
    check_whole_config(config=config, schema=schema, model=model)


def check_config_values(*, config: object, schema: type[BaseModel] | dict[str, Any] | None, model: str) -> None:
    """Raise if a value the config sets doesn't fit the model's config schema.

    Only fields the config sets are judged. A required field it leaves out is
    not an error yet: another config layer or a wrap_model middleware may add
    it, and the model action rejects it if nothing does. ``None`` means
    "clear the default" and isn't checked. A JSON-schema dict is judged the
    same way as a class.
    """
    if schema is None or not isinstance(config, Mapping):
        return
    _raise_config_problems(values=_judged_values(config, schema), schema=schema, model=model, report_missing=False)


def check_whole_config(*, config: object, schema: type[BaseModel] | dict[str, Any], model: str) -> None:
    """Raise if the config the model is about to run with doesn't fit its config schema.

    Unlike ``check_config_values``, a missing required field is an error:
    this runs inside the model action, where nothing is left to fill it.
    """
    if isinstance(schema, type) and isinstance(config, schema):
        return
    if isinstance(config, BaseModel):
        config = config_values(config)
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        return
    _raise_config_problems(values=_judged_values(config, schema), schema=schema, model=model, report_missing=True)


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


def _judged_values(config: Mapping[Any, Any], schema: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    """The config values a schema is judged on.

    ``None`` means "clear the default" and is dropped. ``version`` comes from
    ``ModelRef.version``, which Genkit adds to config itself, so a schema that
    doesn't name it isn't judged on it.
    """
    declared = schema.get('properties', {}) if isinstance(schema, dict) else schema.model_fields
    return {
        key: value
        for key, value in cast(Mapping[str, Any], config).items()
        if value is not None and (key != 'version' or key in declared)
    }


def _raise_config_problems(
    *, values: dict[str, Any], schema: type[BaseModel] | dict[str, Any], model: str, report_missing: bool
) -> None:
    """Raise INVALID_ARGUMENT listing what in ``values`` breaks ``schema``.

    With ``report_missing=False``, a required top-level field left out isn't a
    problem. A field missing inside a nested object always is: the object is
    sent whole and can't be filled later.
    """
    if isinstance(schema, dict):
        described = _json_schema_problems(values, schema=schema, model=model, report_missing=report_missing)
        if described:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{model}: {"; ".join(described)}',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )
        return
    try:
        schema.model_validate(values)
    except ValidationError as e:
        # Pydantic reports a required field left out as type 'missing' at a
        # one-part path.
        problems = [
            err for err in e.errors() if report_missing or not (err['type'] == 'missing' and len(err['loc']) == 1)
        ]
        if not problems:
            return
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'{model}: {_describe_config_problems(problems, values=values, schema=schema)}',
            reason=RuntimeErrorReason.INVALID_INPUT,
            cause=e,
        ) from e


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


def _describe_config_problems(
    problems: Sequence[Mapping[str, Any]], *, values: Mapping[str, Any], schema: type[BaseModel]
) -> str:
    # pydantic binds one spelling of a setting and calls the other unknown;
    # the caller didn't misspell anything, they wrote the setting twice.
    names = config_field_names(schema)
    repeated: dict[str, list[str]] = {}
    unknown: list[str] = []
    for err in problems:
        if err['type'] != 'extra_forbidden':
            continue
        key = _config_path(err['loc'])
        field = names.get(key) if len(err['loc']) == 1 else None
        spellings = [k for k in values if field and names.get(k) == field]
        if field and len(spellings) > 1:
            repeated[field] = sorted(spellings, key=lambda k: k != field)
        else:
            unknown.append(key)
    parts = [f'{_join_words(spellings)} are the same setting; pass one' for spellings in repeated.values()]
    if unknown:
        keys = ', '.join(repr(key) for key in unknown)
        noun = 'key' if len(unknown) == 1 else 'keys'
        parts.append(f"unknown config {noun} {keys}; put provider-only settings in config['extra']")
    parts.extend(
        f'config {_config_path(err["loc"])!r}: {err["msg"]}' for err in problems if err['type'] != 'extra_forbidden'
    )
    return '; '.join(parts)


def _join_words(words: list[str]) -> str:
    """`a and b`, or `a, b, and c` for three or more."""
    if len(words) <= 2:
        return ' and '.join(words)
    return f'{", ".join(words[:-1])}, and {words[-1]}'


def _config_path(loc: tuple[int | str, ...]) -> str:
    return '.'.join(str(part) for part in loc)
