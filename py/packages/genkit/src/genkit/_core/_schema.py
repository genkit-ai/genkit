# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Functions for working with schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from jsonschema.validators import validator_for
from pydantic import TypeAdapter

from genkit._core._error import GenkitError, RuntimeErrorReason
from genkit._core._typing import ModelInfo


class InvalidOutputSchemaError(GenkitError):
    """The caller passed an output_schema that is not a JSON Schema."""

    def __init__(self, *, cause: Exception) -> None:
        super().__init__(
            status='INVALID_ARGUMENT',
            message=f'Invalid output_schema: {cause}',
            cause=cause,
            reason=RuntimeErrorReason.INVALID_SCHEMA,
        )


def to_json_schema(schema: type | dict[str, Any] | str | None) -> dict[str, Any]:
    """Convert a Python type to JSON schema. Pass-through if already a dict."""
    if schema is None:
        return {'type': 'null'}
    if isinstance(schema, dict):
        return schema
    type_adapter = TypeAdapter(schema)
    return type_adapter.json_schema()


def absorb_model_info(
    *,
    options: dict[str, object],
    info: ModelInfo | Mapping[str, Any] | None,
) -> object | None:
    """Copy card fields onto ``options`` and return the info's config schema.

    The Dev UI config form reads ``customOptions``. A schema on the model
    info is that form, so it is not copied under ``configSchema``.
    """
    if isinstance(info, ModelInfo):
        options.update(info.model_dump(by_alias=True, exclude_none=True, exclude={'config_schema'}))
        return info.config_schema
    if not isinstance(info, Mapping):
        return None
    schema: object | None = None
    for key, value in info.items():
        if key in ('configSchema', 'config_schema'):
            if schema is None and value is not None:
                schema = value
            continue
        if key not in options:
            options[key] = value
    return schema


def apply_config_class(*, options: dict[str, object], config_schema: type | dict[str, Any] | None) -> None:
    """A config class also checks ``config=`` on the call, so it wins the form."""
    if config_schema:
        options['customOptions'] = to_json_schema(config_schema)


def check_output_schema(json_schema: dict[str, Any]) -> None:
    """Raise if ``json_schema`` is not itself a JSON Schema.

    generate() calls this before the model so a broken ``output_schema``
    fails the request instead of discarding a turn the caller already paid for.
    """
    try:
        validator_cls = validator_for(json_schema)
        validator_cls.check_schema(json_schema)
    except Exception as error:
        raise InvalidOutputSchemaError(cause=error) from error


def parse_schema(*, data: object, json_schema: dict[str, Any]) -> None:
    """Raise if ``data`` does not satisfy ``json_schema``.

    generate() calls this before returning so a caller who asked for structured
    output does not get a response whose ``.output`` is the wrong shape.
    A broken ``output_schema`` is the same class of error — never an untyped
    exception retry would treat as a transport blip.
    """
    check_output_schema(json_schema)
    validator_cls = validator_for(json_schema)

    instance = cast(
        Mapping[str, Any] | Sequence[Any] | bool | float | int | str | None,
        data,
    )
    errors = sorted(validator_cls(json_schema).iter_errors(instance), key=lambda e: tuple(str(p) for p in e.path))
    if not errors:
        return

    lines = []
    for error in errors:
        path = '/'.join(str(part) for part in error.absolute_path) or '(root)'
        lines.append(f'- {path}: {error.message}')
    # Field paths are enough to debug; the invalid value and the schema
    # are already on the response, and stuffing them into error.message
    # bloated traces.
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message='Schema validation failed. Parse Errors:\n\n' + '\n'.join(lines),
        reason=RuntimeErrorReason.INVALID_INPUT,
    )
