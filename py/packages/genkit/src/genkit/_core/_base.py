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

"""Base model with correct serialization defaults for Genkit types."""

from __future__ import annotations

import base64
from contextvars import ContextVar
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, SerializationInfo, SerializerFunctionWrapHandler, model_serializer
from pydantic.alias_generators import to_camel

# True while GenkitModel._serialize_wire is serializing a wire model, so a
# nested GenkitConfig uses the parent's wire rules (camelCase, no nulls) even
# when the outer dump started from a plain BaseModel.
_IN_GENKIT_MODEL_SERIALIZE: ContextVar[bool] = ContextVar('_in_genkit_model_serialize', default=False)


def _default_serializer(obj: object) -> object:
    """Default serializer for objects not handled by json.dumps."""
    if isinstance(obj, bytes):
        try:
            return base64.b64encode(obj).decode('utf-8')
        except Exception:
            return '<bytes>'
    return str(obj)


def dump_keeping_unknown(model: BaseModel) -> dict[str, Any]:
    # Keep values pydantic doesn't know how to serialize so a later JSON
    # dump can reject them instead of turning them into strings here.
    return model.model_dump(fallback=lambda obj: obj)


class GenkitModel(BaseModel):
    """Base model with correct serialization defaults.

    All Genkit types inherit from this so they serialize to the wire shape
    wherever they sit, including inside a user's own pydantic model:
    - camelCase keys (an explicit ``by_alias=False`` still gets field names)
    - no ``None`` fields (an explicit ``exclude_none=False`` is ignored; config
      models are ``GenkitConfig`` and keep them on their own)
    - ``bytes`` as base64 when a Genkit type is dumped to JSON
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel,
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        ser_json_bytes='base64',
    )

    # No return annotation on purpose: pydantic would publish it as the
    # serialization schema, and Any there blanks out the schema FastAPI
    # shows for these types.
    @model_serializer(mode='wrap')
    def _serialize_wire(self, handler: SerializerFunctionWrapHandler):  # noqa: ANN202
        token = _IN_GENKIT_MODEL_SERIALIZE.set(True)
        try:
            dumped = handler(self)
        finally:
            _IN_GENKIT_MODEL_SERIALIZE.reset(token)
        if not isinstance(dumped, dict):
            return dumped
        return self._drop_none_fields(dumped)

    def _drop_none_fields(self, dumped: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in dumped.items() if value is not None}

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump model with Genkit defaults (by_alias=True, exclude_none=True)."""
        kwargs.setdefault('by_alias', True)
        kwargs.setdefault('exclude_none', True)
        kwargs.setdefault('fallback', _default_serializer)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        """Dump model to JSON with Genkit defaults."""
        kwargs.setdefault('by_alias', True)
        kwargs.setdefault('exclude_none', True)
        kwargs.setdefault('fallback', _default_serializer)
        return super().model_dump_json(**kwargs)


class GenkitConfig(BaseModel):
    """Base for settings passed to a Genkit action: model and middleware config.

    Python callers construct it with field names (snake_case), and plugins
    hand model_dump() output to provider SDKs. Genkit's tooling sees
    camelCase JSON: the Developer UI renders the JSON schema as a form,
    pre-fills it from traces and .prompt files by exact key, and exports
    .prompt files from it. Those formats are shared with the JS and Go SDKs,
    so the schema keys, trace keys and .prompt keys must use one spelling.

    - alias_generator=to_camel: camelCase wire names and schema properties.
    - validate_by_name / validate_by_alias: accept either spelling; all three
      type checkers take the snake_case kwarg.
    - extra='forbid': an unknown key raises instead of being dropped, and so
      does a second spelling of a key already given (maxOutputTokens and
      max_output_tokens in one input).

    model_dump() is not overridden: the default returns field names. Pass
    by_alias=True for Genkit JSON. Nested in a GenkitModel such as
    ModelRequest, the parent's settings apply, so traces stay camelCase.

    Use plain BaseModel for flow and tool input/output; that data is the
    app's, and Genkit does not rename it. An app's own middleware config may
    also be a plain BaseModel: its keys are its field names everywhere, and
    unknown keys still raise.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel,
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True,
    )

    # No return annotation for the same schema reason as GenkitModel._serialize_wire.
    @model_serializer(mode='wrap')
    def _serialize_in_genkit_model(  # noqa: ANN202
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ):
        if not _IN_GENKIT_MODEL_SERIALIZE.get():
            return handler(self)
        if info.by_alias is None or not info.exclude_none:
            return self.model_dump(
                mode=info.mode,
                include=cast(Any, info.include),
                exclude=cast(Any, info.exclude),
                context=info.context,
                by_alias=True if info.by_alias is None else info.by_alias,
                exclude_unset=info.exclude_unset,
                exclude_defaults=info.exclude_defaults,
                exclude_none=True,
                exclude_computed_fields=info.exclude_computed_fields,
                round_trip=info.round_trip,
                serialize_as_any=info.serialize_as_any,
            )
        return handler(self)
