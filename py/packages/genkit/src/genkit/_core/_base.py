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
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer
from pydantic.alias_generators import to_camel


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
    - no ``None`` fields, unless the class sets ``_keep_none_fields``
    - ``bytes`` as base64 when a Genkit type is dumped to JSON
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel,
        extra='forbid',
        populate_by_name=True,
        serialize_by_alias=True,
        ser_json_bytes='base64',
    )

    # Config objects turn this on: an explicit None there clears a model
    # ref's default, so it has to survive the dump.
    _keep_none_fields: ClassVar[bool] = False

    # No return annotation on purpose: pydantic would publish it as the
    # serialization schema, and Any there blanks out the schema FastAPI
    # shows for these types.
    @model_serializer(mode='wrap')
    def _serialize_wire(self, handler: SerializerFunctionWrapHandler):  # noqa: ANN202
        dumped = handler(self)
        if self._keep_none_fields or not isinstance(dumped, dict):
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
