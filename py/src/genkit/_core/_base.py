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

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


def _default_serializer(obj: object) -> object:
    """Default serializer for objects not handled by json.dumps."""
    if isinstance(obj, bytes):
        try:
            return base64.b64encode(obj).decode('utf-8')
        except Exception:
            return '<bytes>'
    return str(obj)


class GenkitModel(BaseModel):
    """Base model with correct serialization defaults.

    All Genkit types inherit from this to ensure consistent serialization:
    - by_alias=True: Use camelCase field names (matching JS SDK)
    - exclude_none=True: Omit null fields (cleaner JSON)
    - fallback=_default_serializer: Handle bytes and other edge cases
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel,
        extra='forbid',
        populate_by_name=True,
    )

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
