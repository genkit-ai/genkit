# Copyright 2026 Google LLC
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

"""Small option types for Interactions-backed Google AI models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing_extensions import Self

ResponseModality = Literal['text', 'image', 'audio']


class ClientOptions(BaseModel):
    """HTTP settings reconstructed across background poll calls.

    Stored on Operation.metadata['clientOptions'] so check/cancel can reuse
    per-request overrides (including apiKey) from start. Wire keys are
    camelCase like the rest of operation metadata.
    """

    model_config = ConfigDict(alias_generator=to_camel, extra='ignore', populate_by_name=True)

    api_key: str | None = None
    api_version: str | None = None
    base_url: str | None = None
    custom_headers: dict[str, str] | None = None
    timeout: float | None = None

    def merge(self, overrides: ClientOptions | dict[str, Any] | None) -> Self:
        """Return a copy with non-null override fields applied."""
        if not overrides:
            return self
        if isinstance(overrides, dict):
            overrides = ClientOptions.model_validate(overrides)
        # Field names (not aliases) — model_copy(update=...) keys off Python attrs.
        update = overrides.model_dump(exclude_none=True)
        return self.model_copy(update=update) if update else self

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> Self:
        """Load options previously persisted on an Operation."""
        raw = (metadata or {}).get('clientOptions')
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            return cls.model_validate(raw)
        return cls()

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize for Operation.metadata (omit unset fields)."""
        return self.model_dump(by_alias=True, exclude_none=True)
