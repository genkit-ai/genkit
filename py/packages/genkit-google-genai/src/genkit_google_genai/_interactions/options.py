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
    """HTTP settings for an Interactions call.

    A tenant key never lives here on a ticket. Check/cancel read it again
    from context.secrets. Tickets may persist timeout, headers, and
    api_version — not api_key or base_url (those would leak a secret or
    let a ticket steer the next request at an attacker host).
    """

    model_config = ConfigDict(alias_generator=to_camel, extra='ignore', populate_by_name=True)

    api_key: str | None = None
    api_version: str | None = None
    base_url: str | None = None
    custom_headers: dict[str, str] | None = None
    # Milliseconds — applied to the HTTP call, not the create body.
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
        """Load transport knobs previously persisted on an Operation.

        Drops api_key and base_url even if an older ticket still has them.
        The key is re-supplied on the run; a ticket must not pick the host.
        """
        raw = (metadata or {}).get('clientOptions')
        if raw is None:
            raw = (metadata or {}).get('client_options')
        parsed: Self
        if isinstance(raw, cls):
            parsed = raw
        elif isinstance(raw, dict):
            parsed = cls.model_validate(raw)
        else:
            return cls()
        return parsed.model_copy(update={'api_key': None, 'base_url': None})

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize transport knobs for Operation.metadata (no key, no host)."""
        dumped = self.model_dump(by_alias=True, exclude_none=True)
        dumped.pop('apiKey', None)
        dumped.pop('api_key', None)
        dumped.pop('baseUrl', None)
        dumped.pop('base_url', None)
        return dumped
