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

"""Action context definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


def joined_headers(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Collapse the same header name into one comma-joined value.

    The same header can arrive twice (two Authorization lines, each hop in
    an X-Forwarded-For chain). A context_provider looks the name up once, so
    keep every part instead of dropping all but the last.
    """
    joined: dict[str, str] = {}
    for key, value in pairs:
        existing = joined.get(key)
        joined[key] = value if existing is None else f'{existing}, {value}'
    return joined


@dataclass
class ContextMetadata:
    """A base class for Context metadata."""

    trace_id: str | None = None


@dataclass
class RequestData:
    """What a context_provider sees for one HTTP request.

    ``headers`` keys are lowercase so ``Authorization`` and ``authorization``
    look the same on every served flow. Repeated names become one
    comma-joined string, so ``headers.get('authorization')`` sees every
    value that arrived.
    """

    request: Any = None
    method: str = ''
    headers: dict[str, str] = field(default_factory=dict)
    input: Any = None
    metadata: ContextMetadata | None = None

    def __post_init__(self) -> None:
        # Header names are case-insensitive on the wire. Lowercase them so
        # Authorization and authorization are the same lookup, and join
        # values when mixed-case keys collapse to one name.
        self.headers = joined_headers((key.lower(), value) for key, value in self.headers.items())


ContextProvider = Callable[[RequestData], dict[str, Any] | Awaitable[dict[str, Any]]]
"""Middleware can read request data and add information to the context that will be passed to the
Action. If middleware throws an error, that error will fail the request and the Action will not
be called.

Expected cases should return a PublicError, which allows the request handler to
know what data is safe to return to end users.
Middleware can provide validation in addition to parsing. For example, an auth middleware can have
policies for validating auth in addition to passing auth context to the Action.
"""
