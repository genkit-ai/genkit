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

"""Backend-agnostic telemetry types. No OpenTelemetry imports."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar('T')


@dataclass(frozen=True)
class SpanMetadata:
    """Description of a span about to be created.

    Providers decide how to encode values. Extra fields beyond name / action_type
    / input / attributes are Genkit product facts (Dev UI path, init, subtype).
    """

    name: str
    action_type: str | None = None
    input: object | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)
    subtype: str | None = None
    init: object | None = None
    metadata: Mapping[str, object] | None = None
    is_root: bool | None = None


class SpanContext(Protocol):
    """Handle to a live span. No backend types leak through."""

    @property
    def trace_id(self) -> str:
        """Trace id, or empty when not instrumented."""
        ...

    @property
    def span_id(self) -> str:
        """Span id, or empty when not instrumented."""
        ...

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        """Attach custom metadata. Safe to call multiple times."""
        ...

    def set_output(self, value: object) -> None:
        """Override genkit:output when the return value is not the span output."""
        ...


@runtime_checkable
class Instrumentation(Protocol):
    """Pluggable provider. ``run_in_new_span`` wraps ``next`` like middleware."""

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: Callable[[SpanContext], Awaitable[T]],
    ) -> T: ...
