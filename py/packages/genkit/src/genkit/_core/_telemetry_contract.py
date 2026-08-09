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

"""Dependency-free telemetry handler types and process-global registry.

Kept separate from OpenTelemetry-backed handlers so handlers and the dispatcher
can share types/registry/annotate-flush without a circular import.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from typing import Any, ClassVar, NamedTuple

from pydantic import BaseModel, ConfigDict


class SpanHookParams(BaseModel):
    """Params passed through the telemetry handler chain for one span enter."""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    name: str
    attributes: dict[str, Any]
    kind: str = 'internal'
    links: list[Any] | None = None


TelemetryHandler = Callable[
    [SpanHookParams, Callable[[], Awaitable[Any]]],
    Awaitable[Any],
]

handlers: list[TelemetryHandler] = []

# Set by the renderer while a span is open so annotate() can mirror frame writes
# onto the live backend span without business code importing OpenTelemetry.
annotate_flush: ContextVar[Callable[[str, Any], None] | None] = ContextVar('genkit_annotate_flush', default=None)

# Set by an enrichment handler (e.g. GenAI conventions) for the active span so
# mid/end annotate() writes can be projected without baking a convention into
# annotate() itself. Nested spans replace and restore like annotate_flush.
AnnotateProjector = Callable[[], None]
annotate_projector: ContextVar[AnnotateProjector | None] = ContextVar('genkit_annotate_projector', default=None)


# Plain frame stack (name/attrs only) for subtree-root / annotate ambient context.
# Lives here so handlers can read the active frame without importing the dispatcher.
class SpanFrame(NamedTuple):
    """One Genkit frame on the logical call stack."""

    name: str
    attrs: dict[str, Any]


frame_stack: ContextVar[tuple[SpanFrame, ...]] = ContextVar('genkit_frame_stack', default=())


def current_frame() -> SpanFrame | None:
    """Return the innermost Genkit frame, if any."""
    frames = frame_stack.get()
    return frames[-1] if frames else None


def get_telemetry_handlers() -> list[TelemetryHandler]:
    """Return the registered handler list (mutable; tests may clear/replace)."""
    return handlers


def register_genkit_telemetry_handler(handler: TelemetryHandler) -> None:
    """Append a telemetry handler to the process-global chain."""
    handlers.append(handler)


def clear_genkit_telemetry_handlers() -> None:
    """Remove all handlers (test helper)."""
    handlers.clear()


def bind_annotate_flush(flush: Callable[[str, Any], None]) -> Token[Callable[[str, Any], None] | None]:
    """Install a flush callback for the active renderer (returns reset token)."""
    return annotate_flush.set(flush)


def unbind_annotate_flush(token: Token[Callable[[str, Any], None] | None]) -> None:
    annotate_flush.reset(token)


def bind_annotate_projector(projector: AnnotateProjector) -> Token[AnnotateProjector | None]:
    """Install a mid/end attr projector for the active enrichment handler."""
    return annotate_projector.set(projector)


def unbind_annotate_projector(token: Token[AnnotateProjector | None]) -> None:
    annotate_projector.reset(token)
