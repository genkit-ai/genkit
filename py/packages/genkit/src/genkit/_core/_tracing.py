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


"""Telemetry dispatcher for the Genkit framework.

Composes registered telemetry handlers around span bodies. Work always runs
inside ``fn`` (the innermost continuation). OpenTelemetry lives in
``_telemetry_handlers`` / ``_trace/``; this module only composes the chain.

Attribute visibility timing
---------------------------
1. Start-known facts must be on ``params.attributes`` / ``start_attributes``
   **before** the renderer creates the span (live Dev UI snapshots at on_start).
2. Mid-run facts go through ``annotate()`` (mirrored live via renderer flush).
3. ``genkit:output`` via ``annotate_output`` during the body; ``genkit:state``
   / error stamped by the renderer before span end.
4. Do not mutate the pre-start attrs dict for live visibility after the span
   has started — use ``annotate`` for mid-run.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, ClassVar, TypeVar, cast, overload

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from ._base import GenkitModel
from ._environment import is_dev_environment
from ._telemetry_contract import (
    SpanFrame,
    SpanHookParams,
    TelemetryHandler,
    annotate_flush,
    annotate_projector,
    clear_genkit_telemetry_handlers,
    current_frame,
    frame_stack,
    get_telemetry_handlers,
    register_genkit_telemetry_handler,
)
from ._telemetry_gen_ai import is_model_span
from ._telemetry_handlers import ensure_dev_telemetry, otel_ai_semantic_conventions, otel_renderer
from ._trace._attrs import TYPE_FACT, Attr, metadata_key
from ._trace._path import build_path

T = TypeVar('T')

initialized = False


class SpanMetadata(GenkitModel):
    """Input parameters for opening a Genkit span via :func:`run_in_new_span`.

    Mapping from SpanMetadata to span attributes (see ``Attr`` for wire names):
      - name                 -> Attr.NAME (and span name)
      - input                -> Attr.INPUT (JSON-serialized at start)
      - type                 -> Attr.TYPE
      - subtype              -> Attr.SUBTYPE
      - metadata[k]          -> metadata_key(k)
      - telemetry_labels[k]  -> <k> verbatim (caller-controlled keys)

    Write ``genkit:output`` during the span body with :func:`annotate_output`.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, extra='forbid', populate_by_name=True, arbitrary_types_allowed=True
    )

    name: str = Field(...)
    input: Any | None = Field(default=None)
    init: Any | None = Field(default=None)
    is_root: bool | None = None
    metadata: dict[str, Any] | None = None
    path: str | None = None
    type: str | None = None
    subtype: str | None = None
    telemetry_labels: dict[str, str] | None = None


# Qualified path of the active span; pushed so nested spans can build child paths.
parent_path_context: ContextVar[str] = ContextVar('genkit_parent_path', default='')


def restore_default_telemetry_handlers() -> None:
    """Reset the chain to the two default handlers (conventions, then renderer)."""
    clear_genkit_telemetry_handlers()
    register_genkit_telemetry_handler(otel_ai_semantic_conventions)
    register_genkit_telemetry_handler(otel_renderer)


def init_telemetry() -> None:
    """Idempotent process setup: default handler chain + optional Dev UI export."""
    global initialized
    if initialized:
        return
    initialized = True
    restore_default_telemetry_handlers()
    if is_dev_environment():
        ensure_dev_telemetry()


def has_genkit_frame_above() -> bool:
    """True when at least one Genkit frame is already on the stack."""
    return bool(frame_stack.get())


def is_subtree_root() -> bool:
    """True when the current Genkit frame has no Genkit frame above it."""
    return len(frame_stack.get()) <= 1


def write_span_attr(key: str, value: Any) -> None:  # noqa: ANN401
    """Write one attr on the current Genkit frame and flush to the live span.

    Does not invoke annotate projectors (so projectors can write derived attrs
    without re-entering projection).
    """
    frame = current_frame()
    if frame is None:
        return
    frame.attrs[key] = value
    flush = annotate_flush.get()
    if flush is not None:
        flush(key, value)


def annotate(key: str, value: Any) -> None:  # noqa: ANN401
    """Buffer a span attribute on the current Genkit frame.

    Deep sites (interrupt / sessionId / snapshot / resumed) call this instead of
    touching OpenTelemetry. While a renderer has a span open it installs a flush
    callback so the write is also mirrored onto the live span for Dev UI.

    If an enrichment handler installed an annotate projector for this span
    (e.g. GenAI conventions), that runs after the write — annotate itself does
    not know about any particular convention.
    """
    if current_frame() is None:
        return
    write_span_attr(key, value)
    projector = annotate_projector.get()
    if projector is not None:
        projector()


def _to_json_attr(value: object) -> str:
    """Serialize an arbitrary object for an input/output span attribute."""
    if isinstance(value, BaseModel):
        return value.model_dump_json(by_alias=True, exclude_none=True)
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def annotate_output(value: Any) -> None:  # noqa: ANN401
    """Write ``genkit:output`` on the current span (JSON-serialized)."""
    annotate(Attr.OUTPUT, _to_json_attr(value))


def start_attributes(
    metadata: SpanMetadata,
    *,
    qualified_path: str,
) -> dict[str, Any]:
    """Attrs known when the span begins (identity/shape + input).

    Put these on ``params.attributes`` before the renderer opens the span so
    live on_start snapshots (Dev UI) already show input/path/type. State and
    output stay out — they aren't known until the body finishes; write those
    before end (renderer / annotate_output) and use ``annotate`` for mid-run
    facts. See module docstring "Attribute visibility timing".
    """
    attrs: dict[str, Any] = {}
    if metadata.telemetry_labels:
        attrs.update(metadata.telemetry_labels)
    attrs.update({
        Attr.NAME: metadata.name,
        Attr.PATH: qualified_path,
        Attr.QUALIFIED_PATH: qualified_path,
    })
    if metadata.type:
        attrs[Attr.TYPE] = metadata.type
    if metadata.subtype:
        attrs[Attr.SUBTYPE] = metadata.subtype
    if metadata.is_root:
        attrs[Attr.IS_ROOT] = True
    if metadata.metadata:
        for meta_key, meta_value in metadata.metadata.items():
            attrs[metadata_key(meta_key)] = str(meta_value)
    if metadata.input is not None:
        attrs[Attr.INPUT] = _to_json_attr(metadata.input)
    if metadata.init is not None:
        attrs[Attr.INIT] = _to_json_attr(metadata.init)
    return attrs


def span_kind_from_attrs(attrs: dict[str, Any]) -> str:
    """Model calls are client spans; everything else is internal."""
    return 'client' if is_model_span(attrs) else 'internal'


async def dispatch(params: SpanHookParams, fn: Callable[[], Awaitable[Any]]) -> Any:  # noqa: ANN401
    """Compose registered handlers into a chain and run it.

    Each handler's ``next_fn`` is the rest of the chain; ``fn`` is the innermost
    async continuation and runs exactly once. Results and exceptions flow back
    out through every handler.
    """
    chain: Callable[[], Awaitable[Any]] = fn
    for handler in reversed(get_telemetry_handlers()):
        next_fn = chain

        async def link(
            h: TelemetryHandler = handler,
            nxt: Callable[[], Awaitable[Any]] = next_fn,
        ) -> Any:  # noqa: ANN401
            return await h(params, nxt)

        chain = link
    return await chain()


@overload
async def run_in_new_span(
    name: SpanMetadata,
    attrs: Callable[[], Awaitable[T]],
    fn: None = None,
    *,
    links: list[Any] | None = None,
) -> T: ...


@overload
async def run_in_new_span(
    name: str,
    attrs: dict[str, Any],
    fn: Callable[[], Awaitable[T]],
    *,
    links: list[Any] | None = None,
) -> T: ...


async def run_in_new_span(
    name: str | SpanMetadata,
    attrs: dict[str, Any] | Callable[[], Awaitable[T]] | None = None,
    fn: Callable[[], Awaitable[T]] | None = None,
    *,
    links: list[Any] | None = None,
) -> Any:  # noqa: ANN401
    """Run ``fn`` inside a new Genkit span via the telemetry handler chain.

    ``fn`` must be async (return an awaitable). Two call shapes:

    * ``await run_in_new_span(name, attrs, fn)``
    * ``await run_in_new_span(metadata, fn)`` — builds attrs/path from SpanMetadata.
    """
    if isinstance(name, SpanMetadata):
        metadata = name
        body = attrs if callable(attrs) else fn
        if not callable(body):
            raise TypeError('run_in_new_span(metadata, fn) requires a callable fn')
        return await run_with_metadata(metadata, cast(Callable[[], Awaitable[Any]], body), links=links)

    if fn is None:
        raise TypeError('run_in_new_span(name, attrs, fn) requires a callable fn')
    if attrs is None or callable(attrs):
        raise TypeError('run_in_new_span(name, attrs, fn) requires an attrs dict')
    return await run_dispatch(name, attrs, fn, links=links)


async def run_with_metadata(
    metadata: SpanMetadata,
    fn: Callable[[], Awaitable[T]],
    *,
    links: list[Any] | None = None,
) -> T:
    if not has_genkit_frame_above() and metadata.is_root is None:
        metadata.is_root = True

    qualified_path = build_path(metadata.name, parent_path_context.get(), metadata.type or '', metadata.subtype)
    metadata.path = qualified_path
    span_attrs = start_attributes(metadata, qualified_path=qualified_path)
    if metadata.type:
        span_attrs[TYPE_FACT] = metadata.type

    # Parent path must stay set across await so nested run_in_new_span sees it.
    async def body_with_path() -> Any:  # noqa: ANN401
        token = parent_path_context.set(qualified_path)
        try:
            return await fn()
        finally:
            parent_path_context.reset(token)

    return await run_dispatch(metadata.name, span_attrs, body_with_path, links=links)


async def run_dispatch(
    name: str,
    attrs: dict[str, Any],
    fn: Callable[[], Awaitable[T]],
    *,
    links: list[Any] | None = None,
) -> T:
    if not has_genkit_frame_above() and Attr.IS_ROOT not in attrs:
        attrs[Attr.IS_ROOT] = True

    # Build params first: pydantic may copy ``attributes``, and the frame must
    # share that same dict so handler start enrichment is visible to annotate().
    params = SpanHookParams(
        name=name,
        attributes=attrs,
        kind=span_kind_from_attrs(attrs),
        links=links,
    )
    frame = SpanFrame(name=name, attrs=params.attributes)
    frame_token = frame_stack.set((*frame_stack.get(), frame))
    try:
        return await dispatch(params, fn)
    finally:
        frame_stack.reset(frame_token)
