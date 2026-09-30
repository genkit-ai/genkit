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

"""Telemetry dispatcher and backend-agnostic types. No OpenTelemetry.

A new backend or exporter is another file in this folder. App code
keeps importing from ``genkit.telemetry``.
"""

from __future__ import annotations

import atexit
import contextlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from ._attrs import METADATA_PREFIX, Attr

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

SpanAttributeValue = str | bool | int | float
SpanState = Literal['success', 'error']


class SpanNext(Protocol[T_co]):
    """``next()`` or ``next(span)``. A logger that does not mint ids calls ``next()``."""

    def __call__(self, span: SpanContext | None = None) -> Awaitable[T_co]: ...


@dataclass(frozen=True)
class SpanMetadata:
    """Description of a span about to be created.

    ``action_type`` is the action kind (``'model'``, ``'flow'``, ``'tool.v2'``...)
    for action spans, or the span's own type (``'util'``, ``'flowStep'``...)
    for plain spans. Providers decide how to encode values.
    """

    name: str
    action_type: str | None = None
    input: object | None = None
    attributes: Mapping[str, SpanAttributeValue] = field(default_factory=dict)


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

    def set_state(self, state: SpanState) -> None:
        """Override genkit:state. ``error`` marks a failed generate that returned."""
        ...


@runtime_checkable
class Instrumentation(Protocol):
    """Pluggable provider. ``run_in_new_span`` wraps ``next`` like middleware."""

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: SpanNext[T],
    ) -> T: ...


@runtime_checkable
class DisposableInstrumentation(Protocol):
    """Optional: a provider that holds a subscription or client.

    ``reset_instrumentation`` calls ``dispose`` so an exporter from the
    last run cannot keep posting after tests tear down.
    """

    def dispose(self) -> None: ...


@runtime_checkable
class FlushableInstrumentation(Protocol):
    """Optional: a provider that buffers exports.

    ``flush_instrumentations`` calls ``flush`` so a trace is on the wire
    before the Developer UI asks for it.
    """

    def flush(self) -> None: ...


instrumentations: list[Instrumentation] = []

# Active SpanContext so set_custom_metadata_attributes can reach it.
current_span: ContextVar[SpanContext | None] = ContextVar('genkit_span_context', default=None)
parent_path_context: ContextVar[str] = ContextVar('genkit_parent_path', default='')

# an action of kind 'util' and a plain 'util' span look the same to a
# provider, but the Traces tab draws them differently. run_in_new_span
# sets this so the poster can tell them apart; providers never see it.
span_is_action: ContextVar[bool] = ContextVar('genkit_span_is_action', default=False)


def describe_value(value: object) -> str:
    """Module-qualified name of a type or of an instance's type."""
    if isinstance(value, type):
        return f'type {value.__module__}.{value.__qualname__}'
    cls = type(value)
    return f'{cls.__module__}.{cls.__qualname__}'


def to_json_attr(value: object) -> str:
    """Serialize an arbitrary object for an input/output span attribute."""
    if isinstance(value, BaseModel):
        return value.model_dump_json(by_alias=True, exclude_none=True)
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def start_attributes(
    metadata: SpanMetadata,
    *,
    qualified_path: str,
    is_action: bool = False,
) -> dict[str, Any]:
    """Attrs known when the span begins (identity/shape + input).

    Live-trace export snapshots the span the instant it starts, so these have to
    be on the span *before* start returns; otherwise Dev UI shows a blank
    in-progress entry until the span ends. State/output stay out — they aren't
    known until the body finishes.
    """
    labels = dict(metadata.attributes or {})
    init = labels.pop(Attr.INIT, None)
    custom = {k: labels.pop(k) for k in list(labels) if k.startswith(METADATA_PREFIX)}
    attrs: dict[str, Any] = dict(labels)
    attrs.update({
        Attr.NAME: metadata.name,
        Attr.PATH: qualified_path,
        Attr.QUALIFIED_PATH: qualified_path,
    })
    if is_action:
        attrs[Attr.TYPE] = 'action'
        if metadata.action_type:
            attrs[Attr.SUBTYPE] = metadata.action_type
    elif metadata.action_type:
        attrs[Attr.TYPE] = metadata.action_type
    attrs.update(custom)
    if metadata.input is not None:
        attrs[Attr.INPUT] = to_json_attr(metadata.input)
    if init is not None:
        attrs[Attr.INIT] = init
    return attrs


def configure_instrumentation(instrumentation: Instrumentation) -> None:
    """Turn on a telemetry backend. Call before ``Genkit()`` to stack backends.

    Each provider wraps the next. ``genkit start`` installs the Developer UI
    HTTP poster when a collector URL is set. ``enable_google_cloud_telemetry()``
    hangs Cloud's exporter; it does not mint spans.
    """
    # The class itself has run_in_new_span, so a forgotten () would pass a
    # Protocol check and then fail on the first span.
    if isinstance(instrumentation, type) or not isinstance(instrumentation, Instrumentation):
        raise TypeError(
            'configure_instrumentation expected an Instrumentation instance, got ' + describe_value(instrumentation)
        )
    instrumentations.append(instrumentation)


def dispose_instrumentations() -> None:
    """Release provider resources. Safe to call more than once."""
    for inst in instrumentations:
        if isinstance(inst, DisposableInstrumentation):
            inst.dispose()


atexit.register(dispose_instrumentations)


def flush_instrumentations() -> None:
    """Wait for in-flight exports on every configured backend."""
    for inst in instrumentations:
        if isinstance(inst, FlushableInstrumentation):
            inst.flush()


def reset_instrumentation() -> None:
    """Remove all providers. Tests and re-init."""
    dispose_instrumentations()
    instrumentations.clear()


def is_instrumented_by(kind: type) -> bool:
    """True when a configured provider is an instance of ``kind``.

    Use ``is_instrumented_by(GenkitBuiltinInstrumentation)`` for the
    Developer UI poster.
    """
    if not isinstance(kind, type):
        raise TypeError('is_instrumented_by expected a type, got ' + describe_value(kind))
    return any(isinstance(i, kind) for i in instrumentations)


def set_custom_metadata_attributes(attributes: Mapping[str, object]) -> None:
    """Write metadata on the active span. No-op outside a span."""
    span = current_span.get()
    if span is not None:
        span.set_metadata(attributes)


def set_span_state(state: SpanState) -> None:
    """Write genkit:state on the active span. No-op outside a span."""
    span = current_span.get()
    if span is not None:
        span.set_state(state)


async def run_in_new_span(
    name: str,
    fn: Callable[[SpanContext], Awaitable[T]],
    *,
    action_type: str | None = None,
    input: object | None = None,
    attributes: Mapping[str, SpanAttributeValue] | None = None,
    is_action: bool = False,
) -> T:
    """Run ``fn`` inside a new span via the configured provider chain.

    No providers → ``fn`` runs with a no-op span (empty ids). Index 0 is
    outermost. The provider list is snapshotted so configure/reset during an
    await cannot break the chain.
    """
    if not inspect.iscoroutinefunction(fn):
        name = getattr(fn, '__qualname__', type(fn).__name__)
        raise TypeError(f'run_in_new_span expected an async callback, got {name}')
    attrs: dict[str, SpanAttributeValue] = dict(attributes) if attributes else {}
    meta = SpanMetadata(
        name=name,
        action_type=action_type,
        input=input,
        attributes=attrs,
    )
    providers = list(instrumentations)
    if not providers:
        return await _run_with_span(NoopSpanContext(), fn)

    spans: list[SpanContext] = []

    async def build(index: int) -> T:
        if index == len(providers):
            return await _run_with_span(CompositeSpanContext(spans), fn)

        async def nxt(span: SpanContext | None = None) -> T:
            if span is not None:
                spans.append(span)
            return await build(index + 1)

        return await providers[index].run_in_new_span(meta, nxt)

    token = span_is_action.set(is_action)
    try:
        return await build(0)
    finally:
        span_is_action.reset(token)


async def _run_with_span(
    span: SpanContext,
    fn: Callable[[SpanContext], Awaitable[T]],
) -> T:
    token = current_span.set(span)
    try:
        return await fn(span)
    finally:
        current_span.reset(token)


class NoopSpanContext:
    """Span used when nothing is configured."""

    @property
    def trace_id(self) -> str:
        return ''

    @property
    def span_id(self) -> str:
        return ''

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        return

    def set_state(self, state: SpanState) -> None:
        return


class CompositeSpanContext:
    """Fans metadata/state to every provider; ids are first non-empty."""

    def __init__(self, spans: list[SpanContext]) -> None:
        self._spans = spans

    @property
    def trace_id(self) -> str:
        return self._first_non_empty(lambda s: s.trace_id)

    @property
    def span_id(self) -> str:
        return self._first_non_empty(lambda s: s.span_id)

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        for span in self._spans:
            with contextlib.suppress(Exception):
                span.set_metadata(metadata)

    def set_state(self, state: SpanState) -> None:
        for span in self._spans:
            with contextlib.suppress(Exception):
                span.set_state(state)

    def _first_non_empty(self, get: Callable[[SpanContext], str]) -> str:
        for span in self._spans:
            value = get(span)
            if value:
                return value
        return ''
