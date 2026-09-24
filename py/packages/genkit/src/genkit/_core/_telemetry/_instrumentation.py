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

import contextlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from ._attrs import Attr, metadata_key

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


class SpanNext(Protocol[T_co]):
    """``next()`` or ``next(span)``. A logger that does not mint ids calls ``next()``."""

    def __call__(self, span: SpanContext | None = None) -> Awaitable[T_co]: ...


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

    def set_state(self, state: str) -> None:
        """Override genkit:state (e.g. State.ERROR)."""
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

    ``reset_instrumentation`` calls ``dispose`` so a log handler from the
    last run cannot keep posting after tests tear down.
    """

    def dispose(self) -> None: ...


instrumentations: list[Instrumentation] = []

# Active SpanContext so set_custom_metadata_attributes can reach it.
current_span: ContextVar[SpanContext | None] = ContextVar('genkit_span_context', default=None)
parent_path_context: ContextVar[str] = ContextVar('genkit_parent_path', default='')
# Prompt playground re-renders on each edit. Those runs are not real traces —
# the flag stays set for the whole action so nested model/tool spans stay
# off the Traces tab too.
suppress_telemetry: ContextVar[bool] = ContextVar('suppress_telemetry', default=False)


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
) -> dict[str, Any]:
    """Attrs known when the span begins (identity/shape + input).

    Live-trace export snapshots the span the instant it starts, so these have to
    be on the span *before* start returns; otherwise Dev UI shows a blank
    in-progress entry until the span ends. State/output stay out — they aren't
    known until the body finishes.
    """
    attrs: dict[str, Any] = {}
    if metadata.attributes:
        attrs.update(metadata.attributes)
    attrs.update({
        Attr.NAME: metadata.name,
        Attr.PATH: qualified_path,
        Attr.QUALIFIED_PATH: qualified_path,
    })
    if metadata.action_type:
        attrs[Attr.TYPE] = metadata.action_type
    if metadata.subtype:
        attrs[Attr.SUBTYPE] = metadata.subtype
    if metadata.is_root:
        attrs[Attr.IS_ROOT] = True
    if metadata.metadata:
        for meta_key, meta_value in metadata.metadata.items():
            attrs[metadata_key(meta_key)] = str(meta_value)
    if metadata.input is not None:
        attrs[Attr.INPUT] = to_json_attr(metadata.input)
    if metadata.init is not None:
        attrs[Attr.INIT] = to_json_attr(metadata.init)
    return attrs


def configure_instrumentation(instrumentation: Instrumentation) -> None:
    """Turn on a telemetry backend. Call before ``Genkit()`` to stack backends.

    Each provider wraps the next. ``genkit start`` installs the Developer UI
    HTTP poster when a collector URL is set. ``enable_google_cloud_telemetry()``
    hangs Cloud's exporter; it does not mint spans.
    """
    # The class itself has run_in_new_span, so a forgotten () would pass a
    # Protocol check and then die inside OTel on the first span.
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


def flush_instrumentations() -> None:
    """Wait for in-flight exports on every configured backend."""
    for inst in instrumentations:
        flush = getattr(inst, 'flush', None)
        if callable(flush):
            flush()


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


def set_span_state(state: str) -> None:
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
    attributes: Mapping[str, str] | None = None,
    subtype: str | None = None,
    init: object | None = None,
    metadata: Mapping[str, object] | None = None,
    is_root: bool | None = None,
) -> T:
    """Run ``fn`` inside a new span via the configured provider chain.

    No providers → ``fn`` runs with a no-op span (empty ids). Index 0 is
    outermost. The provider list is snapshotted so configure/reset during an
    await cannot break the chain.
    """
    if not inspect.iscoroutinefunction(fn):
        name = getattr(fn, '__qualname__', type(fn).__name__)
        raise TypeError(f'run_in_new_span expected an async callback, got {name}')
    meta = SpanMetadata(
        name=name,
        action_type=action_type,
        input=input,
        attributes=dict(attributes) if attributes else {},
        subtype=subtype,
        init=init,
        metadata=metadata,
        is_root=is_root,
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

    return await build(0)


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

    def set_output(self, value: object) -> None:
        return

    def set_state(self, state: str) -> None:
        return


class CompositeSpanContext:
    """Fans metadata/output to every provider; ids are first non-empty."""

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

    def set_output(self, value: object) -> None:
        for span in self._spans:
            with contextlib.suppress(Exception):
                span.set_output(value)

    def set_state(self, state: str) -> None:
        for span in self._spans:
            with contextlib.suppress(Exception):
                span.set_state(state)

    def _first_non_empty(self, get: Callable[[SpanContext], str]) -> str:
        for span in self._spans:
            value = get(span)
            if value:
                return value
        return ''
