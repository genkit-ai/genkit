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

"""Telemetry dispatcher. Composes Instrumentation providers; no OpenTelemetry."""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from typing import TypeVar

from ._instrumentation_api import Instrumentation, SpanContext, SpanMetadata

T = TypeVar('T')

instrumentations: list[Instrumentation] = []

# Active SpanContext so set_custom_metadata_attributes can reach it.
current_span: ContextVar[SpanContext | None] = ContextVar('genkit_span_context', default=None)


def describe_value(value: object) -> str:
    """Module-qualified name of a type or of an instance's type."""
    if isinstance(value, type):
        return f'type {value.__module__}.{value.__qualname__}'
    cls = type(value)
    return f'{cls.__module__}.{cls.__qualname__}'


def configure_instrumentation(instrumentation: Instrumentation) -> None:
    """Turn on a telemetry backend. Call before ``Genkit()`` to stack backends.

    Each provider wraps the next. ``OtelInstrumentation`` is the built-in.
    ``genkit start`` installs that one when a collector is configured.
    """
    # The class itself has run_in_new_span, so a forgotten () would pass a
    # Protocol check and then die inside OTel on the first span.
    if isinstance(instrumentation, type) or not isinstance(instrumentation, Instrumentation):
        raise TypeError(
            'configure_instrumentation expected an Instrumentation instance, got ' + describe_value(instrumentation)
        )
    instrumentations.append(instrumentation)


def reset_instrumentation() -> None:
    """Remove all providers. Tests and re-init."""
    instrumentations.clear()
    from ._otel_instrumentation import reset_developer_ui_collector

    reset_developer_ui_collector()


def is_instrumented_by(kind: type) -> bool:
    """True when a configured provider is an instance of ``kind``.

    Use ``is_instrumented_by(OtelInstrumentation)`` to see whether Genkit
    is already minting OpenTelemetry spans.
    """
    if not isinstance(kind, type):
        raise TypeError('is_instrumented_by expected a type, got ' + describe_value(kind))
    return any(isinstance(i, kind) for i in instrumentations)


def set_custom_metadata_attributes(attributes: Mapping[str, object]) -> None:
    """Write metadata on the active span. No-op outside a span."""
    span = current_span.get()
    if span is not None:
        span.set_metadata(attributes)


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

        async def nxt(span: SpanContext) -> T:
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

    def _first_non_empty(self, get: Callable[[SpanContext], str]) -> str:
        for span in self._spans:
            value = get(span)
            if value:
                return value
        return ''
