#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Dispatcher tests for pluggable Instrumentation providers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

import pytest

from genkit.telemetry import (
    SpanContext,
    SpanMetadata,
    configure_instrumentation,
    is_instrumented_by,
    reset_instrumentation,
    run_in_new_span,
    set_custom_metadata_attributes,
)


class RecordedSpan:
    def __init__(self, label: str, *, trace_id: str = '', span_id: str = '') -> None:
        self.label = label
        self.trace_id = trace_id
        self.span_id = span_id
        self.metadata: list[Mapping[str, object]] = []
        self.outputs: list[object] = []

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        self.metadata.append(metadata)

    def set_output(self, value: object) -> None:
        self.outputs.append(value)


class FakeInstrumentation:
    def __init__(
        self,
        label: str,
        log: list[str],
        *,
        trace_id: str = '',
        span_id: str = '',
    ) -> None:
        self.label = label
        self.log = log
        self.trace_id = trace_id
        self.span_id = span_id
        self.spans: list[RecordedSpan] = []

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: Callable[[SpanContext], Awaitable[object]],
    ) -> object:
        self.log.append(f'enter:{self.label}')
        span = RecordedSpan(self.label, trace_id=self.trace_id, span_id=self.span_id)
        self.spans.append(span)
        try:
            return await next(span)
        finally:
            self.log.append(f'exit:{self.label}')


@pytest.fixture(autouse=True)
def _reset() -> object:
    reset_instrumentation()
    yield
    reset_instrumentation()


@pytest.mark.asyncio
async def test_noop_span_when_nothing_configured() -> None:
    seen: SpanContext | None = None

    async def body(span: SpanContext) -> str:
        nonlocal seen
        seen = span
        return 'ok'

    result = await run_in_new_span('op', body)
    assert result == 'ok'
    assert seen is not None
    assert seen.trace_id == ''
    assert seen.span_id == ''
    seen.set_metadata({'k': 'v'})
    seen.set_output('x')


@pytest.mark.asyncio
async def test_composes_providers_in_registration_order() -> None:
    log: list[str] = []
    configure_instrumentation(FakeInstrumentation('a', log))
    configure_instrumentation(FakeInstrumentation('b', log))

    async def body(_span: SpanContext) -> str:
        log.append('body')
        return 'x'

    await run_in_new_span('op', body)
    assert log == ['enter:a', 'enter:b', 'body', 'exit:b', 'exit:a']


@pytest.mark.asyncio
async def test_in_flight_span_keeps_the_provider_list_it_started_with() -> None:
    log: list[str] = []
    configure_instrumentation(FakeInstrumentation('a', log))
    configure_instrumentation(FakeInstrumentation('b', log))

    async def body(_span: SpanContext) -> None:
        reset_instrumentation()
        configure_instrumentation(FakeInstrumentation('c', log))
        log.append('body')

    await run_in_new_span('op', body)
    assert log == ['enter:a', 'enter:b', 'body', 'exit:b', 'exit:a']
    assert 'enter:c' not in log


@pytest.mark.asyncio
async def test_set_custom_metadata_fans_out() -> None:
    log: list[str] = []
    a = FakeInstrumentation('a', log)
    b = FakeInstrumentation('b', log)
    configure_instrumentation(a)
    configure_instrumentation(b)

    async def body(_span: SpanContext) -> None:
        set_custom_metadata_attributes({'hello': 'world'})

    await run_in_new_span('op', body)
    assert a.spans[0].metadata == [{'hello': 'world'}]
    assert b.spans[0].metadata == [{'hello': 'world'}]


@pytest.mark.asyncio
async def test_set_output_fans_out() -> None:
    log: list[str] = []
    a = FakeInstrumentation('a', log)
    b = FakeInstrumentation('b', log)
    configure_instrumentation(a)
    configure_instrumentation(b)

    async def body(span: SpanContext) -> None:
        span.set_output({'answer': 42})

    await run_in_new_span('op', body)
    assert a.spans[0].outputs == [{'answer': 42}]
    assert b.spans[0].outputs == [{'answer': 42}]


def test_configure_rejects_junk_at_the_boundary() -> None:
    with pytest.raises(TypeError, match='Instrumentation instance'):
        configure_instrumentation(object())  # type: ignore[arg-type]


def test_otel_instrumentation_rejects_junk_tracer_provider() -> None:
    from genkit.telemetry import OtelInstrumentation

    with pytest.raises(TypeError, match='TracerProvider'):
        OtelInstrumentation(tracer_provider=object())  # type: ignore[arg-type]


def test_configure_rejects_the_provider_class() -> None:
    from genkit.telemetry import OtelInstrumentation

    with pytest.raises(
        TypeError,
        match='type genkit._core._otel_instrumentation.OtelInstrumentation',
    ):
        configure_instrumentation(OtelInstrumentation)  # type: ignore[arg-type]


def test_is_instrumented_by_rejects_junk_at_the_boundary() -> None:
    with pytest.raises(TypeError, match='builtins.str'):
        is_instrumented_by('OtelInstrumentation')  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_run_in_new_span_rejects_a_sync_body() -> None:
    def sync_body(_span: SpanContext) -> str:
        return 'ok'

    with pytest.raises(TypeError, match='sync_body'):
        await run_in_new_span('op', sync_body)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_ids_resolve_to_first_non_empty() -> None:
    log: list[str] = []
    configure_instrumentation(FakeInstrumentation('a', log))
    configure_instrumentation(FakeInstrumentation('b', log, trace_id='trace-b', span_id='span-b'))

    seen_trace = ''
    seen_span = ''

    async def body(span: SpanContext) -> None:
        nonlocal seen_trace, seen_span
        seen_trace = span.trace_id
        seen_span = span.span_id

    await run_in_new_span('op', body)
    assert seen_trace == 'trace-b'
    assert seen_span == 'span-b'


@pytest.mark.asyncio
async def test_errors_still_exit_each_provider() -> None:
    log: list[str] = []
    configure_instrumentation(FakeInstrumentation('a', log))
    configure_instrumentation(FakeInstrumentation('b', log))

    async def body(_span: SpanContext) -> None:
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await run_in_new_span('op', body)

    assert log == ['enter:a', 'enter:b', 'exit:b', 'exit:a']


@pytest.mark.asyncio
async def test_action_without_instrumentation_has_empty_ids() -> None:
    from genkit import ActionKind
    from genkit._core._action import Action

    async def noop() -> str:
        return 'ok'

    action = Action(name='plain', kind=ActionKind.CUSTOM, fn=noop)
    result = await action.run()
    assert result.response == 'ok'
    assert result.trace_id == ''
    assert result.span_id == ''
