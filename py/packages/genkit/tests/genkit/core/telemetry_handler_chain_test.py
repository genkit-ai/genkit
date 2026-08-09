#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Handler-chain telemetry tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from typing import Any

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genkit._core._telemetry_contract import SpanHookParams
from genkit._core._telemetry_handlers import (
    ensure_dev_telemetry,
    otel_ai_semantic_conventions,
    otel_renderer,
    start_genkit_span,
)
from genkit._core._tracing import (
    clear_genkit_telemetry_handlers,
    is_subtree_root,
    register_genkit_telemetry_handler,
    restore_default_telemetry_handlers,
    run_in_new_span,
)


@pytest.fixture(autouse=True)
def _reset_handlers() -> Generator[None, None, None]:
    restore_default_telemetry_handlers()
    try:
        yield
    finally:
        restore_default_telemetry_handlers()


@pytest.fixture
def exporter() -> Generator[InMemorySpanExporter, None, None]:
    provider = trace_api.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace_api.set_tracer_provider(provider)
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    try:
        yield exp
    finally:
        exp.clear()


@pytest.mark.asyncio
async def test_fn_runs_once_under_three_handlers() -> None:
    calls: list[str] = []

    async def h1(params: SpanHookParams, next_fn: Callable[[], Awaitable[Any]]) -> Any:
        calls.append('h1-enter')
        out = await next_fn()
        calls.append('h1-exit')
        return out

    async def h2(params: SpanHookParams, next_fn: Callable[[], Awaitable[Any]]) -> Any:
        calls.append('h2-enter')
        out = await next_fn()
        calls.append('h2-exit')
        return out

    async def h3(params: SpanHookParams, next_fn: Callable[[], Awaitable[Any]]) -> Any:
        calls.append('h3-enter')
        out = await next_fn()
        calls.append('h3-exit')
        return out

    clear_genkit_telemetry_handlers()
    register_genkit_telemetry_handler(h1)
    register_genkit_telemetry_handler(h2)
    register_genkit_telemetry_handler(h3)

    fn_calls = {'n': 0}

    async def fn() -> str:
        fn_calls['n'] += 1
        calls.append('fn')
        return 'ok'

    assert await run_in_new_span('work', {'genkit.type': 'util'}, fn) == 'ok'
    assert fn_calls['n'] == 1
    assert calls == [
        'h1-enter',
        'h2-enter',
        'h3-enter',
        'fn',
        'h3-exit',
        'h2-exit',
        'h1-exit',
    ]


@pytest.mark.asyncio
async def test_flow_model_nesting_one_trace_parent_and_transformer_attrs(
    exporter: InMemorySpanExporter,
) -> None:
    async def model_work() -> str:
        return 'reply'

    async def flow_work() -> str:
        return await run_in_new_span(
            'gemini',
            {'genkit.type': 'model', 'model': 'gemini-2.0-flash'},
            model_work,
        )

    assert await run_in_new_span('myFlow', {'genkit.type': 'flow'}, flow_work) == 'reply'

    spans = list(exporter.get_finished_spans())
    assert len(spans) == 2
    by_name = {s.name: s for s in spans}
    flow = by_name['myFlow']
    model = by_name['chat gemini-2.0-flash']

    assert flow.context.trace_id == model.context.trace_id
    assert model.parent is not None
    assert model.parent.span_id == flow.context.span_id

    flow_attrs = dict(flow.attributes or {})
    model_attrs = dict(model.attributes or {})
    assert flow_attrs['gen_ai.operation.name'] == 'invoke_workflow'
    assert flow_attrs['gen_ai.workflow.name'] == 'myFlow'
    assert model_attrs['gen_ai.operation.name'] == 'chat'
    assert model_attrs['gen_ai.request.model'] == 'gemini-2.0-flash'
    assert model.kind == trace_api.SpanKind.CLIENT
    assert flow.kind == trace_api.SpanKind.INTERNAL


@pytest.mark.asyncio
async def test_error_propagates_and_is_recorded(exporter: InMemorySpanExporter) -> None:
    async def boom() -> None:
        raise ValueError('kaboom')

    with pytest.raises(ValueError, match='kaboom'):
        await run_in_new_span('broken', {'genkit.type': 'util'}, boom)

    spans = list(exporter.get_finished_spans())
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == trace_api.StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs['error.type'] == 'ValueError'
    assert any(e.name == 'exception' for e in span.events)


@pytest.mark.asyncio
async def test_mid_execution_annotation_lands_on_current_span(exporter: InMemorySpanExporter) -> None:
    from genkit._core._tracing import annotate, current_frame

    async def work() -> None:
        annotate('sessionId', 'sess-1')
        frame = current_frame()
        assert frame is not None
        assert frame.attrs['sessionId'] == 'sess-1'

    await run_in_new_span('tagged', {'genkit.type': 'util'}, work)

    span = exporter.get_finished_spans()[-1]
    assert dict(span.attributes or {})['sessionId'] == 'sess-1'


@pytest.mark.asyncio
async def test_two_renderers_produce_two_spans(exporter: InMemorySpanExporter) -> None:
    """Documents the one-renderer rule: two renderers ⇒ two nested spans."""

    async def second_renderer(
        params: SpanHookParams,
        next_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        with start_genkit_span(f'double-{params.name}', params.attributes):
            return await next_fn()

    clear_genkit_telemetry_handlers()
    register_genkit_telemetry_handler(otel_ai_semantic_conventions)
    register_genkit_telemetry_handler(otel_renderer)
    register_genkit_telemetry_handler(second_renderer)

    async def work() -> str:
        return 'x'

    await run_in_new_span('once', {'genkit.type': 'util'}, work)

    names = sorted(s.name for s in exporter.get_finished_spans())
    assert names == ['double-once', 'once']


@pytest.mark.asyncio
async def test_subtree_root_true_and_false() -> None:
    seen: list[bool] = []

    async def inner() -> None:
        seen.append(is_subtree_root())

    async def outer() -> None:
        seen.append(is_subtree_root())
        await run_in_new_span('inner', {'genkit.type': 'util'}, inner)

    clear_genkit_telemetry_handlers()  # no OTel needed for frame stack
    await run_in_new_span('outer', {'genkit.type': 'flow'}, outer)

    assert seen == [True, False]


@pytest.mark.xfail(reason='TODO(design): span-end ownership for streams — generator escapes renderer with-block')
@pytest.mark.asyncio
async def test_streaming_span_covers_chunks(exporter: InMemorySpanExporter) -> None:
    """Placeholder: today the span ends before the first chunk is consumed."""

    async def stream_fn() -> Generator[int, None, None]:
        def gen() -> Generator[int, None, None]:
            span_during_fn = trace_api.get_current_span()
            assert span_during_fn.is_recording()
            yield 1
            # After the handler chain returns the generator, the renderer's with
            # block has already exited — so this fails with current ownership.
            assert trace_api.get_current_span().is_recording()
            yield 2

        return gen()

    gen = await run_in_new_span('stream', {'genkit.type': 'util'}, stream_fn)
    chunks = list(gen)
    assert chunks == [1, 2]
    spans: list[ReadableSpan] = list(exporter.get_finished_spans())
    assert len(spans) == 1
    # Desired: span still open across chunk iteration (not true today).
    assert spans[0].status.status_code != trace_api.StatusCode.ERROR


def test_ensure_dev_piggybacks_on_existing_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider registered before Genkit must keep ownership; we only attach."""
    import genkit._core._telemetry_handlers as handlers

    monkeypatch.setenv('GENKIT_ENV', 'dev')
    monkeypatch.setattr(handlers, 'dev_telemetry_installed', False)
    monkeypatch.setattr(handlers, 'init_telemetry_server_exporter', lambda: InMemorySpanExporter())

    provider = trace_api.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace_api.set_tracer_provider(provider)
    assert not handlers.is_placeholder_provider(provider)

    calls = {'n': 0}
    original = provider.add_span_processor

    def counting(processor: object) -> None:
        calls['n'] += 1
        original(processor)

    monkeypatch.setattr(provider, 'add_span_processor', counting)

    ensure_dev_telemetry()
    ensure_dev_telemetry()
    assert calls['n'] == 1
    assert handlers.dev_telemetry_installed is True
