#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What configure_instrumentation and genkit start do to trace ids."""

from __future__ import annotations

import asyncio
import os
import subprocess  # noqa: S404
import sys
from collections.abc import Awaitable, Callable, Generator
from typing import Any, TypeVar

import pytest
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genkit import Genkit
from genkit._core._action import Action
from genkit._core._environment import GENKIT_ENV
from genkit._core._reflection import create_reflection_asgi_app
from genkit._core._registry import Registry
from genkit._core._telemetry._instrumentation import (
    NoopSpanContext,
    SpanContext,
    SpanMetadata,
    instrumentations,
    is_instrumented_by,
    parent_path_context,
    reset_instrumentation,
    run_in_new_span,
    set_custom_metadata_attributes,
    set_span_state,
)
from genkit._core._telemetry._log_exporter import reset_log_export
from genkit._core._telemetry.http import GenkitBuiltinInstrumentation
from genkit.plugin_api import ActionKind
from genkit.telemetry import configure_instrumentation

T = TypeVar('T')


def _hex_id(value: str, length: int) -> bool:
    return len(value) == length and all(c in '0123456789abcdef' for c in value)


async def _joke() -> str:
    return 'Why did the cat cross the road?'


def _hang_exporter(exporter: InMemorySpanExporter) -> None:
    provider = trace_api.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))


@pytest.fixture(autouse=True)
def _isolate_telemetry(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Each test starts with no providers, unset collector env, and its own tracer."""
    reset_instrumentation()
    reset_log_export()
    monkeypatch.delenv(GENKIT_ENV, raising=False)
    monkeypatch.delenv('GENKIT_TELEMETRY_SERVER', raising=False)
    monkeypatch.setattr(Genkit, '_start_reflection_background', lambda self: None)
    isolated = TracerProvider()
    monkeypatch.setattr(trace_api, 'get_tracer_provider', lambda: isolated)
    monkeypatch.setattr(trace_api, 'set_tracer_provider', lambda _provider: None)
    path_token = parent_path_context.set('')
    try:
        yield
    finally:
        parent_path_context.reset(path_token)
        reset_instrumentation()
        reset_log_export()
        isolated.shutdown()


@pytest.mark.asyncio
async def test_a_plain_script_returns_an_answer_and_no_trace_ids() -> None:
    """Genkit() with no telemetry configured still runs; ids stay empty."""
    Genkit()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert result.response == 'Why did the cat cross the road?'
    assert result.trace_id == ''
    assert result.span_id == ''
    assert not is_instrumented_by(GenkitBuiltinInstrumentation)


@pytest.mark.asyncio
async def test_genkit_start_gives_the_developer_ui_real_trace_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under genkit start, Genkit() POSTs traces so the Traces tab gets real ids."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    Genkit()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert is_instrumented_by(GenkitBuiltinInstrumentation)
    assert _hex_id(result.trace_id, 32)
    assert _hex_id(result.span_id, 16)


@pytest.mark.asyncio
async def test_configuring_a_backend_yourself_in_dev_still_adds_the_ui_poster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """configure_instrumentation in dev still adds the UI poster."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    class AlreadyOn:
        async def run_in_new_span(self, metadata: SpanMetadata, next: Callable[..., Awaitable[T]]) -> T:
            return await next()

    yours = AlreadyOn()
    configure_instrumentation(yours)
    Genkit()

    assert len(instrumentations) == 2
    assert instrumentations[0] == yours
    assert isinstance(instrumentations[1], GenkitBuiltinInstrumentation)


@pytest.mark.asyncio
async def test_an_exporter_on_the_process_tracer_does_not_create_spans() -> None:
    """An exporter on the process tracer without configure leaves ids empty."""
    exporter = InMemorySpanExporter()
    _hang_exporter(exporter)
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert result.trace_id == ''
    assert not exporter.get_finished_spans()


@pytest.mark.asyncio
async def test_dev_without_a_collector_stays_untraced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GENKIT_ENV=dev with no collector: empty ids. Stop in the Developer UI will not find the run."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')

    Genkit()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert result.trace_id == ''
    assert result.span_id == ''


@pytest.mark.asyncio
async def test_hanging_an_exporter_after_genkit_does_not_turn_tracing_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hanging an exporter after Genkit() does not turn tracing on; ids stay empty."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')

    Genkit()
    exporter = InMemorySpanExporter()
    _hang_exporter(exporter)
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert result.trace_id == ''
    assert not exporter.get_finished_spans()


def test_importing_genkit_does_not_start_a_tracer() -> None:
    """from genkit import Genkit does not start a tracer or install a provider."""
    script = """
from opentelemetry import trace
from genkit import Genkit  # noqa: F401
from opentelemetry.trace import ProxyTracerProvider, NoOpTracerProvider

provider = trace.get_tracer_provider()
assert isinstance(provider, (ProxyTracerProvider, NoOpTracerProvider))
"""
    env = {k: v for k, v in os.environ.items() if k not in {GENKIT_ENV, 'GENKIT_TELEMETRY_SERVER'}}
    completed = subprocess.run(  # noqa: S603
        [sys.executable, '-c', script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.asyncio
async def test_production_unconfigured_genkit_does_not_leak_into_host_otel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host app global OTel captures app spans, but unconfigured Genkit leaks 0 spans."""
    host_exporter = InMemorySpanExporter()
    host_provider = TracerProvider()
    host_provider.add_span_processor(SimpleSpanProcessor(host_exporter))
    monkeypatch.setattr(trace_api, 'get_tracer_provider', lambda: host_provider)

    tracer = host_provider.get_tracer('app')
    with tracer.start_as_current_span('http.request'):
        pass

    Genkit()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    host_provider.force_flush()
    span_names = [s.name for s in host_exporter.get_finished_spans()]
    assert 'http.request' in span_names
    assert 'joke' not in span_names
    assert result.trace_id == ''


@pytest.mark.asyncio
async def test_dev_mode_never_claims_or_mutates_global_tracer_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dev instrumentation never mutates global OTel."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    global_calls: list[object] = []
    monkeypatch.setattr(trace_api, 'set_tracer_provider', lambda p: global_calls.append(p))

    Genkit()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert _hex_id(result.trace_id, 32)
    assert global_calls == []


@pytest.mark.asyncio
async def test_nested_flow_and_step_share_trace_id(exporter) -> None:
    """Nested actions maintain the same trace_id and different span ids."""

    async def step_fn() -> str:
        return 'step_ok'

    step_action = Action(name='stepAction', kind=ActionKind.UTIL, fn=step_fn)

    async def flow_fn() -> str:
        res = await step_action.run()
        return f'flow_{res.response}'

    flow_action = Action(name='flowAction', kind=ActionKind.FLOW, fn=flow_fn)
    result = await flow_action.run()

    spans = exporter.get_finished_spans()
    flow_span = next(s for s in spans if s.name == 'flowAction')
    step_span = next(s for s in spans if s.name == 'stepAction')

    assert flow_span.trace_id == step_span.trace_id
    assert flow_span.trace_id == result.trace_id
    assert step_span.parent_span_id == flow_span.span_id


@pytest.mark.asyncio
async def test_concurrent_flows_maintain_independent_trace_trees(exporter) -> None:
    """Concurrent flows in asyncio.gather keep independent trace IDs."""

    async def flow1_fn() -> str:
        await asyncio.sleep(0.01)
        return 'done_1'

    async def flow2_fn() -> str:
        await asyncio.sleep(0.01)
        return 'done_2'

    action1 = Action(name='flow1', kind=ActionKind.FLOW, fn=flow1_fn)
    action2 = Action(name='flow2', kind=ActionKind.FLOW, fn=flow2_fn)

    res1, res2 = await asyncio.gather(action1.run(), action2.run())

    assert _hex_id(res1.trace_id, 32)
    assert _hex_id(res2.trace_id, 32)
    assert res1.trace_id != res2.trace_id

    spans = exporter.get_finished_spans()
    span1 = next(s for s in spans if s.name == 'flow1')
    span2 = next(s for s in spans if s.name == 'flow2')
    assert span1.trace_id != span2.trace_id


@pytest.mark.asyncio
async def test_run_in_new_span_snapshots_providers_against_concurrent_mutation(exporter) -> None:
    """Mutating instrumentations while a span runs does not affect the active span."""

    async def body(_span: object) -> str:
        reset_instrumentation()
        return 'mutated'

    res = await run_in_new_span('in_flight', body, action_type='flow')
    assert res == 'mutated'

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == 'in_flight'


@pytest.mark.asyncio
async def test_set_custom_metadata_stamps_the_active_span(exporter) -> None:
    """set_custom_metadata_attributes writes onto the active span."""

    async def flow_fn() -> str:
        set_custom_metadata_attributes({'user_id': 'user_42', 'tier': 'enterprise'})
        return 'ok'

    action = Action(name='metaFlow', kind=ActionKind.FLOW, fn=flow_fn)
    await action.run()

    span = next(s for s in exporter.get_finished_spans() if s.name == 'metaFlow')
    assert span.attributes['genkit:metadata:user_id'] == 'user_42'
    assert span.attributes['genkit:metadata:tier'] == 'enterprise'


def test_set_custom_metadata_is_noop_outside_action() -> None:
    """Calling set_custom_metadata_attributes outside an action does not raise."""
    set_custom_metadata_attributes({'some': 'value'})


def test_set_span_state_is_noop_outside_action() -> None:
    """Calling set_span_state outside an action does not raise."""
    set_span_state('error')


@pytest.mark.asyncio
async def test_failing_custom_provider_does_not_break_other_providers(exporter) -> None:
    """A throwing custom provider span does not crash other providers or action execution."""

    class BrokenSpan(NoopSpanContext):
        def set_metadata(self, metadata: object) -> None:
            raise RuntimeError('custom provider metadata crash')

    class BrokenInstrumentation:
        async def run_in_new_span(
            self,
            metadata: SpanMetadata,
            next: Callable[[SpanContext], Awaitable[T]],
        ) -> T:
            return await next(BrokenSpan())

    configure_instrumentation(BrokenInstrumentation())

    async def flow_fn() -> str:
        set_custom_metadata_attributes({'custom': 'val'})
        return 'success'

    action = Action(name='failingProviderFlow', kind=ActionKind.FLOW, fn=flow_fn)
    result = await action.run()

    assert result.response == 'success'
    span = next(s for s in exporter.get_finished_spans() if s.name == 'failingProviderFlow')
    assert span.attributes['genkit:metadata:custom'] == 'val'


@pytest.mark.asyncio
async def test_logger_provider_can_skip_span_ids() -> None:
    """A provider that only logs can call next() and leave ids empty."""

    class PrintingInstrumentation:
        async def run_in_new_span(
            self,
            metadata: SpanMetadata,
            next: Callable[..., Awaitable[T]],
        ) -> T:
            return await next()

    configure_instrumentation(PrintingInstrumentation())
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert result.response == 'Why did the cat cross the road?'
    assert result.trace_id == ''
    assert result.span_id == ''


@pytest.mark.asyncio
async def test_uninstrumented_run_action_omits_telemetry_payload() -> None:
    """v1 runAction omits telemetry when ids are empty so a blank id is not a broken exporter."""
    registry = Registry()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    registry.register_action_from_instance(action)
    app = create_reflection_asgi_app(registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post('/api/runAction', json={'key': '/flow/joke'})
    assert response.status_code == 200
    body = response.json()
    assert 'telemetry' not in body
    assert 'X-Genkit-Trace-Id' not in response.headers
    assert 'X-Genkit-Span-Id' not in response.headers


@pytest.mark.asyncio
async def test_dev_collector_nested_actions_share_trace_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested actions under genkit start keep one hex trace id."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')
    Genkit()

    async def step_fn() -> str:
        return 'ok'

    step = Action(name='step', kind=ActionKind.UTIL, fn=step_fn)
    inner_from_flow: dict[str, Any] = {}

    async def flow_fn() -> str:
        inner = await step.run()
        inner_from_flow['inner'] = inner
        return inner.response

    flow = Action(name='flow', kind=ActionKind.FLOW, fn=flow_fn)
    outer = await flow.run()
    inner = inner_from_flow['inner']

    assert _hex_id(outer.trace_id, 32)
    assert _hex_id(outer.span_id, 16)
    assert _hex_id(inner.trace_id, 32)
    assert _hex_id(inner.span_id, 16)
    assert inner.trace_id == outer.trace_id
    assert inner.span_id != outer.span_id
