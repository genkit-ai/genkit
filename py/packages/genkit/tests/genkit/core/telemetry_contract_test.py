#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the telemetry product contract."""

from __future__ import annotations

import asyncio
import os
import subprocess  # noqa: S404
import sys
import threading
from collections.abc import Awaitable, Callable, Generator, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, TypeVar
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracerProvider

from genkit import ActionKind, Genkit
from genkit._core._action import Action
from genkit._core._environment import GENKIT_ENV
from genkit._core._instrumentation import (
    NoopSpanContext,
    instrumentations,
    run_in_new_span,
    set_custom_metadata_attributes,
)
from genkit._core._instrumentation_api import Instrumentation, SpanContext, SpanMetadata
from genkit._core._otel_instrumentation import (
    add_custom_exporter,
    maybe_configure_otel_for_exporters,
    parent_path_context,
)
from genkit._core._reflection_v2 import ReflectionServerV2
from genkit._core._registry import Registry
from genkit.telemetry import (
    OtelInstrumentation,
    configure_instrumentation,
    is_instrumented_by,
    reset_instrumentation,
)

T = TypeVar('T')


def _hex_id(value: str, length: int) -> bool:
    return len(value) == length and all(c in '0123456789abcdef' for c in value)


def _flush_exporters_in_provider(provider: TracerProvider) -> None:
    active = getattr(provider, '_active_span_processor', None)
    if active is None:
        return
    processors = getattr(active, '_span_processors', [active])
    for proc in processors:
        exp = getattr(proc, 'span_exporter', None) or getattr(proc, 'exporter', None)
        if exp is not None and hasattr(exp, 'force_flush'):
            exp.force_flush()


def _force_flush() -> None:
    for inst in instrumentations:
        if isinstance(inst, OtelInstrumentation) and inst._tracer_provider is not None:
            inst._tracer_provider.force_flush()
            _flush_exporters_in_provider(inst._tracer_provider)
    provider = trace_api.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush()
        _flush_exporters_in_provider(provider)


async def _joke() -> str:
    return 'Why did the cat cross the road?'


@pytest.fixture(autouse=True)
def _isolate_telemetry(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Each test starts with no providers, unset collector env, and its own tracer."""
    reset_instrumentation()
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
    assert not is_instrumented_by(OtelInstrumentation)


@pytest.mark.asyncio
async def test_genkit_start_gives_the_developer_ui_real_trace_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under genkit start, Genkit() installs OTel so the Traces tab gets real ids."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    Genkit()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert is_instrumented_by(OtelInstrumentation)
    assert _hex_id(result.trace_id, 32)
    assert _hex_id(result.span_id, 16)


@pytest.mark.asyncio
async def test_configuring_otel_yourself_in_dev_stacks_dev_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """configure_instrumentation(OtelInstrumentation()) in dev mode stacks dev instrumentation."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    yours = OtelInstrumentation()
    configure_instrumentation(yours)
    Genkit()

    assert len(instrumentations) == 2
    assert instrumentations[0] == yours
    assert isinstance(instrumentations[1], OtelInstrumentation)


@pytest.mark.asyncio
async def test_an_exporter_alone_does_not_create_spans() -> None:
    """add_custom_exporter attaches exporters only; ids stay empty."""
    exporter = InMemorySpanExporter()
    add_custom_exporter(exporter, 'cloud-trace')
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert result.trace_id == ''
    assert not is_instrumented_by(OtelInstrumentation)
    assert not exporter.get_finished_spans()


@pytest.mark.asyncio
async def test_configure_then_export_sends_spans_to_your_backend() -> None:
    """configure_instrumentation then an exporter: real ids, and the backend sees the span."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_instrumentation(OtelInstrumentation(tracer_provider=provider))

    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert _hex_id(result.trace_id, 32)
    names = [span.name for span in exporter.get_finished_spans()]
    assert 'joke' in names


@pytest.mark.asyncio
async def test_dev_without_a_collector_stays_untraced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GENKIT_ENV=dev with no collector: empty ids. Stop in the Developer UI will not find the run."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')

    Genkit()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert not is_instrumented_by(OtelInstrumentation)
    assert result.trace_id == ''
    assert result.span_id == ''


@pytest.mark.asyncio
async def test_add_custom_exporter_after_genkit_does_not_turn_tracing_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """add_custom_exporter after Genkit() is mailbox-only; ids stay empty."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')

    Genkit()
    exporter = InMemorySpanExporter()
    add_custom_exporter(exporter, 'developer-ui')
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert not is_instrumented_by(OtelInstrumentation)
    assert result.trace_id == ''
    assert not exporter.get_finished_spans()


@pytest.mark.asyncio
async def test_enable_google_cloud_telemetry_is_enough() -> None:
    """enable_google_cloud_telemetry() is enough; you get real hex ids."""
    exporter = InMemorySpanExporter()
    add_custom_exporter(exporter, 'cloud-trace')
    maybe_configure_otel_for_exporters()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()
    _force_flush()

    assert is_instrumented_by(OtelInstrumentation)
    assert _hex_id(result.trace_id, 32)
    names = [span.name for span in exporter.get_finished_spans()]
    assert 'joke' in names


@pytest.mark.asyncio
async def test_enable_does_not_add_a_second_otel_provider() -> None:
    """They already configured OtelInstrumentation; enable only hangs exporters."""
    yours = OtelInstrumentation()
    configure_instrumentation(yours)
    add_custom_exporter(InMemorySpanExporter(), 'cloud-trace')
    maybe_configure_otel_for_exporters()

    assert instrumentations == [yours]


@pytest.mark.asyncio
async def test_enable_hangs_cloud_exporter_on_their_provider() -> None:
    """OtelInstrumentation(tracer_provider=theirs) then enable: Cloud Trace sees their spans."""
    theirs = TracerProvider()
    cloud = InMemorySpanExporter()
    configure_instrumentation(OtelInstrumentation(tracer_provider=theirs))
    add_custom_exporter(cloud, 'cloud-trace')
    maybe_configure_otel_for_exporters()

    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()
    theirs.force_flush()

    assert _hex_id(result.trace_id, 32)
    names = [span.name for span in cloud.get_finished_spans()]
    assert 'joke' in names
    theirs.shutdown()


def test_exporter_refuses_a_provider_it_cannot_attach_to() -> None:
    """Wrong provider type: named TypeError, not a silent empty Cloud Trace."""
    yours = OtelInstrumentation()
    yours._tracer_provider = NoOpTracerProvider()  # pyright: ignore[reportAttributeAccessIssue]
    configure_instrumentation(yours)
    with pytest.raises(TypeError, match='not a TracerProvider'):
        add_custom_exporter(InMemorySpanExporter(), 'cloud-trace')


def test_exporter_reraises_when_their_provider_cannot_add() -> None:
    """Attach failure on theirs must surface; otherwise Cloud Trace stays empty."""
    theirs = TracerProvider()

    def boom(_processor: object) -> None:
        raise RuntimeError('processor dead')

    theirs.add_span_processor = boom  # type: ignore[method-assign]
    configure_instrumentation(OtelInstrumentation(tracer_provider=theirs))
    with pytest.raises(RuntimeError, match='processor dead'):
        add_custom_exporter(InMemorySpanExporter(), 'cloud-trace')
    theirs.shutdown()


def test_exporter_swallows_when_the_global_provider_cannot_add() -> None:
    """Global attach failure stays a log line so enable stays fail-safe."""
    global_provider = trace_api.get_tracer_provider()

    def boom(_processor: object) -> None:
        raise RuntimeError('processor dead')

    global_provider.add_span_processor = boom  # type: ignore[method-assign]
    add_custom_exporter(InMemorySpanExporter(), 'cloud-trace')


def test_init_provider_does_not_rewrite_log_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Booting a tracer must not clobber the process log format."""
    monkeypatch.setattr(trace_api, 'get_tracer_provider', lambda: NoOpTracerProvider())
    created: list[object] = []
    monkeypatch.setattr(trace_api, 'set_tracer_provider', lambda provider: created.append(provider))

    seen: dict[str, bool] = {}

    class FakeInstrumentor:
        def instrument(self, *, set_logging_format: bool) -> None:
            seen['set_logging_format'] = set_logging_format

    monkeypatch.setattr(
        'genkit._core._otel_instrumentation.LoggingInstrumentor',
        FakeInstrumentor,
    )
    from genkit._core._otel_instrumentation import init_provider

    init_provider()
    assert seen['set_logging_format'] is False
    assert created


@pytest.mark.asyncio
async def test_enable_under_genkit_start_leaves_developer_ui_inject_to_genkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under genkit start, enable does not steal the Developer UI collector."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    add_custom_exporter(InMemorySpanExporter(), 'cloud-trace')
    maybe_configure_otel_for_exporters()
    assert not is_instrumented_by(OtelInstrumentation)

    Genkit()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert is_instrumented_by(OtelInstrumentation)
    assert _hex_id(result.trace_id, 32)


@pytest.mark.asyncio
async def test_leftover_collector_url_in_prod_does_not_block_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GENKIT_TELEMETRY_SERVER leftover in prod: enable still turns spans on."""
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    exporter = InMemorySpanExporter()
    add_custom_exporter(exporter, 'cloud-trace')
    maybe_configure_otel_for_exporters()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert is_instrumented_by(OtelInstrumentation)
    assert _hex_id(result.trace_id, 32)


def _handshake_server() -> ReflectionServerV2:
    return ReflectionServerV2(Registry(), 'ws://127.0.0.1:1')


def _capture_handshake_exporters(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    seen: list[object] = []
    real = add_custom_exporter

    def capture(exporter: Any, name: str = 'last') -> None:
        seen.append(exporter)
        real(exporter, name)

    monkeypatch.setattr('genkit._core._otel_instrumentation.add_custom_exporter', capture)
    return seen


def _span_for_export() -> MagicMock:
    span = MagicMock(spec=ReadableSpan)
    ctx = MagicMock()
    ctx.trace_id = 1
    ctx.span_id = 2
    span.context = ctx
    span.name = 'joke'
    span.start_time = 1_000_000_000
    span.end_time = 2_000_000_000
    span.attributes = {}
    span.parent = None
    span.kind = trace_api.SpanKind.INTERNAL
    status = MagicMock()
    status.status_code = trace_api.StatusCode.OK
    status.description = None
    span.status = status
    span.events = ()
    return span


def _start_collector() -> tuple[HTTPServer, list[str]]:
    received: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get('Content-Length', '0'))
            received.append(self.rfile.read(n).decode())
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


@pytest.mark.asyncio
async def test_handshake_url_in_dev_turns_tracing_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UI-only genkit start: handshake URL turns tracing on so the Traces tab fills."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    server, posts = _start_collector()
    url = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        Genkit()
        assert not is_instrumented_by(OtelInstrumentation)

        _handshake_server().apply_handshake_telemetry(url)
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        assert is_instrumented_by(OtelInstrumentation)
        assert _hex_id(result.trace_id, 32)
        assert _hex_id(result.span_id, 16)
        assert posts
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_leftover_collector_url_does_not_drop_handshake_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leftover GENKIT_TELEMETRY_SERVER in the shell still takes today's handshake URL."""
    leftover = _start_collector()
    live = _start_collector()
    leftover_server, leftover_posts = leftover
    live_server, live_posts = live
    leftover_url = f'http://127.0.0.1:{leftover_server.server_address[1]}'
    live_url = f'http://127.0.0.1:{live_server.server_address[1]}'
    try:
        monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', leftover_url)

        Genkit()
        _handshake_server().apply_handshake_telemetry(live_url)
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        assert is_instrumented_by(OtelInstrumentation)
        assert _hex_id(result.trace_id, 32)
        assert live_posts
        assert not leftover_posts
    finally:
        leftover_server.shutdown()
        live_server.shutdown()


@pytest.mark.asyncio
async def test_handshake_skips_when_otel_already_mints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """genkit start -- python: Genkit() already wired the collector; handshake is a no-op."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    Genkit()
    seen = _capture_handshake_exporters(monkeypatch)
    _handshake_server().apply_handshake_telemetry('http://127.0.0.1:4041')

    assert is_instrumented_by(OtelInstrumentation)
    assert seen == []


@pytest.mark.asyncio
async def test_notify_url_in_dev_turns_tracing_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default genkit start POSTs /api/notify; that turns tracing on like v2 handshake."""
    from httpx import ASGITransport, AsyncClient

    from genkit._core._reflection import create_reflection_asgi_app

    monkeypatch.setenv(GENKIT_ENV, 'dev')
    app = create_reflection_asgi_app(Registry())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post('/api/notify', json={'telemetryServerUrl': 'http://127.0.0.1:4041'})
    assert response.status_code == 200

    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert is_instrumented_by(OtelInstrumentation)
    assert _hex_id(result.trace_id, 32)


@pytest.mark.asyncio
async def test_already_minting_hangs_handshake_mailbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """force_dev_export mints first; handshake hangs today's mailbox so the tab can fill."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    server, posts = _start_collector()
    url = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        add_custom_exporter(InMemorySpanExporter(), 'cloud-trace')
        maybe_configure_otel_for_exporters()
        assert is_instrumented_by(OtelInstrumentation)

        _handshake_server().apply_handshake_telemetry(url)
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        assert _hex_id(result.trace_id, 32)
        assert posts
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_plugin_tracer_is_noop_when_uninstrumented() -> None:
    """Imagen/Veo plugin_api.tracer does not mint when tracing is off."""
    from genkit.plugin_api import tracer

    with tracer.start_as_current_span('generate_images') as span:
        ctx = span.get_span_context()
        assert ctx.trace_id == 0


@pytest.mark.asyncio
async def test_plugin_tracer_hangs_on_their_provider() -> None:
    """OtelInstrumentation(tracer_provider=theirs): plugin_api.tracer mints on theirs."""
    from genkit.plugin_api import tracer

    theirs = TracerProvider()
    cloud = InMemorySpanExporter()
    theirs.add_span_processor(SimpleSpanProcessor(cloud))
    configure_instrumentation(OtelInstrumentation(tracer_provider=theirs))

    with tracer.start_as_current_span('generate_images'):
        pass
    theirs.force_flush()

    names = [span.name for span in cloud.get_finished_spans()]
    assert 'generate_images' in names
    theirs.shutdown()


def test_importing_genkit_does_not_start_a_tracer() -> None:
    """from genkit import Genkit does not start a tracer or install a provider."""
    script = """
from opentelemetry import trace
from genkit import Genkit  # noqa: F401
from genkit._core._otel_instrumentation import is_placeholder_provider
from genkit.telemetry import OtelInstrumentation, is_instrumented_by

assert not is_instrumented_by(OtelInstrumentation)
assert is_placeholder_provider(trace.get_tracer_provider())
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
    """Dev instrumentation uses an isolated TracerProvider and never mutates global OTel."""
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
async def test_dev_mode_with_custom_apm_separates_dev_and_remote_exporters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dev UI and remote APM both receive clean spans via isolated providers."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    server, posts = _start_collector()
    dev_url = f'http://127.0.0.1:{server.server_address[1]}'
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', dev_url)

    try:
        remote_exporter = InMemorySpanExporter()
        remote_provider = TracerProvider()
        remote_provider.add_span_processor(SimpleSpanProcessor(remote_exporter))
        configure_instrumentation(OtelInstrumentation(tracer_provider=remote_provider))

        Genkit()
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        assert _hex_id(result.trace_id, 32)
        assert posts
        remote_spans = [s.name for s in remote_exporter.get_finished_spans()]
        assert 'joke' in remote_spans
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_nested_flow_model_tool_shares_trace_id_and_parent_links() -> None:
    """Nested actions maintain the same trace_id and correct parent_span_id links."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_instrumentation(OtelInstrumentation(tracer_provider=provider))

    async def step_fn() -> str:
        return 'step_ok'

    step_action = Action(name='stepAction', kind=ActionKind.UTIL, fn=step_fn)

    async def flow_fn() -> str:
        res = await step_action.run()
        return f'flow_{res.response}'

    flow_action = Action(name='flowAction', kind=ActionKind.FLOW, fn=flow_fn)
    result = await flow_action.run()

    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    flow_span = next(s for s in spans if s.name == 'flowAction')
    step_span = next(s for s in spans if s.name == 'stepAction')

    flow_ctx = flow_span.context
    step_ctx = step_span.context
    assert flow_ctx is not None and step_ctx is not None
    assert flow_ctx.trace_id == step_ctx.trace_id
    assert format(flow_ctx.trace_id, '032x') == result.trace_id
    assert step_span.parent is not None
    assert step_span.parent.span_id == flow_ctx.span_id


@pytest.mark.asyncio
async def test_concurrent_flows_maintain_independent_trace_trees() -> None:
    """Concurrent flows in asyncio.gather keep independent trace IDs and trees."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_instrumentation(OtelInstrumentation(tracer_provider=provider))

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

    provider.force_flush()
    spans = exporter.get_finished_spans()
    span1 = next(s for s in spans if s.name == 'flow1')
    span2 = next(s for s in spans if s.name == 'flow2')
    s1_ctx = span1.context
    s2_ctx = span2.context
    assert s1_ctx is not None and s2_ctx is not None
    assert s1_ctx.trace_id != s2_ctx.trace_id


@pytest.mark.asyncio
async def test_run_in_new_span_snapshots_providers_against_concurrent_mutation() -> None:
    """Mutating instrumentations while a span runs does not affect the active span."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_instrumentation(OtelInstrumentation(tracer_provider=provider))

    async def body(_span: object) -> str:
        reset_instrumentation()
        return 'mutated'

    res = await run_in_new_span('in_flight', body, action_type='flow')
    assert res == 'mutated'

    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == 'in_flight'


@pytest.mark.asyncio
async def test_set_custom_metadata_stamps_all_instrumentations_for_current_action() -> None:
    """set_custom_metadata_attributes fans out to all active provider spans."""
    p1 = TracerProvider()
    e1 = InMemorySpanExporter()
    p1.add_span_processor(SimpleSpanProcessor(e1))

    p2 = TracerProvider()
    e2 = InMemorySpanExporter()
    p2.add_span_processor(SimpleSpanProcessor(e2))

    configure_instrumentation(OtelInstrumentation(tracer_provider=p1))
    configure_instrumentation(OtelInstrumentation(tracer_provider=p2))

    async def flow_fn() -> str:
        set_custom_metadata_attributes({'user_id': 'user_42', 'tier': 'enterprise'})
        return 'ok'

    action = Action(name='metaFlow', kind=ActionKind.FLOW, fn=flow_fn)
    await action.run()

    p1.force_flush()
    p2.force_flush()

    s1 = e1.get_finished_spans()[0]
    s2 = e2.get_finished_spans()[0]

    assert s1.attributes is not None and s1.attributes['genkit:metadata:user_id'] == 'user_42'
    assert s1.attributes is not None and s1.attributes['genkit:metadata:tier'] == 'enterprise'
    assert s2.attributes is not None and s2.attributes['genkit:metadata:user_id'] == 'user_42'
    assert s2.attributes is not None and s2.attributes['genkit:metadata:tier'] == 'enterprise'


def test_set_custom_metadata_is_noop_outside_action() -> None:
    """Calling set_custom_metadata_attributes outside an action does not raise."""
    set_custom_metadata_attributes({'some': 'value'})


@pytest.mark.asyncio
async def test_failing_custom_provider_does_not_break_other_providers() -> None:
    """A throwing custom provider span does not crash other providers or action execution."""
    p1 = TracerProvider()
    e1 = InMemorySpanExporter()
    p1.add_span_processor(SimpleSpanProcessor(e1))

    class BrokenSpan(NoopSpanContext):
        def set_metadata(self, metadata: Mapping[str, object]) -> None:
            raise RuntimeError('custom provider metadata crash')

    class BrokenInstrumentation(Instrumentation):
        async def run_in_new_span(
            self,
            metadata: SpanMetadata,
            next: Callable[[SpanContext], Awaitable[T]],
        ) -> T:
            return await next(BrokenSpan())

    configure_instrumentation(OtelInstrumentation(tracer_provider=p1))
    configure_instrumentation(BrokenInstrumentation())

    async def flow_fn() -> str:
        set_custom_metadata_attributes({'custom': 'val'})
        return 'success'

    action = Action(name='failingProviderFlow', kind=ActionKind.FLOW, fn=flow_fn)
    result = await action.run()

    assert result.response == 'success'
    p1.force_flush()
    spans = e1.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes is not None and spans[0].attributes['genkit:metadata:custom'] == 'val'
