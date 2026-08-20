#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the telemetry product contract."""

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
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
from genkit._core._instrumentation import instrumentations
from genkit._core._otel_instrumentation import (
    add_custom_exporter,
    maybe_configure_otel_for_exporters,
    parent_path_context,
)
from genkit._core._reflection_v2 import ReflectionServerV2
from genkit._core._registry import Registry
from genkit._core._trace._default_exporter import TraceServerExporter
from genkit.telemetry import (
    OtelInstrumentation,
    configure_instrumentation,
    is_instrumented_by,
    reset_instrumentation,
)


def _hex_id(value: str, length: int) -> bool:
    return len(value) == length and all(c in '0123456789abcdef' for c in value)


def _force_flush() -> None:
    provider = trace_api.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    provider.force_flush()


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
async def test_configuring_otel_yourself_means_you_own_the_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """configure_instrumentation(OtelInstrumentation()) first: we will not add a Developer UI provider."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    yours = OtelInstrumentation()
    configure_instrumentation(yours)
    Genkit()

    assert instrumentations == [yours]


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
    yours._tracer_provider = NoOpTracerProvider()
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

    def capture(exporter: object, name: str = 'last') -> None:
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
        def do_POST(self) -> None:
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
        seen = _capture_handshake_exporters(monkeypatch)
        _handshake_server().apply_handshake_telemetry(live_url)
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()

        assert not is_instrumented_by(OtelInstrumentation)
        assert result.trace_id == ''
        assert len(seen) == 1
        exporter = seen[0]
        assert isinstance(exporter, TraceServerExporter)
        assert exporter.telemetry_server_url == live_url

        exporter.export([_span_for_export()])
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
