# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for Dev UI OTLP log export."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from queue import Full
from unittest.mock import MagicMock, patch

import httpx
import pytest
from structlog.testing import capture_logs

from genkit._core._constants import GENKIT_VERSION
from genkit._core._environment import GENKIT_ENV
from genkit._core._logger import get_logger, is_debug_enabled
from genkit._core._trace._log_exporter import (
    BATCH_DELAY_S,
    GENKIT_OTEL_ENABLE_LOGS,
    LOG_ENDPOINT,
    MAX_ATTR_CHARS,
    MAX_ATTRIBUTES,
    SCOPE_NAME,
    SHUTDOWN_FLUSH_TIMEOUT_S,
    build_log_record,
    build_payload,
    emit_log,
    enable_log_export,
    log_export_is_enabled,
    logs_opted_out,
    put_poison_pill,
    reset_log_export,
)
from genkit._core._tracing import SpanMetadata, run_in_new_span


@pytest.fixture
def _reset_export() -> Iterator[None]:
    reset_log_export()
    yield
    reset_log_export()


@pytest.fixture
def _dev_env() -> Iterator[None]:
    with patch.dict(os.environ, {GENKIT_ENV: 'dev', GENKIT_OTEL_ENABLE_LOGS: ''}, clear=False):
        os.environ.pop(GENKIT_OTEL_ENABLE_LOGS, None)
        yield


def test_logs_opted_out_matches_parse_bool() -> None:
    """Only ParseBool false values opt out — ``off`` keeps export on."""
    for value in ['false', 'False', '0', 'f', 'F']:
        with patch.dict(os.environ, {GENKIT_OTEL_ENABLE_LOGS: value}):
            assert logs_opted_out() is True
    for value in ['', 'true', '1', 'off', 'yes']:
        with patch.dict(os.environ, {GENKIT_OTEL_ENABLE_LOGS: value} if value else {}, clear=False):
            if not value:
                os.environ.pop(GENKIT_OTEL_ENABLE_LOGS, None)
            assert logs_opted_out() is False


def test_build_log_record_wire_shape() -> None:
    """OTLP-JSON: string nanoseconds, severity 5/9/13/21, typed attrs only."""
    record = build_log_record(
        level=logging.DEBUG,
        event='looking up weather',
        attrs={
            'city': 'Paris',
            'count': 2,
            'ok': True,
            'temp': 3.5,
            'blob': b'12345',
            'buf': bytearray(b'123'),
            'mem': memoryview(b'abc'),
            '_skip': 1,
            'empty': None,
        },
    )
    assert record['severityNumber'] == 5
    assert record['severityText'] == 'DEBUG'
    assert record['body'] == {'stringValue': 'looking up weather'}
    nano = record['timeUnixNano']
    assert isinstance(nano, str) and nano.isdigit()
    raw_attrs = record['attributes']
    assert isinstance(raw_attrs, list)
    keys: dict[object, object] = {}
    for item in raw_attrs:
        assert isinstance(item, dict)
        keys[item['key']] = item['value']
    assert keys['city'] == {'stringValue': 'Paris'}
    assert keys['count'] == {'intValue': 2}
    assert keys['ok'] == {'boolValue': True}
    assert keys['temp'] == {'doubleValue': 3.5}
    assert keys['blob'] == {'stringValue': '<5 bytes>'}
    assert keys['buf'] == {'stringValue': '<3 bytes>'}
    assert keys['mem'] == {'stringValue': '<3 bytes>'}
    assert '_skip' not in keys
    assert 'empty' not in keys
    assert 'traceId' not in record
    assert 'spanId' not in record

    crit_record = build_log_record(level=logging.CRITICAL, event='panic', attrs={})
    assert crit_record['severityNumber'] == 21
    assert crit_record['severityText'] == 'FATAL'


def test_build_log_record_stamps_active_span() -> None:
    """A record under a span carries lowercase hex ids."""
    captured: dict[str, str] = {}

    def go() -> None:
        record = build_log_record(level=logging.INFO, event='inside', attrs={})
        trace_id = record['traceId']
        span_id = record['spanId']
        assert isinstance(trace_id, str)
        assert isinstance(span_id, str)
        captured['traceId'] = trace_id
        captured['spanId'] = span_id

    from genkit._core._tracing import init_provider

    init_provider()
    with run_in_new_span(metadata=SpanMetadata(name='demo')):
        go()

    assert len(captured['traceId']) == 32
    assert len(captured['spanId']) == 16
    assert captured['traceId'] == captured['traceId'].lower()


def test_build_payload_scope() -> None:
    """One resourceLogs / scopeLogs batch named genkit-py."""
    record = build_log_record(level=logging.INFO, event='hi', attrs={})
    payload = build_payload(records=[record])
    resource_logs = payload['resourceLogs']
    assert isinstance(resource_logs, list)
    first = resource_logs[0]
    assert isinstance(first, dict)
    scope_logs = first['scopeLogs']
    assert isinstance(scope_logs, list)
    first_scope = scope_logs[0]
    assert isinstance(first_scope, dict)
    assert first_scope['scope'] == {'name': SCOPE_NAME, 'version': GENKIT_VERSION}
    assert first_scope['logRecords'] == [record]


@pytest.mark.usefixtures('_reset_export', '_dev_env')
def test_enable_ignored_when_opted_out() -> None:
    with patch.dict(os.environ, {GENKIT_OTEL_ENABLE_LOGS: 'false'}):
        enable_log_export(url='http://127.0.0.1:9')
        assert log_export_is_enabled() is False


@pytest.mark.usefixtures('_reset_export')
def test_enable_ignored_outside_dev() -> None:
    with patch.dict(os.environ, {GENKIT_ENV: 'prod'}):
        enable_log_export(url='http://127.0.0.1:9')
        assert log_export_is_enabled() is False


@pytest.mark.usefixtures('_reset_export', '_dev_env')
def test_export_does_not_stall_on_hung_collector() -> None:
    """emit_log returns while a collector that never reads is still hanging."""
    started = threading.Event()
    release = threading.Event()

    def blocking_post(*_args: object, **_kwargs: object) -> MagicMock:
        started.set()
        release.wait(timeout=5)
        raise httpx.ConnectError('hung')

    with patch('genkit._core._trace._log_exporter.httpx.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client.post.side_effect = blocking_post
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

        enable_log_export(url='http://127.0.0.1:9')
        t0 = time.perf_counter()
        emit_log(level=logging.DEBUG, event='hello', attrs={'city': 'Paris'})
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.5
        assert started.wait(timeout=1)
        release.set()
        from genkit._core._trace._log_exporter import _exporter

        assert _exporter is not None
        assert _exporter.force_flush(timeout_s=2) is True
        assert _exporter.last_result_ok is False


@pytest.mark.usefixtures('_reset_export', '_dev_env')
def test_transport_failure_logs_one_error() -> None:
    """A dead collector is one error line — not a traceback."""
    enable_log_export(url='http://127.0.0.1:1')
    with capture_logs() as entries:
        emit_log(level=logging.INFO, event='hello', attrs={})
        from genkit._core._trace._log_exporter import _exporter

        assert _exporter is not None
        assert _exporter.force_flush(timeout_s=2) is True
        emit_log(level=logging.INFO, event='again', attrs={})
        assert _exporter.force_flush(timeout_s=2) is True

    errors = [e for e in entries if 'Failed to export logs' in str(e.get('event', ''))]
    assert len(errors) == 1
    assert 'exception' not in errors[0]


@pytest.mark.usefixtures('_reset_export', '_dev_env')
def test_overflow_drops_and_warns_once() -> None:
    """A full queue drops the record and warns once — emit never blocks."""
    enable_log_export(url='http://127.0.0.1:1')
    from genkit._core._trace._log_exporter import _exporter

    assert _exporter is not None
    record = build_log_record(level=logging.DEBUG, event='overflow', attrs={})
    with patch.object(_exporter.queue, 'put_nowait', side_effect=Full):
        with capture_logs() as entries:
            t0 = time.perf_counter()
            _exporter.enqueue(record=record)
            _exporter.enqueue(record=record)
            assert time.perf_counter() - t0 < 0.2
        warnings = [e for e in entries if 'queue is full' in str(e.get('event', ''))]
        assert len(warnings) == 1
        assert _exporter.dropped == 2


@pytest.mark.usefixtures('_reset_export', '_dev_env')
def test_posts_otlp_path_and_batches() -> None:
    """Worker POSTs ``/api/otlp`` with a resourceLogs envelope."""
    seen: list[dict[str, object]] = []
    posted = threading.Event()

    def capture_post(*_args: object, **kwargs: object) -> MagicMock:
        body = kwargs.get('content') or (_args[1] if len(_args) > 1 else None)
        if isinstance(body, (bytes, str)):
            seen.append(json.loads(body))
        posted.set()
        response = MagicMock()
        response.status_code = 200
        return response

    with patch('genkit._core._trace._log_exporter.httpx.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client.post.side_effect = capture_post
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

        enable_log_export(url='http://localhost:4033')
        emit_log(level=logging.DEBUG, event='a', attrs={'n': 1})
        emit_log(level=logging.INFO, event='b', attrs={})
        from genkit._core._trace._log_exporter import _exporter

        assert _exporter is not None
        assert _exporter.force_flush(timeout_s=2) is True
        assert posted.wait(timeout=2)

    mock_client.post.assert_called()
    called = mock_client.post.call_args
    url = called.args[0] if called.args else called.kwargs.get('url')
    assert url is not None and url.endswith(LOG_ENDPOINT)
    records = seen[0]['resourceLogs'][0]['scopeLogs'][0]['logRecords']  # type: ignore[index]
    events = [r['body']['stringValue'] for r in records]  # type: ignore[index]
    assert 'a' in events
    assert 'b' in events
    assert BATCH_DELAY_S == 0.1


@pytest.mark.usefixtures('_reset_export', '_dev_env')
def test_get_logger_tees_debug_when_console_is_info() -> None:
    """GENKIT_LOG=info keeps the TTY quiet; the Dev UI still gets debug."""
    enable_log_export(url='http://127.0.0.1:9')
    from genkit._core._logger import configure_structlog_level
    from genkit._core._trace._log_exporter import _exporter

    with patch.dict(os.environ, {'GENKIT_LOG': 'info'}):
        structlog_ok = configure_structlog_level()
        logger = get_logger('tee-test')
        assert is_debug_enabled(logger) is True
        queued: list[dict[str, object]] = []
        assert _exporter is not None
        original = _exporter.enqueue

        def capture(*, record: dict[str, object]) -> None:
            queued.append(record)

        _exporter.enqueue = capture  # type: ignore[method-assign]
        try:
            logger.debug('looking up weather', city='Paris')
            logger.error('Startup failed: %s: %s', 'ValueError', 'boom')
            logger.info('kept on console')
        finally:
            _exporter.enqueue = original  # type: ignore[method-assign]

    events = [r['body']['stringValue'] for r in queued]  # type: ignore[index]
    assert 'looking up weather' in events
    assert 'Startup failed: ValueError: boom' in events
    assert 'kept on console' in events
    assert structlog_ok in {True, False}


@pytest.mark.usefixtures('_reset_export', '_dev_env')
def test_worker_stops_when_client_construction_fails() -> None:
    """Client() raising must flip stopped so debug branches stop building records."""
    with patch('genkit._core._trace._log_exporter.httpx.Client', side_effect=RuntimeError('no client')):
        enable_log_export(url='http://127.0.0.1:9')
        deadline = time.monotonic() + 2.0
        while log_export_is_enabled() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert log_export_is_enabled() is False


@pytest.mark.usefixtures('_reset_export', '_dev_env')
def test_handshake_enables_log_export_when_env_url_missing() -> None:
    """Reflection register/configure URL turns on log export, same as traces."""
    os.environ.pop('GENKIT_TELEMETRY_SERVER', None)
    from genkit._core._reflection_v2 import ReflectionServerV2
    from genkit._core._registry import Registry

    with patch('genkit._core._reflection_v2.add_custom_exporter'):
        server = ReflectionServerV2(registry=Registry(), ws_url='ws://localhost:1')
        server.apply_handshake_telemetry('http://localhost:4033')
        assert log_export_is_enabled() is True
        server.apply_handshake_telemetry('http://localhost:4033')


def test_build_log_record_bounds_and_redacts_attrs() -> None:
    """A generate() that binds 200 keys or an API key shouldn't dump them."""
    attrs: dict[str, object] = {
        'api_key': 'sk-secret',
        'prompt': 'P' * (MAX_ATTR_CHARS + 50),
    }
    extra = MAX_ATTRIBUTES + 10
    attrs.update({f'k{i}': i for i in range(extra)})
    record = build_log_record(level=logging.DEBUG, event='bounded', attrs=attrs)
    raw_attrs = record['attributes']
    assert isinstance(raw_attrs, list)
    assert len(raw_attrs) == MAX_ATTRIBUTES
    assert record['droppedAttributesCount'] == extra - (MAX_ATTRIBUTES - 2)
    keys = {item['key']: item['value'] for item in raw_attrs if isinstance(item, dict)}
    assert keys['api_key'] == {'stringValue': '[redacted]'}
    prompt_value = keys['prompt']
    assert isinstance(prompt_value, dict)
    prompt = prompt_value['stringValue']
    assert isinstance(prompt, str)
    assert prompt.startswith('P' * MAX_ATTR_CHARS)
    assert prompt.endswith(f'...<{50} chars>')


def test_build_log_record_redacts_camel_case_secrets() -> None:
    """apiKey / authToken / apikey must not POST in cleartext to /api/otlp."""
    record = build_log_record(
        level=logging.DEBUG,
        event='calling provider',
        attrs={'apiKey': 'sk-secret', 'authToken': 'tok', 'apikey': 'sk-2', 'city': 'Paris'},
    )
    raw_attrs = record['attributes']
    assert isinstance(raw_attrs, list)
    keys = {item['key']: item['value'] for item in raw_attrs if isinstance(item, dict)}
    assert keys['apiKey'] == {'stringValue': '[redacted]'}
    assert keys['authToken'] == {'stringValue': '[redacted]'}
    assert keys['apikey'] == {'stringValue': '[redacted]'}
    assert keys['city'] == {'stringValue': 'Paris'}


def test_put_poison_pill_lands_when_queue_is_full() -> None:
    """Shutdown must stop the worker even if the last slot is a leftover record."""
    from queue import Queue

    queue: Queue[dict[str, object] | None] = Queue(maxsize=1)
    queue.put_nowait({'leftover': True})
    put_poison_pill(queue=queue)
    assert queue.get_nowait() is None
    queue.task_done()
    queue.join()
    assert queue.empty()


@pytest.mark.usefixtures('_reset_export', '_dev_env')
def test_shutdown_does_not_wait_for_export_timeout() -> None:
    """atexit / Ctrl-C flush the last batch for ~2s, not the 5-minute POST."""
    started = threading.Event()
    release = threading.Event()

    def blocking_post(*_args: object, **_kwargs: object) -> MagicMock:
        started.set()
        release.wait(timeout=10)
        raise httpx.ConnectError('hung')

    with patch('genkit._core._trace._log_exporter.httpx.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client.post.side_effect = blocking_post
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

        enable_log_export(url='http://127.0.0.1:9')
        emit_log(level=logging.DEBUG, event='hello', attrs={})
        assert started.wait(timeout=1)
        t0 = time.perf_counter()
        reset_log_export()
        elapsed = time.perf_counter() - t0
        release.set()

    assert elapsed < SHUTDOWN_FLUSH_TIMEOUT_S + 1.5
    assert SHUTDOWN_FLUSH_TIMEOUT_S == 2.0


@pytest.mark.usefixtures('_reset_export', '_dev_env')
def test_enable_rejects_invalid_url_on_caller_thread() -> None:
    """A typo'd collector URL fails when export starts, not as missing Dev UI logs."""
    enable_log_export(url='not-a-url')
    assert log_export_is_enabled() is False
