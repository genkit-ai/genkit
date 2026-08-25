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

"""Tests for the default telemetry exporter module.

This module tests:
    - TraceServerExporter: Exports spans to a telemetry server
    - extract_span_data: Extracts span data for export
    - create_span_processor: Creates appropriate span processor based on environment
    - init_telemetry_server_exporter: Initializes the telemetry server exporter
"""

import json
import os
import threading
import time
import urllib.error
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult
from structlog.testing import capture_logs

from genkit._core._environment import GENKIT_ENV, GenkitEnvironment
from genkit._core._trace._default_exporter import (
    EXPORT_TIMEOUT_SECONDS,
    TraceServerExporter,
    create_span_processor,
    extract_span_data,
    init_telemetry_server_exporter,
)
from genkit._core._trace._realtime_processor import RealtimeSpanProcessor
from genkit._core._typing import TraceData

# =============================================================================
# Tests for create_span_processor
# =============================================================================


def test_create_span_processor_returns_realtime_in_dev() -> None:
    """Test that RealtimeSpanProcessor is returned in dev mode."""
    mock_exporter = MagicMock()

    with mock.patch.dict(
        os.environ,
        {
            GENKIT_ENV: GenkitEnvironment.DEV,
        },
    ):
        processor = create_span_processor(mock_exporter)
        assert isinstance(processor, RealtimeSpanProcessor)


def test_create_span_processor_returns_batch_in_prod() -> None:
    """Test that BatchSpanProcessor is returned in production mode."""
    mock_exporter = MagicMock()

    with mock.patch.dict(
        os.environ,
        {
            GENKIT_ENV: GenkitEnvironment.PROD,
        },
    ):
        processor = create_span_processor(mock_exporter)
        assert isinstance(processor, BatchSpanProcessor)


def test_create_span_processor_returns_batch_when_no_env_set() -> None:
    """Test that BatchSpanProcessor is returned when no env is set (defaults to prod)."""
    mock_exporter = MagicMock()

    with mock.patch.dict(os.environ, clear=True):
        processor = create_span_processor(mock_exporter)
        assert isinstance(processor, BatchSpanProcessor)


# =============================================================================
# Tests for init_telemetry_server_exporter
# =============================================================================


def test_init_telemetry_server_exporter_returns_exporter_when_url_set() -> None:
    """Test that exporter is returned when GENKIT_TELEMETRY_SERVER is set."""
    with mock.patch.dict(os.environ, {'GENKIT_TELEMETRY_SERVER': 'http://localhost:4000'}):
        exporter = init_telemetry_server_exporter()
        assert exporter is not None
        assert isinstance(exporter, TraceServerExporter)
        assert exporter.telemetry_server_url == 'http://localhost:4000'


def test_init_telemetry_server_exporter_returns_none_when_url_not_set() -> None:
    """Test that None is returned when GENKIT_TELEMETRY_SERVER is not set."""
    with mock.patch.dict(os.environ, clear=True):
        exporter = init_telemetry_server_exporter()
        assert exporter is None


def test_init_telemetry_server_exporter_returns_none_when_url_is_invalid() -> None:
    """A typo'd env var must not crash import — just skip the exporter."""
    with (
        mock.patch.dict(os.environ, {'GENKIT_TELEMETRY_SERVER': 'http://[::1:4033'}),
        capture_logs() as entries,
    ):
        exporter = init_telemetry_server_exporter()

    assert exporter is None
    assert any('GENKIT_TELEMETRY_SERVER is not a valid URL' in str(e.get('event', '')) for e in entries)


# =============================================================================
# Tests for TraceServerExporter
# =============================================================================


def test_telemetry_server_exporter_init_default_endpoint() -> None:
    """Test TraceServerExporter initialization with default endpoint."""
    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')

    assert exporter.telemetry_server_url == 'http://localhost:4000'
    assert exporter.telemetry_server_endpoint == '/api/traces'


def test_telemetry_server_exporter_rejects_malformed_url() -> None:
    """A typo'd IPv6 URL must fail on the caller thread, not kill the worker later."""
    with pytest.raises(ValueError, match='invalid telemetry server URL'):
        TraceServerExporter(telemetry_server_url='http://[::1:4033')


def test_telemetry_server_exporter_rejects_non_http_url() -> None:
    """A websocket URL must fail at init, not as a missing Dev UI trace later."""
    with pytest.raises(ValueError, match='invalid telemetry server URL'):
        TraceServerExporter(telemetry_server_url='ws://localhost:4033')


def test_telemetry_server_exporter_strips_url_whitespace() -> None:
    """A trailing newline in the env var is a common copy-paste, not a bad URL."""
    exporter = TraceServerExporter(telemetry_server_url='  http://localhost:4000\n')
    assert exporter.telemetry_server_url == 'http://localhost:4000'


def test_telemetry_server_exporter_init_custom_endpoint() -> None:
    """Test TraceServerExporter initialization with custom endpoint."""
    exporter = TraceServerExporter(
        telemetry_server_url='http://localhost:4000',
        telemetry_server_endpoint='/custom/traces',
    )

    assert exporter.telemetry_server_url == 'http://localhost:4000'
    assert exporter.telemetry_server_endpoint == '/custom/traces'


def test_telemetry_server_exporter_force_flush_returns_true() -> None:
    """Test that force_flush returns True when the worker queue is empty."""
    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')

    result = exporter.force_flush()
    assert result is True


def test_telemetry_server_exporter_force_flush_respects_timeout() -> None:
    """Test that force_flush can time out while a post is still in flight."""
    started = threading.Event()
    release = threading.Event()

    def blocking_urlopen(*_args: object, **_kwargs: object) -> MagicMock:
        started.set()
        release.wait(timeout=5)
        return mock_urlopen_response()

    with patch(
        'genkit._core._trace._default_exporter.urllib.request.urlopen',
        side_effect=blocking_urlopen,
    ):
        exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')
        exporter.export([create_mock_span()])
        assert started.wait(timeout=1)
        assert exporter.force_flush(timeout_millis=20) is False
        release.set()
        assert exporter.force_flush(timeout_millis=2000) is True


@patch('genkit._core._trace._default_exporter.urllib.request.urlopen')
def test_telemetry_server_exporter_export_sends_http_post(mock_urlopen: MagicMock) -> None:
    """Test that export sends HTTP POST requests for each span."""
    mock_urlopen.return_value = mock_urlopen_response()

    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')

    mock_span = create_mock_span()

    result = exporter.export([mock_span])
    assert exporter.force_flush(timeout_millis=2000) is True

    assert result == SpanExportResult.SUCCESS
    mock_urlopen.assert_called_once()
    assert mock_urlopen.call_args.kwargs['timeout'] == EXPORT_TIMEOUT_SECONDS
    request = mock_urlopen.call_args.args[0]
    assert request.full_url.startswith('http://localhost:4000')
    assert request.get_method() == 'POST'


@patch('genkit._core._trace._default_exporter.urllib.request.urlopen')
def test_telemetry_server_exporter_export_groups_same_trace(mock_urlopen: MagicMock) -> None:
    """A batch of spans on one trace is one POST — BatchSpanProcessor flushes a whole flow."""
    mock_urlopen.return_value = mock_urlopen_response()

    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')
    root = create_mock_span(trace_id=0xABCD, span_id=1, name='root')
    child_a = create_mock_span(trace_id=0xABCD, span_id=2, name='child-a')
    child_b = create_mock_span(trace_id=0xABCD, span_id=3, name='child-b')
    child_a.parent = root.context
    child_b.parent = root.context
    mock_spans = [root, child_a, child_b]

    result = exporter.export(mock_spans)
    assert exporter.force_flush(timeout_millis=2000) is True

    assert result == SpanExportResult.SUCCESS
    assert mock_urlopen.call_count == 1
    body = json.loads(mock_urlopen.call_args.args[0].data)
    assert body['traceId'] == format(0xABCD, '032x')
    assert set(body['spans']) == {format(1, '016x'), format(2, '016x'), format(3, '016x')}
    assert body['displayName'] == 'root'


@patch('genkit._core._trace._default_exporter.urllib.request.urlopen')
def test_telemetry_server_exporter_export_posts_once_per_trace(mock_urlopen: MagicMock) -> None:
    """Two traces in one batch are two POSTs, not one per span."""
    mock_urlopen.return_value = mock_urlopen_response()

    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')
    mock_spans = [
        create_mock_span(trace_id=0xA, span_id=1),
        create_mock_span(trace_id=0xA, span_id=2),
        create_mock_span(trace_id=0xB, span_id=3),
    ]

    result = exporter.export(mock_spans)
    assert exporter.force_flush(timeout_millis=2000) is True

    assert result == SpanExportResult.SUCCESS
    assert mock_urlopen.call_count == 2
    posted = {json.loads(c.args[0].data)['traceId'] for c in mock_urlopen.call_args_list}
    assert posted == {format(0xA, '032x'), format(0xB, '032x')}


def test_export_transport_failure_logs_one_error_and_records_failure() -> None:
    """A dead collector is one error line that names the trace — not a traceback."""
    exporter = TraceServerExporter(telemetry_server_url='http://127.0.0.1:1')
    mock_span = create_mock_span(trace_id=0xABCDEF)

    with capture_logs() as entries:
        started = time.perf_counter()
        result = exporter.export([mock_span])
        elapsed = time.perf_counter() - started
        assert exporter.force_flush(timeout_millis=2000) is True

    assert result == SpanExportResult.SUCCESS
    assert elapsed < 0.5
    assert exporter.last_result == SpanExportResult.FAILURE
    errors = [e for e in entries if 'Failed to save trace' in str(e.get('event', ''))]
    assert len(errors) == 1
    event = errors[0]['event']
    assert format(0xABCDEF, '032x') in event
    assert 'Connection refused' in event or 'URLError' in event
    assert 'exception' not in errors[0]
    assert 'exc_info' not in errors[0]


def test_export_does_not_stall_on_hung_collector() -> None:
    """generate() must return while a collector that accept()s and never reads is still hanging."""
    started = threading.Event()
    release = threading.Event()

    def blocking_urlopen(*_args: object, **_kwargs: object) -> MagicMock:
        started.set()
        release.wait(timeout=5)
        raise urllib.error.URLError('hung')

    with patch(
        'genkit._core._trace._default_exporter.urllib.request.urlopen',
        side_effect=blocking_urlopen,
    ):
        exporter = TraceServerExporter(telemetry_server_url='http://127.0.0.1:9')
        t0 = time.perf_counter()
        result = exporter.export([create_mock_span()])
        elapsed = time.perf_counter() - t0

        assert result == SpanExportResult.SUCCESS
        assert elapsed < 0.5
        assert started.wait(timeout=1)
        release.set()
        assert exporter.force_flush(timeout_millis=2000) is True
        assert exporter.last_result == SpanExportResult.FAILURE


def test_shutdown_does_not_wait_out_hung_post() -> None:
    """Process exit must not sit on the full flush timeout for a wedged collector."""
    started = threading.Event()
    release = threading.Event()

    def blocking_urlopen(*_args: object, **_kwargs: object) -> MagicMock:
        started.set()
        release.wait(timeout=10)
        return mock_urlopen_response()

    with patch(
        'genkit._core._trace._default_exporter.urllib.request.urlopen',
        side_effect=blocking_urlopen,
    ):
        exporter = TraceServerExporter(telemetry_server_url='http://127.0.0.1:9')
        exporter.export([create_mock_span()])
        assert started.wait(timeout=1)
        t0 = time.perf_counter()
        exporter.shutdown()
        elapsed = time.perf_counter() - t0
        assert elapsed < 5
        assert exporter.stopped
        release.set()


def test_batch_stops_after_first_trace_failure() -> None:
    """A transport failure drops the rest of the batch — one error line, not a retry storm."""
    posts = {'n': 0}

    def flaky_urlopen(*_args: object, **_kwargs: object) -> MagicMock:
        posts['n'] += 1
        if posts['n'] == 1:
            raise urllib.error.URLError('reset by peer')
        return mock_urlopen_response()

    with patch(
        'genkit._core._trace._default_exporter.urllib.request.urlopen',
        side_effect=flaky_urlopen,
    ):
        exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')
        with capture_logs() as entries:
            exporter.export([
                create_mock_span(trace_id=0xA, span_id=1),
                create_mock_span(trace_id=0xB, span_id=2),
            ])
            assert exporter.force_flush(timeout_millis=2000) is True

    assert posts['n'] == 1
    errors = [e for e in entries if 'Failed to save trace' in str(e.get('event', ''))]
    assert len(errors) == 1
    assert format(0xA, '032x') in errors[0]['event']


def test_worker_survives_unexpected_post_error() -> None:
    """A non-transport exception must not kill the worker or swallow later traces."""
    posts = {'n': 0}

    def flaky_urlopen(*_args: object, **_kwargs: object) -> MagicMock:
        posts['n'] += 1
        if posts['n'] == 1:
            raise ValueError('not a transport error')
        return mock_urlopen_response()

    with patch(
        'genkit._core._trace._default_exporter.urllib.request.urlopen',
        side_effect=flaky_urlopen,
    ):
        exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')
        with capture_logs() as entries:
            exporter.export([create_mock_span(trace_id=0xA)])
            assert exporter.force_flush(timeout_millis=2000) is True
            exporter.export([create_mock_span(trace_id=0xB)])
            assert exporter.force_flush(timeout_millis=2000) is True

    assert exporter.worker.is_alive()
    assert posts['n'] == 2
    assert exporter.last_result == SpanExportResult.SUCCESS
    errors = [e for e in entries if 'Failed to save trace' in str(e.get('event', ''))]
    assert len(errors) == 1
    assert format(0xA, '032x') in errors[0]['event']


def test_export_encode_bug_is_loud() -> None:
    """A span that cannot serialize must raise — not become a quiet FAILURE."""
    exporter = TraceServerExporter(telemetry_server_url='http://127.0.0.1:1')
    mock_span = create_mock_span()
    mock_span.context = None

    with pytest.raises(TypeError, match='span context is required'), capture_logs() as entries:
        exporter.export([mock_span])

    assert exporter.last_result == SpanExportResult.SUCCESS
    assert not [e for e in entries if 'Failed to save trace' in str(e.get('event', ''))]


@patch('genkit._core._trace._default_exporter.urllib.request.urlopen')
def test_export_skips_non_json_attribute(mock_urlopen: MagicMock) -> None:
    """A bytes attribute is dropped; generate() returns and the rest of the trace POSTs."""
    mock_urlopen.return_value = mock_urlopen_response()
    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')
    mock_span = create_mock_span(attributes={'ok': 'hi', 'payload': b'secret-bytes'})

    with capture_logs() as entries:
        result = exporter.export([mock_span])
        assert exporter.force_flush(timeout_millis=2000) is True

    assert result == SpanExportResult.SUCCESS
    mock_urlopen.assert_called_once()
    body = json.loads(mock_urlopen.call_args.args[0].data)
    span_id = format(67890, '016x')
    assert body['spans'][span_id]['attributes'] == {'ok': 'hi'}
    skipped = [e for e in entries if 'skipped span attribute' in str(e.get('event', ''))]
    assert len(skipped) == 1
    assert "'payload'" in skipped[0]['event']
    assert 'bytes' in skipped[0]['event']


# =============================================================================
# Tests for extract_span_data
# =============================================================================


def test_extract_span_data_requires_context() -> None:
    """A span without context is a TypeError, not an AttributeError inside format()."""
    mock_span = create_mock_span()
    mock_span.context = None
    with pytest.raises(TypeError, match='span context is required'):
        extract_span_data(mock_span)


def test_extract_span_data_basic_fields() -> None:
    """Test that extract_span_data extracts basic span fields correctly."""
    mock_span = create_mock_span(
        trace_id=12345,
        span_id=67890,
        name='test-span',
        start_time=1000000000,  # 1000ms in nanoseconds
        end_time=2000000000,  # 2000ms in nanoseconds
    )

    data = extract_span_data(mock_span)

    assert type(data) is TraceData
    trace_id_hex = format(12345, '032x')
    span_id_hex = format(67890, '016x')

    assert data.trace_id == trace_id_hex
    assert span_id_hex in data.spans

    span_info = data.spans[span_id_hex]
    assert span_info.span_id == span_id_hex
    assert span_info.trace_id == trace_id_hex
    assert span_info.display_name == 'test-span'
    assert span_info.start_time == 1000.0
    assert span_info.end_time == 2000.0


def test_extract_span_data_with_attributes() -> None:
    """Test that extract_span_data includes span attributes."""
    mock_span = create_mock_span(attributes={'key1': 'value1', 'key2': 123})

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data.spans[span_id_hex]
    assert span_info.attributes == {'key1': 'value1', 'key2': 123}


def test_extract_span_data_skips_non_json_attributes() -> None:
    """A value json.dumps cannot store is dropped; neighbors stay on the document."""
    mock_span = create_mock_span(
        attributes={'ok': 'hi', 'payload': b'secret-bytes', 'also': object()},
    )

    with capture_logs() as entries:
        data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    assert data.spans[span_id_hex].attributes == {'ok': 'hi'}
    skipped = [e['event'] for e in entries if 'skipped span attribute' in str(e.get('event', ''))]
    assert any("'payload'" in event and 'bytes' in event for event in skipped)
    assert any("'also'" in event and 'object' in event for event in skipped)


def test_extract_span_data_with_parent_span() -> None:
    """Test that extract_span_data includes parent span ID when present."""
    mock_parent = MagicMock()
    mock_parent.span_id = 11111

    mock_span = create_mock_span()
    mock_span.parent = mock_parent

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    parent_span_id_hex = format(11111, '016x')
    span_info = data.spans[span_id_hex]
    assert span_info.parent_span_id == parent_span_id_hex


def test_extract_span_data_without_parent_span() -> None:
    """Test that extract_span_data omits parent span ID when not present."""
    mock_span = create_mock_span()
    mock_span.parent = None

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data.spans[span_id_hex]
    assert span_info.parent_span_id is None
    assert data.display_name == 'test-span'


def test_extract_span_data_includes_status() -> None:
    """Test that extract_span_data includes span status."""
    mock_span = create_mock_span()

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data.spans[span_id_hex]
    assert span_info.status is not None
    assert span_info.status.code == trace_api.StatusCode.OK.value
    assert span_info.status.message is None


def test_extract_span_data_includes_instrumentation_library() -> None:
    """Test that extract_span_data includes instrumentation library info."""
    mock_span = create_mock_span()

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data.spans[span_id_hex]
    assert span_info.instrumentation_library.name == 'genkit-tracer'
    assert span_info.instrumentation_library.version == 'v1'


def test_extract_span_data_handles_none_times() -> None:
    """Test that extract_span_data handles None start/end times."""
    mock_span = create_mock_span(start_time=None, end_time=None)

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data.spans[span_id_hex]
    assert span_info.start_time == 0
    assert span_info.end_time == 0


def test_extract_span_data_ensures_exception_message_from_status_when_events_empty() -> None:
    """If OTel events are missing but status is ERROR with description, Dev UI still gets a message."""
    mock_span = create_mock_span()
    mock_status = MagicMock()
    mock_status.status_code = trace_api.StatusCode.ERROR
    mock_status.description = 'patched from status only'
    mock_span.status = mock_status
    mock_span.events = ()

    data = extract_span_data(mock_span)
    span_id_hex = format(67890, '016x')
    span_info = data.spans[span_id_hex]
    assert span_info.status is not None
    assert span_info.status.code == 2
    assert span_info.status.message == 'patched from status only'
    assert span_info.time_events is not None
    events = span_info.time_events.time_event
    assert events is not None
    assert len(events) == 1
    assert events[0].annotation.attributes['exception.message'] == 'patched from status only'


def test_extract_span_data_includes_exception_time_events() -> None:
    """OTel exception events must appear as timeEvents so Dev UI shows the message (not plain 'Error')."""
    exc_msg = 'DEV_UI_ERROR_TRACE_TEST_2026: deliberate failure'
    ev = Event(
        'exception',
        attributes={
            'exception.type': 'RuntimeError',
            'exception.message': exc_msg,
            'exception.stacktrace': 'traceback...',
        },
        timestamp=1_500_000_000,
    )
    mock_span = create_mock_span(events=(ev,))

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data.spans[span_id_hex]
    assert span_info.time_events is not None
    events = span_info.time_events.time_event
    assert events is not None
    assert len(events) == 1
    assert events[0].annotation.description == 'exception'
    assert events[0].annotation.attributes['exception.message'] == exc_msg
    assert events[0].time == 1500.0


# =============================================================================
# Helper functions
# =============================================================================


def mock_urlopen_response() -> MagicMock:
    """A urlopen() context manager that drains like a successful POST."""
    response = MagicMock()
    response.read.return_value = b''
    context = MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = None
    return context


def create_mock_span(
    trace_id: int = 12345,
    span_id: int = 67890,
    name: str = 'test-span',
    start_time: int | None = 1000000000,
    end_time: int | None = 2000000000,
    attributes: dict | None = None,
    events: tuple[Event, ...] | None = None,
) -> MagicMock:
    """Create a mock ReadableSpan for testing.

    Args:
        trace_id: The trace ID.
        span_id: The span ID.
        name: The span name.
        start_time: Start time in nanoseconds.
        end_time: End time in nanoseconds.
        attributes: Optional span attributes.

    Returns:
        A MagicMock configured as a ReadableSpan.
    """
    mock_span = MagicMock(spec=ReadableSpan)

    # Configure context
    mock_context = MagicMock()
    mock_context.trace_id = trace_id
    mock_context.span_id = span_id
    mock_span.context = mock_context

    # Configure basic properties
    mock_span.name = name
    mock_span.start_time = start_time
    mock_span.end_time = end_time
    mock_span.attributes = attributes or {}
    mock_span.parent = None

    # Configure kind
    mock_span.kind = trace_api.SpanKind.INTERNAL

    # Configure status
    mock_status = MagicMock()
    mock_status.status_code = trace_api.StatusCode.OK
    mock_status.description = None
    mock_span.status = mock_status

    mock_span.events = events if events is not None else ()

    return mock_span
