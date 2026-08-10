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

import os
from unittest import mock
from unittest.mock import MagicMock, patch

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult

from genkit._core._environment import GENKIT_ENV, GenkitEnvironment
from genkit._core._trace._default_exporter import (
    TraceServerExporter,
    create_span_processor,
    extract_span_data,
    init_telemetry_server_exporter,
)
from genkit._core._trace._realtime_processor import RealtimeSpanProcessor

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


# =============================================================================
# Tests for TraceServerExporter
# =============================================================================


def test_telemetry_server_exporter_init_default_endpoint() -> None:
    """Test TraceServerExporter initialization with default endpoint."""
    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')

    assert exporter.telemetry_server_url == 'http://localhost:4000'
    assert exporter.telemetry_server_endpoint == '/api/traces'


def test_telemetry_server_exporter_init_custom_endpoint() -> None:
    """Test TraceServerExporter initialization with custom endpoint."""
    exporter = TraceServerExporter(
        telemetry_server_url='http://localhost:4000',
        telemetry_server_endpoint='/custom/traces',
    )

    assert exporter.telemetry_server_url == 'http://localhost:4000'
    assert exporter.telemetry_server_endpoint == '/custom/traces'


def test_telemetry_server_exporter_force_flush_returns_true() -> None:
    """Test that force_flush always returns True (no buffering)."""
    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')

    result = exporter.force_flush()
    assert result is True


def test_telemetry_server_exporter_force_flush_ignores_timeout() -> None:
    """Test that force_flush ignores the timeout parameter."""
    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')

    result = exporter.force_flush(timeout_millis=1)
    assert result is True


@patch('genkit._core._trace._default_exporter.httpx.Client')
def test_telemetry_server_exporter_export_sends_http_post(mock_client_class: MagicMock) -> None:
    """Test that export sends HTTP POST requests for each span."""
    # Setup mock client
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')

    # Create a mock span
    mock_span = create_mock_span()

    # Export
    result = exporter.export([mock_span])

    # Verify
    assert result == SpanExportResult.SUCCESS
    mock_client.post.assert_called_once()


@patch('genkit._core._trace._default_exporter.httpx.Client')
def test_telemetry_server_exporter_export_multiple_spans(mock_client_class: MagicMock) -> None:
    """Test that export sends HTTP POST for each span in the sequence."""
    # Setup mock client
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

    exporter = TraceServerExporter(telemetry_server_url='http://localhost:4000')

    # Create multiple mock spans
    mock_spans = [create_mock_span() for _ in range(3)]

    # Export
    result = exporter.export(mock_spans)

    # Verify
    assert result == SpanExportResult.SUCCESS
    assert mock_client.post.call_count == 3


# =============================================================================
# Tests for extract_span_data
# =============================================================================


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

    trace_id_hex = format(12345, '032x')
    span_id_hex = format(67890, '016x')

    assert data['traceId'] == trace_id_hex
    assert 'spans' in data
    assert span_id_hex in data['spans']

    span_info = data['spans'][span_id_hex]
    assert span_info['spanId'] == span_id_hex
    assert span_info['traceId'] == trace_id_hex
    assert span_info['displayName'] == 'test-span'
    assert span_info['startTime'] == 1000.0  # Converted to milliseconds
    assert span_info['endTime'] == 2000.0  # Converted to milliseconds


def test_extract_span_data_with_attributes() -> None:
    """Test that extract_span_data includes span attributes."""
    mock_span = create_mock_span(attributes={'key1': 'value1', 'key2': 123})

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data['spans'][span_id_hex]
    assert span_info['attributes'] == {'key1': 'value1', 'key2': 123}


def test_extract_span_data_with_parent_span() -> None:
    """Test that extract_span_data includes parent span ID when present."""
    mock_parent = MagicMock()
    mock_parent.span_id = 11111

    mock_span = create_mock_span()
    mock_span.parent = mock_parent

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    parent_span_id_hex = format(11111, '016x')
    span_info = data['spans'][span_id_hex]
    assert span_info['parentSpanId'] == parent_span_id_hex


def test_extract_span_data_without_parent_span() -> None:
    """Test that extract_span_data omits parent span ID when not present."""
    mock_span = create_mock_span()
    mock_span.parent = None

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data['spans'][span_id_hex]
    assert 'parentSpanId' not in span_info

    # Root span should have displayName, startTime, endTime at top level
    assert data['displayName'] == 'test-span'


def test_extract_span_data_includes_status() -> None:
    """Test that extract_span_data includes span status."""
    mock_span = create_mock_span()

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data['spans'][span_id_hex]
    assert 'status' in span_info
    assert span_info['status']['code'] == trace_api.StatusCode.OK.value  # OK status is 1
    assert 'message' not in span_info['status']


def test_extract_span_data_includes_instrumentation_library() -> None:
    """Test that extract_span_data includes instrumentation library info."""
    mock_span = create_mock_span()

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data['spans'][span_id_hex]
    assert span_info['instrumentationLibrary'] == {
        'name': 'genkit-tracer',
        'version': 'v1',
    }


def test_extract_span_data_handles_none_times() -> None:
    """Test that extract_span_data handles None start/end times."""
    mock_span = create_mock_span(start_time=None, end_time=None)

    data = extract_span_data(mock_span)

    span_id_hex = format(67890, '016x')
    span_info = data['spans'][span_id_hex]
    assert span_info['startTime'] == 0
    assert span_info['endTime'] == 0


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
    span_info = data['spans'][span_id_hex]
    assert span_info['status']['code'] == 2
    assert span_info['status']['message'] == 'patched from status only'
    te = span_info['timeEvents']['timeEvent']
    assert len(te) == 1
    assert te[0]['annotation']['attributes']['exception.message'] == 'patched from status only'


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
    span_info = data['spans'][span_id_hex]
    assert 'timeEvents' in span_info
    te = span_info['timeEvents']['timeEvent']
    assert len(te) == 1
    assert te[0]['annotation']['description'] == 'exception'
    assert te[0]['annotation']['attributes']['exception.message'] == exc_msg
    assert te[0]['time'] == 1500.0


# =============================================================================
# Helper functions
# =============================================================================


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
