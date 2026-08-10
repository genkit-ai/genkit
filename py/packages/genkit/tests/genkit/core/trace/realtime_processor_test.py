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

"""Tests for RealtimeSpanProcessor.

This module tests the RealtimeSpanProcessor which exports spans both when
they start and when they end, enabling real-time trace visualization.
"""

from collections.abc import Sequence
from unittest.mock import MagicMock

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from genkit._core._trace._realtime_processor import RealtimeSpanProcessor


class MockSpanExporter(SpanExporter):
    """Mock exporter for testing."""

    def __init__(self) -> None:
        """Initialize the mock exporter."""
        self.exported_spans: list[Sequence[ReadableSpan]] = []
        self.shutdown_called = False
        self.force_flush_called = False
        self.force_flush_timeout: int | None = None

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Record exported spans."""
        self.exported_spans.append(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Record shutdown call."""
        self.shutdown_called = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Record force_flush call."""
        self.force_flush_called = True
        self.force_flush_timeout = timeout_millis
        return True


def test_realtime_processor_exports_on_start() -> None:
    """Test that RealtimeSpanProcessor exports spans when they start.

    This verifies the key behavior that differentiates RealtimeSpanProcessor
    from standard processors - it exports spans immediately on start.
    """
    exporter = MockSpanExporter()
    processor = RealtimeSpanProcessor(exporter)

    # Create a mock span
    mock_span = MagicMock(spec=Span)

    # Call on_start
    processor.on_start(mock_span)

    # Verify span was exported (on_start passes list)
    assert len(exporter.exported_spans) == 1
    assert list(exporter.exported_spans[0]) == [mock_span]


def test_realtime_processor_exports_on_end() -> None:
    """Test that RealtimeSpanProcessor exports spans when they end.

    This verifies that completed spans are also exported with full data.
    """
    exporter = MockSpanExporter()
    processor = RealtimeSpanProcessor(exporter)

    # Create a mock ReadableSpan (completed span)
    mock_span = MagicMock(spec=ReadableSpan)

    # Call on_end
    processor.on_end(mock_span)

    # Verify span was exported (SimpleSpanProcessor passes tuple)
    assert len(exporter.exported_spans) == 1
    assert list(exporter.exported_spans[0]) == [mock_span]


def test_realtime_processor_exports_twice_for_full_lifecycle() -> None:
    """Test that a span is exported both on start and end.

    This is the defining characteristic of RealtimeSpanProcessor - each span
    results in two exports for live visualization.
    """
    exporter = MockSpanExporter()
    processor = RealtimeSpanProcessor(exporter)

    # Create mock spans for start and end
    mock_span_start = MagicMock(spec=Span)
    mock_span_end = MagicMock(spec=ReadableSpan)

    # Simulate full span lifecycle
    processor.on_start(mock_span_start)
    processor.on_end(mock_span_end)

    # Verify span was exported twice (on_start uses list, on_end uses tuple)
    assert len(exporter.exported_spans) == 2
    assert list(exporter.exported_spans[0]) == [mock_span_start]
    assert list(exporter.exported_spans[1]) == [mock_span_end]


def test_realtime_processor_force_flush() -> None:
    """Test that force_flush works (inherited from SimpleSpanProcessor)."""
    exporter = MockSpanExporter()
    processor = RealtimeSpanProcessor(exporter)

    result = processor.force_flush(timeout_millis=5000)

    assert result is True


def test_realtime_processor_shutdown_delegates_to_exporter() -> None:
    """Test that shutdown is delegated to the underlying exporter."""
    exporter = MockSpanExporter()
    processor = RealtimeSpanProcessor(exporter)

    # Call shutdown
    processor.shutdown()

    # Verify delegation
    assert exporter.shutdown_called is True


def test_realtime_processor_on_start_with_parent_context() -> None:
    """Test that on_start accepts optional parent_context parameter."""
    exporter = MockSpanExporter()
    processor = RealtimeSpanProcessor(exporter)

    mock_span = MagicMock(spec=Span)
    mock_context = MagicMock(spec=Context)

    # Call on_start with parent_context (should be ignored but accepted)
    processor.on_start(mock_span, parent_context=mock_context)

    # Verify span was still exported
    assert len(exporter.exported_spans) == 1
    assert list(exporter.exported_spans[0]) == [mock_span]


def test_realtime_processor_multiple_spans() -> None:
    """Test that multiple spans can be processed correctly."""
    exporter = MockSpanExporter()
    processor = RealtimeSpanProcessor(exporter)

    # Create multiple mock spans
    spans_start = [MagicMock(spec=Span) for _ in range(3)]
    spans_end = [MagicMock(spec=ReadableSpan) for _ in range(3)]

    # Process all spans
    for span in spans_start:
        processor.on_start(span)
    for span in spans_end:
        processor.on_end(span)

    # Verify all exports
    assert len(exporter.exported_spans) == 6
