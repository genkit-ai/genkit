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

"""Tests for AdjustingTraceExporter.

This module tests the AdjustingTraceExporter which adjusts spans before
exporting, including:
- PII redaction (genkit:input/output)
- Error span marking with HTTP status code
- Failed span marking
- Feature and model marking
- Label normalization (: -> /)
"""

import contextlib
from collections.abc import Mapping, Sequence
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.types import Attributes

from genkit.core.trace.adjusting_exporter import AdjustingTraceExporter, RedactedSpan


class MockSpanExporter(SpanExporter):
    """Mock exporter for testing."""

    def __init__(self) -> None:
        """Initialize the mock exporter."""
        self.exported_spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Record exported spans."""
        self.exported_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """No-op shutdown."""
        pass


def create_mock_span(
    attributes: dict[str, Any] | None = None,
    status_code: StatusCode = StatusCode.OK,
) -> MagicMock:
    """Create a mock ReadableSpan for testing."""
    mock_span = MagicMock(spec=ReadableSpan)
    mock_span.attributes = attributes or {}

    # Configure status
    mock_status = MagicMock(spec=Status)
    mock_status.status_code = status_code
    mock_span.status = mock_status

    return mock_span


def get_attrs(span: ReadableSpan) -> dict[str, Any]:
    """Get attributes from a span as a dict."""
    attrs: Attributes | None = span.attributes
    if attrs is None:
        return {}
    return dict(cast(Mapping[str, Any], attrs))


def test_redacts_input_and_output_by_default() -> None:
    """Test that genkit:input and genkit:output are redacted by default."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter, log_input_and_output=False)

    # Use colon format - will be normalized to slash
    span = create_mock_span(
        attributes={
            'genkit:input': 'sensitive input data',
            'genkit:output': 'sensitive output data',
            'other': 'preserved',
        }
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    # After normalization, colons become slashes
    assert attrs['genkit/input'] == '<redacted>'
    assert attrs['genkit/output'] == '<redacted>'
    assert attrs['other'] == 'preserved'


def test_preserves_input_and_output_when_logging_enabled() -> None:
    """Test that input/output are preserved when log_input_and_output=True."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter, log_input_and_output=True)

    span = create_mock_span(
        attributes={
            'genkit:input': 'sensitive input data',
            'genkit:output': 'sensitive output data',
        }
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    # After normalization, colons become slashes
    assert attrs['genkit/input'] == 'sensitive input data'
    assert attrs['genkit/output'] == 'sensitive output data'


def test_handles_missing_input_output() -> None:
    """Test that spans without input/output are not modified for redaction."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter, log_input_and_output=False)

    span = create_mock_span(attributes={'other': 'value'})

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    assert 'genkit/input' not in attrs
    assert 'genkit/output' not in attrs
    assert attrs['other'] == 'value'


def test_marks_error_span_with_http_status() -> None:
    """Test that error spans get /http/status_code: 599 for GCP display."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter)

    span = create_mock_span(
        attributes={'genkit:name': 'test'},
        status_code=StatusCode.ERROR,
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    assert attrs['/http/status_code'] == '599'


def test_does_not_mark_ok_span_with_http_status() -> None:
    """Test that OK spans do not get HTTP status code marker."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter)

    span = create_mock_span(
        attributes={'genkit:name': 'test'},
        status_code=StatusCode.OK,
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    assert '/http/status_code' not in attrs


def test_marks_failed_span_with_failure_info() -> None:
    """Test that failure source spans get failedSpan and failedPath markers."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter)

    span = create_mock_span(
        attributes={
            'genkit:isFailureSource': True,
            'genkit:name': 'failing-action',
            'genkit:path': '/flow/step1',
        }
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    # After normalization, colons become slashes
    assert attrs['genkit/failedSpan'] == 'failing-action'
    assert attrs['genkit/failedPath'] == '/flow/step1'


def test_does_not_mark_non_failure_span() -> None:
    """Test that non-failure spans do not get failure markers."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter)

    span = create_mock_span(
        attributes={
            'genkit:name': 'normal-action',
            'genkit:path': '/flow/step1',
        }
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    assert 'genkit/failedSpan' not in attrs
    assert 'genkit/failedPath' not in attrs


def test_marks_root_span_with_feature() -> None:
    """Test that root spans get genkit:feature attribute."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter)

    span = create_mock_span(
        attributes={
            'genkit:isRoot': True,
            'genkit:name': 'myFlow',
        }
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    # After normalization, colons become slashes
    assert attrs['genkit/feature'] == 'myFlow'


def test_does_not_mark_non_root_span_with_feature() -> None:
    """Test that non-root spans do not get genkit:feature attribute."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter)

    span = create_mock_span(attributes={'genkit:name': 'myFlow'})

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    assert 'genkit/feature' not in attrs


def test_marks_model_span_with_model_name() -> None:
    """Test that model spans get genkit:model attribute."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter)

    span = create_mock_span(
        attributes={
            'genkit:metadata:subtype': 'model',
            'genkit:name': 'gemini-2.0-flash',
        }
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    # After normalization, colons become slashes
    assert attrs['genkit/model'] == 'gemini-2.0-flash'


def test_does_not_mark_non_model_span_with_model() -> None:
    """Test that non-model spans do not get genkit:model attribute."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter)

    span = create_mock_span(
        attributes={
            'genkit:metadata:subtype': 'tool',
            'genkit:name': 'myTool',
        }
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    assert 'genkit/model' not in attrs


def test_normalizes_labels_colon_to_slash() -> None:
    """Test that colons in attribute keys are replaced with slashes."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter)

    span = create_mock_span(
        attributes={
            'genkit:name': 'test',
            'genkit:type': 'action',
            'genkit:metadata:subtype': 'model',
            'normal_key': 'value',
        }
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])
    # All colons should be replaced with slashes
    assert 'genkit/name' in attrs
    assert 'genkit/type' in attrs
    assert 'genkit/metadata/subtype' in attrs
    assert 'normal_key' in attrs
    # Original colon keys should not exist
    assert 'genkit:name' not in attrs
    assert 'genkit:type' not in attrs


def test_applies_all_transformations_in_order() -> None:
    """Test that all transformations are applied correctly to a complex span."""
    exporter = MockSpanExporter()
    adjusting = AdjustingTraceExporter(exporter, log_input_and_output=False)

    span = create_mock_span(
        attributes={
            'genkit:isRoot': True,
            'genkit:name': 'myFlow',
            'genkit:input': 'sensitive',
            'genkit:output': 'sensitive',
            'genkit:path': '/myFlow',
            'genkit:type': 'flow',
        },
        status_code=StatusCode.ERROR,
    )

    adjusting.export([span])

    attrs = get_attrs(exporter.exported_spans[0])

    # Check redaction (colons normalized to slashes first)
    assert attrs['genkit/input'] == '<redacted>'
    assert attrs['genkit/output'] == '<redacted>'

    # Check error marking
    assert attrs['/http/status_code'] == '599'

    # Check feature marking
    assert attrs['genkit/feature'] == 'myFlow'

    # Check label normalization
    assert 'genkit/name' in attrs
    assert 'genkit:name' not in attrs


def test_error_handler_called_on_export_error() -> None:
    """Test that error_handler is called when export fails."""
    mock_exporter = MagicMock(spec=SpanExporter)
    mock_exporter.export.side_effect = Exception('Export failed')

    errors: list[Exception] = []
    adjusting = AdjustingTraceExporter(
        mock_exporter,
        error_handler=lambda e: errors.append(e),
    )

    span = create_mock_span()

    with contextlib.suppress(Exception):
        adjusting.export([span])

    assert len(errors) == 1
    assert str(errors[0]) == 'Export failed'


@pytest.mark.parametrize(
    ('property_name', 'value'),
    [
        ('dropped_attributes', 0),
        ('dropped_events', 2),
        ('dropped_links', 1),
    ],
)
def test_redacted_span_dropped_properties_delegate_to_inner_span(property_name: str, value: int) -> None:
    """Regression test: RedactedSpan.dropped_* properties must not raise.

    The OTLP trace encoder accesses these properties during serialization.
    Before the fix, RedactedSpan did not call ``super().__init__()`` so the
    private fields required by the base ``ReadableSpan`` properties were
    missing, causing an ``AttributeError``.  This test verifies that the
    overridden properties delegate to the wrapped span correctly.
    """
    inner = create_mock_span()
    setattr(inner, property_name, value)
    span = RedactedSpan(inner, {})
    assert getattr(span, property_name) == value
