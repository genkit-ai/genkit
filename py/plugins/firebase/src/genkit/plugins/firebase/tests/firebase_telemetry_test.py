# Copyright 2025 Google LLC
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

"""Tests for Firebase telemetry functionality."""

from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import ReadableSpan

from genkit.plugins.firebase import add_firebase_telemetry
from genkit.plugins.google_cloud.telemetry.metrics import record_generate_metrics


def _create_model_span(
    model_name: str = 'gemini-pro',
    path: str = '/{myflow,t:flow}',
    output: str = '{"usage": {"inputTokens": 100, "outputTokens": 50}}',
    is_ok: bool = True,
    start_time: int = 1000000000,
    end_time: int = 1500000000,
) -> MagicMock:
    """Helper function to create a model action span for testing.

    Args:
        model_name: The model name for genkit:name attribute
        path: The genkit:path value
        output: The genkit:output JSON string
        is_ok: Whether the span status is ok
        start_time: Span start time in nanoseconds
        end_time: Span end time in nanoseconds

    Returns:
        A mocked ReadableSpan with model action attributes
    """
    mock_span = MagicMock(spec=ReadableSpan)
    mock_span.attributes = {
        'genkit:type': 'action',
        'genkit:metadata:subtype': 'model',
        'genkit:name': model_name,
        'genkit:path': path,
        'genkit:output': output,
    }
    mock_span.status.is_ok = is_ok
    mock_span.start_time = start_time
    mock_span.end_time = end_time
    return mock_span


@patch('genkit.plugins.firebase.telemetry.GcpTelemetry')
def test_firebase_telemetry_initializes_gcp_telemetry(mock_gcp_telemetry_cls: MagicMock) -> None:
    """Test that Firebase telemetry initializes GcpTelemetry with correct defaults."""
    mock_manager = MagicMock()
    mock_gcp_telemetry_cls.return_value = mock_manager

    add_firebase_telemetry()

    mock_gcp_telemetry_cls.assert_called_once()
    kwargs = mock_gcp_telemetry_cls.call_args.kwargs
    assert kwargs['force_dev_export'] is False
    mock_manager.initialize.assert_called_once()


def test_firebase_telemetry_passes_configuration() -> None:
    """Test that configuration options are passed to GcpTelemetry."""
    with patch('genkit.plugins.firebase.telemetry.GcpTelemetry') as mock_gcp_telemetry_cls:
        add_firebase_telemetry(
            project_id='test-project',
            log_input_and_output=True,
            force_dev_export=True,
            disable_metrics=True,
        )

        kwargs = mock_gcp_telemetry_cls.call_args.kwargs
        assert kwargs['project_id'] == 'test-project'
        assert kwargs['log_input_and_output'] is True
        assert kwargs['force_dev_export'] is True
        assert kwargs['disable_metrics'] is True


@patch('genkit.plugins.google_cloud.telemetry.metrics._output_tokens')
@patch('genkit.plugins.google_cloud.telemetry.metrics._input_tokens')
@patch('genkit.plugins.google_cloud.telemetry.metrics._latency')
@patch('genkit.plugins.google_cloud.telemetry.metrics._failures')
@patch('genkit.plugins.google_cloud.telemetry.metrics._requests')
def test_record_generate_metrics_with_model_action(
    mock_requests: MagicMock,
    mock_failures: MagicMock,
    mock_latency: MagicMock,
    mock_input_tokens: MagicMock,
    mock_output_tokens: MagicMock,
) -> None:
    """Test that metrics are recorded for model action spans with usage data."""
    # Setup mocks
    mock_request_counter = MagicMock()
    mock_latency_histogram = MagicMock()
    mock_input_counter = MagicMock()
    mock_output_counter = MagicMock()

    mock_requests.return_value = mock_request_counter
    mock_failures.return_value = MagicMock()
    mock_latency.return_value = mock_latency_histogram
    mock_input_tokens.return_value = mock_input_counter
    mock_output_tokens.return_value = mock_output_counter

    # Create test span using helper
    mock_span = _create_model_span(
        model_name='gemini-pro',
        path='/{myflow,t:flow}',
        output='{"usage": {"inputTokens": 100, "outputTokens": 50}}',
    )

    # Execute
    record_generate_metrics(mock_span)

    # Verify dimensions
    expected_dimensions = {'model': 'gemini-pro', 'source': 'myflow', 'error': 'none'}

    # Verify requests counter
    mock_request_counter.add.assert_called_once_with(1, expected_dimensions)

    # Verify latency (500ms = 1.5s - 1.0s)
    mock_latency_histogram.record.assert_called_once_with(500.0, expected_dimensions)

    # Verify token counts
    mock_input_counter.add.assert_called_once_with(100, expected_dimensions)
    mock_output_counter.add.assert_called_once_with(50, expected_dimensions)
