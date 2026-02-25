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

"""Tests for AWS X-Ray telemetry integration in the amazon-bedrock plugin.

Covers:
- Region resolution from environment variables
- X-Ray OTLP endpoint configuration
- AwsXRayOtlpExporter with SigV4 authentication
- SigV4SigningAdapter for HTTPS requests
- AwsTelemetry manager class
- TimeAdjustedSpan for zero-duration spans
- Log-trace correlation
- Error handling for missing credentials
"""

import os
from unittest import mock

import pytest
import requests

from genkit.plugins.amazon_bedrock.telemetry.tracing import (
    XRAY_OTLP_ENDPOINT_PATTERN,
    AwsAdjustingTraceExporter,
    AwsTelemetry,
    AwsXRayOtlpExporter,
    SigV4SigningAdapter,
    TimeAdjustedSpan,
    _create_sigv4_session,
    _resolve_region,
    add_aws_telemetry,
)


class TestResolveRegion:
    """Tests for the _resolve_region function."""

    def test_explicit_region_takes_precedence(self) -> None:
        """Explicit region parameter should override environment variables."""
        with mock.patch.dict(os.environ, {'AWS_REGION': 'us-east-1'}):
            result = _resolve_region(region='eu-west-1')
            assert result == 'eu-west-1'

    def test_aws_region_env_var(self) -> None:
        """AWS_REGION environment variable should be used when no explicit region."""
        with mock.patch.dict(os.environ, {'AWS_REGION': 'us-west-2'}, clear=True):
            result = _resolve_region()
            assert result == 'us-west-2'

    def test_aws_default_region_fallback(self) -> None:
        """AWS_DEFAULT_REGION should be used as fallback."""
        env = {'AWS_DEFAULT_REGION': 'ap-southeast-1'}
        with mock.patch.dict(os.environ, env, clear=True):
            result = _resolve_region()
            assert result == 'ap-southeast-1'

    def test_aws_region_priority_over_default(self) -> None:
        """AWS_REGION should take priority over AWS_DEFAULT_REGION."""
        env = {
            'AWS_REGION': 'us-east-1',
            'AWS_DEFAULT_REGION': 'us-west-2',
        }
        with mock.patch.dict(os.environ, env, clear=True):
            result = _resolve_region()
            assert result == 'us-east-1'

    def test_no_region_returns_none(self) -> None:
        """Should return None when no region is configured."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _resolve_region()
            assert result is None


class TestXRayOtlpEndpoint:
    """Tests for X-Ray OTLP endpoint configuration."""

    def test_endpoint_pattern_format(self) -> None:
        """Endpoint pattern should produce correct URLs."""
        endpoint = XRAY_OTLP_ENDPOINT_PATTERN.format(region='us-west-2')
        assert endpoint == 'https://xray.us-west-2.amazonaws.com/v1/traces'

    def test_endpoint_pattern_different_regions(self) -> None:
        """Endpoint should work for various AWS regions."""
        regions = ['us-east-1', 'eu-west-1', 'ap-southeast-1', 'sa-east-1']
        for region in regions:
            endpoint = XRAY_OTLP_ENDPOINT_PATTERN.format(region=region)
            assert region in endpoint
            assert endpoint.startswith('https://xray.')
            assert endpoint.endswith('/v1/traces')


class TestAwsXRayOtlpExporter:
    """Tests for the AwsXRayOtlpExporter class."""

    def test_exporter_initialization(self) -> None:
        """Exporter should initialize with region."""
        exporter = AwsXRayOtlpExporter(region='us-west-2')
        assert exporter._region == 'us-west-2'
        assert 'us-west-2' in exporter._endpoint

    def test_exporter_with_error_handler(self) -> None:
        """Exporter should accept error handler callback."""
        errors: list[Exception] = []
        exporter = AwsXRayOtlpExporter(
            region='us-west-2',
            error_handler=lambda e: errors.append(e),
        )
        assert exporter._error_handler is not None

    def test_exporter_uses_sigv4_session(self) -> None:
        """Exporter should use a session with SigV4 signing adapter mounted."""
        exporter = AwsXRayOtlpExporter(region='us-west-2')
        # The OTLP exporter should have a session configured
        assert exporter._otlp_exporter._session is not None


class TestSigV4SigningAdapter:
    """Tests for the SigV4SigningAdapter class."""

    def test_adapter_initialization(self) -> None:
        """Adapter should initialize with credentials and region."""
        mock_credentials = mock.MagicMock()
        adapter = SigV4SigningAdapter(
            credentials=mock_credentials,
            region='us-west-2',
            service='xray',
        )
        assert adapter._credentials == mock_credentials
        assert adapter._region == 'us-west-2'
        assert adapter._service == 'xray'

    def test_adapter_default_service(self) -> None:
        """Adapter should default to xray service."""
        mock_credentials = mock.MagicMock()
        adapter = SigV4SigningAdapter(
            credentials=mock_credentials,
            region='eu-west-1',
        )
        assert adapter._service == 'xray'

    def test_adapter_signs_request(self) -> None:
        """Adapter should add SigV4 headers to request."""
        # Create mock credentials
        mock_credentials = mock.MagicMock()
        mock_credentials.access_key = 'AKIAIOSFODNN7EXAMPLE'
        mock_credentials.secret_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
        mock_credentials.token = None

        adapter = SigV4SigningAdapter(
            credentials=mock_credentials,
            region='us-west-2',
            service='xray',
        )

        # Create a mock request
        request = requests.PreparedRequest()
        request.prepare(
            method='POST',
            url='https://xray.us-west-2.amazonaws.com/v1/traces',
            headers={'Content-Type': 'application/x-protobuf'},
            data=b'test-payload',
        )

        # Mock the parent send method
        with mock.patch.object(
            adapter.__class__.__bases__[0],
            'send',
            return_value=mock.MagicMock(status_code=200),
        ):
            adapter.send(request)

        # Verify that the request now has Authorization header
        assert request.headers is not None
        assert 'Authorization' in request.headers
        assert 'AWS4-HMAC-SHA256' in request.headers['Authorization']

    def test_adapter_handles_none_credentials(self) -> None:
        """Adapter should handle None credentials gracefully."""
        adapter = SigV4SigningAdapter(
            credentials=None,
            region='us-west-2',
        )

        request = requests.PreparedRequest()
        request.prepare(
            method='POST',
            url='https://xray.us-west-2.amazonaws.com/v1/traces',
            headers={},
            data=b'test',
        )

        # Should not raise, just skip signing
        with mock.patch.object(
            adapter.__class__.__bases__[0],
            'send',
            return_value=mock.MagicMock(status_code=200),
        ):
            adapter.send(request)

        # No Authorization header should be set
        assert request.headers is not None
        assert 'Authorization' not in request.headers


class TestCreateSigV4Session:
    """Tests for the _create_sigv4_session function."""

    def test_session_has_adapter_mounted(self) -> None:
        """Session should have SigV4 adapter mounted for HTTPS."""
        mock_credentials = mock.MagicMock()
        session = _create_sigv4_session(
            credentials=mock_credentials,
            region='us-west-2',
        )

        # Get the adapter for an HTTPS URL
        adapter = session.get_adapter('https://xray.us-west-2.amazonaws.com')
        assert isinstance(adapter, SigV4SigningAdapter)

    def test_session_adapter_has_correct_region(self) -> None:
        """Session adapter should be configured with correct region."""
        mock_credentials = mock.MagicMock()
        session = _create_sigv4_session(
            credentials=mock_credentials,
            region='eu-west-1',
            service='xray',
        )

        adapter = session.get_adapter('https://xray.eu-west-1.amazonaws.com')
        assert isinstance(adapter, SigV4SigningAdapter)
        assert adapter._region == 'eu-west-1'
        assert adapter._service == 'xray'


class TestAwsAdjustingTraceExporter:
    """Tests for the AwsAdjustingTraceExporter class."""

    def test_exporter_initialization(self) -> None:
        """Adjusting exporter should wrap base exporter."""
        base_exporter = mock.MagicMock()
        exporter = AwsAdjustingTraceExporter(
            exporter=base_exporter,
            log_input_and_output=False,
            region='us-west-2',
        )
        assert exporter._exporter is base_exporter
        assert exporter._log_input_and_output is False
        assert exporter._region == 'us-west-2'

    def test_exporter_with_logging_enabled(self) -> None:
        """Adjusting exporter should respect log_input_and_output flag."""
        base_exporter = mock.MagicMock()
        exporter = AwsAdjustingTraceExporter(
            exporter=base_exporter,
            log_input_and_output=True,
        )
        assert exporter._log_input_and_output is True


class TestTimeAdjustedSpan:
    """Tests for the TimeAdjustedSpan class."""

    def test_zero_duration_span_adjusted(self) -> None:
        """Spans with zero duration should get minimum 1 microsecond."""
        mock_span = mock.MagicMock()
        mock_span.start_time = 1000000000  # 1 second in nanoseconds
        mock_span.end_time = 1000000000  # Same as start (zero duration)
        mock_span.attributes = {}

        adjusted = TimeAdjustedSpan(mock_span, {})
        # Should add 1000 nanoseconds (1 microsecond)
        assert adjusted.end_time == 1000001000

    def test_none_end_time_adjusted(self) -> None:
        """Spans with None end_time should get adjusted."""
        mock_span = mock.MagicMock()
        mock_span.start_time = 1000000000
        mock_span.end_time = None
        mock_span.attributes = {}

        adjusted = TimeAdjustedSpan(mock_span, {})
        assert adjusted.end_time == 1000001000

    def test_valid_duration_unchanged(self) -> None:
        """Spans with valid duration should remain unchanged."""
        mock_span = mock.MagicMock()
        mock_span.start_time = 1000000000
        mock_span.end_time = 2000000000  # 1 second later
        mock_span.attributes = {}

        adjusted = TimeAdjustedSpan(mock_span, {})
        assert adjusted.end_time == 2000000000


class TestAwsTelemetry:
    """Tests for the AwsTelemetry manager class."""

    def test_initialization_with_explicit_region(self) -> None:
        """Manager should accept explicit region."""
        with mock.patch.dict(os.environ, {}, clear=True):
            telemetry = AwsTelemetry(region='us-west-2')
            assert telemetry.region == 'us-west-2'

    def test_initialization_with_env_region(self) -> None:
        """Manager should use AWS_REGION env var."""
        with mock.patch.dict(os.environ, {'AWS_REGION': 'eu-west-1'}, clear=True):
            telemetry = AwsTelemetry()
            assert telemetry.region == 'eu-west-1'

    def test_initialization_raises_without_region(self) -> None:
        """Manager should raise ValueError without region."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='AWS region is required'):
                AwsTelemetry()

    def test_default_configuration(self) -> None:
        """Manager should have correct defaults."""
        with mock.patch.dict(os.environ, {'AWS_REGION': 'us-west-2'}):
            telemetry = AwsTelemetry()
            assert telemetry.log_input_and_output is False
            assert telemetry.force_dev_export is True
            assert telemetry.disable_traces is False

    def test_inject_trace_context_no_span(self) -> None:
        """Trace context injection should handle no active span."""
        with mock.patch.dict(os.environ, {'AWS_REGION': 'us-west-2'}):
            telemetry = AwsTelemetry()
            event_dict: dict[str, str] = {'message': 'test'}

            with mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.trace') as mock_trace:
                mock_trace.get_current_span.return_value = mock_trace.INVALID_SPAN
                result = telemetry._inject_trace_context(event_dict)

            assert '_X_AMZN_TRACE_ID' not in result


class TestAddAwsTelemetry:
    """Tests for the add_aws_telemetry function."""

    def test_raises_without_region(self) -> None:
        """Should raise ValueError when region is not configured."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='AWS region is required'):
                add_aws_telemetry()

    def test_accepts_explicit_region(self) -> None:
        """Should accept explicit region parameter."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Mock the exporter to avoid actual export
            with mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.add_custom_exporter') as mock_add:
                with mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.propagate'):
                    with mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.trace'):
                        with mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.structlog'):
                            add_aws_telemetry(region='us-west-2')
                            mock_add.assert_called_once()

    def test_skips_in_dev_without_force(self) -> None:
        """Should skip telemetry in dev environment without force_dev_export."""
        with (
            mock.patch.dict(os.environ, {'AWS_REGION': 'us-west-2'}),
            mock.patch(
                'genkit.plugins.amazon_bedrock.telemetry.tracing.is_dev_environment',
                return_value=True,
            ),
            mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.add_custom_exporter') as mock_add,
        ):
            add_aws_telemetry(force_dev_export=False)
            mock_add.assert_not_called()

    def test_exports_in_dev_with_force(self) -> None:
        """Should export telemetry in dev environment with force_dev_export=True."""
        with (
            mock.patch.dict(os.environ, {'AWS_REGION': 'us-west-2'}),
            mock.patch(
                'genkit.plugins.amazon_bedrock.telemetry.tracing.is_dev_environment',
                return_value=True,
            ),
            mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.add_custom_exporter') as mock_add,
        ):
            with mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.propagate'):
                with mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.trace'):
                    with mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.structlog'):
                        add_aws_telemetry(force_dev_export=True)
                        mock_add.assert_called_once()

    def test_disable_traces(self) -> None:
        """Should not add exporter when disable_traces=True."""
        with mock.patch.dict(os.environ, {'AWS_REGION': 'us-west-2'}):
            with mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.add_custom_exporter') as mock_add:
                with mock.patch('genkit.plugins.amazon_bedrock.telemetry.tracing.structlog'):
                    add_aws_telemetry(disable_traces=True)
                    mock_add.assert_not_called()


class TestTelemetryExportFromPlugin:
    """Tests for the add_aws_telemetry export from the main plugin."""

    def test_add_aws_telemetry_exported(self) -> None:
        """add_aws_telemetry should be exported from the main plugin."""
        from genkit.plugins.amazon_bedrock import add_aws_telemetry as exported_fn

        # Verify it's the same function
        assert exported_fn is add_aws_telemetry

    def test_add_aws_telemetry_in_all(self) -> None:
        """add_aws_telemetry should be in __all__."""
        from genkit.plugins import amazon_bedrock

        assert 'add_aws_telemetry' in amazon_bedrock.__all__
