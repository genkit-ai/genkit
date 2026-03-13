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

"""Tests for the GCP telemetry tracing module.

This module tests the integration of GcpAdjustingTraceExporter with the
GCP telemetry plugin, ensuring PII redaction and telemetry recording work correctly.

Tests cover JS/Go parity for:
- Configuration options (project_id, credentials, sampler, etc.)
- PII redaction (log_input_and_output)
- Environment-based export control (force_dev_export)
- Metrics and traces disable flags
- Metric export interval/timeout
"""

import os
from unittest import mock
from unittest.mock import MagicMock, patch

from genkit.core.environment import EnvVar, GenkitEnvironment


def test_add_gcp_telemetry_wraps_with_gcp_adjusting_exporter() -> None:
    """Test that add_gcp_telemetry wraps the exporter with GcpAdjustingTraceExporter."""
    # Set production environment and clear project-related env vars to ensure project_id is None
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.PROD}, clear=False),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit.plugins.google_cloud.telemetry.config.GcpAdjustingTraceExporter') as mock_adjusting,
        patch('genkit.plugins.google_cloud.telemetry.config.add_custom_exporter') as mock_add_exporter,
        patch('genkit.plugins.google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit.plugins.google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit.plugins.google_cloud.telemetry.config.metrics'),
    ):
        # Remove project env vars to ensure project_id is None in the test
        for key in ['FIREBASE_PROJECT_ID', 'GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT']:
            os.environ.pop(key, None)

        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # Create mock instances
        mock_base_exporter = MagicMock()
        mock_gcp_exporter.return_value = mock_base_exporter

        mock_wrapped_exporter = MagicMock()
        mock_adjusting.return_value = mock_wrapped_exporter

        # Call the function
        add_gcp_telemetry()

        # Verify GenkitGCPExporter was created
        mock_gcp_exporter.assert_called_once()

        # Verify GcpAdjustingTraceExporter was created with correct args
        mock_adjusting.assert_called_once()
        call_kwargs = mock_adjusting.call_args.kwargs
        assert call_kwargs['exporter'] == mock_base_exporter
        assert call_kwargs['log_input_and_output'] is False  # Default is redaction enabled
        assert call_kwargs['project_id'] is None

        # Verify the wrapped exporter was added
        mock_add_exporter.assert_called_once_with(mock_wrapped_exporter, 'gcp_telemetry_server')


def test_add_gcp_telemetry_with_log_input_and_output_enabled() -> None:
    """Test that log_input_and_output=True disables PII redaction (JS parity)."""
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.PROD}),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GcpAdjustingTraceExporter') as mock_adjusting,
        patch('genkit.plugins.google_cloud.telemetry.config.add_custom_exporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit.plugins.google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit.plugins.google_cloud.telemetry.config.metrics'),
    ):
        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # Call with log_input_and_output=True (maps to JS: !disableLoggingInputAndOutput)
        add_gcp_telemetry(log_input_and_output=True)

        # Verify log_input_and_output was passed correctly
        call_kwargs = mock_adjusting.call_args.kwargs
        assert call_kwargs['log_input_and_output'] is True


def test_add_gcp_telemetry_with_project_id() -> None:
    """Test that project_id is passed to GcpAdjustingTraceExporter (JS/Go parity)."""
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.PROD}),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GcpAdjustingTraceExporter') as mock_adjusting,
        patch('genkit.plugins.google_cloud.telemetry.config.add_custom_exporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit.plugins.google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit.plugins.google_cloud.telemetry.config.metrics'),
    ):
        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # Call with project_id
        add_gcp_telemetry(project_id='my-test-project')

        # Verify project_id was passed correctly
        call_kwargs = mock_adjusting.call_args.kwargs
        assert call_kwargs['project_id'] == 'my-test-project'


def test_add_gcp_telemetry_skips_in_dev_without_force() -> None:
    """Test that telemetry is skipped in dev environment without force_dev_export (JS/Go parity)."""
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.DEV}),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit.plugins.google_cloud.telemetry.config.add_custom_exporter') as mock_add_exporter,
    ):
        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # Call without force_dev_export (using legacy force_export)
        add_gcp_telemetry(force_dev_export=False)

        # Verify nothing was called
        mock_gcp_exporter.assert_not_called()
        mock_add_exporter.assert_not_called()


def test_add_gcp_telemetry_exports_in_dev_with_force() -> None:
    """Test that telemetry is exported in dev environment with force_dev_export=True (JS/Go parity)."""
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.DEV}),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit.plugins.google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.add_custom_exporter') as mock_add_exporter,
        patch('genkit.plugins.google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit.plugins.google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit.plugins.google_cloud.telemetry.config.metrics'),
    ):
        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # Call with force_dev_export=True (the default)
        add_gcp_telemetry(force_dev_export=True)

        # Verify exporter was created and added
        mock_gcp_exporter.assert_called_once()
        mock_add_exporter.assert_called_once()


def test_add_gcp_telemetry_disable_traces() -> None:
    """Test that disable_traces=True skips trace export (JS/Go parity)."""
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.PROD}),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit.plugins.google_cloud.telemetry.config.add_custom_exporter') as mock_add_exporter,
        patch('genkit.plugins.google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit.plugins.google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit.plugins.google_cloud.telemetry.config.metrics'),
    ):
        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # Call with disable_traces=True (JS/Go: disableTraces)
        add_gcp_telemetry(disable_traces=True)

        # Verify trace exporter was NOT created
        mock_gcp_exporter.assert_not_called()
        mock_add_exporter.assert_not_called()


def test_add_gcp_telemetry_disable_metrics() -> None:
    """Test that disable_metrics=True skips metrics export (JS/Go parity)."""
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.PROD}),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.add_custom_exporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GoogleCloudResourceDetector') as mock_detector,
        patch('genkit.plugins.google_cloud.telemetry.config.CloudMonitoringMetricsExporter') as mock_metric_exp,
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitMetricExporter') as mock_genkit_metric,
        patch('genkit.plugins.google_cloud.telemetry.config.PeriodicExportingMetricReader') as mock_reader,
        patch('genkit.plugins.google_cloud.telemetry.config.metrics'),
    ):
        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # Call with disable_metrics=True (JS/Go: disableMetrics)
        add_gcp_telemetry(disable_metrics=True)

        # Verify metrics exporter was NOT created
        mock_detector.assert_not_called()
        mock_metric_exp.assert_not_called()
        mock_genkit_metric.assert_not_called()
        mock_reader.assert_not_called()


def test_add_gcp_telemetry_custom_metric_interval() -> None:
    """Test that metric_export_interval_ms is passed correctly (JS/Go parity)."""
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.PROD}),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.add_custom_exporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit.plugins.google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.PeriodicExportingMetricReader') as mock_reader,
        patch('genkit.plugins.google_cloud.telemetry.config.metrics'),
    ):
        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # Call with custom metric_export_interval_ms (JS/Go: metricExportIntervalMillis)
        add_gcp_telemetry(metric_export_interval_ms=30000)

        # Verify metric reader was created with correct interval
        mock_reader.assert_called_once()
        call_kwargs = mock_reader.call_args.kwargs
        assert call_kwargs['export_interval_millis'] == 30000
        assert call_kwargs['export_timeout_millis'] == 30000  # Default to interval


def test_add_gcp_telemetry_enforces_minimum_interval() -> None:
    """Test that metric_export_interval_ms enforces minimum 5000ms (GCP requirement)."""
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.PROD}),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.add_custom_exporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit.plugins.google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.PeriodicExportingMetricReader') as mock_reader,
        patch('genkit.plugins.google_cloud.telemetry.config.metrics'),
    ):
        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # Call with interval below minimum
        add_gcp_telemetry(metric_export_interval_ms=1000)

        # Verify metric reader was created with minimum interval (5000ms)
        mock_reader.assert_called_once()
        call_kwargs = mock_reader.call_args.kwargs
        assert call_kwargs['export_interval_millis'] == 5000


def test_resolve_project_id_from_env_vars() -> None:
    """Test project ID resolution from environment variables (JS/Go parity)."""
    from genkit.plugins.google_cloud.telemetry.config import resolve_project_id

    # Test FIREBASE_PROJECT_ID has highest priority
    with mock.patch.dict(
        os.environ,
        {
            'FIREBASE_PROJECT_ID': 'firebase-project',
            'GOOGLE_CLOUD_PROJECT': 'gcp-project',
            'GCLOUD_PROJECT': 'gcloud-project',
        },
    ):
        assert resolve_project_id() == 'firebase-project'

    # Test GOOGLE_CLOUD_PROJECT is second priority
    with mock.patch.dict(
        os.environ,
        {
            'GOOGLE_CLOUD_PROJECT': 'gcp-project',
            'GCLOUD_PROJECT': 'gcloud-project',
        },
        clear=True,
    ):
        assert resolve_project_id() == 'gcp-project'

    # Test GCLOUD_PROJECT is fallback
    with mock.patch.dict(os.environ, {'GCLOUD_PROJECT': 'gcloud-project'}, clear=True):
        assert resolve_project_id() == 'gcloud-project'


def test_resolve_project_id_explicit_takes_precedence() -> None:
    """Test that explicit project_id parameter takes precedence over env vars."""
    from genkit.plugins.google_cloud.telemetry.config import resolve_project_id

    with mock.patch.dict(
        os.environ,
        {'FIREBASE_PROJECT_ID': 'firebase-project'},
    ):
        # Explicit project_id should override env var
        assert resolve_project_id(project_id='explicit-project') == 'explicit-project'


def test_resolve_project_id_from_credentials() -> None:
    """Test project ID resolution from credentials dict (Go parity)."""
    from genkit.plugins.google_cloud.telemetry.config import resolve_project_id

    with mock.patch.dict(os.environ, {}, clear=True):
        # Project ID from credentials
        credentials = {'project_id': 'creds-project'}
        assert resolve_project_id(credentials=credentials) == 'creds-project'


def test_legacy_force_export_parameter() -> None:
    """Test that legacy force_export parameter still works but shows warning."""
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.DEV}),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit.plugins.google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.add_custom_exporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit.plugins.google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit.plugins.google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit.plugins.google_cloud.telemetry.config.metrics'),
        patch('genkit.plugins.google_cloud.telemetry.tracing.logger') as mock_logger,
    ):
        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # Call with legacy force_export parameter
        add_gcp_telemetry(force_export=True)

        # Verify warning was logged about deprecated parameter
        mock_logger.warning.assert_called_once()
        assert 'force_export' in str(mock_logger.warning.call_args)
        assert 'deprecated' in str(mock_logger.warning.call_args)

        # Verify exporter was still created
        mock_gcp_exporter.assert_called_once()


def test_add_gcp_telemetry_is_fail_safe() -> None:
    """Test that add_gcp_telemetry does not crash if initialization fails."""
    with (
        mock.patch.dict(os.environ, {EnvVar.GENKIT_ENV: GenkitEnvironment.PROD}),
        patch(
            'genkit.plugins.google_cloud.telemetry.config.GenkitGCPExporter',
            side_effect=Exception('Auth failed'),
        ),
        patch('genkit.plugins.google_cloud.telemetry.config.handle_tracing_error') as mock_handler,
    ):
        from genkit.plugins.google_cloud.telemetry.tracing import add_gcp_telemetry

        # This should NOT raise an exception
        try:
            add_gcp_telemetry()
        except Exception as e:
            raise AssertionError(f'add_gcp_telemetry raised an exception: {e}') from e

        # Verify error handler was called
        mock_handler.assert_called_once()
