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

"""What enable_google_cloud_telemetry() does to Cloud Trace and the Developer UI."""

import os
from collections.abc import Generator
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from genkit_google_cloud.telemetry.config import resolve_project_id
from genkit_google_cloud.telemetry.tracing import (
    _reset_google_cloud_telemetry,
    enable_google_cloud_telemetry,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from genkit._core._error import GenkitError
from genkit._core._telemetry._instrumentation import reset_instrumentation
from genkit._core._telemetry._log_exporter import reset_log_export

# Environment variable and value constants (matching genkit._core._environment)
_GENKIT_ENV = 'GENKIT_ENV'
_ENV_DEV = 'dev'
_ENV_PROD = 'prod'


@pytest.fixture(autouse=True)
def _reset_instrumentation() -> Generator[None, None, None]:
    reset_instrumentation()
    reset_log_export()
    _reset_google_cloud_telemetry()
    yield
    reset_instrumentation()
    reset_log_export()
    _reset_google_cloud_telemetry()


def test_enable_google_cloud_telemetry_wraps_with_gcp_adjusting_exporter() -> None:
    """enable_google_cloud_telemetry() sends Cloud Trace through the adjusting exporter."""
    # Set production environment and clear project-related env vars to ensure project_id is None
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}, clear=False),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter') as mock_adjusting,
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer') as mock_add_exporter,
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        # Remove project env vars to ensure project_id is None in the test
        for key in ['FIREBASE_PROJECT_ID', 'GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT']:
            os.environ.pop(key, None)

        # Create mock instances
        mock_base_exporter = MagicMock()
        mock_gcp_exporter.return_value = mock_base_exporter

        mock_wrapped_exporter = MagicMock()
        mock_adjusting.return_value = mock_wrapped_exporter

        # Call the function
        enable_google_cloud_telemetry()

        # Verify GenkitGCPExporter was created
        mock_gcp_exporter.assert_called_once()

        # Verify GcpAdjustingTraceExporter was created with correct args
        mock_adjusting.assert_called_once()
        call_kwargs = mock_adjusting.call_args.kwargs
        assert call_kwargs['exporter'] == mock_base_exporter
        assert call_kwargs['log_input_and_output'] is False  # Default is redaction enabled
        assert call_kwargs['project_id'] is None

        # Verify the wrapped exporter was added
        mock_add_exporter.assert_called_once_with(exporter=mock_wrapped_exporter)


def test_enable_google_cloud_telemetry_with_log_input_and_output_enabled() -> None:
    """log_input_and_output=True leaves prompt and response on the Cloud Trace span."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter') as mock_adjusting,
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        # Call with log_input_and_output=True (maps to JS: !disableLoggingInputAndOutput)
        enable_google_cloud_telemetry(log_input_and_output=True)

        # Verify log_input_and_output was passed correctly
        call_kwargs = mock_adjusting.call_args.kwargs
        assert call_kwargs['log_input_and_output'] is True


def test_enable_google_cloud_telemetry_with_project_id() -> None:
    """project_id= lands on the Cloud Trace exporter they get."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter') as mock_adjusting,
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        # Call with project_id
        enable_google_cloud_telemetry(project_id='my-test-project')

        # Verify project_id was passed correctly
        call_kwargs = mock_adjusting.call_args.kwargs
        assert call_kwargs['project_id'] == 'my-test-project'


def test_enable_google_cloud_telemetry_skips_in_dev_without_force() -> None:
    """Under genkit start, enable does nothing unless they pass force_dev_export."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_DEV}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer') as mock_add_exporter,
    ):
        # Call without force_dev_export (using legacy force_export)
        enable_google_cloud_telemetry(force_dev_export=False)

        # Verify nothing was called
        mock_gcp_exporter.assert_not_called()
        mock_add_exporter.assert_not_called()


def test_enable_google_cloud_telemetry_exports_in_dev_with_force() -> None:
    """force_dev_export=True under genkit start still sends Cloud Trace."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_DEV}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer') as mock_add_exporter,
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        enable_google_cloud_telemetry(force_dev_export=True)

        mock_gcp_exporter.assert_called_once()
        mock_add_exporter.assert_called_once()


def test_enable_google_cloud_telemetry_disable_traces() -> None:
    """disable_traces=True skips Cloud Trace and still turns metrics on."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer') as mock_add_exporter,
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        # Call with disable_traces=True (JS/Go: disableTraces)
        enable_google_cloud_telemetry(disable_traces=True)

        # Verify trace exporter was NOT created
        mock_gcp_exporter.assert_not_called()
        mock_add_exporter.assert_not_called()


def test_enable_google_cloud_telemetry_disable_metrics() -> None:
    """disable_metrics=True skips Cloud Monitoring and still turns traces on."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector') as mock_detector,
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter') as mock_metric_exp,
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter') as mock_genkit_metric,
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader') as mock_reader,
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        # Call with disable_metrics=True (JS/Go: disableMetrics)
        enable_google_cloud_telemetry(disable_metrics=True)

        # Verify metrics exporter was NOT created
        mock_detector.assert_not_called()
        mock_metric_exp.assert_not_called()
        mock_genkit_metric.assert_not_called()
        mock_reader.assert_not_called()


def test_enable_google_cloud_telemetry_custom_metric_interval() -> None:
    """metric_export_interval_ms= is the Cloud Monitoring scrape interval."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader') as mock_reader,
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        # Call with custom metric_export_interval_ms (JS/Go: metricExportIntervalMillis)
        enable_google_cloud_telemetry(metric_export_interval_ms=30000)

        # Verify metric reader was created with correct interval
        mock_reader.assert_called_once()
        call_kwargs = mock_reader.call_args.kwargs
        assert call_kwargs['export_interval_millis'] == 30000
        assert call_kwargs['export_timeout_millis'] == 30000  # Default to interval


def test_enable_google_cloud_telemetry_enforces_minimum_interval() -> None:
    """A metric interval under 5s is raised to 5s; Cloud Monitoring rejects faster."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader') as mock_reader,
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        # Call with interval below minimum
        enable_google_cloud_telemetry(metric_export_interval_ms=1000)

        # Verify metric reader was created with minimum interval (5000ms)
        mock_reader.assert_called_once()
        call_kwargs = mock_reader.call_args.kwargs
        assert call_kwargs['export_interval_millis'] == 5000


def test_resolve_project_id_from_env_vars() -> None:
    """GOOGLE_CLOUD_PROJECT / FIREBASE_PROJECT_ID / GCLOUD_PROJECT all resolve a project."""
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
    """An explicit project_id= wins over the environment."""
    with mock.patch.dict(
        os.environ,
        {'FIREBASE_PROJECT_ID': 'firebase-project'},
    ):
        # Explicit project_id should override env var
        assert resolve_project_id(project_id='explicit-project') == 'explicit-project'


def test_resolve_project_id_from_credentials() -> None:
    """A credentials dict with project_id is enough when env is empty."""
    with mock.patch.dict(os.environ, {}, clear=True):
        # Project ID from credentials
        credentials = {'project_id': 'creds-project'}
        assert resolve_project_id(credentials=credentials) == 'creds-project'


def test_legacy_force_export_parameter() -> None:
    """force_export= still works and warns; prefer force_dev_export=."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_DEV}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
        patch('genkit_google_cloud.telemetry.tracing.logger') as mock_logger,
    ):
        # Call with legacy force_export parameter
        enable_google_cloud_telemetry(force_export=True)

        # Verify warning was logged about deprecated parameter
        mock_logger.warning.assert_called_once()
        assert 'force_export' in str(mock_logger.warning.call_args)
        assert 'deprecated' in str(mock_logger.warning.call_args)

        # Verify exporter was still created
        mock_gcp_exporter.assert_called_once()


def test_enable_google_cloud_telemetry_is_fail_safe() -> None:
    """A Cloud Trace auth failure does not crash the process."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch(
            'genkit_google_cloud.telemetry.config.GenkitGCPExporter',
            side_effect=Exception('Auth failed'),
        ),
        patch('genkit_google_cloud.telemetry.config.handle_tracing_error') as mock_handler,
    ):
        # This should NOT raise an exception
        try:
            enable_google_cloud_telemetry()
        except Exception as e:
            raise AssertionError(f'enable_google_cloud_telemetry raised an exception: {e}') from e

        # Verify error handler was called
        mock_handler.assert_called_once()


def test_enable_google_cloud_telemetry_called_twice_raises() -> None:
    """A second enable_google_cloud_telemetry() raises; Cloud still has one exporter."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}, clear=False),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer') as mock_add_exporter,
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        enable_google_cloud_telemetry(project_id='my-project')
        mock_add_exporter.assert_called_once()
        with pytest.raises(GenkitError, match='already called') as raised:
            enable_google_cloud_telemetry(project_id='other')
        assert raised.value.status == 'FAILED_PRECONDITION'
        mock_add_exporter.assert_called_once()


def test_enable_in_dev_without_force_then_again_raises() -> None:
    """First enable in local skips Cloud; a second enable still raises."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_DEV}),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer') as mock_add_exporter,
    ):
        enable_google_cloud_telemetry(force_dev_export=False)
        mock_add_exporter.assert_not_called()
        with pytest.raises(GenkitError, match='already called'):
            enable_google_cloud_telemetry(force_dev_export=True)
        mock_add_exporter.assert_not_called()


def test_leftover_collector_env_in_prod_does_not_block_cloud() -> None:
    """A leftover GENKIT_TELEMETRY_SERVER in prod does not block Cloud spans."""
    with (
        mock.patch.dict(
            os.environ,
            {_GENKIT_ENV: _ENV_PROD, 'GENKIT_TELEMETRY_SERVER': 'http://127.0.0.1:4033'},
            clear=False,
        ),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        enable_google_cloud_telemetry(project_id='my-project')


def _processors(provider: TracerProvider) -> tuple[object, ...]:
    active = getattr(provider, '_active_span_processor', None)
    if active is None:
        return ()
    return tuple(getattr(active, '_span_processors', (active,)))


def test_enable_in_prod_batches_cloud_spans() -> None:
    """enable() in prod hangs Cloud Trace on a batch processor."""
    isolated = TracerProvider()
    try:
        with (
            mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}, clear=False),
            patch(
                'genkit_google_cloud.telemetry.config.trace_api.get_tracer_provider',
                return_value=isolated,
            ),
            patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
            patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
            patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
            patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
            patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
            patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
            patch('genkit_google_cloud.telemetry.config.metrics'),
        ):
            enable_google_cloud_telemetry(project_id='my-project')
        assert any(isinstance(proc, BatchSpanProcessor) for proc in _processors(isolated))
    finally:
        isolated.shutdown()


def test_enable_under_genkit_start_with_force_exports_on_span_end() -> None:
    """force_dev_export=True under genkit start exports Cloud on span end."""
    isolated = TracerProvider()
    try:
        with (
            mock.patch.dict(
                os.environ,
                {_GENKIT_ENV: _ENV_DEV, 'GENKIT_TELEMETRY_SERVER': 'http://127.0.0.1:4033'},
                clear=False,
            ),
            patch(
                'genkit_google_cloud.telemetry.config.trace_api.get_tracer_provider',
                return_value=isolated,
            ),
            patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
            patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
            patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
            patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
            patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
            patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
            patch('genkit_google_cloud.telemetry.config.metrics'),
        ):
            enable_google_cloud_telemetry(force_dev_export=True, project_id='my-project')
        assert any(isinstance(proc, SimpleSpanProcessor) for proc in _processors(isolated))
    finally:
        isolated.shutdown()
