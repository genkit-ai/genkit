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

import inspect
import os
from collections.abc import Generator
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from genkit_google_cloud.telemetry.config import GcpTelemetry, _adc_project_id
from genkit_google_cloud.telemetry.tracing import (
    _reset_google_cloud_telemetry,
    enable_google_cloud_telemetry,
)
from genkit_otel import GenAiInstrumentation
from google.auth.exceptions import DefaultCredentialsError
from opentelemetry import _logs
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from genkit._core._error import GenkitError
from genkit.telemetry import (
    configure_instrumentation,
    is_instrumented_by,
    reset_instrumentation,
)

_GENKIT_ENV = 'GENKIT_ENV'
_ENV_DEV = 'dev'
_ENV_PROD = 'prod'


@pytest.fixture(autouse=True)
def _reset_instrumentation() -> Generator[None, None, None]:
    reset_instrumentation()
    _reset_google_cloud_telemetry()
    yield
    reset_instrumentation()
    _reset_google_cloud_telemetry()


@pytest.fixture(autouse=True)
def _stub_cloud_logging_exporter() -> Generator[MagicMock, None, None]:
    with patch('genkit_google_cloud.telemetry.config.CloudLoggingExporter') as mock_exporter:
        yield mock_exporter


@pytest.fixture(autouse=True)
def _adc_project() -> Generator[MagicMock, None, None]:
    """ADC lookup stub; returns no project unless a test sets one."""
    with patch('genkit_google_cloud.telemetry.config._adc_project_id', return_value=None) as mock_adc:
        yield mock_adc


def test_enable_google_cloud_telemetry_wraps_with_gcp_adjusting_exporter() -> None:
    """enable_google_cloud_telemetry() sends Cloud Trace through the adjusting exporter."""
    # Set production environment and clear project-related env vars to ensure project is None
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
        # Remove project env vars to ensure project is None in the test
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

        # Verify the wrapped exporter was added
        mock_add_exporter.assert_called_once_with(exporter=mock_wrapped_exporter, sampler=None)


def test_enable_google_cloud_telemetry_rejects_log_input_and_output() -> None:
    """log_input_and_output= is gone. Prompt I/O is capture_action_io on GenAiInstrumentation."""
    with pytest.raises(TypeError, match='log_input_and_output'):
        enable_google_cloud_telemetry(log_input_and_output=True)  # ty: ignore[unknown-argument]


def test_enable_google_cloud_telemetry_with_project() -> None:
    """project= lands on the Cloud Trace exporter as its project_id."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        enable_google_cloud_telemetry(project='my-test-project')

        assert mock_gcp_exporter.call_args.kwargs.get('project_id') == 'my-test-project'


def test_enable_google_cloud_telemetry_skips_in_dev_without_force() -> None:
    """Under genkit start, enable does nothing unless they pass force_dev_export."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_DEV}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer') as mock_add_exporter,
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_logger') as mock_add_logger,
    ):
        enable_google_cloud_telemetry(force_dev_export=False)

        # Verify nothing was called
        mock_gcp_exporter.assert_not_called()
        mock_add_exporter.assert_not_called()
        mock_add_logger.assert_not_called()
        assert not is_instrumented_by(GenAiInstrumentation)


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
        assert is_instrumented_by(GenAiInstrumentation)


def test_enable_disable_traces_with_force_dev_export_still_turns_on_genai() -> None:
    """force_dev_export=True and disable_traces=True still turn GenAI on."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_DEV}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer') as mock_add_exporter,
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        enable_google_cloud_telemetry(force_dev_export=True, disable_traces=True)

        mock_gcp_exporter.assert_not_called()
        mock_add_exporter.assert_not_called()
        assert is_instrumented_by(GenAiInstrumentation)


def test_enable_disable_traces_skips_cloud_trace_and_still_turns_on_genai() -> None:
    """disable_traces=True does not hang Cloud Trace; generate is still instrumented."""
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
        enable_google_cloud_telemetry(disable_traces=True)

        mock_gcp_exporter.assert_not_called()
        mock_add_exporter.assert_not_called()
        assert is_instrumented_by(GenAiInstrumentation)


def test_enable_disable_traces_and_metrics_still_turns_on_genai() -> None:
    """disable_traces=True and disable_metrics=True still turn GenAI on."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter') as mock_gcp_exporter,
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer') as mock_add_exporter,
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector') as mock_detector,
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter') as mock_metric_exp,
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter') as mock_genkit_metric,
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader') as mock_reader,
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        enable_google_cloud_telemetry(disable_traces=True, disable_metrics=True)

        mock_gcp_exporter.assert_not_called()
        mock_add_exporter.assert_not_called()
        mock_detector.assert_not_called()
        mock_metric_exp.assert_not_called()
        mock_genkit_metric.assert_not_called()
        mock_reader.assert_not_called()
        assert is_instrumented_by(GenAiInstrumentation)


def test_enable_disable_traces_keeps_their_genai_settings() -> None:
    """A GenAiInstrumentation they registered before enable() keeps its settings."""
    theirs = GenAiInstrumentation(emit_metrics=False)
    configure_instrumentation(theirs)
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        enable_google_cloud_telemetry(disable_traces=True)

    assert is_instrumented_by(GenAiInstrumentation)
    assert theirs.emit_metrics is False


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
        enable_google_cloud_telemetry(disable_metrics=True)

        # Verify metrics exporter was NOT created
        mock_detector.assert_not_called()
        mock_metric_exp.assert_not_called()
        mock_genkit_metric.assert_not_called()
        mock_reader.assert_not_called()
        assert is_instrumented_by(GenAiInstrumentation)


def test_enable_hangs_cloud_logging_so_an_emit_reaches_the_exporter(
    _stub_cloud_logging_exporter: MagicMock,
) -> None:
    """enable() hangs Cloud Logging; an OTel emit reaches that exporter."""
    memory = InMemoryLogRecordExporter()
    _stub_cloud_logging_exporter.return_value = memory
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        enable_google_cloud_telemetry()

    _stub_cloud_logging_exporter.assert_called_once()
    assert _stub_cloud_logging_exporter.call_args.kwargs.get('default_log_name') == 'genkit'
    logger = _logs.get_logger('genkit-genai')
    logger.emit(event_name='gen_ai.client.inference.operation.details', attributes={'probe': '1'})
    provider = _logs.get_logger_provider()
    if isinstance(provider, LoggerProvider):
        provider.force_flush()
    records = memory.get_finished_logs()
    assert records


def test_enable_adds_a_logging_processor_when_they_already_own_a_logger_provider(
    _stub_cloud_logging_exporter: MagicMock,
) -> None:
    """A LoggerProvider they already set keeps their logger; Cloud hangs on it."""
    existing = LoggerProvider()
    added: list[object] = []
    original = existing.add_log_record_processor

    def _spy(processor: object) -> None:
        added.append(processor)
        original(processor)

    existing.add_log_record_processor = _spy  # type: ignore[method-assign]
    _stub_cloud_logging_exporter.return_value = InMemoryLogRecordExporter()
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config._logs.get_logger_provider', return_value=existing),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        enable_google_cloud_telemetry()

    assert added


def test_enable_hangs_cloud_logging_when_traces_are_disabled(
    _stub_cloud_logging_exporter: MagicMock,
) -> None:
    """disable_traces=True still hangs Cloud Logging."""
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_logger') as mock_add_logger,
    ):
        enable_google_cloud_telemetry(disable_traces=True)

    mock_add_logger.assert_called_once()


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


def _exporter_project(mock_ctor: MagicMock) -> str | None:
    mock_ctor.assert_called_once()
    return mock_ctor.call_args.kwargs.get('project_id')


@pytest.mark.parametrize(
    ('env', 'kwargs', 'expected'),
    [
        pytest.param(
            {'FIREBASE_PROJECT_ID': 'firebase-proj', 'GOOGLE_CLOUD_PROJECT': 'gcp-proj'},
            {},
            'gcp-proj',
            id='firebase_ignored_google_cloud_project_wins',
        ),
        pytest.param(
            {'FIREBASE_PROJECT_ID': 'firebase-proj'},
            {'credentials': {'project_id': 'creds-proj'}},
            'creds-proj',
            id='firebase_ignored_credentials_win',
        ),
        pytest.param(
            {'FIREBASE_PROJECT_ID': 'firebase-proj'},
            {},
            None,
            id='firebase_only_sends_no_project',
        ),
        pytest.param(
            {'GOOGLE_CLOUD_PROJECT': 'gcp-proj'},
            {'project': 'explicit-proj'},
            'explicit-proj',
            id='project_beats_google_cloud_project',
        ),
        pytest.param(
            {'GOOGLE_CLOUD_PROJECT': 'gcp-proj', 'GCLOUD_PROJECT': 'gcloud-proj'},
            {},
            'gcp-proj',
            id='google_cloud_project_beats_gcloud_project',
        ),
        pytest.param(
            {'GOOGLE_CLOUD_PROJECT': '', 'GCLOUD_PROJECT': 'gcloud-proj'},
            {},
            'gcloud-proj',
            id='empty_google_cloud_project_falls_through',
        ),
        pytest.param(
            {'GOOGLE_CLOUD_PROJECT': 'gcp-proj'},
            {'credentials': {'project_id': 'creds-proj'}},
            'gcp-proj',
            id='google_cloud_project_beats_credentials',
        ),
        pytest.param(
            {},
            {'credentials': {'project_id': 'creds-proj'}},
            'creds-proj',
            id='credentials_are_last_fallback',
        ),
    ],
)
def test_enable_sends_traces_metrics_and_logs_to_the_resolved_cloud_project(
    monkeypatch: pytest.MonkeyPatch,
    _stub_cloud_logging_exporter: MagicMock,
    env: dict[str, str],
    kwargs: dict[str, Any],
    expected: str | None,
) -> None:
    """project=, then GOOGLE_CLOUD_PROJECT, then GCLOUD_PROJECT, then credentials; FIREBASE_PROJECT_ID is ignored."""
    for key in ('FIREBASE_PROJECT_ID', 'GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT'):
        if key in env:
            monkeypatch.setenv(key, env[key])
        else:
            monkeypatch.delenv(key, raising=False)
    with (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}, clear=False),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter') as traces,
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter') as metrics,
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    ):
        enable_google_cloud_telemetry(**kwargs)
    assert _exporter_project(traces) == expected
    assert _exporter_project(metrics) == expected
    assert _exporter_project(_stub_cloud_logging_exporter) == expected


@pytest.mark.parametrize(
    ('env', 'warns'),
    [
        pytest.param({'FIREBASE_PROJECT_ID': 'firebase-proj', 'GOOGLE_CLOUD_PROJECT': 'gcp-proj'}, True, id='differs'),
        pytest.param({'FIREBASE_PROJECT_ID': 'firebase-proj'}, True, id='only_firebase'),
        pytest.param({'FIREBASE_PROJECT_ID': 'same-proj', 'GOOGLE_CLOUD_PROJECT': 'same-proj'}, False, id='matches'),
        pytest.param({'GOOGLE_CLOUD_PROJECT': 'gcp-proj'}, False, id='unset'),
    ],
)
def test_config_warns_when_firebase_project_id_is_ignored(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], warns: bool
) -> None:
    """A FIREBASE_PROJECT_ID that would have picked a different project logs one warning naming it."""
    for key in ('FIREBASE_PROJECT_ID', 'GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT'):
        if key in env:
            monkeypatch.setenv(key, env[key])
        else:
            monkeypatch.delenv(key, raising=False)
    with patch('genkit_google_cloud.telemetry.config.logger') as logger:
        GcpTelemetry()

    messages = [c.args[0] for c in logger.warning.call_args_list]
    assert any('FIREBASE_PROJECT_ID' in m for m in messages) is warns


def _prod_exporter_patches() -> tuple[Any, ...]:
    return (
        mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_PROD}, clear=False),
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch('genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter'),
        patch('genkit_google_cloud.telemetry.config._hang_exporter_on_process_tracer'),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
    )


def test_firebase_only_falls_back_to_adc_project_and_keeps_log_trace_correlation(
    monkeypatch: pytest.MonkeyPatch,
    _adc_project: MagicMock,
    _stub_cloud_logging_exporter: MagicMock,
) -> None:
    """With only FIREBASE_PROJECT_ID set, the ADC project goes to every exporter and to logging.googleapis.com/trace."""
    for key in ('GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('FIREBASE_PROJECT_ID', 'firebase-proj')
    _adc_project.return_value = 'adc-proj'

    manager = GcpTelemetry()
    env, gcp, adjusting, hang, detector, monitoring, metric_exp, reader, meter = _prod_exporter_patches()
    with env, gcp as traces, adjusting, hang, detector, monitoring as cloud_metrics, metric_exp, reader, meter:
        manager.initialize()

    assert _exporter_project(traces) == 'adc-proj'
    assert _exporter_project(cloud_metrics) == 'adc-proj'
    assert _exporter_project(_stub_cloud_logging_exporter) == 'adc-proj'

    provider = TracerProvider()
    try:
        with provider.get_tracer('test').start_as_current_span('order') as span:
            event = manager._inject_trace_context(MagicMock(), 'info', {'event': 'order placed'})
            trace_id = span.get_span_context().trace_id
    finally:
        provider.shutdown()
    assert event['logging.googleapis.com/trace'] == f'projects/adc-proj/traces/{trace_id:032x}'


def test_explicit_project_skips_adc_lookup(_adc_project: MagicMock) -> None:
    """project= wins, so ADC is never asked."""
    env, gcp, adjusting, hang, detector, monitoring, metric_exp, reader, meter = _prod_exporter_patches()
    with env, gcp as traces, adjusting, hang, detector, monitoring, metric_exp, reader, meter:
        enable_google_cloud_telemetry(project='explicit-proj')

    _adc_project.assert_not_called()
    assert _exporter_project(traces) == 'explicit-proj'


def test_dev_without_force_skips_adc_lookup(monkeypatch: pytest.MonkeyPatch, _adc_project: MagicMock) -> None:
    """Under genkit start with no export, enable() does not probe ADC."""
    for key in ('GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT'):
        monkeypatch.delenv(key, raising=False)
    with mock.patch.dict(os.environ, {_GENKIT_ENV: _ENV_DEV}):
        enable_google_cloud_telemetry()

    _adc_project.assert_not_called()


def test_adc_project_id_is_none_without_default_credentials() -> None:
    """No ADC on the machine means no project, not a raise."""
    from genkit_google_cloud.telemetry import config

    with patch.object(config, 'google_auth_default', side_effect=DefaultCredentialsError('no ADC')):
        assert _adc_project_id() is None
    with patch.object(config, 'google_auth_default', return_value=(MagicMock(), 'adc-proj')):
        assert _adc_project_id() == 'adc-proj'


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
        enable_google_cloud_telemetry(project='my-project')
        mock_add_exporter.assert_called_once()
        with pytest.raises(GenkitError, match='already called') as raised:
            enable_google_cloud_telemetry(project='other')
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
        enable_google_cloud_telemetry(project='my-project')
        assert is_instrumented_by(GenAiInstrumentation)


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
            enable_google_cloud_telemetry(project='my-project')
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
            enable_google_cloud_telemetry(force_dev_export=True, project='my-project')
        assert any(isinstance(proc, SimpleSpanProcessor) for proc in _processors(isolated))
    finally:
        isolated.shutdown()


def test_enable_google_cloud_telemetry_takes_project_like_google_cloud_clients() -> None:
    """The setup kwarg is project=, matching google-cloud-* clients; project_id is only the OTel exporter kwarg."""
    params = inspect.signature(enable_google_cloud_telemetry).parameters
    assert 'project' in params
    assert 'project_id' not in params
