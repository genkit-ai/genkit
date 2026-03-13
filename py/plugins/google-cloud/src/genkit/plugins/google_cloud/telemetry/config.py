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

"""Configuration management for GCP telemetry.

This module handles project ID resolution, telemetry configuration,
and initialization of tracing, metrics, and logging.
"""

import logging
import os
import uuid
from collections.abc import Mapping
from typing import Any

import structlog
from opentelemetry import metrics
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.resourcedetector.gcp_resource_detector import GoogleCloudResourceDetector
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_INSTANCE_ID, SERVICE_NAME, Resource
from opentelemetry.sdk.trace.sampling import Sampler
from opentelemetry.trace import get_current_span, span as trace_span

from genkit.core.environment import is_dev_environment
from genkit.core.logging import get_logger
from genkit.core.tracing import add_custom_exporter

from .constants import (
    DEFAULT_METRIC_EXPORT_INTERVAL_MS,
    DEV_METRIC_EXPORT_INTERVAL_MS,
    MIN_METRIC_EXPORT_INTERVAL_MS,
    PROJECT_ID_ENV_VARS,
)
from .exporters import handle_metric_error, handle_tracing_error
from .metrics_exporter import GenkitMetricExporter
from .trace_exporter import GcpAdjustingTraceExporter, GenkitGCPExporter

logger = get_logger(__name__)


def resolve_project_id(
    project_id: str | None = None,
    credentials: dict[str, Any] | None = None,
) -> str | None:
    """Resolve the GCP project ID from multiple sources.

    Resolution order (matching JS/Go):
    1. Explicit project_id parameter
    2. FIREBASE_PROJECT_ID environment variable
    3. GOOGLE_CLOUD_PROJECT environment variable
    4. GCLOUD_PROJECT environment variable
    5. Project ID from credentials

    Args:
        project_id: Explicitly provided project ID.
        credentials: Optional credentials dict with project_id.

    Returns:
        The resolved project ID or None.
    """
    if project_id:
        return project_id

    # Check environment variables in order of priority
    for env_var in PROJECT_ID_ENV_VARS:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value

    # Check credentials for project_id
    if credentials and 'project_id' in credentials:
        return credentials['project_id']

    return None


class GcpTelemetry:
    """Central manager for GCP Telemetry configuration.

    Encapsulates configuration and manages the lifecycle of Tracing, Metrics,
    and Logging setup, ensuring consistent state (like project_id) across all
    telemetry components.
    """

    def __init__(
        self,
        project_id: str | None = None,
        credentials: dict[str, Any] | None = None,
        sampler: Sampler | None = None,
        log_input_and_output: bool = False,
        force_dev_export: bool = False,
        disable_metrics: bool = False,
        disable_traces: bool = False,
        metric_export_interval_ms: int | None = None,
        metric_export_timeout_ms: int | None = None,
    ) -> None:
        """Initialize the GCP Telemetry manager.

        Args:
            project_id: GCP project ID.
            credentials: Optional credentials dict.
            sampler: Trace sampler.
            log_input_and_output: If False, hides sensitive data.
            force_dev_export: Check to force export in dev environment.
            disable_metrics: If True, metrics are not exported.
            disable_traces: If True, traces are not exported.
            metric_export_interval_ms: Export interval in ms.
            metric_export_timeout_ms: Export timeout in ms.
        """
        self.credentials = credentials
        self.sampler = sampler
        self.log_input_and_output = log_input_and_output
        self.force_dev_export = force_dev_export
        self.disable_metrics = disable_metrics
        self.disable_traces = disable_traces

        # Resolve project ID immediately
        self.project_id = resolve_project_id(project_id, credentials)

        # Determine metric export settings
        is_dev = is_dev_environment()

        default_interval = DEV_METRIC_EXPORT_INTERVAL_MS if is_dev else DEFAULT_METRIC_EXPORT_INTERVAL_MS
        self.metric_export_interval_ms = metric_export_interval_ms or default_interval

        if self.metric_export_interval_ms < MIN_METRIC_EXPORT_INTERVAL_MS:
            logger.warning(
                f'metric_export_interval_ms ({self.metric_export_interval_ms}) is below minimum '
                f'({MIN_METRIC_EXPORT_INTERVAL_MS}), using minimum'
            )
            self.metric_export_interval_ms = MIN_METRIC_EXPORT_INTERVAL_MS

        self.metric_export_timeout_ms = metric_export_timeout_ms or self.metric_export_interval_ms

    def _build_exporter_kwargs(self) -> dict[str, Any]:
        """Build kwargs dict for exporters with project_id and credentials.

        Returns:
            A dict with project_id and/or credentials if available, empty dict otherwise.
        """
        kwargs: dict[str, Any] = {}
        if self.project_id:
            kwargs['project_id'] = self.project_id
        if self.credentials:
            kwargs['credentials'] = self.credentials
        return kwargs

    def initialize(self) -> None:
        """Actuates the telemetry configuration.

        CRITICAL: This method MUST be called to initialize telemetry handlers
        even in dev mode. The 'export' flag controls whether data is sent to
        GCP, but initialization is ALWAYS required for proper operation.
        """
        is_dev = is_dev_environment()
        should_export = self.force_dev_export or not is_dev

        # ALWAYS configure logging (required for telemetry handlers)
        # The export flag is passed down to control Cloud Logging export
        self._configure_logging()

        # Only configure tracing/metrics if exporting (performance optimization)
        if should_export:
            self._configure_tracing()
            self._configure_metrics()
            logger.info(
                'Telemetry fully initialized',
                project_id=self.project_id,
                export_enabled=True,
                environment='dev' if is_dev else 'prod',
                force_dev_export=self.force_dev_export,
            )
        else:
            logger.debug(
                'Telemetry initialized in local-only mode',
                export_enabled=False,
                environment='dev',
                note='Use force_dev_export=True for full AIM visibility in dev',
            )

    def _configure_logging(self) -> None:
        """Configure structlog with Cloud Logging export and trace correlation."""
        from .gcp_logger import gcp_logger

        is_dev = is_dev_environment()
        should_export = self.force_dev_export or not is_dev

        # Initialize the GCP logger for telemetry modules
        gcp_logger.initialize(
            project_id=self.project_id,
            credentials=self.credentials,
            export=should_export,
        )

        # Configure structlog processors for trace correlation
        try:
            current_config = structlog.get_config()
            processors = list(current_config.get('processors', []))

            # Early return if already configured
            if any(getattr(p, '__name__', '') == '_genkit_inject_trace_context' for p in processors):
                return

            # Define processor function that captures self
            def _genkit_inject_trace_context(
                logger_instance: logging.Logger,
                method_name: str,
                event_dict: dict[str, Any],
            ) -> Mapping[str, Any]:
                return self._inject_trace_context(logger_instance, method_name, event_dict)

            # Append processor to chain
            processors.append(_genkit_inject_trace_context)
            structlog.configure(processors=processors)
            logger.debug('Configured structlog for GCP trace correlation')

        except Exception as e:
            logger.warning('Failed to configure structlog for trace correlation', error=str(e))

    def _configure_tracing(self) -> None:
        if self.disable_traces:
            return

        try:
            exporter_kwargs = self._build_exporter_kwargs()
            base_exporter = GenkitGCPExporter(**exporter_kwargs) if exporter_kwargs else GenkitGCPExporter()

            trace_exporter = GcpAdjustingTraceExporter(
                exporter=base_exporter,
                log_input_and_output=self.log_input_and_output,
                project_id=self.project_id,
                error_handler=handle_tracing_error,
            )

            add_custom_exporter(trace_exporter, 'gcp_telemetry_server')
        except Exception as e:
            handle_tracing_error(e)

    def _configure_metrics(self) -> None:
        if self.disable_metrics:
            return

        try:
            resource = Resource.create({
                SERVICE_NAME: 'genkit',
                SERVICE_INSTANCE_ID: str(uuid.uuid4()),
            })

            # Suppress detector warnings during GCP resource detection
            detector_logger = logging.getLogger('opentelemetry.resourcedetector.gcp_resource_detector')
            original_level = detector_logger.level
            detector_logger.setLevel(logging.ERROR)

            try:
                gcp_resource = GoogleCloudResourceDetector(raise_on_error=True).detect()
                resource = resource.merge(gcp_resource)
            except Exception as e:
                # For detection failure log the exception and use the default resource
                detector_logger.warning(f'Google Cloud resource detection failed: {e}')
            finally:
                detector_logger.setLevel(original_level)

            exporter_kwargs = self._build_exporter_kwargs()
            cloud_monitoring_exporter = CloudMonitoringMetricsExporter(**exporter_kwargs)

            metrics_exporter = GenkitMetricExporter(
                exporter=cloud_monitoring_exporter,
                error_handler=handle_metric_error,
            )

            reader = PeriodicExportingMetricReader(
                metrics_exporter,
                export_interval_millis=self.metric_export_interval_ms,
                export_timeout_millis=self.metric_export_timeout_ms,
            )

            provider = MeterProvider(metric_readers=[reader], resource=resource)
            metrics.set_meter_provider(provider)

        except Exception as e:
            handle_metric_error(e)

    def _inject_trace_context(
        self, logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Structlog processor to inject GCP-compatible trace context."""
        # Only inject if event_dict is a dict or mapping
        if not isinstance(event_dict, dict) and not hasattr(event_dict, '__setitem__'):
            return event_dict

        span = get_current_span()
        if span == trace_span.INVALID_SPAN:
            return event_dict

        ctx = span.get_span_context()
        if not ctx.is_valid:
            return event_dict

        if self.project_id:
            event_dict['logging.googleapis.com/trace'] = f'projects/{self.project_id}/traces/{ctx.trace_id:032x}'

        event_dict['logging.googleapis.com/spanId'] = f'{ctx.span_id:016x}'
        event_dict['logging.googleapis.com/trace_sampled'] = '1' if ctx.trace_flags.sampled else '0'

        return event_dict
