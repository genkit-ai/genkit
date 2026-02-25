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

"""Telemetry and tracing functionality for the Microsoft Foundry plugin.

This module provides functionality for collecting and exporting telemetry data
from Genkit operations to Azure. It uses OpenTelemetry for tracing and exports
span data to Azure Application Insights for monitoring and debugging purposes.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Span                │ A "timer" that records how long something took.    │
    │                     │ Like a stopwatch for one task in your code.        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Trace               │ A collection of spans showing a request's journey. │
    │                     │ Like breadcrumbs through your code.                │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Application Insights│ Azure service that collects and visualizes traces. │
    │                     │ Like a detective board connecting all the clues.   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Connection String   │ Your Application Insights "address". Contains the  │
    │                     │ instrumentation key and ingestion endpoint.        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ OTLP                │ OpenTelemetry Protocol - a standard way to send    │
    │                     │ traces. Like a universal shipping label format.    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Exporter            │ Ships your traces to Azure. Like a postal service  │
    │                     │ for your telemetry data.                           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ PII Redaction       │ Removes sensitive data from traces. Like blacking  │
    │                     │ out private info before sharing.                   │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         HOW YOUR CODE GETS TRACED                       │
    │                                                                         │
    │    Your Genkit App                                                      │
    │         │                                                               │
    │         │  (1) You call a flow, model, or tool                          │
    │         ▼                                                               │
    │    ┌─────────┐     ┌─────────┐     ┌─────────┐                          │
    │    │ Flow    │ ──▶ │ Model   │ ──▶ │ Tool    │   Each creates a "span"  │
    │    │ (span)  │     │ (span)  │     │ (span)  │   (a timing record)      │
    │    └─────────┘     └─────────┘     └─────────┘                          │
    │         │               │               │                               │
    │         └───────────────┼───────────────┘                               │
    │                         │                                               │
    │                         │  (2) Spans collected into a trace             │
    │                         ▼                                               │
    │                   ┌───────────┐                                         │
    │                   │   Trace   │   All spans for one request             │
    │                   └─────┬─────┘                                         │
    │                         │                                               │
    │                         │  (3) Adjustments applied                      │
    │                         ▼                                               │
    │           ┌─────────────────────────────┐                               │
    │           │   AdjustingTraceExporter    │                               │
    │           │  • Redact PII (input/output)│                               │
    │           │  • Add error markers        │                               │
    │           └─────────────┬───────────────┘                               │
    │                         │                                               │
    │                         │  (4) Sent to Azure via OTLP                   │
    │                         ▼                                               │
    │              ┌─────────────────────┐                                    │
    │              │ Azure Monitor OTLP  │                                    │
    │              │ Span Exporter       │                                    │
    │              └──────────┬──────────┘                                    │
    │                         │                                               │
    │                         │  (5) HTTPS to Application Insights            │
    │                         ▼                                               │
    │    ════════════════════════════════════════════════════                 │
    │                         │                                               │
    │                         ▼                                               │
    │              ┌─────────────────────┐                                    │
    │              │   Azure Portal      │   View traces in App Insights      │
    │              │   Application       │   Debug latency, errors, etc.      │
    │              │   Insights          │                                    │
    │              └─────────────────────┘                                    │
    └─────────────────────────────────────────────────────────────────────────┘

Configuration Options::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Option                      │ Type     │ Default    │ Description       │
    ├─────────────────────────────┼──────────┼────────────┼───────────────────┤
    │ connection_string           │ str      │ env var    │ App Insights conn │
    │ log_input_and_output        │ bool     │ False      │ Disable redaction │
    │ force_dev_export            │ bool     │ True       │ Export in dev     │
    │ disable_traces              │ bool     │ False      │ Skip traces       │
    │ sampler                     │ Sampler  │ AlwaysOn   │ Trace sampler     │
    └─────────────────────────────┴──────────┴────────────┴───────────────────┘

Connection String Resolution Order:
    1. Explicit connection_string parameter
    2. APPLICATIONINSIGHTS_CONNECTION_STRING environment variable

Usage:
    ```python
    from genkit.plugins.microsoft_foundry import add_azure_telemetry

    # Enable telemetry with default settings (PII redaction enabled)
    add_azure_telemetry()

    # Enable telemetry with explicit connection string
    add_azure_telemetry(connection_string='InstrumentationKey=...')

    # Enable input/output logging (disable PII redaction)
    add_azure_telemetry(log_input_and_output=True)

    # Force export in dev environment
    add_azure_telemetry(force_dev_export=True)
    ```

Azure Documentation References:
    Application Insights:
        - Overview: https://docs.microsoft.com/azure/azure-monitor/app/app-insights-overview
        - Connection String: https://docs.microsoft.com/azure/azure-monitor/app/sdk-connection-string

    Azure Monitor OpenTelemetry:
        - Python SDK: https://pypi.org/project/azure-monitor-opentelemetry-exporter/
        - Configure: https://docs.microsoft.com/azure/azure-monitor/app/opentelemetry-enable

Implementation Notes & Edge Cases
---------------------------------

**Azure Monitor OpenTelemetry Exporter**

We use the ``azure-monitor-opentelemetry-exporter`` package which provides
native Azure Monitor integration. This is the officially supported way to
send OpenTelemetry traces to Application Insights from Python applications.

The exporter requires a connection string which contains:
- InstrumentationKey: Identifies your Application Insights resource
- IngestionEndpoint: The Azure endpoint for data ingestion

**Fallback to OTLP HTTP Exporter**

If the Azure Monitor exporter is not installed (optional dependency), we
fall back to a generic OTLP HTTP exporter. This allows users to send traces
to any OTLP-compatible endpoint, which can then forward to Azure Monitor
via an OpenTelemetry Collector.

**PII Redaction**

By default, model inputs and outputs are redacted to ``<redacted>`` to prevent
accidentally logging sensitive user data. Set ``log_input_and_output=True``
only in trusted environments where PII exposure is acceptable.
"""

import os
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_NAMESPACE,
    SERVICE_VERSION,
    Resource,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import Sampler

from genkit.core.environment import is_dev_environment
from genkit.core.trace.adjusting_exporter import AdjustingTraceExporter
from genkit.core.tracing import add_custom_exporter

logger = structlog.get_logger(__name__)

# Environment variable for Application Insights connection string
APPLICATIONINSIGHTS_CONNECTION_STRING_ENV = 'APPLICATIONINSIGHTS_CONNECTION_STRING'


def _resolve_connection_string(connection_string: str | None = None) -> str | None:
    """Resolve the Application Insights connection string from various sources.

    Resolution order:
    1. Explicit connection_string parameter
    2. APPLICATIONINSIGHTS_CONNECTION_STRING environment variable

    Args:
        connection_string: Explicitly provided connection string.

    Returns:
        The resolved connection string or None if not found.
    """
    if connection_string:
        return connection_string

    return os.environ.get(APPLICATIONINSIGHTS_CONNECTION_STRING_ENV)


def _create_azure_monitor_exporter(
    connection_string: str,
    error_handler: Callable[[Exception], None] | None = None,
) -> SpanExporter:
    """Create an Azure Monitor span exporter.

    Attempts to use the official Azure Monitor exporter if available,
    otherwise falls back to a generic OTLP exporter with instructions.

    Args:
        connection_string: Application Insights connection string.
        error_handler: Optional callback for export errors.

    Returns:
        A SpanExporter configured for Azure Monitor.
    """
    try:
        # Try to import the official Azure Monitor exporter (optional dependency)
        # pyrefly: ignore[missing-import] - Optional dependency, handled by try/except
        from azure.monitor.opentelemetry.exporter import (
            AzureMonitorTraceExporter,
        )

        logger.debug('Using official Azure Monitor OpenTelemetry exporter')
        return AzureMonitorTraceExporter(connection_string=connection_string)

    except ImportError:
        # Fall back to a wrapper that logs helpful instructions
        logger.warning(
            'azure-monitor-opentelemetry-exporter not installed. '
            'Install with: pip install genkit-plugin-microsoft-foundry[monitor] '
            'Falling back to generic OTLP exporter.'
        )
        return AzureOtlpFallbackExporter(
            connection_string=connection_string,
            error_handler=error_handler,
        )


class AzureOtlpFallbackExporter(SpanExporter):
    """Fallback OTLP exporter when Azure Monitor exporter is not available.

    This exporter attempts to send traces via OTLP to Azure's ingestion
    endpoint extracted from the connection string, but the recommended
    approach is to install the official Azure Monitor exporter.

    Args:
        connection_string: Application Insights connection string.
        error_handler: Optional callback for export errors.

    Note:
        For best results, install azure-monitor-opentelemetry-exporter:
        ``pip install genkit-plugin-microsoft-foundry[monitor]``
    """

    def __init__(
        self,
        connection_string: str,
        error_handler: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize the fallback exporter.

        Args:
            connection_string: Application Insights connection string.
            error_handler: Optional callback invoked when export errors occur.
        """
        self._connection_string = connection_string
        self._error_handler = error_handler

        # Parse connection string to extract endpoint
        self._endpoint = self._parse_endpoint(connection_string)

        # Create OTLP exporter with the extracted endpoint
        if self._endpoint:
            self._otlp_exporter: SpanExporter | None = OTLPSpanExporter(
                endpoint=f'{self._endpoint}/v2/track',
            )
        else:
            self._otlp_exporter = None
            logger.error(
                'Could not parse IngestionEndpoint from connection string. '
                'Traces will not be exported. Install azure-monitor-opentelemetry-exporter.'
            )

    def _parse_endpoint(self, connection_string: str) -> str | None:
        """Parse IngestionEndpoint from connection string.

        Args:
            connection_string: Application Insights connection string.

        Returns:
            The ingestion endpoint URL or None if not found.
        """
        # Connection string format: Key1=Value1;Key2=Value2;...
        parts = connection_string.split(';')
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                if key.strip().lower() == 'ingestionendpoint':
                    return value.strip()

        return None

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to Azure via OTLP.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects to export.

        Returns:
            SpanExportResult indicating success or failure.
        """
        if self._otlp_exporter is None:
            return SpanExportResult.FAILURE

        try:
            return self._otlp_exporter.export(spans)
        except Exception as ex:
            logger.error('Error while writing to Azure Application Insights', exc_info=ex)
            if self._error_handler:
                self._error_handler(ex)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        if self._otlp_exporter:
            self._otlp_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending spans.

        Args:
            timeout_millis: Timeout in milliseconds.

        Returns:
            True if flush succeeded.
        """
        if self._otlp_exporter:
            return self._otlp_exporter.force_flush(timeout_millis)
        return True


class AzureTelemetry:
    """Central manager for Azure Telemetry configuration.

    Encapsulates configuration and manages the lifecycle of Tracing and Logging
    setup, ensuring consistent state across all telemetry components.

    Example:
        ```python
        telemetry = AzureTelemetry(connection_string='InstrumentationKey=...')
        telemetry.initialize()
        ```
    """

    def __init__(
        self,
        connection_string: str | None = None,
        sampler: Sampler | None = None,
        log_input_and_output: bool = False,
        force_dev_export: bool = True,
        disable_traces: bool = False,
        service_name: str = 'genkit',
        service_version: str | None = None,
        service_namespace: str | None = None,
        deployment_environment: str | None = None,
    ) -> None:
        """Initialize the Azure Telemetry manager.

        Args:
            connection_string: Application Insights connection string.
            sampler: Trace sampler.
            log_input_and_output: If False, redacts sensitive data.
            force_dev_export: If True, exports even in dev environment.
            disable_traces: If True, traces are not exported.
            service_name: Name of your service (appears in traces as Cloud Role Name).
            service_version: Version of your service.
            service_namespace: Namespace for your service (combined with service_name
                for Cloud Role Name in format "namespace.name").
            deployment_environment: Deployment environment (e.g., "production", "staging").

        Raises:
            ValueError: If connection string cannot be resolved.
        """
        self.sampler = sampler
        self.log_input_and_output = log_input_and_output
        self.force_dev_export = force_dev_export
        self.disable_traces = disable_traces
        self.service_name = service_name
        self.service_version = service_version
        self.service_namespace = service_namespace
        self.deployment_environment = deployment_environment

        # Resolve connection string immediately
        self.connection_string = _resolve_connection_string(connection_string)

        if self.connection_string is None:
            raise ValueError(
                'Azure Application Insights connection string is required. '
                'Set APPLICATIONINSIGHTS_CONNECTION_STRING environment variable '
                'or pass connection_string parameter to add_azure_telemetry().'
            )

    def initialize(self) -> None:
        """Actuate the telemetry configuration.

        Sets up logging with trace correlation and configures tracing export.
        """
        is_dev = is_dev_environment()
        should_export = self.force_dev_export or not is_dev

        if not should_export:
            logger.debug('Telemetry export disabled in dev environment')
            return

        self._configure_logging()
        self._configure_tracing()

    def _configure_logging(self) -> None:
        """Configure structlog with Azure trace correlation.

        Injects trace context into log records for correlation in Application Insights.
        """
        try:
            current_config = structlog.get_config()
            processors = current_config.get('processors', [])

            # Check if our processor is already registered
            if not any(getattr(p, '__name__', '') == 'inject_azure_trace_context' for p in processors):

                def inject_azure_trace_context(
                    _logger: Any,  # noqa: ANN401
                    method_name: str,
                    event_dict: MutableMapping[str, Any],
                ) -> Mapping[str, Any]:
                    """Inject Azure trace context into log event."""
                    return self._inject_trace_context(event_dict)

                new_processors = list(processors)
                # Insert before the last processor (usually the renderer)
                new_processors.insert(max(0, len(new_processors) - 1), inject_azure_trace_context)
                cfg = structlog.get_config()
                structlog.configure(
                    processors=new_processors,
                    wrapper_class=cfg.get('wrapper_class'),
                    context_class=cfg.get('context_class'),
                    logger_factory=cfg.get('logger_factory'),
                    cache_logger_on_first_use=cfg.get('cache_logger_on_first_use'),
                )
                logger.debug('Configured structlog for Azure trace correlation')

        except Exception as e:
            logger.warning('Failed to configure structlog for trace correlation', error=str(e))

    def _configure_tracing(self) -> None:
        """Configure trace export to Azure Application Insights."""
        if self.disable_traces:
            return

        # Connection string is guaranteed to be set by __init__
        assert self.connection_string is not None

        # Create resource with service info
        # Azure Monitor uses these for Cloud Role Name and Cloud Role Instance
        # Cloud Role Name = service.namespace + "." + service.name (or just service.name if no namespace)
        # Cloud Role Instance = service.instance.id
        resource_attrs: dict[str, str] = {
            SERVICE_NAME: self.service_name,
            SERVICE_INSTANCE_ID: str(uuid.uuid4()),
        }

        if self.service_version:
            resource_attrs[SERVICE_VERSION] = self.service_version
        if self.service_namespace:
            resource_attrs[SERVICE_NAMESPACE] = self.service_namespace
        if self.deployment_environment:
            resource_attrs[DEPLOYMENT_ENVIRONMENT] = self.deployment_environment

        resource = Resource.create(resource_attrs)

        # Create TracerProvider
        provider = TracerProvider(
            resource=resource,
            sampler=self.sampler,
        )
        trace.set_tracer_provider(provider)

        # Create the Azure Monitor exporter
        base_exporter = _create_azure_monitor_exporter(
            connection_string=self.connection_string,
            error_handler=lambda e: _handle_tracing_error(e),
        )

        # Wrap with AdjustingTraceExporter for PII redaction
        trace_exporter = AdjustingTraceExporter(
            exporter=base_exporter,
            log_input_and_output=self.log_input_and_output,
            error_handler=lambda e: _handle_tracing_error(e),
        )

        add_custom_exporter(trace_exporter, 'azure_telemetry')

        logger.info(
            'Azure Application Insights telemetry configured',
            connection_string_prefix=self.connection_string[:50] + '...',
        )

    def _inject_trace_context(self, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Inject trace context into log event for Azure correlation.

        Args:
            event_dict: The structlog event dictionary.

        Returns:
            The event dictionary with trace context added.
        """
        span = trace.get_current_span()
        if span == trace.INVALID_SPAN:
            return event_dict

        ctx = span.get_span_context()
        if not ctx.is_valid:
            return event_dict

        # Add standard OpenTelemetry trace context
        # Application Insights uses these for log-trace correlation
        event_dict['trace_id'] = f'{ctx.trace_id:032x}'
        event_dict['span_id'] = f'{ctx.span_id:016x}'

        return event_dict


def add_azure_telemetry(
    connection_string: str | None = None,
    sampler: Sampler | None = None,
    log_input_and_output: bool = False,
    force_dev_export: bool = True,
    disable_traces: bool = False,
    service_name: str = 'genkit',
    service_version: str | None = None,
    service_namespace: str | None = None,
    deployment_environment: str | None = None,
) -> None:
    """Configure Azure telemetry export for traces to Application Insights.

    This function sets up OpenTelemetry export to Azure Application Insights.
    By default, model inputs and outputs are redacted for privacy protection.

    Args:
        connection_string: Application Insights connection string. If not
            provided, uses APPLICATIONINSIGHTS_CONNECTION_STRING environment
            variable.
        sampler: OpenTelemetry trace sampler. Controls which traces are
            collected and exported. Defaults to AlwaysOnSampler. Common options:
            - AlwaysOnSampler: Collect all traces
            - AlwaysOffSampler: Collect no traces
            - TraceIdRatioBasedSampler: Sample a percentage of traces
        log_input_and_output: If True, preserve model input/output in traces.
            Defaults to False (redact for privacy). Only enable this in
            trusted environments where PII exposure is acceptable.
        force_dev_export: If True, export telemetry even in dev environment.
            Defaults to True. Set to False for production-only telemetry.
        disable_traces: If True, traces will not be exported.
            Defaults to False.
        service_name: Name of your service. This appears as the Cloud Role Name
            in Application Insights (combined with service_namespace if provided).
            Defaults to "genkit".
        service_version: Version of your service. Useful for deployment tracking.
        service_namespace: Namespace for your service. When set, Cloud Role Name
            becomes "namespace.name" format.
        deployment_environment: Deployment environment name (e.g., "production",
            "staging", "development"). Useful for filtering traces by environment.

    Raises:
        ValueError: If connection string cannot be resolved from parameters
            or environment.

    Example:
        ```python
        # Default: PII redaction enabled, uses env var
        add_azure_telemetry()

        # Enable input/output logging (disable PII redaction)
        add_azure_telemetry(log_input_and_output=True)

        # Full configuration with service metadata
        add_azure_telemetry(
            connection_string='InstrumentationKey=...',
            service_name='my-genkit-app',
            service_version='1.2.3',
            service_namespace='my-team',
            deployment_environment='production',
        )
        ```

    See Also:
        - Application Insights: https://docs.microsoft.com/azure/azure-monitor/app/app-insights-overview
        - Azure Monitor OpenTelemetry: https://docs.microsoft.com/azure/azure-monitor/app/opentelemetry-enable
        - Cloud Role Name: https://docs.microsoft.com/azure/azure-monitor/app/app-map#understand-the-cloud-role-name-within-the-context-of-an-application-map
    """
    manager = AzureTelemetry(
        connection_string=connection_string,
        sampler=sampler,
        log_input_and_output=log_input_and_output,
        force_dev_export=force_dev_export,
        disable_traces=disable_traces,
        service_name=service_name,
        service_version=service_version,
        service_namespace=service_namespace,
        deployment_environment=deployment_environment,
    )

    manager.initialize()


# Error handling helpers
_tracing_error_logged = False


def _handle_tracing_error(error: Exception) -> None:
    """Handle trace export errors with helpful messages.

    Only logs detailed instructions once to avoid spam.

    Args:
        error: The export error.
    """
    global _tracing_error_logged
    if _tracing_error_logged:
        return

    error_str = str(error).lower()
    if 'connection' in error_str or 'unauthorized' in error_str or '401' in error_str:
        _tracing_error_logged = True
        logger.error(
            'Invalid Application Insights connection string. '
            'Verify your APPLICATIONINSIGHTS_CONNECTION_STRING is correct. '
            'Get this from Azure Portal > Application Insights > Overview. '
            f'Error: {error}'
        )
    elif '403' in error_str or 'forbidden' in error_str:
        _tracing_error_logged = True
        logger.error(
            f'Access denied to Application Insights. Verify the connection string has write access. Error: {error}'
        )
    else:
        logger.error('Error exporting traces to Azure Application Insights', error=str(error))
