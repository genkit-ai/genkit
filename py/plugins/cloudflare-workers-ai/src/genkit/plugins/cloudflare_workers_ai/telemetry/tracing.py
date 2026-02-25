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

"""Telemetry and tracing functionality for the Genkit Cloudflare plugin.

This module provides functionality for collecting and exporting telemetry data
from Genkit operations via OpenTelemetry Protocol (OTLP) to any compatible
backend including Cloudflare's native OTLP endpoints.

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
    │ OTLP                │ OpenTelemetry Protocol - a standard way to send    │
    │                     │ traces. Like a universal language for telemetry.   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Endpoint            │ Where your traces are sent. Any OTLP receiver.     │
    │                     │ Grafana, Honeycomb, Axiom, etc.                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ API Token           │ Your key to authenticate with the endpoint.        │
    │                     │ Sent as a Bearer token in the Authorization header.│
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Exporter            │ Ships your traces to the endpoint. Like a postal   │
    │                     │ service for your telemetry data.                   │
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
    │                         │  (4) Sent via OTLP/HTTP                       │
    │                         ▼                                               │
    │              ┌─────────────────────┐                                    │
    │              │ OTLP Span Exporter  │                                    │
    │              │ (+ Bearer auth)     │                                    │
    │              └──────────┬──────────┘                                    │
    │                         │                                               │
    │                         │  (5) HTTPS to your endpoint                   │
    │                         ▼                                               │
    │    ════════════════════════════════════════════════════                 │
    │                         │                                               │
    │                         ▼                                               │
    │              ┌─────────────────────┐                                    │
    │              │  Your OTLP Backend  │   View traces in your dashboard    │
    │              │  (any compatible)   │   Debug latency, errors, etc.      │
    │              └─────────────────────┘                                    │
    └─────────────────────────────────────────────────────────────────────────┘

Configuration Options::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Option                      │ Type     │ Default    │ Description       │
    ├─────────────────────────────┼──────────┼────────────┼───────────────────┤
    │ endpoint                    │ str      │ env var    │ OTLP endpoint URL │
    │ api_token                   │ str      │ env var    │ Bearer token      │
    │ log_input_and_output        │ bool     │ False      │ Disable redaction │
    │ force_dev_export            │ bool     │ True       │ Export in dev     │
    │ disable_traces              │ bool     │ False      │ Skip traces       │
    │ sampler                     │ Sampler  │ AlwaysOn   │ Trace sampler     │
    └─────────────────────────────┴──────────┴────────────┴───────────────────┘

Endpoint Resolution Order:
    1. Explicit endpoint parameter
    2. CF_OTLP_ENDPOINT environment variable

API Token Resolution Order:
    1. Explicit api_token parameter
    2. CF_API_TOKEN environment variable (same as used by cloudflare-workers-ai plugin)

Usage:
    ```python
    from genkit.plugins.cloudflare_workers_ai import add_cloudflare_telemetry

    # Enable telemetry with default settings (PII redaction enabled)
    add_cloudflare_telemetry()

    # Enable telemetry with explicit endpoint
    add_cloudflare_telemetry(endpoint='https://otel.example.com/v1/traces')

    # Enable input/output logging (disable PII redaction)
    add_cloudflare_telemetry(log_input_and_output=True)

    # Force export in dev environment
    add_cloudflare_telemetry(force_dev_export=True)
    ```

Cloudflare Documentation References:
    Workers Observability:
        - Overview: https://developers.cloudflare.com/workers/observability/
        - OTLP Export: https://developers.cloudflare.com/workers/observability/exporting-opentelemetry-data/
        - Tracing: https://developers.cloudflare.com/workers/observability/traces/

    Compatible Backends:
        - Grafana Cloud: https://grafana.com/docs/grafana-cloud/send-data/otlp/
        - Honeycomb: https://docs.honeycomb.io/send-data/opentelemetry/
        - Axiom: https://axiom.co/docs/send-data/opentelemetry
        - SigNoz: https://signoz.io/docs/instrumentation/opentelemetry/
"""

import os
import uuid
from collections.abc import Mapping, MutableMapping
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
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import Sampler

from genkit.core.environment import is_dev_environment
from genkit.core.trace.adjusting_exporter import AdjustingTraceExporter
from genkit.core.tracing import add_custom_exporter

logger = structlog.get_logger(__name__)

# Environment variables for Cloudflare OTLP configuration
CF_OTLP_ENDPOINT_ENV = 'CF_OTLP_ENDPOINT'
CF_API_TOKEN_ENV = 'CF_API_TOKEN'  # noqa: S105 - env var name, not a secret


def _resolve_endpoint(endpoint: str | None = None) -> str | None:
    """Resolve the OTLP endpoint from various sources.

    Resolution order:
    1. Explicit endpoint parameter
    2. CF_OTLP_ENDPOINT environment variable

    Args:
        endpoint: Explicitly provided OTLP endpoint.

    Returns:
        The resolved endpoint or None if not found.
    """
    if endpoint:
        return endpoint

    return os.environ.get(CF_OTLP_ENDPOINT_ENV)


def _resolve_api_token(api_token: str | None = None) -> str | None:
    """Resolve the API token from various sources.

    Resolution order:
    1. Explicit api_token parameter
    2. CF_API_TOKEN environment variable

    Args:
        api_token: Explicitly provided API token.

    Returns:
        The resolved API token or None if not found.
    """
    if api_token:
        return api_token

    return os.environ.get(CF_API_TOKEN_ENV)


class CfTelemetry:
    """Central manager for Cloudflare Telemetry configuration.

    Encapsulates configuration and manages the lifecycle of Tracing setup,
    ensuring consistent state across all telemetry components.

    Example:
        ```python
        telemetry = CfTelemetry(endpoint='https://otel.example.com/v1/traces')
        telemetry.initialize()
        ```
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_token: str | None = None,
        sampler: Sampler | None = None,
        log_input_and_output: bool = False,
        force_dev_export: bool = True,
        disable_traces: bool = False,
        service_name: str = 'genkit',
        service_version: str | None = None,
        service_namespace: str | None = None,
        deployment_environment: str | None = None,
    ) -> None:
        """Initialize the Cloudflare Telemetry manager.

        Args:
            endpoint: OTLP traces endpoint URL.
            api_token: API token for Bearer authentication.
            sampler: Trace sampler.
            log_input_and_output: If False, redacts sensitive data.
            force_dev_export: If True, exports even in dev environment.
            disable_traces: If True, traces are not exported.
            service_name: Name of your service (appears in traces).
            service_version: Version of your service.
            service_namespace: Namespace for your service.
            deployment_environment: Deployment environment (e.g., "production").

        Raises:
            ValueError: If endpoint cannot be resolved.
        """
        self.sampler = sampler
        self.log_input_and_output = log_input_and_output
        self.force_dev_export = force_dev_export
        self.disable_traces = disable_traces
        self.service_name = service_name
        self.service_version = service_version
        self.service_namespace = service_namespace
        self.deployment_environment = deployment_environment

        # Resolve configuration
        self.endpoint = _resolve_endpoint(endpoint)
        self.api_token = _resolve_api_token(api_token)

        if self.endpoint is None:
            raise ValueError(
                'OTLP endpoint is required. '
                'Set CF_OTLP_ENDPOINT environment variable '
                'or pass endpoint parameter to add_cloudflare_telemetry().'
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
        """Configure structlog with trace correlation.

        Injects trace context into log records for correlation.
        """
        try:
            current_config = structlog.get_config()
            processors = current_config.get('processors', [])

            # Check if our processor is already registered
            if not any(getattr(p, '__name__', '') == 'inject_cf_trace_context' for p in processors):

                def inject_cf_trace_context(
                    _logger: Any,  # noqa: ANN401
                    method_name: str,
                    event_dict: MutableMapping[str, Any],
                ) -> Mapping[str, Any]:
                    """Inject trace context into log event."""
                    return self._inject_trace_context(event_dict)

                new_processors = list(processors)
                # Insert before the last processor (usually the renderer)
                new_processors.insert(max(0, len(new_processors) - 1), inject_cf_trace_context)
                cfg = structlog.get_config()
                structlog.configure(
                    processors=new_processors,
                    wrapper_class=cfg.get('wrapper_class'),
                    context_class=cfg.get('context_class'),
                    logger_factory=cfg.get('logger_factory'),
                    cache_logger_on_first_use=cfg.get('cache_logger_on_first_use'),
                )
                logger.debug('Configured structlog for trace correlation')

        except Exception as e:
            logger.warning('Failed to configure structlog for trace correlation', error=str(e))

    def _configure_tracing(self) -> None:
        """Configure trace export via OTLP."""
        if self.disable_traces:
            return

        # Endpoint is guaranteed to be set by __init__
        assert self.endpoint is not None

        # Create resource with service info
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

        # Build headers for OTLP exporter
        headers: dict[str, str] = {}
        if self.api_token:
            headers['Authorization'] = f'Bearer {self.api_token}'

        # Create OTLP exporter
        base_exporter = OTLPSpanExporter(
            endpoint=self.endpoint,
            headers=headers,
        )

        # Wrap with AdjustingTraceExporter for PII redaction
        trace_exporter = AdjustingTraceExporter(
            exporter=base_exporter,
            log_input_and_output=self.log_input_and_output,
            error_handler=lambda e: _handle_tracing_error(e),
        )

        add_custom_exporter(trace_exporter, 'cf_telemetry')

        logger.info(
            'Cloudflare OTLP telemetry configured',
            endpoint=self.endpoint,
        )

    def _inject_trace_context(self, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Inject trace context into log event.

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
        event_dict['trace_id'] = f'{ctx.trace_id:032x}'
        event_dict['span_id'] = f'{ctx.span_id:016x}'

        return event_dict


def add_cloudflare_telemetry(
    endpoint: str | None = None,
    api_token: str | None = None,
    sampler: Sampler | None = None,
    log_input_and_output: bool = False,
    force_dev_export: bool = True,
    disable_traces: bool = False,
    service_name: str = 'genkit',
    service_version: str | None = None,
    service_namespace: str | None = None,
    deployment_environment: str | None = None,
) -> None:
    """Configure Cloudflare-compatible OTLP telemetry export.

    This function sets up OpenTelemetry export via OTLP to any compatible
    endpoint. By default, model inputs and outputs are redacted for privacy.

    Args:
        endpoint: OTLP traces endpoint URL. If not provided, uses
            CF_OTLP_ENDPOINT environment variable.
        api_token: API token for Bearer authentication. If not provided,
            uses CF_API_TOKEN environment variable.
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
        service_name: Name of your service (appears in traces). Defaults to "genkit".
        service_version: Version of your service. Useful for deployment tracking.
        service_namespace: Namespace for your service.
        deployment_environment: Deployment environment name (e.g., "production",
            "staging", "development"). Useful for filtering traces by environment.

    Raises:
        ValueError: If endpoint cannot be resolved from parameters
            or environment.

    Example:
        ```python
        # Default: PII redaction enabled, uses env vars
        add_cloudflare_telemetry()

        # With explicit endpoint
        add_cloudflare_telemetry(endpoint='https://otel.example.com/v1/traces')

        # Full configuration with service metadata
        add_cloudflare_telemetry(
            endpoint='https://otel.example.com/v1/traces',
            api_token='your-token',
            service_name='my-genkit-app',
            service_version='1.2.3',
            deployment_environment='production',
        )
        ```

    See Also:
        - Cloudflare Workers OTLP: https://developers.cloudflare.com/workers/observability/exporting-opentelemetry-data/
        - OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/
    """
    manager = CfTelemetry(
        endpoint=endpoint,
        api_token=api_token,
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
    if 'connection' in error_str or 'network' in error_str:
        _tracing_error_logged = True
        logger.error(
            f'Failed to connect to OTLP endpoint. Verify CF_OTLP_ENDPOINT is correct and reachable. Error: {error}'
        )
    elif '401' in error_str or 'unauthorized' in error_str:
        _tracing_error_logged = True
        logger.error(f'Authentication failed. Verify CF_API_TOKEN is correct. Error: {error}')
    elif '403' in error_str or 'forbidden' in error_str:
        _tracing_error_logged = True
        logger.error(f'Access denied to OTLP endpoint. Verify your API token has write access. Error: {error}')
    else:
        logger.error('Error exporting traces via OTLP', error=str(error))
