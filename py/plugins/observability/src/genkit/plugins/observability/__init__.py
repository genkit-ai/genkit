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

"""Observability Plugin for Genkit - Third-party telemetry backends.

This plugin provides a unified way to export Genkit telemetry to any
OTLP-compatible backend with simple presets for popular services like
Sentry, Honeycomb, Datadog, Grafana Cloud, and Axiom.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ OTLP                │ OpenTelemetry Protocol. The universal language     │
    │                     │ for sending traces. Sentry, Honeycomb, all speak   │
    │                     │ it. Like USB but for observability data.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Backend Preset      │ Pre-configured settings for a service. Just add    │
    │                     │ your API key and you're done! No URLs to remember. │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Sentry              │ Error tracking + tracing. Great for debugging      │
    │                     │ crashes and performance issues.                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Honeycomb           │ Observability platform built for debugging.        │
    │                     │ Query your traces like a database.                 │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Datadog             │ Full-stack monitoring. Traces, metrics, logs,      │
    │                     │ all in one place.                                  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Grafana Cloud       │ Open-source observability suite. Tempo for traces, │
    │                     │ Loki for logs, Prometheus for metrics.             │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Axiom               │ Log and trace ingestion at scale. Query everything │
    │                     │ with SQL-like syntax.                              │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  HOW OBSERVABILITY EXPORT WORKS                         │
    │                                                                         │
    │    Your Genkit App                                                      │
    │         │                                                               │
    │         │  (1) Flows, models, tools create spans                        │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  TracerProvider │   Collects all spans from your app               │
    │    │  (OpenTelemetry)│                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Batch and export via OTLP                            │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  OTLP Exporter  │   Sends to your chosen backend                   │
    │    │  (HTTP POST)    │                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) View in your dashboard                               │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Sentry /       │   Query, alert, debug your traces                │
    │    │  Honeycomb /    │                                                  │
    │    │  Datadog / etc  │                                                  │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

When to Use This Plugin::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     WHEN TO USE WHAT                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   "I'm on AWS and want X-Ray"           → aws plugin (SigV4, X-Ray)     │
    │   "I'm on GCP and want Cloud Trace"     → google-cloud plugin (ADC)     │
    │   "I'm on Azure and want App Insights"  → azure plugin (Azure Monitor)  │
    │                                                                         │
    │   "I'm on AWS but want Honeycomb"       → THIS PLUGIN                   │
    │   "I'm on GCP but want Sentry"          → THIS PLUGIN                   │
    │   "I'm multi-cloud, want Datadog"       → THIS PLUGIN                   │
    │   "I don't care, just give me traces"   → THIS PLUGIN                   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         OBSERVABILITY PLUGIN                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  __init__.py - Plugin Entry Point                                       │
    │  ├── configure_telemetry() - Main configuration function                │
    │  ├── Backend enum - Supported backend types                             │
    │  └── package_name() - Plugin identification                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  backends/ - Backend-Specific Presets                                   │
    │  ├── base.py         - BackendConfig base class                         │
    │  ├── sentry.py       - Sentry OTLP configuration                        │
    │  ├── honeycomb.py    - Honeycomb API configuration                      │
    │  ├── datadog.py      - Datadog OTLP ingestion                           │
    │  ├── grafana.py      - Grafana Cloud Tempo                              │
    │  ├── axiom.py        - Axiom OTLP ingestion                             │
    │  └── custom.py       - Custom OTLP endpoint                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Supported Backends:                                                     │
    │  ┌────────────┬────────────┬────────────┬────────────┬────────────┐     │
    │  │  Sentry    │ Honeycomb  │  Datadog   │  Grafana   │   Axiom    │     │
    │  │  ✅        │  ✅        │  ✅        │  ✅        │  ✅        │     │
    │  └────────────┴────────────┴────────────┴────────────┴────────────┘     │
    └─────────────────────────────────────────────────────────────────────────┘

Example:
    ```python
    from genkit.plugins.observability import configure_telemetry

    # Sentry
    configure_telemetry(backend='sentry', sentry_dsn='https://...')

    # Honeycomb
    configure_telemetry(backend='honeycomb', honeycomb_api_key='...')

    # Datadog
    configure_telemetry(backend='datadog', datadog_api_key='...')

    # Custom OTLP endpoint
    configure_telemetry(
        backend='custom',
        endpoint='https://my-collector/v1/traces',
        headers={'Authorization': 'Bearer ...'},
    )
    ```

Trademark Notice:
    All trademarks (Sentry, Honeycomb, Datadog, Grafana, Axiom) are property
    of their respective owners. This is a community plugin.

See Also:
    - OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/
    - Sentry OTLP: https://docs.sentry.io/platforms/python/tracing/
    - Honeycomb: https://docs.honeycomb.io/send-data/opentelemetry/
    - Datadog OTLP: https://docs.datadoghq.com/opentelemetry/
    - Grafana Cloud: https://grafana.com/docs/grafana-cloud/send-data/otlp/
    - Axiom: https://axiom.co/docs/send-data/opentelemetry
"""

import base64
import os
import re
import sys
import uuid
from collections.abc import Mapping, MutableMapping
from typing import Any

# StrEnum is available in Python 3.11+, use strenum package for 3.10
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_VERSION,
    Resource,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import Sampler

from genkit.core.environment import is_dev_environment
from genkit.core.trace.adjusting_exporter import AdjustingTraceExporter
from genkit.core.tracing import add_custom_exporter

logger = structlog.get_logger(__name__)


class Backend(StrEnum):
    """Supported observability backends.

    Each backend has its own OTLP endpoint and authentication mechanism.
    Use the corresponding configure_telemetry() parameters for each backend.
    """

    SENTRY = 'sentry'
    HONEYCOMB = 'honeycomb'
    DATADOG = 'datadog'
    GRAFANA = 'grafana'
    AXIOM = 'axiom'
    CUSTOM = 'custom'


# Environment variable mappings for each backend
# These use the official env var names from each provider's documentation
_ENV_VARS = {
    Backend.SENTRY: {'dsn': 'SENTRY_DSN'},  # https://docs.sentry.io/platforms/python/configuration/options/
    Backend.HONEYCOMB: {  # https://docs.honeycomb.io/configure/environments/manage-api-keys/
        'api_key': 'HONEYCOMB_API_KEY',
        'dataset': 'HONEYCOMB_DATASET',
        'api_endpoint': 'HONEYCOMB_API_ENDPOINT',
    },
    Backend.DATADOG: {
        'api_key': 'DD_API_KEY',
        'site': 'DD_SITE',
    },  # https://docs.datadoghq.com/agent/guide/environment-variables/
    Backend.GRAFANA: {  # Genkit-specific (Grafana uses standard OTEL_* vars)
        'endpoint': 'GRAFANA_OTLP_ENDPOINT',
        'api_key': 'GRAFANA_API_KEY',
    },
    Backend.AXIOM: {'token': 'AXIOM_TOKEN', 'dataset': 'AXIOM_DATASET'},  # https://axiom.co/docs/reference/tokens
}


def _get_backend_config(
    backend: Backend | str,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[str, dict[str, str]]:
    """Get OTLP endpoint and headers for a backend.

    Args:
        backend: The backend type.
        **kwargs: Backend-specific configuration.

    Returns:
        Tuple of (endpoint, headers).

    Raises:
        ValueError: If required configuration is missing.
    """
    backend = Backend(backend) if isinstance(backend, str) else backend

    if backend == Backend.SENTRY:
        dsn = kwargs.get('sentry_dsn') or os.environ.get('SENTRY_DSN')
        if not dsn:
            raise ValueError('Sentry DSN is required. Set SENTRY_DSN or pass sentry_dsn parameter.')

        # Sentry OTLP endpoint and authentication
        # See: https://docs.sentry.io/concepts/otlp/
        #
        # The OTLP endpoint and public key are available in Sentry project settings:
        # Settings > Projects > [Project] > Client Keys (DSN) > OpenTelemetry (OTLP) tab
        #
        # DSN format: https://{public_key}@{org}.ingest.{region}.sentry.io/{project_id}
        # OTLP endpoint: Project-specific URL from Sentry settings
        # Auth header: x-sentry-auth: sentry sentry_key={public_key}
        #
        # Parse DSN to extract the public key for authentication
        # Modern DSN format with region: https://{key}@{org}.ingest.{region}.sentry.io/{project}
        dsn_pattern = r'https://([^@]+)@([^/]+)/(\d+)'
        match = re.match(dsn_pattern, dsn)
        if match:
            public_key, host, project_id = match.groups()
            # Construct the OTLP endpoint from the DSN host
            # The host already includes the org and region info
            endpoint = f'https://{host}/api/{project_id}/otlp/v1/traces'
            # Sentry OTLP uses x-sentry-auth header with the format: sentry sentry_key=<key>
            return (endpoint, {'x-sentry-auth': f'sentry sentry_key={public_key}'})
        else:
            raise ValueError(
                f'Invalid Sentry DSN format: {dsn}. '
                'Expected format: https://{{public_key}}@{{host}}/{{project_id}}. '
                'Get your DSN from Sentry: Settings > Projects > [Project] > Client Keys (DSN)'
            )

    elif backend == Backend.HONEYCOMB:
        api_key = kwargs.get('honeycomb_api_key') or os.environ.get('HONEYCOMB_API_KEY')
        if not api_key:
            raise ValueError('Honeycomb API key is required. Set HONEYCOMB_API_KEY or pass honeycomb_api_key.')

        # Honeycomb supports custom API endpoints via HONEYCOMB_API_ENDPOINT
        # See: https://docs.honeycomb.io/configure/environments/manage-api-keys/
        # US (default): https://api.honeycomb.io
        # EU: https://api.eu1.honeycomb.io
        api_endpoint = kwargs.get('honeycomb_api_endpoint') or os.environ.get('HONEYCOMB_API_ENDPOINT')
        base_url = api_endpoint.rstrip('/') if api_endpoint else 'https://api.honeycomb.io'

        # Dataset header is only needed for Honeycomb Classic
        # Modern Honeycomb environments auto-create datasets from service.name
        # See: https://docs.honeycomb.io/send-data/python/opentelemetry-sdk/
        dataset = kwargs.get('honeycomb_dataset') or os.environ.get('HONEYCOMB_DATASET')

        headers = {'x-honeycomb-team': api_key}
        if dataset:
            # Only add dataset header for Classic environments
            headers['x-honeycomb-dataset'] = dataset

        return (f'{base_url}/v1/traces', headers)

    elif backend == Backend.DATADOG:
        api_key = kwargs.get('datadog_api_key') or os.environ.get('DD_API_KEY')
        if not api_key:
            raise ValueError('Datadog API key is required. Set DD_API_KEY or pass datadog_api_key.')

        # Datadog OTLP intake endpoint
        # See: https://docs.datadoghq.com/opentelemetry/setup/otlp_ingest/
        # Supported sites: datadoghq.com, datadoghq.eu, us3.datadoghq.com, us5.datadoghq.com, ap1.datadoghq.com
        site = kwargs.get('datadog_site') or os.environ.get('DD_SITE') or 'datadoghq.com'

        # The OTLP/HTTP endpoint format for direct ingestion
        # Note: This sends directly to Datadog without the Agent
        return (
            f'https://otlp.{site}/v1/traces',
            {'DD-API-KEY': api_key},
        )

    elif backend == Backend.GRAFANA:
        # Grafana Cloud OTLP configuration
        # See: https://grafana.com/docs/grafana-cloud/monitor-applications/application-observability/setup/collector/opentelemetry-collector/
        #
        # Authentication uses Basic auth with instance_id:api_key encoded as Base64
        # The endpoint and credentials are available in Grafana Cloud portal:
        # My Account > [Stack] > OpenTelemetry > Configure
        endpoint = kwargs.get('grafana_endpoint') or os.environ.get('GRAFANA_OTLP_ENDPOINT')
        user_id = kwargs.get('grafana_user_id') or os.environ.get('GRAFANA_USER_ID')
        api_key = kwargs.get('grafana_api_key') or os.environ.get('GRAFANA_API_KEY')

        if not endpoint:
            raise ValueError(
                'Grafana endpoint is required. Set GRAFANA_OTLP_ENDPOINT or pass grafana_endpoint. '
                'Find it in Grafana Cloud: My Account > [Stack] > OpenTelemetry > Configure'
            )
        if not user_id:
            raise ValueError(
                'Grafana user ID is required. Set GRAFANA_USER_ID or pass grafana_user_id. '
                'Find it in Grafana Cloud: My Account > [Stack] > OpenTelemetry > Configure'
            )
        if not api_key:
            raise ValueError(
                'Grafana API key is required. Set GRAFANA_API_KEY or pass grafana_api_key. '
                'Generate one in Grafana Cloud: My Account > [Stack] > OpenTelemetry > Configure'
            )

        # Ensure endpoint ends with /v1/traces
        if not endpoint.endswith('/v1/traces'):
            endpoint = endpoint.rstrip('/') + '/v1/traces'

        # Encode credentials as Base64 for Basic auth
        credentials = f'{user_id}:{api_key}'
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        return (endpoint, {'Authorization': f'Basic {encoded_credentials}'})

    elif backend == Backend.AXIOM:
        token = kwargs.get('axiom_api_token') or os.environ.get('AXIOM_TOKEN')
        if not token:
            raise ValueError('Axiom API token is required. Set AXIOM_TOKEN or pass axiom_api_token.')

        dataset = kwargs.get('axiom_dataset') or os.environ.get('AXIOM_DATASET') or 'genkit'

        return (
            'https://api.axiom.co/v1/traces',
            {'Authorization': f'Bearer {token}', 'X-Axiom-Dataset': dataset},
        )

    elif backend == Backend.CUSTOM:
        endpoint = kwargs.get('endpoint')
        if not endpoint:
            raise ValueError('Custom endpoint is required. Pass endpoint parameter.')

        headers = kwargs.get('headers') or {}
        return (endpoint, headers)

    else:
        raise ValueError(f'Unknown backend: {backend}')


def _inject_trace_context(event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Inject trace context into log event for correlation.

    This enables linking logs to traces in observability backends.
    Adds trace_id and span_id fields to log records when a span is active.

    Args:
        event_dict: The structlog event dictionary.

    Returns:
        The event dictionary with trace context added.
    """
    # Only inject if event_dict is a dict or mapping
    if not isinstance(event_dict, dict) and not hasattr(event_dict, '__setitem__'):
        return event_dict

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


def _configure_logging() -> None:
    """Configure structlog with trace correlation.

    Injects trace context into log records so logs can be correlated
    with traces in observability backends like Sentry, Honeycomb, etc.
    """
    try:
        current_config = structlog.get_config()
        processors = current_config.get('processors', [])

        # Check if our processor is already registered
        if not any(getattr(p, '__name__', '') == 'inject_observability_trace_context' for p in processors):

            def inject_observability_trace_context(
                _logger: Any,  # noqa: ANN401
                method_name: str,
                event_dict: MutableMapping[str, Any],
            ) -> Mapping[str, Any]:
                """Inject trace context into log event."""
                return _inject_trace_context(event_dict)

            new_processors = list(processors)
            # Insert before the last processor (usually the renderer)
            new_processors.insert(max(0, len(new_processors) - 1), inject_observability_trace_context)
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


def configure_telemetry(
    backend: Backend | str,
    *,
    # Common options
    service_name: str = 'genkit-app',
    service_version: str = '1.0.0',
    environment: str | None = None,
    sampler: Sampler | None = None,
    log_input_and_output: bool = False,
    force_dev_export: bool = True,
    disable_traces: bool = False,
    # Sentry
    sentry_dsn: str | None = None,
    # Honeycomb
    honeycomb_api_key: str | None = None,
    honeycomb_dataset: str | None = None,
    honeycomb_api_endpoint: str | None = None,
    # Datadog
    datadog_api_key: str | None = None,
    datadog_site: str = 'datadoghq.com',
    # Grafana Cloud
    grafana_endpoint: str | None = None,
    grafana_user_id: str | None = None,
    grafana_api_key: str | None = None,
    # Axiom
    axiom_api_token: str | None = None,
    axiom_dataset: str | None = None,
    # Custom OTLP
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """Configure telemetry export to a third-party backend.

    This function sets up OpenTelemetry export via OTLP to any of the supported
    backends. By default, model inputs and outputs are redacted for privacy.

    Args:
        backend: Which backend to use (sentry, honeycomb, datadog, grafana, axiom, custom).
        service_name: Name of your service (appears in traces). Defaults to "genkit-app".
        service_version: Version of your service. Defaults to "1.0.0".
        environment: Environment name (production, staging, etc.).
        sampler: OpenTelemetry trace sampler. Defaults to AlwaysOnSampler.
        log_input_and_output: If True, preserve model input/output in traces.
            Defaults to False (redact for privacy).
        force_dev_export: If True, export telemetry even in dev environment.
            Defaults to True.
        disable_traces: If True, traces will not be exported.

        # Sentry-specific:
        sentry_dsn: Sentry SDK DSN (for backend="sentry"). Can also use SENTRY_DSN env var.
            This is the standard Sentry DSN from your project settings, NOT the OTLP-specific
            endpoint. Find it in Sentry: Settings > Projects > [Project] > Client Keys (DSN).
            Format: https://{public_key}@{host}/{project_id}
            Example: https://abc123@o123456.ingest.us.sentry.io/4507654321

        # Honeycomb-specific:
        honeycomb_api_key: Honeycomb API key. Can also use HONEYCOMB_API_KEY env var.
        honeycomb_dataset: Honeycomb dataset name. Only needed for Honeycomb Classic.
            Modern Honeycomb environments auto-create datasets from service.name.
        honeycomb_api_endpoint: Honeycomb API endpoint. Defaults to US (https://api.honeycomb.io).
            For EU, use https://api.eu1.honeycomb.io. Can also use HONEYCOMB_API_ENDPOINT env var.

        # Datadog-specific:
        datadog_api_key: Datadog API key. Can also use DD_API_KEY env var.
        datadog_site: Datadog site (e.g., "datadoghq.com", "datadoghq.eu").

        # Grafana Cloud-specific:
        grafana_endpoint: Grafana Cloud OTLP endpoint. Can also use GRAFANA_OTLP_ENDPOINT env var.
            Find it in Grafana Cloud: My Account > [Stack] > OpenTelemetry > Configure.
        grafana_user_id: Grafana Cloud instance ID (numeric). Can also use GRAFANA_USER_ID env var.
            This is the instance ID shown in the OTLP configuration page.
        grafana_api_key: Grafana Cloud API key (token). Can also use GRAFANA_API_KEY env var.
            Generate one in Grafana Cloud with MetricsPublisher role.

        # Axiom-specific:
        axiom_api_token: Axiom API token. Can also use AXIOM_TOKEN env var.
        axiom_dataset: Axiom dataset name. Defaults to "genkit".

        # Custom OTLP:
        endpoint: Custom OTLP endpoint URL (for backend="custom").
        headers: Custom headers for authentication (for backend="custom").

    Raises:
        ValueError: If required backend-specific configuration is missing.

    Example:
        ```python
        # Sentry
        configure_telemetry(backend='sentry', sentry_dsn='https://...')

        # Honeycomb
        configure_telemetry(backend='honeycomb', honeycomb_api_key='...')

        # Datadog
        configure_telemetry(backend='datadog', datadog_api_key='...')

        # Grafana Cloud
        configure_telemetry(
            backend='grafana',
            grafana_endpoint='https://otlp-gateway-prod-us-central-0.grafana.net/otlp',
            grafana_user_id='123456',
            grafana_api_key='glc_...',
        )

        # Axiom
        configure_telemetry(backend='axiom', axiom_api_token='xaat-...')

        # Custom
        configure_telemetry(
            backend='custom',
            endpoint='https://my-collector/v1/traces',
            headers={'Authorization': 'Bearer ...'},
        )
        ```

    See Also:
        - OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/
    """
    is_dev = is_dev_environment()
    should_export = force_dev_export or not is_dev

    if not should_export:
        logger.debug('Telemetry export disabled in dev environment')
        return

    if disable_traces:
        logger.debug('Trace export disabled')
        return

    # Configure structured logging with trace correlation
    # This enables linking logs to traces in observability backends
    _configure_logging()

    # Get backend-specific configuration
    otlp_endpoint, otlp_headers = _get_backend_config(
        backend,
        sentry_dsn=sentry_dsn,
        honeycomb_api_key=honeycomb_api_key,
        honeycomb_dataset=honeycomb_dataset,
        honeycomb_api_endpoint=honeycomb_api_endpoint,
        datadog_api_key=datadog_api_key,
        datadog_site=datadog_site,
        grafana_endpoint=grafana_endpoint,
        grafana_user_id=grafana_user_id,
        grafana_api_key=grafana_api_key,
        axiom_api_token=axiom_api_token,
        axiom_dataset=axiom_dataset,
        endpoint=endpoint,
        headers=headers,
    )

    # Create resource with service info
    resource_attrs = {
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        SERVICE_INSTANCE_ID: str(uuid.uuid4()),
    }
    if environment:
        resource_attrs[DEPLOYMENT_ENVIRONMENT] = environment

    resource = Resource.create(resource_attrs)

    # Create TracerProvider
    provider = TracerProvider(
        resource=resource,
        sampler=sampler,
    )
    trace.set_tracer_provider(provider)

    # Create OTLP exporter
    base_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers=otlp_headers,
    )

    # Wrap with AdjustingTraceExporter for PII redaction
    trace_exporter = AdjustingTraceExporter(
        exporter=base_exporter,
        log_input_and_output=log_input_and_output,
        error_handler=lambda e: logger.error('Error exporting traces', error=str(e)),
    )

    add_custom_exporter(trace_exporter, 'observability')

    logger.info(
        'Observability telemetry configured',
        backend=str(backend),
        service_name=service_name,
    )


def package_name() -> str:
    """Get the package name for the observability plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit.plugins.observability'


__all__ = ['Backend', 'configure_telemetry', 'package_name']
