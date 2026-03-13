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

"""Telemetry and tracing functionality for the Genkit AWS plugin.

This module provides functionality for collecting and exporting telemetry data
from Genkit operations to AWS. It uses OpenTelemetry for tracing and exports
span data to AWS X-Ray for monitoring and debugging purposes.

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
    │ X-Ray               │ AWS service that collects and visualizes traces.   │
    │                     │ Like a detective board connecting all the clues.   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ OTLP                │ OpenTelemetry Protocol - a standard way to send    │
    │                     │ traces. Like a universal shipping label format.    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ SigV4               │ AWS's way of signing requests to prove identity.   │
    │                     │ Like a secret handshake with AWS.                  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Propagator          │ Passes trace IDs between services. Like a relay    │
    │                     │ baton so X-Ray can connect the dots.               │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ID Generator        │ Creates trace IDs in X-Ray's special format.       │
    │                     │ X-Ray needs timestamps baked into IDs.             │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Sampler             │ Decides which traces to keep. Like a bouncer       │
    │                     │ deciding which requests get recorded.              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Exporter            │ Ships your traces to X-Ray. Like a postal service  │
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
    │           │  AwsAdjustingTraceExporter  │                               │
    │           │  • Redact PII (input/output)│                               │
    │           │  • Fix zero-duration spans  │                               │
    │           │  • Add error markers        │                               │
    │           └─────────────┬───────────────┘                               │
    │                         │                                               │
    │                         │  (4) Sent to AWS via OTLP/HTTP                │
    │                         ▼                                               │
    │              ┌─────────────────────┐                                    │
    │              │ AwsXRayOtlpExporter │                                    │
    │              │ (with SigV4 auth)   │                                    │
    │              └──────────┬──────────┘                                    │
    │                         │                                               │
    │                         │  (5) HTTPS to X-Ray endpoint                  │
    │                         ▼                                               │
    │    ════════════════════════════════════════════════════                 │
    │                         │                                               │
    │                         ▼                                               │
    │              ┌─────────────────────┐                                    │
    │              │     AWS X-Ray       │   View traces in AWS Console       │
    │              │   (your traces!)    │   Debug latency, errors, etc.      │
    │              └─────────────────────┘                                    │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         TELEMETRY DATA FLOW                             │
    │                                                                         │
    │  Genkit Actions (flows, models, tools)                                  │
    │         │                                                               │
    │         ▼                                                               │
    │  ┌─────────────────┐                                                    │
    │  │ OpenTelemetry   │  Creates spans with genkit:* attributes            │
    │  │ Tracer          │  (type, name, input, output, state, path, etc.)    │
    │  │ + AwsXRayIdGen  │  Uses X-Ray-compatible trace ID format             │
    │  └────────┬────────┘                                                    │
    │           │                                                             │
    │           ▼                                                             │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │           AwsAdjustingTraceExporter                             │    │
    │  │  ┌─────────────────────────────────────────────────────────┐    │    │
    │  │  │ AdjustingTraceExporter._adjust()                        │    │    │
    │  │  │    - Redact genkit:input/output → "<redacted>"          │    │    │
    │  │  │    - Mark error spans with /http/status_code: 599       │    │    │
    │  │  │    - Normalize labels for X-Ray compatibility           │    │    │
    │  │  │    - TimeAdjustedSpan ensures end > start               │    │    │
    │  │  └─────────────────────────────────────────────────────────┘    │    │
    │  └────────────────────────┬────────────────────────────────────────┘    │
    │                           │                                             │
    │                           ▼                                             │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │                AwsXRayOtlpExporter                               │    │
    │  │  ┌─────────────────────────────────────────────────────────┐    │    │
    │  │  │ OTLP/HTTP Export with SigV4 Authentication              │    │    │
    │  │  │ Endpoint: https://xray.{region}.amazonaws.com/v1/traces │    │    │
    │  │  └─────────────────────────────────────────────────────────┘    │    │
    │  └────────────────────────┬────────────────────────────────────────┘    │
    │                           │                                             │
    │                           ▼                                             │
    │  ┌─────────────────┐                                                    │
    │  │ AWS X-Ray       │                                                    │
    │  │ Service         │                                                    │
    │  └─────────────────┘                                                    │
    └─────────────────────────────────────────────────────────────────────────┘

Key Components:
    1. **AwsTelemetry**: Central manager class that encapsulates configuration
       and manages the lifecycle of tracing and logging setup.

    2. **AwsXRayOtlpExporter**: Custom OTLP/HTTP exporter with SigV4 signing.
       Uses botocore credentials to authenticate requests to AWS X-Ray.

    3. **AwsAdjustingTraceExporter**: Extends AdjustingTraceExporter to add
       AWS-specific handling before spans are adjusted and exported.

    4. **TimeAdjustedSpan**: Wrapper that ensures spans have non-zero duration
       (X-Ray requires end_time > start_time).

    5. **AwsXRayIdGenerator**: From opentelemetry-sdk-extension-aws, generates
       X-Ray-compatible trace IDs with Unix timestamp in first 32 bits.

    6. **AwsXRayPropagator**: From opentelemetry-propagator-aws-xray, handles
       trace context propagation across AWS services.

X-Ray Trace ID Requirements:
    AWS X-Ray requires trace IDs to have Unix epoch time in the first 32 bits.
    The AwsXRayIdGenerator ensures compatibility. Spans with timestamps older
    than 30 days are rejected by X-Ray.

Log-Trace Correlation:
    The AwsTelemetry manager injects X-Ray trace context into structlog logs,
    enabling correlation between logs and traces in CloudWatch. The format
    uses `_X_AMZN_TRACE_ID` which matches AWS X-Ray conventions.

Configuration Options::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Option                      │ Type     │ Default    │ Description       │
    ├─────────────────────────────┼──────────┼────────────┼───────────────────┤
    │ region                      │ str      │ AWS_REGION │ AWS region        │
    │ log_input_and_output        │ bool     │ False      │ Disable redaction │
    │ force_dev_export            │ bool     │ True       │ Export in dev     │
    │ disable_traces              │ bool     │ False      │ Skip traces       │
    │ sampler                     │ Sampler  │ AlwaysOn   │ Trace sampler     │
    └─────────────────────────────┴──────────┴────────────┴───────────────────┘

Region Resolution Order:
    1. Explicit region parameter
    2. AWS_REGION environment variable
    3. AWS_DEFAULT_REGION environment variable

Usage:
    ```python
    from genkit.plugins.amazon_bedrock import add_aws_telemetry

    # Enable telemetry with default settings (PII redaction enabled)
    add_aws_telemetry()

    # Enable telemetry with explicit region
    add_aws_telemetry(region='us-west-2')

    # Enable input/output logging (disable PII redaction)
    add_aws_telemetry(log_input_and_output=True)

    # Force export in dev environment
    add_aws_telemetry(force_dev_export=True)
    ```

AWS Documentation References:
    X-Ray:
        - Overview: https://docs.aws.amazon.com/xray/
        - OTLP Endpoint: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch-OTLPEndpoint.html
        - IAM Role: AWSXrayWriteOnlyPolicy

    ADOT Python:
        - Getting Started: https://aws-otel.github.io/docs/getting-started/python-sdk
        - Migration Guide: https://docs.aws.amazon.com/xray/latest/devguide/migrate-xray-to-opentelemetry-python.html
"""

import os
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any, cast

import requests
import structlog
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session as BotocoreSession
from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagators.aws import AwsXRayPropagator
from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator
from opentelemetry.sdk.resources import SERVICE_INSTANCE_ID, SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import Sampler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from genkit.core.environment import is_dev_environment
from genkit.core.trace.adjusting_exporter import AdjustingTraceExporter, RedactedSpan
from genkit.core.tracing import add_custom_exporter

logger = structlog.get_logger(__name__)

# X-Ray OTLP endpoint pattern
XRAY_OTLP_ENDPOINT_PATTERN = 'https://xray.{region}.amazonaws.com/v1/traces'


def _resolve_region(region: str | None = None) -> str | None:
    """Resolve the AWS region from various sources.

    Resolution order:
    1. Explicit region parameter
    2. AWS_REGION environment variable
    3. AWS_DEFAULT_REGION environment variable

    Args:
        region: Explicitly provided AWS region.

    Returns:
        The resolved region or None if not found.
    """
    if region:
        return region

    # Check environment variables in order of priority
    for env_var in ('AWS_REGION', 'AWS_DEFAULT_REGION'):
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value

    return None


class AwsTelemetry:
    """Central manager for AWS Telemetry configuration.

    Encapsulates configuration and manages the lifecycle of Tracing and Logging
    setup, ensuring consistent state (like region) across all telemetry components.

    Example:
        ```python
        telemetry = AwsTelemetry(region='us-west-2')
        telemetry.initialize()
        ```
    """

    def __init__(
        self,
        region: str | None = None,
        sampler: Sampler | None = None,
        log_input_and_output: bool = False,
        force_dev_export: bool = True,
        disable_traces: bool = False,
    ) -> None:
        """Initialize the AWS Telemetry manager.

        Args:
            region: AWS region for X-Ray endpoint.
            sampler: Trace sampler.
            log_input_and_output: If False, redacts sensitive data.
            force_dev_export: If True, exports even in dev environment.
            disable_traces: If True, traces are not exported.

        Raises:
            ValueError: If region cannot be resolved.
        """
        self.sampler = sampler
        self.log_input_and_output = log_input_and_output
        self.force_dev_export = force_dev_export
        self.disable_traces = disable_traces

        # Resolve region immediately
        self.region = _resolve_region(region)

        if self.region is None:
            raise ValueError(
                'AWS region is required. Set AWS_REGION environment variable '
                'or pass region parameter to add_aws_telemetry().'
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
        """Configure structlog with X-Ray trace correlation.

        Injects X-Ray trace ID into log records for correlation in CloudWatch.
        """
        try:
            current_config = structlog.get_config()
            processors = current_config.get('processors', [])

            # Check if our processor is already registered (by function name)
            # Note: Use the exact function name being added, not the method name
            if not any(getattr(p, '__name__', '') == 'inject_xray_trace_context' for p in processors):

                def inject_xray_trace_context(
                    _logger: Any,  # noqa: ANN401
                    method_name: str,
                    event_dict: MutableMapping[str, Any],
                ) -> Mapping[str, Any]:
                    """Inject X-Ray trace context into log event."""
                    return self._inject_trace_context(event_dict)

                new_processors = list(processors)
                # Insert before the last processor (usually the renderer)
                new_processors.insert(max(0, len(new_processors) - 1), inject_xray_trace_context)
                cfg = structlog.get_config()
                structlog.configure(
                    processors=new_processors,
                    wrapper_class=cfg.get('wrapper_class'),
                    context_class=cfg.get('context_class'),
                    logger_factory=cfg.get('logger_factory'),
                    cache_logger_on_first_use=cfg.get('cache_logger_on_first_use'),
                )
                logger.debug('Configured structlog for AWS X-Ray trace correlation')

        except Exception as e:
            logger.warning('Failed to configure structlog for trace correlation', error=str(e))

    def _configure_tracing(self) -> None:
        """Configure trace export to AWS X-Ray."""
        if self.disable_traces:
            return

        # Region is guaranteed to be set by __init__ (raises ValueError if None)
        assert self.region is not None

        # Configure X-Ray propagator for trace context
        propagate.set_global_textmap(AwsXRayPropagator())

        # Create resource with service info
        resource = Resource.create({
            SERVICE_NAME: 'genkit',
            SERVICE_INSTANCE_ID: str(uuid.uuid4()),
        })

        # Create TracerProvider with X-Ray ID generator
        provider = TracerProvider(
            resource=resource,
            id_generator=AwsXRayIdGenerator(),
            sampler=self.sampler,
        )
        trace.set_tracer_provider(provider)

        # Create the base X-Ray OTLP exporter
        base_exporter = AwsXRayOtlpExporter(
            region=self.region,
            error_handler=lambda e: _handle_tracing_error(e),
        )

        # Wrap with AwsAdjustingTraceExporter for PII redaction
        trace_exporter = AwsAdjustingTraceExporter(
            exporter=base_exporter,
            log_input_and_output=self.log_input_and_output,
            region=self.region,
            error_handler=lambda e: _handle_tracing_error(e),
        )

        add_custom_exporter(trace_exporter, 'aws_xray_telemetry')

        logger.info(
            'AWS X-Ray telemetry configured',
            region=self.region,
            endpoint=XRAY_OTLP_ENDPOINT_PATTERN.format(region=self.region),
        )

    def _inject_trace_context(self, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Inject AWS X-Ray trace context into log event.

        Adds X-Ray trace ID in the format expected by CloudWatch Logs for
        correlation with X-Ray traces.

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

        # Format trace ID for X-Ray (first 8 chars are timestamp, rest is random)
        # X-Ray trace ID format: 1-{timestamp}-{random}
        trace_id_hex = f'{ctx.trace_id:032x}'
        xray_trace_id = f'1-{trace_id_hex[:8]}-{trace_id_hex[8:]}'

        # Add X-Ray trace header format for CloudWatch correlation
        sampled = '1' if ctx.trace_flags.sampled else '0'
        event_dict['_X_AMZN_TRACE_ID'] = f'Root={xray_trace_id};Parent={ctx.span_id:016x};Sampled={sampled}'

        return event_dict


class SigV4SigningAdapter(HTTPAdapter):
    """HTTP adapter that signs requests with AWS SigV4 authentication.

    This adapter intercepts all HTTP requests and signs them with AWS SigV4
    before sending. This enables the OTLP exporter to authenticate with
    AWS services like X-Ray that require SigV4.

    Example:
        ```python
        session = requests.Session()
        adapter = SigV4SigningAdapter(
            credentials=botocore_session.get_credentials(),
            region='us-west-2',
            service='xray',
        )
        session.mount('https://', adapter)
        ```
    """

    def __init__(
        self,
        credentials: Any,  # noqa: ANN401 - botocore credentials type
        region: str,
        service: str = 'xray',
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the SigV4 signing adapter.

        Args:
            credentials: Botocore credentials object.
            region: AWS region for signing.
            service: AWS service name for signing (default: 'xray').
            **kwargs: Additional arguments passed to HTTPAdapter.
        """
        super().__init__(**kwargs)
        self._credentials = credentials
        self._region = region
        self._service = service

    def send(  # type: ignore[override]
        self,
        request: requests.PreparedRequest,
        **kwargs: Any,  # noqa: ANN401
    ) -> requests.Response:
        """Send the request after signing it with SigV4.

        Args:
            request: The prepared request to send.
            **kwargs: Additional arguments passed to the parent send method.

        Returns:
            The response from the server.
        """
        if self._credentials is not None:
            # Sign the request
            aws_request = AWSRequest(
                method=request.method or 'POST',
                url=request.url or '',
                headers=dict(request.headers) if request.headers else {},
                data=request.body or b'',
            )
            SigV4Auth(self._credentials, self._service, self._region).add_auth(aws_request)

            # Update the original request with signed headers
            if request.headers is None:
                request.headers = cast(Any, {})
            request.headers.update(dict(aws_request.headers))

        return super().send(request, **kwargs)


def _create_sigv4_session(
    credentials: Any,  # noqa: ANN401 - botocore credentials type
    region: str,
    service: str = 'xray',
) -> requests.Session:
    """Create a requests Session that signs all requests with SigV4.

    Args:
        credentials: Botocore credentials object.
        region: AWS region for signing.
        service: AWS service name for signing.

    Returns:
        A configured requests Session with SigV4 signing.
    """
    session = requests.Session()

    # Configure retry logic
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )

    # Create signing adapter with retry
    adapter = SigV4SigningAdapter(
        credentials=credentials,
        region=region,
        service=service,
        max_retries=retry,
    )

    # Mount the signing adapter for HTTPS requests
    session.mount('https://', adapter)

    return session


class AwsXRayOtlpExporter(SpanExporter):
    """OTLP/HTTP exporter with AWS SigV4 authentication for X-Ray.

    This exporter sends spans via OTLP/HTTP to the AWS X-Ray endpoint,
    signing each request with AWS SigV4 authentication using botocore.

    The SigV4 signing is implemented via a custom requests Session adapter
    that intercepts all outgoing HTTP requests and adds the required
    Authorization, X-Amz-Date, and X-Amz-Security-Token headers.

    Args:
        region: AWS region for the X-Ray endpoint.
        error_handler: Optional callback for export errors.

    Example:
        ```python
        exporter = AwsXRayOtlpExporter(region='us-west-2')
        ```

    Note:
        Uses standard AWS credential chain (environment variables, IAM role,
        credential file, etc.) via botocore.
    """

    def __init__(
        self,
        region: str,
        error_handler: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize the X-Ray OTLP exporter.

        Args:
            region: AWS region for the X-Ray endpoint.
            error_handler: Optional callback invoked when export errors occur.
        """
        self._region = region
        self._error_handler = error_handler
        self._endpoint = XRAY_OTLP_ENDPOINT_PATTERN.format(region=region)

        # Initialize botocore session for SigV4 signing
        self._botocore_session = BotocoreSession()
        self._credentials = self._botocore_session.get_credentials()

        if self._credentials is None:
            logger.warning(
                'No AWS credentials found for SigV4 signing. X-Ray export will fail without valid credentials.'
            )

        # Create a session with SigV4 signing adapter
        signing_session = _create_sigv4_session(
            credentials=self._credentials,
            region=region,
            service='xray',
        )

        # Create the underlying OTLP exporter with signing session
        self._otlp_exporter = OTLPSpanExporter(
            endpoint=self._endpoint,
            session=signing_session,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to AWS X-Ray via OTLP/HTTP with SigV4 authentication.

        All requests are automatically signed with AWS SigV4 using the
        credentials from the botocore session.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects to export.

        Returns:
            SpanExportResult indicating success or failure.
        """
        try:
            return self._otlp_exporter.export(spans)
        except Exception as ex:
            logger.error('Error while writing to X-Ray', exc_info=ex)
            if self._error_handler:
                self._error_handler(ex)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._otlp_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending spans.

        Args:
            timeout_millis: Timeout in milliseconds.

        Returns:
            True if flush succeeded.
        """
        return self._otlp_exporter.force_flush(timeout_millis)


class TimeAdjustedSpan(RedactedSpan):
    """Wraps a span to ensure non-zero duration.

    X-Ray requires end_time > start_time. This wrapper ensures spans have
    at least 1 microsecond duration.
    """

    @property
    def end_time(self) -> int | None:
        """Return the span end time, adjusted to be > start_time.

        Returns:
            The end time, with minimum 1 microsecond after start if needed.
        """
        start = self._span.start_time
        end = self._span.end_time

        # X-Ray requires end_time > start_time.
        # If the span is unfinished (end_time is None) or has zero duration,
        # we provide a minimum 1 microsecond duration.
        if start is not None and (end is None or end <= start):
            return start + 1000  # 1 microsecond in nanoseconds

        return end


class AwsAdjustingTraceExporter(AdjustingTraceExporter):
    """AWS-specific span exporter that adds X-Ray compatibility.

    This extends the base AdjustingTraceExporter to handle AWS X-Ray
    specific requirements before spans are adjusted and exported.

    Example:
        ```python
        exporter = AwsAdjustingTraceExporter(
            exporter=AwsXRayOtlpExporter(region='us-west-2'),
            log_input_and_output=False,
            region='us-west-2',
        )
        ```
    """

    def __init__(
        self,
        exporter: SpanExporter,
        log_input_and_output: bool = False,
        region: str | None = None,
        error_handler: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize the AWS adjusting trace exporter.

        Args:
            exporter: The underlying SpanExporter to wrap.
            log_input_and_output: If True, preserve input/output in spans.
                Defaults to False (redact for privacy).
            region: AWS region (for future use in telemetry).
            error_handler: Optional callback invoked when export errors occur.
        """
        super().__init__(
            exporter=exporter,
            log_input_and_output=log_input_and_output,
            error_handler=error_handler,
        )
        # Store region for potential future AWS-specific telemetry
        self._region = region

    def _adjust(self, span: ReadableSpan) -> ReadableSpan:
        """Apply all adjustments to a span including time adjustment.

        This overrides the base method to add time adjustment for X-Ray
        compatibility (end_time must be > start_time).

        Args:
            span: The span to adjust.

        Returns:
            The adjusted span with guaranteed non-zero duration.
        """
        # Apply standard adjustments from base class
        span = super()._adjust(span)

        # Fix start/end times for X-Ray (must be end > start)
        return TimeAdjustedSpan(span, dict(span.attributes) if span.attributes else {})


def add_aws_telemetry(
    region: str | None = None,
    sampler: Sampler | None = None,
    log_input_and_output: bool = False,
    force_dev_export: bool = True,
    disable_traces: bool = False,
) -> None:
    """Configure AWS telemetry export for traces to X-Ray.

    This function sets up OpenTelemetry export to AWS X-Ray. By default,
    model inputs and outputs are redacted for privacy protection.

    Args:
        region: AWS region for X-Ray endpoint. If not provided, uses
            AWS_REGION or AWS_DEFAULT_REGION environment variables.
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

    Raises:
        ValueError: If region cannot be resolved from parameters or environment.

    Example:
        ```python
        # Default: PII redaction enabled, uses AWS_REGION env var
        add_aws_telemetry()

        # Enable input/output logging (disable PII redaction)
        add_aws_telemetry(log_input_and_output=True)

        # Force export in dev environment with specific region
        add_aws_telemetry(force_dev_export=True, region='us-west-2')
        ```

    Note:
        This implementation currently sends traces to an OTLP endpoint.
        For collector-less export with SigV4 authentication, either:
        1. Run an ADOT collector locally that handles authentication
        2. Use ADOT auto-instrumentation with environment variables:
           - OTEL_PYTHON_DISTRO=aws_distro
           - OTEL_PYTHON_CONFIGURATOR=aws_configurator
           - OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://xray.{region}.amazonaws.com/v1/traces

    See Also:
        - AWS X-Ray: https://docs.aws.amazon.com/xray/
        - ADOT Python: https://aws-otel.github.io/docs/getting-started/python-sdk
    """
    manager = AwsTelemetry(
        region=region,
        sampler=sampler,
        log_input_and_output=log_input_and_output,
        force_dev_export=force_dev_export,
        disable_traces=disable_traces,
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
    if 'permission' in error_str or 'denied' in error_str or '403' in error_str:
        _tracing_error_logged = True
        logger.error(
            'Unable to send traces to AWS X-Ray. '
            'Ensure the IAM role/user has the "AWSXrayWriteOnlyPolicy" policy '
            'or xray:PutTraceSegments permission. '
            f'Error: {error}'
        )
    elif 'credential' in error_str or 'unauthorized' in error_str or '401' in error_str:
        _tracing_error_logged = True
        logger.error(
            'AWS credentials not found or invalid. '
            'Configure credentials via environment variables (AWS_ACCESS_KEY_ID, '
            'AWS_SECRET_ACCESS_KEY), IAM role, or ~/.aws/credentials file. '
            f'Error: {error}'
        )
    else:
        logger.error('Error exporting traces to AWS X-Ray', error=str(error))
