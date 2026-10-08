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


"""Telemetry and tracing functionality for the Genkit Google Cloud plugin.

This module configures OpenTelemetry exporters to send traces to Cloud Trace,
metrics to Cloud Monitoring, and log records to Cloud Logging.

Usage:
    ```python
    from genkit import Genkit
    from genkit_google_genai import GoogleAI
    from genkit_google_cloud import enable_google_cloud_telemetry

    enable_google_cloud_telemetry(project_id='my-project')

    # 2. All subsequent Genkit actions automatically export telemetry
    ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))
    await ai.generate(prompt='Hello, world!')
    ```

Requirements:
    - Requires Google Cloud Application Default Credentials (ADC) or explicit credentials.

See Also:
    - Cloud Trace: https://cloud.google.com/trace/docs
    - Cloud Monitoring: https://cloud.google.com/monitoring/docs
"""

from typing import Any

from opentelemetry.sdk.trace.sampling import Sampler

from genkit import GenkitError

from ._config import GcpTelemetry, _reject_unusable_cloud_setup

# Once per process: the app (or a test) may call enable_google_cloud_telemetry
# once. A second call raises so Cloud Trace does not get two exporters.
_enable_google_cloud_telemetry_already_called = False


def _reset_google_cloud_telemetry() -> None:
    """Clear the once-per-process latch. Tests only."""
    global _enable_google_cloud_telemetry_already_called
    _enable_google_cloud_telemetry_already_called = False


def enable_google_cloud_telemetry(
    project_id: str | None = None,
    credentials: dict[str, Any] | None = None,
    sampler: Sampler | None = None,
    force_dev_export: bool = False,
    disable_metrics: bool = False,
    disable_traces: bool = False,
    metric_export_interval_ms: int | None = None,
    metric_export_timeout_ms: int | None = None,
) -> None:
    """Attach Cloud Trace and Cloud Monitoring exporters.

    Call this once from the app. A second call raises. This hangs Cloud
    Trace, Monitoring, and Logging on the process-global OpenTelemetry
    providers and turns on ``GenAiInstrumentation`` unless one is already
    minting. Under ``genkit start``, ``Genkit()`` still attaches the
    Developer UI collector.

    If the app already set an SDK ``TracerProvider``, Cloud Trace is added
    next to that exporter. Set your provider first; calling this helper
    and then ``trace.set_tracer_provider(...)`` leaves the app's exporter
    unused. A provider that is not the SDK class raises
    ``FAILED_PRECONDITION``. The same check applies to the process logger.

    Cloud exporters are skipped when ``GENKIT_ENV=dev`` and
    ``force_dev_export=False``. ``disable_traces=True`` skips Cloud Trace
    only; GenAI still turns on. Prompt and reply text are not written
    on GenAI spans. To put raw action I/O on the span, register
    ``GenAiInstrumentation(capture_action_io=True)`` before ``Genkit()``.
    Log records the instrumentation emits (content-capture log events)
    go to Cloud Logging. Sampler and provider checks still run under
    ``GENKIT_ENV=dev``.

    Args:
        project_id: Google Cloud project ID. Wins over ``GOOGLE_CLOUD_PROJECT``,
            ``GCLOUD_PROJECT``, the project on ``credentials``, and the
            Application Default Credentials project, in that order. Required
            when using external credentials (e.g., Workload Identity
            Federation).
        credentials: Service account credentials dict for authenticating with
            Google Cloud. Primarily for use outside of GCP. On GCP, credentials
            are typically inferred via Application Default Credentials (ADC).
        sampler: Sampler used when this helper creates the process tracer.
            Pass ``ALWAYS_ON``, ``ALWAYS_OFF``, or ``TraceIdRatioBased(0.1)``
            from ``opentelemetry.sdk.trace.sampling``. With no ``sampler=``,
            the SDK default applies (parent-based, always on, unless
            ``OTEL_TRACES_SAMPLER`` says otherwise). If the app already set
            an SDK ``TracerProvider``, leave ``sampler=`` off: Cloud Trace
            joins that provider and uses its sampler. Passing ``sampler=``
            then raises ``INVALID_ARGUMENT``, since a provider's sampler is
            fixed when it is built. ``sampler=`` with ``disable_traces=True``
            also raises.
        force_dev_export: If True, export Cloud telemetry even when
            ``GENKIT_ENV=dev``. Defaults to False.
        disable_metrics: If True, Cloud Monitoring is not hung. Traces and
            logs may still be exported. Defaults to False.
        disable_traces: If True, Cloud Trace is not hung. GenAI still
            turns on. Metrics and logs may still be exported. Defaults to False.
        metric_export_interval_ms: Metrics export interval in milliseconds.
            GCP requires a minimum of 5000ms. Defaults to 60000ms.
        metric_export_timeout_ms: Timeout for metrics export in milliseconds.
            Defaults to the export interval if not specified.

    Example:
        ```python
        enable_google_cloud_telemetry()

        # Force export in dev environment with specific project
        enable_google_cloud_telemetry(force_dev_export=True, project_id='my-project')

        # Disable metrics but keep traces
        enable_google_cloud_telemetry(disable_metrics=True)

        # Custom metric export interval (minimum 5000ms)
        enable_google_cloud_telemetry(metric_export_interval_ms=30000)

        # With custom credentials for non-GCP environments
        enable_google_cloud_telemetry(
            project_id='my-project',
            credentials={'type': 'service_account', ...},
        )
        ```

        Sample 10% of traces. ``ParentBased`` keeps the caller's decision
        when a request arrives with a ``traceparent``, so a trace that
        started upstream is not cut in half:

        ```python
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

        enable_google_cloud_telemetry(
            project_id='my-project',
            sampler=ParentBased(TraceIdRatioBased(0.1)),
        )
        # => about 1 in 10 new traces reach Cloud Trace
        #    a request whose caller sampled it is always kept
        ```

        If the app already set a ``TracerProvider``, put the sampler on it
        and leave ``sampler=`` off:

        ```python
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

        # 1. App's provider, with the sampler
        trace.set_tracer_provider(TracerProvider(sampler=ParentBased(TraceIdRatioBased(0.1))))

        # 2. Cloud Trace joins that provider and inherits its sampler
        enable_google_cloud_telemetry(project_id='my-project')
        # => Cloud Trace gets the same 10% the app's exporter gets

        # Passing sampler= here instead raises:
        # enable_google_cloud_telemetry(sampler=ParentBased(TraceIdRatioBased(0.1)))
        # => GenkitError INVALID_ARGUMENT: a tracer provider is already set;
        #    pass TracerProvider(sampler=...) when you create it instead of sampler=
        ```

    See Also:
        - Cloud Trace: https://cloud.google.com/trace/docs
        - Cloud Monitoring: https://cloud.google.com/monitoring/docs
        - Cloud Logging: https://cloud.google.com/logging/docs
    """
    global _enable_google_cloud_telemetry_already_called
    if _enable_google_cloud_telemetry_already_called:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message='enable_google_cloud_telemetry() was already called. Call it once from the app.',
        )
    # Before the flag is set, so a rejected call can be fixed and retried.
    _reject_unusable_cloud_setup(sampler=sampler, disable_traces=disable_traces)
    _enable_google_cloud_telemetry_already_called = True

    manager = GcpTelemetry(
        project_id=project_id,
        credentials=credentials,
        sampler=sampler,
        force_dev_export=force_dev_export,
        disable_metrics=disable_metrics,
        disable_traces=disable_traces,
        metric_export_interval_ms=metric_export_interval_ms,
        metric_export_timeout_ms=metric_export_timeout_ms,
    )

    manager.initialize()
