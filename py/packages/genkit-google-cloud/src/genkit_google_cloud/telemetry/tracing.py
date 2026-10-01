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

import structlog
from opentelemetry.sdk.trace.sampling import Sampler

from genkit._core._error import GenkitError

from .config import GcpTelemetry

logger = structlog.get_logger(__name__)

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
    # Legacy parameter name for backwards compatibility
    force_export: bool | None = None,
) -> None:
    """Attach Cloud Trace and Cloud Monitoring exporters.

    Call this once from the app. A second call raises. This hangs Cloud
    Trace, Monitoring, and Logging on the process-global OpenTelemetry
    providers and turns on ``GenAiInstrumentation`` unless one is already
    minting. Under ``genkit start``, ``Genkit()`` still attaches the
    Developer UI collector.

    Cloud exporters are skipped when ``GENKIT_ENV=dev`` and
    ``force_dev_export=False``. ``disable_traces=True`` skips Cloud Trace
    only; GenAI still turns on. Prompt and reply text are not written
    on GenAI spans. To put raw action I/O on the span, register
    ``GenAiInstrumentation(capture_action_io=True)`` before ``Genkit()``.
    Log records the instrumentation emits (content-capture log events)
    go to Cloud Logging.

    Args:
        project_id: Google Cloud project ID. If provided, takes precedence over
            environment variables and credentials. Required when using external
            credentials (e.g., Workload Identity Federation).
        credentials: Service account credentials dict for authenticating with
            Google Cloud. Primarily for use outside of GCP. On GCP, credentials
            are typically inferred via Application Default Credentials (ADC).
        sampler: OpenTelemetry trace sampler. Controls which traces are collected
            and exported. Defaults to AlwaysOnSampler. Common options:
            - AlwaysOnSampler: Collect all traces
            - AlwaysOffSampler: Collect no traces
            - TraceIdRatioBasedSampler: Sample a percentage of traces
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
        force_export: Deprecated. Use force_dev_export instead.

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
    _enable_google_cloud_telemetry_already_called = True

    # Handle legacy force_export parameter
    if force_export is not None:
        logger.warning('force_export is deprecated, use force_dev_export instead')
        force_dev_export = force_export

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
