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

This module configures OpenTelemetry exporters to send distributed traces to
Google Cloud Trace and metrics to Google Cloud Monitoring. It includes automatic
PII redaction and error span adjustment.

Usage:
    ```python
    from genkit import Genkit
    from genkit_googleai import GoogleAI
    from genkit_googlecloud import enable_googlecloud_telemetry

    # Enable telemetry with default settings (PII redaction enabled)
    enable_googlecloud_telemetry(project_id='my-project')

    ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')
    response = await ai.generate(prompt='Hello, world!')
    ```

Requirements:
    - Requires Google Cloud Application Default Credentials (ADC) or explicit credentials.
    - Set ``log_input_and_output=True`` only in trusted environments where prompt/response logging is permitted.

See Also:
    - Cloud Trace: https://cloud.google.com/trace/docs
    - Cloud Monitoring: https://cloud.google.com/monitoring/docs
"""

from typing import Any

import structlog
from opentelemetry.sdk.trace.sampling import Sampler

from .config import GcpTelemetry

logger = structlog.get_logger(__name__)


def enable_googlecloud_telemetry(
    project_id: str | None = None,
    credentials: dict[str, Any] | None = None,
    sampler: Sampler | None = None,
    log_input_and_output: bool = False,
    force_dev_export: bool = False,
    disable_metrics: bool = False,
    disable_traces: bool = False,
    metric_export_interval_ms: int | None = None,
    metric_export_timeout_ms: int | None = None,
    # Legacy parameter name for backwards compatibility
    force_export: bool | None = None,
) -> None:
    """Configure GCP telemetry export for traces and metrics.

    This function sets up OpenTelemetry export to Google Cloud Trace and
    Cloud Monitoring. By default, model inputs and outputs are redacted
    for privacy protection.

    Configuration options match the JavaScript (GcpTelemetryConfigOptions) and
    Go (FirebaseTelemetryOptions/GoogleCloudTelemetryOptions) implementations.

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
        log_input_and_output: If True, preserve model input/output in traces
            and logs. Defaults to False (redact for privacy). Only enable this
            in trusted environments where PII exposure is acceptable.
            Maps to JS: !disableLoggingInputAndOutput
        force_dev_export: If True, export telemetry even in dev environment.
            Defaults to True. Set to False for production-only telemetry.
            Maps to JS: forceDevExport
        disable_metrics: If True, metrics will not be exported. Traces and
            logs may still be exported. Defaults to False.
            Maps to JS/Go: disableMetrics
        disable_traces: If True, traces will not be exported. Metrics and
            logs may still be exported. Defaults to False.
            Maps to JS/Go: disableTraces
        metric_export_interval_ms: Metrics export interval in milliseconds.
            GCP requires a minimum of 5000ms. Defaults to 60000ms.
            Dev environment uses 5000ms, production uses 300000ms by default
            in JS/Go (but we use 60000ms for consistent behavior).
            Maps to JS/Go: metricExportIntervalMillis
        metric_export_timeout_ms: Timeout for metrics export in milliseconds.
            Defaults to the export interval if not specified.
            Maps to JS/Go: metricExportTimeoutMillis
        force_export: Deprecated. Use force_dev_export instead.

    Example:
        ```python
        # Default: PII redaction enabled
        enable_googlecloud_telemetry()

        # Enable input/output logging (disable PII redaction)
        enable_googlecloud_telemetry(log_input_and_output=True)

        # Force export in dev environment with specific project
        enable_googlecloud_telemetry(force_dev_export=True, project_id='my-project')

        # Disable metrics but keep traces
        enable_googlecloud_telemetry(disable_metrics=True)

        # Custom metric export interval (minimum 5000ms)
        enable_googlecloud_telemetry(metric_export_interval_ms=30000)

        # With custom credentials for non-GCP environments
        enable_googlecloud_telemetry(
            project_id='my-project',
            credentials={'type': 'service_account', ...},
        )
        ```

    Note:
        This matches the JavaScript implementation's GcpTelemetryConfigOptions
        and Go's FirebaseTelemetryOptions/GoogleCloudTelemetryOptions.

    See Also:
        - JS: js/plugins/google-cloud/src/types.ts (GcpTelemetryConfigOptions)
        - Go: go/plugins/firebase/telemetry.go (FirebaseTelemetryOptions)
        - Go: go/plugins/googlecloud/types.go (GoogleCloudTelemetryOptions)
    """
    # Handle legacy force_export parameter
    if force_export is not None:
        logger.warning('force_export is deprecated, use force_dev_export instead')
        force_dev_export = force_export

    manager = GcpTelemetry(
        project_id=project_id,
        credentials=credentials,
        sampler=sampler,
        log_input_and_output=log_input_and_output,
        force_dev_export=force_dev_export,
        disable_metrics=disable_metrics,
        disable_traces=disable_traces,
        metric_export_interval_ms=metric_export_interval_ms,
        metric_export_timeout_ms=metric_export_timeout_ms,
    )

    manager.initialize()
