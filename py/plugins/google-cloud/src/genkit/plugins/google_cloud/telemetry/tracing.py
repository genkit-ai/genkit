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

This module provides functionality for collecting and exporting telemetry data
from Genkit operations to Google Cloud. It uses OpenTelemetry for tracing and
exports span data to Google Cloud Trace for monitoring and debugging purposes.

Architecture Overview:
    The telemetry system follows a pipeline architecture that processes spans
    (traces) and metrics before exporting them to Google Cloud:

    ```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         TELEMETRY DATA FLOW                             │
    │                                                                         │
    │  Genkit Actions (flows, models, tools)                                  │
    │         │                                                               │
    │         ▼                                                               │
    │  ┌─────────────────┐                                                    │
    │  │ OpenTelemetry   │  Creates spans with genkit:* attributes            │
    │  │ Tracer          │  (type, name, input, output, state, path, etc.)    │
    │  └────────┬────────┘                                                    │
    │           │                                                             │
    │           ▼                                                             │
    │  ┌─────────────────────────────────────────────────────────────┐        │
    │  │           GcpAdjustingTraceExporter                         │        │
    │  │  ┌─────────────────────────────────────────────────────┐    │        │
    │  │  │ 1. _tick_telemetry()                                │    │        │
    │  │  │    - pathsTelemetry.tick()    → Error metrics/logs  │    │        │
    │  │  │    - featuresTelemetry.tick() → Feature metrics     │    │        │
    │  │  │    - generateTelemetry.tick() → Model metrics       │    │        │
    │  │  │    - actionTelemetry.tick()   → Action I/O logs     │    │        │
    │  │  │    - engagementTelemetry.tick() → Feedback metrics  │    │        │
    │  │  │    - Sets genkit:rootState for root spans           │    │        │
    │  │  └─────────────────────────────────────────────────────┘    │        │
    │  │  ┌─────────────────────────────────────────────────────┐    │        │
    │  │  │ 2. AdjustingTraceExporter._adjust()                 │    │        │
    │  │  │    - Redact genkit:input/output → "<redacted>"      │    │        │
    │  │  │    - Mark error spans with /http/status_code: 599   │    │        │
    │  │  │    - Mark failed spans with genkit:failedSpan       │    │        │
    │  │  │    - Mark root spans with genkit:feature            │    │        │
    │  │  │    - Mark model spans with genkit:model             │    │        │
    │  │  │    - Normalize labels (: → /) for GCP compatibility │    │        │
    │  │  └─────────────────────────────────────────────────────┘    │        │
    │  └────────────────────────┬────────────────────────────────────┘        │
    │                           │                                             │
    │           ┌───────────────┴───────────────┐                             │
    │           ▼                               ▼                             │
    │  ┌─────────────────┐             ┌─────────────────┐                    │
    │  │ GenkitGCPExporter│             │ Cloud Logging   │                    │
    │  │ (Cloud Trace)   │             │ (via structlog) │                    │
    │  └────────┬────────┘             └─────────────────┘                    │
    │           │                                                             │
    │           ▼                                                             │
    │  ┌─────────────────┐                                                    │
    │  │ Google Cloud    │                                                    │
    │  │ Trace API       │                                                    │
    │  └─────────────────┘                                                    │
    │                                                                         │
    │  ─────────────────────── METRICS PIPELINE ────────────────────────      │
    │                                                                         │
    │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
    │  │ OpenTelemetry   │───▶│ GenkitMetric    │───▶│ Cloud Monitoring│      │
    │  │ Meter           │    │ Exporter        │    │ API             │      │
    │  │ (counters,      │    │ (adjusts start  │    │                 │      │
    │  │  histograms)    │    │  times for      │    │                 │      │
    │  └─────────────────┘    │  DELTA→CUMUL.)  │    └─────────────────┘      │
    │                         └─────────────────┘                             │
    └─────────────────────────────────────────────────────────────────────────┘
    ```

Key Components:
    1. **GcpAdjustingTraceExporter**: Extends AdjustingTraceExporter to add
       GCP-specific telemetry recording before spans are adjusted and exported.

    2. **AdjustingTraceExporter** (from genkit.core.trace): Base class that
       handles PII redaction, error marking, and label normalization.

    3. **GenkitGCPExporter**: Extends CloudTraceSpanExporter with retry logic
       for reliable delivery to Google Cloud Trace.

    4. **GenkitMetricExporter**: Wraps CloudMonitoringMetricsExporter and
       adjusts start times to prevent overlap when GCP converts DELTA to
       CUMULATIVE aggregation.

    5. **Telemetry Handlers** (in separate modules):
       - feature.py: Tracks root span requests/latency
       - path.py: Tracks error paths and failure metrics
       - generate.py: Tracks model usage (tokens, latency, media)
       - action.py: Logs tool and action I/O
       - engagement.py: Tracks user feedback and acceptance

Telemetry Types and When They Fire:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Telemetry Type │ Condition                    │ What It Records         │
    ├────────────────┼──────────────────────────────┼─────────────────────────┤
    │ paths          │ Always (for all spans)       │ Error paths, failures   │
    │ features       │ genkit:isRoot = true         │ Request count, latency  │
    │ generate       │ type=action, subtype=model   │ Tokens, latency, media  │
    │ action         │ type in (action,flow,...)    │ Input/output logs       │
    │ engagement     │ type=userEngagement          │ Feedback, acceptance    │
    └────────────────┴──────────────────────────────┴─────────────────────────┘

Span Attributes Used:
    The system reads these genkit:* attributes from spans:
    - genkit:type - Span type (action, flow, flowStep, util, userEngagement)
    - genkit:metadata:subtype - Subtype (model, tool, etc.)
    - genkit:isRoot - Whether this is the root/entry span
    - genkit:name - Action/flow name
    - genkit:path - Hierarchical path like /{flow,t:flow}/{step,t:flowStep}
    - genkit:input - JSON-encoded input data
    - genkit:output - JSON-encoded output data
    - genkit:state - Span state (success, error)
    - genkit:isFailureSource - Whether this span is the source of a failure

Configuration Options (matching JS/Go parity):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Option                      │ Type     │ Default    │ Description       │
    ├─────────────────────────────┼──────────┼────────────┼───────────────────┤
    │ project_id                  │ str      │ Auto       │ GCP project ID    │
    │ credentials                 │ dict     │ ADC        │ Service account   │
    │ log_input_and_output        │ bool     │ False      │ Disable redaction │
    │ force_dev_export            │ bool     │ True       │ Export in dev     │
    │ disable_metrics             │ bool     │ False      │ Skip metrics      │
    │ disable_traces              │ bool     │ False      │ Skip traces       │
    │ metric_export_interval_ms   │ int      │ 60000      │ Export interval   │
    │ metric_export_timeout_ms    │ int      │ None       │ Export timeout    │
    │ sampler                     │ Sampler  │ AlwaysOn   │ Trace sampler     │
    └─────────────────────────────┴──────────┴────────────┴───────────────────┘

Project ID Resolution Order:
    1. Explicit project_id parameter
    2. FIREBASE_PROJECT_ID environment variable
    3. GOOGLE_CLOUD_PROJECT environment variable
    4. GCLOUD_PROJECT environment variable
    5. project_id from credentials dict

Usage:
    ```python
    from genkit.plugins.google_cloud import add_gcp_telemetry

    # Enable telemetry with default settings (PII redaction enabled)
    add_gcp_telemetry()

    # Enable telemetry with input/output logging (disable PII redaction)
    add_gcp_telemetry(log_input_and_output=True)

    # Force export even in dev environment
    add_gcp_telemetry(force_dev_export=True)

    # Disable metrics but keep traces
    add_gcp_telemetry(disable_metrics=True)

    # Custom metric export interval (minimum 5000ms for GCP)
    add_gcp_telemetry(metric_export_interval_ms=30000)
    ```

Caveats:
    - By default, model inputs and outputs are redacted for privacy
    - Set log_input_and_output=True only in trusted environments
    - In dev environment, telemetry is skipped unless force_dev_export=True
    - GCP requires minimum 5000ms metric export interval (see quotas link below)

GCP Documentation References:
    Cloud Trace:
        - Overview: https://cloud.google.com/trace/docs
        - IAM Roles: https://cloud.google.com/trace/docs/iam
        - Required role: roles/cloudtrace.agent (Cloud Trace Agent)

    Cloud Monitoring:
        - Overview: https://cloud.google.com/monitoring/docs
        - Quotas & Limits: https://cloud.google.com/monitoring/quotas
        - Required role: roles/monitoring.metricWriter (Monitoring Metric Writer)
          or roles/telemetry.metricsWriter (Cloud Telemetry Metrics Writer)

    OpenTelemetry GCP Exporters:
        - Documentation: https://google-cloud-opentelemetry.readthedocs.io/
        - Cloud Trace Exporter: https://google-cloud-opentelemetry.readthedocs.io/en/stable/cloud_trace/cloud_trace.html
        - Cloud Monitoring Exporter: https://google-cloud-opentelemetry.readthedocs.io/en/stable/cloud_monitoring/cloud_monitoring.html

Cross-Language Parity:
    This implementation maintains parity with:
    - JavaScript: js/plugins/google-cloud/src/gcpOpenTelemetry.ts
    - Go: go/plugins/googlecloud/googlecloud.go
    - Go: go/plugins/firebase/telemetry.go (FirebaseTelemetryOptions)

    Key parity points:
    - Same configuration options with equivalent semantics
    - Same telemetry dispatch logic (when each handler fires)
    - Same metrics names and dimensions
    - Same span adjustment pipeline (redaction, marking, normalization)
    - Same project ID resolution order
"""

from typing import Any

import structlog
from opentelemetry.sdk.trace.sampling import Sampler

from .config import GcpTelemetry

logger = structlog.get_logger(__name__)


def add_gcp_telemetry(
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
        add_gcp_telemetry()

        # Enable input/output logging (disable PII redaction)
        add_gcp_telemetry(log_input_and_output=True)

        # Force export in dev environment with specific project
        add_gcp_telemetry(force_dev_export=True, project_id='my-project')

        # Disable metrics but keep traces
        add_gcp_telemetry(disable_metrics=True)

        # Custom metric export interval (minimum 5000ms)
        add_gcp_telemetry(metric_export_interval_ms=30000)

        # With custom credentials for non-GCP environments
        add_gcp_telemetry(
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
