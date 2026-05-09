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


"""Google Cloud Plugin for Genkit.

This plugin provides Google Cloud observability integration for Genkit,
enabling telemetry export to Cloud Trace and Cloud Monitoring.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Telemetry           │ Data about how your app is running. Like a        │
    │                     │ fitness tracker for your code.                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Cloud Trace         │ Shows the path requests take through your app.    │
    │                     │ Like GPS tracking for your API calls.             │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Cloud Monitoring    │ Graphs and alerts for your app's health.          │
    │                     │ Like a heart rate monitor dashboard.              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Span                │ One step in a request's journey. Like one         │
    │                     │ leg of a relay race.                              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Trace               │ All spans for one request connected together.     │
    │                     │ The complete story of one API call.               │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Metrics             │ Numbers that describe your app (requests/sec,     │
    │                     │ error rate, latency). Like a report card.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ PII Redaction       │ Hiding sensitive data in traces. Like blurring    │
    │                     │ faces in photos before sharing.                   │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    HOW TELEMETRY FLOWS TO GOOGLE CLOUD                  │
    │                                                                         │
    │    Your Genkit App                                                      │
    │         │                                                               │
    │         │  (1) App runs flows, calls models, uses tools                 │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  OpenTelemetry  │   Automatically creates spans for each           │
    │    │  SDK            │   operation (you don't write this code!)         │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Spans collected and processed                        │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  GCP Exporters  │   • Redact PII (input/output)                    │
    │    │                 │   • Add error markers                            │
    │    │                 │   • Batch for efficiency                         │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) HTTPS to Google Cloud                                │
    │             ▼                                                           │
    │    ════════════════════════════════════════════════════                 │
    │             │  Internet                                                 │
    │             ▼                                                           │
    │    ┌─────────────────────────────────────────────────────┐              │
    │    │  Google Cloud Console                               │              │
    │    │  ┌──────────────┐  ┌──────────────┐                │              │
    │    │  │ Cloud Trace  │  │ Cloud        │                │              │
    │    │  │ (waterfall   │  │ Monitoring   │                │              │
    │    │  │  diagrams)   │  │ (dashboards) │                │              │
    │    │  └──────────────┘  └──────────────┘                │              │
    │    └─────────────────────────────────────────────────────┘              │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      Google Cloud Plugin                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Plugin Entry Point (__init__.py)                                       │
    │  └── add_gcp_telemetry() - Enable Cloud Trace/Monitoring export         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  telemetry/__init__.py - Telemetry Module                               │
    │  └── Re-exports from submodules                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  telemetry/tracing.py - Distributed Tracing                             │
    │  ├── Cloud Trace exporter configuration                                 │
    │  └── OpenTelemetry integration                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  telemetry/metrics.py - Metrics Collection                              │
    │  ├── Cloud Monitoring exporter                                          │
    │  └── Custom Genkit metrics                                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  telemetry/action.py - Action Instrumentation                           │
    │  └── Automatic span creation for Genkit actions                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     Telemetry Data Flow                                 │
    │                                                                         │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐      │
    │  │ Genkit App   │───►│ OpenTelemetry│───►│ Google Cloud         │      │
    │  │ (actions,    │    │ SDK          │    │ (Trace, Monitoring)  │      │
    │  │  flows)      │    └──────────────┘    └──────────────────────┘      │
    │  └──────────────┘                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

Example:
    ```python
    from genkit.plugins.google_cloud import add_gcp_telemetry

    # Enable telemetry export to Google Cloud
    add_gcp_telemetry()

    # Traces and metrics are now exported to:
    # - Cloud Trace (distributed tracing)
    # - Cloud Monitoring (metrics)
    ```

Caveats:
    - Requires Google Cloud credentials (ADC or explicit)
    - Telemetry is disabled by default in development mode (GENKIT_ENV=dev)
    - Requires opentelemetry and google-cloud-* packages

See Also:
    - Cloud Trace: https://cloud.google.com/trace
    - Cloud Monitoring: https://cloud.google.com/monitoring
    - Genkit documentation: https://genkit.dev/
"""

from .telemetry import add_gcp_telemetry


def package_name() -> str:
    """Get the package name for the Google Cloud plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit.plugins.google_cloud'


__all__ = ['add_gcp_telemetry', 'package_name']
