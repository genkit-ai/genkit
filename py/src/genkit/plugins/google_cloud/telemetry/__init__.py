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

"""Google Cloud telemetry integration for Genkit.

This package provides telemetry export to Google Cloud's observability suite,
enabling monitoring and debugging of Genkit applications through Cloud Trace,
Cloud Monitoring, and Cloud Logging.

Module Structure:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Module          │ Purpose                                               │
    ├─────────────────┼───────────────────────────────────────────────────────┤
    │ tracing.py      │ Main entry point, exporters, configuration            │
    │ feature.py      │ Root span metrics (requests, latency)                 │
    │ path.py         │ Error path tracking and failure metrics               │
    │ generate.py     │ Model/generate metrics (tokens, latency, media)       │
    │ action.py       │ Action I/O logging (tools, flows)                     │
    │ engagement.py   │ User feedback and acceptance metrics                  │
    │ metrics.py      │ Metric definitions and lazy initialization            │
    │ utils.py        │ Shared utilities (truncation, path parsing, logging)  │
    └─────────────────┴───────────────────────────────────────────────────────┘

Quick Start:
    ```python
    from genkit.plugins.google_cloud import add_gcp_telemetry

    # Enable telemetry with defaults (PII redaction enabled)
    add_gcp_telemetry()

    # Or with custom options
    add_gcp_telemetry(
        project_id='my-project',
        log_input_and_output=True,  # Disable PII redaction (caution!)
    )
    ```

Cross-Language Parity:
    This implementation maintains feature parity with:
    - JavaScript: js/plugins/google-cloud/src/gcpOpenTelemetry.ts
    - Go: go/plugins/googlecloud/ and go/plugins/firebase/telemetry.go

See Also:
    - tracing.py module docstring for detailed architecture documentation

GCP Documentation:
    Cloud Trace:
        - Overview: https://cloud.google.com/trace/docs
        - IAM Roles: https://cloud.google.com/trace/docs/iam

    Cloud Monitoring:
        - Overview: https://cloud.google.com/monitoring/docs
        - Quotas & Limits: https://cloud.google.com/monitoring/quotas

    OpenTelemetry GCP:
        - Python Exporters: https://google-cloud-opentelemetry.readthedocs.io/
"""

from .tracing import add_gcp_telemetry

__all__ = ['add_gcp_telemetry']
