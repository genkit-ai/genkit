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

"""Cloudflare telemetry integration for Genkit.

This package provides telemetry export via OpenTelemetry Protocol (OTLP)
to any compatible observability backend, including Cloudflare's native
OTLP endpoints and third-party platforms.

Module Structure::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Module          │ Purpose                                               │
    ├─────────────────┼───────────────────────────────────────────────────────┤
    │ tracing.py      │ Main entry point, OTLP/HTTP exporter configuration    │
    └─────────────────┴───────────────────────────────────────────────────────┘

Quick Start:
    ```python
    from genkit.plugins.cloudflare_workers_ai import add_cloudflare_telemetry

    # Enable telemetry with defaults (uses CF_OTLP_ENDPOINT env var)
    add_cloudflare_telemetry()

    # Or with explicit endpoint and token
    add_cloudflare_telemetry(endpoint='https://otel.example.com/v1/traces', api_token='your-api-token')

    # Or disable PII redaction (caution!)
    add_cloudflare_telemetry(log_input_and_output=True)
    ```

Cross-Language Parity:
    This implementation maintains feature parity with:
    - AWS plugin: py/plugins/aws/
    - Google Cloud plugin: py/plugins/google-cloud/
    - Azure plugin: py/plugins/azure/

See Also:
    - tracing.py module docstring for detailed architecture documentation

Cloudflare Documentation:
    Workers Observability:
        - Overview: https://developers.cloudflare.com/workers/observability/
        - OTLP Export: https://developers.cloudflare.com/workers/observability/exporting-opentelemetry-data/

    OpenTelemetry:
        - Python SDK: https://opentelemetry.io/docs/languages/python/
"""

from .tracing import add_cloudflare_telemetry

__all__ = ['add_cloudflare_telemetry']
