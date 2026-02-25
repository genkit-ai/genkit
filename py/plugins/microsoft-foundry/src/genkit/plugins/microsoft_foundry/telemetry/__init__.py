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

"""Azure telemetry integration for Microsoft Foundry plugin.

This package provides telemetry export to Azure's observability suite,
enabling monitoring and debugging of Genkit applications through Azure
Application Insights for distributed tracing.

Module Structure::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Module          │ Purpose                                               │
    ├─────────────────┼───────────────────────────────────────────────────────┤
    │ tracing.py      │ Main entry point, Azure Monitor OTLP exporter         │
    └─────────────────┴───────────────────────────────────────────────────────┘

Quick Start:
    ```python
    from genkit.plugins.microsoft_foundry import add_azure_telemetry

    # Enable telemetry with defaults (uses APPLICATIONINSIGHTS_CONNECTION_STRING)
    add_azure_telemetry()

    # Or with explicit connection string
    add_azure_telemetry(connection_string='InstrumentationKey=...')

    # Or disable PII redaction (caution!)
    add_azure_telemetry(log_input_and_output=True)
    ```

Cross-Language Parity:
    This implementation maintains feature parity with:
    - AWS plugin: py/plugins/aws/
    - Google Cloud plugin: py/plugins/google-cloud/

See Also:
    - tracing.py module docstring for detailed architecture documentation

Azure Documentation:
    Application Insights:
        - Overview: https://docs.microsoft.com/azure/azure-monitor/app/app-insights-overview
        - Connection String: https://docs.microsoft.com/azure/azure-monitor/app/sdk-connection-string

    OpenTelemetry Azure:
        - Azure Monitor Exporter: https://docs.microsoft.com/azure/azure-monitor/app/opentelemetry-enable
        - Python SDK: https://pypi.org/project/azure-monitor-opentelemetry-exporter/
"""

from .tracing import add_azure_telemetry

__all__ = ['add_azure_telemetry']
