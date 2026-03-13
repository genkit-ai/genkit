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

"""AWS telemetry integration for Genkit.

This package provides telemetry export to AWS's observability suite,
enabling monitoring and debugging of Genkit applications through AWS X-Ray
for distributed tracing and CloudWatch for metrics and logs.

Module Structure::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Module          │ Purpose                                               │
    ├─────────────────┼───────────────────────────────────────────────────────┤
    │ tracing.py      │ Main entry point, X-Ray OTLP exporter, SigV4 auth     │
    └─────────────────┴───────────────────────────────────────────────────────┘

Quick Start:
    ```python
    from genkit.plugins.amazon_bedrock import add_aws_telemetry

    # Enable telemetry with defaults (uses AWS_REGION env var)
    add_aws_telemetry()

    # Or with explicit region
    add_aws_telemetry(region='us-west-2')

    # Or disable PII redaction (caution!)
    add_aws_telemetry(log_input_and_output=True)
    ```

Cross-Language Parity:
    This implementation maintains feature parity with:
    - Google Cloud plugin: py/plugins/google-cloud/

See Also:
    - tracing.py module docstring for detailed architecture documentation

AWS Documentation:
    X-Ray:
        - Overview: https://docs.aws.amazon.com/xray/
        - OTLP Endpoint: https://docs.aws.amazon.com/xray/latest/devguide/xray-api-sendingdata.html
        - IAM Roles: AWSXrayWriteOnlyPolicy

    CloudWatch:
        - Overview: https://docs.aws.amazon.com/cloudwatch/
        - OTLP Endpoint: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch-OTLPEndpoint.html

    OpenTelemetry AWS:
        - ADOT Python: https://aws-otel.github.io/docs/getting-started/python-sdk
        - SDK Extension: https://pypi.org/project/opentelemetry-sdk-extension-aws/
"""

from .tracing import add_aws_telemetry

__all__ = ['add_aws_telemetry']
