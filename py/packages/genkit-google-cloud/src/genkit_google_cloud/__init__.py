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
enabling telemetry export to Cloud Trace, Cloud Monitoring, and Cloud Logging.

Example:
    ```python
    from genkit import Genkit
    from genkit_googleai import GoogleAI
    from genkit_google_cloud import enable_google_cloud_telemetry


    # 1. Enable Google Cloud Trace and Monitoring export
    enable_google_cloud_telemetry(project_id='my-project')

    # 2. All subsequent Genkit actions automatically export telemetry
    ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')
    await ai.generate(prompt='Hello, world!')
    # => Traces exported asynchronously to Cloud Trace (latency, tokens, status)
    ```

Requirements:
    - Requires Google Cloud Application Default Credentials (ADC) or explicit credentials.
    - Telemetry export is disabled by default in local dev environments unless explicitly configured.

See Also:
    - Cloud Trace: https://cloud.google.com/trace
    - Cloud Monitoring: https://cloud.google.com/monitoring
"""

from .telemetry import add_gcp_telemetry, enable_google_cloud_telemetry


def package_name() -> str:
    """Get the package name for the Google Cloud plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit_google_cloud'


__all__ = [
    'add_gcp_telemetry',
    'enable_google_cloud_telemetry',
    'package_name',
]
