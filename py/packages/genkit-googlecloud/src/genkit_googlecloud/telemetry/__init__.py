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

Example:
    ```python
    from genkit import Genkit
    from genkit_googleai import GoogleAI
    from genkit_googlecloud import enable_googlecloud_telemetry

    enable_googlecloud_telemetry(project_id='my-project')

    ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')
    response = await ai.generate(prompt='Hello, world!')
    ```

See Also:
    - Cloud Trace: https://cloud.google.com/trace/docs
    - Cloud Monitoring: https://cloud.google.com/monitoring/docs
"""

from .tracing import enable_googlecloud_telemetry

__all__ = ['enable_googlecloud_telemetry']
