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

"""Firebase constants and configuration models."""

from collections.abc import Callable
from typing import Annotated, Any

from google.cloud.firestore_v1 import DocumentSnapshot
from opentelemetry.sdk.trace.sampling import Sampler
from pydantic import BaseModel, Field

MetadataTransformFn = Callable[[DocumentSnapshot], dict[str, Any]]


class FirebaseTelemetryConfig(BaseModel):
    """Configuration for Firebase telemetry export to Google Cloud Observability.

    Args:
        project_id: Firebase project ID. Auto-detected from environment if None.
        credentials: Service account credentials dictionary.
        sampler: OpenTelemetry trace sampler.
        log_input_and_output: If True, logs feature inputs/outputs. WARNING: May log PII.
        force_dev_export: If True, exports telemetry in dev mode (GENKIT_ENV=dev).
        disable_metrics: If True, disables metrics export.
        disable_traces: If True, disables trace export.
        metric_export_interval_ms: Metrics export interval in ms. Minimum: 1000ms.
        metric_export_timeout_ms: Metrics export timeout in ms.
    """

    project_id: str | None = None
    credentials: dict[str, Any] | None = None
    sampler: Sampler | None = None
    log_input_and_output: bool = False
    force_dev_export: bool = False
    disable_metrics: bool = False
    disable_traces: bool = False
    metric_export_interval_ms: Annotated[int, Field(ge=1000)] | None = None
    metric_export_timeout_ms: int | None = None
    model_config = {'arbitrary_types_allowed': True}
