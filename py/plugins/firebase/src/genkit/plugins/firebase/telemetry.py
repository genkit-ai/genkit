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

"""Firebase telemetry integration."""

from genkit.core.logging import get_logger
from genkit.plugins.google_cloud.telemetry.config import GcpTelemetry

from .constant import FirebaseTelemetryConfig

logger = get_logger(__name__)


def add_firebase_telemetry(config: FirebaseTelemetryConfig) -> None:
    """Add Firebase telemetry export to Google Cloud Observability.

    Exports traces to Cloud Trace, metrics to Cloud Monitoring, and logs to
    Cloud Logging. In development (GENKIT_ENV=dev), telemetry is disabled by
    default unless force_dev_export is set to True.

    Args:
        config: FirebaseTelemetryConfig object with telemetry configuration.
    """
    manager = GcpTelemetry(
        project_id=config.project_id,
        credentials=config.credentials,
        sampler=config.sampler,
        log_input_and_output=config.log_input_and_output,
        force_dev_export=config.force_dev_export,
        disable_metrics=config.disable_metrics,
        disable_traces=config.disable_traces,
        metric_export_interval_ms=config.metric_export_interval_ms,
        metric_export_timeout_ms=config.metric_export_timeout_ms,
    )

    if not manager.project_id:
        logger.warning(
            'Firebase project ID not found. Set FIREBASE_PROJECT_ID, GOOGLE_CLOUD_PROJECT, '
            'or GCLOUD_PROJECT environment variable, or pass project_id parameter.'
        )

    manager.initialize()
