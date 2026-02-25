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

"""Feature telemetry for GCP.

This module tracks feature-level metrics (requests, latencies) and logs
input/output for root spans, matching the JavaScript implementation.

Metrics Recorded:
    - genkit/feature/requests: Counter for root span calls
    - genkit/feature/latency: Histogram for root span latency (ms)

Cross-Language Parity:
    - JavaScript: js/plugins/google-cloud/src/telemetry/feature.ts
    - Go: go/plugins/googlecloud/feature.go

See Also:
    - Cloud Monitoring Custom Metrics: https://cloud.google.com/monitoring/custom-metrics
"""

from __future__ import annotations

import structlog
from opentelemetry import metrics
from opentelemetry.sdk.trace import ReadableSpan

from genkit.core import GENKIT_VERSION

from .gcp_logger import gcp_logger
from .utils import (
    create_common_log_attributes,
    extract_error_name,
    to_display_path,
    truncate,
    truncate_path,
)

logger = structlog.get_logger(__name__)

# Lazy-initialized metrics
_feature_counter: metrics.Counter | None = None
_feature_latency: metrics.Histogram | None = None


def _get_feature_counter() -> metrics.Counter:
    """Get or create the feature requests counter."""
    global _feature_counter
    if _feature_counter is None:
        meter = metrics.get_meter('genkit')
        _feature_counter = meter.create_counter(
            'genkit/feature/requests',
            description='Counts calls to genkit features.',
            unit='1',
        )
    return _feature_counter


def _get_feature_latency() -> metrics.Histogram:
    """Get or create the feature latency histogram."""
    global _feature_latency
    if _feature_latency is None:
        meter = metrics.get_meter('genkit')
        _feature_latency = meter.create_histogram(
            'genkit/feature/latency',
            description='Latencies when calling Genkit features.',
            unit='ms',
        )
    return _feature_latency


class FeaturesTelemetry:
    """Telemetry handler for Genkit features (root spans)."""

    def tick(
        self,
        span: ReadableSpan,
        log_input_and_output: bool,
        project_id: str | None = None,
    ) -> None:
        """Record telemetry for a feature span.

        Args:
            span: The span to record telemetry for.
            log_input_and_output: Whether to log input/output.
            project_id: Optional GCP project ID.
        """
        attrs = span.attributes or {}
        name = str(attrs.get('genkit:name', '<unknown>'))
        path = str(attrs.get('genkit:path', ''))
        state = str(attrs.get('genkit:state', ''))

        # Calculate latency
        latency_ms = 0.0
        if span.end_time and span.start_time:
            latency_ms = (span.end_time - span.start_time) / 1_000_000

        if state == 'success':
            self._write_feature_success(name, latency_ms)
        elif state == 'error':
            error_name = extract_error_name(list(span.events)) or '<unknown>'
            self._write_feature_failure(name, latency_ms, error_name)
        else:
            logger.warning('Unknown state', state=state)
            return

        if log_input_and_output:
            input_val = truncate(str(attrs.get('genkit:input', '')))
            output_val = truncate(str(attrs.get('genkit:output', '')))
            session_id = str(attrs.get('genkit:sessionId', '')) or None
            thread_name = str(attrs.get('genkit:threadName', '')) or None

            if input_val:
                self._write_log(span, 'Input', name, path, input_val, project_id, session_id, thread_name)
            if output_val:
                self._write_log(span, 'Output', name, path, output_val, project_id, session_id, thread_name)

    def _write_feature_success(self, feature_name: str, latency_ms: float) -> None:
        """Record success metrics for a feature."""
        dimensions = {
            'name': feature_name[:256],
            'status': 'success',
            'source': 'py',
            'sourceVersion': GENKIT_VERSION,
        }
        _get_feature_counter().add(1, dimensions)
        _get_feature_latency().record(latency_ms, dimensions)

    def _write_feature_failure(
        self,
        feature_name: str,
        latency_ms: float,
        error_name: str,
    ) -> None:
        """Record failure metrics for a feature."""
        dimensions = {
            'name': feature_name[:256],
            'status': 'failure',
            'source': 'py',
            'sourceVersion': GENKIT_VERSION,
            'error': error_name[:256],
        }
        _get_feature_counter().add(1, dimensions)
        _get_feature_latency().record(latency_ms, dimensions)

    def _write_log(
        self,
        span: ReadableSpan,
        tag: str,
        feature_name: str,
        qualified_path: str,
        content: str,
        project_id: str | None,
        session_id: str | None,
        thread_name: str | None,
    ) -> None:
        """Write a structured log entry to Cloud Logging."""
        path = truncate_path(to_display_path(qualified_path))
        metadata = {
            **create_common_log_attributes(span, project_id),
            'path': path,
            'qualifiedPath': qualified_path,
            'featureName': feature_name,
            'content': content,
        }
        if session_id:
            metadata['sessionId'] = session_id
        if thread_name:
            metadata['threadName'] = thread_name

        gcp_logger.log_structured(f'{tag}[{path}, {feature_name}]', metadata)


# Singleton instance
features_telemetry = FeaturesTelemetry()
