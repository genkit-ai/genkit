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

"""Path telemetry for GCP.

This module tracks path-level failure metrics and logs errors,
matching the JavaScript implementation.

Metrics Recorded:
    - genkit/feature/path/requests: Counter for unique flow paths
    - genkit/feature/path/latency: Histogram for path latency (ms)

Cross-Language Parity:
    - JavaScript: js/plugins/google-cloud/src/telemetry/paths.ts
    - Go: go/plugins/googlecloud/paths.go

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
    extract_error_message,
    extract_error_name,
    extract_error_stack,
    extract_outer_feature_name_from_path,
    to_display_path,
    truncate_path,
)

logger = structlog.get_logger(__name__)

# Lazy-initialized metrics
_path_counter: metrics.Counter | None = None
_path_latency: metrics.Histogram | None = None


def _get_path_counter() -> metrics.Counter:
    """Get or create the path requests counter."""
    global _path_counter
    if _path_counter is None:
        meter = metrics.get_meter('genkit')
        _path_counter = meter.create_counter(
            'genkit/feature/path/requests',
            description='Tracks unique flow paths per flow.',
            unit='1',
        )
    return _path_counter


def _get_path_latency() -> metrics.Histogram:
    """Get or create the path latency histogram."""
    global _path_latency
    if _path_latency is None:
        meter = metrics.get_meter('genkit')
        _path_latency = meter.create_histogram(
            'genkit/feature/path/latency',
            description='Latencies per flow path.',
            unit='ms',
        )
    return _path_latency


class PathsTelemetry:
    """Telemetry handler for Genkit paths (error tracking)."""

    def tick(
        self,
        span: ReadableSpan,
        log_input_and_output: bool,
        project_id: str | None = None,
    ) -> None:
        """Record telemetry for a path span.

        Only ticks metrics for failing, leaf spans (isFailureSource).

        Args:
            span: The span to record telemetry for.
            log_input_and_output: Whether to log input/output (unused here).
            project_id: Optional GCP project ID.
        """
        attrs = span.attributes or {}

        path = str(attrs.get('genkit:path', ''))
        is_failure_source = bool(attrs.get('genkit:isFailureSource'))
        state = str(attrs.get('genkit:state', ''))

        # Only tick metrics for failing, leaf spans
        if not path or not is_failure_source or state != 'error':
            return

        session_id = str(attrs.get('genkit:sessionId', '')) or None
        thread_name = str(attrs.get('genkit:threadName', '')) or None

        events = list(span.events)
        error_name = extract_error_name(events) or '<unknown>'
        error_message = extract_error_message(events) or '<unknown>'
        error_stack = extract_error_stack(events) or ''

        # Calculate latency
        latency_ms = 0.0
        if span.end_time and span.start_time:
            latency_ms = (span.end_time - span.start_time) / 1_000_000

        path_dimensions = {
            'featureName': extract_outer_feature_name_from_path(path)[:256],
            'status': 'failure',
            'error': error_name[:256],
            'path': path[:256],
            'source': 'py',
            'sourceVersion': GENKIT_VERSION,
        }
        _get_path_counter().add(1, path_dimensions)
        _get_path_latency().record(latency_ms, path_dimensions)

        display_path = truncate_path(to_display_path(path))
        log_attrs = {
            **create_common_log_attributes(span, project_id),
            'path': display_path,
            'qualifiedPath': path,
            'name': error_name,
            'message': error_message,
            'stack': error_stack,
            'source': 'py',
            'sourceVersion': GENKIT_VERSION,
        }
        if session_id:
            log_attrs['sessionId'] = session_id
        if thread_name:
            log_attrs['threadName'] = thread_name

        gcp_logger.log_structured_error(f'Error[{display_path}, {error_name}]', log_attrs)


# Singleton instance
paths_telemetry = PathsTelemetry()
