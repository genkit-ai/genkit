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

"""Trace exporting functionality for GCP telemetry.

This module contains all trace-specific exporters and span wrappers
for Google Cloud Trace integration.
"""

from collections.abc import Callable, Sequence

import structlog
from google.api_core import exceptions as core_exceptions, retry as retries
from google.cloud.trace_v2 import BatchWriteSpansRequest
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from genkit.core.trace.adjusting_exporter import AdjustingTraceExporter, RedactedSpan

from .action import action_telemetry
from .constants import (
    MIN_SPAN_DURATION_NS,
    TRACE_RETRY_DEADLINE,
    TRACE_RETRY_INITIAL,
    TRACE_RETRY_MAXIMUM,
    TRACE_RETRY_MULTIPLIER,
)
from .engagement import engagement_telemetry
from .feature import features_telemetry
from .generate import generate_telemetry
from .path import paths_telemetry

logger = structlog.get_logger(__name__)


class GenkitGCPExporter(CloudTraceSpanExporter):
    """Exports spans to Google Cloud Trace with retry logic.

    This exporter extends the base CloudTraceSpanExporter to add
    robust retry handling for transient failures.

    Note:
        The parent class uses google.auth.default() to get the project ID.
    """

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export the spans to Cloud Trace with retry logic.

        Iterates through the provided spans and exports them to GCP.

        Note:
            Leverages span transformation and formatting from opentelemetry-exporter-gcp-trace.
            See: https://cloud.google.com/python/docs/reference/cloudtrace/latest

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects to export.

        Returns:
            SpanExportResult.SUCCESS upon successful processing (does not guarantee
            server-side success), or SpanExportResult.FAILURE if an error occurs.
        """
        try:
            self.client.batch_write_spans(
                request=BatchWriteSpansRequest(
                    name=f'projects/{self.project_id}',
                    spans=self._translate_to_cloud_trace(spans),
                ),
                retry=retries.Retry(
                    initial=TRACE_RETRY_INITIAL,
                    maximum=TRACE_RETRY_MAXIMUM,
                    multiplier=TRACE_RETRY_MULTIPLIER,
                    predicate=retries.if_exception_type(
                        core_exceptions.DeadlineExceeded,
                    ),
                    deadline=TRACE_RETRY_DEADLINE,
                ),
            )
        except Exception as ex:
            logger.error('Error while writing to Cloud Trace', exc_info=ex)
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS


class TimeAdjustedSpan(RedactedSpan):
    """Wraps a span to ensure non-zero duration for GCP requirements.

    Google Cloud Trace requires end_time > start_time. This wrapper
    ensures that all spans meet this requirement by adding a minimum
    duration if needed.
    """

    @property
    def end_time(self) -> int | None:
        """Return the span end time, adjusted to meet GCP requirements.

        Returns:
            The span end time, guaranteed to be > start_time if start_time exists.
        """
        start = self._span.start_time
        end = self._span.end_time

        # GCP requires end_time > start_time.
        # If the span is unfinished (end_time is None) or has zero duration,
        # we provide a minimum duration.
        if start is not None:
            if end is None or end <= start:
                return start + MIN_SPAN_DURATION_NS

        return end


class GcpAdjustingTraceExporter(AdjustingTraceExporter):
    """GCP-specific span exporter that adds telemetry recording.

    This extends the base AdjustingTraceExporter to add GCP-specific telemetry
    recording (metrics and logs) for each span, matching the JavaScript
    implementation in gcpOpenTelemetry.ts.

    The telemetry handlers record:
    - Feature metrics (requests, latency) for root spans
    - Path metrics for failure tracking
    - Generate metrics (tokens, latency) for model actions
    - Action logs for tools and generate
    - Engagement metrics for user feedback

    Example:
        ```python
        exporter = GcpAdjustingTraceExporter(
            exporter=GenkitGCPExporter(),
            log_input_and_output=False,
            project_id='my-project',
        )
        ```
    """

    def __init__(
        self,
        exporter: SpanExporter,
        log_input_and_output: bool = False,
        project_id: str | None = None,
        error_handler: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize the GCP adjusting trace exporter.

        Args:
            exporter: The underlying SpanExporter to wrap.
            log_input_and_output: If True, preserve input/output in spans and logs.
                Defaults to False (redact for privacy).
            project_id: Optional GCP project ID for log correlation.
            error_handler: Optional callback invoked when export errors occur.
        """
        super().__init__(
            exporter=exporter,
            log_input_and_output=log_input_and_output,
            project_id=project_id,
            error_handler=error_handler,
        )

    def _adjust(self, span: ReadableSpan) -> ReadableSpan:
        """Apply all adjustments to a span including telemetry.

        This overrides the base method to add telemetry recording before
        the standard adjustments (redaction, marking, normalization).

        Args:
            span: The span to adjust.

        Returns:
            The adjusted span with telemetry recorded and time adjusted.
        """
        # Record telemetry before adjustments (uses original attributes)
        span = self._tick_telemetry(span)

        # Apply standard adjustments from base class
        span = super()._adjust(span)

        # Fix start/end times for GCP (must be end > start)
        return TimeAdjustedSpan(span, dict(span.attributes) if span.attributes else {})

    def _tick_telemetry(self, span: ReadableSpan) -> ReadableSpan:
        """Record telemetry for a span and apply root state marking.

        This matches the JavaScript tickTelemetry method in gcpOpenTelemetry.ts.
        It calls the appropriate telemetry handlers based on span type.

        Args:
            span: The span to record telemetry for.

        Returns:
            The span, potentially with genkit:rootState added for root spans.
        """
        attrs = span.attributes or {}
        if 'genkit:type' not in attrs:
            return span

        span_type = attrs.get('genkit:type', '')
        subtype = attrs.get('genkit:metadata:subtype', '')
        is_root = bool(attrs.get('genkit:isRoot'))

        try:
            # Always record path telemetry for error tracking
            paths_telemetry.tick(span, self._log_input_and_output, self._project_id)

            if is_root:
                # Report top level feature request and latency only for root spans
                features_telemetry.tick(span, self._log_input_and_output, self._project_id)

                # Set root state explicitly
                # (matches JS: span.attributes['genkit:rootState'] = span.attributes['genkit:state'])
                state = attrs.get('genkit:state')
                if state:
                    new_attrs = dict(attrs)
                    new_attrs['genkit:rootState'] = state
                    span = RedactedSpan(span, new_attrs)
            else:
                if span_type == 'action' and subtype == 'model':
                    # Report generate metrics for all model actions
                    generate_telemetry.tick(span, self._log_input_and_output, self._project_id)

                if span_type == 'action' and subtype == 'tool':
                    # TODO(#4359): Report input and output for tool actions (matching JS comment)
                    pass

                if span_type in ('action', 'flow', 'flowStep', 'util'):
                    # Report request and latency metrics for all actions
                    action_telemetry.tick(span, self._log_input_and_output, self._project_id)

            if span_type == 'userEngagement':
                # Report user acceptance and feedback metrics
                engagement_telemetry.tick(span, self._log_input_and_output, self._project_id)

        except Exception as e:
            logger.warning('Error recording telemetry', error=str(e))

        return span
