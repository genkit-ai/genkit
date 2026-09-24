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

from genkit._core._telemetry._adjusting_exporter import AdjustingTraceExporter, RedactedSpan

from .constants import (
    MIN_SPAN_DURATION_NS,
    TRACE_RETRY_DEADLINE,
    TRACE_RETRY_INITIAL,
    TRACE_RETRY_MAXIMUM,
    TRACE_RETRY_MULTIPLIER,
)

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
        """Span end time, adjusted to meet GCP requirements.

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
    """Cloud Trace exporter with PII redaction and a non-zero duration."""

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
        span = super()._adjust(span)
        return TimeAdjustedSpan(span, dict(span.attributes) if span.attributes else {})

