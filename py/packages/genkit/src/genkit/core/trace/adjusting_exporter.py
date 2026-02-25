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

"""Adjusting trace exporter for PII redaction and span enhancement.

This module provides an exporter wrapper that adjusts spans before exporting,
primarily for redacting sensitive input/output data (PII protection) and
augmenting span attributes for cloud observability platforms like Google Cloud Trace.

Overview:
    When exporting traces to cloud services like Google Cloud Trace, you often
    want to redact model inputs and outputs to protect potentially sensitive
    user data (PII). The AdjustingTraceExporter wraps any SpanExporter and
    modifies spans before they're exported.

Key Features:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Feature                 │ Description                                   │
    ├─────────────────────────┼───────────────────────────────────────────────┤
    │ PII Redaction           │ Replaces genkit:input/output with <redacted>  │
    │ Error Marking           │ Adds /http/status_code:599 for GCP red marker │
    │ Label Normalization     │ Replaces : with / in attribute keys for GCP   │
    │ Failed Span Marking     │ Marks failure source with genkit:failedSpan   │
    │ Feature Marking         │ Marks root spans with genkit:feature          │
    │ Model Marking           │ Marks model spans with genkit:model           │
    │ Configurable Logging    │ Optional: keep input/output for debugging     │
    │ Error Callbacks         │ Custom handling for export errors             │
    └─────────────────────────┴───────────────────────────────────────────────┘

Usage:
    ```python
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    from genkit.core.trace import AdjustingTraceExporter

    # Wrap the cloud exporter with redaction
    base_exporter = CloudTraceSpanExporter()
    exporter = AdjustingTraceExporter(
        exporter=base_exporter,
        log_input_and_output=False,  # Redact by default
    )

    # Use with a span processor
    processor = BatchSpanProcessor(exporter)
    ```

Caveats:
    - Redaction is shallow - only top-level genkit:input/output are affected
    - Setting log_input_and_output=True disables redaction (use with caution)
    - The wrapped exporter must implement the standard SpanExporter interface

See Also:
    - JavaScript AdjustingTraceExporter: js/plugins/google-cloud/src/gcpOpenTelemetry.ts
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, cast

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationInfo, InstrumentationScope
from opentelemetry.trace import Link, SpanContext, SpanKind, Status, StatusCode
from opentelemetry.util.types import Attributes

from genkit.core._compat import override


class RedactedSpan(ReadableSpan):
    """A span wrapper that redacts sensitive attributes.

    This class wraps a ReadableSpan and overrides the attributes property
    to return redacted values for sensitive fields like genkit:input and
    genkit:output.
    """

    def __init__(self, span: ReadableSpan, redacted_attributes: dict[str, Any]) -> None:
        """Initialize a RedactedSpan.

        Args:
            span: The original span to wrap.
            redacted_attributes: The attributes with redacted values.
        """
        self._span = span
        self._redacted_attributes = redacted_attributes

    @property
    @override
    def name(self) -> str:
        """Return the span name."""
        return self._span.name

    @property
    @override
    def context(self) -> SpanContext:
        """Return the span context."""
        return cast(SpanContext, self._span.context)

    @override
    def get_span_context(self) -> SpanContext:
        """Return the span context."""
        return cast(SpanContext, self._span.get_span_context())

    @property
    @override
    def parent(self) -> SpanContext | None:
        """Return the parent span context."""
        return self._span.parent

    @property
    @override
    def start_time(self) -> int | None:
        """Return the span start time."""
        return self._span.start_time

    @property
    @override
    def end_time(self) -> int | None:
        """Return the span end time."""
        return self._span.end_time

    @property
    @override
    def status(self) -> Status:
        """Return the span status."""
        return self._span.status

    @property
    @override
    def attributes(self) -> Attributes:
        """Return the redacted attributes."""
        return self._redacted_attributes

    @property
    @override
    def events(self) -> Sequence[Event]:
        """Return the span events."""
        return self._span.events

    @property
    @override
    def links(self) -> Sequence[Link]:
        """Return the span links."""
        return self._span.links

    @property
    @override
    def kind(self) -> SpanKind:
        """Return the span kind."""
        return self._span.kind

    @property
    @override
    def resource(self) -> Resource:
        """Return the span resource."""
        return self._span.resource

    @property
    @override
    def instrumentation_info(self) -> InstrumentationInfo | None:
        """Return the instrumentation info."""
        # pyrefly: ignore[deprecated] - Required override for ReadableSpan interface compatibility
        return self._span.instrumentation_info

    @property
    @override
    def instrumentation_scope(self) -> InstrumentationScope | None:
        """Return the instrumentation scope."""
        return self._span.instrumentation_scope

    @property
    @override
    def dropped_attributes(self) -> int:
        """Return the number of dropped attributes from the wrapped span.

        The base ReadableSpan implementation accesses ``self._attributes``
        which is set by ``ReadableSpan.__init__``.  Since RedactedSpan
        intentionally skips ``super().__init__()`` (to avoid duplicating
        span state), the private ``_attributes`` field does not exist.
        This override delegates to the wrapped span instead.
        """
        return self._span.dropped_attributes

    @property
    @override
    def dropped_events(self) -> int:
        """Return the number of dropped events from the wrapped span.

        Delegates to the wrapped span for the same reason as
        :pyattr:`dropped_attributes`.
        """
        return self._span.dropped_events

    @property
    @override
    def dropped_links(self) -> int:
        """Return the number of dropped links from the wrapped span.

        Delegates to the wrapped span for the same reason as
        :pyattr:`dropped_attributes`.
        """
        return self._span.dropped_links


class AdjustingTraceExporter(SpanExporter):
    """Adjusts spans before exporting for PII redaction and enhancement.

    This exporter wraps another SpanExporter and modifies spans before they
    are exported. The primary use case is redacting model input/output to
    protect user privacy when exporting to cloud observability platforms.

    Attributes:
        exporter: The wrapped SpanExporter.
        log_input_and_output: If True, don't redact input/output.
        project_id: Optional project ID for cloud-specific features.
        error_handler: Optional callback for export errors.

    Example:
        ```python
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from genkit.core.trace import AdjustingTraceExporter

        exporter = AdjustingTraceExporter(
            exporter=CloudTraceSpanExporter(),
            log_input_and_output=False,
        )
        ```
    """

    REDACTED_VALUE: ClassVar[str] = '<redacted>'

    def __init__(
        self,
        exporter: SpanExporter,
        log_input_and_output: bool = False,
        project_id: str | None = None,
        error_handler: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize the AdjustingTraceExporter.

        Args:
            exporter: The underlying SpanExporter to wrap.
            log_input_and_output: If True, preserve input/output in spans.
                Defaults to False (redact for privacy).
            project_id: Optional project ID for cloud-specific features.
            error_handler: Optional callback invoked when export errors occur.
        """
        self._exporter: SpanExporter = exporter
        self._log_input_and_output: bool = log_input_and_output
        self._project_id: str | None = project_id
        self._error_handler: Callable[[Exception], None] | None = error_handler

    @override
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans after adjusting them.

        Applies transformations to each span (redaction, marking, etc.)
        before passing them to the underlying exporter.

        Args:
            spans: The spans to export.

        Returns:
            The result from the underlying exporter.
        """
        adjusted_spans = [self._adjust(span) for span in spans]

        try:
            result = self._exporter.export(adjusted_spans)
            return result
        except Exception as e:
            if self._error_handler:
                self._error_handler(e)
            raise

    @override
    def shutdown(self) -> None:
        """Shut down the underlying exporter."""
        self._exporter.shutdown()

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force the underlying exporter to flush.

        Args:
            timeout_millis: Maximum time to wait for flush.

        Returns:
            True if flush succeeded.
        """
        if hasattr(self._exporter, 'force_flush'):
            return self._exporter.force_flush(timeout_millis)
        return True

    def _adjust(self, span: ReadableSpan) -> ReadableSpan:
        """Apply all adjustments to a span.

        This method applies the same transformations as the JavaScript
        implementation in gcpOpenTelemetry.ts:
        1. Redact input/output (if not logging)
        2. Mark error spans with HTTP status code for GCP
        3. Mark failed spans with failure source info
        4. Mark root spans with genkit:feature
        5. Mark model spans with genkit:model
        6. Normalize attribute labels (: -> /)

        Args:
            span: The span to adjust.

        Returns:
            The adjusted span (possibly a RedactedSpan wrapper).
        """
        span = self._redact_input_output(span)
        span = self._mark_error_span_as_error(span)
        span = self._mark_failed_span(span)
        span = self._mark_genkit_feature(span)
        span = self._mark_genkit_model(span)
        span = self._normalize_labels(span)
        return span

    def _redact_input_output(self, span: ReadableSpan) -> ReadableSpan:
        """Redact genkit:input and genkit:output attributes.

        If log_input_and_output is True, the span is returned unchanged.
        Otherwise, these sensitive fields are replaced with '<redacted>'.

        Args:
            span: The span to potentially redact.

        Returns:
            The original span or a RedactedSpan wrapper.
        """
        if self._log_input_and_output:
            return span

        attributes = dict(span.attributes) if span.attributes else {}
        has_input = 'genkit:input' in attributes
        has_output = 'genkit:output' in attributes

        if not has_input and not has_output:
            return span

        # Create redacted attributes
        redacted = {**attributes}
        if has_input:
            redacted['genkit:input'] = self.REDACTED_VALUE
        if has_output:
            redacted['genkit:output'] = self.REDACTED_VALUE

        return RedactedSpan(span, redacted)

    def _mark_error_span_as_error(self, span: ReadableSpan) -> ReadableSpan:
        """Mark error spans with HTTP status code for GCP Trace display.

        This is a workaround for GCP Trace to mark a span with a red
        exclamation mark indicating that it is an error. GCP requires
        an HTTP status code to show the error indicator.

        Args:
            span: The span to potentially mark.

        Returns:
            The span with /http/status_code: 599 if it's an error span.
        """
        if not span.status or span.status.status_code != StatusCode.ERROR:
            return span

        attributes = dict(span.attributes) if span.attributes else {}
        attributes['/http/status_code'] = '599'
        return RedactedSpan(span, attributes)

    def _mark_failed_span(self, span: ReadableSpan) -> ReadableSpan:
        """Mark spans that are the source of a failure.

        Adds genkit:failedSpan and genkit:failedPath attributes to spans
        that have genkit:isFailureSource set.

        Args:
            span: The span to potentially mark.

        Returns:
            The span with failure markers if applicable.
        """
        attributes = dict(span.attributes) if span.attributes else {}

        if not attributes.get('genkit:isFailureSource'):
            return span

        attributes['genkit:failedSpan'] = attributes.get('genkit:name', '')
        attributes['genkit:failedPath'] = attributes.get('genkit:path', '')
        return RedactedSpan(span, attributes)

    def _mark_genkit_feature(self, span: ReadableSpan) -> ReadableSpan:
        """Mark root spans with the genkit:feature attribute.

        This helps identify the top-level feature being executed.

        Args:
            span: The span to potentially mark.

        Returns:
            The span with genkit:feature if it's a root span.
        """
        attributes = dict(span.attributes) if span.attributes else {}

        is_root = attributes.get('genkit:isRoot')
        name = attributes.get('genkit:name')

        if not is_root or not name:
            return span

        attributes['genkit:feature'] = name
        return RedactedSpan(span, attributes)

    def _mark_genkit_model(self, span: ReadableSpan) -> ReadableSpan:
        """Mark model spans with the genkit:model attribute.

        This helps identify which model was used in a span.

        Args:
            span: The span to potentially mark.

        Returns:
            The span with genkit:model if it's a model action.
        """
        attributes = dict(span.attributes) if span.attributes else {}

        subtype = attributes.get('genkit:metadata:subtype')
        name = attributes.get('genkit:name')

        if subtype != 'model' or not name:
            return span

        attributes['genkit:model'] = name
        return RedactedSpan(span, attributes)

    def _normalize_labels(self, span: ReadableSpan) -> ReadableSpan:
        """Normalize attribute labels by replacing : with /.

        GCP Cloud Trace has specific requirements for label keys.
        This ensures compatibility by replacing colons with slashes.

        Args:
            span: The span with attributes to normalize.

        Returns:
            The span with normalized attribute keys.
        """
        attributes = dict(span.attributes) if span.attributes else {}

        # Replace : with / in all attribute keys
        normalized: dict[str, Any] = {}
        for key, value in attributes.items():
            normalized[key.replace(':', '/')] = value

        return RedactedSpan(span, normalized)
