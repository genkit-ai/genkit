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

"""Adjusting trace exporter for PII redaction and span enhancement."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from genkit._core._compat import override


def _copy_attrs(span: ReadableSpan) -> dict[str, Any]:
    """Return a mutable copy of span attributes."""
    return dict(span.attributes) if span.attributes else {}


class RedactedSpan(ReadableSpan):
    """A span wrapper that overrides attributes while delegating everything else."""

    # pyrefly:ignore[bad-override]
    _attributes: dict[str, Any]

    def __init__(self, span: ReadableSpan, attributes: dict[str, Any]) -> None:
        self._span = span
        self._attributes = attributes

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._span, name)

    @property
    def attributes(self) -> dict[str, Any]:
        """Return the modified attributes."""
        # pyrefly: ignore[bad-return] - dict[str, Any] is compatible with Mapping at runtime
        return self._attributes


class AdjustingTraceExporter(SpanExporter):
    """Wraps a SpanExporter to redact PII and enhance spans for cloud plugins (GCP, AWS)."""

    REDACTED: ClassVar[str] = '<redacted>'

    def __init__(
        self,
        exporter: SpanExporter,
        log_input_and_output: bool = False,
        project_id: str | None = None,
        error_handler: Callable[[Exception], None] | None = None,
    ) -> None:
        self._exporter = exporter
        self._log_input_and_output = log_input_and_output
        self._project_id = project_id
        self._error_handler = error_handler

    @property
    def project_id(self) -> str | None:
        return self._project_id

    @property
    def log_input_and_output(self) -> bool:
        return self._log_input_and_output

    @override
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        adjusted = [self._adjust(span) for span in spans]
        try:
            return self._exporter.export(adjusted)
        except Exception as e:
            if self._error_handler:
                self._error_handler(e)
            raise

    @override
    def shutdown(self) -> None:
        self._exporter.shutdown()

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        if hasattr(self._exporter, 'force_flush'):
            return self._exporter.force_flush(timeout_millis)
        return True

    def _adjust(self, span: ReadableSpan) -> ReadableSpan:
        """Apply all adjustments to a span."""
        span = self._redact_pii(span)
        span = self._mark_error(span)
        span = self._mark_failure_source(span)
        span = self._mark_feature(span)
        span = self._mark_model(span)
        span = self._normalize_labels(span)
        return span

    def _redact_pii(self, span: ReadableSpan) -> ReadableSpan:
        if self._log_input_and_output:
            return span
        attrs = _copy_attrs(span)
        keys_to_redact = [k for k in ('genkit:input', 'genkit:output') if k in attrs]
        if not keys_to_redact:
            return span
        for key in keys_to_redact:
            attrs[key] = self.REDACTED
        return RedactedSpan(span, attrs)

    def _mark_error(self, span: ReadableSpan) -> ReadableSpan:
        if not span.status or span.status.status_code != StatusCode.ERROR:
            return span
        attrs = _copy_attrs(span)
        attrs['/http/status_code'] = '599'
        return RedactedSpan(span, attrs)

    def _mark_failure_source(self, span: ReadableSpan) -> ReadableSpan:
        attrs = _copy_attrs(span)
        if not attrs.get('genkit:isFailureSource'):
            return span
        attrs['genkit:failedSpan'] = attrs.get('genkit:name', '')
        attrs['genkit:failedPath'] = attrs.get('genkit:path', '')
        return RedactedSpan(span, attrs)

    def _mark_feature(self, span: ReadableSpan) -> ReadableSpan:
        attrs = _copy_attrs(span)
        if not attrs.get('genkit:isRoot') or not attrs.get('genkit:name'):
            return span
        attrs['genkit:feature'] = attrs['genkit:name']
        return RedactedSpan(span, attrs)

    def _mark_model(self, span: ReadableSpan) -> ReadableSpan:
        attrs = _copy_attrs(span)
        if attrs.get('genkit:metadata:subtype') != 'model' or not attrs.get('genkit:name'):
            return span
        attrs['genkit:model'] = attrs['genkit:name']
        return RedactedSpan(span, attrs)

    def _normalize_labels(self, span: ReadableSpan) -> ReadableSpan:
        attrs = _copy_attrs(span)
        normalized = {k.replace(':', '/'): v for k, v in attrs.items()}
        return RedactedSpan(span, normalized)
