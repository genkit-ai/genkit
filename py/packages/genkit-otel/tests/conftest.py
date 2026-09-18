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

from __future__ import annotations

from collections.abc import Iterator

import pytest
from genkit_otel import GenAiInstrumentation
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class OtelHarness:
    """In-memory OTel leftover: spans, metrics, and log records a later reader sees."""

    def __init__(self) -> None:
        self.spans = InMemorySpanExporter()
        self.logs = InMemoryLogRecordExporter()
        self.metrics = InMemoryMetricReader()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.spans))
        self.meter_provider = MeterProvider(metric_readers=[self.metrics])
        self.logger_provider = LoggerProvider()
        self.logger_provider.add_log_record_processor(SimpleLogRecordProcessor(self.logs))

    def instrumentation(self, **kwargs: object) -> GenAiInstrumentation:
        return GenAiInstrumentation(
            tracer=self.tracer_provider.get_tracer('test'),
            meter=self.meter_provider.get_meter('test'),
            otel_logger=self.logger_provider.get_logger('test'),
            **kwargs,
        )

    def span_named(self, name: str):
        matches = [s for s in self.spans.get_finished_spans() if s.name == name]
        return matches[-1] if matches else None

    def attr(self, span, key: str):
        attrs = span.attributes or {}
        return attrs.get(key)

    def clear(self) -> None:
        self.spans.clear()
        self.logs.clear()


@pytest.fixture
def harness() -> Iterator[OtelHarness]:
    h = OtelHarness()
    try:
        yield h
    finally:
        h.clear()
