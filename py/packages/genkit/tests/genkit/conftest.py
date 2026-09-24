# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Hex ids and in-memory Developer UI spans for tests (no OpenTelemetry minting)."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Generator, Mapping
from typing import TypeVar

import pytest

from genkit._core._telemetry._instrumentation import (
    SpanContext,
    SpanMetadata,
    reset_instrumentation,
)
from genkit._core._telemetry.http import ActiveSpan, DirectHttpInstrumentation
from genkit.telemetry import configure_instrumentation

T = TypeVar('T')


class MemoryCollectorSink:
    def __init__(self) -> None:
        self._spans: list[ActiveSpan] = []

    def export_spans(self, spans: list[ActiveSpan], *, resource_attributes: dict[str, object]) -> None:
        self._spans.extend(spans)

    def export_logs(self, payload: dict[str, object]) -> None:
        return

    def flush(self) -> None:
        return

    def shutdown(self) -> None:
        return

    def get_finished_spans(self) -> list[ActiveSpan]:
        by_id: dict[str, ActiveSpan] = {}
        for span in self._spans:
            by_id[span.span_id] = span
        return list(by_id.values())

    def clear(self) -> None:
        self._spans.clear()


class HexSpanContext:
    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        return

    def set_output(self, value: object) -> None:
        return

    def set_state(self, state: str) -> None:
        return


class HexInstrumentation:
    """Mints hex ids without OpenTelemetry."""

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: Callable[[SpanContext], Awaitable[T]],
    ) -> T:
        return await next(HexSpanContext())


def recording_http_instrumentation() -> tuple[DirectHttpInstrumentation, MemoryCollectorSink]:
    sink = MemoryCollectorSink()
    return DirectHttpInstrumentation(sink, capture_logs=False), sink


@pytest.fixture
def exporter() -> Generator[MemoryCollectorSink, None, None]:
    inst, sink = recording_http_instrumentation()
    reset_instrumentation()
    configure_instrumentation(inst)
    try:
        yield sink
    finally:
        reset_instrumentation()


@pytest.fixture
def hex_ids() -> Generator[None, None, None]:
    reset_instrumentation()
    configure_instrumentation(HexInstrumentation())
    try:
        yield
    finally:
        reset_instrumentation()
