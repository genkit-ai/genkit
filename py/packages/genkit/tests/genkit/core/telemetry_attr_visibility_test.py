#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Pin attribute visibility: start-known attrs at on_start, annotate/outcome by on_end."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from opentelemetry import context as otel_context, trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider

from genkit._core._trace._attrs import Attr, State
from genkit._core._tracing import (
    SpanMetadata,
    annotate,
    restore_default_telemetry_handlers,
    run_in_new_span,
)


class AttrSnapshotProcessor(SpanProcessor):
    """Records attribute dicts seen at on_start and on_end for each span."""

    def __init__(self) -> None:
        self.on_start_by_name: dict[str, dict[str, Any]] = {}
        self.on_end_by_name: dict[str, dict[str, Any]] = {}

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        del parent_context
        self.on_start_by_name[span.name] = dict(span.attributes or {})

    def on_end(self, span: ReadableSpan) -> None:
        self.on_end_by_name[span.name] = dict(span.attributes or {})

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return True


@pytest.fixture(autouse=True)
def _reset_handlers() -> Generator[None, None, None]:
    restore_default_telemetry_handlers()
    try:
        yield
    finally:
        restore_default_telemetry_handlers()


@pytest.fixture
def snapshots() -> Generator[AttrSnapshotProcessor, None, None]:
    provider = trace_api.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace_api.set_tracer_provider(provider)
    processor = AttrSnapshotProcessor()
    provider.add_span_processor(processor)
    try:
        yield processor
    finally:
        processor.shutdown()


@pytest.mark.asyncio
async def test_input_visible_at_on_start_annotate_and_state_by_on_end(
    snapshots: AttrSnapshotProcessor,
) -> None:
    """Start attrs (input) at on_start; annotate + genkit:state only by on_end."""

    async def work() -> str:
        annotate('custom', 'x')
        return 'done'

    result = await run_in_new_span(
        SpanMetadata(name='visibilityTurn', type='util', input={'prompt': 'hi'}),
        work,
    )
    assert result == 'done'

    start = snapshots.on_start_by_name['visibilityTurn']
    end = snapshots.on_end_by_name['visibilityTurn']

    # 1) genkit:input present in the live on_start snapshot
    assert start[Attr.INPUT] == '{"prompt": "hi"}'
    assert start[Attr.TYPE] == 'util'
    assert Attr.PATH in start

    # 2) annotate mid-run is absent from on_start, present by on_end
    assert 'custom' not in start
    assert end['custom'] == 'x'

    # 3) genkit:state success lands before end, never at on_start
    assert Attr.STATE not in start
    assert end[Attr.STATE] == State.SUCCESS
