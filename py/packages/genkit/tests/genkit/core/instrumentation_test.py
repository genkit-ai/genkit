#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What configure_instrumentation and run_in_new_span do when you stack backends."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

import pytest

from genkit._core._action import Action
from genkit._core._telemetry._instrumentation import (
    SpanContext,
    SpanMetadata,
    flush_instrumentations,
    is_instrumented_by,
    reset_instrumentation,
    run_in_new_span,
    set_custom_metadata_attributes,
    set_span_state,
)
from genkit.plugin_api import ActionKind
from genkit.telemetry import FlushableInstrumentation, configure_instrumentation


class RecordedSpan:
    def __init__(self, label: str, *, trace_id: str = '', span_id: str = '') -> None:
        self.label = label
        self.trace_id = trace_id
        self.span_id = span_id
        self.metadata: list[Mapping[str, object]] = []
        self.states: list[str] = []

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        self.metadata.append(metadata)

    def set_state(self, state: str) -> None:
        self.states.append(state)


class FakeInstrumentation:
    def __init__(
        self,
        label: str,
        log: list[str],
        *,
        trace_id: str = '',
        span_id: str = '',
    ) -> None:
        self.label = label
        self.log = log
        self.trace_id = trace_id
        self.span_id = span_id
        self.spans: list[RecordedSpan] = []
        self.seen: list[SpanMetadata] = []

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: Callable[[SpanContext], Awaitable[object]],
    ) -> object:
        self.seen.append(metadata)
        self.log.append(f'enter:{self.label}')
        span = RecordedSpan(self.label, trace_id=self.trace_id, span_id=self.span_id)
        self.spans.append(span)
        try:
            return await next(span)
        finally:
            self.log.append(f'exit:{self.label}')


@pytest.fixture(autouse=True)
def _reset() -> object:
    reset_instrumentation()
    yield
    reset_instrumentation()


@pytest.mark.asyncio
async def test_noop_span_when_nothing_configured() -> None:
    """No backend configured: the action still runs and ids stay empty."""
    seen: SpanContext | None = None

    async def body(span: SpanContext) -> str:
        nonlocal seen
        seen = span
        return 'ok'

    result = await run_in_new_span('op', body)
    assert result == 'ok'
    assert seen is not None
    assert seen.trace_id == ''
    assert seen.span_id == ''
    seen.set_metadata({'k': 'v'})


@pytest.mark.asyncio
async def test_composes_providers_in_registration_order() -> None:
    """Two configure_instrumentation calls wrap the action in registration order."""
    log: list[str] = []
    configure_instrumentation(FakeInstrumentation('a', log))
    configure_instrumentation(FakeInstrumentation('b', log))

    async def body(_span: SpanContext) -> str:
        log.append('body')
        return 'x'

    await run_in_new_span('op', body)
    assert log == ['enter:a', 'enter:b', 'body', 'exit:b', 'exit:a']


@pytest.mark.asyncio
async def test_in_flight_span_keeps_the_backends_it_started_with() -> None:
    """configure_instrumentation during an in-flight span does not join that span."""
    log: list[str] = []
    configure_instrumentation(FakeInstrumentation('a', log))
    configure_instrumentation(FakeInstrumentation('b', log))

    async def body(_span: SpanContext) -> None:
        reset_instrumentation()
        configure_instrumentation(FakeInstrumentation('c', log))
        log.append('body')

    await run_in_new_span('op', body)
    assert log == ['enter:a', 'enter:b', 'body', 'exit:b', 'exit:a']
    assert 'enter:c' not in log


@pytest.mark.asyncio
async def test_set_custom_metadata_writes_to_every_backend() -> None:
    """set_custom_metadata_attributes copies onto every configured backend."""
    log: list[str] = []
    a = FakeInstrumentation('a', log)
    b = FakeInstrumentation('b', log)
    configure_instrumentation(a)
    configure_instrumentation(b)

    async def body(_span: SpanContext) -> None:
        set_custom_metadata_attributes({'hello': 'world'})

    await run_in_new_span('op', body)
    assert a.spans[0].metadata == [{'hello': 'world'}]
    assert b.spans[0].metadata == [{'hello': 'world'}]


@pytest.mark.asyncio
async def test_set_state_writes_to_every_backend() -> None:
    """span.set_state copies onto every configured backend."""
    log: list[str] = []
    a = FakeInstrumentation('a', log)
    b = FakeInstrumentation('b', log)
    configure_instrumentation(a)
    configure_instrumentation(b)

    async def body(span: SpanContext) -> None:
        span.set_state('error')

    await run_in_new_span('op', body)
    assert a.spans[0].states == ['error']
    assert b.spans[0].states == ['error']


@pytest.mark.asyncio
async def test_set_span_state_writes_to_every_backend() -> None:
    """set_span_state copies onto every configured backend."""
    log: list[str] = []
    a = FakeInstrumentation('a', log)
    b = FakeInstrumentation('b', log)
    configure_instrumentation(a)
    configure_instrumentation(b)

    async def body(_span: SpanContext) -> None:
        set_span_state('error')

    await run_in_new_span('op', body)
    assert a.spans[0].states == ['error']
    assert b.spans[0].states == ['error']


def test_configure_rejects_a_non_instrumentation() -> None:
    """configure_instrumentation(object()) raises; it does not silently no-op."""
    with pytest.raises(TypeError, match='Instrumentation instance'):
        configure_instrumentation(object())  # type: ignore[arg-type]


def test_configure_rejects_the_class_instead_of_an_instance() -> None:
    """Passing FakeInstrumentation the class, not an instance, raises."""
    with pytest.raises(
        TypeError,
        match='FakeInstrumentation',
    ):
        configure_instrumentation(FakeInstrumentation)  # type: ignore[arg-type]


def test_is_instrumented_by_rejects_a_string_name() -> None:
    """is_instrumented_by wants the class, not the name as a string."""
    with pytest.raises(TypeError, match='builtins.str'):
        is_instrumented_by('HexInstrumentation')  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_run_in_new_span_rejects_a_sync_body() -> None:
    """run_in_new_span only accepts an async body."""

    def sync_body(_span: SpanContext) -> str:
        return 'ok'

    with pytest.raises(TypeError, match='sync_body'):
        await run_in_new_span('op', sync_body)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_trace_id_comes_from_the_first_backend_that_has_one() -> None:
    """The action's trace_id is the first non-empty id in the backend chain."""
    log: list[str] = []
    configure_instrumentation(FakeInstrumentation('a', log))
    configure_instrumentation(FakeInstrumentation('b', log, trace_id='trace-b', span_id='span-b'))

    seen_trace = ''
    seen_span = ''

    async def body(span: SpanContext) -> None:
        nonlocal seen_trace, seen_span
        seen_trace = span.trace_id
        seen_span = span.span_id

    await run_in_new_span('op', body)
    assert seen_trace == 'trace-b'
    assert seen_span == 'span-b'


@pytest.mark.asyncio
async def test_a_raised_error_still_closes_every_backend() -> None:
    """A raised error still closes every backend in reverse order."""
    log: list[str] = []
    configure_instrumentation(FakeInstrumentation('a', log))
    configure_instrumentation(FakeInstrumentation('b', log))

    async def body(_span: SpanContext) -> None:
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await run_in_new_span('op', body)

    assert log == ['enter:a', 'enter:b', 'exit:b', 'exit:a']


async def _ok() -> str:
    return 'ok'


@pytest.mark.asyncio
async def test_provider_sees_real_action_kind() -> None:
    """A provider sees action_type 'model', 'flow', 'tool.v2' for those actions."""
    rec = FakeInstrumentation('rec', [])
    configure_instrumentation(rec)

    for name, kind in (('m', ActionKind.MODEL), ('f', ActionKind.FLOW), ('t', ActionKind.TOOL)):
        await Action(name=name, kind=kind, fn=_ok).run()

    assert [(m.name, m.action_type) for m in rec.seen] == [('m', 'model'), ('f', 'flow'), ('t', 'tool.v2')]


@pytest.mark.asyncio
async def test_is_action_does_not_appear_on_provider_attributes() -> None:
    """is_action=True is not a key on metadata.attributes; providers see the real labels only."""
    rec = FakeInstrumentation('rec', [])
    configure_instrumentation(rec)

    await Action(name='labeled', kind=ActionKind.FLOW, fn=_ok, span_metadata={'k': 'v'}).run(
        telemetry_labels={'genkitx:ignore-trace': 'true'},
    )

    async def body(_span: SpanContext) -> None:
        return None

    await run_in_new_span('plain', body, action_type='util', is_action=True)

    assert len(rec.seen) == 2
    assert rec.seen[0].attributes == {'genkitx:ignore-trace': 'true', 'genkit:metadata:k': 'v'}
    assert rec.seen[1].attributes == {}


def test_telemetry_exports_run_in_new_span() -> None:
    """from genkit.telemetry import run_in_new_span is how you wrap your own span."""
    from genkit._core._telemetry._instrumentation import run_in_new_span as impl
    from genkit.telemetry import run_in_new_span as exported

    assert exported is impl
    import genkit.telemetry as telemetry

    assert 'run_in_new_span' in telemetry.__all__


def test_span_metadata_attributes_keep_bool_int_and_float() -> None:
    """SpanMetadata.attributes keeps bool, int, and float instead of requiring strings."""
    meta = SpanMetadata(name='step', attributes={'cached': True, 'retries': 3, 'cost': 0.002})
    assert meta.attributes == {'cached': True, 'retries': 3, 'cost': 0.002}


def test_telemetry_exports_provider_types() -> None:
    """SpanNext, DisposableInstrumentation, FlushableInstrumentation import from genkit.telemetry."""
    import genkit.telemetry as telemetry

    for name in ('SpanNext', 'DisposableInstrumentation', 'FlushableInstrumentation'):
        assert name in telemetry.__all__
        assert getattr(telemetry, name) is not None


def test_flush_instrumentations_flushes_flushable() -> None:
    """flush_instrumentations calls flush() on a provider implementing FlushableInstrumentation."""

    class Buffered(FakeInstrumentation):
        def __init__(self) -> None:
            super().__init__('buffered', [])
            self.flushed = 0

        def flush(self) -> None:
            self.flushed += 1

    buffered = Buffered()
    plain = FakeInstrumentation('plain', [])
    configure_instrumentation(buffered)
    configure_instrumentation(plain)
    assert isinstance(buffered, FlushableInstrumentation)
    assert not isinstance(plain, FlushableInstrumentation)

    flush_instrumentations()

    assert buffered.flushed == 1
