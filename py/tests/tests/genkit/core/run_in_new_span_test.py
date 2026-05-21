#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fattened ``run_in_new_span`` helper and Action delegation.

Covers attributes ``run_in_new_span`` writes (name, path, qualifiedPath, input, output, state,
error, metadata) plus a regression test that ``Action._run_with_telemetry`` records
the original exception text in ``genkit:error`` rather than the wrapped GenkitError message.
"""

from collections.abc import Generator, Sequence

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from genkit._core._action import Action, ActionKind
from genkit._core._error import GenkitError
from genkit._core._tracing import SpanMetadata, _parent_path_context, run_in_new_span


@pytest.fixture(autouse=True)
def _reset_parent_path() -> Generator[None, None, None]:
    """Each test starts with an empty parent-path context to keep paths independent."""
    token = _parent_path_context.set('')
    try:
        yield
    finally:
        _parent_path_context.reset(token)


@pytest.fixture
def exporter() -> Generator[InMemorySpanExporter, None, None]:
    """Provide an in-memory span exporter wired into the global tracer provider."""
    provider = trace_api.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace_api.set_tracer_provider(provider)
    exp = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exp)
    provider.add_span_processor(processor)
    try:
        yield exp
    finally:
        exp.clear()


def _by_name(spans: Sequence[ReadableSpan], name: str) -> ReadableSpan:
    matches = [s for s in spans if s.name == name]
    assert matches, f'no span named {name!r} in {[s.name for s in spans]}'
    return matches[-1]


def test_writes_name_path_and_state_success(exporter: InMemorySpanExporter) -> None:
    with run_in_new_span(SpanMetadata(name='hello', type='util')):
        pass

    span = _by_name(exporter.get_finished_spans(), 'hello')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:name'] == 'hello'
    assert attrs['genkit:type'] == 'util'
    assert attrs['genkit:state'] == 'success'
    assert attrs['genkit:path'] == '/{hello,t:util}'
    assert attrs['genkit:qualifiedPath'] == '/{hello,t:util}'


def test_writes_input_from_metadata(exporter: InMemorySpanExporter) -> None:
    class Payload(BaseModel):
        msg: str

    with run_in_new_span(SpanMetadata(name='echo', type='action', subtype='tool', input=Payload(msg='hi'))):
        pass

    span = _by_name(exporter.get_finished_spans(), 'echo')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:input'] == '{"msg":"hi"}'
    assert attrs['genkit:path'] == '/{echo,t:action,s:tool}'
    assert attrs['genkit:metadata:subtype'] == 'tool'


def test_writes_output_from_metadata_on_success(exporter: InMemorySpanExporter) -> None:
    meta = SpanMetadata(name='answer', type='util')
    with run_in_new_span(meta):
        meta.output = {'result': 42}

    span = _by_name(exporter.get_finished_spans(), 'answer')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:output'] == '{"result": 42}'
    assert attrs['genkit:state'] == 'success'


def test_records_error_attributes(exporter: InMemorySpanExporter) -> None:
    with pytest.raises(RuntimeError, match='boom'):
        with run_in_new_span(SpanMetadata(name='broken', type='util')):
            raise RuntimeError('boom')

    span = _by_name(exporter.get_finished_spans(), 'broken')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:state'] == 'error'
    assert attrs['genkit:error'] == 'boom'
    assert span.status.status_code == trace_api.StatusCode.ERROR


def test_nested_path_inherits_parent_qualified_path(exporter: InMemorySpanExporter) -> None:
    with run_in_new_span(SpanMetadata(name='outer', type='flow')):
        with run_in_new_span(SpanMetadata(name='inner', type='flowStep')):
            pass

    inner = _by_name(exporter.get_finished_spans(), 'inner')
    inner_attrs = dict(inner.attributes or {})
    assert inner_attrs['genkit:qualifiedPath'] == '/{outer,t:flow}/{inner,t:flowStep}'


def test_metadata_metadata_dict_is_flattened_and_telemetry_labels_pass_through(
    exporter: InMemorySpanExporter,
) -> None:
    with run_in_new_span(
        SpanMetadata(
            name='step',
            type='flowStep',
            metadata={'flow:name': 'pipeline', 'attempt': 2},
            telemetry_labels={'genkit:custom:tag': 'foo'},
        )
    ):
        pass

    span = _by_name(exporter.get_finished_spans(), 'step')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:metadata:flow:name'] == 'pipeline'
    assert attrs['genkit:metadata:attempt'] == '2'
    # Raw telemetry_labels pass through without the genkit:metadata: prefix.
    assert attrs['genkit:custom:tag'] == 'foo'


@pytest.mark.asyncio
async def test_action_span_metadata_uses_short_keys(exporter: InMemorySpanExporter) -> None:
    """``Action.span_metadata`` uses short keys; ``run_in_new_span`` adds ``genkit:metadata:`` once.

    Locks in the simplified contract introduced alongside this refactor: framework call
    sites (e.g. ``_flow.py``, ``_resource.py``) pass short keys like ``flow:name``, and
    the helper produces ``genkit:metadata:flow:name`` on the span.
    """

    async def noop() -> str:
        return 'ok'

    action = Action(
        name='myFlow',
        kind=ActionKind.FLOW,
        fn=noop,
        span_metadata={'flow:name': 'myFlow'},
    )
    await action.run()

    span = _by_name(exporter.get_finished_spans(), 'myFlow')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:metadata:flow:name'] == 'myFlow'
    assert 'genkit:metadata:genkit:metadata:flow:name' not in attrs


@pytest.mark.asyncio
async def test_action_error_attribute_keeps_original_text(exporter: InMemorySpanExporter) -> None:
    """Regression: the action span should record ``str(original_e)`` in ``genkit:error``,

    not the wrapped GenkitError's ``"Error while running action ..."`` message. This
    locks in the SoC contract: ``run_in_new_span`` records the exception it sees, and
    ``_run_with_telemetry`` wraps GenkitError OUTSIDE the with-block so the wrap
    doesn't clobber the recorded attribute.
    """

    async def kaboom(_: str | None) -> None:
        raise ValueError('original boom')

    action = Action(name='kaboomAction', kind=ActionKind.CUSTOM, fn=kaboom)

    with pytest.raises(GenkitError):
        await action.run()

    span = _by_name(exporter.get_finished_spans(), 'kaboomAction')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:error'] == 'original boom'
    assert attrs['genkit:type'] == 'action'
    assert attrs['genkit:metadata:subtype'] == 'custom'
    assert attrs['genkit:state'] == 'error'


@pytest.mark.asyncio
async def test_action_context_telemetry_sanitizes_unserializable(exporter: InMemorySpanExporter) -> None:
    """Verify that unserializable values in action context are dropped from tracing metadata.

    Also verify that JSON-serializable values are kept.
    """
    import json

    class UnserializableObject:
        def __repr__(self) -> str:
            return 'Unserializable'

    async def noop() -> str:
        return 'ok'

    action = Action(
        name='sanitizedFlow',
        kind=ActionKind.FLOW,
        fn=noop,
    )

    # We pass a context dictionary with both serializable and unserializable values,
    # including nested dictionaries and lists.
    complex_context: dict[str, object] = {
        'auth': {
            'user_id': 123,
            'token': 'secret_token',
            'raw_connection': UnserializableObject(),  # should be dropped
        },
        'serializable_list': [1, 'two', {'nested_key': 'nested_val'}],
        'unserializable_list': [1, UnserializableObject(), 3],  # UnserializableObject should be dropped, keeping [1, 3]
        'top_level_unserializable': UnserializableObject(),  # should be dropped entirely
    }

    await action.run(context=complex_context)

    span = _by_name(exporter.get_finished_spans(), 'sanitizedFlow')
    attrs = dict(span.attributes or {})

    # The context key is mapped under genkit:metadata:context
    assert 'genkit:metadata:context' in attrs
    context_attr = attrs['genkit:metadata:context']
    assert isinstance(context_attr, str)
    context_json = json.loads(context_attr)

    # Assertions
    assert context_json['auth']['user_id'] == 123
    assert context_json['auth']['token'] == 'secret_token'
    assert context_json['auth']['raw_connection'] == 'Unserializable'

    assert context_json['serializable_list'] == [1, 'two', {'nested_key': 'nested_val'}]
    assert context_json['unserializable_list'] == [1, 'Unserializable', 3]
    assert context_json['top_level_unserializable'] == 'Unserializable'


@pytest.mark.asyncio
async def test_action_context_telemetry_circular_references(exporter: InMemorySpanExporter) -> None:
    """Verify that circular references inside the context are proactively detected and dropped."""
    import json

    async def noop() -> str:
        return 'ok'

    action = Action(
        name='circularFlow',
        kind=ActionKind.FLOW,
        fn=noop,
    )

    # Setup a context dictionary with circular references
    circular_context: dict[str, object] = {
        'key': 'val',
    }
    circular_context['self'] = circular_context

    await action.run(context=circular_context)

    span = _by_name(exporter.get_finished_spans(), 'circularFlow')
    attrs = dict(span.attributes or {})

    assert 'genkit:metadata:context' in attrs
    context_attr = attrs['genkit:metadata:context']
    assert isinstance(context_attr, str)
    context_json = json.loads(context_attr)

    # 'key' is serializable, and 'self' circular reference should be safely cut off with '[Circular]'
    assert context_json == {'key': 'val', 'self': '[Circular]'}
