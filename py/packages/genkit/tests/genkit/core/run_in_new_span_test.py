#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What shows up on a span: path, input, output, errors, and redacted context."""

import asyncio
import json
import logging
from collections.abc import Generator, Sequence

import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, Part, Role
from genkit._ai._tools import Interrupt, ToolRunContext
from genkit._core._action import Action, ActionRunContext
from genkit._core._telemetry._attrs import metadata_key
from genkit._core._telemetry._http import ActiveSpan
from genkit._core._telemetry._instrumentation import (
    parent_path_context,
    start_attributes,
)
from genkit.model import ModelRequest
from genkit.plugin_api import ActionKind
from genkit.telemetry import (
    SpanMetadata,
    run_in_new_span,
)


@pytest.fixture(autouse=True)
def _reset_parent_path() -> Generator[None, None, None]:
    """Each test starts with an empty parent-path context to keep paths independent."""
    token = parent_path_context.set('')
    try:
        yield
    finally:
        parent_path_context.reset(token)


def _by_name(spans: Sequence[ActiveSpan], name: str) -> ActiveSpan:
    matches = [s for s in spans if s.name == name]
    assert matches, f'no span named {name!r} in {[s.name for s in spans]}'
    return matches[-1]


def test_start_attributes_includes_input_excludes_outcome() -> None:
    """Start-known attrs include input; state/output wait until the body finishes."""
    attrs = start_attributes(
        SpanMetadata(
            name='myTool',
            action_type='tool.v2',
            input='in',
            attributes={
                'user:label': 'x',
                'genkit:init': '{"sessionId": "s"}',
                'genkit:metadata:key': 'value',
            },
        ),
        qualified_path='/{chatFlow,t:flow}/{myTool,t:action,s:tool.v2}',
        is_action=True,
    )
    assert list(attrs.items()) == [
        ('user:label', 'x'),
        ('genkit:name', 'myTool'),
        ('genkit:path', '/{chatFlow,t:flow}/{myTool,t:action,s:tool.v2}'),
        ('genkit:qualifiedPath', '/{chatFlow,t:flow}/{myTool,t:action,s:tool.v2}'),
        ('genkit:type', 'action'),
        ('genkit:metadata:subtype', 'tool.v2'),
        ('genkit:metadata:key', 'value'),
        ('genkit:input', '"in"'),
        ('genkit:init', '{"sessionId": "s"}'),
    ]
    for forbidden in ('genkit:state', 'genkit:output', 'genkit:isRoot'):
        assert forbidden not in attrs


def test_start_attributes_json_input() -> None:
    """A dict input is JSON on genkit:input."""
    attrs = start_attributes(
        SpanMetadata(name='echo', action_type='custom', input={'msg': 'hi'}),
        qualified_path='/{echo,t:action,s:custom}',
        is_action=True,
    )
    assert attrs['genkit:input'] == '{"msg": "hi"}'


@pytest.mark.asyncio
async def test_writes_name_path_and_state_success(exporter) -> None:
    """A successful span has name, path, and state=success."""

    async def body(_span: object) -> None:
        return None

    await run_in_new_span('hello', body, action_type='util')

    span = _by_name(exporter.get_finished_spans(), 'hello')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:name'] == 'hello'
    assert attrs['genkit:type'] == 'util'
    assert attrs['genkit:state'] == 'success'
    assert attrs['genkit:path'] == '/{hello,t:util}'
    assert attrs['genkit:qualifiedPath'] == '/{hello,t:util}'
    assert 'genkit:output' not in attrs


@pytest.mark.asyncio
async def test_writes_input_from_metadata(exporter) -> None:
    class Payload(BaseModel):
        msg: str

    async def body(_span: object) -> None:
        return None

    await run_in_new_span(
        'echo',
        body,
        action_type='tool.v2',
        input=Payload(msg='hi'),
        is_action=True,
    )

    span = _by_name(exporter.get_finished_spans(), 'echo')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:input'] == '{"msg":"hi"}'
    assert attrs['genkit:path'] == '/{echo,t:action,s:tool.v2}'
    assert attrs['genkit:metadata:subtype'] == 'tool.v2'


@pytest.mark.asyncio
async def test_init_reaches_dev_ui(exporter) -> None:
    """An action run with init writes it as JSON on genkit:init."""

    async def noop() -> str:
        return 'ok'

    action = Action(name='agentRun', kind=ActionKind.CUSTOM, fn=noop)
    await action.run(init={'sessionId': 'session-123'})

    span = _by_name(exporter.get_finished_spans(), 'agentRun')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:init'] == '{"sessionId": "session-123"}'


@pytest.mark.asyncio
async def test_dev_ui_action_span_attributes_unchanged(exporter) -> None:
    """An action span shows as genkit:type=action with its kind in genkit:metadata:subtype."""

    async def noop() -> str:
        return 'ok'

    action = Action(name='getWeather', kind=ActionKind.TOOL, fn=noop)
    await action.run()

    span = _by_name(exporter.get_finished_spans(), 'getWeather')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:type'] == 'action'
    assert attrs['genkit:metadata:subtype'] == 'tool.v2'
    assert attrs['genkit:path'] == '/{getWeather,t:action,s:tool.v2}'
    assert attrs['genkit:qualifiedPath'] == '/{getWeather,t:action,s:tool.v2}'


@pytest.mark.asyncio
async def test_dev_ui_plain_util_span_is_not_an_action(exporter) -> None:
    """ai.generate()'s helper span is genkit:type=util with no subtype, even though 'util' is also an action kind."""
    ai = Genkit(model='echoModel')

    async def echo(_req: ModelRequest) -> ModelResponse:
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('hi')]))

    ai.define_model(name='echoModel', fn=echo)
    await ai.generate(prompt='hello')

    span = _by_name(exporter.get_finished_spans(), 'generate')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:type'] == 'util'
    assert 'genkit:metadata:subtype' not in attrs
    assert attrs['genkit:path'] == '/{generate,t:util}'


@pytest.mark.asyncio
async def test_custom_metadata_keeps_string_format(exporter) -> None:
    """A True action metadata value is written as the string "True"."""

    async def noop() -> str:
        return 'ok'

    action = Action(name='flagged', kind=ActionKind.FLOW, fn=noop, span_metadata={'flow:beta': True})
    await action.run()

    span = _by_name(exporter.get_finished_spans(), 'flagged')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:metadata:flow:beta'] == 'True'


@pytest.mark.asyncio
async def test_ignore_trace_label_on_span(exporter) -> None:
    """A run labeled genkitx:ignore-trace=true still exports that label on its span."""

    async def noop() -> str:
        return 'ok'

    action = Action(name='playground', kind=ActionKind.EXECUTABLE_PROMPT, fn=noop)
    await action.run(telemetry_labels={'genkitx:ignore-trace': 'true'})

    span = _by_name(exporter.get_finished_spans(), 'playground')
    attrs = dict(span.attributes or {})
    assert attrs['genkitx:ignore-trace'] == 'true'


@pytest.mark.asyncio
async def test_writes_output_from_return_value_on_success(exporter) -> None:
    async def body(_span: object) -> dict[str, int]:
        return {'result': 42}

    await run_in_new_span('answer', body, action_type='util')

    span = _by_name(exporter.get_finished_spans(), 'answer')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:output'] == '{"result": 42}'
    assert attrs['genkit:state'] == 'success'


@pytest.mark.asyncio
async def test_records_error_attributes(exporter) -> None:
    async def body(_span: object) -> None:
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await run_in_new_span('broken', body, action_type='util')

    span = _by_name(exporter.get_finished_spans(), 'broken')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:state'] == 'error'
    assert attrs['genkit:error'] == 'boom'
    assert span.status_code == 2


@pytest.mark.asyncio
async def test_cancelled_span_leaves_state_unset(exporter, caplog: pytest.LogCaptureFixture) -> None:
    """Abort/timeout is unfinished work — neither success nor error."""

    async def body(_span: object) -> None:
        raise asyncio.CancelledError()

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(asyncio.CancelledError):
            await run_in_new_span('abortedTurn', body, action_type='util')

    span = _by_name(exporter.get_finished_spans(), 'abortedTurn')
    attrs = dict(span.attributes or {})
    assert 'genkit:state' not in attrs
    assert 'genkit:error' not in attrs
    assert span.status_code != 2
    assert not any('Error in run_in_new_span' in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_tool_interrupt_is_not_recorded_as_span_error(exporter, caplog: pytest.LogCaptureFixture) -> None:
    """Tool interrupts are control flow — the tool span must not look like a failure.

    Drives a real ``@ai.tool`` that raises ``Interrupt``; the caller gets that
    same ``Interrupt`` back.
    """
    ai = Genkit()

    @ai.tool(name='transfer')
    async def transfer(inp: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        raise Interrupt({'reason': 'needs_approval'})

    action = await ai._registry.resolve_action(kind=ActionKind.TOOL, name='transfer')
    assert action is not None

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Interrupt) as ei:
            await action.run({'amount': 100})

    assert ei.value.metadata == {'reason': 'needs_approval'}

    span = _by_name(exporter.get_finished_spans(), 'transfer')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:state'] == 'success'
    assert 'genkit:error' not in attrs
    assert span.status_code != 2
    assert json.loads(attrs['genkit:metadata:interrupt']) == {'reason': 'needs_approval'}
    assert not any('Error in run_in_new_span' in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_nested_path_inherits_parent_qualified_path(exporter) -> None:
    async def inner(_span: object) -> None:
        return None

    async def outer(_span: object) -> None:
        await run_in_new_span('inner', inner, action_type='flowStep')

    await run_in_new_span('outer', outer, action_type='flow')

    inner_span = _by_name(exporter.get_finished_spans(), 'inner')
    inner_attrs = dict(inner_span.attributes or {})
    assert inner_attrs['genkit:qualifiedPath'] == '/{outer,t:flow}/{inner,t:flowStep}'


@pytest.mark.asyncio
async def test_run_step_metadata_is_flattened(exporter) -> None:
    """ai.run(metadata=...) lands each key as genkit:metadata:<k> with str() values."""
    ai = Genkit()

    async def step() -> None:
        return None

    await ai.run(name='step', fn=step, metadata={'flow:name': 'pipeline', 'attempt': 2})

    span = _by_name(exporter.get_finished_spans(), 'step')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:type'] == 'flowStep'
    assert attrs['genkit:metadata:flow:name'] == 'pipeline'
    assert attrs['genkit:metadata:attempt'] == '2'


@pytest.mark.asyncio
async def test_plain_span_attributes_pass_through(exporter) -> None:
    """Attributes on a plain span land as-is, without the genkit:metadata: prefix."""

    async def body(_span: object) -> None:
        return None

    await run_in_new_span('step', body, action_type='flowStep', attributes={'genkit:custom:tag': 'foo'})

    span = _by_name(exporter.get_finished_spans(), 'step')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:custom:tag'] == 'foo'


@pytest.mark.asyncio
async def test_action_span_metadata_uses_short_keys(exporter) -> None:
    """``Action.span_metadata`` uses short keys; the action runner adds ``genkit:metadata:`` once.

    Framework call sites (e.g. ``_flow.py``) pass short keys like ``flow:name``,
    and the span gets ``genkit:metadata:flow:name``.
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
async def test_action_error_attribute_keeps_original_text(exporter) -> None:
    """The action span records the body's own error text in ``genkit:error``."""

    async def kaboom(_: str | None) -> None:
        raise ValueError('original boom')

    action = Action(name='kaboomAction', kind=ActionKind.CUSTOM, fn=kaboom)

    with pytest.raises(ValueError, match='original boom'):
        await action.run()

    span = _by_name(exporter.get_finished_spans(), 'kaboomAction')
    attrs = dict(span.attributes or {})
    assert attrs['genkit:error'] == 'original boom'
    assert attrs['genkit:type'] == 'action'
    assert attrs['genkit:metadata:subtype'] == 'custom'
    assert attrs['genkit:state'] == 'error'


@pytest.mark.asyncio
async def test_action_context_telemetry_sanitizes_unserializable(exporter) -> None:
    """Verify that unserializable values in action context are dropped from tracing metadata.

    Also verify that JSON-serializable values are kept.
    """

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
        'session': {
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
    assert context_json['session']['user_id'] == 123
    assert context_json['session']['token'] == 'secret_token'
    assert context_json['session']['raw_connection'] == 'Unserializable'

    assert context_json['serializable_list'] == [1, 'two', {'nested_key': 'nested_val'}]
    assert context_json['unserializable_list'] == [1, 'Unserializable', 3]
    assert context_json['top_level_unserializable'] == 'Unserializable'


@pytest.mark.asyncio
async def test_action_context_telemetry_redacts_auth_and_secrets(exporter) -> None:
    """The Context panel hides identity and keys. The live action still sees them."""
    seen: dict[str, object] = {}

    async def peek(_input: object, ctx: ActionRunContext) -> str:
        seen['secrets'] = ctx.context['secrets']
        seen['auth'] = ctx.context['auth']
        return 'ok'

    action = Action(
        name='redactContext',
        kind=ActionKind.CUSTOM,
        fn=peek,
    )

    await action.run(
        context={
            'auth': {'token': 'ya29'},
            'secrets': {'api_key': 'sk-live'},
            'locale': 'en-US',
            'nested': {'auth': 'this stays'},
        }
    )

    assert seen['secrets'] == {'api_key': 'sk-live'}
    assert seen['auth'] == {'token': 'ya29'}

    span = _by_name(exporter.get_finished_spans(), 'redactContext')
    attrs = dict(span.attributes or {})
    raw_context = attrs['genkit:metadata:context']
    assert isinstance(raw_context, str)
    context_json = json.loads(raw_context)

    assert context_json['auth'] == '<redacted>'
    assert context_json['secrets'] == '<redacted>'
    assert context_json['locale'] == 'en-US'
    assert context_json['nested']['auth'] == 'this stays'


@pytest.mark.asyncio
async def test_action_context_telemetry_whole_bag_redaction(exporter) -> None:
    """Top-level auth and secrets bags are replaced in full regardless of key names."""

    async def noop(_input: object, _ctx: ActionRunContext) -> str:
        return 'ok'

    action = Action(
        name='multiSecretFlow',
        kind=ActionKind.FLOW,
        fn=noop,
    )

    await action.run(
        context={
            'auth': {'uid': '123', 'roles': ['admin'], 'custom': {'nested': True}},
            'secrets': {'api_key': 'k1', 'signing_key': 'k2', 'token_data': {'hash': 'abc'}},
            'request_id': 'req-987',
        }
    )

    span = _by_name(exporter.get_finished_spans(), 'multiSecretFlow')
    attrs = dict(span.attributes or {})
    raw_context = attrs['genkit:metadata:context']
    assert isinstance(raw_context, str)
    context_json = json.loads(raw_context)

    assert context_json['auth'] == '<redacted>'
    assert context_json['secrets'] == '<redacted>'
    assert context_json['request_id'] == 'req-987'


@pytest.mark.asyncio
async def test_action_context_telemetry_circular_references(exporter) -> None:
    """Verify that circular references inside the context are proactively detected and dropped."""

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


def test_metadata_key_prevents_double_prefix() -> None:
    assert metadata_key('flow:name') == 'genkit:metadata:flow:name'
    assert metadata_key('genkit:metadata:flow:name') == 'genkit:metadata:flow:name'


def test_start_attributes_precedence_over_telemetry_labels() -> None:
    meta = SpanMetadata(
        name='realName',
        attributes={
            'genkit:name': 'fakeName',
            'genkit:path': 'fakePath',
            'user:label': 'custom',
        },
    )
    attrs = start_attributes(meta, qualified_path='/realPath')
    assert attrs['genkit:name'] == 'realName'
    assert attrs['genkit:path'] == '/realPath'
    assert attrs['genkit:qualifiedPath'] == '/realPath'
    assert attrs['user:label'] == 'custom'
