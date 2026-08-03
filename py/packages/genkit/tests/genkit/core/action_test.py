#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import pytest

from genkit._core._action import (
    Action,
    ActionKind,
    ActionRunContext,
    BidiAction,
    DapQualifiedName,
    _action_context,
    create_action_key,
    get_current_context,
    parse_action_key,
    parse_dap_qualified_name,
    parse_plugin_name_from_action_name,
)
from genkit._core._error import GenkitError


def test_action_enum_behaves_like_str() -> None:
    """Ensure the ActionType behaves like a string.

    This test verifies that the ActionType enum values can be compared
    directly with strings and that the correct variants are used.
    """
    assert ActionKind.CUSTOM == 'custom'
    assert ActionKind.EMBEDDER == 'embedder'
    assert ActionKind.EVALUATOR == 'evaluator'
    assert ActionKind.EXECUTABLE_PROMPT == 'executable-prompt'
    assert ActionKind.AGENT == 'agent'
    assert ActionKind.FLOW == 'flow'
    assert ActionKind.MODEL == 'model'
    assert ActionKind.PROMPT == 'prompt'
    assert ActionKind.TOOL == 'tool'
    assert ActionKind.UTIL == 'util'


def test_parse_action_key_valid() -> None:
    """Parse action key valid."""
    test_cases = [
        ('/prompt/my-prompt', (ActionKind.PROMPT, 'my-prompt')),
        ('/model/gpt-4', (ActionKind.MODEL, 'gpt-4')),
        (
            '/model/vertexai/gemini-1.0',
            (ActionKind.MODEL, 'vertexai/gemini-1.0'),
        ),
        ('/custom/test-action', (ActionKind.CUSTOM, 'test-action')),
        ('/flow/my-flow', (ActionKind.FLOW, 'my-flow')),
        ('/agent/my-agent', (ActionKind.AGENT, 'my-agent')),
    ]

    for key, expected in test_cases:
        kind, name = parse_action_key(key)
        assert kind == expected[0]
        assert name == expected[1]


def test_parse_action_key_invalid_format() -> None:
    """Parse action key invalid format."""
    invalid_keys = [
        'invalid_key',  # Missing separator
        '/missing-kind',  # Missing kind
        'missing-name/',  # Missing name
        '',  # Empty string
        '/',  # Just separator
    ]

    for key in invalid_keys:
        with pytest.raises(ValueError, match='Invalid action key format'):
            parse_action_key(key)


def test_parse_dap_qualified_name() -> None:
    """Parse provider:innerKind/innerName segments."""
    assert parse_dap_qualified_name('my-dap:tool/echo') == DapQualifiedName('my-dap', 'tool', 'echo')
    assert parse_dap_qualified_name('plugin/foo:model/bar') is None
    assert parse_dap_qualified_name('plain-name') is None
    assert parse_dap_qualified_name('no-slash:toolonly') is None
    assert parse_dap_qualified_name(':tool/x') is None


def test_create_action_key() -> None:
    """Create action key."""
    assert create_action_key(ActionKind.CUSTOM, 'foo') == '/custom/foo'
    assert create_action_key(ActionKind.MODEL, 'foo') == '/model/foo'
    assert create_action_key(ActionKind.PROMPT, 'foo') == '/prompt/foo'
    assert create_action_key(ActionKind.TOOL, 'foo') == '/tool/foo'
    assert create_action_key(ActionKind.UTIL, 'foo') == '/util/foo'
    assert create_action_key(ActionKind.AGENT, 'foo') == '/agent/foo'


def test_sync_action_rejected() -> None:
    """Sync functions are rejected - all actions must be async."""

    def sync_foo() -> str:
        return 'syncFoo'

    with pytest.raises(TypeError, match='Action handlers must be async functions'):
        Action(name='syncFoo', kind=ActionKind.CUSTOM, fn=sync_foo)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_define_async_action() -> None:
    """Define and run an async action."""

    async def async_foo() -> str:
        """An async action that returns 'asyncFoo'."""
        return 'asyncFoo'

    action = Action(name='asyncFoo', kind=ActionKind.CUSTOM, fn=async_foo)

    assert (await action.run()).response == 'asyncFoo'
    assert (await async_foo()) == 'asyncFoo'


@pytest.mark.asyncio
async def test_define_async_action_with_input() -> None:
    """Define and run an async action with input."""

    async def async_foo(input: str) -> str:
        """An async action that returns 'asyncFoo' with an input."""
        return f'asyncFoo {input}'

    action = Action(name='asyncFoo', kind=ActionKind.CUSTOM, fn=async_foo)

    assert (await action.run('foo')).response == 'asyncFoo foo'
    assert (await async_foo('foo')) == 'asyncFoo foo'


@pytest.mark.asyncio
async def test_define_async_action_with_input_and_context() -> None:
    """Define and run async action with input and context."""

    async def async_foo(input: str, ctx: ActionRunContext) -> str:
        """An async action that returns 'syncFoo' with an input and context."""
        return f'syncFoo {input} {ctx.context["foo"]}'

    action = Action(name='syncFoo', kind=ActionKind.CUSTOM, fn=async_foo)

    assert (await action.run('foo', context={'foo': 'bar'})).response == 'syncFoo foo bar'
    assert (await async_foo('foo', ActionRunContext(context={'foo': 'bar'}))) == 'syncFoo foo bar'


@pytest.mark.asyncio
async def test_streaming_action_with_callback() -> None:
    """Streaming action with on_chunk callback."""

    async def foo(
        input: str,
        ctx: ActionRunContext,
    ) -> int:
        ctx.send_chunk('1')
        ctx.send_chunk('2')
        return 3

    action = Action(name='foo', kind=ActionKind.CUSTOM, fn=foo)

    chunks: list[object] = []
    result = await action.run('foo', on_chunk=chunks.append)

    assert result.response == 3
    assert chunks == ['1', '2']


@pytest.mark.asyncio
async def test_streaming_action_with_stream_method() -> None:
    """Streaming action using the stream() method."""

    async def foo(
        input: str,
        ctx: ActionRunContext,
    ) -> int:
        ctx.send_chunk('1')
        ctx.send_chunk('2')
        return 3

    action = Action(name='foo', kind=ActionKind.CUSTOM, fn=foo)

    chunks: list[object] = []
    result = action.stream('foo')
    async for chunk in result.stream:
        chunks.append(chunk)

    assert await result.response == 3
    assert chunks == ['1', '2']


def test_parse_plugin_name_from_action_name() -> None:
    """Parse plugin name from the action name."""
    assert parse_plugin_name_from_action_name('foo') is None
    assert parse_plugin_name_from_action_name('foo/bar') == 'foo'
    assert parse_plugin_name_from_action_name('foo/bar/baz') == 'foo'


@pytest.mark.asyncio
async def test_propagates_context_via_contextvar() -> None:
    """Context is properly propagated via contextvar."""

    async def foo(_: str | None, ctx: ActionRunContext) -> str:
        return json.dumps(ctx.context)

    foo_action = cast(Action[str | None, str], Action(name='foo', kind=ActionKind.CUSTOM, fn=foo))

    async def bar() -> str:
        return (await foo_action.run()).response

    bar_action = cast(Action[None, str], Action(name='bar', kind=ActionKind.CUSTOM, fn=bar))

    async def baz() -> str:
        return (await bar_action.run()).response

    baz_action = cast(Action[None, str], Action(name='baz', kind=ActionKind.CUSTOM, fn=baz))

    first = baz_action.run(context={'foo': 'bar'})
    second = baz_action.run(context={'bar': 'baz'})

    assert (await second).response == '{"bar": "baz"}'
    assert (await first).response == '{"foo": "bar"}'


@pytest.mark.asyncio
async def test_action_raises_errors() -> None:
    """Action raises error with necessary metadata."""

    async def foo(_: str | None, ctx: ActionRunContext) -> None:
        raise Exception('oops')

    action = Action(name='fooAction', kind=ActionKind.CUSTOM, fn=foo)

    with pytest.raises(GenkitError, match=r'.*Error while running action fooAction.*') as e:
        await action.run()

    assert 'stack' in e.value.details
    assert 'trace_id' in e.value.details
    assert str(e.value.cause) == 'oops'


@pytest.mark.asyncio
async def test_run_raises_on_none_input_when_input_required() -> None:
    """run() raises GenkitError when input is None but the action requires it."""

    async def typed_fn(input: str) -> str:
        return f'got {input}'

    action = Action(name='typedAction', kind=ActionKind.CUSTOM, fn=typed_fn)

    with pytest.raises(GenkitError, match=r'.*requires input but none was provided.*'):
        await action.run(input=None)


@pytest.mark.asyncio
async def test_run_succeeds_with_valid_input() -> None:
    """run() succeeds when valid input is provided."""

    async def typed_fn(input: str) -> str:
        return f'got {input}'

    action = Action(name='typedAction', kind=ActionKind.CUSTOM, fn=typed_fn)

    result = await action.run(input='hello')
    assert result.response == 'got hello'


@pytest.mark.asyncio
async def test_run_no_input_type_allows_none() -> None:
    """run() allows None input when action has no input type."""

    async def no_input_fn() -> str:
        return 'no input needed'

    action = Action(name='noInputAction', kind=ActionKind.CUSTOM, fn=no_input_fn)

    result = await action.run(input=None)
    assert result.response == 'no input needed'


@pytest.mark.asyncio
async def test_action_context_isolation_sequential_and_nested() -> None:
    """Action context is isolated and does not bleed sequentially or permanently override in nested runs."""

    # 1. Sequential isolation test
    async def get_context(_: None, ctx: ActionRunContext) -> dict[str, object] | None:
        return ctx.context

    tool_action = Action(name='getContext', kind=ActionKind.TOOL, fn=get_context)

    # First run sets context
    res1 = await tool_action.run(context={'auth': 'user1'})
    assert res1.response == {'auth': 'user1'}

    # Second run does NOT set context (should be empty/None)
    res2 = await tool_action.run()
    assert res2.response == {}  # Bleeding check

    # 2. Nested isolation test (parent calls child with overrides)
    async def child_fn(_: None, ctx: ActionRunContext) -> dict[str, object] | None:
        return ctx.context

    child_action = Action(name='childAction', kind=ActionKind.CUSTOM, fn=child_fn)

    async def parent_fn(_: None, ctx: ActionRunContext) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        # Run child action with different context
        child_res = await child_action.run(context={'auth': 'child_secret'})
        # Return parent context (which should still be parent's original context!)
        return ctx.context, child_res.response

    parent_action = Action(name='parentAction', kind=ActionKind.CUSTOM, fn=parent_fn)

    # Run parent action with its own context
    res = await parent_action.run(context={'auth': 'parent_secret'})
    parent_ctx, child_ctx = res.response

    assert child_ctx == {'auth': 'child_secret'}
    assert parent_ctx == {'auth': 'parent_secret'}  # Permanent override check

    assert get_current_context() is None


@pytest.mark.asyncio
async def test_action_context_reset_on_exception() -> None:
    """ContextVar is reset even when the action fn raises (#5555)."""

    async def boom(_: None, ctx: ActionRunContext) -> None:
        raise RuntimeError('explode')

    action = Action(name='boom', kind=ActionKind.CUSTOM, fn=boom)

    with pytest.raises(GenkitError, match='explode'):
        await action.run(context={'auth': 'transient'})

    assert get_current_context() is None
    assert _action_context.get(None) is None


@pytest.mark.asyncio
async def test_action_explicit_empty_context_does_not_inherit() -> None:
    """Passing context={} sets an empty bag instead of inheriting ambient (#5555)."""

    async def get_context(_: None, ctx: ActionRunContext) -> dict[str, object]:
        return dict(ctx.context)

    action = Action(name='getContext', kind=ActionKind.TOOL, fn=get_context)

    async def parent(_: None, ctx: ActionRunContext) -> dict[str, object]:
        child = await action.run(context={})
        return child.response  # type: ignore[return-value]

    parent_action = Action(name='parent', kind=ActionKind.CUSTOM, fn=parent)
    res = await parent_action.run(context={'auth': 'parent'})
    assert res.response == {}


@pytest.mark.asyncio
async def test_stream_resets_action_context() -> None:
    """stream() goes through run() and must not leave ambient context (#5555)."""

    async def echo(_: None, ctx: ActionRunContext) -> dict[str, object]:
        return dict(ctx.context)

    action = Action(name='echo', kind=ActionKind.CUSTOM, fn=echo)
    stream = action.stream(context={'auth': 'streamer'})
    chunks = [c async for c in stream.stream]
    assert chunks == []
    assert await stream.response == {'auth': 'streamer'}
    assert get_current_context() is None


@pytest.mark.asyncio
async def test_action_context_is_shallow_copied() -> None:
    """Action.run must not mutate the caller's context dict."""

    async def mutate(_: None, ctx: ActionRunContext) -> None:
        ctx.context['mutated'] = True

    action = Action(name='mutate', kind=ActionKind.CUSTOM, fn=mutate)
    caller_ctx: dict[str, object] = {'auth': 'user1'}
    await action.run(context=caller_ctx)
    assert caller_ctx == {'auth': 'user1'}


@pytest.mark.asyncio
async def test_bidi_stream_context_isolation_sequential_and_nested() -> None:
    """BidiAction.stream_bidi must isolate context like Action.run (#5555)."""

    async def bidi_fn(
        init: None,
        inputs: AsyncIterator[Any],
        send_chunk: Callable[[Any], None],
    ) -> dict[str, object] | None:
        # Drain the one-shot input stream; return ambient action context.
        async for _ in inputs:
            pass
        return get_current_context()

    bidi = BidiAction(name='bidiCtx', kind=ActionKind.CUSTOM, bidi_fn=bidi_fn)

    conn1 = await bidi.stream_bidi(context={'auth': 'user1'})
    await conn1.close()
    assert await conn1.output() == {'auth': 'user1'}
    assert get_current_context() is None

    # Second session with no context must not see user1.
    conn2 = await bidi.stream_bidi()
    await conn2.close()
    out2 = await conn2.output()
    assert out2 is None or out2 == {}
    assert get_current_context() is None

    # Nested: parent action calls bidi with a different context and must keep its own.
    async def parent_fn(_: None, ctx: ActionRunContext) -> tuple[dict[str, object], dict[str, object] | None]:
        conn = await bidi.stream_bidi(context={'auth': 'child_secret'})
        await conn.close()
        child_out = await conn.output()
        return dict(ctx.context), child_out

    parent = Action(name='parentBidi', kind=ActionKind.CUSTOM, fn=parent_fn)
    res = await parent.run(context={'auth': 'parent_secret'})
    parent_ctx, child_ctx = res.response
    assert parent_ctx == {'auth': 'parent_secret'}
    assert child_ctx == {'auth': 'child_secret'}
    assert get_current_context() is None


@pytest.mark.asyncio
async def test_run_defaulted_input_arg_allows_none() -> None:
    """A function with a Python default for its input should be callable with no input.

    Otherwise `await my_flow()` on `async def my_flow(name: str = 'world')`
    would surprise the caller with INVALID_ARGUMENT — typing accepts the
    call but the runtime would reject it.
    """

    async def greet(name: str = 'world') -> str:
        return f'hi {name}'

    action = Action(name='greet', kind=ActionKind.CUSTOM, fn=greet)

    assert (await action.run(input='Alice')).response == 'hi Alice'
    assert (await action.run(input=None)).response == 'hi world'
    assert (await action.run()).response == 'hi world'


@pytest.mark.asyncio
async def test_run_defaulted_input_arg_allows_none_with_ctx() -> None:
    """Same as above but for 2-arg (input + ctx) actions."""

    async def greet(name: str = 'world', ctx: ActionRunContext | None = None) -> str:
        return f'hi {name}'

    action = Action(name='greet_ctx', kind=ActionKind.CUSTOM, fn=greet)

    assert (await action.run(input='Bob')).response == 'hi Bob'
    assert (await action.run(input=None)).response == 'hi world'
    assert (await action.run()).response == 'hi world'
