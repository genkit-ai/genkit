#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import asyncio
import contextvars
import json
import pathlib
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

import pytest
import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import assert_type

from genkit import Document, Genkit, Message, ModelResponse, ModelResponseChunk, MultipartToolResponse, Part, Tool, tool
from genkit._ai._formats._types import FormatDef, Formatter, FormatterConfig
from genkit._ai._generate import DEFAULT_MAX_TURNS, ChunkAccumulator, augment_with_context, generate_action
from genkit._ai._model import text_from_content, text_from_message
from genkit._ai._tools import Interrupt, ToolRunContext, define_tool
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError, PublicError, RuntimeErrorReason, wrap_http_error
from genkit._core._model import GenerateActionOptions, ModelRequest, Resume
from genkit._core._registry import Registry
from genkit._core._typing import (
    FinishReason,
    GenerateActionOutputConfig,
    GenerationUsage,
    Role,
    ToolRequest,
)
from genkit.middleware import (
    BaseMiddleware,
    GenerateHookParams,
    GenerateMiddleware,
    GenerateMiddlewareContext,
    MiddlewareRef,
    ModelHookParams,
    ToolHookParams,
)
from genkit.plugin_api import ActionKind, MiddlewarePlugin, new_middleware
from genkit.testing import (
    ScriptedModel,
    define_echo_model,
    define_scripted_model,
)


def _to_dict(obj: object) -> object:
    """Convert object to dict for test comparisons."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _to_json(obj: object, indent: int | None = None) -> str:
    """Local test helper: serialize to JSON for assertion error messages.

    Uses model_dump_json for BaseModel, json.dumps for dicts/other.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=indent)
    return json.dumps(obj, indent=indent)


@pytest.fixture
def setup_test() -> tuple[Genkit, ScriptedModel]:
    """Setup the test."""
    ai = Genkit()

    pm, _ = define_scripted_model(ai)

    @ai.tool(name='testTool')
    async def test_tool() -> object:
        """description"""  # noqa: D403, D415
        return 'tool called'

    return (ai, pm)


@pytest.mark.asyncio
async def test_simple_text_generate_request(
    setup_test: tuple[Genkit, ScriptedModel],
) -> None:
    """Test that the generate action can generate text."""
    ai, pm = setup_test

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('bye')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part.from_text('hi')],
                ),
            ],
        ),
    )

    assert response.text == 'bye'
    assert response.finish_reason == FinishReason.STOP
    assert response.finish_message is None
    assert response.error is None
    assert response.message is not None
    assert response.message.text == 'bye'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[0].text == 'hi'
    assert response.messages[1] == response.message


@pytest.mark.asyncio
async def test_generate_user_from_text_model_from_text_reads_without_root(
    setup_test: tuple[Genkit, ScriptedModel],
) -> None:
    ai, pm = setup_test
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('hello')]),
        )
    )

    response = await ai.generate(
        model='scriptedModel',
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
    )

    assert response.message is not None
    assert response.message.content[0].text == 'hello'
    assert response.message.content[0].media is None


@pytest.mark.asyncio
async def test_generate_user_text_and_media_model_sees_both_parts(
    setup_test: tuple[Genkit, ScriptedModel],
) -> None:
    ai, pm = setup_test
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )
    )

    await ai.generate(
        model='scriptedModel',
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part.from_text('caption'),
                    Part.from_media('https://example.com/x.png'),
                ],
            )
        ],
    )

    assert pm.last_request is not None
    parts = pm.last_request.messages[0].content
    assert parts[0].text == 'caption'
    assert parts[1].media is not None
    assert parts[1].media.url == 'https://example.com/x.png'
    assert parts[1].text is None


def _queue_ok(pm: ScriptedModel) -> None:
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )
    )


def _sent_text(pm: ScriptedModel) -> list[tuple[str, list[str | None]]]:
    assert pm.last_request is not None
    return [(m.role, [p.text for p in m.content]) for m in pm.last_request.messages]


_AS_WRITTEN_CASES = [
    pytest.param('hello {{name}}', id='unfilled variable'),
    pytest.param('Reply like {"dish": {{', id='unclosed braces'),
    pytest.param('<<<dotprompt:role:system>>> hi', id='role marker'),
    pytest.param('Describe {{media url="https://example.com/x.png"}}', id='media helper'),
    pytest.param('{{> persona}} hi', id='registered partial'),
    pytest.param('{{shout "hey"}} hi', id='registered helper'),
]


def _register_generate_string_fixtures(ai: Genkit) -> None:
    ai.define_partial('persona', 'You are a pirate.')
    ai.define_helper('shout', lambda *args: 'HEY')


@pytest.mark.asyncio
@pytest.mark.parametrize('prompt', _AS_WRITTEN_CASES)
async def test_generate_prompt_string_is_sent_as_written(
    setup_test: tuple[Genkit, ScriptedModel],
    prompt: str,
) -> None:
    """`ai.generate(prompt=...)` is not a template; `define_prompt` is where templating lives."""
    ai, pm = setup_test
    _register_generate_string_fixtures(ai)
    _queue_ok(pm)

    await ai.generate(model='scriptedModel', prompt=prompt)

    assert _sent_text(pm) == [(Role.USER, [prompt])]


@pytest.mark.asyncio
@pytest.mark.parametrize('system', _AS_WRITTEN_CASES)
async def test_generate_system_string_is_sent_as_written(
    setup_test: tuple[Genkit, ScriptedModel],
    system: str,
) -> None:
    """`ai.generate(system=...)` reaches the model unchanged as the system message."""
    ai, pm = setup_test
    _register_generate_string_fixtures(ai)
    _queue_ok(pm)

    await ai.generate(model='scriptedModel', system=system, prompt='hi')

    assert _sent_text(pm) == [(Role.SYSTEM, [system]), (Role.USER, ['hi'])]


@pytest.mark.asyncio
async def test_generate_messages_string_raises_type_error(
    setup_test: tuple[Genkit, ScriptedModel],
) -> None:
    """`ai.generate(messages='hello {{name}}')` raises TypeError and the model is never called."""
    ai, pm = setup_test

    with pytest.raises(TypeError, match='messages must be a list of Message'):
        await ai.generate(model='scriptedModel', messages='hello {{name}}')  # type: ignore[arg-type]

    assert pm.last_request is None


@pytest.mark.asyncio
async def test_generate_stream_chunk_text_from_factory_part(
    setup_test: tuple[Genkit, ScriptedModel],
) -> None:
    ai, pm = setup_test
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('hi')]),
        )
    )
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('h')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('i')]),
        ],
    ]

    stream_result = ai.generate_stream(model='scriptedModel', prompt='do it')
    texts: list[str] = []
    async for chunk in stream_result.stream:
        texts.append(chunk.text)
    assert texts == ['h', 'i']


@pytest.mark.asyncio
async def test_generate_stream_earlier_chunk_accumulated_text_stays_put(
    setup_test: tuple[Genkit, ScriptedModel],
) -> None:
    """An earlier stream chunk's accumulated_text does not grow as later chunks arrive."""
    ai, pm = setup_test
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('abc')]),
        )
    )
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('a')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('b')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('c')]),
        ],
    ]

    stream_result = ai.generate_stream(model='scriptedModel', prompt='do it')
    first: ModelResponseChunk | None = None
    async for chunk in stream_result.stream:
        if first is None:
            first = chunk
    assert first is not None
    assert first.accumulated_text == 'a'


@pytest.mark.asyncio
async def test_generate_stream_chunk_output_uses_format_parser(
    setup_test: tuple[Genkit, ScriptedModel],
) -> None:
    """Streaming with output_format='array' puts the parsed list on chunk.output."""
    ai, pm = setup_test
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('[{"id": 1}]')]),
        )
    )
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('[{"id": 1}]')]),
        ],
    ]

    stream_result = ai.generate_stream(model='scriptedModel', prompt='list', output_format='array')
    chunks = [chunk async for chunk in stream_result.stream]
    assert chunks[0].output == [{'id': 1}]


@pytest.mark.asyncio
async def test_simulates_doc_grounding(
    setup_test: tuple[Genkit, ScriptedModel],
) -> None:
    """Test that docs are correctly grounded and injected into prompt."""
    ai, pm = setup_test

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('bye')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part.from_text('hi')],
                ),
            ],
            docs=[Document(content=[Part.from_text('doc content 1')])],
        ),
    )

    grounded_msg = Message(
        role=Role.USER,
        content=[
            Part.from_text('hi'),
            Part.from_text(
                '\n\nUse the following information to complete your task:' + '\n\n- [0]: doc content 1\n\n',
                metadata={'purpose': 'context'},
            ),
        ],
    )

    # the model receives the grounded prompt with docs injected as a context part.
    assert pm.last_request is not None
    assert pm.last_request.messages[0] == grounded_msg

    # the returned request is the conversation we persist: the clean turn. The
    # docs ride along as structured data, not inlined into the message.
    assert response.request is not None
    assert response.request.messages is not None
    assert response.request.messages[0] == Message(role=Role.USER, content=[Part.from_text('hi')])
    assert response.request.docs is not None


# --------------------------------------------------------------------------- #
# Unit tests for the private augment_with_context helper                     #
# --------------------------------------------------------------------------- #


def test_augment_with_context_ignores_no_docs() -> None:
    """No docs -> request returned unchanged (same object identity)."""
    req = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part.from_text('hi')]),
        ],
    )

    transformed_req = augment_with_context(req)

    assert transformed_req is req


def test_augment_with_context_adds_docs_as_context() -> None:
    """Docs are injected as a context-purpose part appended to the last user message."""
    req = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part.from_text('hi')]),
        ],
        docs=[
            Document(content=[Part.from_text('doc content 1')]),
            Document(content=[Part.from_text('doc content 2')]),
        ],
    )

    transformed_req = augment_with_context(req)

    assert transformed_req == ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part.from_text('hi'),
                    Part.from_text(
                        '\n\nUse the following information to complete '
                        + 'your task:\n\n'
                        + '- [0]: doc content 1\n'
                        + '- [1]: doc content 2\n\n',
                        metadata={'purpose': 'context'},
                    ),
                ],
            )
        ],
        docs=[
            Document(content=[Part.from_text('doc content 1')]),
            Document(content=[Part.from_text('doc content 2')]),
        ],
    )


def test_augment_with_context_does_not_mutate_input() -> None:
    """Input request and its messages are not mutated; helper returns a deepcopy."""
    original_user_msg = Message(role=Role.USER, content=[Part.from_text('hi')])
    req = ModelRequest(
        messages=[original_user_msg],
        docs=[Document(content=[Part.from_text('doc content 1')])],
    )
    original_content_len = len(original_user_msg.content)

    transformed_req = augment_with_context(req)

    assert transformed_req is not req
    assert transformed_req.messages[0] is not original_user_msg
    assert len(original_user_msg.content) == original_content_len
    assert len(transformed_req.messages[0].content) == original_content_len + 1


def test_augment_with_context_skips_when_context_already_rendered() -> None:
    """Already-rendered context (purpose=context, no pending flag) is left untouched.

    If a message already contains a context part that was previously rendered
    (non-pending), augment_with_context should return the original request
    unchanged rather than injecting the docs again.
    """
    req = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part.from_text('this is already context', metadata={'purpose': 'context'}),
                    Part.from_text('hi'),
                ],
            ),
        ],
        docs=[
            Document(content=[Part.from_text('doc content 1')]),
        ],
    )

    transformed_req = augment_with_context(req)

    assert transformed_req is req


def test_augment_with_context_with_purpose_part() -> None:
    """A pending context placeholder is replaced in-place with the rendered docs.

    Prompts can include a Part with metadata={'purpose': 'context', 'pending': True}
    as a placeholder.  augment_with_context locates it and swaps it out for the
    actual rendered document context, preserving the surrounding parts.
    """
    req = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part.from_text('insert context here', metadata={'purpose': 'context', 'pending': True}),
                    Part.from_text('hi'),
                ],
            ),
        ],
        docs=[
            Document(content=[Part.from_text('doc content 1')]),
        ],
    )

    transformed_req = augment_with_context(req)

    assert transformed_req == ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part.from_text(
                        '\n\nUse the following information to complete '
                        + 'your task:\n\n'
                        + '- [0]: doc content 1\n\n',
                        metadata={'purpose': 'context'},
                    ),
                    Part.from_text('hi'),
                ],
            )
        ],
        docs=[
            Document(content=[Part.from_text('doc content 1')]),
        ],
    )


# --------------------------------------------------------------------------- #
# Middleware class definitions shared by tests below                           #
# --------------------------------------------------------------------------- #


# Module-level Genkit so `@ai.middleware(...)` can stamp + register the
# classes below at import time. Tests that need a fresh registry construct
# their own `ai = Genkit(...)` locally.
ai = Genkit()
define_echo_model(ai)


@ai.middleware(name='pre_mw')
class PreMiddleware(BaseMiddleware):
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        txt = ''.join(text_from_message(m) for m in params.request.messages)
        return await next_fn(
            ModelHookParams(
                request=ModelRequest(
                    messages=[
                        Message(role=Role.USER, content=[Part.from_text(f'PRE {txt}')]),
                    ],
                ),
            ),
            ctx,
        )


@ai.middleware(name='post_mw')
class PostMiddleware(BaseMiddleware):
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        resp: ModelResponse = await next_fn(params, ctx)
        assert resp.message is not None
        txt = text_from_message(resp.message)
        return ModelResponse(
            finish_reason=resp.finish_reason,
            message=Message(role=Role.USER, content=[Part.from_text(f'{txt} POST')]),
        )


class ExtensionMiddlewarePlugin(MiddlewarePlugin):
    """Test plugin subclass; mirrors ``genkit_middleware.Middleware``."""

    name = 'extension-middleware'


class PostMiddlewarePlugin(ExtensionMiddlewarePlugin):
    middleware = [new_middleware(PostMiddleware, name='post_mw')]


class PrePostMiddlewarePlugin(ExtensionMiddlewarePlugin):
    middleware = [
        new_middleware(PreMiddleware, name='pre_mw'),
        new_middleware(PostMiddleware, name='post_mw'),
    ]


@pytest.mark.asyncio
async def test_generate_accepts_inline_base_middleware_instance() -> None:
    """Inline ``BaseMiddleware`` instances in ``use=`` run without registration."""
    ai = Genkit()
    define_echo_model(ai)

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        use=[PreMiddleware(), PostMiddleware()],
    )

    assert response.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_generate_interleaves_inline_instances_and_middleware_refs() -> None:
    """Inline instances and ``MiddlewareRef`` entries preserve ``use=`` ordering together."""
    ai = Genkit(plugins=[PostMiddlewarePlugin()])
    define_echo_model(ai)

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        use=[PreMiddleware(), MiddlewareRef(name='post_mw')],
    )

    assert response.text == '[ECHO] user: "PRE hi" POST'


class _PrefixConfig(BaseModel):
    prefix: str = 'DEFAULT'


@ai.middleware(name='configured_prefix_mw')
class ConfiguredPrefixMiddleware(BaseMiddleware[_PrefixConfig]):
    """Inline middleware driven purely by a pydantic config field."""

    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        txt = ''.join(text_from_message(m) for m in params.request.messages)
        return await next_fn(
            ModelHookParams(
                request=ModelRequest(
                    messages=[
                        Message(role=Role.USER, content=[Part.from_text(f'{self.config.prefix} {txt}')]),
                    ],
                ),
            ),
            ctx,
        )


class ConfiguredPrefixMiddlewarePlugin(ExtensionMiddlewarePlugin):
    middleware = [new_middleware(ConfiguredPrefixMiddleware, name='configured_prefix_mw')]


@pytest.mark.asyncio
async def test_generate_inline_instance_uses_pydantic_fields() -> None:
    """Config fields passed at construction time drive inline behavior."""
    ai = Genkit()
    define_echo_model(ai)

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        use=[ConfiguredPrefixMiddleware(prefix='[TRACE]')],
    )

    assert response.text == '[ECHO] user: "[TRACE] hi"'


@pytest.mark.asyncio
async def test_generate_inline_instance_accepts_config_object() -> None:
    """``Retry(config=RetryConfig(...))`` attaches a validated config instance."""
    ai = Genkit()
    define_echo_model(ai)

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        use=[ConfiguredPrefixMiddleware(config=_PrefixConfig(prefix='[CFG]'))],
    )

    assert response.text == '[ECHO] user: "[CFG] hi"'


@pytest.mark.asyncio
async def test_generate_middleware_ref_config_instantiates_class() -> None:
    """``MiddlewareRef(config=...)`` feeds ``**config`` into the class constructor."""
    ai = Genkit(plugins=[ConfiguredPrefixMiddlewarePlugin()])
    define_echo_model(ai)

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        use=[MiddlewareRef(name='configured_prefix_mw', config={'prefix': '[SPAN]'})],
    )

    assert response.text == '[ECHO] user: "[SPAN] hi"'


@pytest.mark.asyncio
async def test_ai_middleware_decorator_registers_on_the_app() -> None:
    """``@ai.middleware`` registers the class so it's resolvable by name."""
    local_ai = Genkit()
    define_echo_model(local_ai)

    @local_ai.middleware(name='live_prefix_mw')
    class LivePrefixMiddleware(BaseMiddleware[_PrefixConfig]):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            txt = ''.join(text_from_message(m) for m in params.request.messages)
            return await next_fn(
                ModelHookParams(
                    request=ModelRequest(
                        messages=[
                            Message(role=Role.USER, content=[Part.from_text(f'{self.config.prefix} {txt}')]),
                        ],
                    ),
                ),
                ctx,
            )

    response = await local_ai.generate(
        model='echoModel',
        prompt='hi',
        use=[MiddlewareRef(name='live_prefix_mw', config={'prefix': '[LIVE]'})],
    )

    assert response.text == '[ECHO] user: "[LIVE] hi"'


def test_middleware_validation_raises_correct_errors() -> None:
    """Verify that registering middleware with invalid names raises expected errors."""
    local_ai = Genkit()

    # 1. Test @ai.middleware decorator raising ValueError
    with pytest.raises(ValueError, match='middleware name must be one path-free token'):

        @local_ai.middleware(name='invalid/name')
        class InvalidDecoratorMw(BaseMiddleware):
            pass

    with pytest.raises(ValueError, match='middleware name must be a non-empty string'):

        @local_ai.middleware(name='  ')
        class InvalidDecoratorMwEmpty(BaseMiddleware):
            pass

    # 2. Test new_middleware helper raising ValueError on bad names
    with pytest.raises(ValueError, match='GenerateMiddleware name must be one path-free token'):
        new_middleware(PreMiddleware, name='invalid/name')

    with pytest.raises(ValueError, match='GenerateMiddleware name must be a non-empty string'):
        new_middleware(PreMiddleware, name='')

    # 3. Test new_middleware helper behavior
    desc = new_middleware(PreMiddleware, name='custom_mw', description='custom desc')
    assert isinstance(desc, GenerateMiddleware)
    assert desc.name == 'custom_mw'
    assert desc.description == 'custom desc'

    with pytest.raises(TypeError, match='pass either config= or keyword config fields'):
        ConfiguredPrefixMiddleware(config=_PrefixConfig(prefix='x'), prefix='y')

    class _WrongConfig(BaseModel):
        other: str = 'x'

    with pytest.raises(TypeError, match='expected config type'):
        ConfiguredPrefixMiddleware(config=_WrongConfig())  # type: ignore[arg-type]


def test_base_middleware_rejects_explicit_config_class() -> None:
    with pytest.raises(TypeError, match='must not define Config'):

        class _Bad(BaseMiddleware):
            class Config(BaseModel):
                x: int = 1


def test_base_middleware_infers_config_from_generic() -> None:
    """``BaseMiddleware[RetryConfig]`` sets ``Config`` without a redundant alias."""

    class _RetryConfig(BaseModel):
        max_retries: int = 3

    class _Retry(BaseMiddleware[_RetryConfig]):
        pass

    assert _Retry.Config is _RetryConfig
    assert _Retry(max_retries=5).config.max_retries == 5
    schema = cast(dict[str, Any], new_middleware(_Retry, name='retry').config_schema)
    assert schema['properties']['max_retries']['type'] == 'integer'


@pytest.mark.asyncio
async def test_util_generate_action_runs_use_middleware() -> None:
    """The Dev UI hits ``/util/generate`` directly with ``use=[MiddlewareRef(...)]``.

    That entry point skips the in-process ``generate_action`` veneer, so
    middleware resolution has to live in ``run_generate``, not in
    the veneer. Without that, a hook the user configured in the UI silently
    drops on the floor — exactly the bug this test pins down.
    """
    action = await ai.registry.resolve_action(kind=ActionKind.UTIL, name='generate')
    assert action is not None

    action_response = await action.run(
        GenerateActionOptions(
            model='echoModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            use=[MiddlewareRef(name='configured_prefix_mw', config={'prefix': '[DEV-UI]'})],
        ),
    )
    response = cast(ModelResponse, action_response.response)

    assert response.text == '[ECHO] user: "[DEV-UI] hi"'


@pytest.mark.asyncio
async def test_prompt_call_runs_middleware_declared_on_prompt() -> None:
    """``ai.define_prompt(use=[...])`` actually runs those middleware on call."""
    ai = Genkit()
    define_echo_model(ai)

    my_prompt = ai.define_prompt(
        model='echoModel',
        prompt='hi',
        use=[PreMiddleware(), PostMiddleware()],
    )

    response = await my_prompt()

    assert response.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_prompt_call_runs_per_call_middleware() -> None:
    """``my_prompt(use=[...])`` per-call middleware run too."""
    ai = Genkit()
    define_echo_model(ai)

    my_prompt = ai.define_prompt(model='echoModel', prompt='hi')

    response = await my_prompt(use=[PreMiddleware(), PostMiddleware()])

    assert response.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_prompt_call_use_interleaves_inline_and_refs() -> None:
    """Prompts mix inline ``BaseMiddleware`` and ``MiddlewareRef`` like ``generate``."""
    ai = Genkit(plugins=[PostMiddlewarePlugin()])
    define_echo_model(ai)

    my_prompt = ai.define_prompt(
        model='echoModel',
        prompt='hi',
        use=[PreMiddleware(), MiddlewareRef(name='post_mw')],
    )

    response = await my_prompt()

    assert response.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_prompt_per_call_use_overrides_prompt_use() -> None:
    """Per-call ``use=`` replaces the prompt's declared ``use``, matching ``opts.tools`` semantics."""
    ai = Genkit()
    define_echo_model(ai)

    my_prompt = ai.define_prompt(
        model='echoModel',
        prompt='hi',
        use=[PreMiddleware()],
    )

    response = await my_prompt(use=[PostMiddleware()])

    assert response.text == '[ECHO] user: "hi" POST'


@pytest.mark.asyncio
async def test_prompt_stream_runs_middleware() -> None:
    """``.stream()`` shares the middleware path with ``__call__``."""
    ai = Genkit()
    define_echo_model(ai)

    my_prompt = ai.define_prompt(
        model='echoModel',
        prompt='hi',
        use=[PreMiddleware(), PostMiddleware()],
    )

    streamed = my_prompt.stream()
    response = await streamed.response

    assert response.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_generate_applies_middleware() -> None:
    """When middleware is provided, apply it via MiddlewareRef resolution."""
    ai = Genkit(plugins=[PrePostMiddlewarePlugin()])
    define_echo_model(ai)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='echoModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part.from_text('hi')],
                ),
            ],
            use=[MiddlewareRef(name='pre_mw'), MiddlewareRef(name='post_mw')],
        ),
    )

    assert response.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_generate_middleware_next_fn_args_optional() -> None:
    """Can call next function without modifying params (pass params through)."""
    ai = Genkit(plugins=[PostMiddlewarePlugin()])
    define_echo_model(ai)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='echoModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part.from_text('hi')],
                ),
            ],
            use=[MiddlewareRef(name='post_mw')],
        ),
    )

    assert response.text == '[ECHO] user: "hi" POST'


class RefuseTheAnswerMiddleware(BaseMiddleware):
    """Rejects the model's answer after the provider has already billed for it."""

    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        await next_fn(params, ctx)
        raise GenkitError(status='INTERNAL', message='the rendered card is not in the catalog')


@pytest.mark.asyncio
async def test_middleware_refusing_the_answer_keeps_the_tokens_it_cost() -> None:
    """A refused answer still costs money, and the response still reports the bill.

    Middleware that inspects what the model said — a validator, a renderer, a
    safety pass — runs after the provider has charged for it. Refusing drops
    ``message`` so the bad answer never reaches history you can send again, but
    ``usage`` and ``custom`` stay put, so cost accounting and provider trace ids
    survive the failure.
    """
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('here is a broken card')]),
            usage=GenerationUsage(input_tokens=11, output_tokens=22, total_tokens=33),
            custom={'provider_trace_id': 'abc-123'},
        )
    ]

    response = await ai.generate(
        model='scriptedModel',
        prompt='card please',
        use=[RefuseTheAnswerMiddleware()],
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.usage is not None
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 22
    assert response.usage.total_tokens == 33
    assert response.custom == {'provider_trace_id': 'abc-123'}


@pytest.mark.asyncio
async def test_a_turn_that_never_reached_the_model_reports_no_tokens() -> None:
    """Nothing was billed, so nothing is reported.

    The same middleware refusing before the model runs leaves ``usage`` empty.
    A failed turn never invents a cost it did not incur.
    """

    class RefuseBeforeTheModel(BaseMiddleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            raise GenkitError(status='INTERNAL', message='refused before the call')

    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('never sent')]),
            usage=GenerationUsage(input_tokens=11, output_tokens=22, total_tokens=33),
        )
    ]

    response = await ai.generate(model='scriptedModel', prompt='hi', use=[RefuseBeforeTheModel()])

    assert response.finish_reason == FinishReason.FAILED
    assert response.usage is not None
    assert response.usage.output_tokens is None
    assert pm.request_count == 0


@ai.middleware(name='add_ctx')
class AddContextMiddleware(BaseMiddleware):
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        ctx.custom_context['banana'] = True
        return await next_fn(params, ctx)


@ai.middleware(name='inject_ctx')
class InjectContextMiddleware(BaseMiddleware):
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        txt = ''.join(text_from_message(m) for m in params.request.messages)
        return await next_fn(
            ModelHookParams(
                request=ModelRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content=[Part.from_text(f'{txt} {ctx.custom_context}')],
                        ),
                    ],
                ),
            ),
            ctx,
        )


class ContextMiddlewarePlugin(ExtensionMiddlewarePlugin):
    middleware = [
        new_middleware(AddContextMiddleware, name='add_ctx'),
        new_middleware(InjectContextMiddleware, name='inject_ctx'),
    ]


@pytest.mark.asyncio
async def test_generate_middleware_can_modify_context() -> None:
    """Test that middleware can modify custom_context on the shared generate ctx."""
    ai = Genkit(plugins=[ContextMiddlewarePlugin()])
    define_echo_model(ai)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='echoModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part.from_text('hi')],
                ),
            ],
            use=[MiddlewareRef(name='add_ctx'), MiddlewareRef(name='inject_ctx')],
        ),
        context={'foo': 'bar'},
    )

    assert response.text == '''[ECHO] user: "hi {'foo': 'bar', 'banana': True}"'''


@pytest.mark.asyncio
async def test_generate_middleware_can_modify_stream() -> None:
    """Test that middleware can intercept and modify streaming chunks."""
    ai = Genkit()

    @ai.middleware(name='mod_stream_mw')
    class ModifyStreamMiddleware(BaseMiddleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            if ctx.on_chunk:
                ctx.send_chunk(
                    ModelResponseChunk(
                        role=Role.MODEL,
                        content=[Part.from_text('something extra before')],
                    )
                )

            downstream = ctx.on_chunk

            def chunk_handler(chunk: ModelResponseChunk) -> None:
                if downstream:
                    downstream(
                        ModelResponseChunk(
                            role=Role.MODEL,
                            content=[Part.from_text(f'intercepted: {text_from_content(chunk.content)}')],
                        )
                    )

            previous = ctx.replace_on_chunk(chunk_handler)
            resp = await next_fn(params, ctx)
            ctx.replace_on_chunk(previous)
            if ctx.on_chunk:
                ctx.send_chunk(
                    ModelResponseChunk(
                        role=Role.MODEL,
                        content=[Part.from_text('something extra after')],
                    )
                )
            return resp

    pm, _ = define_scripted_model(ai)

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('bye')]),
        )
    )
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('1')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('2')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('3')]),
        ]
    ]

    got_chunks = []

    def collect_chunks(c: ModelResponseChunk) -> None:
        got_chunks.append(text_from_content(c.content))

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part.from_text('hi')],
                ),
            ],
            use=[MiddlewareRef(name='mod_stream_mw')],
        ),
        on_chunk=collect_chunks,
    )

    assert response.text == 'bye'
    assert got_chunks == [
        'something extra before',
        'intercepted: 1',
        'intercepted: 2',
        'intercepted: 3',
        'something extra after',
    ]


@pytest.mark.asyncio
async def test_stream_interception_chains_across_model_and_generate_hooks() -> None:
    """wrap_model can intercept streaming; wrap_generate can modify the response.

    Matches JS behaviour ('can intercept and modify the stream from model and
    generate interceptors'):
    - wrap_generate installs gen_chunk_handler on ctx.on_chunk
    - wrap_model installs model_chunk_handler on ctx.on_chunk (wrapping gen_chunk_handler)
    - intercept_model_stream captures ctx.on_chunk at install time so it picks up the full chain
    - Raw chunks flow: model → framework wrapper → model_chunk_handler → gen_chunk_handler → caller
    """
    ai = Genkit()
    chunk_intercepts: list[str] = []

    @ai.middleware(name='chain_stream_mw')
    class ChainStreamMiddleware(BaseMiddleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            downstream = ctx.on_chunk

            def model_chunk_handler(chunk: ModelResponseChunk) -> None:
                text = text_from_content(chunk.content)
                chunk_intercepts.append(f'model_mw: {text}')
                if downstream:
                    downstream(
                        ModelResponseChunk(
                            role=Role.MODEL,
                            content=[Part.from_text(text.upper())],
                        )
                    )

            previous = ctx.replace_on_chunk(model_chunk_handler)
            resp = await next_fn(params, ctx)
            ctx.replace_on_chunk(previous)
            return resp

        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            downstream = ctx.on_chunk

            def gen_chunk_handler(chunk: ModelResponseChunk) -> None:
                text = text_from_content(chunk.content)
                chunk_intercepts.append(f'gen_mw: {text}')
                if downstream:
                    downstream(
                        ModelResponseChunk(
                            role=Role.MODEL,
                            content=[Part.from_text(f'[{text}]')],
                        )
                    )

            previous = ctx.replace_on_chunk(gen_chunk_handler)
            resp = await next_fn(params, ctx)
            ctx.replace_on_chunk(previous)

            # Also modify the final response text.
            assert resp.message is not None
            original_text = text_from_message(resp.message)
            return ModelResponse(
                finish_reason=resp.finish_reason,
                message=Message(
                    role=Role.MODEL,
                    content=[Part.from_text(f'modified_result: {original_text}')],
                ),
            )

    pm, _ = define_scripted_model(ai)

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('chunk1chunk2')]),
        )
    )
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('chunk1')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('chunk2')]),
        ]
    ]

    final_chunks: list[str] = []

    def collect(c: ModelResponseChunk) -> None:
        final_chunks.append(text_from_content(c.content))

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[
                Message(role=Role.USER, content=[Part.from_text('test streaming mw')]),
            ],
            use=[MiddlewareRef(name='chain_stream_mw')],
        ),
        on_chunk=collect,
    )

    # Both wrap_model AND wrap_generate chunk handlers are called in order.
    assert chunk_intercepts == [
        'model_mw: chunk1',
        'gen_mw: CHUNK1',
        'model_mw: chunk2',
        'gen_mw: CHUNK2',
    ]

    # Chunks arrive at the caller with both transformations applied:
    # wrap_model uppercases, then wrap_generate bracket-wraps.
    assert final_chunks == ['[CHUNK1]', '[CHUNK2]']

    # wrap_generate CAN still modify the final response — this works.
    assert response.text == 'modified_result: chunk1chunk2'


@pytest.mark.asyncio
async def test_wrap_generate_called_per_turn() -> None:
    """wrap_generate is invoked for each turn of the generate loop.

    This is the two-turn regression test: verifies middleware runs on *every*
    recursive run_wrap_generate call (turn 0 + turn 1 after tool response).
    """
    # Each test-local class closes over its own list so the test can inspect
    # what wrap_generate saw across the (potentially many) fresh instances the
    # registry mints per call.
    iters_a: list[int] = []
    iters_b: list[int] = []

    class TrackerA(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            iters_a.append(params.iteration)
            return await next_fn(params, ctx)

    class TrackerB(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            iters_b.append(params.iteration)
            return await next_fn(params, ctx)

    class GenerateTrackerPlugin(MiddlewarePlugin):
        name = 'extension-middleware'

        def list_middleware(self) -> list[GenerateMiddleware]:
            return [
                new_middleware(TrackerA, name='track_gen', description='track generate'),
                new_middleware(TrackerB, name='track_gen2', description='track generate 2'),
            ]

    ai = Genkit(plugins=[GenerateTrackerPlugin()])
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='testTool')
    async def _test_tool() -> object:
        return 'tool called'

    # No tools: single turn → wrap_generate called once with iteration=0
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    )
    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            use=[MiddlewareRef(name='track_gen')],
        ),
    )
    assert response.text == 'done'
    assert iters_a == [0]

    # With tools: two turns (model→tool→model) → wrap_generate called for each
    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='testTool', input={}, ref='r1'))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('final')]),
        )
    )
    response2 = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            tools=['testTool'],
            use=[MiddlewareRef(name='track_gen2')],
        ),
    )
    assert response2.text == 'final'
    assert iters_b == [0, 1]


@pytest.mark.asyncio
async def test_wrap_tool_called_on_tool_execution() -> None:
    """wrap_tool is invoked for each tool execution."""
    tool_names: list[str] = []

    class Tracker(BaseMiddleware):
        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            tool_req = params.tool_request_part.tool_request
            assert tool_req is not None
            tool_names.append(tool_req.name)
            return await next_fn(params, ctx)

    class ToolTrackerPlugin(MiddlewarePlugin):
        name = 'extension-middleware'

        def list_middleware(self) -> list[GenerateMiddleware]:
            return [new_middleware(Tracker, name='track_tool', description='track tool')]

    ai = Genkit(plugins=[ToolTrackerPlugin()])
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='myTool')
    async def my_tool() -> object:
        return 'result'

    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='myTool', input={}, ref='r1'))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            tools=['myTool'],
            use=[MiddlewareRef(name='track_tool')],
        ),
    )
    assert response.text == 'done'
    assert tool_names == ['myTool']


def test_wrap_tool_params_from_tool_request() -> None:
    scratch = Registry()

    async def fn() -> str:
        return ''

    tool = define_tool(scratch, fn, name='lookup').action()
    params = ToolHookParams(tool_request_part=Part.from_tool_request(name='lookup'), tool=tool)
    assert params.tool_request_part.tool_request is not None
    assert params.tool_request_part.tool_request.name == 'lookup'
    assert params.tool_request_part.text is None


def test_wrap_tool_params_text_part_raises() -> None:
    scratch = Registry()

    async def fn() -> str:
        return ''

    tool = define_tool(scratch, fn, name='lookup').action()
    with pytest.raises(ValidationError, match='tool request'):
        ToolHookParams(tool_request_part=Part.from_text('hi'), tool=tool)


@pytest.mark.asyncio
async def test_generate_context_reaches_tool_run() -> None:
    """``generate(context=...)`` is piped into ``tool.run`` as ``ToolRunContext.context``."""
    seen: list[dict[str, object]] = []

    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='ctxTool')
    async def ctx_tool(_: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        seen.append(dict(ctx.context))
        return 'ok'

    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='ctxTool', input={}, ref='r1'))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            tools=['ctxTool'],
        ),
        context={'user_id': 'u-123'},
    )

    assert response.text == 'done'
    assert seen == [{'user_id': 'u-123'}]


@pytest.mark.asyncio
async def test_generate_without_context_uses_enclosing_flow_context() -> None:
    """``ai.generate()`` inside a flow, with no ``context=``, gives middleware and tools the flow's context.

    Tools would inherit it on their own; middleware only sees what reaches the run.
    """
    seen: list[tuple[str, dict[str, object]]] = []

    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    # The engine builds middleware from its class, so the recorder closes over `seen`.
    class RecordContext(BaseMiddleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            seen.append(('middleware', dict(ctx.custom_context)))
            return await next_fn(params, ctx)

    @ai.tool(name='check_allergies')
    async def check_allergies(_: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        seen.append(('tool', dict(ctx.context)))
        return 'no nuts'

    pm.responses = [
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='check_allergies', input={}, ref='r1'))],
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        ),
    ]

    @ai.flow()
    async def plan_order(_: None) -> str:
        return (
            await ai.generate(
                model='scriptedModel',
                prompt='Plan the order.',
                tools=['check_allergies'],
                use=[RecordContext()],
            )
        ).text

    auth = {'auth': {'uid': 'diner-42'}}
    await plan_order.run(context=auth)

    # Model turn that asks for the tool, the tool run, then the model turn that answers.
    assert seen == [('middleware', auth), ('tool', auth), ('middleware', auth)]


@pytest.mark.asyncio
async def test_generate_resume_context_reaches_tool_run() -> None:
    """``generate(resume=..., context=...)`` pipes ``context`` to ``ToolRunContext.context``."""
    seen: list[dict[str, object]] = []

    ai = Genkit()

    @ai.tool(name='ctx_res_tool')
    async def ctx_res_tool(inp: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        seen.append(dict(ctx.context))
        return 'resumed_value'

    intr_trp = Part.from_tool_request(name='ctx_res_tool', ref='ref-abc', input={}, metadata={'interrupt': True})
    restart_trp = Part.from_tool_request(
        name='ctx_res_tool', ref='ref-abc', input={'approved': True}, metadata={'resumed': True}
    )

    pm, _ = define_scripted_model(ai)
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('all done')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[
                Message(role=Role.USER, content=[Part.from_text('hi')]),
                Message(role=Role.MODEL, content=[intr_trp]),
            ],
            tools=['ctx_res_tool'],
            resume=Resume(restart=[restart_trp]),
        ),
        context={'session_token': 's-999'},
    )

    assert response.text == 'all done'
    assert seen == [{'session_token': 's-999'}]


@pytest.mark.asyncio
async def test_wrap_tool_middleware_custom_context_reaches_tool_run() -> None:
    """Mutations to ``ctx.custom_context`` in ``wrap_tool`` are visible in ``ToolRunContext``."""
    seen: list[dict[str, object]] = []

    ai = Genkit()

    @ai.middleware(name='enrich_tool_ctx', description='add context before tool runs')
    class EnrichToolContextMiddleware(BaseMiddleware):
        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            ctx.custom_context['added_by_mw'] = True
            return await next_fn(params, ctx)

    pm, _ = define_scripted_model(ai)

    @ai.tool(name='ctxTool')
    async def ctx_tool(_: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        seen.append(dict(ctx.context))
        return 'ok'

    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='ctxTool', input={}, ref='r1'))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            tools=['ctxTool'],
            use=[MiddlewareRef(name='enrich_tool_ctx')],
        ),
        context={'user_id': 'u-123'},
    )

    assert response.text == 'done'
    assert seen == [{'user_id': 'u-123', 'added_by_mw': True}]


@pytest.mark.asyncio
async def test_wrap_tool_custom_context_visible_to_generate_and_model_on_next_turn() -> None:
    """``wrap_tool`` mutations to ``custom_context`` appear in later ``wrap_generate`` / ``wrap_model`` calls."""
    generate_ctx: list[tuple[int, dict[str, object]]] = []
    model_ctx: list[dict[str, object]] = []

    ai = Genkit()

    @ai.middleware(name='enrich_and_track', description='enrich tool ctx and track hook visibility')
    class EnrichAndTrackMiddleware(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            generate_ctx.append((params.iteration, dict(ctx.custom_context)))
            return await next_fn(params, ctx)

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            model_ctx.append(dict(ctx.custom_context))
            return await next_fn(params, ctx)

        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            ctx.custom_context['added_by_mw'] = True
            return await next_fn(params, ctx)

    pm, _ = define_scripted_model(ai)

    @ai.tool(name='ctxTool')
    async def ctx_tool() -> str:
        return 'ok'

    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='ctxTool', input={}, ref='r1'))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    )

    caller_ctx = {'user_id': 'u-123'}
    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            tools=['ctxTool'],
            use=[MiddlewareRef(name='enrich_and_track')],
        ),
        context=caller_ctx,
    )

    assert response.text == 'done'
    assert caller_ctx == {'user_id': 'u-123'}
    assert generate_ctx == [
        (0, {'user_id': 'u-123'}),
        (1, {'user_id': 'u-123', 'added_by_mw': True}),
    ]
    assert model_ctx == [
        {'user_id': 'u-123'},
        {'user_id': 'u-123', 'added_by_mw': True},
    ]


@pytest.mark.asyncio
async def test_middleware_wrap_tool_interrupt_handled_as_interrupt_not_crash() -> None:
    """Interrupt raised by wrap_tool middleware is converted to an interrupt part.

    This is a regression test: before the fix, a middleware-raised Interrupt
    bypassed execute_tool_request's except block and propagated uncaught through
    asyncio.gather, crashing generation instead of surfacing as a tool interrupt.
    """
    from genkit._ai._tools import Interrupt

    ai = Genkit()

    @ai.middleware(name='interrupt_all', description='interrupt all tools')
    class InterruptingMiddleware(BaseMiddleware):
        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            raise Interrupt({'blocked': True})

    pm, _ = define_scripted_model(ai)

    @ai.tool(name='blockedTool')
    async def blocked_tool() -> str:
        return 'should not run'

    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='blockedTool', input={}, ref='r1'))],
            ),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('do it')])],
            tools=['blockedTool'],
            use=[MiddlewareRef(name='interrupt_all')],
        ),
    )
    assert response.finish_reason == FinishReason.INTERRUPTED
    assert response.finish_message == 'One or more tool calls resulted in interrupts.'
    assert response.error is None
    assert response.message is not None
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[0].text == 'do it'
    assert response.messages[1] == response.message
    interrupt_parts = [
        p for p in response.message.content if p.tool_request is not None and p.metadata and 'interrupt' in p.metadata
    ]
    assert len(interrupt_parts) == 1
    assert interrupt_parts[0].metadata is not None
    assert interrupt_parts[0].metadata['interrupt'] == {'blocked': True}


@pytest.mark.asyncio
async def test_middleware_contributed_tools_available_to_model() -> None:
    """Middleware.tools() returns Tool handles scoped to the generate call (child registry).

    The contributed tool is resolvable by the model during the call but must not
    appear in the root registry afterward — mirroring Go's Hooks.Tools + NewChild.
    """

    ai = Genkit()

    @ai.middleware(name='tool_provider_mw')
    class ToolProviderMiddleware(BaseMiddleware):
        """Middleware that contributes a tool dynamically per generate() call."""

        def tools(self, ctx: GenerateMiddlewareContext) -> list[Tool]:
            async def middleware_tool() -> str:
                """A tool injected by middleware."""
                return 'from_middleware_tool'

            return [tool(middleware_tool)]

    pm, _ = define_scripted_model(ai)

    # Turn 1: model calls the middleware-contributed tool
    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='middleware_tool', input={}, ref='r1'))],
            ),
        )
    )
    # Turn 2: model returns final answer after tool result
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            use=[MiddlewareRef(name='tool_provider_mw')],
        ),
    )
    assert _tool_output(response.messages[2]) == 'from_middleware_tool'
    assert response.text == 'done'

    # The contributed tool must NOT be visible in the root registry after the call.
    assert await ai.registry.resolve_action(ActionKind.TOOL, 'middleware_tool') is None


@pytest.mark.asyncio
async def test_middleware_tool_already_on_the_request_raises() -> None:
    """A middleware tool with the same name as tools= is a bad argument."""
    ai = Genkit(model='echoModel')
    define_echo_model(ai)

    @ai.tool(name='ping')
    async def ping() -> str:
        return 'pong'

    @ai.middleware(name='also_ping')
    class AlsoPing(BaseMiddleware):
        def tools(self, ctx: GenerateMiddlewareContext) -> list[Tool]:
            async def ping() -> str:
                return 'from_mw'

            return [tool(ping)]

    with pytest.raises(GenkitError, match="tool 'ping' is contributed by middleware") as raised:
        await ai.generate(prompt='hi', tools=['ping'], use=[AlsoPing()])
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in raised.value.original_message


@pytest.mark.asyncio
async def test_two_middleware_contributing_the_same_tool_raises() -> None:
    """Two hooks cannot contribute the same tool name on one call."""
    ai = Genkit(model='echoModel')
    define_echo_model(ai)

    def _ping_mw(name: str) -> type[BaseMiddleware]:
        @ai.middleware(name=name)
        class PingMw(BaseMiddleware):
            def tools(self, ctx: GenerateMiddlewareContext) -> list[Tool]:
                async def ping() -> str:
                    return name

                return [tool(ping)]

        return PingMw

    with pytest.raises(GenkitError, match="tool 'ping' is contributed by middleware") as raised:
        await ai.generate(prompt='hi', use=[_ping_mw('ping_a')(), _ping_mw('ping_b')()])
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT


@pytest.mark.asyncio
async def test_middleware_in_one_call_share_an_isolated_registry() -> None:
    """Middleware in the same generate() call share an isolated registry.

    This verifies:

    - **Cooperation:** Middleware A contributes a tool via ``tools()`` and
      middleware B resolves it through ``ctx.registry`` in the same call
      (proves both middleware see the same per-call child registry, so they
      can pass tools and other actions to one another).
    - **Isolation:** Anything middleware writes via ``ctx.registry`` does NOT
      survive the call (proves writes are auto-cleaned and cannot leak into the
      root registry or across concurrent generate() calls).
    """
    seen_by_b: list[str] = []
    ai = Genkit()

    @ai.middleware(name='provider_mw')
    class ProviderMW(BaseMiddleware):
        def tools(self, ctx: GenerateMiddlewareContext) -> list[Tool]:
            async def shared_tool() -> str:
                """Shared by all middleware in the call."""
                return 'shared_ok'

            return [tool(shared_tool)]

    @ai.middleware(name='looker_mw')
    class LookerMW(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            # Resolve the tool ProviderMW just contributed — only works if
            # both middleware share the same per-call registry scope.
            tool = await ctx.ai.registry.resolve_action(ActionKind.TOOL, 'shared_tool')
            if tool is not None:
                seen_by_b.append(tool.name)
            # Also exercise the write path: anything we register through
            # ctx.ai.registry must not survive the call.
            scratch = Registry()

            async def leaky_tool() -> str:
                """Should not survive the call."""
                return 'nope'

            leak = define_tool(scratch, leaky_tool, name='leaky_tool').action()
            ctx.ai.registry.register_action_from_instance(leak)
            return await next_fn(params, ctx)

    pm, _ = define_scripted_model(ai)
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            use=[
                MiddlewareRef(name='provider_mw'),
                MiddlewareRef(name='looker_mw'),
            ],
        ),
    )
    assert response.text == 'ok'
    assert seen_by_b == ['shared_tool'], f'looker middleware should have resolved shared_tool, saw: {seen_by_b}'
    # Neither tool may leak into the root registry after the call ends.
    assert await ai.registry.resolve_action(ActionKind.TOOL, 'shared_tool') is None
    assert await ai.registry.resolve_action(ActionKind.TOOL, 'leaky_tool') is None


@pytest.mark.asyncio
async def test_queue_drain_streams_each_message_at_one_index() -> None:
    """Queued tool middleware messages stream as exactly one chunk per message.

    Regression: the old queue-drain path called ``make_chunk(USER, ...)`` for
    each queued message AND then did ``message_index += 1``. ``make_chunk``
    *also* advanced the index (role flip from MODEL to USER), so each queued
    message bumped the counter twice — leaving a hole in the stream sequence.
    The fix emits queued chunks directly and increments once per message.
    """

    ai = Genkit()

    @ai.middleware(name='enqueuing_mw')
    class EnqueuingMW(BaseMiddleware):
        """After each tool call, queue an extra USER message for the next turn."""

        def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
            super().__init__(**kwargs)
            self._queued: list[Message] = []

        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            if self._queued:
                queued = list(self._queued)
                self._queued.clear()
                if ctx.on_chunk:
                    for msg in queued:
                        ctx.send_chunk(
                            ModelResponseChunk(
                                role=msg.role,
                                content=msg.content,
                                index=params.message_index,
                            )
                        )
                options = params.options.model_copy()
                options.messages = [*options.messages, *queued]
                params = params.model_copy(update={'options': options})
            return await next_fn(params, ctx)

        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            result = await next_fn(params, ctx)
            self._queued.append(Message(role=Role.USER, content=[Part.from_text('extra-context')]))
            return result

    pm, _ = define_scripted_model(ai)

    @ai.tool(name='trigger')
    async def trigger() -> str:
        return 'triggered'

    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='trigger', input={}, ref='r1'))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('final')]),
        )
    )

    streamed: list[ModelResponseChunk] = []
    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('go')])],
            tools=['trigger'],
            use=[MiddlewareRef(name='enqueuing_mw')],
        ),
        on_chunk=streamed.append,
    )
    assert response.text == 'final'

    user_chunks = [c for c in streamed if c.role == Role.USER]
    assert len(user_chunks) == 1, (
        f'expected exactly one streamed user chunk for the queued message, saw '
        f'{[(c.role, c.index) for c in user_chunks]}'
    )
    indices = [c.index or 0 for c in streamed]
    assert indices == sorted(indices), f'indices not monotonic: {indices}'


@pytest.mark.asyncio
async def test_restart_path_routes_through_wrap_tool_middleware() -> None:
    """Restarting a tool via ``resume_restart`` must invoke ``wrap_tool`` middleware.

    Regression: ``resolve_resumed_tool`` used to call
    ``run_tool_after_restart`` directly, skipping the middleware chain. That
    silently bypassed ToolApproval / Filesystem / etc. on every restart.
    """
    invocations: list[str] = []
    ai = Genkit()

    @ai.middleware(name='recording_mw')
    class RecordingMW(BaseMiddleware):
        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            invocations.append(params.tool.name)
            return await next_fn(params, ctx)

    pm, _ = define_scripted_model(ai)

    @ai.tool(name='approveMe')
    async def approve_me() -> str:
        return 'approved'

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('final')]),
        )
    )

    interrupt_part = Part.from_tool_request(name='approveMe', input={}, ref='r1', metadata={'interrupt': True})

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[
                Message(role=Role.USER, content=[Part.from_text('do it')]),
                Message(
                    role=Role.MODEL,
                    content=[interrupt_part],
                ),
            ],
            tools=['approveMe'],
            use=[MiddlewareRef(name='recording_mw')],
            resume=Resume(
                restart=[
                    Part.from_tool_request(
                        name='approveMe',
                        input={},
                        ref='r1',
                        metadata={'resumed': {'tool_approved': True}},
                    )
                ],
            ),
        ),
    )
    assert response.text == 'final'
    assert invocations == ['approveMe'], f'expected wrap_tool to fire once on restart, saw: {invocations}'


@pytest.mark.asyncio
async def test_generate_restart_without_approval_returns_interrupted() -> None:
    """Restarting without approval pauses again; they can approve on the next call."""
    ai = Genkit(model='scriptedModel')

    @ai.middleware(name='approval_mw')
    class ApprovalMW(BaseMiddleware):
        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            metadata = params.tool_request_part.metadata or {}
            resumed = metadata.get('resumed')
            if isinstance(resumed, dict) and resumed.get('toolApproved'):
                return await next_fn(params, ctx)
            raise Interrupt({'message': f'Tool not in approved list: {params.tool.name}'})

    pm, _ = define_scripted_model(ai)

    @ai.tool(name='sensitiveTool')
    async def sensitive_tool() -> str:
        return 'done'

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('final')]),
        )
    ]

    interrupt_part = Part.from_tool_request(name='sensitiveTool', input={}, ref='r1', metadata={'interrupt': True})
    history = [
        Message(role=Role.USER, content=[Part.from_text('do it')]),
        Message(role=Role.MODEL, content=[interrupt_part]),
    ]

    response = await ai.generate(
        messages=history,
        tools=['sensitiveTool'],
        use=[ApprovalMW()],
        resume_restart=interrupt_part.restart(),
    )
    assert response.finish_reason == FinishReason.INTERRUPTED
    assert response.finish_message == 'One or more tool calls resulted in interrupts.'
    assert response.error is None
    assert response.message is not None
    assert response.messages[-1] == response.message
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.interrupts
    assert response.interrupts[0].metadata is not None
    assert response.interrupts[0].metadata['interrupt'] == {
        'message': 'Tool not in approved list: sensitiveTool',
    }

    approved = await ai.generate(
        messages=response.messages,
        tools=['sensitiveTool'],
        use=[ApprovalMW()],
        resume_restart=response.interrupts[0].restart(
            resumed_metadata={'toolApproved': True},
        ),
    )
    assert approved.finish_reason == FinishReason.STOP
    assert approved.text == 'final'
    assert approved.error is None
    assert approved.message is not None
    assert approved.messages[-1] == approved.message
    assert [m.role for m in approved.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(approved.messages[2]) == 'done'


@pytest.mark.asyncio
async def test_generate_restart_interrupt_returns_interrupted() -> None:
    """A tool that pauses again on restart returns INTERRUPTED they can answer."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='hold')
    async def hold() -> str:
        raise Interrupt({'hold': True})

    pm.responses = [_model_calls_tool(name='hold', ref='1')]

    first = await ai.generate(prompt='hi', tools=['hold'])
    assert first.finish_reason == FinishReason.INTERRUPTED
    assert first.interrupts

    response = await ai.generate(
        messages=first.messages,
        tools=['hold'],
        resume_restart=first.interrupts[0].restart(),
    )
    assert response.finish_reason == FinishReason.INTERRUPTED
    assert response.finish_message == 'One or more tool calls resulted in interrupts.'
    assert response.error is None
    assert response.message is not None
    assert response.messages[-1] == response.message
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.interrupts
    assert response.interrupts[0].metadata is not None
    assert response.interrupts[0].metadata['interrupt'] == {'hold': True}


@pytest.mark.asyncio
async def test_parallel_tool_requests_all_complete() -> None:
    """Multiple tool requests in one model turn are resolved together (asyncio.gather); all succeed."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='tool_a')
    async def tool_a() -> str:
        return 'a_ok'

    @ai.tool(name='tool_b')
    async def tool_b() -> str:
        return 'b_ok'

    @ai.tool(name='tool_c')
    async def tool_c() -> str:
        return 'c_ok'

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part.from_text('call three'),
                    Part(tool_request=ToolRequest(name='tool_a', ref='ref-a', input={})),
                    Part(tool_request=ToolRequest(name='tool_b', ref='ref-b', input={})),
                    Part(tool_request=ToolRequest(name='tool_c', ref='ref-c', input={})),
                ],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('after_tools')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[
                Message(role=Role.USER, content=[Part.from_text('hi')]),
            ],
            tools=['tool_a', 'tool_b', 'tool_c'],
        ),
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.finish_message is None
    assert response.text == 'after_tools'
    assert response.error is None
    assert response.message is not None
    assert response.message.text == 'after_tools'
    assert [message.role for message in response.messages] == [
        Role.USER,
        Role.MODEL,
        Role.TOOL,
        Role.MODEL,
    ]
    assert response.messages[-1] == response.message
    tool_responses = [part.tool_response for part in response.messages[2].content if part.tool_response is not None]
    assert [(part.name, part.ref, part.output) for part in tool_responses] == [
        ('tool_a', 'ref-a', 'a_ok'),
        ('tool_b', 'ref-b', 'b_ok'),
        ('tool_c', 'ref-c', 'c_ok'),
    ]


@pytest.mark.asyncio
async def test_generate_inline_tool_without_root_registration() -> None:
    """Passing a Tool from another registry into ``ai.generate`` resolves for that call only."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    other = Registry()

    async def inline_yell() -> str:
        return 'HEY'

    inline_tool = define_tool(other, inline_yell, name='inline_yell')

    assert await ai.registry.resolve_action(ActionKind.TOOL, 'inline_yell') is None

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(tool_request=ToolRequest(name='inline_yell', ref='ref-y', input={})),
                ],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('after_inline')]),
        )
    )

    response = await ai.generate(
        model='scriptedModel',
        prompt='call it',
        tools=[inline_tool],
    )

    assert response.text == 'after_inline'
    assert await ai.registry.resolve_action(ActionKind.TOOL, 'inline_yell') is None


@pytest.mark.asyncio
async def test_parallel_tool_requests_one_interrupt_keeps_pending_output_for_others(
    setup_test: tuple[Genkit, ScriptedModel],
) -> None:
    """With asyncio.gather in resolve_tool_requests: one interrupt still records pendingOutput for others."""
    ai, pm = setup_test

    @ai.tool(name='tool_a')
    async def tool_a() -> str:
        return 'a_ok'

    @ai.tool(name='tool_b')
    async def tool_b() -> None:
        raise Interrupt({'stop': True})

    @ai.tool(name='tool_c')
    async def tool_c() -> str:
        return 'c_ok'

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part.from_text('call three'),
                    Part(tool_request=ToolRequest(name='tool_a', ref='ref-a', input={})),
                    Part(tool_request=ToolRequest(name='tool_b', ref='ref-b', input={})),
                    Part(tool_request=ToolRequest(name='tool_c', ref='ref-c', input={})),
                ],
            ),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[
                Message(role=Role.USER, content=[Part.from_text('hi')]),
            ],
            tools=['tool_a', 'tool_b', 'tool_c'],
        ),
    )

    assert response.finish_reason == FinishReason.INTERRUPTED
    assert response.message is not None
    parts = response.message.content
    assert len(parts) == 4
    assert parts[0].text == 'call three'
    assert parts[1].tool_request is not None
    assert parts[2].tool_request is not None
    assert parts[3].tool_request is not None
    assert parts[1].metadata and parts[1].metadata.get('pendingOutput') == 'a_ok'
    assert parts[2].metadata and parts[2].metadata.get('interrupt') == {'stop': True}
    assert parts[3].metadata and parts[3].metadata.get('pendingOutput') == 'c_ok'


@pytest.mark.asyncio
async def test_generate_and_model_middleware_execution_order() -> None:
    """wrap_generate and wrap_model run in the correct nested order.

    Matches JS: 'runs generate and model middleware in the correct order'.
    Expected: generateBefore → modelBefore → modelExecution → modelAfter → generateAfter
    """
    execution_order: list[str] = []
    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('response')]),
        )
    )

    @ai.middleware(name='order_mw')
    class OrderMiddleware(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            execution_order.append('generateBefore')
            resp = await next_fn(params, ctx)
            execution_order.append('generateAfter')
            return resp

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            execution_order.append('modelBefore')
            resp = await next_fn(params, ctx)
            execution_order.append('modelAfter')
            return resp

    # The scripted model appends to execution_order when called.
    pm.responses.copy()
    pm.responses.clear()

    def model_side_effect(request: ModelRequest) -> ModelResponse:
        execution_order.append('modelExecution')
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('response')]),
        )

    pm.response_cb = model_side_effect
    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            use=[MiddlewareRef(name='order_mw')],
        ),
    )

    assert response.text == 'response'
    assert execution_order == [
        'generateBefore',
        'modelBefore',
        'modelExecution',
        'modelAfter',
        'generateAfter',
    ]


@pytest.mark.asyncio
async def test_generate_model_tool_middleware_ordering_across_turns() -> None:
    """All three hooks (generate, model, tool) fire in correct order across a two-turn tool flow.

    Matches JS: 'runs tool middleware correctly'.
    Turn 1: model returns a tool request → tool executes
    Turn 2: model returns final text response
    Expected order mirrors the JS assertion.
    """
    execution_order: list[str] = []
    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='orderTool')
    async def order_tool() -> str:
        execution_order.append('toolExecution')
        return 'tool result'

    turn = 0

    def model_side_effect(request: ModelRequest) -> ModelResponse:
        nonlocal turn
        turn += 1
        execution_order.append('modelExecution')
        if turn == 1:
            return ModelResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(tool_request=ToolRequest(name='orderTool', input={}, ref='r1'))],
                ),
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('final response')]),
        )

    pm.response_cb = model_side_effect

    # The middleware tracks a turn counter internally, matching JS's `turnCount`.
    turn_counter: list[int] = [0]

    @ai.middleware(name='full_order_mw')
    class FullOrderMiddleware(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            turn_counter[0] += 1
            t = turn_counter[0]
            execution_order.append(f'generateBefore-{t}')
            resp = await next_fn(params, ctx)
            execution_order.append(f'generateAfter-{t}')
            return resp

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            execution_order.append(f'modelBefore-{turn_counter[0]}')
            resp = await next_fn(params, ctx)
            execution_order.append(f'modelAfter-{turn_counter[0]}')
            return resp

        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            execution_order.append(f'toolBefore-{turn_counter[0]}')
            resp = await next_fn(params, ctx)
            execution_order.append(f'toolAfter-{turn_counter[0]}')
            return resp

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            tools=['orderTool'],
            use=[MiddlewareRef(name='full_order_mw')],
        ),
    )

    assert response.text == 'final response'
    assert execution_order == [
        'generateBefore-1',
        'modelBefore-1',
        'modelExecution',
        'modelAfter-1',
        'toolBefore-1',
        'toolExecution',
        'toolAfter-1',
        'generateBefore-2',
        'modelBefore-2',
        'modelExecution',
        'modelAfter-2',
        'generateAfter-2',
        'generateAfter-1',
    ]


@pytest.mark.asyncio
async def test_middleware_contributed_tool_resolvable_during_restart() -> None:
    """Tools injected by middleware.tools() are resolvable during a resume/restart flow.

    Matches JS: 'should resolve tools injected by middleware during restarts'.
    Scenario: middleware contributes a tool, that tool gets interrupted, then
    resume.restart can still find and execute it through the middleware pipeline.
    """
    ai = Genkit()

    @ai.middleware(name='tool_injector_mw')
    class ToolInjectorMiddleware(BaseMiddleware):
        def tools(self, ctx: GenerateMiddlewareContext) -> list:
            async def injected_tool() -> str:
                """A tool contributed by middleware."""
                return 'injected_success'

            return [tool(injected_tool, name='injectedTool')]

    pm, _ = define_scripted_model(ai)

    # The model will be called after restart — return a final response.
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done after restart')]),
        )
    )

    # Simulate: model previously called injectedTool, it was interrupted,
    # now we resume with restart.
    interrupt_part = Part.from_tool_request(name='injectedTool', input={}, ref='r1', metadata={'interrupt': True})

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[
                Message(role=Role.USER, content=[Part.from_text('do it')]),
                Message(
                    role=Role.MODEL,
                    content=[interrupt_part],
                ),
            ],
            use=[MiddlewareRef(name='tool_injector_mw')],
            resume=Resume(
                restart=[
                    Part.from_tool_request(name='injectedTool', input={}, ref='r1'),
                ],
            ),
        ),
    )
    assert response.text == 'done after restart'


##########################################################################
# run tests from /tests/specs/generate.yaml
##########################################################################

specs = []
spec_path = pathlib.Path(__file__).parent / '../../../../../../tests/specs/generate.yaml'
with spec_path.resolve().open() as stream:
    tests_spec = yaml.safe_load(stream)
    specs = tests_spec['tests']


@pytest.mark.parametrize(
    'spec',
    specs,
)
@pytest.mark.asyncio
async def test_generate_action_spec(spec: dict[str, Any]) -> None:
    """Run tests based on external generate action specifications."""
    ai = Genkit()

    model_name = spec.get('input', {}).get('model', 'scriptedModel')
    pm, _ = define_scripted_model(ai, name=model_name)

    @ai.tool(name='testTool')
    async def test_tool() -> object:
        """description"""  # noqa: D403, D415
        return 'tool called'

    if 'modelResponses' in spec:
        pm.responses = [TypeAdapter(ModelResponse).validate_python(resp) for resp in spec['modelResponses']]

    if 'streamChunks' in spec:
        pm.chunks = []
        for stream_chunks in spec['streamChunks']:
            converted = []
            if stream_chunks:
                for chunk in stream_chunks:
                    converted.append(TypeAdapter(ModelResponseChunk).validate_python(chunk))
            pm.chunks.append(converted)

    action = await ai.registry.resolve_action(kind=ActionKind.UTIL, name='generate')
    assert action is not None

    response = None
    chunks: list[ModelResponseChunk] | None = None
    if spec.get('stream'):
        chunks = []
        captured_chunks = chunks  # Capture list reference for closure

        def on_chunk(chunk: ModelResponseChunk) -> None:
            captured_chunks.append(chunk)

        action_response = await action.run(
            TypeAdapter(GenerateActionOptions).validate_python(spec['input']),  # type: ignore[arg-type]
            on_chunk=on_chunk,  # type: ignore[misc]
        )
        response = action_response.response
    else:
        action_response = await action.run(
            TypeAdapter(GenerateActionOptions).validate_python(spec['input']),
        )
        response = action_response.response

    if 'expectChunks' in spec:
        got = clean_schema(chunks)
        want = clean_schema(spec['expectChunks'])
        assert isinstance(got, list) and isinstance(want, list)
        if not is_equal_lists(got, want):
            raise AssertionError(
                f'{_to_json(got, indent=2)}\n\nis not equal to expected:\n\n{_to_json(want, indent=2)}'
            )

    if 'expectResponse' in spec:
        got = clean_schema(_to_dict(response))
        want = clean_schema(spec['expectResponse'])
        if got != want:
            raise AssertionError(
                f'{_to_json(got, indent=2)}\n\nis not equal to expected:\n\n{_to_json(want, indent=2)}'
            )


def is_equal_lists(a: Sequence[object], b: Sequence[object]) -> bool:
    """Deep compare two lists of actions."""
    if len(a) != len(b):
        return False

    return all(_to_dict(a[i]) == _to_dict(b[i]) for i in range(len(a)))


primitives = (bool, str, int, float, type(None))


def is_primitive(obj: object) -> bool:
    """Check if an object is a primitive type."""
    return isinstance(obj, primitives)


def clean_schema(d: object) -> object:
    """Remove $schema keys and other non-relevant parts from a dict recursively."""
    if is_primitive(d):
        return d
    if isinstance(d, dict):
        out: dict[str, object] = {}
        d_dict = cast(dict[str, object], d)
        for key in d_dict:
            # Skip $schema and latencyMs (dynamic value that varies between runs)
            if key not in ('$schema', 'latencyMs'):
                out[key] = clean_schema(d_dict[key])
        return out
    elif isinstance(d, (list, tuple)):
        return [clean_schema(i) for i in d]
    else:
        return d


def test_chunk_accumulator_make_kwargs_only() -> None:
    """``ChunkAccumulator.make`` requires keyword-only arguments."""
    acc = ChunkAccumulator(message_index=0, formatter=None)
    raw_chunk = ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('hi')])

    with pytest.raises(TypeError):
        acc.make(Role.MODEL, raw_chunk)  # type: ignore[misc]

    wrapped = acc.make(role=Role.MODEL, chunk=raw_chunk)
    assert wrapped.index == 0


@pytest.mark.asyncio
async def test_wrap_generate_middleware_cannot_inject_a_tool_after_the_door() -> None:
    """Tools on the generate() call are the ones the model sees. A hook cannot add one."""
    ai = Genkit()
    captured_tool_names: list[list[str]] = []

    @ai.tool(name='dynamic_mw_tool')
    async def dynamic_mw_tool() -> str:
        return 'ok'

    class DynCfg(BaseModel):
        pass

    @ai.middleware(name='dynamic_tool_mw')
    class DynamicToolMiddleware(BaseMiddleware[DynCfg]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            new_opts = params.options.model_copy()
            tools = list(new_opts.tools or [])
            tools.append('dynamic_mw_tool')
            new_opts.tools = tools
            return await next_fn(params.model_copy(update={'options': new_opts}), ctx)

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            names = [t.name for t in (params.request.tools or [])]
            captured_tool_names.append(names)
            return await next_fn(params, ctx)

    define_echo_model(ai)

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        use=[DynamicToolMiddleware()],
    )
    assert response.text == '[ECHO] user: "hi"'
    assert captured_tool_names == [[]]


@pytest.mark.asyncio
async def test_generate_with_empty_messages_lets_the_model_speak_first(
    setup_test: tuple[Genkit, ScriptedModel],
) -> None:
    """An empty conversation is a valid start; the model writes the first turn."""
    ai, pm = setup_test
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('hello')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(model='scriptedModel', messages=[]),
    )
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'hello'
    assert response.error is None
    assert response.message is not None
    assert [m.role for m in response.messages] == [Role.MODEL]
    assert pm.request_count == 1
    assert pm.last_request is not None
    assert pm.last_request.messages == []


@pytest.mark.asyncio
async def test_generate_without_prompt_lets_the_model_speak_first() -> None:
    """Omitting prompt and messages is the same empty start."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('hello')]),
        )
    )

    response = await ai.generate()
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'hello'
    assert response.error is None
    assert [m.role for m in response.messages] == [Role.MODEL]
    assert pm.last_request is not None
    assert pm.last_request.messages == []


@pytest.mark.asyncio
async def test_generate_rejects_negative_max_turns() -> None:
    """A negative cap is not a generation; they get the argument back before a request."""
    ai = Genkit(model='scriptedModel')
    define_scripted_model(ai)

    with pytest.raises(GenkitError, match='max turns cannot be negative, got -1') as raised:
        await ai.generate(prompt='hi', max_turns=-1)
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in raised.value.original_message


@pytest.mark.asyncio
async def test_generate_max_turns_zero_drops_first_tool_round() -> None:
    """A cap of zero is zero tool rounds, not the omitted default."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]

    response = await ai.generate(prompt='hi', tools=['lookup'], max_turns=0)

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Exceeded maximum tool call iterations (0)'
    assert response.message is None
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'


@pytest.mark.asyncio
async def test_generate_returns_on_blocked_finish() -> None:
    """Blocked is a refusal that still returns so the text stays on the response."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.BLOCKED,
            finish_message='safety',
            message=Message(role=Role.MODEL, content=[Part.from_text('nope')]),
        )
    ]

    response = await ai.generate(prompt='hi')
    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'safety'
    assert response.text == 'nope'
    assert response.output is None
    assert response.error is None
    assert response.message is not None
    assert response.message.text == 'nope'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[0].text == 'hi'
    assert response.messages[1] == response.message


@pytest.mark.asyncio
async def test_generate_blocked_without_message_keeps_prior_history() -> None:
    """A provider can block without returning refusal content."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.BLOCKED,
            finish_message='safety',
        )
    ]

    response = await ai.generate(prompt='hi')

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'safety'
    assert response.message is None
    assert response.error is None
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'


@pytest.mark.asyncio
async def test_generate_aborted_with_content_preserves_terminal_message() -> None:
    """A model response aborted with partial content retains that model message."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.ABORTED,
            finish_message='interrupted by provider',
            message=Message(role=Role.MODEL, content=[Part.from_text('partial text')]),
        )
    ]

    response = await ai.generate(prompt='hi')

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'interrupted by provider'
    assert response.text == 'partial text'
    assert response.message is not None
    assert response.message.text == 'partial text'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[-1] == response.message


@pytest.mark.asyncio
async def test_generate_aborted_without_content_preserves_prior_history() -> None:
    """A model response aborted with no content preserves prior history and sets message=None."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.ABORTED,
            finish_message='interrupted by provider',
        )
    ]

    response = await ai.generate(prompt='hi')

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'interrupted by provider'
    assert response.message is None
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'


@pytest.mark.asyncio
async def test_generate_schema_failure_preserves_history() -> None:
    """A schema parsing failure preserves the prompt and the raw terminal model reply."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Recipe(BaseModel):
        title: str

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('not json')]),
        )
    ]

    response = await ai.generate(prompt='give me a recipe', output_schema=Recipe)
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'not json'
    assert response.output is None
    assert response.finish_message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert 'not valid JSON' in response.error.message
    assert response.error.details == {'reason': 'INVALID_OUTPUT'}
    assert response.message is not None
    assert response.message.text == 'not json'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[0].text == 'give me a recipe'
    assert response.messages[1] == response.message


def _model_calls_tool(*, name: str, ref: str, input: dict[str, object] | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name=name, input=input if input is not None else {}, ref=ref))],
        ),
    )


def _tool_output(message: Message) -> object:
    part = message.content[0]
    assert part.tool_response is not None
    return part.tool_response.output


@pytest.mark.asyncio
async def test_return_tool_requests_keeps_model_message_without_running_tool() -> None:
    """The caller can deliberately take responsibility for an open tool request."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    called = False

    @ai.tool(name='lookup')
    async def lookup() -> str:
        nonlocal called
        called = True
        return '72F'

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]

    response = await ai.generate(
        prompt='let me run it',
        tools=['lookup'],
        return_tool_requests=True,
    )

    assert called is False
    assert response.finish_reason == FinishReason.STOP
    assert response.finish_message is None
    assert response.error is None
    assert response.message is not None
    assert response.message.tool_requests[0].tool_request == ToolRequest(name='lookup', input={}, ref='r1')
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[0].text == 'let me run it'
    assert response.messages[1] == response.message


@pytest.mark.asyncio
async def test_max_turns_budget_is_spent_entirely_on_tool_rounds() -> None:
    """max_turns counts tool rounds, and every one of them may complete.

    A run that uses the whole budget still gets its final answer: the cap
    allows max_turns tool rounds plus the model call that replies, so the
    history comes back as max_turns model/tool pairs followed by the answer.
    """
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    tool_calls = 0

    @ai.tool(name='step')
    async def step() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return 'ok'

    pm.responses = [
        *[_model_calls_tool(name='step', ref=str(index)) for index in range(DEFAULT_MAX_TURNS)],
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        ),
    ]

    response = await ai.generate(prompt='finish the work', tools=['step'])

    # Each round contributes a MODEL then a TOOL message after the leading USER
    # message, so the model turns land on odd indices and their outputs on even.
    model_turns = range(1, 2 * DEFAULT_MAX_TURNS + 1, 2)
    tool_turns = range(2, 2 * DEFAULT_MAX_TURNS + 2, 2)

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'done'
    assert response.error is None
    assert tool_calls == DEFAULT_MAX_TURNS
    assert pm.request_count == DEFAULT_MAX_TURNS + 1
    assert [message.role for message in response.messages] == [
        Role.USER,
        *[role for _ in range(DEFAULT_MAX_TURNS) for role in (Role.MODEL, Role.TOOL)],
        Role.MODEL,
    ]
    assert [_tool_request(response.messages[index]).ref for index in model_turns] == [
        str(index) for index in range(DEFAULT_MAX_TURNS)
    ]
    assert [_tool_output(response.messages[index]) for index in tool_turns] == ['ok'] * DEFAULT_MAX_TURNS
    assert response.messages[-1] == response.message


@pytest.mark.asyncio
async def test_max_turns_exceeded_leaves_the_unanswered_round_out_of_history() -> None:
    """Going past the cap drops the round that never finished, so the history stays resendable.

    The model asked for one more tool call than the budget allows. That last
    request was never answered, so it is left out of response.messages -- what
    you get back is the completed rounds only, which you can send again as-is.
    """
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    tool_calls = 0

    @ai.tool(name='step')
    async def step() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return 'ok'

    pm.responses = [_model_calls_tool(name='step', ref=str(index)) for index in range(DEFAULT_MAX_TURNS + 1)]

    response = await ai.generate(prompt='keep going', tools=['step'])

    model_turns = range(1, 2 * DEFAULT_MAX_TURNS + 1, 2)
    tool_turns = range(2, 2 * DEFAULT_MAX_TURNS + 2, 2)

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == f'Exceeded maximum tool call iterations ({DEFAULT_MAX_TURNS})'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'ABORTED'
    assert response.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert response.error.details == {'reason': 'MAX_TURNS_EXCEEDED'}
    assert response.error.message == response.finish_message
    assert tool_calls == DEFAULT_MAX_TURNS
    assert pm.request_count == DEFAULT_MAX_TURNS + 1
    assert [message.role for message in response.messages] == [
        Role.USER,
        *[role for _ in range(DEFAULT_MAX_TURNS) for role in (Role.MODEL, Role.TOOL)],
    ]
    assert [_tool_request(response.messages[index]).ref for index in model_turns] == [
        str(index) for index in range(DEFAULT_MAX_TURNS)
    ]
    assert [_tool_output(response.messages[index]) for index in tool_turns] == ['ok'] * DEFAULT_MAX_TURNS


@pytest.mark.asyncio
async def test_closed_history_after_tool_turn_keeps_intermediate_messages() -> None:
    """A dead turn after a tool turn still has that tool request and reply on .messages."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Recipe(BaseModel):
        title: str

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('not json')]),
        ),
    ]

    response = await ai.generate(
        prompt='give me a recipe',
        output_schema=Recipe,
        output_instructions=False,
        tools=['lookup'],
    )
    assert response.finish_reason == FinishReason.STOP
    assert response.finish_message is None
    assert response.text == 'not json'
    assert response.output is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert response.error.details == {'reason': 'INVALID_OUTPUT'}
    assert 'not valid JSON' in response.error.message
    assert response.message is not None
    assert response.message.text == 'not json'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_request(response.messages[1]).name == 'lookup'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.messages[3].text == 'not json'


@pytest.mark.asyncio
async def test_generate_blocked_after_tool_turn_keeps_prior_history() -> None:
    """A blocked model response without content after a tool turn preserves completed tool rounds."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.BLOCKED,
            finish_message='safety',
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'])

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'safety'
    assert response.message is None
    assert response.error is None
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.messages[-1].role == Role.TOOL


@pytest.mark.asyncio
async def test_generate_blocked_with_content_after_tool_turn_preserves_refusal() -> None:
    """A blocked model response with refusal content after a tool turn includes the terminal model reply."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.BLOCKED,
            finish_message='safety',
            message=Message(role=Role.MODEL, content=[Part.from_text('cannot continue')]),
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'])

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'safety'
    assert response.text == 'cannot continue'
    assert response.error is None
    assert response.message is not None
    assert response.message.text == 'cannot continue'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.messages[-1] == response.message


@pytest.mark.asyncio
async def test_generate_aborted_after_tool_turn_keeps_prior_history() -> None:
    """A provider abort without content after a tool turn keeps the closed rounds."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.ABORTED,
            finish_message='interrupted by provider',
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'])

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'interrupted by provider'
    assert response.message is None
    assert response.error is None
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.messages[-1].role == Role.TOOL


@pytest.mark.asyncio
async def test_provider_abort_with_text_after_tool_keeps_the_reply() -> None:
    """After lookup returns, a provider abort that includes text keeps that model reply."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.ABORTED,
            finish_message='interrupted by provider',
            message=Message(role=Role.MODEL, content=[Part.from_text('partial text')]),
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'])

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'interrupted by provider'
    assert response.text == 'partial text'
    assert response.error is None
    assert response.message is not None
    assert response.message.text == 'partial text'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.messages[-1] == response.message


def _tool_request(message: Message) -> ToolRequest:
    assert message.tool_requests
    req = message.tool_requests[0].tool_request
    assert req is not None
    return req


@pytest.mark.asyncio
async def test_generate_successful_tool_turn_preserves_full_history() -> None:
    """A completed tool turn followed by a successful reply contains all 4 messages."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('the weather is 72F')]),
        ),
    ]

    response = await ai.generate(prompt='weather?', tools=['lookup'])

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'the weather is 72F'
    assert response.message is not None
    assert response.message.text == 'the weather is 72F'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert response.messages[0].text == 'weather?'
    assert _tool_request(response.messages[1]).name == 'lookup'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.messages[-1] == response.message


@pytest.mark.asyncio
async def test_generate_completes_within_max_turns_and_preserves_all_rounds() -> None:
    """Multi-round tool execution completing within max_turns returns all message rounds."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='step_one')
    async def step_one() -> str:
        return 'done 1'

    @ai.tool(name='step_two')
    async def step_two() -> str:
        return 'done 2'

    pm.responses = [
        _model_calls_tool(name='step_one', ref='r1'),
        _model_calls_tool(name='step_two', ref='r2'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('all finished')]),
        ),
    ]

    response = await ai.generate(prompt='run both', tools=['step_one', 'step_two'], max_turns=3)

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'all finished'
    assert response.message is not None
    assert [message.role for message in response.messages] == [
        Role.USER,
        Role.MODEL,
        Role.TOOL,
        Role.MODEL,
        Role.TOOL,
        Role.MODEL,
    ]
    assert response.messages[0].text == 'run both'
    assert _tool_request(response.messages[1]).name == 'step_one'
    assert _tool_output(response.messages[2]) == 'done 1'
    assert _tool_request(response.messages[3]).name == 'step_two'
    assert _tool_output(response.messages[4]) == 'done 2'
    assert response.messages[-1] == response.message


@pytest.mark.asyncio
async def test_generate_schema_failure_after_tool_turn_preserves_history() -> None:
    """Output schema failure after a tool turn preserves completed tool rounds and terminal reply."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Recipe(BaseModel):
        title: str

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('not json')]),
        ),
    ]

    response = await ai.generate(
        prompt='give me a recipe',
        output_schema=Recipe,
        output_instructions=False,
        tools=['lookup'],
    )

    # Bad JSON is not a dead turn: the turn closed, the model text stands.
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'not json'
    assert response.output is None
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert response.message is not None
    assert response.message.text == 'not json'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_request(response.messages[1]).name == 'lookup'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.messages[-1] == response.message


@pytest.mark.asyncio
async def test_max_turns_after_tool_turn_drops_unanswered_request() -> None:
    """Hitting max tool turns keeps closed rounds and drops the unanswered call."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        _model_calls_tool(name='lookup', ref='r2'),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'], max_turns=1)

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message is not None
    assert 'maximum tool call iterations' in response.finish_message
    assert response.message is None
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.messages[-1].role == Role.TOOL


@pytest.mark.asyncio
async def test_unknown_tool_on_first_turn_drops_unanswered_request() -> None:
    """An unknown tool on the first turn leaves only the user prompt to resend."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [_model_calls_tool(name='ghost', ref='r1')]

    response = await ai.generate(prompt='keep going', tools=['lookup'])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'Tool ghost not found'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'NOT_FOUND'
    assert response.error.reason is RuntimeErrorReason.TOOL_NOT_FOUND
    assert response.error.details == {'reason': 'TOOL_NOT_FOUND'}
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'keep going'


@pytest.mark.asyncio
async def test_unknown_tool_after_tool_turn_drops_unanswered_request() -> None:
    """An unknown tool drops the unanswered request and keeps closed rounds."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        _model_calls_tool(name='ghost', ref='r2'),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'Tool ghost not found'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'NOT_FOUND'
    assert response.error.reason is RuntimeErrorReason.TOOL_NOT_FOUND
    assert response.error.details == {'reason': 'TOOL_NOT_FOUND'}
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.messages[-1].role == Role.TOOL


def _rewrite_tools_middleware(mutate: Callable[[list[str]], list[str]]) -> BaseMiddleware:
    """wrap_generate that rewrites options.tools after the door has already resolved."""

    class EmptyMwCfg(BaseModel):
        pass

    class RewriteTools(BaseMiddleware[EmptyMwCfg]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            new_opts = params.options.model_copy()
            new_opts.tools = mutate(list(new_opts.tools or []))
            return await next_fn(params.model_copy(update={'options': new_opts}), ctx)

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await next_fn(params, ctx)

    return RewriteTools()


def _model_says(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
    )


@pytest.mark.asyncio
async def test_wrap_generate_injected_tool_does_not_run() -> None:
    """A hook-added name is not a tool they passed. The model asking for it is unknown."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran: list[str] = []

    @ai.tool(name='sneaky')
    async def sneaky() -> str:
        ran.append('sneaky')
        return 'injected-ran'

    pm.responses = [_model_calls_tool(name='sneaky', ref='r1')]

    response = await ai.generate(prompt='hi', use=[_rewrite_tools_middleware(lambda tools: [*tools, 'sneaky'])])

    assert ran == []
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'Tool sneaky not found'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'NOT_FOUND'
    assert response.error.reason is RuntimeErrorReason.TOOL_NOT_FOUND
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'


@pytest.mark.asyncio
async def test_wrap_generate_cannot_swap_the_named_tool_for_another() -> None:
    """They named lookup. A hook that writes sneaky cannot run sneaky."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran: list[str] = []

    @ai.tool(name='lookup')
    async def lookup() -> str:
        ran.append('lookup')
        return '72F'

    @ai.tool(name='sneaky')
    async def sneaky() -> str:
        ran.append('sneaky')
        return 'injected-ran'

    pm.responses = [_model_calls_tool(name='sneaky', ref='r1')]

    response = await ai.generate(
        prompt='hi',
        tools=['lookup'],
        use=[_rewrite_tools_middleware(lambda _tools: ['sneaky'])],
    )

    assert ran == []
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'Tool sneaky not found'
    assert response.message is None
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.TOOL_NOT_FOUND
    assert [message.role for message in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_wrap_generate_cannot_strip_the_named_tool() -> None:
    """They named lookup. A hook that clears tools= still runs lookup."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran: list[str] = []

    @ai.tool(name='lookup')
    async def lookup() -> str:
        ran.append('lookup')
        return '72F'

    pm.responses = [_model_calls_tool(name='lookup', ref='r1'), _model_says('done')]

    response = await ai.generate(
        prompt='hi',
        tools=['lookup'],
        use=[_rewrite_tools_middleware(lambda _tools: [])],
    )

    assert ran == ['lookup']
    assert response.finish_reason == FinishReason.STOP
    assert response.error is None
    assert response.message is not None
    assert response.message.text == 'done'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_wrap_generate_named_tool_still_runs_after_a_swap() -> None:
    """They named lookup. The model asking for lookup still runs it after the hook wrote sneaky."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran: list[str] = []

    @ai.tool(name='lookup')
    async def lookup() -> str:
        ran.append('lookup')
        return '72F'

    @ai.tool(name='sneaky')
    async def sneaky() -> str:
        ran.append('sneaky')
        return 'injected-ran'

    pm.responses = [_model_calls_tool(name='lookup', ref='r1'), _model_says('done')]

    response = await ai.generate(
        prompt='hi',
        tools=['lookup'],
        use=[_rewrite_tools_middleware(lambda _tools: ['sneaky'])],
    )

    assert ran == ['lookup']
    assert response.finish_reason == FinishReason.STOP
    assert response.error is None
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_wrap_generate_injected_tool_after_closed_round_does_not_run() -> None:
    """After a closed lookup round, a hook-added sneaky is still unknown."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran: list[str] = []

    @ai.tool(name='lookup')
    async def lookup() -> str:
        ran.append('lookup')
        return '72F'

    @ai.tool(name='sneaky')
    async def sneaky() -> str:
        ran.append('sneaky')
        return 'injected-ran'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        _model_calls_tool(name='sneaky', ref='r2'),
    ]

    response = await ai.generate(
        prompt='keep going',
        tools=['lookup'],
        use=[_rewrite_tools_middleware(lambda tools: [*tools, 'sneaky'])],
    )

    assert ran == ['lookup']
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'Tool sneaky not found'
    assert response.message is None
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.TOOL_NOT_FOUND
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_wrap_generate_cannot_replace_the_named_tool_action() -> None:
    """They named lookup. Re-registering that name in a hook still runs the door Action."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran: list[str] = []

    @ai.tool(name='lookup')
    async def lookup() -> str:
        ran.append('lookup')
        return '72F'

    scratch = Registry()

    async def impostor() -> str:
        ran.append('impostor')
        return 'IMPOSTOR'

    leak = define_tool(scratch, impostor, name='lookup').action()

    class SwapBody(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            ctx.ai.registry.register_action_from_instance(leak)
            return await next_fn(params, ctx)

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await next_fn(params, ctx)

    pm.responses = [_model_calls_tool(name='lookup', ref='r1'), _model_says('done')]

    response = await ai.generate(prompt='hi', tools=['lookup'], use=[SwapBody()])

    assert ran == ['lookup']
    assert response.finish_reason == FinishReason.STOP
    assert response.error is None
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_wrap_generate_cannot_replace_the_named_tool_after_a_closed_round() -> None:
    """After a closed lookup round, a hook re-register still cannot swap the Action."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran: list[str] = []

    @ai.tool(name='lookup')
    async def lookup() -> str:
        ran.append('lookup')
        return '72F'

    scratch = Registry()

    async def impostor() -> str:
        ran.append('impostor')
        return 'IMPOSTOR'

    leak = define_tool(scratch, impostor, name='lookup').action()

    class SwapBody(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            ctx.ai.registry.register_action_from_instance(leak)
            return await next_fn(params, ctx)

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await next_fn(params, ctx)

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        _model_calls_tool(name='lookup', ref='r2'),
        _model_says('done'),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[SwapBody()])

    assert ran == ['lookup', 'lookup']
    assert response.finish_reason == FinishReason.STOP
    assert response.error is None
    assert [message.role for message in response.messages] == [
        Role.USER,
        Role.MODEL,
        Role.TOOL,
        Role.MODEL,
        Role.TOOL,
        Role.MODEL,
    ]
    assert _tool_output(response.messages[2]) == '72F'
    assert _tool_output(response.messages[4]) == '72F'


@pytest.mark.asyncio
async def test_resume_restart_cannot_replace_the_named_tool_action() -> None:
    """They named lookup. Restart still runs the door Action after a hook re-register."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran: list[str] = []

    @ai.tool(name='lookup')
    async def lookup(inp: dict) -> str:
        ran.append('lookup')
        if not inp.get('ok'):
            raise Interrupt({'hold': True})
        return '72F'

    scratch = Registry()

    async def impostor(inp: dict) -> str:
        ran.append('impostor')
        return 'IMPOSTOR'

    leak = define_tool(scratch, impostor, name='lookup').action()

    class SwapBody(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            ctx.ai.registry.register_action_from_instance(leak)
            return await next_fn(params, ctx)

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await next_fn(params, ctx)

    pm.responses = [_model_calls_tool(name='lookup', ref='r1'), _model_says('done')]

    first = await ai.generate(prompt='hi', tools=['lookup'])
    assert first.finish_reason == FinishReason.INTERRUPTED
    assert ran == ['lookup']

    second = await ai.generate(
        messages=list(first.messages),
        tools=['lookup'],
        resume_restart=first.interrupts[0].restart(replace_input={'ok': True}),
        use=[SwapBody()],
    )

    assert ran == ['lookup', 'lookup']
    assert second.finish_reason == FinishReason.STOP
    assert second.error is None
    assert [message.role for message in second.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(second.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_tool_failure_after_tool_turn_keeps_closed_rounds() -> None:
    """A tool that blows up drops the unfinished round and returns the closed ones."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    calls = {'n': 0}

    @ai.tool(name='lookup')
    async def lookup() -> str:
        calls['n'] += 1
        if calls['n'] > 1:
            raise RuntimeError('db down')
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        _model_calls_tool(name='lookup', ref='r2'),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert isinstance(response.error.details, dict)
    assert response.error.details['reason'] == 'TOOL_FAILED'
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_tool_runtime_error_message_is_internal() -> None:
    """A tool RuntimeError does not publish its text on the returned response."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        raise RuntimeError('connection to postgres://user:pw@host refused')

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]

    response = await ai.generate(prompt='hi', tools=['lookup'])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.message == 'internal error'
    assert 'postgres' not in (response.finish_message or '')
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_tool_public_error_message_is_the_sentence() -> None:
    """PublicError is how the tool author publishes the sentence on the response."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup_order')
    async def lookup_order() -> str:
        raise PublicError('NOT_FOUND', 'No order 99')

    pm.responses = [_model_calls_tool(name='lookup_order', ref='r1')]

    response = await ai.generate(prompt='hi', tools=['lookup_order'])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'No order 99'
    assert response.error is not None
    assert response.error.status == 'NOT_FOUND'
    assert response.error.message == 'No order 99'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_output_schema_does_not_replace_tool_failure() -> None:
    """Output validation cannot mask the runtime failure that ended generation."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Recipe(BaseModel):
        title: str

    @ai.tool(name='broken')
    async def broken() -> str:
        raise RuntimeError('db down')

    pm.responses = [_model_calls_tool(name='broken', ref='r1')]

    response = await ai.generate(
        prompt='make a recipe',
        tools=['broken'],
        output_schema=Recipe,
        output_instructions=False,
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.message is None
    assert response.output is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_provider_failed_finish_is_not_relabelled_invalid_output() -> None:
    """When the provider reports a failure, you see that failure, not a schema complaint.

    A provider can return finish_reason='failed' with text attached. That text
    will not match your output schema, but the schema is not the problem, so
    Genkit leaves the model's own failure in place instead of relabelling it
    as invalid output. Read finish_reason and finish_message to find out what
    went wrong; response.output is None and response.error stays unset.
    """
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Recipe(BaseModel):
        title: str

    pm.responses = [
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part.from_text('upstream blew up')]),
            finish_reason=FinishReason.FAILED,
            finish_message='provider failed',
        )
    ]

    response = await ai.generate(prompt='make a recipe', output_schema=Recipe, output_instructions=False)

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is None
    assert response.output is None
    assert response.text == 'upstream blew up'


@pytest.mark.asyncio
async def test_one_parallel_tool_failure_drops_successful_sibling() -> None:
    """A partial set of tool responses is not a resendable conversation round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    stable_calls = 0

    @ai.tool(name='stable')
    async def stable() -> str:
        nonlocal stable_calls
        stable_calls += 1
        return f'ok-{stable_calls}'

    @ai.tool(name='broken')
    async def broken() -> str:
        raise RuntimeError('db down')

    pm.responses = [
        _model_calls_tool(name='stable', ref='closed'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(tool_request=ToolRequest(name='stable', input={}, ref='ok')),
                    Part(tool_request=ToolRequest(name='broken', input={}, ref='bad')),
                ],
            ),
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['stable', 'broken'])

    assert response.finish_reason == FinishReason.FAILED
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'closed'
    assert _tool_output(response.messages[2]) == 'ok-1'


@pytest.mark.asyncio
async def test_sibling_interrupt_goes_with_failed_tool_round() -> None:
    """A tool failure drops the whole open round, including a sibling interrupt."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pause_done = asyncio.Event()

    @ai.tool(name='pauser')
    async def pauser() -> str:
        pause_done.set()
        raise Interrupt({'reason': 'approval'})

    @ai.tool(name='boom')
    async def boom() -> str:
        await pause_done.wait()
        raise RuntimeError('boom')

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(tool_request=ToolRequest(name='pauser', input={}, ref='pause')),
                    Part(tool_request=ToolRequest(name='boom', input={}, ref='bad')),
                ],
            ),
        )
    ]

    response = await ai.generate(prompt='start', tools=['pauser', 'boom'])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.message is None
    assert response.interrupts == []
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'start'


@pytest.mark.asyncio
async def test_first_turn_model_failure_returns_closed_history() -> None:
    """A model that dies on the first turn returns the user prompt and no model message."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    def die(_request: ModelRequest) -> ModelResponse:
        raise RuntimeError('model exploded')

    pm.response_cb = die

    response = await ai.generate(prompt='start')

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'start'


@pytest.mark.asyncio
async def test_model_failure_after_tool_turn_keeps_closed_rounds() -> None:
    """A model that dies on the next turn still returns the closed tool turn."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    seen = {'n': 0}

    def once_then_die(_request: ModelRequest) -> ModelResponse:
        seen['n'] += 1
        if seen['n'] == 1:
            return _model_calls_tool(name='lookup', ref='r1')
        raise RuntimeError('stream died')

    pm.response_cb = once_then_die

    response = await ai.generate(prompt='keep going', tools=['lookup'])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.details is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_chat_model_returning_dict_after_tool_keeps_closed_rounds() -> None:
    """A chat model that returns a dict on the next turn still keeps the closed tool round."""
    ai = Genkit()
    seen = {'n': 0}

    async def model_fn(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse | dict[str, object]:
        seen['n'] += 1
        if seen['n'] == 1:
            return _model_calls_tool(name='lookup', ref='r1')
        return {'text': 'nope'}

    ai.define_model(name='plain-dict', fn=model_fn)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    response = await ai.generate(model='plain-dict', prompt='keep going', tools=['lookup'])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Model 'plain-dict' did not return a ModelResponse" in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.operation is None


class _CityQuery(BaseModel):
    city: str


@pytest.mark.asyncio
async def test_generate_invalid_tool_args_drops_unanswered_request() -> None:
    """A tool request whose args fail the schema drops that unanswered call."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup(query: _CityQuery) -> str:
        return '72F'

    pm.response_cb = lambda _request: _model_calls_tool(name='lookup', ref='r1')

    response = await ai.generate(prompt='weather in paris?', tools=['lookup'])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'lookup' in response.finish_message
    assert 'Invalid input' in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'weather in paris?'


@pytest.mark.asyncio
async def test_generate_invalid_tool_args_after_tool_turn_keeps_closed_rounds() -> None:
    """Bad tool args on the next turn still keep the closed tool round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup(query: _CityQuery) -> str:
        return '72F'

    seen = {'n': 0}

    def once_ok_then_empty(_request: ModelRequest) -> ModelResponse:
        seen['n'] += 1
        if seen['n'] == 1:
            return _model_calls_tool(name='lookup', ref='r1', input={'city': 'NYC'})
        return _model_calls_tool(name='lookup', ref='r2')

    pm.response_cb = once_ok_then_empty

    response = await ai.generate(prompt='keep going', tools=['lookup'])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'lookup' in response.finish_message
    assert 'Invalid input' in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert response.error.message == response.finish_message
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'


class _Unserializable:
    pass


@pytest.mark.asyncio
async def test_generate_unserializable_tool_output_drops_unanswered_request() -> None:
    """A tool that ran but cannot serialize its return drops that unanswered call."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> object:
        return _Unserializable()

    pm.response_cb = lambda _request: _model_calls_tool(name='lookup', ref='r1')

    response = await ai.generate(prompt='weather in paris?', tools=['lookup'])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'lookup' in response.finish_message
    assert 'not JSON-serializable' in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'weather in paris?'


@pytest.mark.asyncio
async def test_generate_unserializable_tool_output_after_tool_turn_keeps_closed_rounds() -> None:
    """Unserializable output on the next turn still keeps the closed tool round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    calls = {'n': 0}

    @ai.tool(name='lookup')
    async def lookup() -> object:
        calls['n'] += 1
        if calls['n'] == 1:
            return '72F'
        return _Unserializable()

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        _model_calls_tool(name='lookup', ref='r2'),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'lookup' in response.finish_message
    assert 'not JSON-serializable' in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert response.error.message == response.finish_message
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_failed_history_can_be_reused_without_rerunning_closed_tool() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    tool_calls = 0

    @ai.tool(name='lookup')
    async def lookup() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return '72F'

    def fail_after_tool(_request: ModelRequest) -> ModelResponse:
        if pm.request_count == 1:
            return _model_calls_tool(name='lookup', ref='r1')
        raise RuntimeError('provider disconnected')

    pm.response_cb = fail_after_tool
    failed = await ai.generate(prompt='start', tools=['lookup'])

    assert failed.finish_reason == FinishReason.FAILED
    assert failed.message is None
    assert failed.error is not None
    assert failed.error.status == 'INTERNAL'
    assert [message.role for message in failed.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(failed.messages[1]).ref == 'r1'
    assert _tool_output(failed.messages[2]) == '72F'
    assert tool_calls == 1
    assert pm.request_count == 2

    pm.response_cb = lambda _request: ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text('recovered')]),
    )
    recovered = await ai.generate(messages=failed.messages, prompt='try again', tools=['lookup'])

    assert recovered.finish_reason == FinishReason.STOP
    assert recovered.finish_message is None
    assert recovered.error is None
    assert recovered.message is not None
    assert recovered.message.text == 'recovered'
    assert [message.role for message in recovered.messages] == [
        Role.USER,
        Role.MODEL,
        Role.TOOL,
        Role.USER,
        Role.MODEL,
    ]
    assert _tool_request(recovered.messages[1]).ref == 'r1'
    assert _tool_output(recovered.messages[2]) == '72F'
    assert pm.last_request is not None
    assert [message.role for message in pm.last_request.messages] == [
        Role.USER,
        Role.MODEL,
        Role.TOOL,
        Role.USER,
    ]
    assert _tool_request(pm.last_request.messages[1]).ref == 'r1'
    assert _tool_output(pm.last_request.messages[2]) == '72F'
    assert pm.last_request.messages[3].text == 'try again'
    assert tool_calls == 1
    assert pm.request_count == 3


@pytest.mark.asyncio
async def test_generate_stream_closes_with_structured_max_turns_response() -> None:
    ai = Genkit(model='streamLimitModel')
    model_calls = 0
    tool_calls = 0

    class Recipe(BaseModel):
        title: str

    @ai.tool(name='lookup')
    async def lookup() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return '72F'

    async def stream_limit_model(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('first')]))
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('second')]))
        return _model_calls_tool(name='lookup', ref='unopened')

    ai.define_model(name='streamLimitModel', fn=stream_limit_model)
    stream = ai.generate_stream(
        prompt='make a recipe',
        tools=['lookup'],
        max_turns=0,
        output_schema=Recipe,
        output_instructions=False,
    )

    chunks = [chunk async for chunk in stream]
    response = await stream.response

    assert [text_from_content(chunk.content) for chunk in chunks] == ['first', 'second']
    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Exceeded maximum tool call iterations (0)'
    assert response.message is None
    assert response.output is None
    assert response.error is not None
    assert response.error.status == 'ABORTED'
    assert response.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert response.error.details == {'reason': 'MAX_TURNS_EXCEEDED'}
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'make a recipe'
    assert model_calls == 1
    assert tool_calls == 0


def _model_dies_after_hello_wor(ai: Genkit, name: str) -> None:
    async def model(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Hello, ')]))
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('wor')]))
        raise RuntimeError('stream died')

    ai.define_model(name=name, fn=model)


async def _run_dies_midway(surface: str) -> tuple[list[str] | None, ModelResponse[Any]]:
    """Run the dies-after-'Hello, wor' model through one generate surface."""
    ai = Genkit(model='diesMidway')
    _model_dies_after_hello_wor(ai, 'diesMidway')
    if surface == 'generate':
        return None, await ai.generate(prompt='start')
    if surface == 'generate_stream':
        stream = ai.generate_stream(prompt='start')
    else:
        stream = ai.define_prompt(name='greet', prompt='start').stream()
    chunks = [chunk.text async for chunk in stream]
    return chunks, await stream.response


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['generate_stream', 'prompt.stream', 'generate'])
async def test_model_dies_midway_ends_failed_with_empty_reply(surface: str) -> None:
    """The model dies after two chunks; each surface ends FAILED/INTERNAL with no half answer in history."""
    chunks, response = await _run_dies_midway(surface)

    if chunks is not None:
        assert chunks == ['Hello, ', 'wor']
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.text == ''
    assert response.message is None
    assert response.output is None
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'start'


@pytest.mark.asyncio
async def test_generate_stream_fails_midway_with_schema_output_is_none() -> None:
    """With output_schema, a stream that dies partway leaves final.output as None."""

    class City(BaseModel):
        name: str
        population: int

    ai = Genkit(model='diesMidJson')

    async def model(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('{"name": "Pa')]))
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('ris", "population": 21')]))
        raise RuntimeError('stream died')

    ai.define_model(name='diesMidJson', fn=model)

    stream = ai.generate_stream(prompt='a city', output_schema=City)
    chunks = [chunk async for chunk in stream]
    response = await stream.response

    assert len(chunks) == 2
    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.text == ''
    assert response.message is None
    assert response.output is None
    assert [message.role for message in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_stream_fails_after_tool_turn_keeps_closed_round() -> None:
    """The model dies on its second turn; history keeps the finished tool round and drops the half answer."""
    ai = Genkit(model='diesAfterTool')
    model_calls = 0
    tool_calls = 0

    @ai.tool(name='lookup')
    async def lookup() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return '72F'

    async def model(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return _model_calls_tool(name='lookup', ref='closed')
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial-1')]))
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial-2')]))
        raise RuntimeError('stream broke')

    ai.define_model(name='diesAfterTool', fn=model)

    stream = ai.generate_stream(prompt='start', tools=['lookup'])
    chunks = [chunk async for chunk in stream]
    response = await stream.response

    model_chunks = [chunk for chunk in chunks if chunk.role == Role.MODEL]
    assert [chunk.text for chunk in model_chunks] == ['partial-1', 'partial-2']
    assert [chunk.role for chunk in chunks] == [Role.TOOL, Role.MODEL, Role.MODEL]
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.text == ''
    assert response.message is None
    assert response.output is None
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'closed'
    assert _tool_output(response.messages[2]) == '72F'
    assert response.messages[-1].role == Role.TOOL
    assert model_calls == 2
    assert tool_calls == 1


def test_generate_stream_does_not_accept_timeout() -> None:
    """generate_stream has no timeout=; the async for waits until generate finishes."""
    ai = Genkit()
    with pytest.raises(TypeError, match='timeout'):
        ai.generate_stream(prompt='hi', timeout=5)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_generate_on_chunk_failure_returns_closed_history() -> None:
    """A dead on_chunk sink on the first model token returns the user prompt."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial')])]]

    def on_chunk(_: ModelResponseChunk) -> None:
        raise RuntimeError('model sink closed')

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        ),
        on_chunk=on_chunk,
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'model sink closed'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'


@pytest.mark.asyncio
async def test_generate_on_chunk_failure_echoes_full_request() -> None:
    """A streaming callback that raises still reports the request the turn sent."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='testTool')
    async def _test_tool() -> object:
        """description"""  # noqa: D403, D415
        return 'tool called'

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial')])]]

    def on_chunk(_: ModelResponseChunk) -> None:
        raise RuntimeError('model sink closed')

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            docs=[Document(content=[Part.from_text('doc content 1')])],
            config={'temperature': 0.5},
            tools=['testTool'],
            tool_choice='required',
            output=GenerateActionOutputConfig(format='json'),
        ),
        on_chunk=on_chunk,
    )

    assert response.finish_reason == FinishReason.FAILED
    request = response.request
    assert request is not None
    assert request.docs, 'docs dropped from echoed request'
    assert request.config == {'temperature': 0.5}, 'config dropped from echoed request'
    assert request.tools, 'tools dropped from echoed request'
    assert request.tool_choice == 'required', 'tool_choice dropped from echoed request'
    assert request.output is not None
    assert request.output.format == 'json', 'output dropped from echoed request'


@pytest.mark.asyncio
async def test_generate_on_chunk_genkit_error_is_internal_not_the_sink_reason() -> None:
    """A sink that raises GenkitError is still a dead pipe, not TOOL_FAILED."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial')])]]

    def on_chunk(_: ModelResponseChunk) -> None:
        raise GenkitError(
            status='NOT_FOUND',
            message='sink closed',
            reason=RuntimeErrorReason.TOOL_FAILED,
        )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        ),
        on_chunk=on_chunk,
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'sink closed'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.details is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'


@pytest.mark.asyncio
async def test_first_turn_task_cancel_raises() -> None:
    """Cancelling the generate task raises CancelledError; it does not return a response."""
    ai = Genkit(model='waitingModel')
    started = asyncio.Event()
    model_calls = 0

    async def waiting_model(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        started.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    ai.define_model(name='waitingModel', fn=waiting_model)
    task = asyncio.create_task(ai.generate(prompt='wait'))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model_calls == 1


@pytest.mark.asyncio
async def test_wait_for_generate_raises_timeout() -> None:
    """asyncio.wait_for around generate still raises TimeoutError."""
    ai = Genkit(model='slow')

    async def slow_model(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        await asyncio.sleep(5)
        raise AssertionError('unreachable')

    ai.define_model(name='slow', fn=slow_model)
    # asyncio.TimeoutError, not the builtin: on 3.10 they are different classes
    # and wait_for raises the asyncio one. 3.11 aliased them.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(ai.generate(model='slow', prompt='x'), timeout=0.3)


@pytest.mark.asyncio
async def test_task_cancel_after_tool_turn_raises() -> None:
    """Cancelling the generate task after a closed tool round still raises CancelledError."""
    ai = Genkit(model='cancelAfterToolModel')
    second_started = asyncio.Event()
    model_calls = 0

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    async def cancel_after_tool(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return _model_calls_tool(name='lookup', ref='r1')
        second_started.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    ai.define_model(name='cancelAfterToolModel', fn=cancel_after_tool)
    task = asyncio.create_task(ai.generate(prompt='keep going', tools=['lookup']))
    await second_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model_calls == 2


class _NoopGenerateMiddleware(BaseMiddleware):
    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await next_fn(params, ctx)


def _five_noop_middleware() -> list[BaseMiddleware]:
    return [_NoopGenerateMiddleware() for _ in range(5)]


def _four_hundred_noop_middleware() -> list[BaseMiddleware]:
    return [_NoopGenerateMiddleware() for _ in range(400)]


async def _assert_no_extra_tasks(before: set[int]) -> None:
    await asyncio.sleep(0.05)
    still_running = [
        task
        for task in asyncio.all_tasks()
        if id(task) not in before and task is not asyncio.current_task() and not task.done()
    ]
    assert still_running == []


def _always_requests_tool(pm: ScriptedModel, *, name: str = 'step') -> None:
    def always_tool(_request: ModelRequest) -> ModelResponse:
        return _model_calls_tool(name=name, ref=str(pm.request_count + 1))

    pm.response_cb = always_tool


@pytest.mark.asyncio
async def test_generate_with_five_middleware_and_max_turns_50_returns_aborted_after_51_model_calls() -> None:
    """ai.generate with five no-op middleware and max_turns=50 aborts after 51 model calls."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    _always_requests_tool(pm)
    response = await ai.generate(
        prompt='keep going',
        tools=['step'],
        max_turns=50,
        use=_five_noop_middleware(),
    )

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Exceeded maximum tool call iterations (50)'
    assert response.error is not None
    assert response.error.status == 'ABORTED'
    assert response.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert pm.request_count == 51


@pytest.mark.asyncio
async def test_generate_stream_with_five_middleware_and_max_turns_50_returns_aborted_after_51_model_calls() -> None:
    """ai.generate_stream with five no-op middleware and max_turns=50 closes aborted after 51 model calls."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    _always_requests_tool(pm)
    stream = ai.generate_stream(
        prompt='keep going',
        tools=['step'],
        max_turns=50,
        use=_five_noop_middleware(),
    )
    _ = [chunk async for chunk in stream]
    response = await stream.response

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Exceeded maximum tool call iterations (50)'
    assert response.error is not None
    assert response.error.status == 'ABORTED'
    assert response.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert pm.request_count == 51


@pytest.mark.asyncio
async def test_generate_with_five_middleware_and_max_turns_200_returns_aborted() -> None:
    """ai.generate with five middleware and max_turns=200 aborts at 200; no RecursionError."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    _always_requests_tool(pm)
    response = await ai.generate(
        prompt='keep going',
        tools=['step'],
        max_turns=200,
        use=_five_noop_middleware(),
    )

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Exceeded maximum tool call iterations (200)'
    assert response.error is not None
    assert response.error.status == 'ABORTED'
    assert response.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert pm.request_count == 201


@pytest.mark.asyncio
async def test_generate_stream_with_five_middleware_and_max_turns_200_returns_aborted() -> None:
    """ai.generate_stream with five middleware and max_turns=200 closes aborted at 200."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    _always_requests_tool(pm)
    stream = ai.generate_stream(
        prompt='keep going',
        tools=['step'],
        max_turns=200,
        use=_five_noop_middleware(),
    )
    _ = [chunk async for chunk in stream]
    response = await stream.response

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Exceeded maximum tool call iterations (200)'
    assert response.error is not None
    assert response.error.status == 'ABORTED'
    assert response.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert pm.request_count == 201


@pytest.mark.asyncio
async def test_generate_eager_task_factory_two_hundred_tool_turns_returns() -> None:
    """max_turns=200 under eager factory returns ABORTED; no RecursionError."""
    eager_factory = getattr(asyncio, 'eager_task_factory', None)
    if eager_factory is None:
        pytest.skip('asyncio.eager_task_factory requires Python 3.12')

    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    loop.set_task_factory(eager_factory)
    try:
        ai = Genkit(model='scriptedModel')
        pm, _ = define_scripted_model(ai)

        @ai.tool(name='step')
        async def step() -> str:
            return 'ok'

        _always_requests_tool(pm)
        response = await ai.generate(
            prompt='keep going',
            tools=['step'],
            max_turns=200,
        )

        assert response.finish_reason == FinishReason.ABORTED
        assert response.finish_message == 'Exceeded maximum tool call iterations (200)'
        assert response.error is not None
        assert response.error.status == 'ABORTED'
        assert response.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
        assert pm.request_count == 201
    finally:
        loop.set_task_factory(previous)


@pytest.mark.asyncio
async def test_generate_turn_0_middleware_after_next_sees_the_final_answer() -> None:
    """wrap_generate after await next on the first wrap sees the final stop text."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    after_next: list[tuple[FinishReason | None, str]] = []

    class AfterNext(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            response = await next_fn(params, ctx)
            if params.iteration == 0:
                after_next.append((response.finish_reason, response.text))
            return response

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    pm.responses = [
        _model_calls_tool(name='step', ref='r1'),
        _model_calls_tool(name='step', ref='r2'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('final answer')]),
        ),
    ]

    response = await ai.generate(prompt='go', tools=['step'], use=[AfterNext()], max_turns=5)

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'final answer'
    assert after_next == [(FinishReason.STOP, 'final answer')]


@pytest.mark.asyncio
async def test_generate_error_on_a_later_turn_returns_failed_with_that_error() -> None:
    """A later-turn model exception returns failed/INTERNAL; closed tool rounds stay on messages."""
    ai = Genkit(model='laterBoomModel')
    model_calls = 0

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    async def later_boom(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls <= 2:
            return _model_calls_tool(name='step', ref=str(model_calls))
        raise RuntimeError('boom-turn-3')

    ai.define_model(name='laterBoomModel', fn=later_boom)
    response = await ai.generate(
        prompt='go',
        tools=['step'],
        max_turns=10,
        use=_five_noop_middleware(),
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert model_calls == 3
    assert [message.role for message in response.messages] == [
        Role.USER,
        Role.MODEL,
        Role.TOOL,
        Role.MODEL,
        Role.TOOL,
    ]
    assert _tool_output(response.messages[2]) == 'ok'
    assert _tool_output(response.messages[4]) == 'ok'


@pytest.mark.asyncio
async def test_generate_middleware_awaiting_next_inside_taskgroup_returns_stop() -> None:
    """Middleware that awaits next inside asyncio.TaskGroup still returns stop and the final text."""
    task_group_cls = getattr(asyncio, 'TaskGroup', None)
    if task_group_cls is None:
        pytest.skip('asyncio.TaskGroup requires Python 3.11')

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class InTaskGroup(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            async with task_group_cls() as tg:
                nested = tg.create_task(next_fn(params, ctx))
            return nested.result()

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    pm.responses = [
        _model_calls_tool(name='step', ref='r1'),
        _model_calls_tool(name='step', ref='r2'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done via taskgroup')]),
        ),
    ]

    response = await ai.generate(prompt='go', tools=['step'], use=[InTaskGroup()], max_turns=5)

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'done via taskgroup'


@pytest.mark.asyncio
async def test_generate_cancelled_during_a_later_turn_stops_calling_the_model() -> None:
    """Cancelling generate while turn 3 runs cancels that turn; the model is not called again."""
    ai = Genkit(model='hangOnTurn3Model')
    started = asyncio.Event()
    model_calls = 0

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    async def hang_on_turn_3(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls < 4:
            return _model_calls_tool(name='step', ref=str(model_calls))
        started.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    ai.define_model(name='hangOnTurn3Model', fn=hang_on_turn_3)
    before = {id(task) for task in asyncio.all_tasks()}
    task = asyncio.create_task(ai.generate(prompt='go', tools=['step'], max_turns=20))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model_calls == 4
    await _assert_no_extra_tasks(before)


def test_generate_system_exit_on_a_later_turn_unwinds_middleware_before_the_loop_exits() -> None:
    """SystemExit from the model on turn 2 unwinds through middleware and the caller in order.

    asyncio re-raises SystemExit out of the loop instead of into the awaiting task.
    Without carrying it back across each hop, outer frames would only see
    CancelledError when asyncio.run shuts down.
    """
    seen: list[str] = []

    class RecordExit(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            try:
                return await next_fn(params, ctx)
            except BaseException as exc:
                seen.append(f'middleware:{type(exc).__name__}')
                raise

    async def main() -> None:
        ai = Genkit(model='exitOnTurn2Model')
        model_calls = 0

        @ai.tool(name='step')
        async def step() -> str:
            return 'ok'

        async def exit_on_turn_2(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
            nonlocal model_calls
            model_calls += 1
            if model_calls == 1:
                return _model_calls_tool(name='step', ref='r1')
            raise SystemExit(3)

        ai.define_model(name='exitOnTurn2Model', fn=exit_on_turn_2)
        try:
            await ai.generate(prompt='go', tools=['step'], use=[RecordExit()])
        except BaseException as exc:
            seen.append(f'caller:{type(exc).__name__}')
            raise

    with pytest.raises(SystemExit) as raised:
        asyncio.run(main())

    assert raised.value.code == 3
    assert seen
    assert seen[-1] == 'caller:SystemExit'
    assert all(entry.endswith(':SystemExit') for entry in seen)
    assert 'middleware:SystemExit' in seen


@pytest.mark.asyncio
async def test_generate_with_400_middleware_returns_the_model_text() -> None:
    """ai.generate with 400 middleware returns the model text, and the outer wrap sees it after next."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    after_next: list[str] = []

    class AfterNext(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            response = await next_fn(params, ctx)
            after_next.append(response.text)
            return response

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('from a deep list')]),
        ),
    ]

    response = await ai.generate(
        prompt='go',
        use=[AfterNext(), *_four_hundred_noop_middleware()],
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'from a deep list'
    assert after_next == ['from a deep list']


@pytest.mark.asyncio
async def test_generate_eager_task_factory_four_hundred_middleware_returns() -> None:
    """400 middleware under the eager task factory returns; no RecursionError."""
    eager_factory = getattr(asyncio, 'eager_task_factory', None)
    if eager_factory is None:
        pytest.skip('asyncio.eager_task_factory requires Python 3.12')

    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    loop.set_task_factory(eager_factory)
    try:
        ai = Genkit(model='scriptedModel')
        pm, _ = define_scripted_model(ai)
        pm.responses = [_model_says('from a deep list')]
        response = await ai.generate(prompt='go', use=_four_hundred_noop_middleware())

        assert response.finish_reason == FinishReason.STOP
        assert response.text == 'from a deep list'
    finally:
        loop.set_task_factory(previous)


@pytest.mark.asyncio
async def test_generate_with_400_middleware_runs_the_tool_and_returns_the_model_text() -> None:
    """ai.generate with 400 middleware runs the tool once, then returns the model text."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    tool_runs = 0

    @ai.tool(name='step')
    async def step() -> str:
        nonlocal tool_runs
        tool_runs += 1
        return 'ok'

    pm.responses = [
        _model_calls_tool(name='step', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('after the tool')]),
        ),
    ]

    response = await ai.generate(
        prompt='go',
        tools=['step'],
        use=_four_hundred_noop_middleware(),
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'after the tool'
    assert tool_runs == 1
    assert pm.request_count == 2


@pytest.mark.asyncio
async def test_generate_with_400_middleware_cancelled_while_the_model_hangs_raises_cancelled_error() -> None:
    """Cancelling generate while the model is in a 400-middleware call raises CancelledError."""
    ai = Genkit(model='hangModel')
    started = asyncio.Event()

    async def hang(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    ai.define_model(name='hangModel', fn=hang)
    before = {id(task) for task in asyncio.all_tasks()}
    task = asyncio.create_task(
        ai.generate(prompt='go', use=_four_hundred_noop_middleware()),
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_no_extra_tasks(before)


def _recording_layer(name: str, steps: list[str]) -> BaseMiddleware:
    class Recording(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            steps.append(f'{name} generate before {params.iteration}')
            response = await next_fn(params, ctx)
            steps.append(f'{name} generate after {params.iteration}')
            return response

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            steps.append(f'{name} model before')
            response = await next_fn(params, ctx)
            steps.append(f'{name} model after')
            return response

        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            steps.append(f'{name} tool before')
            response = await next_fn(params, ctx)
            steps.append(f'{name} tool after')
            return response

    return Recording()


@pytest.mark.asyncio
async def test_generate_middleware_before_and_after_stay_in_order_across_a_tool_turn() -> None:
    """Outer and inner middleware around 400 hooks keep before/after order through a tool turn."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    steps: list[str] = []

    @ai.tool(name='step')
    async def step() -> str:
        steps.append('tool')
        return 'ok'

    pm.responses = [
        _model_calls_tool(name='step', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('final answer')]),
        ),
    ]

    response = await ai.generate(
        prompt='go',
        tools=['step'],
        use=[
            _recording_layer('outer', steps),
            *_four_hundred_noop_middleware(),
            _recording_layer('inner', steps),
        ],
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'final answer'
    assert response.message is not None
    assert response.messages[-1] == response.message
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(response.messages[2]) == 'ok'
    assert steps == [
        'outer generate before 0',
        'inner generate before 0',
        'outer model before',
        'inner model before',
        'inner model after',
        'outer model after',
        'outer tool before',
        'inner tool before',
        'tool',
        'inner tool after',
        'outer tool after',
        'outer generate before 1',
        'inner generate before 1',
        'outer model before',
        'inner model before',
        'inner model after',
        'outer model after',
        'inner generate after 1',
        'outer generate after 1',
        'inner generate after 0',
        'outer generate after 0',
    ]


@pytest.mark.asyncio
async def test_generate_outer_middleware_text_is_what_the_caller_gets() -> None:
    """The outermost middleware's rewritten text is the text ai.generate returns."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_model_says('secret')]

    class Rewrite(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            response = await next_fn(params, ctx)
            return response.model_copy(
                update={'message': Message(role=Role.MODEL, content=[Part.from_text('rewritten')])},
            )

    response = await ai.generate(
        prompt='go',
        use=[Rewrite(), *_four_hundred_noop_middleware()],
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'rewritten'
    assert response.message is not None
    assert response.messages[-1] == response.message


@pytest.mark.asyncio
async def test_generate_middleware_that_returns_without_calling_next_skips_the_model() -> None:
    """An inner middleware that returns its own response never calls the model."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_model_says('should not run')]
    seen: list[str] = []

    class SeeIt(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            response = await next_fn(params, ctx)
            seen.append(response.text)
            return response

    class StopHere(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part.from_text('stopped early')]),
            )

    response = await ai.generate(
        prompt='go',
        use=[SeeIt(), *_four_hundred_noop_middleware(), StopHere()],
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'stopped early'
    assert seen == ['stopped early']
    assert pm.request_count == 0


@pytest.mark.asyncio
async def test_generate_middleware_value_set_before_next_is_what_the_model_returns() -> None:
    """A value set before await next is what the model reads, through 400 hooks."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    marker: contextvars.ContextVar[str] = contextvars.ContextVar('marker', default='unset')

    class SetMarker(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            marker.set('from-middleware')
            return await next_fn(params, ctx)

    def read_marker(_request: ModelRequest) -> ModelResponse:
        return _model_says(marker.get())

    pm.response_cb = read_marker
    response = await ai.generate(
        prompt='go',
        use=[SetMarker(), *_four_hundred_noop_middleware()],
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'from-middleware'


@pytest.mark.asyncio
async def test_generate_outer_middleware_after_next_still_reads_the_value_it_set() -> None:
    """After await next, the outer middleware still reads the value it set, not the inner one."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_model_says('model text')]
    marker: contextvars.ContextVar[str] = contextvars.ContextVar('marker', default='unset')

    class Outer(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            marker.set('outer')
            await next_fn(params, ctx)
            return _model_says(marker.get())

    class Inner(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            marker.set('inner')
            return await next_fn(params, ctx)

    response = await ai.generate(
        prompt='go',
        use=[Outer(), *_four_hundred_noop_middleware(), Inner()],
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'outer'


@pytest.mark.asyncio
async def test_generate_without_use_and_with_logging_use_leave_same_contextvar() -> None:
    """A model-set ContextVar is the same after generate with use=[] and with use=[LogMw()]."""
    marker: contextvars.ContextVar[str] = contextvars.ContextVar('marker', default='unset')

    class LogMw(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await next_fn(params, ctx)

    async def run(*, use: list[BaseMiddleware]) -> tuple[list[str], str]:
        ai = Genkit(model='scriptedModel')
        pm, _ = define_scripted_model(ai)

        @ai.tool(name='step')
        async def step() -> str:
            return 'ok'

        seen: list[str] = []

        def reply(_request: ModelRequest) -> ModelResponse:
            seen.append(marker.get())
            marker.set('from-model')
            if pm.request_count == 1:
                return _model_calls_tool(name='step', ref='r1')
            return _model_says('done')

        pm.response_cb = reply
        token = marker.set('before')
        try:
            await ai.generate(prompt='go', tools=['step'], use=use)
            return seen, marker.get()
        finally:
            marker.reset(token)

    seen_empty, after_empty = await run(use=[])
    seen_log, after_log = await run(use=[LogMw()])
    assert after_empty == after_log
    assert seen_empty == seen_log


@pytest.mark.asyncio
async def test_generate_with_400_middleware_second_turn_error_keeps_the_closed_tool_round() -> None:
    """A second-turn middleware error keeps the closed tool round and drops that turn."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class DenySecondTurn(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            if params.iteration == 1:
                raise RuntimeError('hook denied')
            return await next_fn(params, ctx)

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]
    response = await ai.generate(
        prompt='keep going',
        tools=['lookup'],
        use=[DenySecondTurn(), *_four_hundred_noop_middleware()],
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_with_400_middleware_model_hook_error_returns_internal() -> None:
    """A model hook that raises returns failed/INTERNAL and does not call the model."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_model_says('should not run')]

    class BoomModel(BaseMiddleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            raise RuntimeError('model hook failed')

    response = await ai.generate(
        prompt='go',
        use=[*_four_hundred_noop_middleware(), BoomModel()],
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [message.role for message in response.messages] == [Role.USER]
    assert pm.request_count == 0


@pytest.mark.asyncio
async def test_generate_with_400_middleware_tool_hook_error_returns_internal() -> None:
    """A tool hook that raises returns failed/INTERNAL and drops the open tool round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class BoomTool(BaseMiddleware):
        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            raise RuntimeError('tool hook failed')

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]
    response = await ai.generate(
        prompt='go',
        tools=['lookup'],
        use=[*_four_hundred_noop_middleware(), BoomTool()],
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [message.role for message in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_with_400_middleware_one_of_two_tools_failing_drops_the_open_round() -> None:
    """One of two tools raising drops both outputs; the caller gets the user message only."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='stable')
    async def stable() -> str:
        return 'ok'

    @ai.tool(name='broken')
    async def broken() -> str:
        raise RuntimeError('db down')

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(tool_request=ToolRequest(name='stable', input={}, ref='ok')),
                    Part(tool_request=ToolRequest(name='broken', input={}, ref='bad')),
                ],
            ),
        ),
    ]
    response = await ai.generate(
        prompt='go',
        tools=['stable', 'broken'],
        use=_four_hundred_noop_middleware(),
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.TOOL_FAILED
    assert response.message is None
    assert [message.role for message in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_with_400_middleware_tool_interrupt_pauses_instead_of_failing() -> None:
    """A tool hook that interrupts pauses the generate instead of returning failed."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='blocked')
    async def blocked() -> str:
        return 'should not run'

    class InterruptTool(BaseMiddleware):
        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            raise Interrupt({'blocked': True})

    pm.responses = [_model_calls_tool(name='blocked', ref='r1')]
    response = await ai.generate(
        prompt='do it',
        tools=['blocked'],
        use=[*_four_hundred_noop_middleware(), InterruptTool()],
    )

    assert response.finish_reason == FinishReason.INTERRUPTED
    assert response.finish_message == 'One or more tool calls resulted in interrupts.'
    assert response.error is None
    assert response.message is not None
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[-1] == response.message
    interrupt_parts = [
        part
        for part in response.message.content
        if part.tool_request is not None and part.metadata and 'interrupt' in part.metadata
    ]
    assert len(interrupt_parts) == 1
    assert interrupt_parts[0].metadata is not None
    assert interrupt_parts[0].metadata['interrupt'] == {'blocked': True}


@pytest.mark.asyncio
async def test_generate_with_400_middleware_runs_both_tools() -> None:
    """ai.generate with 400 middleware runs both tools, then returns the model text."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='alpha')
    async def alpha() -> str:
        return 'a'

    @ai.tool(name='beta')
    async def beta() -> str:
        return 'b'

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(tool_request=ToolRequest(name='alpha', input={}, ref='a')),
                    Part(tool_request=ToolRequest(name='beta', input={}, ref='b')),
                ],
            ),
        ),
        _model_says('both ran'),
    ]
    response = await ai.generate(
        prompt='go',
        tools=['alpha', 'beta'],
        use=_four_hundred_noop_middleware(),
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'both ran'
    assert response.message is not None
    assert response.messages[-1] == response.message
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert [part.tool_response.output for part in response.messages[2].content if part.tool_response] == ['a', 'b']


@pytest.mark.asyncio
async def test_generate_stream_with_400_middleware_returns_chunks_and_the_model_text() -> None:
    """ai.generate_stream with 400 middleware yields the chunk text and then the final text."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_model_says('final')]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('c0')])]]

    stream = ai.generate_stream(prompt='go', use=_four_hundred_noop_middleware())
    chunks = [text_from_content(chunk.content) async for chunk in stream]
    response = await stream.response

    assert chunks == ['c0']
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'final'
    assert response.message is not None
    assert response.messages[-1] == response.message


@pytest.mark.asyncio
async def test_generate_stream_with_400_middleware_includes_chunks_from_the_later_turn() -> None:
    """A streaming wrap on turn 0 still sees chunks the later turn emits, through 400 hooks."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    turn0_chunks: list[str] = []

    class Turn0ChunkSpy(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            if params.iteration != 0:
                return await next_fn(params, ctx)
            previous = ctx.on_chunk

            def handler(chunk: ModelResponseChunk) -> None:
                turn0_chunks.append(text_from_content(chunk.content))
                if previous is not None:
                    previous(chunk)

            ctx.replace_on_chunk(handler)
            try:
                return await next_fn(params, ctx)
            finally:
                ctx.replace_on_chunk(previous)

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    pm.responses = [
        _model_calls_tool(name='step', ref='r1'),
        _model_says('final'),
    ]
    pm.chunks = [
        [ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('c0')])],
        [ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('c1')])],
    ]

    stream = ai.generate_stream(
        prompt='go',
        tools=['step'],
        use=[Turn0ChunkSpy(), *_four_hundred_noop_middleware()],
    )
    chunks = [text_from_content(chunk.content) async for chunk in stream]
    response = await stream.response

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'final'
    # The tool round is the empty chunk between the two model chunks.
    assert chunks == ['c0', '', 'c1']
    assert turn0_chunks == ['c0', '', 'c1']


@pytest.mark.asyncio
async def test_generate_with_400_middleware_times_out_while_the_model_hangs() -> None:
    """asyncio.wait_for around a 400-middleware generate raises TimeoutError and leaves nothing running."""
    ai = Genkit(model='hangModel')

    async def hang(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    ai.define_model(name='hangModel', fn=hang)
    before = {id(task) for task in asyncio.all_tasks()}
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            ai.generate(prompt='go', use=_four_hundred_noop_middleware()),
            timeout=0.2,
        )
    await _assert_no_extra_tasks(before)


@pytest.mark.asyncio
async def test_generate_cancelled_with_400_middleware_finishes_inner_cleanup_before_outer() -> None:
    """Cancelling generate finishes the inner middleware cleanup, then the outer one."""
    ai = Genkit(model='hangModel')
    started = asyncio.Event()
    cleanup: list[str] = []

    async def hang(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    class Outer(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            try:
                return await next_fn(params, ctx)
            finally:
                cleanup.append('outer')

    class Inner(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            try:
                return await next_fn(params, ctx)
            finally:
                cleanup.append('inner')
                await asyncio.sleep(0.15)

    ai.define_model(name='hangModel', fn=hang)
    before = {id(task) for task in asyncio.all_tasks()}
    task = asyncio.create_task(
        ai.generate(
            prompt='go',
            use=[Outer(), *_four_hundred_noop_middleware(), Inner()],
        ),
    )
    await started.wait()
    started_at = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = time.monotonic() - started_at

    assert cleanup == ['inner', 'outer']
    assert elapsed >= 0.15
    await _assert_no_extra_tasks(before)


@pytest.mark.asyncio
async def test_generate_with_400_middleware_cancelled_during_the_tool_raises_cancelled_error() -> None:
    """Cancelling generate while the tool is running finishes the tool cleanup, then raises CancelledError."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    started = asyncio.Event()
    cleanup: list[str] = []

    @ai.tool(name='step')
    async def step() -> str:
        try:
            started.set()
            await asyncio.Event().wait()
            return 'ok'
        finally:
            cleanup.append('tool')
            await asyncio.sleep(0.1)

    pm.responses = [_model_calls_tool(name='step', ref='r1')]
    before = {id(task) for task in asyncio.all_tasks()}
    task = asyncio.create_task(
        ai.generate(prompt='go', tools=['step'], use=_four_hundred_noop_middleware()),
    )
    await started.wait()
    started_at = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleanup == ['tool']
    assert time.monotonic() - started_at >= 0.1
    await _assert_no_extra_tasks(before)


@pytest.mark.asyncio
async def test_generate_with_400_middleware_cancelled_on_a_later_turn_raises_cancelled_error() -> None:
    """Cancelling generate while a later turn's model hangs raises CancelledError and stops there."""
    ai = Genkit(model='hangOnTurn2Model')
    started = asyncio.Event()
    model_calls = 0

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    async def hang_on_turn_2(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return _model_calls_tool(name='step', ref='r1')
        started.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    ai.define_model(name='hangOnTurn2Model', fn=hang_on_turn_2)
    before = {id(task) for task in asyncio.all_tasks()}
    task = asyncio.create_task(
        ai.generate(
            prompt='go',
            tools=['step'],
            max_turns=10,
            use=_four_hundred_noop_middleware(),
        ),
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert model_calls == 2
    await _assert_no_extra_tasks(before)


@pytest.mark.asyncio
async def test_generate_stream_with_400_middleware_cancelled_while_the_model_hangs_raises_cancelled_error() -> None:
    """Cancelling the generate_stream response while the model hangs raises CancelledError."""
    ai = Genkit(model='hangModel')
    started = asyncio.Event()

    async def hang(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    ai.define_model(name='hangModel', fn=hang)
    before = {id(task) for task in asyncio.all_tasks()}
    stream = ai.generate_stream(prompt='go', use=_four_hundred_noop_middleware())
    await started.wait()
    response_task = stream.response
    assert isinstance(response_task, asyncio.Task)
    response_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await response_task
    await _assert_no_extra_tasks(before)


@pytest.mark.asyncio
async def test_generate_turn_0_middleware_sees_chunks_from_later_turns() -> None:
    """A streaming wrap_generate on turn 0 still intercepts chunks emitted by later turns."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    turn0_chunks: list[str] = []

    class Turn0ChunkSpy(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            if params.iteration != 0:
                return await next_fn(params, ctx)
            previous = ctx.on_chunk

            def handler(chunk: ModelResponseChunk) -> None:
                turn0_chunks.append(text_from_content(chunk.content))
                if previous is not None:
                    previous(chunk)

            ctx.replace_on_chunk(handler)
            try:
                return await next_fn(params, ctx)
            finally:
                ctx.replace_on_chunk(previous)

    @ai.tool(name='step')
    async def step() -> str:
        return 'ok'

    pm.responses = [
        _model_calls_tool(name='step', ref='r1'),
        _model_calls_tool(name='step', ref='r2'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('final')]),
        ),
    ]
    pm.chunks = [
        [ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('c0')])],
        [ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('c1')])],
        [ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('c2')])],
    ]

    stream = ai.generate_stream(prompt='go', tools=['step'], use=[Turn0ChunkSpy()], max_turns=5)
    _ = [chunk async for chunk in stream]
    response = await stream.response

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'final'
    assert 'c0' in turn0_chunks
    assert 'c1' in turn0_chunks
    assert 'c2' in turn0_chunks


@pytest.mark.asyncio
async def test_recovered_middleware_failure_uses_latest_closed_history() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    tool_calls = 0

    @ai.tool(name='lookup')
    async def lookup() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='recover_once')
    class RecoverOnce(BaseMiddleware[Config]):
        recovered = False

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            try:
                return await next_fn(params, ctx)
            except Exception:
                if self.recovered:
                    raise
                self.recovered = True
                return _model_calls_tool(name='lookup', ref='recovered-round')

    def always_fail(_request: ModelRequest) -> ModelResponse:
        if pm.request_count == 1:
            raise RuntimeError('transient failure')
        raise RuntimeError('later failure')

    pm.response_cb = always_fail
    response = await ai.generate(prompt='start', tools=['lookup'], use=[RecoverOnce()])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'recovered-round'
    assert _tool_output(response.messages[2]) == '72F'
    assert pm.request_count == 2
    assert tool_calls == 1


@pytest.mark.asyncio
async def test_abort_after_tool_turn_keeps_closed_rounds() -> None:
    """Abort after a tool returns: the closed round stays, the next model call does not run."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    abort_signal = asyncio.Event()

    @ai.tool(name='lookup')
    async def lookup() -> str:
        abort_signal.set()
        return '72F'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        _model_calls_tool(name='lookup', ref='r2'),
    ]

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('keep going')])],
            tools=['lookup'],
        ),
        abort_signal=abort_signal,
    )
    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Generation aborted.'
    assert response.error is not None
    assert response.error.status == 'CANCELLED'
    assert response.error.reason is None
    assert response.error.details is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_abort_during_first_tool_drops_unfinished_round() -> None:
    """Abort while the first tool is running: only the user message stays."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    abort_signal = asyncio.Event()
    started = asyncio.Event()

    @ai.tool(name='lookup')
    async def lookup() -> str:
        started.set()
        await abort_signal.wait()
        raise GenkitError(status='ABORTED', message='Task aborted')

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]

    async def run() -> ModelResponse:
        return await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='scriptedModel',
                messages=[Message(role=Role.USER, content=[Part.from_text('keep going')])],
                tools=['lookup'],
            ),
            abort_signal=abort_signal,
        )

    task = asyncio.create_task(run())
    await started.wait()
    abort_signal.set()
    response = await asyncio.wait_for(task, timeout=2.0)
    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message is not None
    assert 'aborted' in response.finish_message.lower()
    assert response.error is not None
    assert response.error.status == 'CANCELLED'
    assert response.error.reason is None
    assert response.error.details is not None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_already_cancelled_generate_returns_prompt() -> None:
    """A generate cancelled before it starts returns the prompt and does not call the model."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    abort_signal = asyncio.Event()
    abort_signal.set()

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('keep going')])],
        ),
        abort_signal=abort_signal,
    )
    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Generation aborted.'
    assert response.error is not None
    assert response.error.status == 'CANCELLED'
    assert response.error.reason is None
    assert response.error.details is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.messages[0].text == 'keep going'
    assert pm.request_count == 0

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )
    )
    continued = await ai.generate(messages=response.messages)
    assert continued.finish_reason == FinishReason.STOP
    assert continued.error is None
    assert continued.message is not None
    assert continued.messages[-1] == continued.message
    assert [m.role for m in continued.messages] == [Role.USER, Role.MODEL]
    assert continued.messages[0].text == 'keep going'
    assert continued.messages[-1].text == 'ok'


@pytest.mark.asyncio
async def test_already_cancelled_generate_returns_prior_messages() -> None:
    """A generate cancelled before it starts returns the messages they passed in."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    abort_signal = asyncio.Event()
    abort_signal.set()

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[
                Message(role=Role.USER, content=[Part.from_text('first')]),
                Message(role=Role.MODEL, content=[Part.from_text('ok')]),
                Message(role=Role.USER, content=[Part.from_text('again')]),
            ],
        ),
        abort_signal=abort_signal,
    )
    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Generation aborted.'
    assert response.error is not None
    assert response.error.status == 'CANCELLED'
    assert response.error.reason is None
    assert response.error.details is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.USER]
    assert response.messages[0].text == 'first'
    assert response.messages[1].text == 'ok'
    assert response.messages[2].text == 'again'
    assert pm.request_count == 0


@pytest.mark.asyncio
async def test_already_cancelled_unknown_model_returns() -> None:
    """Cancelled generate returns even when the model name would not resolve."""
    ai = Genkit()
    abort_signal = asyncio.Event()
    abort_signal.set()

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='nope/ghost',
            messages=[Message(role=Role.USER, content=[Part.from_text('keep going')])],
        ),
        abort_signal=abort_signal,
    )
    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Generation aborted.'
    assert response.error is not None
    assert response.error.status == 'CANCELLED'
    assert response.error.reason is None
    assert response.error.details is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.messages[0].text == 'keep going'


@pytest.mark.asyncio
async def test_abort_during_first_model_call_returns_prompt() -> None:
    """Abort while the first model call is running: only the user message stays."""
    ai = Genkit(model='hangingModel')
    abort_signal = asyncio.Event()
    started = asyncio.Event()

    async def hang(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        started.set()
        await abort_signal.wait()
        raise GenkitError(status='ABORTED', message='Generation aborted.')

    ai.define_model(name='hangingModel', fn=hang)

    async def run() -> ModelResponse:
        return await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='hangingModel',
                messages=[Message(role=Role.USER, content=[Part.from_text('keep going')])],
            ),
            abort_signal=abort_signal,
        )

    task = asyncio.create_task(run())
    await started.wait()
    abort_signal.set()
    response = await asyncio.wait_for(task, timeout=2.0)
    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Generation aborted.'
    assert response.error is not None
    assert response.error.status == 'CANCELLED'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.messages[0].text == 'keep going'


@pytest.mark.asyncio
async def test_abort_during_later_model_call_keeps_closed_round() -> None:
    """Abort while a later model call is running: the closed tool round stays."""
    ai = Genkit(model='hangAfterToolModel')
    abort_signal = asyncio.Event()
    second_started = asyncio.Event()
    model_calls = 0

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    async def hang_after_tool(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return _model_calls_tool(name='lookup', ref='r1')
        second_started.set()
        await abort_signal.wait()
        raise GenkitError(status='ABORTED', message='Generation aborted.')

    ai.define_model(name='hangAfterToolModel', fn=hang_after_tool)

    async def run() -> ModelResponse:
        return await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='hangAfterToolModel',
                messages=[Message(role=Role.USER, content=[Part.from_text('keep going')])],
                tools=['lookup'],
            ),
            abort_signal=abort_signal,
        )

    task = asyncio.create_task(run())
    await second_started.wait()
    abort_signal.set()
    response = await asyncio.wait_for(task, timeout=2.0)
    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Generation aborted.'
    assert response.error is not None
    assert response.error.status == 'CANCELLED'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'r1'
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_user_retry_middleware_makes_no_model_call_after_stop() -> None:
    """A user's retry loop that calls next_fn again after a stop does not bill another model call."""
    ai = Genkit()
    calls: list[str] = []

    async def flaky(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        calls.append('flaky')
        ctx.abort_signal.set()
        raise GenkitError(status='UNAVAILABLE', message='primary down')

    ai.define_model(name='flaky', fn=flaky)

    class MyRetry(BaseMiddleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            last: GenkitError | None = None
            for _ in range(3):
                try:
                    return await next_fn(params, ctx)
                except GenkitError as e:
                    last = e
            assert last is not None
            raise last

    response = await ai.generate(model='flaky', prompt='hi', use=[MyRetry()])

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Generation aborted.'
    assert response.error is not None
    assert response.error.status == 'CANCELLED'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]
    assert calls == ['flaky']


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['ABORTED', 'NOT_FOUND', 'INVALID_ARGUMENT', 'FAILED_PRECONDITION'])
async def test_provider_status_failure_keeps_closed_rounds(status: str) -> None:
    """A provider status is failure data after generation starts, not a setup error."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    seen = {'n': 0}

    def once_then_fail(_request: ModelRequest) -> ModelResponse:
        seen['n'] += 1
        if seen['n'] == 1:
            return _model_calls_tool(name='lookup', ref='r1')
        raise GenkitError(status=cast(Any, status), message='provider stopped')

    pm.response_cb = once_then_fail

    response = await ai.generate(prompt='keep going', tools=['lookup'])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'provider stopped' in response.finish_message
    assert response.error is not None
    assert response.error.status == status
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_middleware_failure_keeps_closed_rounds() -> None:
    """A generate hook that fails between turns returns the conversation entering that turn."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_second_turn')
    class DenySecondTurn(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            if params.iteration == 1:
                raise RuntimeError('hook denied')
            return await next_fn(params, ctx)

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[DenySecondTurn()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.details is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_middleware_rewrite_then_raise_keeps_one_model_turn() -> None:
    """A later wrap_generate that raises after an inner rewrite keeps one model message."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_model_says('secret')]

    class Config(BaseModel):
        pass

    @ai.middleware(name='redact')
    class Redact(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            result = await next_fn(params, ctx)
            return result.model_copy(
                update={
                    'message': Message(role=Role.MODEL, content=[Part.from_text('REDACTED')]),
                }
            )

    @ai.middleware(name='outer_boom')
    class OuterBoom(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            raise RuntimeError('outer failed')

    response = await ai.generate(prompt='hi', use=[OuterBoom(), Redact()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1].text == 'REDACTED'


@pytest.mark.asyncio
async def test_generate_middleware_genkit_error_keeps_closed_rounds() -> None:
    """A GenkitError from wrap_generate after a closed tool round is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_second_turn_typed')
    class DenySecondTurn(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            if params.iteration == 1:
                raise GenkitError(
                    status='FAILED_PRECONDITION',
                    message='hook denied',
                    reason=RuntimeErrorReason.INVALID_INPUT,
                )
            return await next_fn(params, ctx)

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[DenySecondTurn()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'hook denied'
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in (response.error.message or '')
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_first_turn_middleware_genkit_error_drops_unanswered_model() -> None:
    """A wrap_generate GenkitError after generate has entered is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('hi')]),
        )
    ]

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_first_turn')
    class DenyFirstTurn(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message='hook denied',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )

    response = await ai.generate(prompt='start', use=[DenyFirstTurn()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'hook denied'
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is RuntimeErrorReason.INVALID_INPUT
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_first_turn_middleware_runtime_error_drops_unanswered_model() -> None:
    """A wrap_generate RuntimeError after generate has entered is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('hi')]),
        )
    ]

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_first_turn_rt')
    class DenyFirstTurn(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            raise RuntimeError('hook denied')

    response = await ai.generate(prompt='start', use=[DenyFirstTurn()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_middleware_genkit_error_after_next_fn_keeps_closed_rounds() -> None:
    """A wrap_generate GenkitError after next_fn still leaves the closed tool round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_after_next_fn')
    class DenyAfterNextFn(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message='hook after closed',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[DenyAfterNextFn()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'hook after closed'
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in (response.error.message or '')
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_middleware_genkit_error_after_next_fn_keeps_model_turn() -> None:
    """A wrap_generate GenkitError after a completed model turn is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_after_model')
    class DenyAfterModel(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message='hook after closed',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[DenyAfterModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'hook after closed'
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is RuntimeErrorReason.INVALID_INPUT
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1].text == 'done'


@pytest.mark.asyncio
async def test_generate_middleware_validation_error_after_next_fn_keeps_model_turn() -> None:
    """A wrap_generate schema error after a completed model turn is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Recipe(BaseModel):
        title: str

    class Config(BaseModel):
        pass

    @ai.middleware(name='validate_after_model')
    class DenyAfterModel(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            Recipe.model_validate({'nope': 1})
            raise AssertionError('unreachable')

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[DenyAfterModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1].text == 'done'


@pytest.mark.asyncio
async def test_generate_middleware_validation_error_after_next_fn_keeps_closed_rounds() -> None:
    """A wrap_generate schema error after next_fn still leaves the closed tool round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Recipe(BaseModel):
        title: str

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='validate_after_closed')
    class DenyAfterNextFn(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            Recipe.model_validate({'nope': 1})
            raise AssertionError('unreachable')

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[DenyAfterNextFn()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_middleware_action_input_error_after_next_fn_keeps_model_turn() -> None:
    """An invalid model input error after a completed model turn is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_action_after_model')
    class DenyAfterModel(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message="Invalid input for model 'conforming': title missing",
                cause=ValidationError.from_exception_data('Recipe', []),
            )

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[DenyAfterModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Invalid input for model 'conforming'" in response.finish_message
    assert 'title missing' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1].text == 'done'


@pytest.mark.asyncio
async def test_generate_wrap_model_action_input_error_after_next_fn_drops_unanswered_model() -> None:
    """A wrap_model invalid model input after the model returned is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_model_after_next')
    class DenyAfterModel(BaseMiddleware[Config]):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message="Invalid input for model 'conforming': title missing",
                cause=ValidationError.from_exception_data('Recipe', []),
            )

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[DenyAfterModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Invalid input for model 'conforming'" in response.finish_message
    assert 'title missing' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_model_action_input_error_after_next_fn_keeps_closed_rounds() -> None:
    """A wrap_model invalid model input after next_fn still leaves the closed tool round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_model_after_closed')
    class DenyAfterModel(BaseMiddleware[Config]):
        seen = 0

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            self.seen += 1
            result = await next_fn(params, ctx)
            if self.seen > 1:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message="Invalid input for model 'conforming': title missing",
                    cause=ValidationError.from_exception_data('Recipe', []),
                )
            return result

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[DenyAfterModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Invalid input for model 'conforming'" in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_wrap_model_action_input_error_before_next_fn_drops_unanswered_model() -> None:
    """A wrap_model invalid model input after generate has entered is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_model_before_next')
    class DenyBeforeModel(BaseMiddleware[Config]):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message="Invalid input for model 'conforming': title missing",
                cause=ValidationError.from_exception_data('Recipe', []),
            )

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[DenyBeforeModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Invalid input for model 'conforming'" in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_model_action_input_error_before_next_fn_keeps_closed_rounds() -> None:
    """A wrap_model invalid model input before the next model call still leaves the closed tool round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_model_before_second')
    class DenyBeforeSecond(BaseMiddleware[Config]):
        seen = 0

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            self.seen += 1
            if self.seen > 1:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message="Invalid input for model 'conforming': title missing",
                    cause=ValidationError.from_exception_data('Recipe', []),
                )
            return await next_fn(params, ctx)

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[DenyBeforeSecond()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Invalid input for model 'conforming'" in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_wrap_model_action_input_error_after_short_circuit_next_fn_drops_unanswered_model() -> None:
    """A wrap_model Invalid input after a cache-style next_fn is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='cache_model')
    class CacheModel(BaseMiddleware[Config]):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part.from_text('short-circuit')]),
            )

    @ai.middleware(name='deny_after_cache')
    class DenyAfterCache(BaseMiddleware[Config]):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message="Invalid input for model 'conforming': title missing",
                cause=ValidationError.from_exception_data('Recipe', []),
            )

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('unused')]),
        )
    ]

    response = await ai.generate(prompt='hi', use=[DenyAfterCache(), CacheModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Invalid input for model 'conforming'" in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_generate_dict_after_next_fn_keeps_model_turn() -> None:
    """A wrap_generate dict after a completed model turn is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='dump_after_model')
    class DumpAfterModel(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            return {'text': 'nope'}  # type: ignore[return-value]

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[DumpAfterModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'did not return a ModelResponse' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1].text == 'done'


@pytest.mark.asyncio
async def test_generate_wrap_generate_dict_after_next_fn_keeps_closed_rounds() -> None:
    """A wrap_generate dict after next_fn still leaves the closed tool round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='dump_after_closed')
    class DumpAfterClosed(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            return {'text': 'nope'}  # type: ignore[return-value]

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[DumpAfterClosed()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'did not return a ModelResponse' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_wrap_generate_dict_before_next_fn_drops_unanswered_model() -> None:
    """A wrap_generate dict after generate has entered is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='dump_before_next')
    class DumpBeforeNext(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return {'text': 'nope'}  # type: ignore[return-value]

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[DumpBeforeNext()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'did not return a ModelResponse' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_generate_dict_after_short_circuit_next_fn_keeps_model_turn() -> None:
    """A wrap_generate dict after a cache-style next_fn is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='cache_generate')
    class CacheGenerate(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part.from_text('cached')]),
                usage=GenerationUsage(input_tokens=11, output_tokens=7, total_tokens=18),
            )

    @ai.middleware(name='dump_after_cache')
    class DumpAfterCache(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            return {'text': 'nope'}  # type: ignore[return-value]

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('unused')]),
        )
    ]

    response = await ai.generate(prompt='hi', use=[DumpAfterCache(), CacheGenerate()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'did not return a ModelResponse' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1].text == 'cached'
    assert response.usage is not None
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.total_tokens == 18


@pytest.mark.asyncio
async def test_generate_wrap_generate_action_input_error_after_short_circuit_next_fn_keeps_model_turn() -> None:
    """A wrap_generate Invalid input after a cache-style next_fn is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='cache_generate_input')
    class CacheGenerate(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part.from_text('cached')]),
                usage=GenerationUsage(input_tokens=11, output_tokens=7, total_tokens=18),
            )

    @ai.middleware(name='deny_after_cache_generate')
    class DenyAfterCache(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message="Invalid input for model 'conforming': title missing",
                cause=ValidationError.from_exception_data('Recipe', []),
            )

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('unused')]),
        )
    ]

    response = await ai.generate(prompt='hi', use=[DenyAfterCache(), CacheGenerate()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Invalid input for model 'conforming'" in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1].text == 'cached'
    assert response.usage is not None
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.total_tokens == 18


@pytest.mark.asyncio
async def test_generate_wrap_generate_dict_after_short_circuit_dict_next_fn_drops_unanswered_model() -> None:
    """A wrap_generate dict after a cache-style dict next_fn is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='cache_generate_dict')
    class CacheGenerate(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return {'text': 'cached-dict'}  # type: ignore[return-value]

    @ai.middleware(name='dump_after_cache_dict')
    class DumpAfterCache(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            return {'text': 'nope'}  # type: ignore[return-value]

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('unused')]),
        )
    ]

    response = await ai.generate(prompt='hi', use=[DumpAfterCache(), CacheGenerate()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'did not return a ModelResponse' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_model_dict_after_short_circuit_dict_next_fn_drops_unanswered_model() -> None:
    """A wrap_model dict after a cache-style dict next_fn is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='cache_model_dict')
    class CacheModel(BaseMiddleware[Config]):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return {'text': 'cached-dict'}  # type: ignore[return-value]

    @ai.middleware(name='dump_after_cache_model_dict')
    class DumpAfterCache(BaseMiddleware[Config]):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            return {'text': 'nope'}  # type: ignore[return-value]

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('unused')]),
        )
    ]

    response = await ai.generate(prompt='hi', use=[DumpAfterCache(), CacheModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'did not return a ModelResponse' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_model_dict_before_next_fn_drops_unanswered_model() -> None:
    """A wrap_model dict after generate has entered is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='dump_model_before_next')
    class DumpBeforeModel(BaseMiddleware[Config]):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return {'text': 'nope'}  # type: ignore[return-value]

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[DumpBeforeModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'did not return a ModelResponse' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_model_runtime_error_before_next_fn_drops_unanswered_model() -> None:
    """A wrap_model RuntimeError after generate has entered is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='boom_model_before_next')
    class BoomBeforeModel(BaseMiddleware[Config]):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            raise RuntimeError('boom')

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[BoomBeforeModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_model_dict_after_next_fn_drops_unanswered_model() -> None:
    """A wrap_model dict after the model returned is still resendable."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='dump_model_after_next')
    class DumpAfterModel(BaseMiddleware[Config]):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            return {'text': 'nope'}  # type: ignore[return-value]

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[DumpAfterModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'did not return a ModelResponse' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_middleware_action_input_error_after_next_fn_keeps_closed_rounds() -> None:
    """An invalid model input error after next_fn still leaves the closed tool round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='deny_action_after_closed')
    class DenyAfterNextFn(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message="Invalid input for model 'conforming': title missing",
                cause=ValidationError.from_exception_data('Recipe', []),
            )

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[DenyAfterNextFn()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Invalid input for model 'conforming'" in response.finish_message
    assert 'title missing' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_middleware_interrupt_after_next_fn_keeps_model_turn() -> None:
    """A wrap_generate Interrupt after a completed model turn is a dead turn, not a tool pause."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Config(BaseModel):
        pass

    @ai.middleware(name='interrupt_after_model')
    class InterruptAfterModel(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            raise Interrupt({'paused': True})

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]

    response = await ai.generate(prompt='keep going', use=[InterruptAfterModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1].text == 'done'


@pytest.mark.asyncio
async def test_generate_middleware_interrupt_after_next_fn_keeps_closed_rounds() -> None:
    """A wrap_generate Interrupt after next_fn still leaves the closed tool round."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='interrupt_after_closed')
    class InterruptAfterNextFn(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            await next_fn(params, ctx)
            raise Interrupt({'paused': True})

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        ),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[InterruptAfterNextFn()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_on_chunk_validation_error_returns_closed_history() -> None:
    """A dead on_chunk sink that raises ValidationError is still a dead pipe."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Recipe(BaseModel):
        title: str

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial')])]]

    def on_chunk(_: ModelResponseChunk) -> None:
        Recipe.model_validate({'nope': 1})

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        ),
        on_chunk=on_chunk,
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'title' in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'


@pytest.mark.asyncio
async def test_generate_format_parse_error_keeps_model_text() -> None:
    """A custom format that cannot parse the model text still returns that text."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Recipe(BaseModel):
        title: str

    class BoomFormat(FormatDef):
        def __init__(self) -> None:
            super().__init__('boom', FormatterConfig(format='json'))

        def handle(self, schema: dict[str, object] | None) -> Formatter[object, object]:
            def message_parser(_msg: Message) -> object:
                raise TypeError('parser exploded')

            def chunk_parser(_chunk: ModelResponseChunk) -> object:
                return None

            return Formatter(
                message_parser=message_parser,
                chunk_parser=chunk_parser,
                instructions=None,
            )

    ai.define_format(BoomFormat())
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('not a recipe')]),
        )
    ]

    response = await ai.generate(
        prompt='give me a recipe',
        output_schema=Recipe,
        output_format='boom',
        output_instructions=False,
    )
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'not a recipe'
    assert response.output is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert 'not valid JSON' in response.error.message
    assert response.error.details == {'reason': 'INVALID_OUTPUT'}
    assert response.message is not None
    assert response.message.text == 'not a recipe'
    assert response.messages[-1] == response.message
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]


@pytest.mark.asyncio
async def test_util_generate_dead_turn_paints_span_error(exporter) -> None:
    """Dev UI /util/generate after a dead turn must not look like a win on the action span."""

    class Recipe(BaseModel):
        title: str

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('not json')]),
        )
    ]

    action = await ai.registry.resolve_action(kind=ActionKind.UTIL, name='generate')
    assert action is not None
    action_response = await action.run(
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('give me a recipe')])],
            output=GenerateActionOutputConfig(
                json_schema=Recipe.model_json_schema(),
                schema_type=Recipe,
            ),
        )
    )
    response = cast(ModelResponse, action_response.response)
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'not json'
    assert response.error is not None
    generate_spans = [span for span in exporter.get_finished_spans() if span.name == 'generate']
    assert generate_spans
    attrs = generate_spans[-1].attributes or {}
    assert attrs.get('genkit:state') == 'error'


@pytest.mark.asyncio
async def test_failed_generate_marks_span_error(exporter) -> None:
    """ai.generate() returning a failed response marks its generate span genkit:state=error."""

    class Recipe(BaseModel):
        title: str

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('not json')]),
        )
    ]

    response = await ai.generate(prompt='give me a recipe', output_schema=Recipe)
    assert response.error is not None
    generate_spans = [span for span in exporter.get_finished_spans() if span.name == 'generate']
    assert generate_spans
    attrs = generate_spans[-1].attributes or {}
    assert attrs.get('genkit:type') == 'util'
    assert attrs.get('genkit:state') == 'error'


@pytest.mark.asyncio
async def test_generate_middleware_dropping_failure_still_keeps_closed_rounds() -> None:
    """A hook that discards an inner failure cannot discard its completed history."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return '72F'

    class Config(BaseModel):
        pass

    @ai.middleware(name='drop_failure_response')
    class DropFailureResponse(BaseMiddleware[Config]):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            response = await next_fn(params, ctx)
            if response.finish_reason == FinishReason.FAILED:
                raise RuntimeError('hook discarded failure')
            return response

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1'),
        _model_calls_tool(name='ghost', ref='r2'),
    ]

    response = await ai.generate(prompt='keep going', tools=['lookup'], use=[DropFailureResponse()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.details is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_returns_typed_output_when_schema_matches() -> None:
    """Matching JSON is parsed and handed back as the schema type."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class Recipe(BaseModel):
        title: str

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('{"title": "Soup"}')]),
        )
    ]

    response = await ai.generate(prompt='give me a recipe', output_schema=Recipe)
    assert response.finish_reason == FinishReason.STOP
    assert response.finish_message is None
    assert response.error is None
    assert response.output is not None
    assert response.output.title == 'Soup'
    assert response.message is not None
    assert response.message.text == '{"title": "Soup"}'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1] == response.message


@pytest.mark.asyncio
async def test_model_response_output_is_typed_optional() -> None:
    """ai.generate(output_schema=City).output is typed City | None.

    assert_type is a no-op at runtime; the ty CI job enforces it. The runtime
    INVALID_OUTPUT / output is None behavior is covered by the INVALID_OUTPUT tests.
    """

    class City(BaseModel):
        name: str
        population: int

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Paris is pretty big.')]),
        )
    ]

    response = await ai.generate(prompt='a city', output_schema=City)

    assert_type(response.output, City | None)


@pytest.mark.asyncio
async def test_generate_enum_off_list_is_error_finish() -> None:
    """An enum reply that is not one of the listed values is invalid output."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('MAYBE')]),
        )
    ]

    response = await ai.generate(
        prompt='classify',
        output_format='enum',
        output_schema={'type': 'string', 'enum': ['POSITIVE', 'NEGATIVE', 'NEUTRAL']},
    )
    assert response.finish_reason == FinishReason.STOP
    assert response.output is None
    assert response.text == 'MAYBE'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert response.error.details == {'reason': 'INVALID_OUTPUT'}
    assert response.message is not None
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1] == response.message


@pytest.mark.asyncio
async def test_generate_array_of_scalars() -> None:
    """A JSON array of strings is the output, not a parse miss."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('["a", "b"]')]),
        )
    ]

    response = await ai.generate(
        prompt='list',
        output_format='array',
        output_schema={'type': 'array', 'items': {'type': 'string'}},
    )
    assert response.finish_reason == FinishReason.STOP
    assert response.finish_message is None
    assert response.error is None
    assert response.output == ['a', 'b']


@pytest.mark.asyncio
async def test_generate_unknown_format_is_invalid_argument() -> None:
    """An unresolved output format fails before the model is called."""
    ai = Genkit(model='scriptedModel')
    define_scripted_model(ai)

    with pytest.raises(GenkitError, match='Unable to resolve format') as raised:
        await ai.generate(prompt='hi', output_format='no-such-format')
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'ACTION_NOT_FOUND' not in raised.value.original_message


@pytest.mark.asyncio
async def test_unknown_middleware_raises_with_invalid_input() -> None:
    """A middleware name that is not registered is a bad argument, not a missing action."""
    ai = Genkit(model='scriptedModel')
    define_scripted_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='hi', use=[MiddlewareRef(name='ghost')])
    error = raised.value
    assert error.status == 'NOT_FOUND'
    assert error.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'ghost' in error.original_message
    assert 'ACTION_NOT_FOUND' not in error.original_message


@pytest.mark.asyncio
async def test_unknown_model_raises_with_model_not_found() -> None:
    """A missing model raises; the message stays human and reason is MODEL_NOT_FOUND."""
    ai = Genkit()

    with pytest.raises(GenkitError) as raised:
        await ai.generate(model='nope/ghost', prompt='hi')
    error = raised.value
    assert error.status == 'NOT_FOUND'
    assert error.reason is RuntimeErrorReason.MODEL_NOT_FOUND
    assert "Failed to resolve model 'nope/ghost'" in error.original_message
    assert 'MODEL_NOT_FOUND' not in error.original_message


@pytest.mark.asyncio
async def test_unknown_tool_on_request_raises_with_tool_not_found() -> None:
    """A tool name that is not registered raises before generate starts."""
    ai = Genkit()
    define_echo_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(model='echoModel', prompt='hi', tools=['ghost'])
    error = raised.value
    assert error.status == 'NOT_FOUND'
    assert error.reason is RuntimeErrorReason.TOOL_NOT_FOUND
    assert 'Unable to resolve tool ghost' in error.original_message
    assert 'TOOL_NOT_FOUND' not in error.original_message


@pytest.mark.asyncio
async def test_generate_with_tool_ref_slash_retriever_raises_tool_not_found() -> None:
    """generate(tools=['/retriever/x']) raises NOT_FOUND 'Unable to resolve tool /retriever/x'."""
    ai = Genkit()
    define_echo_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(model='echoModel', prompt='hi', tools=['/retriever/x'])
    error = raised.value
    assert error.status == 'NOT_FOUND'
    assert error.reason is RuntimeErrorReason.TOOL_NOT_FOUND
    assert error.original_message == 'Unable to resolve tool /retriever/x'


@pytest.mark.asyncio
async def test_generate_without_model_or_default_raises_model_not_found() -> None:
    """generate() with no model and no constructor default is MODEL_NOT_FOUND."""
    ai = Genkit()

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='hi')
    error = raised.value
    assert error.status == 'INVALID_ARGUMENT'
    assert error.reason is RuntimeErrorReason.MODEL_NOT_FOUND
    assert 'No model configured' in error.original_message
    assert 'MODEL_NOT_FOUND' not in error.original_message


@pytest.mark.asyncio
async def test_generate_jsonl_with_object_schema_raises_invalid_schema() -> None:
    """jsonl needs an array of objects; a lone object schema is INVALID_SCHEMA."""
    ai = Genkit(model='scriptedModel')
    define_scripted_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='hi', output_format='jsonl', output_schema={'type': 'object'})
    error = raised.value
    assert error.status == 'INVALID_ARGUMENT'
    assert error.reason is RuntimeErrorReason.INVALID_SCHEMA
    assert 'jsonl' in error.original_message
    assert 'INVALID_SCHEMA' not in error.original_message


@pytest.mark.asyncio
async def test_generate_enum_with_object_schema_raises_invalid_schema() -> None:
    """enum needs a string schema; an object schema is INVALID_SCHEMA."""
    ai = Genkit(model='scriptedModel')
    define_scripted_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='hi', output_format='enum', output_schema={'type': 'object'})
    error = raised.value
    assert error.status == 'INVALID_ARGUMENT'
    assert error.reason is RuntimeErrorReason.INVALID_SCHEMA
    assert 'enum' in error.original_message
    assert 'INVALID_SCHEMA' not in error.original_message


@pytest.mark.asyncio
async def test_generate_array_with_object_schema_raises_invalid_schema() -> None:
    """array format needs an array schema; an object schema is INVALID_SCHEMA."""
    ai = Genkit(model='scriptedModel')
    define_scripted_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='hi', output_format='array', output_schema={'type': 'object'})
    error = raised.value
    assert error.status == 'INVALID_ARGUMENT'
    assert error.reason is RuntimeErrorReason.INVALID_SCHEMA
    assert 'array' in error.original_message
    assert 'INVALID_SCHEMA' not in error.original_message


@pytest.mark.asyncio
async def test_generate_output_returns_none_on_plain_text_reply() -> None:
    """Reading .output on a plain conversational turn yields None without throwing."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Hello world')]),
        )
    ]

    response = await ai.generate(prompt='hi')
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'Hello world'
    assert response.output is None
    assert response.error is None


@pytest.mark.asyncio
async def test_generate_four_output_scenarios_with_schema() -> None:
    """Validate all four output scenarios when output_schema is provided:
    1. Returns text only -> dead-turn with INVALID_OUTPUT
    2. Returns text + unparseable output -> dead-turn with INVALID_OUTPUT
    3. Returns text + parseable output -> extracts structured output, error is None
    4. Returns parseable output only -> extracts structured output, error is None
    """

    class Dish(BaseModel):
        dish: str

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    raw_text_only = 'Sorry, I cannot help with that.'
    raw_text_unparseable = 'Here is your recipe:\n```json\n{dish: bad_json\n```\nHope you like it!'
    raw_text_parseable = 'Here is your recipe:\n```json\n{"dish": "Smoked Salmon Tartine"}\n```\nHope you like it!'
    raw_parseable_only = '{"dish": "Smoked Salmon Tartine"}'

    # 1. returns text only
    pm.reset()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text(raw_text_only)]),
        )
    ]
    r1 = await ai.generate(prompt='give me dish', output_schema=Dish)
    assert r1.finish_reason == FinishReason.STOP
    assert r1.text == raw_text_only
    assert r1.output is None
    assert r1.finish_message is None
    assert r1.error is not None
    assert r1.error.status == 'INTERNAL'
    assert r1.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert r1.error.details == {'reason': 'INVALID_OUTPUT'}
    assert 'not valid JSON' in r1.error.message
    assert r1.messages[-1].text == raw_text_only

    # 2. returns text + unparseable output
    pm.reset()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text(raw_text_unparseable)]),
        )
    ]
    r2 = await ai.generate(prompt='give me dish', output_schema=Dish)
    assert r2.finish_reason == FinishReason.STOP
    assert r2.text == raw_text_unparseable
    assert r2.output is None
    assert r2.finish_message is None
    assert r2.error is not None
    assert r2.error.status == 'INTERNAL'
    assert r2.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert r2.error.details == {'reason': 'INVALID_OUTPUT'}
    assert 'not valid JSON' in r2.error.message
    assert r2.messages[-1].text == raw_text_unparseable

    # 3. returns text + parseable output
    pm.reset()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text(raw_text_parseable)]),
        )
    ]
    r3 = await ai.generate(prompt='give me dish', output_schema=Dish)
    assert r3.finish_reason == FinishReason.STOP
    assert r3.text == raw_text_parseable
    assert r3.output == Dish(dish='Smoked Salmon Tartine')
    assert r3.finish_message is None
    assert r3.error is None
    assert r3.messages[-1].text == raw_text_parseable

    # 4. returns parseable output only
    pm.reset()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text(raw_parseable_only)]),
        )
    ]
    r4 = await ai.generate(prompt='give me dish', output_schema=Dish)
    assert r4.finish_reason == FinishReason.STOP
    assert r4.text == raw_parseable_only
    assert r4.output == Dish(dish='Smoked Salmon Tartine')
    assert r4.finish_message is None
    assert r4.error is None
    assert r4.messages[-1].text == raw_parseable_only


@pytest.mark.asyncio
async def test_generate_four_output_scenarios_with_format_json_no_schema() -> None:
    """Validate all four output scenarios when format='json' is requested without schema:
    1. Returns text only -> dead-turn with INVALID_OUTPUT
    2. Returns text + unparseable output -> dead-turn with INVALID_OUTPUT
    3. Returns text + parseable output -> extracts dict, error is None
    4. Returns parseable output only -> extracts dict, error is None
    """
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    raw_text_only = 'Sorry, I cannot help with that.'
    raw_text_unparseable = 'Here is your recipe:\n```json\n{dish: bad_json\n```\nHope you like it!'
    raw_text_parseable = 'Here is your recipe:\n```json\n{"dish": "Smoked Salmon Tartine"}\n```\nHope you like it!'
    raw_parseable_only = '{"dish": "Smoked Salmon Tartine"}'

    # 1. returns text only
    pm.reset()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text(raw_text_only)]),
        )
    ]
    r1 = await ai.generate(prompt='give me dish', output_format='json')
    assert r1.finish_reason == FinishReason.STOP
    assert r1.text == raw_text_only
    assert r1.output is None
    assert r1.finish_message is None
    assert r1.error is not None
    assert r1.error.status == 'INTERNAL'
    assert r1.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert r1.error.details == {'reason': 'INVALID_OUTPUT'}
    assert 'not valid JSON' in r1.error.message
    assert r1.messages[-1].text == raw_text_only

    # 2. returns text + unparseable output
    pm.reset()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text(raw_text_unparseable)]),
        )
    ]
    r2 = await ai.generate(prompt='give me dish', output_format='json')
    assert r2.finish_reason == FinishReason.STOP
    assert r2.text == raw_text_unparseable
    assert r2.output is None
    assert r2.finish_message is None
    assert r2.error is not None
    assert r2.error.status == 'INTERNAL'
    assert r2.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert r2.error.details == {'reason': 'INVALID_OUTPUT'}
    assert 'not valid JSON' in r2.error.message
    assert r2.messages[-1].text == raw_text_unparseable

    # 3. returns text + parseable output
    pm.reset()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text(raw_text_parseable)]),
        )
    ]
    r3 = await ai.generate(prompt='give me dish', output_format='json')
    assert r3.finish_reason == FinishReason.STOP
    assert r3.text == raw_text_parseable
    assert r3.output == {'dish': 'Smoked Salmon Tartine'}
    assert r3.finish_message is None
    assert r3.error is None
    assert r3.messages[-1].text == raw_text_parseable

    # 4. returns parseable output only
    pm.reset()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text(raw_parseable_only)]),
        )
    ]
    r4 = await ai.generate(prompt='give me dish', output_format='json')
    assert r4.finish_reason == FinishReason.STOP
    assert r4.text == raw_parseable_only
    assert r4.output == {'dish': 'Smoked Salmon Tartine'}
    assert r4.finish_message is None
    assert r4.error is None
    assert r4.messages[-1].text == raw_parseable_only


@pytest.mark.asyncio
async def test_generate_array_format_no_schema_unparseable_records_invalid_output() -> None:
    """format='array' without schema marks INVALID_OUTPUT when model emits unparseable prose."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Sorry, I cannot do that.')]),
        )
    ]

    response = await ai.generate(prompt='give me list', output_format='array')
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'Sorry, I cannot do that.'
    assert response.output is None
    assert response.finish_message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert response.error.details == {'reason': 'INVALID_OUTPUT'}
    assert response.messages[-1].text == 'Sorry, I cannot do that.'


@pytest.mark.asyncio
async def test_generate_array_format_object_reply_records_invalid_output() -> None:
    """format='array' marks INVALID_OUTPUT when model returns a JSON object instead of a list."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('{"dish": "Tartine"}')]),
        )
    ]

    response = await ai.generate(prompt='give me list', output_format='array')
    assert response.finish_reason == FinishReason.STOP
    assert response.text == '{"dish": "Tartine"}'
    assert response.output is None
    assert response.finish_message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert response.error.details == {'reason': 'INVALID_OUTPUT'}
    assert 'not valid for the requested format' in response.error.message
    assert response.messages[-1].text == '{"dish": "Tartine"}'


@pytest.mark.asyncio
async def test_generate_json_format_blocked_preserves_blocked_reason() -> None:
    """An abnormal finish like BLOCKED keeps its finish reason and does not mark INVALID_OUTPUT."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.BLOCKED,
            message=Message(role=Role.MODEL, content=[]),
        )
    ]

    response = await ai.generate(prompt='sensitive', output_format='json')
    assert response.finish_reason == FinishReason.BLOCKED
    assert response.output is None
    assert response.error is None


@pytest.mark.asyncio
async def test_generate_json_format_with_tool_call_validates_only_final_turn() -> None:
    """Intermediate tool request turns do not fail format validation before final reply."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup(query: str) -> str:
        return 'special ingredient'

    pm.responses = [
        _model_calls_tool(name='lookup', ref='r1', input='secret'),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('{"result": "special ingredient"}')]),
        ),
    ]

    response = await ai.generate(
        prompt='find it and output json',
        output_format='json',
        tools=['lookup'],
    )
    assert response.finish_reason == FinishReason.STOP
    assert response.finish_message is None
    assert response.error is None
    assert response.output == {'result': 'special ingredient'}
    assert response.messages[-1].text == '{"result": "special ingredient"}'


@pytest.mark.asyncio
async def test_generate_with_failing_tool_still_returns_internal_error() -> None:
    """A tool raising ValueError still fails the response with INTERNAL and `finish_message == 'internal error'`."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        raise ValueError('db password is hunter2')

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]

    response = await ai.generate(prompt='weather?', tools=['lookup'])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.message == 'internal error'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_with_public_error_tool_returns_its_message() -> None:
    """A tool raising PublicError still puts its message on `finish_message`."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        raise PublicError('NOT_FOUND', 'no forecast for Atlantis')

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]

    response = await ai.generate(prompt='weather?', tools=['lookup'])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'no forecast for Atlantis'
    assert response.error is not None
    assert response.error.status == 'NOT_FOUND'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_with_interrupting_tool_still_returns_interrupted() -> None:
    """A tool raising Interrupt still yields an interrupted response with the tool request."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='transfer')
    async def transfer() -> str:
        raise Interrupt({'reason': 'needs_approval'})

    pm.responses = [_model_calls_tool(name='transfer', ref='r1')]

    response = await ai.generate(prompt='send $100', tools=['transfer'])

    assert response.finish_reason == FinishReason.INTERRUPTED
    assert response.message is not None
    assert response.messages[-1] == response.message
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    [interrupt] = response.interrupts
    assert interrupt.tool_request is not None
    assert interrupt.tool_request.ref == 'r1'
    assert interrupt.metadata is not None
    assert interrupt.metadata['interrupt'] == {'reason': 'needs_approval'}


@pytest.mark.asyncio
async def test_generate_with_failing_streaming_callback_returns_callback_message() -> None:
    """An `on_chunk` that raises still yields its own message on the failed response."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial')])]]

    def on_chunk(_: ModelResponseChunk) -> None:
        raise RuntimeError('model sink closed')

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        ),
        on_chunk=on_chunk,
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'model sink closed'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_with_model_raising_wrapped_provider_500_returns_internal_error() -> None:
    """A model raising wrap_http_error(..., status_code=500) keeps provider text off finish_message."""
    ai = Genkit()

    async def down(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        raise wrap_http_error(
            RuntimeError('upstream said: db password rejected'),
            status_code=500,
        )

    ai.define_model(name='down', fn=down)

    response = await ai.generate(model='down', prompt='hi')

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert 'password' not in (response.finish_message or '')


@pytest.mark.asyncio
async def test_generate_with_genkit_error_wrapping_public_error_uses_outer_message() -> None:
    """A hand-built GenkitError(cause=PublicError) uses the outer message, not the inner one."""
    ai = Genkit()

    async def down(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(
            status='INTERNAL',
            message='outer wrapper',
            cause=PublicError('NOT_FOUND', 'no order 99'),
        )

    ai.define_model(name='down', fn=down)

    response = await ai.generate(model='down', prompt='hi')

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'outer wrapper'
    assert 'no order 99' not in (response.finish_message or '')
    assert response.error is not None
    assert response.error.status == 'INTERNAL'


class _City(BaseModel):
    name: str
    population: int


def _reply(text: str, finish_reason: FinishReason = FinishReason.STOP) -> ModelResponse:
    return ModelResponse(
        finish_reason=finish_reason,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
    )


@pytest.mark.asyncio
async def test_generate_output_schema_cut_off_with_length_returns_none() -> None:
    """A reply cut off at `{"name":"X","population": 21` with finish LENGTH gives `output is None` and INVALID_OUTPUT.

    `finish_reason` stays LENGTH, so the caller can tell a token cap from bad JSON.
    """
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('{"name":"X","population": 21', FinishReason.LENGTH)]

    response = await ai.generate(prompt='a city', output_schema=_City)

    assert response.output is None
    assert response.finish_reason == FinishReason.LENGTH
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert response.text == '{"name":"X","population": 21'


@pytest.mark.asyncio
async def test_generate_output_schema_cut_off_with_stop_returns_none() -> None:
    """The same cut-off reply with finish STOP also gives `output is None` and INVALID_OUTPUT."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('{"name":"X","population": 21')]

    response = await ai.generate(prompt='a city', output_schema=_City)

    assert response.output is None
    assert response.finish_reason == FinishReason.STOP
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT


@pytest.mark.asyncio
async def test_generate_output_schema_complete_json_with_length_returns_instance() -> None:
    """A complete object that also hit the token cap is still a City with no error."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('{"name":"X","population": 2100}', FinishReason.LENGTH)]

    response = await ai.generate(prompt='a city', output_schema=_City)

    assert response.output == _City(name='X', population=2100)
    assert response.error is None
    assert response.finish_reason == FinishReason.LENGTH


@pytest.mark.asyncio
async def test_generate_output_in_code_fence_returns_instance() -> None:
    """A complete object inside a ```json fence with prose around it still parses to a City."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('Here you go:\n```json\n{"name": "Paris", "population": 2100000}\n```\nEnjoy!')]

    response = await ai.generate(prompt='a city', output_schema=_City)

    assert response.output == _City(name='Paris', population=2100000)
    assert response.error is None


@pytest.mark.asyncio
async def test_generate_json_format_cut_off_returns_none() -> None:
    """`output_format='json'` with no schema gives `output is None` and INVALID_OUTPUT for a cut-off object."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('{"a": 1, "b": [1, 2', FinishReason.LENGTH)]

    response = await ai.generate(prompt='json please', output_format='json')

    assert response.output is None
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT


@pytest.mark.asyncio
async def test_generate_json_format_complete_returns_dict() -> None:
    """`output_format='json'` with no schema returns the dict for a complete object."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('{"a": 1, "b": [1, 2]}')]

    response = await ai.generate(prompt='json please', output_format='json')

    assert response.output == {'a': 1, 'b': [1, 2]}
    assert response.error is None


@pytest.mark.asyncio
async def test_generate_json_format_brace_in_prose_returns_later_object() -> None:
    """generate(output_format='json') returns {'a': 1} when the model replies `Fill in {name}: {"a": 1}`."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('Fill in {name}: {"a": 1}')]

    response = await ai.generate(prompt='fill the template', output_format='json')

    assert response.output == {'a': 1}
    assert response.error is None


@pytest.mark.asyncio
async def test_generate_output_schema_brace_in_prose_wrong_shape_returns_none() -> None:
    """generate(output_schema=City) on `{x} {"other": 1}` gives output is None and INVALID_OUTPUT."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('{x} {"other": 1}')]

    response = await ai.generate(prompt='a city', output_schema=_City)

    assert response.output is None
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT


@pytest.mark.asyncio
async def test_generate_output_schema_cut_off_with_length_error_says_cut_off() -> None:
    """generate(output_schema=City) cut off at the token limit says so in the INVALID_OUTPUT message."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('{"name":"X","population": 21', FinishReason.LENGTH)]

    response = await ai.generate(prompt='a city', output_schema=_City)

    assert response.output is None
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert 'cut off at the token limit' in response.error.message
    assert 'finish_reason=length' in response.error.message


@pytest.mark.asyncio
async def test_generate_output_schema_bad_json_with_stop_error_says_not_valid_json() -> None:
    """generate(output_schema=City) on `{not json}` with finish stop keeps the not-valid-JSON message."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('{not json}')]

    response = await ai.generate(prompt='a city', output_schema=_City)

    assert response.output is None
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert 'not valid JSON' in response.error.message


@pytest.mark.asyncio
async def test_generate_array_format_cut_off_keeps_finished_items() -> None:
    """generate(output_format='array') keeps finished items when the model hits the token cap mid-item."""
    text = '[{"a":1},{"a":2},{"a":'
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply(text, FinishReason.LENGTH)]

    response = await ai.generate(prompt='list 50 recipes', output_format='array')

    assert response.output == [{'a': 1}, {'a': 2}]
    assert response.error is None
    assert response.finish_reason == FinishReason.LENGTH

    stream_ai = Genkit(model='scriptedModel')
    stream_pm, _ = define_scripted_model(stream_ai)
    stream_pm.responses = [_reply(text, FinishReason.LENGTH)]
    stream_pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(text)])]]
    stream = stream_ai.generate_stream(prompt='list 50 recipes', output_format='array')
    chunks = [chunk.output async for chunk in stream.stream]
    streamed = await stream.response
    assert streamed.output == response.output
    assert chunks[-1] == response.output
    assert streamed.finish_reason == FinishReason.LENGTH
    assert streamed.error is None


@pytest.mark.asyncio
async def test_generate_array_format_cut_off_before_first_item_returns_none() -> None:
    """generate(output_format='array') gives output is None and INVALID_OUTPUT when the model stops at `[{"a":`."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('[{"a":', FinishReason.LENGTH)]

    response = await ai.generate(prompt='list 50 recipes', output_format='array')

    assert response.output is None
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert response.finish_reason == FinishReason.LENGTH


@pytest.mark.asyncio
async def test_generate_array_format_string_items_returns_list() -> None:
    """generate(output_format='array') returns ["a", "b"] when the model replies with that array."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('["a", "b"]')]

    response = await ai.generate(prompt='a list', output_format='array')

    assert response.output == ['a', 'b']
    assert response.error is None


@pytest.mark.asyncio
async def test_generate_jsonl_format_drops_cut_off_last_line() -> None:
    """`output_format='jsonl'` keeps the complete lines and drops a cut-off last line."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('{"name": "a"}\n{"name": "b"}\n{"name": "c', FinishReason.LENGTH)]

    response = await ai.generate(prompt='lines', output_format='jsonl')

    assert response.output == [{'name': 'a'}, {'name': 'b'}]
    assert response.error is None


@pytest.mark.asyncio
async def test_generate_custom_format_parser_output_is_used_for_cut_off_reply() -> None:
    """A custom FormatDef whose parser closes a cut-off object still gets its City back."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    class ClosingFormat(FormatDef):
        def __init__(self) -> None:
            super().__init__('closing', FormatterConfig(format='json'))

        def handle(self, schema: dict[str, object] | None) -> Formatter[object, object]:
            return Formatter(
                message_parser=lambda msg: json.loads(msg.text + '}'),
                chunk_parser=lambda _chunk: None,
                instructions=None,
            )

    ai.define_format(ClosingFormat())
    pm.responses = [_reply('{"name":"X","population": 21', FinishReason.LENGTH)]

    response = await ai.generate(
        prompt='a city',
        output_schema=_City,
        output_format='closing',
        output_instructions=False,
    )

    assert response.output == _City(name='X', population=21)
    assert response.error is None
