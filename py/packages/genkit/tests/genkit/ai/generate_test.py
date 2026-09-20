#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import asyncio
import json
import pathlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

import pytest
import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

from genkit import ActionKind, Document, Genkit, Message, MiddlewareRef, ModelResponse, ModelResponseChunk, Part
from genkit._ai._formats._types import FormatDef, Formatter, FormatterConfig
from genkit._ai._generate import ChunkAccumulator, augment_with_context, generate_action
from genkit._ai._model import text_from_content, text_from_message
from genkit._ai._resource import ResourceInput, ResourceOutput, define_resource
from genkit._ai._testing import (
    ProgrammableModel,
    define_echo_model,
    define_programmable_model,
)
from genkit._ai._tools import Interrupt, ToolRunContext, define_tool, restart_tool
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError, PublicError, RuntimeErrorReason
from genkit._core._model import GenerateActionOptions, ModelRequest, Resume
from genkit._core._registry import Registry
from genkit._core._typing import (
    FinishReason,
    GenerateActionOutputConfig,
    GenerationUsage,
    Resource1,
    Role,
    ToolRequest,
)
from genkit.middleware import (
    BaseMiddleware,
    GenerateHookParams,
    GenerateMiddleware,
    GenerateMiddlewareContext,
    ModelHookParams,
    MultipartToolResponse,
    ToolHookParams,
)
from genkit.plugin_api import MiddlewarePlugin, new_middleware


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
def setup_test() -> tuple[Genkit, ProgrammableModel]:
    """Setup the test."""
    ai = Genkit()

    pm, _ = define_programmable_model(ai)

    @ai.tool(name='testTool')
    async def test_tool() -> object:
        """description"""  # noqa: D403, D415
        return 'tool called'

    return (ai, pm)


@pytest.mark.asyncio
async def test_simple_text_generate_request(
    setup_test: tuple[Genkit, ProgrammableModel],
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
            model='programmableModel',
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
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    ai, pm = setup_test
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('hello')]),
        )
    )

    response = await ai.generate(
        model='programmableModel',
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
    )

    assert response.message is not None
    assert response.message.content[0].text == 'hello'
    assert response.message.content[0].media is None


@pytest.mark.asyncio
async def test_generate_user_text_and_media_model_sees_both_parts(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    ai, pm = setup_test
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )
    )

    await ai.generate(
        model='programmableModel',
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


@pytest.mark.asyncio
async def test_generate_stream_chunk_text_from_factory_part(
    setup_test: tuple[Genkit, ProgrammableModel],
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

    stream_result = ai.generate_stream(model='programmableModel', prompt='do it')
    texts: list[str] = []
    async for chunk in stream_result.stream:
        texts.append(chunk.text)
    assert texts == ['h', 'i']


@pytest.mark.asyncio
async def test_simulates_doc_grounding(
    setup_test: tuple[Genkit, ProgrammableModel],
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
            model='programmableModel',
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

    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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

    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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
    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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
            model='programmableModel',
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
    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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
    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            tools=['ctxTool'],
        ),
        context={'user_id': 'u-123'},
    )

    assert response.text == 'done'
    assert seen == [{'user_id': 'u-123'}]


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

    pm, _ = define_programmable_model(ai)
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('all done')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
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

    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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

    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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

    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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
    """Middleware.tools() contributes actions scoped to the generate call (child registry).

    The contributed tool is resolvable by the model during the call but must not
    appear in the root registry afterward — mirroring Go's Hooks.Tools + NewChild.
    """

    ai = Genkit()

    @ai.middleware(name='tool_provider_mw')
    class ToolProviderMiddleware(BaseMiddleware):
        """Middleware that contributes a tool dynamically per generate() call."""

        def tools(self, ctx: GenerateMiddlewareContext) -> list:
            # Build a tool action on a throw-away registry; the generate engine
            # will adopt it into a call-scoped child registry.
            scratch = Registry()

            async def provided_tool() -> str:
                """A tool injected by middleware."""
                return 'from_middleware_tool'

            t = define_tool(scratch, provided_tool, name='middleware_tool')
            return [t.action()]

    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            use=[MiddlewareRef(name='tool_provider_mw')],
        ),
    )
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
        def tools(self, ctx: GenerateMiddlewareContext) -> list:
            scratch = Registry()

            async def ping() -> str:
                return 'from_mw'

            return [define_tool(scratch, ping, name='ping').action()]

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
            def tools(self, ctx: GenerateMiddlewareContext) -> list:
                scratch = Registry()

                async def ping() -> str:
                    return name

                return [define_tool(scratch, ping, name='ping').action()]

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
        def tools(self, ctx: GenerateMiddlewareContext) -> list:
            scratch = Registry()

            async def shared_tool() -> str:
                """Shared by all middleware in the call."""
                return 'shared_ok'

            return [define_tool(scratch, shared_tool, name='shared_tool').action()]

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

    pm, _ = define_programmable_model(ai)
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
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

    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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

    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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
    ai = Genkit(model='programmableModel')

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

    pm, _ = define_programmable_model(ai)

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
        resume_restart=restart_tool(interrupt=interrupt_part),
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
        resume_restart=restart_tool(
            interrupt=response.interrupts[0],
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
        resume_restart=restart_tool(interrupt=first.interrupts[0]),
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
    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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
    pm, _ = define_programmable_model(ai)

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
        model='programmableModel',
        prompt='call it',
        tools=[inline_tool],
    )

    assert response.text == 'after_inline'
    assert await ai.registry.resolve_action(ActionKind.TOOL, 'inline_yell') is None


@pytest.mark.asyncio
async def test_parallel_tool_requests_one_interrupt_keeps_pending_output_for_others(
    setup_test: tuple[Genkit, ProgrammableModel],
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
            model='programmableModel',
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
    pm, _ = define_programmable_model(ai)

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

    # The programmable model appends to execution_order when called.
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
            model='programmableModel',
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
    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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
            scratch = Registry()

            async def injected_tool() -> str:
                """A tool contributed by middleware."""
                return 'injected_success'

            return [define_tool(scratch, injected_tool, name='injectedTool').action()]

    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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

    pm, _ = define_programmable_model(ai)

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
    setup_test: tuple[Genkit, ProgrammableModel],
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
        GenerateActionOptions(model='programmableModel', messages=[]),
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    define_programmable_model(ai)

    with pytest.raises(GenkitError, match='max turns cannot be negative, got -1') as raised:
        await ai.generate(prompt='hi', max_turns=-1)
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in raised.value.original_message


@pytest.mark.asyncio
async def test_generate_max_turns_zero_drops_first_tool_round() -> None:
    """A cap of zero is zero tool rounds, not the omitted default."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
async def test_default_max_turns_allows_fifty_tool_rounds_to_finish() -> None:
    """The default leaves room for fifty completed tool rounds and a final reply."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    tool_calls = 0

    @ai.tool(name='step')
    async def step() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return 'ok'

    pm.responses = [
        *[_model_calls_tool(name='step', ref=str(index)) for index in range(50)],
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        ),
    ]

    response = await ai.generate(prompt='finish the work', tools=['step'])

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'done'
    assert response.error is None
    assert tool_calls == 50
    assert pm.request_count == 51
    assert [message.role for message in response.messages] == [
        Role.USER,
        *[role for _ in range(50) for role in (Role.MODEL, Role.TOOL)],
        Role.MODEL,
    ]
    assert [_tool_request(response.messages[index]).ref for index in range(1, 101, 2)] == [
        str(index) for index in range(50)
    ]
    assert [_tool_output(response.messages[index]) for index in range(2, 102, 2)] == ['ok'] * 50
    assert response.messages[-1] == response.message


@pytest.mark.asyncio
async def test_default_max_turns_drops_fifty_first_tool_round() -> None:
    """The fifty-first open tool round is excluded from resendable history."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    tool_calls = 0

    @ai.tool(name='step')
    async def step() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return 'ok'

    pm.responses = [_model_calls_tool(name='step', ref=str(index)) for index in range(51)]

    response = await ai.generate(prompt='keep going', tools=['step'])

    assert response.finish_reason == FinishReason.ABORTED
    assert response.finish_message == 'Exceeded maximum tool call iterations (50)'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'ABORTED'
    assert response.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert response.error.details == {'reason': 'MAX_TURNS_EXCEEDED'}
    assert response.error.message == response.finish_message
    assert tool_calls == 50
    assert pm.request_count == 51
    assert [message.role for message in response.messages] == [
        Role.USER,
        *[role for _ in range(50) for role in (Role.MODEL, Role.TOOL)],
    ]
    assert [_tool_request(response.messages[index]).ref for index in range(1, 101, 2)] == [
        str(index) for index in range(50)
    ]
    assert [_tool_output(response.messages[index]) for index in range(2, 101, 2)] == ['ok'] * 50


@pytest.mark.asyncio
async def test_closed_history_after_tool_turn_keeps_intermediate_messages() -> None:
    """A dead turn after a tool turn still has that tool request and reply on .messages."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
        resume_restart=restart_tool(interrupt=first.interrupts[0], replace_input={'ok': True}),
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    """A model reporting ``failed`` owns the failure; output parsing does not overwrite it.

    The turn above is a failure Genkit boxed itself. This one arrives from the
    provider with text attached, so there is something to parse and a wrong
    answer available: stamping INVALID_OUTPUT here would blame the schema for
    a model-side failure. Matches Go, where FinishReasonFailed is in
    FinishReason.isAbnormal and so never reaches the parser.
    """
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    assert isinstance(response.error.details, dict)
    assert 'reason' not in response.error.details
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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


@pytest.mark.asyncio
async def test_midstream_model_failure_keeps_chunks_and_prior_closed_history() -> None:
    ai = Genkit(model='midstreamFailureModel')
    model_calls = 0
    tool_calls = 0
    chunks: list[ModelResponseChunk] = []

    @ai.tool(name='lookup')
    async def lookup() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return '72F'

    async def midstream_failure_model(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return _model_calls_tool(name='lookup', ref='closed')
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial-1')]))
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial-2')]))
        raise RuntimeError('stream broke')

    ai.define_model(name='midstreamFailureModel', fn=midstream_failure_model)
    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='midstreamFailureModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('start')])],
            tools=['lookup'],
        ),
        on_chunk=chunks.append,
    )

    model_chunks = [chunk for chunk in chunks if chunk.role == Role.MODEL]
    assert [text_from_content(chunk.content) for chunk in model_chunks] == ['partial-1', 'partial-2']
    assert [chunk.role for chunk in chunks] == [Role.TOOL, Role.MODEL, Role.MODEL]
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_request(response.messages[1]).ref == 'closed'
    assert _tool_output(response.messages[2]) == '72F'
    assert model_calls == 2
    assert tool_calls == 1


@pytest.mark.asyncio
async def test_first_turn_midstream_failure_keeps_chunks_and_drops_unfinished_message() -> None:
    """Chunks already sent; the unfinished model message is not on resendable history."""
    ai = Genkit(model='firstTurnMidstreamModel')

    async def first_turn_midstream(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Hello, ')]))
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('wor')]))
        raise RuntimeError('stream died')

    ai.define_model(name='firstTurnMidstreamModel', fn=first_turn_midstream)
    stream = ai.generate_stream(prompt='start')
    chunks = [chunk async for chunk in stream]
    response = await stream.response

    assert [text_from_content(chunk.content) for chunk in chunks] == ['Hello, ', 'wor']
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'start'


def test_generate_stream_does_not_accept_timeout() -> None:
    """generate_stream has no timeout=; the async for waits until generate finishes."""
    ai = Genkit()
    with pytest.raises(TypeError, match='timeout'):
        ai.generate_stream(prompt='hi', timeout=5)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_generate_on_chunk_failure_returns_closed_history() -> None:
    """A dead on_chunk sink on the first model token returns the user prompt."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
            model='programmableModel',
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
async def test_generate_on_chunk_genkit_error_is_internal_not_the_sink_reason() -> None:
    """A sink that raises GenkitError is still a dead pipe, not TOOL_FAILED."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
            model='programmableModel',
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


@pytest.mark.asyncio
async def test_recovered_middleware_failure_uses_latest_closed_history() -> None:
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
            model='programmableModel',
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
                model='programmableModel',
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    abort_signal = asyncio.Event()
    abort_signal.set()

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    abort_signal = asyncio.Event()
    abort_signal.set()

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
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
@pytest.mark.parametrize('status', ['ABORTED', 'NOT_FOUND', 'INVALID_ARGUMENT', 'FAILED_PRECONDITION'])
async def test_provider_status_failure_keeps_closed_rounds(status: str) -> None:
    """A provider status is failure data after generation starts, not a setup error."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    """An Invalid input for action error after a completed model turn is still resendable."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
                message="Invalid input for action 'conforming': title missing",
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
    assert "Invalid input for action 'conforming'" in response.finish_message
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
    """A wrap_model Invalid input for action after the model returned is still resendable."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
                message="Invalid input for action 'conforming': title missing",
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
    assert "Invalid input for action 'conforming'" in response.finish_message
    assert 'title missing' in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_model_action_input_error_after_next_fn_keeps_closed_rounds() -> None:
    """A wrap_model Invalid input for action after next_fn still leaves the closed tool round."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
                    message="Invalid input for action 'conforming': title missing",
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
    assert "Invalid input for action 'conforming'" in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_wrap_model_action_input_error_before_next_fn_drops_unanswered_model() -> None:
    """A wrap_model Invalid input for action after generate has entered is still resendable."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
                message="Invalid input for action 'conforming': title missing",
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
    assert "Invalid input for action 'conforming'" in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_model_action_input_error_before_next_fn_keeps_closed_rounds() -> None:
    """A wrap_model Invalid input for action before the next model call still leaves the closed tool round."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
                    message="Invalid input for action 'conforming': title missing",
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
    assert "Invalid input for action 'conforming'" in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_output(response.messages[2]) == '72F'


@pytest.mark.asyncio
async def test_generate_wrap_model_action_input_error_after_short_circuit_next_fn_drops_unanswered_model() -> None:
    """A wrap_model Invalid input after a cache-style next_fn is still resendable."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
                message="Invalid input for action 'conforming': title missing",
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
    assert "Invalid input for action 'conforming'" in response.finish_message
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is None
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_wrap_generate_dict_after_next_fn_keeps_model_turn() -> None:
    """A wrap_generate dict after a completed model turn is still resendable."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
                message="Invalid input for action 'conforming': title missing",
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
    assert "Invalid input for action 'conforming'" in response.finish_message
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    """An Invalid input for action error after next_fn still leaves the closed tool round."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
                message="Invalid input for action 'conforming': title missing",
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
    assert "Invalid input for action 'conforming'" in response.finish_message
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
            model='programmableModel',
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
async def test_util_generate_dead_turn_paints_span_error() -> None:
    """Dev UI /util/generate after a dead turn must not look like a win on the action span."""
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    class Recipe(BaseModel):
        title: str

    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('not json')]),
        )
    ]

    provider = trace_api.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace_api.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    try:
        action = await ai.registry.resolve_action(kind=ActionKind.UTIL, name='generate')
        assert action is not None
        action_response = await action.run(
            GenerateActionOptions(
                model='programmableModel',
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
    finally:
        exporter.clear()
        if hasattr(provider, '_active_span_processor'):
            provider._active_span_processor._span_processors = tuple(
                p for p in provider._active_span_processor._span_processors if p is not processor
            )


@pytest.mark.asyncio
async def test_generate_middleware_dropping_failure_still_keeps_closed_rounds() -> None:
    """A hook that discards an inner failure cannot discard its completed history."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

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
    assert response.output.title == 'Soup'
    assert response.message is not None
    assert response.message.text == '{"title": "Soup"}'
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[1] == response.message


@pytest.mark.asyncio
async def test_generate_enum_off_list_is_error_finish() -> None:
    """An enum reply that is not one of the listed values is invalid output."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
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
    ai = Genkit(model='programmableModel')
    define_programmable_model(ai)

    with pytest.raises(GenkitError, match='Unable to resolve format') as raised:
        await ai.generate(prompt='hi', output_format='no-such-format')
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'ACTION_NOT_FOUND' not in raised.value.original_message


@pytest.mark.asyncio
async def test_unknown_middleware_raises_with_invalid_input() -> None:
    """A middleware name that is not registered is a bad argument, not a missing action."""
    ai = Genkit(model='programmableModel')
    define_programmable_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='hi', use=[MiddlewareRef(name='ghost')])
    error = raised.value
    assert error.status == 'NOT_FOUND'
    assert error.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'ghost' in error.original_message
    assert 'ACTION_NOT_FOUND' not in error.original_message


@pytest.mark.asyncio
async def test_unmatched_resource_raises_with_invalid_input() -> None:
    """A resource URI that matches nothing is a bad argument, not a missing action."""
    ai = Genkit(model='programmableModel')
    define_programmable_model(ai)

    async def other_resource(inp: ResourceInput, ctx: ActionRunContext) -> ResourceOutput:
        return ResourceOutput(content=[Part.from_text('other')])

    define_resource(ai.registry, {'uri': 'test://other'}, other_resource)

    with pytest.raises(GenkitError) as raised:
        await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[
                    Message(
                        role=Role.USER,
                        content=[Part(resource=Resource1(uri='test://missing'))],
                    )
                ],
                resources=['test://other'],
            ),
        )
    error = raised.value
    assert error.status == 'NOT_FOUND'
    assert error.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'test://missing' in error.original_message
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
    ai = Genkit(model='programmableModel')
    define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    define_programmable_model(ai)

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
    ai = Genkit(model='programmableModel')
    define_programmable_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='hi', output_format='array', output_schema={'type': 'object'})
    error = raised.value
    assert error.status == 'INVALID_ARGUMENT'
    assert error.reason is RuntimeErrorReason.INVALID_SCHEMA
    assert 'array' in error.original_message
    assert 'INVALID_SCHEMA' not in error.original_message


@pytest.mark.asyncio
async def test_generate_unknown_resource_name_raises_not_found() -> None:
    """A resource name that is not registered is NOT_FOUND, same as an unmatched URI."""
    ai = Genkit(model='programmableModel')
    define_programmable_model(ai)

    with pytest.raises(GenkitError) as raised:
        await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[
                    Message(
                        role=Role.USER,
                        content=[Part(resource=Resource1(uri='test://file'))],
                    )
                ],
                resources=['ghost'],
            ),
        )
    error = raised.value
    assert error.status == 'NOT_FOUND'
    assert error.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'ghost' in error.original_message
    assert 'INVALID_INPUT' not in error.original_message


@pytest.mark.asyncio
async def test_generate_numeric_resource_raises_invalid_input() -> None:
    """A resource entry that is not a name or action is a bad argument."""
    ai = Genkit(model='programmableModel')
    define_programmable_model(ai)

    with pytest.raises(GenkitError) as raised:
        await generate_action(
            ai.registry,
            GenerateActionOptions.model_construct(
                model='programmableModel',
                messages=[
                    Message(
                        role=Role.USER,
                        content=[Part(resource=Resource1(uri='test://file'))],
                    )
                ],
                resources=[123],
            ),
        )
    error = raised.value
    assert error.status == 'INVALID_ARGUMENT'
    assert error.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'Resources must be strings or actions' in error.original_message
    assert 'INVALID_INPUT' not in error.original_message
