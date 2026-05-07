#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import json
import pathlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, ClassVar, cast

import pytest
import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from genkit import ActionKind, Document, Genkit, Message, MiddlewareRef, ModelResponse, ModelResponseChunk
from genkit._ai._generate import _augment_with_context, generate_action
from genkit._ai._model import text_from_content, text_from_message
from genkit._ai._testing import (
    ProgrammableModel,
    define_echo_model,
    define_programmable_model,
)
from genkit._ai._tools import Interrupt, define_tool
from genkit._core._model import GenerateActionOptions, ModelRequest
from genkit._core._registry import Registry
from genkit._core._typing import (
    DocumentPart,
    FinishReason,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)
from genkit.middleware import (
    BaseMiddleware,
    GenerateHookParams,
    MiddlewareDesc,
    ModelHookParams,
    MultipartToolResponse,
    ToolHookParams,
    middleware_plugin,
    new_middleware,
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
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='bye'))]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(TextPart(text='hi'))],
                ),
            ],
        ),
    )

    assert response.text == 'bye'


@pytest.mark.asyncio
async def test_simulates_doc_grounding(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    """Test that docs are correctly grounded and injected into prompt."""
    ai, pm = setup_test

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='bye'))]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(TextPart(text='hi'))],
                ),
            ],
            docs=[Document(content=[DocumentPart(TextPart(text='doc content 1'))])],
        ),
    )

    assert response.request is not None
    assert response.request.messages is not None
    assert response.request.messages[0] == Message(
        role=Role.USER,
        content=[
            Part(TextPart(text='hi')),
            Part(
                root=TextPart(
                    text='\n\nUse the following information to complete your task:' + '\n\n- [0]: doc content 1\n\n',
                    metadata={'purpose': 'context'},
                )
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# Unit tests for the private _augment_with_context helper                     #
# --------------------------------------------------------------------------- #

def test_augment_with_context_ignores_no_docs() -> None:
    """No docs -> request returned unchanged (same object identity)."""
    req = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
        ],
    )

    transformed_req = _augment_with_context(req)

    assert transformed_req is req


def test_augment_with_context_adds_docs_as_context() -> None:
    """Docs are injected as a context-purpose part appended to the last user message."""
    req = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
        ],
        docs=[
            Document(content=[DocumentPart(root=TextPart(text='doc content 1'))]),
            Document(content=[DocumentPart(root=TextPart(text='doc content 2'))]),
        ],
    )

    transformed_req = _augment_with_context(req)

    assert transformed_req == ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='hi')),
                    Part(
                        root=TextPart(
                            text='\n\nUse the following information to complete '
                            + 'your task:\n\n'
                            + '- [0]: doc content 1\n'
                            + '- [1]: doc content 2\n\n',
                            metadata={'purpose': 'context'},
                        )
                    ),
                ],
            )
        ],
        docs=[
            Document(content=[DocumentPart(root=TextPart(text='doc content 1'))]),
            Document(content=[DocumentPart(root=TextPart(text='doc content 2'))]),
        ],
    )


def test_augment_with_context_does_not_mutate_input() -> None:
    """Input request and its messages are not mutated; helper returns a deepcopy."""
    original_user_msg = Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])
    req = ModelRequest(
        messages=[original_user_msg],
        docs=[Document(content=[DocumentPart(root=TextPart(text='doc content 1'))])],
    )
    original_content_len = len(original_user_msg.content)

    transformed_req = _augment_with_context(req)

    assert transformed_req is not req
    assert transformed_req.messages[0] is not original_user_msg
    assert len(original_user_msg.content) == original_content_len
    assert len(transformed_req.messages[0].content) == original_content_len + 1


def test_augment_with_context_should_not_modify_non_pending_part() -> None:
    """An existing non-pending context part means the request is returned unchanged."""
    req = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(
                        root=TextPart(
                            text='this is already context',
                            metadata={'purpose': 'context'},
                        )
                    ),
                    Part(root=TextPart(text='hi')),
                ],
            ),
        ],
        docs=[
            Document(content=[DocumentPart(root=TextPart(text='doc content 1'))]),
        ],
    )

    transformed_req = _augment_with_context(req)

    assert transformed_req is req


def test_augment_with_context_with_purpose_part() -> None:
    """A pending purpose=context part is replaced with the rendered docs context."""
    req = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(
                        root=TextPart(
                            text='insert context here',
                            metadata={'purpose': 'context', 'pending': True},
                        )
                    ),
                    Part(root=TextPart(text='hi')),
                ],
            ),
        ],
        docs=[
            Document(content=[DocumentPart(root=TextPart(text='doc content 1'))]),
        ],
    )

    transformed_req = _augment_with_context(req)

    assert transformed_req == ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(
                        root=TextPart(
                            text='\n\nUse the following information to complete '
                            + 'your task:\n\n'
                            + '- [0]: doc content 1\n\n',
                            metadata={'purpose': 'context'},
                        )
                    ),
                    Part(root=TextPart(text='hi')),
                ],
            )
        ],
        docs=[
            Document(content=[DocumentPart(root=TextPart(text='doc content 1'))]),
        ],
    )


# --------------------------------------------------------------------------- #
# Middleware class definitions shared by tests below                           #
# --------------------------------------------------------------------------- #

class PreMiddleware(BaseMiddleware):
    name: ClassVar[str] = 'pre_mw'

    async def wrap_model(self, params: ModelHookParams, next_fn: Callable) -> ModelResponse:
        txt = ''.join(text_from_message(m) for m in params.request.messages)
        return await next_fn(
            ModelHookParams(
                request=ModelRequest(
                    messages=[
                        Message(role=Role.USER, content=[Part(TextPart(text=f'PRE {txt}'))]),
                    ],
                ),
                on_chunk=params.on_chunk,
                context=params.context,
            )
        )


class PostMiddleware(BaseMiddleware):
    name: ClassVar[str] = 'post_mw'

    async def wrap_model(self, params: ModelHookParams, next_fn: Callable) -> ModelResponse:
        resp: ModelResponse = await next_fn(params)
        assert resp.message is not None
        txt = text_from_message(resp.message)
        return ModelResponse(
            finish_reason=resp.finish_reason,
            message=Message(role=Role.USER, content=[Part(TextPart(text=f'{txt} POST'))]),
        )


def test_generate_action_options_use_is_middleware_ref_only() -> None:
    """``GenerateActionOptions.use`` is the wire form; only ``MiddlewareRef`` entries allowed.

    Inline ``BaseMiddleware`` instances flow through the ``Genkit.generate(use=...)`` veneer,
    which resolves them locally and hands them to ``generate_action`` via ``resolved_middleware=``;
    they must never appear in the JSON-serialized action options.
    """
    expected: tuple[type[BaseException], ...] = (ValidationError, TypeError)
    with pytest.raises(expected):
        GenerateActionOptions(
            model='echoModel',
            messages=[Message(role=Role.USER, content=[Part(TextPart(text='hi'))])],
            use=cast(list[MiddlewareRef], [PreMiddleware()]),
        )


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
    ai = Genkit(plugins=[middleware_plugin([new_middleware(PostMiddleware)])])
    define_echo_model(ai)

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        use=[PreMiddleware(), MiddlewareRef(name='post_mw')],
    )

    assert response.text == '[ECHO] user: "PRE hi" POST'


class ConfiguredPrefixMiddleware(BaseMiddleware):
    """Inline middleware driven purely by a pydantic config field."""

    name: ClassVar[str] = 'configured_prefix_mw'
    prefix: str = 'DEFAULT'

    async def wrap_model(self, params: ModelHookParams, next_fn: Callable) -> ModelResponse:
        txt = ''.join(text_from_message(m) for m in params.request.messages)
        return await next_fn(
            ModelHookParams(
                request=ModelRequest(
                    messages=[
                        Message(role=Role.USER, content=[Part(TextPart(text=f'{self.prefix} {txt}'))]),
                    ],
                ),
                on_chunk=params.on_chunk,
                context=params.context,
            )
        )


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
async def test_generate_middleware_ref_config_instantiates_class() -> None:
    """``MiddlewareRef(config=...)`` feeds ``**config`` into the class constructor."""
    ai = Genkit(plugins=[middleware_plugin([new_middleware(ConfiguredPrefixMiddleware)])])
    define_echo_model(ai)

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        use=[MiddlewareRef(name='configured_prefix_mw', config={'prefix': '[SPAN]'})],
    )

    assert response.text == '[ECHO] user: "[SPAN] hi"'


@pytest.mark.asyncio
async def test_define_middleware_registers_on_the_fly() -> None:
    """``ai.define_middleware(cls)`` makes the definition resolvable by name."""
    ai = Genkit()
    define_echo_model(ai)
    ai.define_middleware(ConfiguredPrefixMiddleware)

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        use=[MiddlewareRef(name='configured_prefix_mw', config={'prefix': '[LIVE]'})],
    )

    assert response.text == '[ECHO] user: "[LIVE] hi"'


@pytest.mark.asyncio
async def test_generate_applies_middleware() -> None:
    """When middleware is provided, apply it via MiddlewareRef resolution."""
    ai = Genkit(
        plugins=[
            middleware_plugin([
                new_middleware(PreMiddleware),
                new_middleware(PostMiddleware),
            ])
        ],
    )
    define_echo_model(ai)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='echoModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(TextPart(text='hi'))],
                ),
            ],
            use=[MiddlewareRef(name='pre_mw'), MiddlewareRef(name='post_mw')],
        ),
    )

    assert response.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_generate_middleware_next_fn_args_optional() -> None:
    """Can call next function without modifying params (pass params through)."""
    ai = Genkit(plugins=[middleware_plugin([new_middleware(PostMiddleware)])])
    define_echo_model(ai)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='echoModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(TextPart(text='hi'))],
                ),
            ],
            use=[MiddlewareRef(name='post_mw')],
        ),
    )

    assert response.text == '[ECHO] user: "hi" POST'


class AddContextMiddleware(BaseMiddleware):
    name: ClassVar[str] = 'add_ctx'

    async def wrap_model(self, params: ModelHookParams, next_fn: Callable) -> ModelResponse:
        return await next_fn(
            ModelHookParams(
                request=params.request,
                on_chunk=params.on_chunk,
                context={**params.context, 'banana': True},
            )
        )


class InjectContextMiddleware(BaseMiddleware):
    name: ClassVar[str] = 'inject_ctx'

    async def wrap_model(self, params: ModelHookParams, next_fn: Callable) -> ModelResponse:
        txt = ''.join(text_from_message(m) for m in params.request.messages)
        return await next_fn(
            ModelHookParams(
                request=ModelRequest(
                    messages=[
                        Message(
                            role=Role.USER,
                            content=[Part(TextPart(text=f'{txt} {params.context}'))],
                        ),
                    ],
                ),
                on_chunk=params.on_chunk,
                context=params.context,
            )
        )


@pytest.mark.asyncio
async def test_generate_middleware_can_modify_context() -> None:
    """Test that middleware can modify context via ModelHookParams.context."""
    ai = Genkit(
        plugins=[
            middleware_plugin([
                new_middleware(AddContextMiddleware),
                new_middleware(InjectContextMiddleware),
            ])
        ],
    )
    define_echo_model(ai)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='echoModel',
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(TextPart(text='hi'))],
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

    class ModifyStreamMiddleware(BaseMiddleware):
        name: ClassVar[str] = 'mod_stream_mw'

        async def wrap_model(self, params: ModelHookParams, next_fn: Callable) -> ModelResponse:
            if params.on_chunk:
                params.on_chunk(
                    ModelResponseChunk(
                        role=Role.MODEL,
                        content=[Part(TextPart(text='something extra before'))],
                    )
                )

            def chunk_handler(chunk: ModelResponseChunk) -> None:
                if params.on_chunk:
                    params.on_chunk(
                        ModelResponseChunk(
                            role=Role.MODEL,
                            content=[Part(TextPart(text=f'intercepted: {text_from_content(chunk.content)}'))],
                        )
                    )

            new_params = ModelHookParams(
                request=params.request,
                on_chunk=chunk_handler,
                context=params.context,
            )
            resp = await next_fn(new_params)
            if params.on_chunk:
                params.on_chunk(
                    ModelResponseChunk(
                        role=Role.MODEL,
                        content=[Part(TextPart(text='something extra after'))],
                    )
                )
            return resp

    ai = Genkit(plugins=[middleware_plugin([new_middleware(ModifyStreamMiddleware)])])
    pm, _ = define_programmable_model(ai)

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='bye'))]),
        )
    )
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part(TextPart(text='1'))]),
            ModelResponseChunk(role=Role.MODEL, content=[Part(TextPart(text='2'))]),
            ModelResponseChunk(role=Role.MODEL, content=[Part(TextPart(text='3'))]),
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
                    content=[Part(TextPart(text='hi'))],
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


class TrackGenerateMiddleware(BaseMiddleware):
    """Middleware that records wrap_generate calls per turn."""

    iterations: list[int] = Field(default_factory=list)

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        next_fn: Callable[[GenerateHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self.iterations.append(params.iteration)
        return await next_fn(params)


@pytest.mark.asyncio
async def test_wrap_generate_called_per_turn() -> None:
    """wrap_generate is invoked for each turn of the generate loop.

    This is the two-turn regression test: verifies middleware runs on *every*
    recursive _generate_action call (turn 0 + turn 1 after tool response).
    """
    track_mw = TrackGenerateMiddleware()
    track_mw2 = TrackGenerateMiddleware()
    ai = Genkit(
        plugins=[
            middleware_plugin([
                MiddlewareDesc(
                    name='track_gen',
                    description='track generate',
                    factory=lambda _opts, _reg=None: track_mw,
                ),
                MiddlewareDesc(
                    name='track_gen2',
                    description='track generate 2',
                    factory=lambda _opts, _reg=None: track_mw2,
                ),
            ])
        ],
    )
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='testTool')
    async def _test_tool() -> object:
        return 'tool called'

    # No tools: single turn → wrap_generate called once with iteration=0
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='done'))]),
        )
    )
    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part(TextPart(text='hi'))])],
            use=[MiddlewareRef(name='track_gen')],
        ),
    )
    assert response.text == 'done'
    assert track_mw.iterations == [0]

    # With tools: two turns (model→tool→model) → wrap_generate called for each
    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='testTool', input={}, ref='r1')))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='final'))]),
        )
    )
    response2 = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part(TextPart(text='hi'))])],
            tools=['testTool'],
            use=[MiddlewareRef(name='track_gen2')],
        ),
    )
    assert response2.text == 'final'
    assert track_mw2.iterations == [0, 1]


class TrackToolMiddleware(BaseMiddleware):
    """Middleware that records wrap_tool calls."""

    tool_names: list[str] = Field(default_factory=list)

    async def wrap_tool(
        self,
        params: ToolHookParams,
        next_fn: Callable[
            [ToolHookParams],
            Awaitable[tuple[MultipartToolResponse | None, ToolRequestPart | None]],
        ],
    ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
        self.tool_names.append(params.tool_request_part.tool_request.name)
        return await next_fn(params)


@pytest.mark.asyncio
async def test_wrap_tool_called_on_tool_execution() -> None:
    """wrap_tool is invoked for each tool execution."""
    track_mw = TrackToolMiddleware()
    ai = Genkit(
        plugins=[
            middleware_plugin([
                MiddlewareDesc(
                    name='track_tool',
                    description='track tool',
                    factory=lambda _opts, _reg=None: track_mw,
                ),
            ])
        ],
    )
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='myTool')
    async def my_tool() -> object:
        return 'result'

    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='myTool', input={}, ref='r1')))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='done'))]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part(TextPart(text='hi'))])],
            tools=['myTool'],
            use=[MiddlewareRef(name='track_tool')],
        ),
    )
    assert response.text == 'done'
    assert track_mw.tool_names == ['myTool']


@pytest.mark.asyncio
async def test_middleware_wrap_tool_interrupt_handled_as_interrupt_not_crash() -> None:
    """Interrupt raised by wrap_tool middleware is converted to an interrupt part.

    This is a regression test: before the fix, a middleware-raised Interrupt
    bypassed _resolve_tool_request's except block and propagated uncaught through
    asyncio.gather, crashing generation instead of surfacing as a tool interrupt.
    """
    from genkit._ai._tools import Interrupt

    class InterruptingMiddleware(BaseMiddleware):
        name: ClassVar[str] = 'interrupt_all'

        async def wrap_tool(
            self,
            params: ToolHookParams,
            next_fn: Callable[[ToolHookParams], Awaitable[tuple[MultipartToolResponse | None, ToolRequestPart | None]]],
        ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
            raise Interrupt({'blocked': True})

    ai = Genkit(
        plugins=[
            middleware_plugin([
                MiddlewareDesc(
                    name='interrupt_all',
                    description='interrupt all tools',
                    factory=lambda _opts, _reg=None: InterruptingMiddleware(),
                ),
            ])
        ],
    )
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='blockedTool')
    async def blocked_tool() -> str:
        return 'should not run'

    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='blockedTool', input={}, ref='r1')))],
            ),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part(TextPart(text='do it'))])],
            tools=['blockedTool'],
            use=[MiddlewareRef(name='interrupt_all')],
        ),
    )
    assert response.finish_reason == FinishReason.INTERRUPTED
    assert response.message is not None
    interrupt_parts = [
        p for p in response.message.content
        if isinstance(p.root, ToolRequestPart) and p.root.metadata and 'interrupt' in p.root.metadata
    ]
    assert len(interrupt_parts) == 1
    assert interrupt_parts[0].root.metadata['interrupt'] == {'blocked': True}


@pytest.mark.asyncio
async def test_middleware_contributed_tools_available_to_model() -> None:
    """Middleware.tools() contributes actions scoped to the generate call (child registry).

    The contributed tool is resolvable by the model during the call but must not
    appear in the root registry afterward — mirroring Go's Hooks.Tools + NewChild.
    """

    class ToolProviderMiddleware(BaseMiddleware):
        """Middleware that contributes a tool dynamically per generate() call."""

        name: ClassVar[str] = 'tool_provider_mw'

        def tools(self, enqueue_parts: Callable[[list[Part]], None] | None = None) -> list:
            # Build a tool action on a throw-away registry; the generate engine
            # will adopt it into a call-scoped child registry.
            scratch = Registry()

            async def provided_tool() -> str:
                """A tool injected by middleware."""
                return 'from_middleware_tool'

            t = define_tool(scratch, provided_tool, name='middleware_tool')
            return [t.action()]

    ai = Genkit(plugins=[middleware_plugin([new_middleware(ToolProviderMiddleware)])])
    pm, _ = define_programmable_model(ai)

    # Turn 1: model calls the middleware-contributed tool
    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='middleware_tool', input={}, ref='r1')
                        )
                    )
                ],
            ),
        )
    )
    # Turn 2: model returns final answer after tool result
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='done'))]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part(TextPart(text='hi'))])],
            use=[MiddlewareRef(name='tool_provider_mw')],
        ),
    )
    assert response.text == 'done'

    # The contributed tool must NOT be visible in the root registry after the call.
    assert await ai.registry.resolve_action(ActionKind.TOOL, 'middleware_tool') is None


@pytest.mark.asyncio
async def test_middleware_self_registry_is_per_call_scope() -> None:
    """``self._registry`` points to the per-call child, not the root.

    Two assertions:

    - Middleware A contributes a tool via ``tools()`` and middleware B
      resolves it through ``self._registry`` in the same call (proves the
      shared per-call scope).
    - Anything middleware writes via ``self._registry.register_action`` does
      NOT survive after the call (proves writes are auto-cleaned).
    """
    seen_by_b: list[str] = []

    class ProviderMW(BaseMiddleware):
        name: ClassVar[str] = 'provider_mw'

        def tools(self, enqueue_parts: Callable[[list[Part]], None] | None = None) -> list:
            scratch = Registry()

            async def shared_tool() -> str:
                """Shared by all middleware in the call."""
                return 'shared_ok'

            return [define_tool(scratch, shared_tool, name='shared_tool').action()]

    class LookerMW(BaseMiddleware):
        name: ClassVar[str] = 'looker_mw'

        async def wrap_generate(
            self,
            params: GenerateHookParams,
            next_fn: Callable[[GenerateHookParams], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            assert self._registry is not None
            # Resolve the tool ProviderMW just contributed — only works if
            # both middleware share the same per-call registry scope.
            tool = await self._registry.resolve_action(ActionKind.TOOL, 'shared_tool')
            if tool is not None:
                seen_by_b.append(tool.name)
            # Also exercise the write path: anything we register through
            # self._registry must not survive the call.
            scratch = Registry()

            async def leaky_tool() -> str:
                """Should not survive the call."""
                return 'nope'

            leak = define_tool(scratch, leaky_tool, name='leaky_tool').action()
            self._registry.register_action_from_instance(leak)
            return await next_fn(params)

    ai = Genkit(
        plugins=[
            middleware_plugin([
                new_middleware(ProviderMW),
                new_middleware(LookerMW),
            ])
        ],
    )
    pm, _ = define_programmable_model(ai)
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='ok'))]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part(TextPart(text='hi'))])],
            use=[
                MiddlewareRef(name='provider_mw'),
                MiddlewareRef(name='looker_mw'),
            ],
        ),
    )
    assert response.text == 'ok'
    assert seen_by_b == ['shared_tool'], (
        f'looker middleware should have resolved shared_tool, saw: {seen_by_b}'
    )
    # Neither tool may leak into the root registry after the call ends.
    assert await ai.registry.resolve_action(ActionKind.TOOL, 'shared_tool') is None
    assert await ai.registry.resolve_action(ActionKind.TOOL, 'leaky_tool') is None


@pytest.mark.asyncio
async def test_inline_middleware_instance_is_not_mutated_across_calls() -> None:
    """Inline ``BaseMiddleware`` instances passed in ``use=`` must not have their
    ``_registry`` mutated in place — the engine clones with ``model_copy()``.
    """

    class IdentityMW(BaseMiddleware):
        name: ClassVar[str] = 'identity_mw'

    ai_a = Genkit()
    ai_b = Genkit()
    define_programmable_model(ai_a)
    define_programmable_model(ai_b)
    shared = IdentityMW()

    from genkit._ai._generate import normalize_middleware

    child_a = ai_a.registry.new_child()
    child_b = ai_b.registry.new_child()
    refs_a = normalize_middleware(child_a, [shared])
    refs_b = normalize_middleware(child_b, [shared])

    assert refs_a
    assert refs_b
    # Caller's instance is untouched.
    assert shared._registry is None
    # Each normalization registered a distinct cloned instance into its own child registry.
    inst_a = child_a.lookup_value('middleware', 'identity_mw')
    inst_b = child_b.lookup_value('middleware', 'identity_mw')
    assert inst_a is not None
    assert inst_b is not None


@pytest.mark.asyncio
async def test_queue_drain_streams_each_message_at_one_index() -> None:
    """Queued tool middleware messages stream as exactly one chunk per message.

    Regression: the old queue-drain path called ``make_chunk(USER, ...)`` for
    each queued message AND then did ``message_index += 1``. ``make_chunk``
    *also* advanced the index (role flip from MODEL to USER), so each queued
    message bumped the counter twice — leaving a hole in the stream sequence.
    The fix emits queued chunks directly and increments once per message.
    """

    class EnqueuingMW(BaseMiddleware):
        """After each tool call, enqueue an extra USER part for the next turn."""

        name: ClassVar[str] = 'enqueuing_mw'

        async def wrap_tool(
            self,
            params: ToolHookParams,
            next_fn: Callable[
                [ToolHookParams],
                Awaitable[tuple[MultipartToolResponse | None, ToolRequestPart | None]],
            ],
        ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
            result = await next_fn(params)
            if params.enqueue_parts:
                params.enqueue_parts([Part(TextPart(text='extra-context'))])
            return result

    ai = Genkit(plugins=[middleware_plugin([new_middleware(EnqueuingMW)])])
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='trigger')
    async def trigger() -> str:
        return 'triggered'

    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='trigger', input={}, ref='r1')))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='final'))]),
        )
    )

    streamed: list[ModelResponseChunk] = []
    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part(TextPart(text='go'))])],
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

    Regression: ``_resolve_resumed_tool_request`` used to call
    ``run_tool_after_restart`` directly, skipping the middleware chain. That
    silently bypassed ToolApproval / Filesystem / etc. on every restart.
    """
    invocations: list[str] = []

    class RecordingMW(BaseMiddleware):
        name: ClassVar[str] = 'recording_mw'

        async def wrap_tool(
            self,
            params: ToolHookParams,
            next_fn: Callable[
                [ToolHookParams],
                Awaitable[tuple[MultipartToolResponse | None, ToolRequestPart | None]],
            ],
        ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
            invocations.append(params.tool.name)
            return await next_fn(params)

    ai = Genkit(plugins=[middleware_plugin([new_middleware(RecordingMW)])])
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='approveMe')
    async def approve_me() -> str:
        return 'approved'

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='final'))]),
        )
    )

    interrupt_part = ToolRequestPart(
        tool_request=ToolRequest(name='approveMe', input={}, ref='r1'),
        metadata={'interrupt': True},
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[
                Message(role=Role.USER, content=[Part(TextPart(text='do it'))]),
                Message(role=Role.MODEL, content=[Part(root=interrupt_part)]),
            ],
            tools=['approveMe'],
            use=[MiddlewareRef(name='recording_mw')],
            resume={
                'restart': [
                    {
                        'toolRequest': {'name': 'approveMe', 'input': {}, 'ref': 'r1'},
                        'metadata': {'resumed': {'toolApproved': True}},
                    }
                ],
            },
        ),
    )
    assert response.text == 'final'
    assert invocations == ['approveMe'], (
        f'expected wrap_tool to fire once on restart, saw: {invocations}'
    )


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
                    Part(TextPart(text='call three')),
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='tool_a', ref='ref-a', input={}),
                        )
                    ),
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='tool_b', ref='ref-b', input={}),
                        )
                    ),
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='tool_c', ref='ref-c', input={}),
                        )
                    ),
                ],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='after_tools'))]),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[
                Message(role=Role.USER, content=[Part(TextPart(text='hi'))]),
            ],
            tools=['tool_a', 'tool_b', 'tool_c'],
        ),
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'after_tools'


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
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='inline_yell', ref='ref-y', input={}),
                        )
                    ),
                ],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='after_inline'))]),
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
                    Part(TextPart(text='call three')),
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='tool_a', ref='ref-a', input={}),
                        )
                    ),
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='tool_b', ref='ref-b', input={}),
                        )
                    ),
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='tool_c', ref='ref-c', input={}),
                        )
                    ),
                ],
            ),
        )
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[
                Message(role=Role.USER, content=[Part(TextPart(text='hi'))]),
            ],
            tools=['tool_a', 'tool_b', 'tool_c'],
        ),
    )

    assert response.finish_reason == FinishReason.INTERRUPTED
    assert response.message is not None
    parts = response.message.content
    assert len(parts) == 4
    assert parts[0].root == TextPart(text='call three')
    a_root = parts[1].root
    b_root = parts[2].root
    c_root = parts[3].root
    assert isinstance(a_root, ToolRequestPart)
    assert isinstance(b_root, ToolRequestPart)
    assert isinstance(c_root, ToolRequestPart)
    assert a_root.metadata and a_root.metadata.get('pendingOutput') == 'a_ok'
    assert b_root.metadata and b_root.metadata.get('interrupt') == {'stop': True}
    assert c_root.metadata and c_root.metadata.get('pendingOutput') == 'c_ok'


##########################################################################
# run tests from /tests/specs/generate.yaml
##########################################################################

specs = []
spec_path = pathlib.Path(__file__).parent / '../../../../../../tests/specs/generate.yaml'
with spec_path.resolve().open() as stream:
    tests_spec = yaml.safe_load(stream)
    specs = tests_spec['tests']
    specs = [x for x in tests_spec['tests'] if x['name'] == 'calls tools']


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
            ai.registry,
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
