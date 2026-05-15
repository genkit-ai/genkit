#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import json
import pathlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

import pytest
import yaml
from pydantic import BaseModel, TypeAdapter

from genkit import ActionKind, Document, Genkit, Message, ModelResponse, ModelResponseChunk
from genkit._ai._generate import generate_action
from genkit._ai._model import text_from_content, text_from_message
from genkit._ai._testing import (
    ProgrammableModel,
    define_echo_model,
    define_programmable_model,
)
from genkit._ai._tools import Interrupt, define_tool
from genkit._core._action import ActionRunContext
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


@pytest.mark.asyncio
async def test_generate_applies_middleware(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    """When middleware is provided, apply it."""
    ai, *_ = setup_test
    define_echo_model(ai)

    async def pre_middle(
        req: ModelRequest,
        ctx: ActionRunContext,
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        txt = ''.join(text_from_message(m) for m in req.messages)
        return await next(
            ModelRequest(
                messages=[
                    Message(role=Role.USER, content=[Part(TextPart(text=f'PRE {txt}'))]),
                ],
            ),
            ctx,
        )

    async def post_middle(
        req: ModelRequest,
        ctx: ActionRunContext,
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        resp: ModelResponse = await next(req, ctx)
        assert resp.message is not None
        txt = text_from_message(resp.message)
        return ModelResponse(
            finish_reason=resp.finish_reason,
            message=Message(role=Role.USER, content=[Part(TextPart(text=f'{txt} POST'))]),
        )

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
        ),
        middleware=[pre_middle, post_middle],
    )

    assert response.text == '[ECHO] user: "PRE hi" POST'


@pytest.mark.asyncio
async def test_generate_middleware_next_fn_args_optional(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    """Can call next function without args (convenience)."""
    ai, *_ = setup_test
    define_echo_model(ai)

    async def post_middle(
        req: ModelRequest,
        ctx: ActionRunContext,
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        resp: ModelResponse = await next(req, ctx)
        assert resp.message is not None
        txt = text_from_message(resp.message)
        return ModelResponse(
            finish_reason=resp.finish_reason,
            message=Message(role=Role.USER, content=[Part(TextPart(text=f'{txt} POST'))]),
        )

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
        ),
        middleware=[post_middle],
    )

    assert response.text == '[ECHO] user: "hi" POST'


@pytest.mark.asyncio
async def test_generate_middleware_can_modify_context(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    """Test that middleware can modify context."""
    ai, *_ = setup_test
    define_echo_model(ai)

    async def add_context(
        req: ModelRequest,
        ctx: ActionRunContext,
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await next(req, ActionRunContext(context={**ctx.context, 'banana': True}))

    async def inject_context(
        req: ModelRequest,
        ctx: ActionRunContext,
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        txt = ''.join(text_from_message(m) for m in req.messages)
        return await next(
            ModelRequest(
                messages=[
                    Message(
                        role=Role.USER,
                        content=[Part(TextPart(text=f'{txt} {ctx.context}'))],
                    ),
                ],
            ),
            ctx,
        )

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
        ),
        middleware=[add_context, inject_context],
        context={'foo': 'bar'},
    )

    assert response.text == '''[ECHO] user: "hi {'foo': 'bar', 'banana': True}"'''


@pytest.mark.asyncio
async def test_generate_middleware_can_modify_stream(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    """Test that middleware can modify streams."""
    ai, pm = setup_test

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

    async def modify_stream(
        req: ModelRequest,
        ctx: ActionRunContext,
        on_chunk: Callable[[ModelResponseChunk], None] | None,
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # 4-param streaming middleware signature
        if on_chunk:
            on_chunk(
                ModelResponseChunk(
                    role=Role.MODEL,
                    content=[Part(TextPart(text='something extra before'))],
                )
            )

        def chunk_handler(chunk: ModelResponseChunk) -> None:
            if on_chunk:
                on_chunk(
                    ModelResponseChunk(
                        role=Role.MODEL,
                        content=[Part(TextPart(text=f'intercepted: {text_from_content(chunk.content)}'))],
                    )
                )

        resp = await next(req, ctx, chunk_handler)
        if on_chunk:
            on_chunk(
                ModelResponseChunk(
                    role=Role.MODEL,
                    content=[Part(TextPart(text='something extra after'))],
                )
            )
        return resp

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
        ),
        middleware=[modify_stream],
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
