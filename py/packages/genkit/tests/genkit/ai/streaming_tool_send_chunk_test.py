#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for typed tool chunk streaming via ``ToolContext[ChunkT].send_chunk``."""

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import (
    FinishReason,
    Genkit,
    Interrupt,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    Role,
    ToolContext,
)
from genkit._ai._agents._client import StreamedMessageAccumulator
from genkit._core._action import StreamResponse
from genkit._core._middleware import BaseMiddleware, GenerateHookParams, GenerateMiddlewareContext
from genkit._core._model import AgentStreamChunk, ModelRequest
from genkit._core._typing import ToolRequest
from genkit.testing import define_scripted_model


class Step(BaseModel):
    """Typed progress payload emitted by a tool."""

    stage: str
    percent: int


class Recipe(BaseModel):
    """Typed model output produced after the tool turn."""

    title: str
    steps: list[str]


def _text_part(text: str) -> Part:
    return Part(text=text)


def _model_calls(*tools: tuple[str, str]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name=name, ref=ref, input={})) for name, ref in tools],
        ),
    )


def _model_says(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[_text_part(text)]),
    )


def _partial_parts(chunks: Sequence[ModelResponseChunk[Any]]) -> list[Part]:
    return [part for chunk in chunks if chunk.role == Role.TOOL for part in chunk.tool_responses if part.is_partial]


def _final_parts(chunks: Sequence[ModelResponseChunk[Any]]) -> list[Part]:
    return [part for chunk in chunks if chunk.role == Role.TOOL for part in chunk.tool_responses if not part.is_partial]


def _message_tool_response_parts(messages: Sequence[Message]) -> list[Part]:
    return [part for msg in messages if msg.role == Role.TOOL for part in msg.content if part.tool_response is not None]


def _tool_outputs(messages: Sequence[Message]) -> list[object]:
    return [part.tool_response.output for part in _message_tool_response_parts(messages) if part.tool_response]


async def _collect_stream(ai: Genkit, **kwargs: object) -> tuple[list[ModelResponseChunk[Any]], ModelResponse[Any]]:
    stream = ai.generate_stream(**kwargs)  # type: ignore[arg-type]
    chunks: list[ModelResponseChunk[Any]] = []
    async for chunk in stream.stream:
        chunks.append(chunk)
    return chunks, await stream.response


@pytest.mark.asyncio
async def test_tool_stream_yields_typed_chunks_and_output() -> None:
    """Direct tool.stream(input) yields ChunkT directly and resolves OutputT directly."""
    ai = Genkit()

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[Step]) -> str:
        ctx.send_chunk(Step(stage='compile', percent=30))
        ctx.send_chunk(Step(stage='rollout', percent=90))
        return 'https://service.run'

    stream = deploy.stream({})
    assert_type(stream, StreamResponse[Step, str])
    got: list[Step] = []
    async for chunk in stream.stream:
        assert_type(chunk, Step)
        got.append(chunk)
    result = await stream.response
    assert_type(result, str)

    assert got == [
        Step(stage='compile', percent=30),
        Step(stage='rollout', percent=90),
    ]
    assert result == 'https://service.run'


@pytest.mark.asyncio
async def test_tool_stream_returns_pydantic_output_directly() -> None:
    """Direct tool.stream(input) resolves a Pydantic model OutputT instance directly."""
    ai = Genkit()

    @ai.tool(name='cook')
    async def cook(_req: dict[str, Any], ctx: ToolContext[Step]) -> Recipe:
        ctx.send_chunk(Step(stage='prep', percent=50))
        return Recipe(title='Tartine', steps=['toast', 'top'])

    stream = cook.stream({})
    assert_type(stream, StreamResponse[Step, Recipe])
    got = [chunk async for chunk in stream.stream]
    recipe = await stream.response
    assert_type(recipe, Recipe)

    assert got == [Step(stage='prep', percent=50)]
    assert isinstance(recipe, Recipe)
    assert recipe == Recipe(title='Tartine', steps=['toast', 'top'])


@pytest.mark.asyncio
async def test_tool_call_without_stream_is_noop_for_send_chunk() -> None:
    """Direct await tool(input) ignores send_chunk and returns OutputT directly."""
    ai = Genkit()

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[Step]) -> str:
        ctx.send_chunk(Step(stage='compile', percent=30))
        return 'done'

    res = await deploy({})
    assert res == 'done'


@pytest.mark.asyncio
async def test_bare_tool_decorator_streams_typed_chunks_and_output() -> None:
    """Bare @ai.tool (without parentheses) infers Tool[InputT, OutputT, ChunkT] and streams ChunkT."""
    ai = Genkit()

    @ai.tool
    async def deploy(_req: dict[str, Any], ctx: ToolContext[Step]) -> Recipe:
        ctx.send_chunk(Step(stage='bake', percent=80))
        return Recipe(title='Sourdough', steps=['mix', 'bake'])

    stream = deploy.stream({})
    assert_type(stream, StreamResponse[Step, Recipe])
    got = [chunk async for chunk in stream.stream]
    recipe = await stream.response
    assert_type(recipe, Recipe)

    assert got == [Step(stage='bake', percent=80)]
    assert recipe == Recipe(title='Sourdough', steps=['mix', 'bake'])


@pytest.mark.asyncio
async def test_generate_stream_emits_partial_tool_response_before_final() -> None:
    """ctx.send_chunk(val) during generate_stream emits partial tool_response before the final one."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[Step]) -> str:
        ctx.send_chunk(Step(stage='uploading', percent=50))
        return 'https://example.run'

    pm.responses = [_model_calls(('deploy', 'call-1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])

    partials = _partial_parts(chunks)
    finals = _final_parts(chunks)
    assert len(partials) == 1
    assert len(finals) == 1

    progress = partials[0]
    assert progress.is_partial is True
    assert progress.tool_response is not None
    assert progress.tool_response.name == 'deploy'
    assert progress.tool_response.ref == 'call-1'
    assert Step.model_validate(progress.tool_response.output) == Step(stage='uploading', percent=50)

    final = finals[0]
    assert final.is_partial is False
    assert final.tool_response is not None
    assert final.tool_response.name == 'deploy'
    assert final.tool_response.ref == 'call-1'
    assert final.tool_response.output == 'https://example.run'

    partial_chunk_idx = next(i for i, c in enumerate(chunks) if progress in c.tool_responses)
    final_chunk_idx = next(i for i, c in enumerate(chunks) if final in c.tool_responses)
    assert partial_chunk_idx < final_chunk_idx

    assert all(not p.is_partial for p in _message_tool_response_parts(response.messages))
    assert _tool_outputs(response.messages) == ['https://example.run']


@pytest.mark.asyncio
async def test_generate_stream_preserves_each_partial_chunk_in_order() -> None:
    """Repeated progress values remain separate updates without merging or deduplication."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[dict[str, int]]) -> str:
        ctx.send_chunk({'percent': 10})
        ctx.send_chunk({'percent': 10})
        ctx.send_chunk({'percent': 90})
        return 'ready'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, _response = await _collect_stream(ai, prompt='go', tools=[deploy])

    assert [p.tool_response.output for p in _partial_parts(chunks) if p.tool_response] == [
        {'percent': 10},
        {'percent': 10},
        {'percent': 90},
    ]


@pytest.mark.asyncio
async def test_generate_stream_excludes_partial_chunks_from_next_model_turn_and_response_messages() -> None:
    """Partial tool chunks are never added to ModelRequest.messages on the next turn or response.messages."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    seen_requests: list[ModelRequest[Any]] = []

    class RecordModelRequests(BaseMiddleware):
        async def wrap_model(self, params: Any, ctx: GenerateMiddlewareContext, next_fn: Any) -> ModelResponse[Any]:
            seen_requests.append(params.request.model_copy(deep=True))
            return await next_fn(params, ctx)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('transient-step')
        return 'final-result'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    _chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], use=[RecordModelRequests()])

    assert len(seen_requests) == 2
    second_turn_parts = _message_tool_response_parts(seen_requests[1].messages)
    assert len(second_turn_parts) == 1
    assert second_turn_parts[0].is_partial is False
    assert second_turn_parts[0].tool_response is not None
    assert second_turn_parts[0].tool_response.output == 'final-result'

    response_parts = _message_tool_response_parts(response.messages)
    assert len(response_parts) == 1
    assert response_parts[0].is_partial is False
    assert _tool_outputs(response.messages) == ['final-result']


@pytest.mark.asyncio
async def test_generate_stream_multi_round_tool_calls_keep_partial_chunks_ephemeral() -> None:
    """After a finished tool round, a later tool's send_chunk still emits partial tool responses."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='weather')
    async def weather(_req: dict[str, Any]) -> str:
        return '72F'

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('live')
        return 'done'

    pm.responses = [
        _model_calls(('weather', 'w1')),
        _model_calls(('deploy', 'd1')),
        _model_says('ok'),
    ]
    chunks, response = await _collect_stream(ai, prompt='do both', tools=[weather, deploy])

    partials = _partial_parts(chunks)
    assert len(partials) == 1
    assert partials[0].tool_response is not None
    assert partials[0].tool_response.name == 'deploy'
    assert partials[0].tool_response.ref == 'd1'
    assert partials[0].tool_response.output == 'live'
    assert [m.role for m in response.messages[:3]] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_outputs(response.messages) == ['72F', 'done']


@pytest.mark.asyncio
async def test_generate_stream_output_schema_returns_none_on_tool_chunks_and_parses_model_chunks() -> None:
    """chunk.output is None on role=Role.TOOL chunks even when output_schema=Recipe is configured."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[dict[str, str]]) -> str:
        ctx.send_chunk({'title': 'Not a recipe'})
        return 'done'

    recipe_json = '{"title": "Pie", "steps": ["bake"]}'
    pm.chunks = [
        [
            ModelResponseChunk(
                role=Role.MODEL,
                content=[
                    _text_part(recipe_json),
                    Part(tool_request=ToolRequest(name='deploy', ref='r1', input={})),
                ],
            )
        ],
        [ModelResponseChunk(role=Role.MODEL, content=[_text_part(recipe_json)])],
    ]
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    _text_part(recipe_json),
                    Part(tool_request=ToolRequest(name='deploy', ref='r1', input={})),
                ],
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[_text_part(recipe_json)]),
        ),
    ]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], output_schema=Recipe)

    tool_chunks = [c for c in chunks if c.role == Role.TOOL]
    assert len(tool_chunks) == 2
    assert all(c.output is None for c in tool_chunks)
    assert all(c.accumulated_text == '' for c in tool_chunks)

    model_chunks = [c for c in chunks if c.role == Role.MODEL]
    assert model_chunks
    last_output = model_chunks[-1].output
    assert isinstance(last_output, Recipe)
    assert last_output.title == 'Pie'
    final_output = response.output
    assert isinstance(final_output, Recipe)
    assert final_output.title == 'Pie'


def test_part_is_partial_requires_tool_response_and_exact_boolean_true() -> None:
    """Part.is_partial is True only on tool_response parts with metadata={'partial': True}."""
    values: list[object] = [True, False, 'true', 1, None]
    parts = [
        Part.from_tool_response(
            name='deploy',
            ref='call-1',
            output='x',
            metadata={'partial': value} if value is not None else None,
        )
        for value in values
    ]
    assert [p.is_partial for p in parts] == [True, False, False, False, False]

    text_with_partial_meta = Part.from_text('hello', metadata={'partial': True})
    assert text_with_partial_meta.is_partial is False


def test_model_response_chunk_tool_responses_returns_only_response_parts() -> None:
    """ModelResponseChunk.tool_responses returns partial and final tool responses while ignoring other parts."""
    partial = Part.from_tool_response(name='deploy', ref='call-1', output='half', metadata={'partial': True})
    final = Part.from_tool_response(name='deploy', ref='call-1', output='ready')
    chunk = ModelResponseChunk(
        role=Role.TOOL,
        content=[_text_part('ignored'), partial, final],
    )
    assert chunk.tool_responses == [partial, final]


@pytest.mark.asyncio
async def test_generate_non_streaming_reports_not_streaming_and_ignores_send_chunk() -> None:
    """await ai.generate() has ctx.is_streaming == False and ignores send_chunk."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    streaming_states: list[bool] = []

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        streaming_states.append(ctx.is_streaming)
        ctx.send_chunk('live')
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    response = await ai.generate(prompt='go', tools=[deploy])

    assert response.text == 'ok'
    assert streaming_states == [False]
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_tool_context_reports_is_streaming_true() -> None:
    """ai.generate_stream() sets ctx.is_streaming == True on ToolContext."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    streaming_states: list[bool] = []

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        streaming_states.append(ctx.is_streaming)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    await _collect_stream(ai, prompt='go', tools=[deploy])
    assert streaming_states == [True]


@pytest.mark.asyncio
async def test_generate_stream_concurrent_tools_attribute_name_and_ref_per_tool() -> None:
    """Concurrent tools attribute each partial chunk's name and ref to the emitting tool request."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    first_sent = asyncio.Event()
    second_sent = asyncio.Event()

    @ai.tool(name='alpha')
    async def alpha(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('a1')
        first_sent.set()
        await second_sent.wait()
        ctx.send_chunk('a2')
        return 'A'

    @ai.tool(name='beta')
    async def beta(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        await first_sent.wait()
        ctx.send_chunk('b1')
        second_sent.set()
        return 'B'

    pm.responses = [_model_calls(('alpha', 'ref-a'), ('beta', 'ref-b')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[alpha, beta])

    partials = _partial_parts(chunks)
    assert [
        (p.tool_response.name, p.tool_response.ref, p.tool_response.output) for p in partials if p.tool_response
    ] == [
        ('alpha', 'ref-a', 'a1'),
        ('beta', 'ref-b', 'b1'),
        ('alpha', 'ref-a', 'a2'),
    ]
    finals = _final_parts(chunks)
    assert [(p.tool_response.name, p.tool_response.ref, p.tool_response.output) for p in finals if p.tool_response] == [
        ('alpha', 'ref-a', 'A'),
        ('beta', 'ref-b', 'B'),
    ]
    assert len({c.index for c in chunks if c.role == Role.TOOL}) == 1
    assert _tool_outputs(response.messages) == ['A', 'B']


@pytest.mark.asyncio
async def test_generate_stream_same_named_tools_distinguish_progress_by_ref() -> None:
    """Separate calls to the same tool remain distinguishable through their ref."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(inp: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk(f'step-{inp["id"]}')
        return str(inp['id'])

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(tool_request=ToolRequest(name='deploy', ref='ref-a', input={'id': 'a'})),
                    Part(tool_request=ToolRequest(name='deploy', ref='ref-b', input={'id': 'b'})),
                ],
            ),
        ),
        _model_says('ok'),
    ]
    chunks, _response = await _collect_stream(ai, prompt='go', tools=[deploy])

    assert {(p.tool_response.ref, p.tool_response.output) for p in _partial_parts(chunks) if p.tool_response} == {
        ('ref-a', 'step-a'),
        ('ref-b', 'step-b'),
    }


@pytest.mark.asyncio
async def test_generate_stream_drops_late_send_chunk_after_tool_returns() -> None:
    """Late send_chunk after a tool returns is dropped because active is False."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    send_late = asyncio.Event()
    late_finished = asyncio.Event()

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        async def later() -> None:
            await send_late.wait()
            ctx.send_chunk('late')
            late_finished.set()

        asyncio.create_task(later())
        ctx.send_chunk('on-time')
        return 'ready'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, _response = await _collect_stream(ai, prompt='go', tools=[deploy])
    send_late.set()
    await late_finished.wait()

    assert [p.tool_response.output for p in _partial_parts(chunks) if p.tool_response] == ['on-time']


@pytest.mark.asyncio
async def test_generate_stream_drops_late_send_chunk_after_tool_interrupts() -> None:
    """Late send_chunk after a tool interrupts is dropped."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    send_late = asyncio.Event()
    late_finished = asyncio.Event()

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        async def later() -> None:
            await send_late.wait()
            ctx.send_chunk('late')
            late_finished.set()

        asyncio.create_task(later())
        ctx.send_chunk('on-time')
        raise Interrupt({'reason': 'approval'})

    pm.responses = [_model_calls(('deploy', 'r1'))]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    send_late.set()
    await late_finished.wait()

    assert response.finish_reason == FinishReason.INTERRUPTED
    assert [p.tool_response.output for p in _partial_parts(chunks) if p.tool_response] == ['on-time']


@pytest.mark.asyncio
async def test_generate_stream_nested_tool_call_does_not_inherit_outer_chunk_sender() -> None:
    """A nested direct tool call inside an active tool does not inherit _tool_generate_chunk_sender."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='inner_helper')
    async def inner_helper(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('leaked-from-inner')
        return 'inner-done'

    @ai.tool(name='outer_deploy')
    async def outer_deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('outer-before')
        inner_res = await inner_helper({})
        ctx.send_chunk(f'outer-after:{inner_res}')
        return 'outer-done'

    pm.responses = [_model_calls(('outer_deploy', 'outer-1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[outer_deploy])

    partials = _partial_parts(chunks)
    assert [
        (p.tool_response.name, p.tool_response.ref, p.tool_response.output) for p in partials if p.tool_response
    ] == [
        ('outer_deploy', 'outer-1', 'outer-before'),
        ('outer_deploy', 'outer-1', 'outer-after:inner-done'),
    ]
    assert _tool_outputs(response.messages) == ['outer-done']


@pytest.mark.asyncio
async def test_generate_stream_interrupt_after_send_chunk_keeps_partial_out_of_messages() -> None:
    """send_chunk followed by Interrupt streams the partial chunk but excludes it from response.messages."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('waiting')
        raise Interrupt({'hold': True})

    pm.responses = [_model_calls(('deploy', 'r1'))]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])

    assert [p.tool_response.output for p in _partial_parts(chunks) if p.tool_response] == ['waiting']
    assert _final_parts(chunks) == []
    assert response.finish_reason == FinishReason.INTERRUPTED
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert _message_tool_response_parts(response.messages) == []


@pytest.mark.asyncio
async def test_generate_stream_interrupted_sibling_keeps_completed_output_pending_without_partials() -> None:
    """A mixed tool round persists completed output on pendingOutput metadata and drops partial chunks."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='complete')
    async def complete(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('working-a')
        return 'A'

    @ai.tool(name='pause')
    async def pause(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('working-b')
        raise Interrupt({'reason': 'approval'})

    pm.responses = [_model_calls(('complete', 'a'), ('pause', 'b'))]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[complete, pause])

    assert {p.tool_response.output for p in _partial_parts(chunks) if p.tool_response} == {'working-a', 'working-b'}
    assert _final_parts(chunks) == []
    assert response.finish_reason == FinishReason.INTERRUPTED
    requests = response.message.tool_requests if response.message is not None else []
    assert requests[0].metadata == {'pendingOutput': 'A'}
    assert requests[1].metadata == {'interrupt': {'reason': 'approval'}}
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]


@pytest.mark.asyncio
async def test_generate_stream_resume_restart_emits_fresh_partial_without_replaying_first_run() -> None:
    """Restarting an interrupted tool emits fresh partial chunks and never replays the first run's partials."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    runs = {'n': 0}

    @ai.tool(name='pay')
    async def pay(inp: dict[str, Any], ctx: ToolContext[str]) -> str:
        runs['n'] += 1
        if not inp.get('ok'):
            ctx.send_chunk('first-run')
            raise Interrupt({'hold': True})
        ctx.send_chunk('second-run')
        return 'paid'

    pm.responses = [_model_calls(('pay', 'p1')), _model_says('final')]
    first_chunks, first = await _collect_stream(ai, prompt='go', tools=[pay])
    assert [p.tool_response.output for p in _partial_parts(first_chunks) if p.tool_response] == ['first-run']

    restart = first.interrupts[0].restart(replace_input={'ok': True})
    second_chunks, second = await _collect_stream(
        ai,
        messages=list(first.messages),
        tools=[pay],
        resume_restart=restart,
    )
    assert [p.tool_response.output for p in _partial_parts(second_chunks) if p.tool_response] == ['second-run']
    assert [p.tool_response.output for p in _final_parts(second_chunks) if p.tool_response] == ['paid']
    assert all(not p.is_partial for p in _message_tool_response_parts(second.messages))
    assert _tool_outputs(second.messages) == ['paid']
    assert runs['n'] == 2


@pytest.mark.asyncio
async def test_generate_stream_resume_respond_emits_no_partial_chunks() -> None:
    """Answering an interrupt via resume_respond emits only the final tool response, no partials."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    runs = {'n': 0}

    @ai.tool(name='pay')
    async def pay(_inp: dict[str, Any], ctx: ToolContext[str]) -> str:
        runs['n'] += 1
        ctx.send_chunk('first-run')
        raise Interrupt({'hold': True})

    pm.responses = [_model_calls(('pay', 'p1')), _model_says('final')]
    _first_chunks, first = await _collect_stream(ai, prompt='go', tools=[pay])

    respond = first.interrupts[0].respond('approved')
    second_chunks, second = await _collect_stream(
        ai,
        messages=list(first.messages),
        tools=[pay],
        resume_respond=respond,
    )
    assert _partial_parts(second_chunks) == []
    assert [p.tool_response.output for p in _final_parts(second_chunks) if p.tool_response] == ['approved']
    assert second.finish_reason == FinishReason.STOP
    assert runs['n'] == 1


@pytest.mark.asyncio
async def test_generate_stream_abort_after_send_chunk_keeps_chunk_off_messages() -> None:
    """Abort after send_chunk streams the partial chunk and keeps it off response.messages."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('live')
        ctx.abort_signal.set()
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])

    assert response.finish_reason == FinishReason.ABORTED
    assert response.error is not None
    assert response.error.status == 'CANCELLED'
    assert [p.tool_response.output for p in _partial_parts(chunks) if p.tool_response] == ['live']
    assert all(not p.is_partial for p in _message_tool_response_parts(response.messages))


@pytest.mark.asyncio
async def test_generate_stream_max_turns_after_send_chunk_keeps_chunk_off_messages() -> None:
    """Hitting max_turns after send_chunk keeps the partial chunk off response.messages."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('live')
        return '72F'

    pm.responses = [_model_calls(('lookup', 'r1')), _model_calls(('lookup', 'r2'))]
    chunks, response = await _collect_stream(ai, prompt='keep going', tools=[lookup], max_turns=1)

    assert [p.tool_response.output for p in _partial_parts(chunks) if p.tool_response] == ['live']
    assert all(not p.is_partial for p in _message_tool_response_parts(response.messages))
    assert _tool_outputs(response.messages) == ['72F']


@pytest.mark.asyncio
async def test_generate_stream_return_tool_requests_never_runs_tool() -> None:
    """return_tool_requests=True does not run the tool, so send_chunk never fires."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran = {'n': 0}

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ran['n'] += 1
        ctx.send_chunk('live')
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1'))]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], return_tool_requests=True)
    assert ran['n'] == 0
    assert _partial_parts(chunks) == []
    assert response.tool_requests


@pytest.mark.asyncio
async def test_generate_stream_tool_failure_after_send_chunk_emits_no_final_tool_response() -> None:
    """A tool that raises after send_chunk leaves the partial chunk on the stream and emits no final response."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('started')
        raise RuntimeError('deployment failed')

    pm.responses = [_model_calls(('deploy', 'r1'))]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])

    assert [p.tool_response.output for p in _partial_parts(chunks) if p.tool_response] == ['started']
    assert _final_parts(chunks) == []
    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.details == {'reason': 'TOOL_FAILED'}


@pytest.mark.asyncio
async def test_generate_stream_error_during_partial_chunk_is_swallowed_and_tool_returns() -> None:
    """A listener error on a partial tool chunk is dropped; the tool still returns its final response."""

    class BoomOnPartial(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Any,
        ) -> ModelResponse[Any]:
            downstream = ctx.on_chunk

            def handler(chunk: ModelResponseChunk[Any]) -> None:
                if chunk.role == Role.TOOL and any(p.is_partial for p in chunk.tool_responses):
                    raise RuntimeError('sink')
                if downstream is not None:
                    downstream(chunk)

            previous = ctx.replace_on_chunk(handler)
            try:
                return await next_fn(params, ctx)
            finally:
                ctx.replace_on_chunk(previous)

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('live')
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], use=[BoomOnPartial()])
    assert _partial_parts(chunks) == []
    assert [p.tool_response.output for p in _final_parts(chunks) if p.tool_response] == ['done']
    assert _tool_outputs(response.messages) == ['done']
    assert response.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_generate_stream_error_on_final_tool_response_fails_generate() -> None:
    """A listener error on the final tool response chunk fails generate."""

    class BoomOnFinal(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Any,
        ) -> ModelResponse[Any]:
            downstream = ctx.on_chunk

            def handler(chunk: ModelResponseChunk[Any]) -> None:
                if chunk.role == Role.TOOL and any(not p.is_partial for p in chunk.tool_responses):
                    raise RuntimeError('final sink')
                if downstream is not None:
                    downstream(chunk)

            previous = ctx.replace_on_chunk(handler)
            try:
                return await next_fn(params, ctx)
            finally:
                ctx.replace_on_chunk(previous)

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[str]) -> str:
        ctx.send_chunk('live')
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], use=[BoomOnFinal()])
    assert [p.tool_response.output for p in _partial_parts(chunks) if p.tool_response] == ['live']
    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_define_prompt_stream_surfaces_partial_and_final_tool_responses() -> None:
    """define_prompt(...).stream() emits partial and final tool responses like generate_stream."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[Step]) -> str:
        ctx.send_chunk(Step(stage='verify', percent=75))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    prompt = ai.define_prompt(prompt='go', tools=[deploy])
    stream = prompt.stream()
    chunks = [chunk async for chunk in stream.stream]
    response = await stream.response

    partials = _partial_parts(chunks)
    assert len(partials) == 1
    assert partials[0].tool_response is not None
    assert Step.model_validate(partials[0].tool_response.output) == Step(stage='verify', percent=75)
    assert all(not p.is_partial for p in _message_tool_response_parts(response.messages))
    assert _tool_outputs(response.messages) == ['done']


def test_partial_tool_response_serializes_with_canonical_wire_shape() -> None:
    """Wire JSON serialization produces camelCase toolResponse and metadata.partial == True."""
    chunk = ModelResponseChunk(
        role=Role.TOOL,
        index=1,
        content=[
            Part.from_tool_response(
                name='deploy',
                ref='call-1',
                output=Step(stage='uploading', percent=50).model_dump(mode='json'),
                metadata={'partial': True},
            )
        ],
    )

    assert json.loads(chunk.model_dump_json(by_alias=True, exclude_none=True)) == {
        'role': 'tool',
        'index': 1.0,
        'content': [
            {
                'toolResponse': {
                    'ref': 'call-1',
                    'name': 'deploy',
                    'output': {'stage': 'uploading', 'percent': 50},
                },
                'metadata': {'partial': True},
            }
        ],
    }


@pytest.mark.asyncio
async def test_agent_send_stream_surfaces_tool_responses_and_excludes_partials_from_messages() -> None:
    """Agent chat.send_stream() surfaces partial and final tool_responses while excluding partials from messages."""
    from genkit.exp import Genkit as ExpGenkit

    ai = ExpGenkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict[str, Any], ctx: ToolContext[Step]) -> str:
        ctx.send_chunk(Step(stage='uploading', percent=50))
        return 'done'

    ai.define_prompt(name='shipAgent', model='scriptedModel', tools=[deploy])
    agent = ai.define_prompt_agent(name='shipAgent')
    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]

    chat = agent.chat()
    turn = chat.send_stream('go')
    partial_outputs: list[Step] = []
    final_outputs: list[object] = []
    async for item in turn.stream:
        for tr_part in item.tool_responses:
            assert tr_part.tool_response is not None
            if tr_part.is_partial:
                partial_outputs.append(Step.model_validate(tr_part.tool_response.output))
            else:
                final_outputs.append(tr_part.tool_response.output)
    await turn.response

    assert partial_outputs == [Step(stage='uploading', percent=50)]
    assert final_outputs == ['done']

    tool_msgs = [m for m in chat.messages if m.role == Role.TOOL]
    assert len(tool_msgs) == 1
    assert len(tool_msgs[0].content) == 1
    assert tool_msgs[0].content[0].is_partial is False
    assert tool_msgs[0].content[0].tool_response is not None
    assert tool_msgs[0].content[0].tool_response.output == 'done'


def test_streamed_message_accumulator_skips_partial_parts_without_flushing() -> None:
    """StreamedMessageAccumulator ignores partial tool response chunks and keeps only final parts."""
    acc = StreamedMessageAccumulator()
    acc.add(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.MODEL,
                index=0,
                content=[Part.from_tool_request(name='deploy', ref='r1', input={})],
            )
        )
    )
    acc.add(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.TOOL,
                index=1,
                content=[
                    Part.from_tool_response(
                        name='deploy',
                        ref='r1',
                        output={'stage': 'uploading', 'percent': 50},
                        metadata={'partial': True},
                    )
                ],
            )
        )
    )
    acc.add(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.TOOL,
                index=1,
                content=[Part.from_tool_response(name='deploy', ref='r1', output='done')],
            )
        )
    )

    msgs = acc.messages()
    assert [m.role for m in msgs] == [Role.MODEL, Role.TOOL]
    assert len(msgs[1].content) == 1
    assert msgs[1].content[0].is_partial is False
    assert msgs[1].content[0].tool_response is not None
    assert msgs[1].content[0].tool_response.output == 'done'
