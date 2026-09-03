#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Product contract tests for progress streamed by automatically executed tools."""

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk
from genkit._ai._generate import generate_action
from genkit._ai._testing import ProgrammableModel, define_programmable_model
from genkit._ai._tools import Interrupt, ToolRunContext, respond_to_interrupt, restart_tool
from genkit._core._error import GenkitError
from genkit._core._model import GenerateActionOptions
from genkit._core._typing import (
    FinishReason,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponsePart,
)


class Progress(BaseModel):
    """Structured progress emitted by the test tools."""

    step: str
    percent: int


class Report(BaseModel):
    """Structured model result produced after the tool turn."""

    summary: str


def model_calls_tool(*, name: str = 'deploy', ref: str = 'call-1', input: object = None) -> ModelResponse:
    """Build a model response containing one tool request."""
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(name=name, ref=ref, input={} if input is None else input)
                    )
                )
            ],
        ),
    )


def model_finishes(text: str = 'done') -> ModelResponse:
    """Build a final model response."""
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=text))]),
    )


def setup_model(ai: Genkit, *responses: ModelResponse) -> ProgrammableModel:
    """Register a programmable model with the supplied turn responses."""
    pm, _ = define_programmable_model(ai)
    pm.responses = list(responses)
    return pm


async def collect_stream(ai: Genkit, *, tools: list[object]) -> tuple[list[ModelResponseChunk[Any]], ModelResponse]:
    """Collect one public generate stream and its final response."""
    stream = ai.generate_stream(model='programmableModel', prompt='deploy it', tools=tools)
    chunks = [chunk async for chunk in stream.stream]
    return chunks, await stream.response


def partials(chunks: list[ModelResponseChunk[Any]]) -> list[ToolResponsePart]:
    """Return all partial tool responses from a stream."""
    return [response for chunk in chunks for response in chunk.tool_responses if is_partial(response)]


def finals(chunks: list[ModelResponseChunk[Any]]) -> list[ToolResponsePart]:
    """Return all authoritative tool responses from a stream."""
    return [response for chunk in chunks for response in chunk.tool_responses if not is_partial(response)]


def message_tool_responses(message: Message) -> list[ToolResponsePart]:
    """Return tool responses from a durable message."""
    return [part.root for part in message.content if isinstance(part.root, ToolResponsePart)]


def is_partial(response: ToolResponsePart) -> bool:
    """Return whether a tool response is transient progress."""
    return (response.metadata or {}).get('partial') is True


@pytest.mark.asyncio
async def test_generate_stream_tool_send_partial_emits_attributed_progress_before_final_response() -> None:
    """Each update identifies its tool call and arrives before the authoritative response."""
    ai = Genkit()
    setup_model(ai, model_calls_tool(), model_finishes())

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(Progress(step='uploading', percent=50))
        return 'https://example.run'

    chunks, response = await collect_stream(ai, tools=[deploy])

    [progress] = partials(chunks)
    [final] = finals(chunks)
    assert progress.tool_response.name == 'deploy'
    assert progress.tool_response.ref == 'call-1'
    assert Progress.model_validate(progress.tool_response.output) == Progress(step='uploading', percent=50)
    assert chunks.index(next(c for c in chunks if progress in c.tool_responses)) < chunks.index(
        next(c for c in chunks if final in c.tool_responses)
    )
    assert final.tool_response.output == 'https://example.run'
    assert all(not is_partial(part) for message in response.messages for part in message_tool_responses(message))


@pytest.mark.asyncio
async def test_generate_stream_tool_send_partial_preserves_each_update() -> None:
    """Repeated progress values remain separate updates rather than being merged or deduplicated."""
    ai = Genkit()
    setup_model(ai, model_calls_tool(), model_finishes())

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial({'percent': 10})
        ctx.send_partial({'percent': 10})
        ctx.send_partial({'percent': 90})
        return 'ready'

    chunks, _ = await collect_stream(ai, tools=[deploy])

    assert [part.tool_response.output for part in partials(chunks)] == [
        {'percent': 10},
        {'percent': 10},
        {'percent': 90},
    ]


@pytest.mark.asyncio
async def test_generate_stream_typed_model_output_continues_after_tool_progress() -> None:
    """Tool chunks expose no model output while the following model turn remains typed."""
    ai = Genkit()
    pm = setup_model(
        ai,
        model_calls_tool(),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='{"summary":"ready"}'))]),
        ),
    )
    pm.chunks = [
        [
            ModelResponseChunk(
                role=Role.MODEL,
                content=[
                    Part(root=TextPart(text='{"summary":"stale"}')),
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='deploy', ref='call-1', input={}))),
                ],
            )
        ],
        [ModelResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='{"summary":"ready"}'))])],
    ]

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial({'not': 'a report'})
        return 'ready'

    stream = ai.generate_stream(
        model='programmableModel',
        prompt='deploy it',
        tools=[deploy],
        output_schema=Report,
    )
    chunks = [chunk async for chunk in stream.stream]
    response = await stream.response

    tool_chunks = [chunk for chunk in chunks if chunk.role == Role.TOOL]
    assert len(tool_chunks) == 2
    assert all(chunk.output is None for chunk in tool_chunks)
    assert all(chunk.accumulated_text == '' for chunk in tool_chunks)
    model_outputs = [chunk.output for chunk in chunks if chunk.role == Role.MODEL]
    assert isinstance(model_outputs[-1], Report)
    assert model_outputs[-1].summary == 'ready'
    assert isinstance(response.output, Report)
    assert response.output.summary == 'ready'


def test_tool_response_part_only_exact_true_is_partial() -> None:
    """Only the canonical boolean metadata value marks a response as transient."""
    response = ToolRequest(name='deploy', ref='call-1', input={})
    values = [True, False, 'true', 1, None]
    parts = [
        ToolResponsePart.model_validate({
            'toolResponse': {'name': response.name, 'ref': response.ref, 'output': 'x'},
            'metadata': {'partial': value} if value is not None else None,
        })
        for value in values
    ]

    assert [is_partial(part) for part in parts] == [True, False, False, False, False]


def test_model_response_chunk_tool_responses_returns_only_response_parts() -> None:
    """The chunk accessor returns partial and final responses while ignoring unrelated parts."""
    partial = ToolResponsePart.model_validate({
        'toolResponse': {'name': 'deploy', 'ref': 'call-1', 'output': 'half'},
        'metadata': {'partial': True},
    })
    final = ToolResponsePart.model_validate({
        'toolResponse': {'name': 'deploy', 'ref': 'call-1', 'output': 'ready'},
    })
    chunk = ModelResponseChunk(
        role=Role.TOOL,
        content=[Part(root=TextPart(text='ignored')), Part(root=partial), Part(root=final)],
    )

    assert chunk.tool_responses == [partial, final]


@pytest.mark.asyncio
async def test_generate_tool_send_partial_is_noop_without_streaming() -> None:
    """Non-streaming generation runs the tool normally without offering progress delivery."""
    ai = Genkit()
    setup_model(ai, model_calls_tool(), model_finishes())
    streaming_states: list[bool] = []

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        streaming_states.append(ctx.is_streaming)
        ctx.send_partial({'percent': 50})
        return 'ready'

    response = await ai.generate(model='programmableModel', prompt='deploy it', tools=[deploy])

    assert response.text == 'done'
    assert streaming_states == [False]
    assert message_tool_responses(response.messages[2])[0].tool_response.output == 'ready'


@pytest.mark.asyncio
async def test_generate_stream_tool_context_reports_streaming_available() -> None:
    """A tool can avoid computing progress unless its automatic generate call is streaming."""
    ai = Genkit()
    setup_model(ai, model_calls_tool(), model_finishes())
    streaming_states: list[bool] = []

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        streaming_states.append(ctx.is_streaming)
        return 'ready'

    await collect_stream(ai, tools=[deploy])

    assert streaming_states == [True]


@pytest.mark.asyncio
async def test_direct_tool_stream_send_partial_is_noop_and_send_chunk_stays_raw() -> None:
    """Direct action streaming keeps raw chunks and has no model-bound partial-response channel."""
    ai = Genkit()

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial({'percent': 50})
        ctx.send_chunk('raw')
        return 'ready'

    stream = deploy.action().stream({})
    chunks = [chunk async for chunk in stream.stream]

    assert chunks == ['raw']
    assert await stream.response == 'ready'


@pytest.mark.asyncio
async def test_generate_stream_tool_send_chunk_does_not_inject_raw_chunk() -> None:
    """Automatic generation forwards only structured partial responses from a tool."""
    ai = Genkit()
    setup_model(ai, model_calls_tool(), model_finishes())

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk('raw')
        ctx.send_partial('structured')
        return 'ready'

    chunks, _ = await collect_stream(ai, tools=[deploy])

    assert [part.tool_response.output for part in partials(chunks)] == ['structured']
    assert all(chunk.text != 'raw' for chunk in chunks)


@pytest.mark.asyncio
async def test_generate_stream_concurrent_tools_preserve_attribution_and_per_tool_order() -> None:
    """Concurrent updates may interleave while each tool's order, name, and ref remain stable."""
    ai = Genkit()
    setup_model(
        ai,
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='first', ref='a', input={}))),
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='second', ref='b', input={}))),
                ],
            ),
        ),
        model_finishes(),
    )
    first_sent = asyncio.Event()
    second_sent = asyncio.Event()

    @ai.tool(name='first')
    async def first(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial('a1')
        first_sent.set()
        await second_sent.wait()
        ctx.send_partial('a2')
        return 'A'

    @ai.tool(name='second')
    async def second(_: dict, ctx: ToolRunContext) -> str:
        await first_sent.wait()
        ctx.send_partial('b1')
        second_sent.set()
        return 'B'

    chunks, _ = await collect_stream(ai, tools=[first, second])
    progress = partials(chunks)

    assert [(part.tool_response.ref, part.tool_response.output) for part in progress] == [
        ('a', 'a1'),
        ('b', 'b1'),
        ('a', 'a2'),
    ]
    assert [part.tool_response.ref for part in finals(chunks)] == ['a', 'b']
    assert len({chunk.index for chunk in chunks if chunk.role == Role.TOOL}) == 1


@pytest.mark.asyncio
async def test_generate_stream_same_named_tools_distinguishes_progress_by_ref() -> None:
    """Separate calls to one tool remain distinguishable through their model-assigned references."""
    ai = Genkit()
    setup_model(
        ai,
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='deploy', ref='a', input={'id': 'a'}))),
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='deploy', ref='b', input={'id': 'b'}))),
                ],
            )
        ),
        model_finishes(),
    )

    @ai.tool(name='deploy')
    async def deploy(inp: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(inp['id'])
        return inp['id']

    chunks, _ = await collect_stream(ai, tools=[deploy])

    assert {(part.tool_response.ref, part.tool_response.output) for part in partials(chunks)} == {
        ('a', 'a'),
        ('b', 'b'),
    }


@pytest.mark.asyncio
async def test_generate_stream_progress_callback_failure_does_not_fail_tool() -> None:
    """Losing transient progress does not change the authoritative tool or model result."""
    ai = Genkit()
    setup_model(ai, model_calls_tool(), model_finishes())
    delivered: list[ModelResponseChunk[Any]] = []

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial('halfway')
        return 'ready'

    def on_chunk(chunk: ModelResponseChunk[Any]) -> None:
        if any(is_partial(part) for part in chunk.tool_responses):
            raise RuntimeError('sink closed')
        delivered.append(chunk)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='deploy'))])],
            tools=['deploy'],
        ),
        on_chunk=on_chunk,
    )

    assert response.text == 'done'
    assert finals(delivered)[0].tool_response.output == 'ready'


@pytest.mark.asyncio
async def test_generate_stream_tool_failure_after_progress_emits_no_final_tool_response() -> None:
    """A failed tool leaves delivered progress visible without inventing an authoritative response."""
    ai = Genkit()
    setup_model(ai, model_calls_tool())

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial('started')
        raise RuntimeError('deployment failed')

    stream = ai.generate_stream(model='programmableModel', prompt='deploy', tools=[deploy])
    chunks = [chunk async for chunk in stream.stream]

    assert [part.tool_response.output for part in partials(chunks)] == ['started']
    assert finals(chunks) == []
    with pytest.raises(GenkitError, match='deployment failed'):
        await stream.response


@pytest.mark.asyncio
async def test_generate_stream_tool_interrupt_after_progress_keeps_progress_out_of_history() -> None:
    """An interrupted turn exposes live progress but retains only the resumable model request."""
    ai = Genkit()
    setup_model(ai, model_calls_tool())

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial('waiting')
        raise Interrupt({'reason': 'approval'})

    chunks, response = await collect_stream(ai, tools=[deploy])

    assert [part.tool_response.output for part in partials(chunks)] == ['waiting']
    assert finals(chunks) == []
    assert response.finish_reason == FinishReason.INTERRUPTED
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]
    assert all(not message_tool_responses(message) for message in response.messages)


@pytest.mark.asyncio
async def test_generate_stream_interrupted_sibling_keeps_completed_output_pending() -> None:
    """A mixed tool round persists completed output but none of either tool's progress."""
    ai = Genkit()
    setup_model(
        ai,
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='complete', ref='a', input={}))),
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='pause', ref='b', input={}))),
                ],
            )
        ),
    )

    @ai.tool(name='complete')
    async def complete(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial('working-a')
        return 'A'

    @ai.tool(name='pause')
    async def pause(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial('working-b')
        raise Interrupt({'reason': 'approval'})

    chunks, response = await collect_stream(ai, tools=[complete, pause])

    assert {part.tool_response.output for part in partials(chunks)} == {'working-a', 'working-b'}
    assert finals(chunks) == []
    assert response.finish_reason == FinishReason.INTERRUPTED
    requests = response.message.tool_requests if response.message is not None else []
    assert requests[0].metadata == {'pendingOutput': 'A'}
    assert requests[1].metadata == {'interrupt': {'reason': 'approval'}}
    assert [message.role for message in response.messages] == [Role.USER, Role.MODEL]


@pytest.mark.asyncio
async def test_generate_stream_restarted_tool_emits_fresh_progress_without_replay() -> None:
    """Restart executes with a new progress stream and never replays the interrupted run."""
    ai = Genkit()
    setup_model(ai, model_calls_tool(), model_finishes('resumed'))
    runs = 0

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        nonlocal runs
        runs += 1
        ctx.send_partial(f'run-{runs}')
        if not ctx.is_resumed():
            raise Interrupt({'reason': 'approval'})
        return 'ready'

    first_chunks, first = await collect_stream(ai, tools=[deploy])
    assert [part.tool_response.output for part in partials(first_chunks)] == ['run-1']

    stream = ai.generate_stream(
        model='programmableModel',
        messages=list(first.messages),
        tools=[deploy],
        resume_restart=[restart_tool(interrupt=first.interrupts[0])],
    )
    second_chunks = [chunk async for chunk in stream.stream]
    second = await stream.response

    assert [part.tool_response.output for part in partials(second_chunks)] == ['run-2']
    assert second.finish_reason == FinishReason.STOP
    assert runs == 2


@pytest.mark.asyncio
async def test_generate_stream_resume_with_response_emits_no_tool_progress() -> None:
    """Supplying an interrupted tool's response does not impersonate live execution."""
    ai = Genkit()
    setup_model(ai, model_calls_tool(), model_finishes('resumed'))
    runs = 0

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        nonlocal runs
        runs += 1
        ctx.send_partial('run-1')
        raise Interrupt({'reason': 'approval'})

    _, first = await collect_stream(ai, tools=[deploy])
    stream = ai.generate_stream(
        model='programmableModel',
        messages=list(first.messages),
        tools=[deploy],
        resume_respond=[respond_to_interrupt('approved', interrupt=first.interrupts[0])],
    )
    chunks = [chunk async for chunk in stream.stream]
    response = await stream.response

    assert partials(chunks) == []
    assert response.finish_reason == FinishReason.STOP
    assert runs == 1


@pytest.mark.asyncio
async def test_generate_stream_abort_after_progress_stops_without_final_tool_response() -> None:
    """Cancellation retains already delivered progress and never reports tool completion."""
    ai = Genkit()
    setup_model(ai, model_calls_tool())
    abort = asyncio.Event()

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial('started')
        abort.set()
        await asyncio.Event().wait()
        return 'unreachable'

    delivered: list[ModelResponseChunk[Any]] = []
    with pytest.raises(GenkitError, match='Task aborted'):
        await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='deploy'))])],
                tools=['deploy'],
            ),
            on_chunk=delivered.append,
            abort_signal=abort,
        )

    assert [part.tool_response.output for part in partials(delivered)] == ['started']
    assert finals(delivered) == []


@pytest.mark.asyncio
async def test_generate_stream_drops_progress_sent_after_tool_returns() -> None:
    """A retained tool context cannot leak late progress into a later model turn."""
    ai = Genkit()
    setup_model(ai, model_calls_tool(), model_finishes())
    send_late = asyncio.Event()
    late_finished = asyncio.Event()

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        async def later() -> None:
            await send_late.wait()
            ctx.send_partial('late')
            late_finished.set()

        asyncio.create_task(later())
        ctx.send_partial('on-time')
        return 'ready'

    chunks, _ = await collect_stream(ai, tools=[deploy])
    send_late.set()
    await late_finished.wait()

    assert [part.tool_response.output for part in partials(chunks)] == ['on-time']


@pytest.mark.asyncio
async def test_generate_stream_drops_progress_sent_after_tool_interrupts() -> None:
    """A retained interrupted context cannot emit progress after the turn becomes resumable."""
    ai = Genkit()
    setup_model(ai, model_calls_tool())
    send_late = asyncio.Event()
    late_finished = asyncio.Event()

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        async def later() -> None:
            await send_late.wait()
            ctx.send_partial('late')
            late_finished.set()

        asyncio.create_task(later())
        ctx.send_partial('on-time')
        raise Interrupt({'reason': 'approval'})

    chunks, response = await collect_stream(ai, tools=[deploy])
    send_late.set()
    await late_finished.wait()

    assert response.finish_reason == FinishReason.INTERRUPTED
    assert [part.tool_response.output for part in partials(chunks)] == ['on-time']


@pytest.mark.asyncio
async def test_generate_stream_return_tool_requests_emits_no_tool_progress() -> None:
    """Returning tool requests to the caller never starts a progress-capable tool execution."""
    ai = Genkit()
    setup_model(ai, model_calls_tool())
    calls = 0

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        nonlocal calls
        calls += 1
        ctx.send_partial('unexpected')
        return 'ready'

    stream = ai.generate_stream(
        model='programmableModel',
        prompt='deploy',
        tools=[deploy],
        return_tool_requests=True,
    )
    chunks = [chunk async for chunk in stream.stream]
    response = await stream.response

    assert calls == 0
    assert partials(chunks) == []
    assert response.tool_requests[0].tool_request.name == 'deploy'


@pytest.mark.asyncio
async def test_generate_stream_unknown_tool_emits_no_tool_progress() -> None:
    """A model request for an unavailable tool fails before any tool can emit progress."""
    ai = Genkit()
    setup_model(ai, model_calls_tool(name='missing'))

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial('unexpected')
        return 'ready'

    stream = ai.generate_stream(model='programmableModel', prompt='deploy', tools=[deploy])
    chunks = [chunk async for chunk in stream.stream]
    response = await stream.response

    assert partials(chunks) == []
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'Tool missing not found'


@pytest.mark.asyncio
async def test_generate_stream_max_turns_refuses_tool_without_progress() -> None:
    """A refused tool round emits neither transient progress nor an authoritative response."""
    ai = Genkit()
    setup_model(ai, model_calls_tool())
    calls = 0

    @ai.tool(name='deploy')
    async def deploy(_: dict, ctx: ToolRunContext) -> str:
        nonlocal calls
        calls += 1
        ctx.send_partial('unexpected')
        return 'ready'

    stream = ai.generate_stream(
        model='programmableModel',
        prompt='deploy',
        tools=[deploy],
        max_turns=0,
    )
    chunks = [chunk async for chunk in stream.stream]
    response = await stream.response

    assert calls == 0
    assert partials(chunks) == []
    assert response.finish_reason == FinishReason.ABORTED
    assert 'maximum tool call iterations' in (response.finish_message or '')


def test_partial_tool_response_serializes_with_canonical_wire_shape() -> None:
    """Network transports receive camelCase tool identity, output, and partial metadata."""
    chunk = ModelResponseChunk(
        ModelResponseChunk(
            role=Role.TOOL,
            content=[
                Part(
                    root=ToolResponsePart.model_validate({
                        'toolResponse': {
                            'name': 'deploy',
                            'ref': 'call-1',
                            'output': Progress(step='uploading', percent=50),
                        },
                        'metadata': {'partial': True},
                    })
                )
            ],
        ),
        index=1,
    )

    assert json.loads(chunk.model_dump_json()) == {
        'role': 'tool',
        'index': 1.0,
        'content': [
            {
                'toolResponse': {
                    'ref': 'call-1',
                    'name': 'deploy',
                    'output': {'step': 'uploading', 'percent': 50},
                },
                'metadata': {'partial': True},
            }
        ],
    }
