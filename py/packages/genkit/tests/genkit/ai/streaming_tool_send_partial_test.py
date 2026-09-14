#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Pins ``ToolRunContext.send_partial`` during ``generate_stream``.

``send_partial(value)`` shows up as a tool-role chunk with one
``ToolResponsePart``: this tool's name and ref, ``metadata.partial is True``,
and ``output`` is the value they passed. History only keeps the return.
``chunk.output`` is not that Progress. ``tool.stream()`` does not yield it.
Agent ``send_stream`` carries the same generate chunk (walk its parts).
"""

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, restart_tool
from genkit._ai._agents._client import StreamedMessageAccumulator
from genkit._ai._generate import mark_stream_only
from genkit._ai._testing import define_programmable_model
from genkit._ai._tools import Interrupt, ToolRunContext
from genkit._core._error import GenkitError
from genkit._core._middleware import BaseMiddleware, GenerateHookParams, GenerateMiddlewareContext
from genkit._core._typing import (
    AgentStreamChunk,
    FinishReason,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)


class Progress(BaseModel):
    step: str
    percent: int


class Recipe(BaseModel):
    title: str
    steps: list[str]


def _text_part(text: str) -> Part:
    return Part(TextPart(text=text))


def _model_calls(*tools: tuple[str, str]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[
                Part(root=ToolRequestPart(tool_request=ToolRequest(name=name, ref=ref, input={})))
                for name, ref in tools
            ],
        ),
    )


def _model_says(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[_text_part(text)]),
    )


def _message_texts(messages: Sequence[Message]) -> list[str]:
    texts: list[str] = []
    for msg in messages:
        for part in msg.content:
            text_val = getattr(part.root, 'text', None)
            if text_val:
                texts.append(str(text_val))
    return texts


def _tool_outputs(messages: Sequence[Message]) -> list[object]:
    out: list[object] = []
    for msg in messages:
        if msg.role != Role.TOOL:
            continue
        for part in msg.content:
            root = part.root
            if isinstance(root, ToolResponsePart):
                out.append(root.tool_response.output)
    return out


def _has_partial_part(messages: Sequence[Message]) -> bool:
    for msg in messages:
        for part in msg.content:
            root = part.root
            if isinstance(root, ToolResponsePart) and (root.metadata or {}).get('partial') is True:
                return True
    return False


def _partials(chunks: Sequence[ModelResponseChunk]) -> list[ToolResponsePart]:
    out: list[ToolResponsePart] = []
    for chunk in chunks:
        if chunk.role != Role.TOOL:
            continue
        for part in chunk.content:
            root = part.root
            if isinstance(root, ToolResponsePart) and (root.metadata or {}).get('partial') is True:
                out.append(root)
    return out


def _assert_stamped(part: ToolResponsePart, *, name: str, ref: str, output: object) -> None:
    assert part.tool_response.name == name
    assert part.tool_response.ref == ref
    assert (part.metadata or {}).get('partial') is True
    assert part.tool_response.output == output


async def _collect_stream(ai: Genkit, **kwargs: object) -> tuple[list[ModelResponseChunk], ModelResponse]:
    stream = ai.generate_stream(**kwargs)  # type: ignore[arg-type]
    chunks: list[ModelResponseChunk] = []
    async for chunk in stream.stream:
        chunks.append(chunk)
    return chunks, await stream.response


@pytest.mark.asyncio
async def test_generate_stream_send_partial_progress_is_tool_role_stamped_part() -> None:
    """send_partial(Progress) is a tool-role chunk with name, ref, and partial metadata."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(progress)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    sent = _partials(chunks)
    assert len(sent) == 1
    _assert_stamped(sent[0], name='deploy', ref='r1', output=progress)
    assert not _has_partial_part(response.messages)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_two_send_partials_are_two_progress_values() -> None:
    """Two send_partial calls are two chunks: 50 then 100, not one merged Progress."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    first = Progress(step='uploading', percent=50)
    second = Progress(step='health', percent=100)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(first)
        ctx.send_partial(second)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    sent = _partials(chunks)
    assert len(sent) == 2
    _assert_stamped(sent[0], name='deploy', ref='r1', output=first)
    _assert_stamped(sent[1], name='deploy', ref='r1', output=second)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_return_still_streams_final_tool_response() -> None:
    """After send_partial, the return still streams as a later tool-role response without partial."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(Progress(step='uploading', percent=50))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    finals = [
        chunk
        for chunk in chunks
        if chunk.role == Role.TOOL
        and any(
            isinstance(p.root, ToolResponsePart) and (p.root.metadata or {}).get('partial') is not True
            for p in chunk.content
        )
    ]
    assert finals
    assert _tool_outputs(response.messages) == ['done']
    assert not _has_partial_part(response.messages)


@pytest.mark.asyncio
async def test_generate_stream_after_a_finished_tool_round_send_partial_is_still_stamped() -> None:
    """A later tool's send_partial is still stamped; the first closed round stays."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='weather')
    async def weather(_req: dict) -> str:
        return '72F'

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(progress)
        return 'done'

    pm.responses = [
        _model_calls(('weather', 'w1')),
        _model_calls(('deploy', 'd1')),
        _model_says('ok'),
    ]
    chunks, response = await _collect_stream(ai, prompt='do both', tools=[weather, deploy])
    sent = _partials(chunks)
    _assert_stamped(sent[0], name='deploy', ref='d1', output=progress)
    assert _tool_outputs(response.messages)[0] == '72F'
    assert not _has_partial_part(response.messages)


@pytest.mark.asyncio
async def test_generate_stream_send_partial_dict_stays_a_dict() -> None:
    """send_partial({'percent': 50}) puts that dict on the part; it is not a Progress."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    payload = {'percent': 50}

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(payload)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    sent = _partials(chunks)
    _assert_stamped(sent[0], name='deploy', ref='r1', output=payload)
    assert isinstance(sent[0].tool_response.output, dict)
    assert not isinstance(sent[0].tool_response.output, Progress)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_send_partial_number_is_the_number() -> None:
    """send_partial(50) puts 50 on the part."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(50)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=50)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_send_partial_none_is_a_partial_with_no_output() -> None:
    """send_partial(None) is still a stamped partial; output is None."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(None)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=None)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_send_partial_part_is_the_part_on_output() -> None:
    """send_partial(Part(...)) puts that Part on output; it does not raise."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    part = _text_part('live')

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(part)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=part)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_send_partial_model_response_chunk_is_on_output() -> None:
    """send_partial(ModelResponseChunk(...)) puts that chunk on output; it does not raise."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    payload = ModelResponseChunk(role=Role.TOOL, content=[_text_part('x')])

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(payload)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=payload)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_send_partial_chunk_output_is_not_the_progress() -> None:
    """chunk.output is not the Progress they sent; the Progress is on the part."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(progress)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    sent_chunks = [
        chunk
        for chunk in chunks
        if chunk.role == Role.TOOL and any((p.root.metadata or {}).get('partial') is True for p in chunk.content)
    ]
    assert sent_chunks
    assert sent_chunks[0].output is None
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=progress)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_send_chunk_then_send_partial_keeps_both() -> None:
    """send_chunk text then send_partial Progress: both show up; only the second is stamped."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('svc-00042 is live'))
        ctx.send_partial(progress)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    text_chunks = [
        chunk
        for chunk in chunks
        if chunk.role == Role.TOOL and any(isinstance(p.root, TextPart) for p in chunk.content)
    ]
    assert text_chunks[0].text == 'svc-00042 is live'
    for part in text_chunks[0].content:
        assert not isinstance(part.root, ToolResponsePart)
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=progress)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_send_chunk_still_has_no_tool_name() -> None:
    """A send_chunk next to send_partial still has no name or ref."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        ctx.send_partial(Progress(step='uploading', percent=50))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, _response = await _collect_stream(ai, prompt='go', tools=[deploy])
    text_chunks = [
        chunk
        for chunk in chunks
        if chunk.role == Role.TOOL and any(isinstance(p.root, TextPart) for p in chunk.content)
    ]
    for part in text_chunks[0].content:
        root = part.root
        if isinstance(root, ToolResponsePart):
            raise AssertionError('send_chunk text must not be stamped as a tool response')


@pytest.mark.asyncio
async def test_generate_stream_two_tools_send_partial_parts_have_name_and_ref() -> None:
    """Two tools send_partial; each part has that tool's name and ref."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    a_val = Progress(step='alpha', percent=1)
    b_val = Progress(step='beta', percent=2)

    @ai.tool(name='alpha')
    async def alpha(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(a_val)
        return 'a'

    @ai.tool(name='beta')
    async def beta(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(b_val)
        return 'b'

    pm.responses = [_model_calls(('alpha', 'a1'), ('beta', 'b1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[alpha, beta])
    sent = _partials(chunks)
    by_name = {p.tool_response.name: p for p in sent}
    _assert_stamped(by_name['alpha'], name='alpha', ref='a1', output=a_val)
    _assert_stamped(by_name['beta'], name='beta', ref='b1', output=b_val)
    assert set(_tool_outputs(response.messages)) == {'a', 'b'}


@pytest.mark.asyncio
async def test_generate_stream_two_tools_send_partial_can_tell_them_apart() -> None:
    """Two tools send_partial; you can tell which progress is which from name and ref."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='alpha')
    async def alpha(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial({'who': 'alpha'})
        return 'a'

    @ai.tool(name='beta')
    async def beta(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial({'who': 'beta'})
        return 'b'

    pm.responses = [_model_calls(('alpha', 'a1'), ('beta', 'b1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[alpha, beta])
    sent = _partials(chunks)
    named = {(p.tool_response.name, p.tool_response.ref) for p in sent}
    assert named == {('alpha', 'a1'), ('beta', 'b1')}
    by_name = {p.tool_response.name: p.tool_response.output for p in sent}
    assert by_name['alpha'] == {'who': 'alpha'}
    assert by_name['beta'] == {'who': 'beta'}
    assert set(_tool_outputs(response.messages)) == {'a', 'b'}


@pytest.mark.asyncio
async def test_generate_send_partial_does_not_appear_on_the_stream() -> None:
    """await generate() has no stream; send_partial is a no-op and messages have the return."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(Progress(step='uploading', percent=50))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    response = await ai.generate(prompt='go', tools=[deploy])
    assert not _has_partial_part(response.messages)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_tool_stream_send_partial_does_not_yield() -> None:
    """tool.stream() does not yield the Progress; only send_chunk Parts if any."""
    ai = Genkit()
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        ctx.send_partial(progress)
        return 'done'

    stream = deploy.action().stream({})
    got: list[object] = []
    async for chunk in stream.stream:
        got.append(chunk)
    assert len(got) == 1
    assert isinstance(got[0], Part)
    assert got[0].root.text == 'live'
    assert (await stream.response).output == 'done'


@pytest.mark.asyncio
async def test_tool_run_send_partial_without_a_stream_is_noop() -> None:
    """deploy.run() with no listener: send_partial does not error; return is unchanged."""
    ai = Genkit()

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(Progress(step='uploading', percent=50))
        return 'done'

    assert (await deploy({})).output == 'done'


@pytest.mark.asyncio
async def test_generate_stream_send_partial_progress_is_not_in_response_messages() -> None:
    """No message in the finished response contains the Progress or a partial part."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(progress)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    _chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    assert not _has_partial_part(response.messages)
    assert progress not in _tool_outputs(response.messages)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_return_tool_requests_never_runs_send_partial() -> None:
    """return_tool_requests=True does not run the tool, so send_partial never fires."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    ran = {'n': 0}

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ran['n'] += 1
        ctx.send_partial(Progress(step='uploading', percent=50))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1'))]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], return_tool_requests=True)
    assert ran['n'] == 0
    assert _partials(chunks) == []
    assert response.message is not None
    assert response.message.tool_requests


@pytest.mark.asyncio
async def test_generate_stream_recipe_schema_does_not_apply_to_send_partial() -> None:
    """output_schema=Recipe does not turn a send_partial chunk into a Recipe."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)
    recipe_json = '{"title": "Pie", "steps": ["bake"]}'

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(progress)
        return 'done'

    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[_text_part(recipe_json)])]]
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    _text_part(recipe_json),
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='deploy', ref='r1', input={}))),
                ],
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[_text_part(recipe_json)]),
        ),
    ]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], output_schema=Recipe)
    sent_chunks = [
        chunk
        for chunk in chunks
        if chunk.role == Role.TOOL and any((p.root.metadata or {}).get('partial') is True for p in chunk.content)
    ]
    assert sent_chunks
    assert not isinstance(sent_chunks[0].output, Recipe)
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=progress)
    assert not _has_partial_part(response.messages)


@pytest.mark.asyncio
async def test_generate_stream_model_chunks_are_still_recipe_next_to_send_partial() -> None:
    """Model-role chunks are still Recipe beside a send_partial."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    recipe_json = '{"title": "Pie", "steps": ["bake"]}'

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(Progress(step='uploading', percent=50))
        return 'done'

    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[_text_part(recipe_json)])]]
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    _text_part(recipe_json),
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='deploy', ref='r1', input={}))),
                ],
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[_text_part(recipe_json)]),
        ),
    ]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], output_schema=Recipe)
    model_chunks = [c for c in chunks if c.role == Role.MODEL]
    assert model_chunks
    assert isinstance(model_chunks[0].output, Recipe)
    assert model_chunks[0].output.title == 'Pie'
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_interrupt_after_send_partial_streamed_but_not_in_messages() -> None:
    """send_partial then Interrupt: the progress was streamed; messages do not keep it."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(progress)
        raise Interrupt({'hold': True})

    pm.responses = [_model_calls(('deploy', 'r1'))]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=progress)
    assert response.finish_reason == FinishReason.INTERRUPTED
    assert not _has_partial_part(response.messages)


@pytest.mark.asyncio
async def test_generate_stream_resume_does_not_replay_send_partial() -> None:
    """Resume after interrupt does not replay the first run's Progress."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    first = Progress(step='first', percent=10)
    runs = {'n': 0}

    @ai.tool(name='pay')
    async def pay(inp: dict, ctx: ToolRunContext) -> str:
        runs['n'] += 1
        if not inp.get('ok'):
            ctx.send_partial(first)
            raise Interrupt({'hold': True})
        return 'paid'

    pm.responses = [_model_calls(('pay', 'p1')), _model_says('final')]
    first_chunks, first_resp = await _collect_stream(ai, prompt='go', tools=[pay])
    _assert_stamped(_partials(first_chunks)[0], name='pay', ref='p1', output=first)
    restart = restart_tool(interrupt=first_resp.interrupts[0], replace_input={'ok': True})
    second_chunks, second = await _collect_stream(
        ai,
        messages=list(first_resp.messages),
        tools=[pay],
        resume_restart=restart,
    )
    assert all(p.tool_response.output != first for p in _partials(second_chunks))
    assert not _has_partial_part(second.messages)
    assert runs['n'] == 2


@pytest.mark.asyncio
async def test_generate_stream_restarted_tool_send_partial_is_a_new_chunk() -> None:
    """A restarted tool's send_partial is new Progress, not a replay of the first run."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    first = Progress(step='first', percent=10)
    second = Progress(step='second', percent=90)

    @ai.tool(name='pay')
    async def pay(inp: dict, ctx: ToolRunContext) -> str:
        if not inp.get('ok'):
            ctx.send_partial(first)
            raise Interrupt({'hold': True})
        ctx.send_partial(second)
        return 'paid'

    pm.responses = [_model_calls(('pay', 'p1')), _model_says('final')]
    _first_chunks, first_resp = await _collect_stream(ai, prompt='go', tools=[pay])
    restart = restart_tool(interrupt=first_resp.interrupts[0], replace_input={'ok': True})
    second_chunks, second_resp = await _collect_stream(
        ai,
        messages=list(first_resp.messages),
        tools=[pay],
        resume_restart=restart,
    )
    sent = _partials(second_chunks)
    assert any(p.tool_response.output == second for p in sent)
    assert all(p.tool_response.output != first for p in sent)
    assert not _has_partial_part(second_resp.messages)
    assert _tool_outputs(second_resp.messages)[-1] == 'paid'


@pytest.mark.asyncio
async def test_generate_stream_abort_after_send_partial_keeps_progress_off_messages() -> None:
    """Abort after send_partial: the progress was streamed; messages do not contain it."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(progress)
        ctx.abort_signal.set()
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks: list[ModelResponseChunk] = []
    stream = ai.generate_stream(prompt='go', tools=[deploy])
    with pytest.raises(GenkitError) as ei:
        async for chunk in stream.stream:
            chunks.append(chunk)
        await stream.response
    assert ei.value.status == 'ABORTED'
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=progress)


@pytest.mark.asyncio
async def test_generate_stream_max_turns_after_send_partial_keeps_progress_off_messages() -> None:
    """Hitting max turns after send_partial keeps the Progress off messages."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='lookup')
    async def lookup(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(progress)
        return '72F'

    pm.responses = [_model_calls(('lookup', 'r1')), _model_calls(('lookup', 'r2'))]
    chunks, response = await _collect_stream(ai, prompt='keep going', tools=[lookup], max_turns=1)
    _assert_stamped(_partials(chunks)[0], name='lookup', ref='r1', output=progress)
    assert not _has_partial_part(response.messages)
    assert _tool_outputs(response.messages) == ['72F']


@pytest.mark.asyncio
async def test_define_prompt_stream_send_partial_is_stamped_progress() -> None:
    """define_prompt(...).stream() + send_partial is the same stamped tool-role part."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(progress)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    prompt = ai.define_prompt(prompt='go', tools=[deploy])
    stream = prompt.stream()
    chunks: list[ModelResponseChunk] = []
    async for chunk in stream.stream:
        chunks.append(chunk)
    response = await stream.response
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=progress)
    assert not _has_partial_part(response.messages)


@pytest.mark.asyncio
async def test_agent_send_stream_includes_tool_send_partial() -> None:
    """Agent send_stream has the same stamped generate chunk; walk the parts."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(progress)
        return 'done'

    ai.define_prompt(name='shipAgent', model='programmableModel', tools=[deploy])
    agent = ai.define_prompt_agent(name='shipAgent')
    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]

    turn = agent.chat().send_stream('go')
    found: list[ToolResponsePart] = []
    async for item in turn.stream:
        raw = item.raw
        model_chunk = raw.model_chunk if raw is not None else None
        if model_chunk is None:
            continue
        for part in model_chunk.content:
            root = part.root
            if isinstance(root, ToolResponsePart) and (root.metadata or {}).get('partial') is True:
                found.append(root)
    await turn.response
    _assert_stamped(found[0], name='deploy', ref='r1', output=progress)


@pytest.mark.asyncio
async def test_agent_chat_messages_keep_return_not_mid_tool_updates() -> None:
    """After send, chat.messages has the return; the next turn does not see mid-tool parts."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('svc-00042 is live'))
        ctx.send_partial(progress)
        return 'done'

    ai.define_prompt(name='shipAgent', model='programmableModel', tools=[deploy])
    agent = ai.define_prompt_agent(name='shipAgent')
    pm.responses = [
        _model_calls(('deploy', 'r1')),
        _model_says('ok'),
        _model_says('again-ok'),
    ]

    chat = agent.chat()
    await chat.send('go')
    assert _tool_outputs(chat.messages) == ['done']
    assert not _has_partial_part(chat.messages)
    assert 'svc-00042 is live' not in _message_texts(chat.messages)

    turn = chat.send_stream('again')
    async for _item in turn.stream:
        pass
    await turn.response
    req = pm.last_request
    assert req is not None
    assert 'svc-00042 is live' not in _message_texts(req.messages)
    assert not _has_partial_part(req.messages)
    assert _tool_outputs(req.messages) == ['done']


@pytest.mark.asyncio
async def test_agent_next_turn_drops_send_chunk_tool_response_part() -> None:
    """send_chunk(ToolResponsePart) is streamed; chat.messages and the next turn keep only the return."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    stamped = Part(
        root=ToolResponsePart(tool_response=ToolResponse(name='deploy', ref='mine', output='uploading')),
    )

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(stamped)
        return 'done'

    ai.define_prompt(name='shipAgent', model='programmableModel', tools=[deploy])
    agent = ai.define_prompt_agent(name='shipAgent')
    pm.responses = [
        _model_calls(('deploy', 'r1')),
        _model_says('ok'),
        _model_says('again-ok'),
    ]

    chat = agent.chat()
    await chat.send('go')
    assert _tool_outputs(chat.messages) == ['done']

    turn = chat.send_stream('again')
    async for _item in turn.stream:
        pass
    await turn.response
    req = pm.last_request
    assert req is not None
    assert _tool_outputs(req.messages) == ['done']
    assert _tool_outputs(chat.messages) == ['done']


@pytest.mark.asyncio
async def test_agent_next_turn_drops_send_chunk_tool_request_part() -> None:
    """send_chunk(ToolRequestPart) is streamed; the next turn does not replay it."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    sneak = Part(root=ToolRequestPart(tool_request=ToolRequest(name='sneak', ref='x', input={})))

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(sneak)
        return 'done'

    ai.define_prompt(name='shipAgent', model='programmableModel', tools=[deploy])
    agent = ai.define_prompt_agent(name='shipAgent')
    pm.responses = [
        _model_calls(('deploy', 'r1')),
        _model_says('ok'),
        _model_says('again-ok'),
    ]

    chat = agent.chat()
    await chat.send('go')
    sneak_names = [
        part.root.tool_request.name
        for msg in chat.messages
        for part in msg.content
        if isinstance(part.root, ToolRequestPart)
    ]
    assert 'sneak' not in sneak_names

    turn = chat.send_stream('again')
    async for _item in turn.stream:
        pass
    await turn.response
    req = pm.last_request
    assert req is not None
    req_names = [
        part.root.tool_request.name
        for msg in req.messages
        for part in msg.content
        if isinstance(part.root, ToolRequestPart)
    ]
    assert 'sneak' not in req_names


def test_agent_history_still_drops_send_chunk_tool_response_after_wire_hop() -> None:
    """After dump/validate, send_chunk(ToolResponsePart) still stays off history."""
    mid = ModelResponseChunk(
        role=Role.TOOL,
        content=[
            Part(root=ToolResponsePart(tool_response=ToolResponse(name='deploy', ref='mine', output='uploading'))),
        ],
    )
    mark_stream_only(mid)
    dumped = AgentStreamChunk.model_validate(
        AgentStreamChunk(model_chunk=mid).model_dump(by_alias=True, exclude_none=True)
    )
    acc = StreamedMessageAccumulator()
    acc.add(dumped)
    assert _tool_outputs(acc.messages()) == []


@pytest.mark.asyncio
async def test_generate_stream_json_send_chunk_then_send_partial_output_stays_empty() -> None:
    """After a json-looking send_chunk, send_partial and the return have chunk.output None."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    leftover = '{"title": "Pie", "steps": ["mix"]}'
    progress = Progress(step='uploading', percent=50)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part(leftover))
        ctx.send_partial(progress)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], output_schema=Recipe)
    for chunk in chunks:
        if chunk.role == Role.TOOL:
            assert chunk.output is None
    _assert_stamped(_partials(chunks)[0], name='deploy', ref='r1', output=progress)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_error_during_send_partial_tool_still_returns() -> None:
    """A listener error on the send_partial chunk is dropped; the return still lands."""

    class BoomOnPartial(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn,
        ) -> ModelResponse:
            downstream = ctx.on_chunk

            def handler(chunk: ModelResponseChunk) -> None:
                if any((p.root.metadata or {}).get('partial') is True for p in chunk.content):
                    raise RuntimeError('sink')
                if downstream is not None:
                    downstream(chunk)

            previous = ctx.replace_on_chunk(handler)
            try:
                return await next_fn(params, ctx)
            finally:
                ctx.replace_on_chunk(previous)

    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(Progress(step='uploading', percent=50))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    _chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], use=[BoomOnPartial()])
    assert _tool_outputs(response.messages) == ['done']
    assert response.finish_reason == FinishReason.STOP


@pytest.mark.asyncio
async def test_generate_stream_error_on_final_tool_response_fails_generate() -> None:
    """A listener error on the return tool-role chunk fails generate."""

    class BoomOnFinal(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn,
        ) -> ModelResponse:
            downstream = ctx.on_chunk

            def handler(chunk: ModelResponseChunk) -> None:
                if chunk.role == Role.TOOL and any(
                    isinstance(p.root, ToolResponsePart) and (p.root.metadata or {}).get('partial') is not True
                    for p in chunk.content
                ):
                    raise RuntimeError('final sink')
                if downstream is not None:
                    downstream(chunk)

            previous = ctx.replace_on_chunk(handler)
            try:
                return await next_fn(params, ctx)
            finally:
                ctx.replace_on_chunk(previous)

    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_partial(Progress(step='uploading', percent=50))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    stream = ai.generate_stream(prompt='go', tools=[deploy], use=[BoomOnFinal()])
    with pytest.raises(RuntimeError, match='final sink'):
        async for _chunk in stream.stream:
            pass
    with pytest.raises(RuntimeError, match='final sink'):
        await stream.response
