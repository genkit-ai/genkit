#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Pins ``ToolRunContext.send_chunk`` during ``generate_stream``.

``send_chunk(Part | list[Part])`` shows up as a tool-role
``ModelResponseChunk`` on ``generate_stream``. History only keeps the
tool's return. ``tool.stream()`` still yields the Part you sent.
"""

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, restart_tool
from genkit._ai._testing import define_programmable_model
from genkit._ai._tools import Interrupt, ToolRunContext, normalize_send_chunk_parts
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._middleware import BaseMiddleware, GenerateHookParams, GenerateMiddlewareContext
from genkit._core._typing import (
    FinishReason,
    Media,
    MediaPart,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)


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


def _send_chunks(chunks: Sequence[ModelResponseChunk]) -> list[ModelResponseChunk]:
    return [
        chunk
        for chunk in chunks
        if chunk.role == Role.TOOL and any(isinstance(p.root, TextPart) for p in chunk.content)
    ]


async def _collect_stream(ai: Genkit, **kwargs: object) -> tuple[list[ModelResponseChunk], ModelResponse]:
    stream = ai.generate_stream(**kwargs)  # type: ignore[arg-type]
    chunks: list[ModelResponseChunk] = []
    async for chunk in stream.stream:
        chunks.append(chunk)
    return chunks, await stream.response


@pytest.mark.asyncio
async def test_generate_stream_send_chunk_one_text_part_is_tool_role_text() -> None:
    """Tool send_chunk of one text part is a tool-role chunk with that text."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('svc-00042 is live'))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])

    sent = _send_chunks(chunks)
    assert len(sent) == 1
    assert sent[0].role == Role.TOOL
    assert sent[0].text == 'svc-00042 is live'
    assert 'svc-00042 is live' not in _message_texts(response.messages)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_send_chunk_list_keeps_every_part() -> None:
    """send_chunk([text, media]) is one tool-role chunk with both parts."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    media = Part(root=MediaPart(media=Media(url='data:image/png;base64,QQ==', content_type='image/png')))

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk([_text_part('chart:'), media])
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])

    sent = _send_chunks(chunks)
    assert len(sent) == 1
    assert sent[0].text == 'chart:'
    assert len(sent[0].content) == 2
    assert isinstance(sent[0].content[1].root, MediaPart)
    assert 'chart:' not in _message_texts(response.messages)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_return_still_streams_final_tool_response() -> None:
    """After send_chunk, the return still streams as a later tool-role response."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])

    finals = [
        chunk
        for chunk in chunks
        if chunk.role == Role.TOOL and any(isinstance(p.root, ToolResponsePart) for p in chunk.content)
    ]
    assert finals
    assert finals[0].text == ''
    assert _tool_outputs(response.messages) == ['done']
    assert response.messages[-1].role in {Role.MODEL, Role.TOOL}


@pytest.mark.asyncio
async def test_generate_stream_after_a_finished_tool_round_send_chunk_still_tool_role() -> None:
    """A later tool's send_chunk is still role=tool; the first closed round stays."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='weather')
    async def weather(_req: dict) -> str:
        return '72F'

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        return 'done'

    pm.responses = [
        _model_calls(('weather', 'w1')),
        _model_calls(('deploy', 'd1')),
        _model_says('ok'),
    ]
    chunks, response = await _collect_stream(ai, prompt='do both', tools=[weather, deploy])

    sent = _send_chunks(chunks)
    assert sent[0].role == Role.TOOL
    assert sent[0].text == 'live'
    assert [m.role for m in response.messages[:3]] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tool_outputs(response.messages)[0] == '72F'
    assert 'live' not in _message_texts(response.messages)


@pytest.mark.asyncio
async def test_tool_stream_send_chunk_one_part_is_the_part() -> None:
    """tool.stream() yields the Part you sent, not a generate ModelResponseChunk."""
    ai = Genkit()

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
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
async def test_generate_stream_send_chunk_media_only_has_empty_text() -> None:
    """A media-only send_chunk has empty .text and keeps the media part."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    media = Part(root=MediaPart(media=Media(url='https://example.com/a.png', content_type='image/png')))

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(media)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, _response = await _collect_stream(ai, prompt='go', tools=[deploy])

    tool_chunks = [c for c in chunks if c.role == Role.TOOL]
    sent = [c for c in tool_chunks if any(isinstance(p.root, MediaPart) for p in c.content)]
    assert sent[0].text == ''
    assert isinstance(sent[0].content[0].root, MediaPart)


@pytest.mark.asyncio
async def test_generate_stream_send_chunk_tool_response_part_is_forwarded() -> None:
    """A ToolResponsePart is a legal send_chunk part and shows up as-is."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    stamped = Part(
        root=ToolResponsePart(tool_response=ToolResponse(name='deploy', ref='mine', output='uploading')),
    )

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(stamped)
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])

    sent = [c for c in chunks if c.role == Role.TOOL and any(isinstance(p.root, ToolResponsePart) for p in c.content)]
    first = sent[0].content[0].root
    assert isinstance(first, ToolResponsePart)
    assert first.tool_response.name == 'deploy'
    assert first.tool_response.ref == 'mine'
    assert first.tool_response.output == 'uploading'
    assert (sent[0].custom or {}).get('keepInHistory') is False
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_two_send_chunks_accumulated_text_is_hello() -> None:
    """Two text send_chunks concatenate on .accumulated_text."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('hel'))
        ctx.send_chunk(_text_part('lo'))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])

    sent = _send_chunks(chunks)
    assert sent[0].text == 'hel'
    assert sent[1].text == 'lo'
    assert sent[1].accumulated_text == 'hello'
    assert 'hel' not in _message_texts(response.messages)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_two_tools_send_chunk_chunks_have_no_tool_name() -> None:
    """Two tools send_chunk text; generate does not stamp name or ref."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='alpha')
    async def alpha(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('from-alpha'))
        return 'a'

    @ai.tool(name='beta')
    async def beta(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('from-beta'))
        return 'b'

    pm.responses = [_model_calls(('alpha', 'a1'), ('beta', 'b1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[alpha, beta])

    sent = _send_chunks(chunks)
    assert {c.text for c in sent} == {'from-alpha', 'from-beta'}
    for chunk in sent:
        for part in chunk.content:
            root = part.root
            if isinstance(root, ToolResponsePart):
                raise AssertionError('send_chunk text must not be stamped as a tool response')
    assert _tool_outputs(response.messages) == ['a', 'b'] or set(_tool_outputs(response.messages)) == {'a', 'b'}


@pytest.mark.asyncio
async def test_generate_stream_two_tools_send_chunk_accumulated_text_includes_both() -> None:
    """The later tool-role send_chunk's accumulated_text includes both tools."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='alpha')
    async def alpha(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('from-alpha'))
        return 'a'

    @ai.tool(name='beta')
    async def beta(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('from-beta'))
        return 'b'

    pm.responses = [_model_calls(('alpha', 'a1'), ('beta', 'b1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[alpha, beta])

    sent = _send_chunks(chunks)
    later = sent[-1]
    assert 'from-alpha' in later.accumulated_text
    assert 'from-beta' in later.accumulated_text
    assert set(_tool_outputs(response.messages)) == {'a', 'b'}


@pytest.mark.asyncio
async def test_generate_send_chunk_does_not_appear_on_the_stream() -> None:
    """await generate() has no stream; send_chunk is a no-op and messages have the return."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    response = await ai.generate(prompt='go', tools=[deploy])
    assert 'live' not in _message_texts(response.messages)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_tool_run_send_chunk_without_a_stream_is_noop() -> None:
    """deploy.run() with no on_chunk: send_chunk does not error; return is unchanged."""
    ai = Genkit()

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        return 'done'

    assert (await deploy({})).output == 'done'


@pytest.mark.asyncio
async def test_generate_stream_send_chunk_text_is_not_in_response_messages() -> None:
    """No message in the finished response contains the send_chunk string."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('not-in-history'))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    _chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    assert 'not-in-history' not in _message_texts(response.messages)
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_return_tool_requests_never_runs_send_chunk() -> None:
    """return_tool_requests=True does not run the tool, so send_chunk never fires."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    ran = {'n': 0}

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ran['n'] += 1
        ctx.send_chunk(_text_part('live'))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1'))]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], return_tool_requests=True)
    assert ran['n'] == 0
    assert _send_chunks(chunks) == []
    assert response.message is not None
    assert response.message.tool_requests


@pytest.mark.asyncio
async def test_generate_stream_recipe_schema_does_not_apply_to_send_chunk() -> None:
    """output_schema=Recipe does not turn a tool send_chunk into a Recipe."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        return 'done'

    recipe_json = '{"title": "Pie", "steps": ["bake"]}'
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

    sent = _send_chunks(chunks)
    assert sent
    assert sent[0].output is None
    assert 'live' not in _message_texts(response.messages)


@pytest.mark.asyncio
async def test_generate_stream_model_chunks_are_still_recipe_next_to_send_chunk() -> None:
    """Model-role chunks are still leftover Recipe beside a tool send_chunk."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        return 'done'

    recipe_json = '{"title": "Pie", "steps": ["bake"]}'
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
async def test_generate_stream_json_send_chunk_does_not_leave_output_on_later_tool_chunks() -> None:
    """A json-looking send_chunk does not become leftover chunk.output on later tool-role chunks."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    leftover = '{"title": "Pie", "steps": ["mix"]}'
    recipe_json = '{"title": "Pie", "steps": ["bake"]}'

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part(leftover))
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
    tool_chunks = [chunk for chunk in chunks if chunk.role == Role.TOOL]
    assert tool_chunks
    for chunk in tool_chunks:
        assert chunk.output is None
    model_chunks = [chunk for chunk in chunks if chunk.role == Role.MODEL]
    assert isinstance(model_chunks[-1].output, Recipe)
    assert model_chunks[-1].output.steps == ['bake']
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_interrupt_after_send_chunk_streamed_but_not_in_messages() -> None:
    """send_chunk then Interrupt: the chunk was streamed; messages do not keep it."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        raise Interrupt({'hold': True})

    pm.responses = [_model_calls(('deploy', 'r1'))]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])

    assert _send_chunks(chunks)[0].text == 'live'
    assert response.finish_reason == FinishReason.INTERRUPTED
    assert 'live' not in _message_texts(response.messages)


@pytest.mark.asyncio
async def test_generate_stream_resume_does_not_replay_send_chunk() -> None:
    """Resume after interrupt does not replay the first run's send_chunk text."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    runs = {'n': 0}

    @ai.tool(name='pay')
    async def pay(inp: dict, ctx: ToolRunContext) -> str:
        runs['n'] += 1
        if not inp.get('ok'):
            ctx.send_chunk(_text_part('first-run'))
            raise Interrupt({'hold': True})
        return 'paid'

    pm.responses = [_model_calls(('pay', 'p1')), _model_says('final')]
    first_chunks, first = await _collect_stream(ai, prompt='go', tools=[pay])
    assert _send_chunks(first_chunks)[0].text == 'first-run'

    restart = restart_tool(interrupt=first.interrupts[0], replace_input={'ok': True})
    second_chunks, second = await _collect_stream(
        ai,
        messages=list(first.messages),
        tools=[pay],
        resume_restart=restart,
    )
    assert all(c.text != 'first-run' for c in _send_chunks(second_chunks))
    assert 'first-run' not in _message_texts(second.messages)
    assert runs['n'] == 2


@pytest.mark.asyncio
async def test_generate_stream_restarted_tool_send_chunk_is_a_new_chunk() -> None:
    """A restarted tool's send_chunk is new text, not a replay of the first run."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='pay')
    async def pay(inp: dict, ctx: ToolRunContext) -> str:
        if not inp.get('ok'):
            ctx.send_chunk(_text_part('first-run'))
            raise Interrupt({'hold': True})
        ctx.send_chunk(_text_part('second-run'))
        return 'paid'

    pm.responses = [_model_calls(('pay', 'p1')), _model_says('final')]
    _first_chunks, first = await _collect_stream(ai, prompt='go', tools=[pay])
    restart = restart_tool(interrupt=first.interrupts[0], replace_input={'ok': True})
    second_chunks, second = await _collect_stream(
        ai,
        messages=list(first.messages),
        tools=[pay],
        resume_restart=restart,
    )
    sent = _send_chunks(second_chunks)
    assert any(c.text == 'second-run' for c in sent)
    assert all(c.text != 'first-run' for c in sent)
    assert 'second-run' not in _message_texts(second.messages)
    assert _tool_outputs(second.messages)[-1] == 'paid'


@pytest.mark.asyncio
async def test_generate_stream_abort_after_send_chunk_keeps_chunk_off_messages() -> None:
    """Abort after send_chunk: the chunk was streamed; messages do not contain it."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
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
    assert _send_chunks(chunks)[0].text == 'live'


@pytest.mark.asyncio
async def test_generate_stream_max_turns_after_send_chunk_keeps_chunk_off_messages() -> None:
    """Hitting max turns after send_chunk keeps the chunk off messages."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='lookup')
    async def lookup(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        return '72F'

    pm.responses = [_model_calls(('lookup', 'r1')), _model_calls(('lookup', 'r2'))]
    chunks, response = await _collect_stream(ai, prompt='keep going', tools=[lookup], max_turns=1)
    assert _send_chunks(chunks)[0].text == 'live'
    assert 'live' not in _message_texts(response.messages)
    assert _tool_outputs(response.messages) == ['72F']


@pytest.mark.asyncio
async def test_define_prompt_stream_send_chunk_is_tool_role_text() -> None:
    """define_prompt(...).stream() + send_chunk is the same tool-role text."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    prompt = ai.define_prompt(prompt='go', tools=[deploy])
    stream = prompt.stream()
    chunks: list[ModelResponseChunk] = []
    async for chunk in stream.stream:
        chunks.append(chunk)
    response = await stream.response
    assert _send_chunks(chunks)[0].text == 'live'
    assert 'live' not in _message_texts(response.messages)


@pytest.mark.asyncio
async def test_agent_send_stream_includes_tool_send_chunk() -> None:
    """Agent send_stream surfaces the generate tool-role send_chunk as that same chunk."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk(_text_part('live'))
        return 'done'

    ai.define_prompt(name='shipAgent', model='programmableModel', tools=[deploy])
    agent = ai.define_prompt_agent(name='shipAgent')
    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]

    turn = agent.chat().send_stream('go')
    texts: list[str] = []
    async for item in turn.stream:
        raw = item.raw
        model_chunk = raw.model_chunk if raw is not None else None
        if model_chunk is not None and model_chunk.role == Role.TOOL and item.text:
            texts.append(item.text)
    await turn.response
    assert 'live' in texts


@pytest.mark.asyncio
async def test_send_chunk_dict_raises_invalid_argument() -> None:
    """send_chunk({'percent': 50}) raises INVALID_ARGUMENT."""
    ctx = ToolRunContext(ActionRunContext())
    with pytest.raises(GenkitError) as ei:
        ctx.send_chunk({'percent': 50})  # type: ignore[arg-type]
    assert ei.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_send_chunk_model_response_chunk_raises_invalid_argument() -> None:
    """Passing a ModelResponseChunk to send_chunk raises INVALID_ARGUMENT."""
    ctx = ToolRunContext(ActionRunContext())
    with pytest.raises(GenkitError) as ei:
        ctx.send_chunk(ModelResponseChunk(role=Role.TOOL, content=[_text_part('x')]))  # type: ignore[arg-type]
    assert ei.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_send_chunk_none_raises_invalid_argument() -> None:
    """send_chunk(None) raises INVALID_ARGUMENT."""
    ctx = ToolRunContext(ActionRunContext())
    with pytest.raises(GenkitError) as ei:
        ctx.send_chunk(None)  # type: ignore[arg-type]
    assert ei.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_send_chunk_empty_list_is_a_tool_role_chunk_with_no_parts() -> None:
    """send_chunk([]) is a tool-role chunk with empty content."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='deploy')
    async def deploy(_req: dict, ctx: ToolRunContext) -> str:
        ctx.send_chunk([])
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy])
    empty = [c for c in chunks if c.role == Role.TOOL and c.content == []]
    assert empty
    assert empty[0].text == ''
    assert _tool_outputs(response.messages) == ['done']


@pytest.mark.asyncio
async def test_generate_stream_error_during_send_chunk_tool_still_returns() -> None:
    """A listener error on the send_chunk chunk is dropped; the return still lands."""

    class BoomOnSend(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn,
        ) -> ModelResponse:
            downstream = ctx.on_chunk

            def handler(chunk: ModelResponseChunk) -> None:
                if chunk.role == Role.TOOL and chunk.text == 'live':
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
        ctx.send_chunk(_text_part('live'))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    _chunks, response = await _collect_stream(ai, prompt='go', tools=[deploy], use=[BoomOnSend()])
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
                if chunk.role == Role.TOOL and any(isinstance(p.root, ToolResponsePart) for p in chunk.content):
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
        ctx.send_chunk(_text_part('live'))
        return 'done'

    pm.responses = [_model_calls(('deploy', 'r1')), _model_says('ok')]
    stream = ai.generate_stream(prompt='go', tools=[deploy], use=[BoomOnFinal()])
    with pytest.raises(RuntimeError, match='final sink'):
        async for _chunk in stream.stream:
            pass
        await stream.response
    with pytest.raises(RuntimeError, match='final sink'):
        await stream.response


def test_normalize_send_chunk_parts_accepts_part_and_list() -> None:
    """normalize_send_chunk_parts turns one Part into a one-item list."""
    part = _text_part('x')
    assert normalize_send_chunk_parts(part) == [part]
    assert normalize_send_chunk_parts([part, part]) == [part, part]
