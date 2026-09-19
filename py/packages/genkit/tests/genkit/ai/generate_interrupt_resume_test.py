# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for interrupt, resume, and restart behavior in ``generate_action``."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from genkit import Genkit, Message, MiddlewareRef, ModelResponse, Part
from genkit._ai._generate import generate_action
from genkit._ai._testing import define_programmable_model
from genkit._ai._tools import (
    Interrupt,
    MultipartToolResponse,
    ToolRunContext,
    respond_to_interrupt,
    response,
    restart_tool,
)
from genkit._core._error import RuntimeErrorReason
from genkit._core._model import GenerateActionOptions, Resume
from genkit._core._typing import (
    FinishReason,
    Media,
    Role,
)
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ToolHookParams


def _wire(messages: list[Message]) -> list[dict[str, Any]]:
    """Messages as JSON-shaped dicts (``model_dump`` with aliases) for comparing to expected wire."""
    return [m.model_dump(mode='json', exclude_none=True, by_alias=True) for m in messages]


def _gen_opts(
    ai: Genkit,
    *,
    tools: list[str],
    messages: list[Message],
    resume: Resume | None = None,
    use: list[MiddlewareRef] | None = None,
) -> GenerateActionOptions:
    return GenerateActionOptions(
        model='programmableModel',
        messages=messages,
        tools=tools,
        resume=resume,
        use=use,
    )


def _png() -> Part:
    return Part.from_media('data:image/png;base64,abc', content_type='image/png')


WIRE_PNG = {'media': {'contentType': 'image/png', 'url': 'data:image/png;base64,abc'}}


@pytest.mark.asyncio
async def test_normal_two_arg_tools_see_no_resume_context() -> None:
    """Two tools in one batch, no interrupt: ``ToolRunContext`` should stay empty of resume fields.

    The model asks for both tools in one turn; each tool records whether it thinks it's a resume
    (it shouldn't), and we compare the whole conversation to the expected wire.
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    seen: list[tuple[bool, object | None, object | None]] = []

    @ai.tool(name='u1')
    async def u1(_: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        seen.append((ctx.is_resumed(), ctx.resumed_metadata, ctx.original_input))
        return 'a'

    @ai.tool(name='u2')
    async def u2(_: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        seen.append((ctx.is_resumed(), ctx.resumed_metadata, ctx.original_input))
        return 'b'

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'text': 'go'},
                    {'toolRequest': {'ref': '1', 'name': 'u1', 'input': {}}},
                    {'toolRequest': {'ref': '2', 'name': 'u2', 'input': {}}},
                ],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'done'}]}),
        )
    )

    r = await generate_action(
        ai.registry,
        _gen_opts(
            ai, tools=['u1', 'u2'], messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})]
        ),
    )
    assert seen == [(False, None, None), (False, None, None)]
    assert _wire(r.messages) == [
        {
            'role': 'user',
            'content': [{'text': 'hi'}],
        },
        {
            'role': 'model',
            'content': [
                {'text': 'go'},
                {'toolRequest': {'ref': '1', 'name': 'u1', 'input': {}}},
                {'toolRequest': {'ref': '2', 'name': 'u2', 'input': {}}},
            ],
        },
        {
            'role': 'tool',
            'content': [
                {'toolResponse': {'ref': '1', 'name': 'u1', 'output': 'a'}},
                {'toolResponse': {'ref': '2', 'name': 'u2', 'output': 'b'}},
            ],
        },
        {
            'role': 'model',
            'content': [{'text': 'done'}],
        },
    ]


@pytest.mark.asyncio
async def test_interrupt_wires_trp_metadata_interrupt_and_stops() -> None:
    """When the tool raises ``Interrupt``, the interrupt payload lands on the TRP metadata, finish is
    ``INTERRUPTED``, and we never get a ``role=tool`` row yet—only user + model in the history.
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='intr')
    async def intr(_: dict) -> str:  # noqa: ARG001
        raise Interrupt({'reason': 'x'})

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'text': 'call'},
                    {'toolRequest': {'ref': 'r1', 'name': 'intr', 'input': {}}},
                ],
            }),
        )
    )

    r = await generate_action(
        ai.registry,
        _gen_opts(ai, tools=['intr'], messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})]),
    )
    assert r.finish_reason == FinishReason.INTERRUPTED
    assert _wire(r.messages) == [
        {
            'role': 'user',
            'content': [{'text': 'hi'}],
        },
        {
            'role': 'model',
            'content': [
                {'text': 'call'},
                {
                    'toolRequest': {'ref': 'r1', 'name': 'intr', 'input': {}},
                    'metadata': {'interrupt': {'reason': 'x'}},
                },
            ],
        },
    ]


@pytest.mark.asyncio
async def test_resume_respond_trp_gets_resolved_interrupt_and_tool_trp() -> None:
    """Follow-up generate with ``Resume(respond=[...])``: the stuck TRP picks up ``resolvedInterrupt``,
    the tool reply shows up under ``interruptResponse``, and the model can answer again. Compares
    wire before and after the interrupt.
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='intr')
    async def intr(_: dict) -> str:  # noqa: ARG001
        raise Interrupt({'reason': 'x'})

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'text': 'call'},
                    {'toolRequest': {'ref': 'r1', 'name': 'intr', 'input': {}}},
                ],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'after resume'}]}),
        )
    )

    first = await generate_action(
        ai.registry,
        _gen_opts(ai, tools=['intr'], messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})]),
    )
    assert first.finish_reason == FinishReason.INTERRUPTED
    assert _wire(first.messages) == [
        {
            'role': 'user',
            'content': [{'text': 'hi'}],
        },
        {
            'role': 'model',
            'content': [
                {'text': 'call'},
                {
                    'toolRequest': {'ref': 'r1', 'name': 'intr', 'input': {}},
                    'metadata': {'interrupt': {'reason': 'x'}},
                },
            ],
        },
    ]

    reply = respond_to_interrupt({'bar': 2}, interrupt=first.interrupts[0])

    second = await generate_action(
        ai.registry,
        _gen_opts(ai, tools=['intr'], messages=list(first.messages), resume=Resume(respond=[reply])),
    )

    assert second.finish_reason == FinishReason.STOP
    assert _wire(second.messages) == [
        {
            'role': 'user',
            'content': [{'text': 'hi'}],
        },
        {
            'role': 'model',
            'content': [
                {'text': 'call'},
                {
                    'toolRequest': {'ref': 'r1', 'name': 'intr', 'input': {}},
                    'metadata': {'resolvedInterrupt': {'reason': 'x'}},
                },
            ],
        },
        {
            'role': 'tool',
            'content': [
                {
                    'toolResponse': {'ref': 'r1', 'name': 'intr', 'output': {'bar': 2}},
                    'metadata': {'interruptResponse': True},
                },
            ],
            'metadata': {'resumed': True},
        },
        {
            'role': 'model',
            'content': [{'text': 'after resume'}],
        },
    ]


async def _interrupted_generate() -> tuple[Genkit, ModelResponse]:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='intr')
    async def intr(_: dict) -> str:  # noqa: ARG001
        raise Interrupt({'reason': 'x'})

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'text': 'call'},
                    {'toolRequest': {'ref': 'r1', 'name': 'intr', 'input': {}}},
                ],
            }),
        )
    )
    first = await generate_action(
        ai.registry,
        _gen_opts(ai, tools=['intr'], messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})]),
    )
    assert first.finish_reason == FinishReason.INTERRUPTED
    return ai, first


@pytest.mark.asyncio
async def test_resume_respond_text_part_raises() -> None:
    """resume_respond needs a tool response part."""
    ai, first = await _interrupted_generate()
    with pytest.raises(ValueError, match='tool response'):
        await ai.generate(
            model='programmableModel',
            messages=list(first.messages),
            tools=['intr'],
            resume_respond=Part.from_text('hi'),
        )


@pytest.mark.asyncio
async def test_resume_respond_list_text_part_raises() -> None:
    """resume_respond rejects a list of text parts."""
    ai, first = await _interrupted_generate()
    with pytest.raises(ValueError, match='tool response'):
        await ai.generate(
            model='programmableModel',
            messages=list(first.messages),
            tools=['intr'],
            resume_respond=[Part.from_text('hi')],
        )


@pytest.mark.asyncio
async def test_resume_restart_text_part_raises() -> None:
    """resume_restart needs a tool request part."""
    ai, first = await _interrupted_generate()
    with pytest.raises(ValueError, match='tool request'):
        await ai.generate(
            model='programmableModel',
            messages=list(first.messages),
            tools=['intr'],
            resume_restart=Part.from_text('hi'),
        )


@pytest.mark.asyncio
async def test_tool_either_interrupts_or_returns() -> None:
    """Same tool, two independent generate calls with different inputs.

    First call: ``preapproved=False`` → tool raises Interrupt → finish is INTERRUPTED, no tool row.
    Second call: ``preapproved=True`` → tool returns 42 → finish is STOP, full tool+model rows present.
    Both results are wire-asserted in full.
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='bank_transfer')
    async def bank_transfer(inp: dict) -> int:
        if not inp.get('preapproved'):
            raise Interrupt({'reason': 'awaiting_approval'})
        return 42

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'text': 't'},
                    {
                        'toolRequest': {
                            'ref': 'g',
                            'name': 'bank_transfer',
                            'input': {'preapproved': False},
                        },
                    },
                ],
            }),
        )
    )
    r_fail = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['bank_transfer'],
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})],
        ),
    )
    assert r_fail.finish_reason == FinishReason.INTERRUPTED
    assert _wire(r_fail.messages) == [
        {
            'role': 'user',
            'content': [{'text': 'hi'}],
        },
        {
            'role': 'model',
            'content': [
                {'text': 't'},
                {
                    'toolRequest': {
                        'ref': 'g',
                        'name': 'bank_transfer',
                        'input': {'preapproved': False},
                    },
                    'metadata': {'interrupt': {'reason': 'awaiting_approval'}},
                },
            ],
        },
    ]

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'text': 't2'},
                    {
                        'toolRequest': {
                            'ref': 'g2',
                            'name': 'bank_transfer',
                            'input': {'preapproved': True},
                        },
                    },
                ],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'ok'}]}),
        )
    )
    r_ok = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['bank_transfer'],
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})],
        ),
    )
    assert r_ok.finish_reason == FinishReason.STOP
    assert _wire(r_ok.messages) == [
        {
            'role': 'user',
            'content': [{'text': 'hi'}],
        },
        {
            'role': 'model',
            'content': [
                {'text': 't2'},
                {
                    'toolRequest': {
                        'ref': 'g2',
                        'name': 'bank_transfer',
                        'input': {'preapproved': True},
                    },
                },
            ],
        },
        {
            'role': 'tool',
            'content': [
                {'toolResponse': {'ref': 'g2', 'name': 'bank_transfer', 'output': 42}},
            ],
        },
        {
            'role': 'model',
            'content': [{'text': 'ok'}],
        },
    ]


@pytest.mark.asyncio
async def test_resume_restart_runs_tool_second_time_and_resolved_interrupt_on_model() -> None:
    """``Resume(restart=[...])`` reruns the tool with new input after an interrupt. The tool runs
    twice (tracked in ``calls``); the second pass shows ``resolvedInterrupt`` on the model TRP and a
    plain ``toolResponse`` (no ``interruptResponse`` on that path).
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    calls: list[str] = []

    @ai.tool(name='pay')
    async def pay(inp: dict) -> str:
        calls.append('run')
        if not inp.get('ok'):
            raise Interrupt({'hold': True})
        return 'paid'

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'text': 'x'},
                    {'toolRequest': {'ref': 'p1', 'name': 'pay', 'input': {}}},
                ],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'final'}]}),
        )
    )
    # ^ Queued for the second generate call (after restart re-runs the tool).

    first = await generate_action(
        ai.registry,
        _gen_opts(ai, tools=['pay'], messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})]),
    )
    assert first.finish_reason == FinishReason.INTERRUPTED
    assert _wire(first.messages) == [
        {
            'role': 'user',
            'content': [{'text': 'hi'}],
        },
        {
            'role': 'model',
            'content': [
                {'text': 'x'},
                {
                    'toolRequest': {'ref': 'p1', 'name': 'pay', 'input': {}},
                    'metadata': {'interrupt': {'hold': True}},
                },
            ],
        },
    ]

    restart_trp = restart_tool(
        interrupt=first.interrupts[0], replace_input={'ok': True}, resumed_metadata={'by': 'test'}
    )

    second = await generate_action(
        ai.registry,
        _gen_opts(ai, tools=['pay'], messages=list(first.messages), resume=Resume(restart=[restart_trp])),
    )

    assert second.finish_reason == FinishReason.STOP
    assert calls == ['run', 'run']
    assert _wire(second.messages) == [
        {
            'role': 'user',
            'content': [{'text': 'hi'}],
        },
        {
            'role': 'model',
            'content': [
                {'text': 'x'},
                {
                    'toolRequest': {'ref': 'p1', 'name': 'pay', 'input': {}},
                    'metadata': {'resolvedInterrupt': {'hold': True}},
                },
            ],
        },
        {
            'role': 'tool',
            'content': [
                {'toolResponse': {'ref': 'p1', 'name': 'pay', 'output': 'paid'}},
            ],
            'metadata': {'resumed': True},
        },
        {
            'role': 'model',
            'content': [{'text': 'final'}],
        },
    ]


@pytest.mark.asyncio
async def test_resume_top_level_metadata_lands_on_tool_message() -> None:
    """Top-level ``Resume(metadata=...)`` is stamped onto the resolved tool message's
    ``metadata.resumed`` (matching JS ``resumed: resume.metadata || true``), rather than
    being flattened to ``True``.
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='pay')
    async def pay(inp: dict) -> str:
        if not inp.get('ok'):
            raise Interrupt({'hold': True})
        return 'paid'

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [{'toolRequest': {'ref': 'p1', 'name': 'pay', 'input': {}}}],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'final'}]}),
        )
    )

    first = await generate_action(
        ai.registry,
        _gen_opts(ai, tools=['pay'], messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})]),
    )
    restart_trp = restart_tool(interrupt=first.interrupts[0], replace_input={'ok': True})

    second = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['pay'],
            messages=list(first.messages),
            resume=Resume(restart=[restart_trp], metadata={'approved_by': 'test'}),
        ),
    )

    tool_msg = next(m for m in second.messages if m.role == 'tool')
    assert tool_msg.metadata == {'resumed': {'approved_by': 'test'}}


@pytest.mark.asyncio
async def test_mixed_resume_one_respond_one_restart() -> None:
    """Two tool calls both interrupt in one turn; the next generate fills in a ``respond`` for one
    ref and a ``restart`` for the other. Expect one tool message with two parts (respond path still
    has ``interruptResponse``; restart path does not).
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='a')
    async def a_tool(_: dict) -> str:  # noqa: ARG001
        raise Interrupt({'tool': 'a'})

    @ai.tool(name='b')
    async def b_tool(inp: dict) -> str:
        if inp.get('ok'):
            return 'b-done'
        raise Interrupt({'tool': 'b'})

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'text': 'both'},
                    {'toolRequest': {'ref': 'ra', 'name': 'a', 'input': {}}},
                    {'toolRequest': {'ref': 'rb', 'name': 'b', 'input': {}}},
                ],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'end'}]}),
        )
    )

    first = await generate_action(
        ai.registry,
        _gen_opts(
            ai, tools=['a', 'b'], messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})]
        ),
    )
    assert first.finish_reason == FinishReason.INTERRUPTED
    assert _wire(first.messages) == [
        {
            'role': 'user',
            'content': [{'text': 'hi'}],
        },
        {
            'role': 'model',
            'content': [
                {'text': 'both'},
                {
                    'toolRequest': {'ref': 'ra', 'name': 'a', 'input': {}},
                    'metadata': {'interrupt': {'tool': 'a'}},
                },
                {
                    'toolRequest': {'ref': 'rb', 'name': 'b', 'input': {}},
                    'metadata': {'interrupt': {'tool': 'b'}},
                },
            ],
        },
    ]

    ia = next(p for p in first.interrupts if p.tool_request is not None and p.tool_request.name == 'a')
    ib = next(p for p in first.interrupts if p.tool_request is not None and p.tool_request.name == 'b')

    second = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['a', 'b'],
            messages=list(first.messages),
            resume=Resume(
                respond=[respond_to_interrupt({'done': True}, interrupt=ia)],
                restart=[restart_tool(interrupt=ib, replace_input={'ok': True})],
            ),
        ),
    )

    assert second.finish_reason == FinishReason.STOP
    assert _wire(second.messages) == [
        {
            'role': 'user',
            'content': [{'text': 'hi'}],
        },
        {
            'role': 'model',
            'content': [
                {'text': 'both'},
                {
                    'toolRequest': {'ref': 'ra', 'name': 'a', 'input': {}},
                    'metadata': {'resolvedInterrupt': {'tool': 'a'}},
                },
                {
                    'toolRequest': {'ref': 'rb', 'name': 'b', 'input': {}},
                    'metadata': {'resolvedInterrupt': {'tool': 'b'}},
                },
            ],
        },
        {
            'role': 'tool',
            'content': [
                {
                    'toolResponse': {'ref': 'ra', 'name': 'a', 'output': {'done': True}},
                    'metadata': {'interruptResponse': True},
                },
                # Restart path: tool re-runs and returns its own output; no interruptResponse metadata.
                {'toolResponse': {'ref': 'rb', 'name': 'b', 'output': 'b-done'}},
            ],
            'metadata': {'resumed': True},
        },
        {
            'role': 'model',
            'content': [{'text': 'end'}],
        },
    ]


@pytest.mark.asyncio
async def test_mixed_one_interrupts_one_succeeds_pending_output_in_wire() -> None:
    """Two tools in one turn: ``a`` interrupts, ``b`` succeeds.

    Turn 1: both run in parallel. ``b``'s output is stashed as ``pendingOutput``
    on its TRP in the model message (no tool message yet). finish=INTERRUPTED.

    Turn 2: resume with ``respond=[...]`` for ``a`` only — no action needed for
    ``b``. The framework reconstructs ``b``'s tool response from the stashed
    output, strips ``pendingOutput`` from ``b``'s model TRP, and
    marks the tool response ``source: pending`` on the wire.
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='a')
    async def a_tool(_: dict) -> str:  # noqa: ARG001
        raise Interrupt({'reason': 'needs_approval'})

    @ai.tool(name='b')
    async def b_tool(_: dict) -> int:  # noqa: ARG001
        return 42

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'toolRequest': {'ref': 'ra', 'name': 'a', 'input': {}}},
                    {'toolRequest': {'ref': 'rb', 'name': 'b', 'input': {}}},
                ],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'done'}]}),
        )
    )

    first = await generate_action(
        ai.registry,
        _gen_opts(
            ai, tools=['a', 'b'], messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})]
        ),
    )
    assert first.finish_reason == FinishReason.INTERRUPTED
    # b's output is stashed in pendingOutput on its TRP; no tool message yet.
    assert _wire(first.messages) == [
        {'role': 'user', 'content': [{'text': 'hi'}]},
        {
            'role': 'model',
            'content': [
                {
                    'toolRequest': {'ref': 'ra', 'name': 'a', 'input': {}},
                    'metadata': {'interrupt': {'reason': 'needs_approval'}},
                },
                {
                    'toolRequest': {'ref': 'rb', 'name': 'b', 'input': {}},
                    'metadata': {'pendingOutput': 42},
                },
            ],
        },
    ]

    ia = first.interrupts[0]
    second = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['a', 'b'],
            messages=list(first.messages),
            resume=Resume(respond=[respond_to_interrupt({'approved': True}, interrupt=ia)]),
        ),
    )

    assert second.finish_reason == FinishReason.STOP
    assert _wire(second.messages) == [
        {'role': 'user', 'content': [{'text': 'hi'}]},
        {
            'role': 'model',
            'content': [
                {
                    'toolRequest': {'ref': 'ra', 'name': 'a', 'input': {}},
                    'metadata': {'resolvedInterrupt': {'reason': 'needs_approval'}},
                },
                {'toolRequest': {'ref': 'rb', 'name': 'b', 'input': {}}},
            ],
        },
        {
            'role': 'tool',
            'content': [
                {
                    'toolResponse': {'ref': 'ra', 'name': 'a', 'output': {'approved': True}},
                    'metadata': {'interruptResponse': True},
                },
                {
                    # b ran on turn 1; output reconstructed from pendingOutput stash.
                    'toolResponse': {'ref': 'rb', 'name': 'b', 'output': 42},
                    'metadata': {'source': 'pending'},
                },
            ],
            'metadata': {'resumed': True},
        },
        {'role': 'model', 'content': [{'text': 'done'}]},
    ]


@pytest.mark.asyncio
async def test_pending_multipart_response_survives_wire_round_trip() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    media_calls = 0

    @ai.tool(name='pause')
    async def pause(_: dict) -> str:  # noqa: ARG001
        raise Interrupt({'reason': 'approval'})

    @ai.tool(name='media')
    async def media() -> str:
        nonlocal media_calls
        media_calls += 1
        return 'described'

    @ai.middleware(name='multipart_media')
    class MultipartMedia(BaseMiddleware):
        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            response = await next_fn(params, ctx)
            requested = params.tool_request_part.tool_request
            if requested is None or requested.name != 'media':
                return response
            return MultipartToolResponse(
                output=response.output,
                content=[Part(media=Media(url='data:image/png;base64,AAA', content_type='image/png'))],
                metadata={'traceId': 'media-1'},
            )

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'toolRequest': {'ref': 'pause-1', 'name': 'pause', 'input': {}}},
                    {
                        'toolRequest': {'ref': 'media-1', 'name': 'media', 'input': {}},
                        'metadata': {'requestTag': 'keep'},
                    },
                ],
            }),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.model_validate({'text': 'done'})]),
        ),
    ]
    options = GenerateActionOptions(
        model='programmableModel',
        messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'start'}]})],
        tools=['pause', 'media'],
        use=[MiddlewareRef(name='multipart_media')],
    )

    first = await generate_action(ai.registry, options)

    assert first.finish_reason == FinishReason.INTERRUPTED
    assert first.message is not None
    pending = next(
        part for part in first.message.content if part.tool_request is not None and part.tool_request.name == 'media'
    )
    assert pending.metadata is not None
    assert pending.metadata['requestTag'] == 'keep'
    assert pending.metadata['pendingOutput'] == 'described'
    assert pending.metadata['pendingMetadata'] == {'traceId': 'media-1'}
    pending_content = pending.metadata['pendingContent']
    assert isinstance(pending_content, list)
    pending_media = Part.model_validate(pending_content[0])
    assert pending_media.media == Media(url='data:image/png;base64,AAA', content_type='image/png')

    messages = [Message.model_validate(message) for message in json.loads(json.dumps(_wire(first.messages)))]
    interrupt = next(
        part for part in messages[-1].content if part.tool_request is not None and part.tool_request.name == 'pause'
    )
    second = await generate_action(
        ai.registry,
        options.model_copy(
            update={
                'messages': messages,
                'resume': Resume(respond=[respond_to_interrupt('approved', interrupt=interrupt)]),
            }
        ),
    )

    assert second.finish_reason == FinishReason.STOP
    assert media_calls == 1
    revised_media = next(
        part
        for part in second.messages[1].content
        if part.tool_request is not None and part.tool_request.name == 'media'
    )
    assert revised_media.metadata == {'requestTag': 'keep'}
    tool_message = second.messages[2]
    media_response = next(
        part for part in tool_message.content if part.tool_response is not None and part.tool_response.name == 'media'
    )
    assert media_response.tool_response is not None
    assert media_response.tool_response.output == 'described'
    assert media_response.tool_response.content is not None
    replayed_media = Part.model_validate(media_response.tool_response.content[0])
    assert replayed_media.media == Media(url='data:image/png;base64,AAA', content_type='image/png')
    assert media_response.metadata == {'traceId': 'media-1', 'source': 'pending'}
    assert not {'pendingOutput', 'pendingContent', 'pendingMetadata'} & set(media_response.metadata)


@pytest.mark.asyncio
async def test_restarted_tools_run_concurrently_and_keep_request_order() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    beta_done = asyncio.Event()
    calls: list[str] = []

    @ai.tool(name='alpha')
    async def alpha(inp: dict) -> str:
        calls.append('alpha')
        if not inp.get('restart'):
            raise Interrupt({'tool': 'alpha'})
        await beta_done.wait()
        return 'A'

    @ai.tool(name='beta')
    async def beta(inp: dict) -> str:
        calls.append('beta')
        if not inp.get('restart'):
            raise Interrupt({'tool': 'beta'})
        beta_done.set()
        return 'B'

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'toolRequest': {'ref': 'alpha-1', 'name': 'alpha', 'input': {}}},
                    {'toolRequest': {'ref': 'beta-1', 'name': 'beta', 'input': {}}},
                ],
            }),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.model_validate({'text': 'done'})]),
        ),
    ]
    first = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['alpha', 'beta'],
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'start'}]})],
        ),
    )
    interrupts = {p.tool_request.name: p for p in first.interrupts if p.tool_request is not None}

    second = await asyncio.wait_for(
        generate_action(
            ai.registry,
            _gen_opts(
                ai,
                tools=['alpha', 'beta'],
                messages=list(first.messages),
                resume=Resume(
                    restart=[
                        restart_tool(interrupt=interrupts['alpha'], replace_input={'restart': True}),
                        restart_tool(interrupt=interrupts['beta'], replace_input={'restart': True}),
                    ]
                ),
            ),
        ),
        timeout=2,
    )

    assert second.finish_reason == FinishReason.STOP
    assert calls.count('alpha') == 2
    assert calls.count('beta') == 2
    tool_message = next(message for message in second.messages if message.role == Role.TOOL)
    tool_responses = []
    for part in tool_message.content:
        assert part.tool_response is not None
        tool_responses.append(part.tool_response)
    assert [response.name for response in tool_responses] == ['alpha', 'beta']
    assert [response.output for response in tool_responses] == ['A', 'B']


@pytest.mark.asyncio
async def test_resume_without_matching_replies_is_still_resendable() -> None:
    """An interrupted TRP with an empty ``Resume()`` is a failed response they can resend."""
    ai = Genkit()
    _, _ = define_programmable_model(ai)
    messages = [
        Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]}),
        Message.model_validate({
            'role': 'model',
            'content': [
                {
                    'toolRequest': {'ref': 'z', 'name': 'missing', 'input': {}},
                    'metadata': {'interrupt': True},
                },
            ],
        }),
    ]

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=messages,
            resume=Resume(),
        ),
    )
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'unresolved tool request' in response.finish_message.lower()
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is RuntimeErrorReason.UNRESOLVED_TOOL_REQUEST
    assert 'UNRESOLVED_TOOL_REQUEST' not in response.error.message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[0].text == 'hi'
    requested = response.messages[1].tool_requests[0].tool_request
    assert requested is not None
    assert requested.ref == 'z'


@pytest.mark.asyncio
async def test_resume_on_empty_messages_is_still_resendable() -> None:
    """resume= on an empty conversation is a failed response, not an IndexError."""
    ai = Genkit()
    _, _ = define_programmable_model(ai)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[],
            resume=Resume(),
        ),
    )
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "cannot 'resume'" in response.finish_message.lower()
    assert response.error is not None
    assert response.error.reason is RuntimeErrorReason.INVALID_RESUME
    assert response.message is None
    assert response.messages == []


@pytest.mark.asyncio
async def test_resume_on_user_turn_is_still_resendable() -> None:
    """resume= on a user turn is a failed response of the history they already had."""
    ai = Genkit()
    _, _ = define_programmable_model(ai)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'only user'}]})],
            resume=Resume(),
        ),
    )
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "cannot 'resume'" in response.finish_message.lower()
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is RuntimeErrorReason.INVALID_RESUME
    assert 'INVALID_RESUME' not in response.error.message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.messages[0].text == 'only user'


@pytest.mark.asyncio
async def test_resume_on_text_only_model_turn_is_still_resendable() -> None:
    """A model turn with no tool request is not a resume point; they still get that history back."""
    ai = Genkit()
    _, _ = define_programmable_model(ai)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[
                Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]}),
                Message.model_validate({'role': 'model', 'content': [{'text': 'hello'}]}),
            ],
            resume=Resume(),
        ),
    )
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "cannot 'resume'" in response.finish_message.lower()
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is RuntimeErrorReason.INVALID_RESUME
    assert 'INVALID_RESUME' not in response.error.message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.messages[0].text == 'hi'
    assert response.messages[1].text == 'hello'


@pytest.mark.asyncio
async def test_resume_on_tool_turn_is_still_resendable() -> None:
    """A transcript that already ends on a tool message is not a resume point."""
    ai = Genkit()
    _, _ = define_programmable_model(ai)

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[
                Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]}),
                Message.model_validate({
                    'role': 'model',
                    'content': [{'toolRequest': {'ref': 'z', 'name': 'echo', 'input': {}}}],
                }),
                Message.model_validate({
                    'role': 'tool',
                    'content': [{'toolResponse': {'ref': 'z', 'name': 'echo', 'output': 'ok'}}],
                }),
            ],
            resume=Resume(),
        ),
    )
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "cannot 'resume'" in response.finish_message.lower()
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is RuntimeErrorReason.INVALID_RESUME
    assert 'INVALID_RESUME' not in response.error.message
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert response.messages[0].text == 'hi'
    requested = response.messages[1].tool_requests[0].tool_request
    assert requested is not None
    assert requested.ref == 'z'


@pytest.mark.asyncio
async def test_restarted_tool_that_interrupts_again_returns_interrupted() -> None:
    """A tool that pauses again on restart returns INTERRUPTED they can answer."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='hold')
    async def hold(_: dict) -> str:  # noqa: ARG001
        raise Interrupt({'hold': True})

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [{'toolRequest': {'ref': '1', 'name': 'hold', 'input': {}}}],
            }),
        )
    )
    first = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})],
            tools=['hold'],
        ),
    )

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=list(first.messages),
            tools=['hold'],
            resume=Resume(restart=[restart_tool(interrupt=first.interrupts[0])]),
        ),
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


async def _screenshot_confirm_interrupted() -> tuple[Genkit, Any]:
    """Screenshot finishes with a PNG; confirm interrupts. Caller resumes or mutates the stash."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    png = Part.from_media('data:image/png;base64,abc', content_type='image/png')

    @ai.tool(name='confirm')
    async def confirm(_: dict) -> None:  # noqa: ARG001
        raise Interrupt({'needs_approval': True})

    @ai.tool(name='screenshot')
    async def screenshot(_: dict) -> MultipartToolResponse:  # noqa: ARG001
        return response({'ok': True, 'label': 'lab'}, parts=[png], metadata={'src': 'cam'})

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [
                    {'toolRequest': {'ref': 'c1', 'name': 'confirm', 'input': {}}},
                    {'toolRequest': {'ref': 's1', 'name': 'screenshot', 'input': {}}},
                ],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'posted'}]}),
        )
    )

    first = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['confirm', 'screenshot'],
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'post the lab'}]})],
        ),
    )
    assert first.finish_reason == FinishReason.INTERRUPTED
    return ai, first


def _with_shot_pending(first: Any, **pending: object) -> list[Message]:
    messages = [m.model_copy(deep=True) for m in first.messages]
    part = messages[1].content[1]
    assert part.metadata is not None
    part.metadata.update(pending)
    return messages


@pytest.mark.asyncio
async def test_mixed_interrupt_preserves_sibling_media_on_resume() -> None:
    """Screenshot finishes with a PNG; confirm interrupts. Resume must still send the PNG."""
    ai, first = await _screenshot_confirm_interrupted()
    shot_meta = first.messages[1].content[1].metadata
    assert shot_meta is not None
    assert shot_meta.get('pendingOutput') == {'ok': True, 'label': 'lab'}
    assert shot_meta.get('pendingContent') == [
        {'media': {'contentType': 'image/png', 'url': 'data:image/png;base64,abc'}}
    ]
    assert shot_meta.get('pendingMetadata') == {'src': 'cam'}

    second = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['confirm', 'screenshot'],
            messages=list(first.messages),
            resume=Resume(respond=[respond_to_interrupt({'approved': True}, interrupt=first.interrupts[0])]),
        ),
    )
    assert second.finish_reason == FinishReason.STOP
    tool_msg = next(m for m in second.messages if m.role == 'tool')
    shot_resp = tool_msg.content[1].tool_response
    assert shot_resp is not None
    assert shot_resp.output == {'ok': True, 'label': 'lab'}
    assert shot_resp.content == [{'media': {'contentType': 'image/png', 'url': 'data:image/png;base64,abc'}}]
    assert tool_msg.content[1].metadata == {'src': 'cam', 'source': 'pending'}


@pytest.mark.asyncio
async def test_resume_rejects_hollow_pending_content() -> None:
    """A saved conversation whose pending screenshot has no live payload fails on the response."""
    ai, first = await _screenshot_confirm_interrupted()
    response = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['confirm', 'screenshot'],
            messages=_with_shot_pending(first, pendingContent=[{}]),
            resume=Resume(respond=[respond_to_interrupt({'approved': True}, interrupt=first.interrupts[0])]),
        ),
    )
    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is RuntimeErrorReason.INVALID_PART
    assert 'screenshot' in (response.finish_message or '')
    assert 'pendingContent' in (response.finish_message or '')
    assert response.message is None
    assert Role.USER in [m.role for m in response.messages]
    assert Role.MODEL in [m.role for m in response.messages]


@pytest.mark.asyncio
async def test_resume_rejects_text_and_media_pending_content() -> None:
    """A saved conversation whose pending screenshot is caption plus image on one part fails on the response."""
    ai, first = await _screenshot_confirm_interrupted()
    response = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['confirm', 'screenshot'],
            messages=_with_shot_pending(
                first,
                pendingContent=[{'text': 'caption', 'media': {'url': 'https://x'}}],
            ),
            resume=Resume(respond=[respond_to_interrupt({'approved': True}, interrupt=first.interrupts[0])]),
        ),
    )
    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is RuntimeErrorReason.INVALID_PART
    assert 'pendingContent' in (response.finish_message or '')
    assert 'exactly one' in (response.finish_message or '')
    assert response.message is None
    assert Role.USER in [m.role for m in response.messages]
    assert Role.MODEL in [m.role for m in response.messages]


@pytest.mark.asyncio
async def test_resume_rejects_non_dict_pending_metadata() -> None:
    """A saved conversation whose pending screenshot metadata is not a dict fails on the response."""
    ai, first = await _screenshot_confirm_interrupted()
    response = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['confirm', 'screenshot'],
            messages=_with_shot_pending(first, pendingMetadata='cam'),
            resume=Resume(respond=[respond_to_interrupt({'approved': True}, interrupt=first.interrupts[0])]),
        ),
    )
    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert response.error.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'screenshot' in (response.finish_message or '')
    assert 'pendingMetadata' in (response.finish_message or '')
    assert response.message is None
    assert Role.USER in [m.role for m in response.messages]
    assert Role.MODEL in [m.role for m in response.messages]


async def _restart_screenshot(*, with_passthrough: bool = False) -> tuple[Any, Any]:
    """Interrupt a screenshot tool, then restart it so it returns a PNG + metadata."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    use: list[MiddlewareRef] | None = None
    if with_passthrough:

        @ai.middleware(name='passthrough_mw')
        class PassthroughMW(BaseMiddleware):
            async def wrap_tool(
                self,
                params: ToolHookParams,
                ctx: GenerateMiddlewareContext,
                next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
            ) -> MultipartToolResponse:
                return await next_fn(params, ctx)

        use = [MiddlewareRef(name='passthrough_mw')]

    @ai.tool(name='shot')
    async def shot(inp: dict) -> MultipartToolResponse:
        if not inp.get('ok'):
            raise Interrupt({'hold': True})
        return response({'ok': True}, parts=[_png()], metadata={'camera': 'rear'})

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [{'toolRequest': {'ref': 's1', 'name': 'shot', 'input': {}}}],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'posted'}]}),
        )
    )

    first = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['shot'],
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'snap'}]})],
            use=use,
        ),
    )
    assert first.finish_reason == FinishReason.INTERRUPTED

    second = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['shot'],
            messages=list(first.messages),
            resume=Resume(restart=[restart_tool(interrupt=first.interrupts[0], replace_input={'ok': True})]),
            use=use,
        ),
    )
    tool_msg = next(m for m in second.messages if m.role == 'tool')
    return tool_msg.content[0].tool_response, tool_msg.content[0].metadata


@pytest.mark.asyncio
async def test_resume_restart_copies_parts_and_metadata() -> None:
    """After approval, the restarted screenshot must still send the PNG and its metadata."""
    tr, meta = await _restart_screenshot()
    assert tr.output == {'ok': True}
    assert tr.content == [WIRE_PNG]
    assert meta == {'camera': 'rear'}


@pytest.mark.asyncio
async def test_resume_restart_keeps_parts_and_metadata_through_middleware() -> None:
    """A passthrough wrap_tool must not drop the restarted screenshot's media or metadata."""
    tr, meta = await _restart_screenshot(with_passthrough=True)
    assert tr.output == {'ok': True}
    assert tr.content == [WIRE_PNG]
    assert meta == {'camera': 'rear'}
