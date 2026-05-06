# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for interrupt, resume, and restart behavior in ``generate_action``."""

from __future__ import annotations

from typing import Any

import pytest

from genkit import Genkit, Message, ModelResponse
from genkit._ai._generate import generate_action
from genkit._ai._testing import define_programmable_model
from genkit._ai._tools import Interrupt, ToolRunContext, respond_to_interrupt
from genkit._core._error import GenkitError
from genkit._core._model import GenerateActionOptions
from genkit._core._typing import FinishReason, Resume


def _wire(messages: list[Message]) -> list[dict[str, Any]]:
    """Messages as JSON-shaped dicts (``model_dump`` with aliases) for comparing to expected wire."""
    return [m.model_dump(mode='json', exclude_none=True, by_alias=True) for m in messages]


def _gen_opts(
    ai: Genkit, *, tools: list[str], messages: list[Message], resume: Resume | None = None
) -> GenerateActionOptions:
    return GenerateActionOptions(
        model='programmableModel',
        messages=messages,
        tools=tools,
        resume=resume,
    )


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

    restart_trp = pay.restart({'ok': True}, interrupt=first.interrupts[0], resumed_metadata={'by': 'test'})

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

    ia = next(p for p in first.interrupts if p.tool_request.name == 'a')
    ib = next(p for p in first.interrupts if p.tool_request.name == 'b')

    second = await generate_action(
        ai.registry,
        _gen_opts(
            ai,
            tools=['a', 'b'],
            messages=list(first.messages),
            resume=Resume(
                respond=[respond_to_interrupt({'done': True}, interrupt=ia)],
                restart=[b_tool.restart({'ok': True}, interrupt=ib, resumed_metadata=None)],
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
async def test_resume_without_matching_replies_raises() -> None:
    """Hand-built history with an interrupted TRP but an empty ``Resume()``: expect ``GenkitError``
    and a message that mentions replies or restarts.
    """
    ai = Genkit()
    _, _ = define_programmable_model(ai)

    with pytest.raises(GenkitError) as ei:
        await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[
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
                ],
                resume=Resume(),
            ),
        )
    assert ei.value.status == 'INVALID_ARGUMENT'
    assert 'unresolved tool request' in ei.value.original_message.lower()


@pytest.mark.asyncio
async def test_resume_requires_last_message_model_with_tool_requests() -> None:
    """Can't resume when the transcript ends on a user turn: ``GenkitError``, and the message should
    mention needing a model message.
    """
    ai = Genkit()
    _, _ = define_programmable_model(ai)

    with pytest.raises(GenkitError) as ei:
        await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'only user'}]})],
                resume=Resume(),
            ),
        )
    assert ei.value.status == 'FAILED_PRECONDITION'
    assert "cannot 'resume'" in ei.value.original_message.lower()
