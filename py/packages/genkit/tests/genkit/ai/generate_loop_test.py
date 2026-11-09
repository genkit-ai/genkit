#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Turn-loop regressions: usage totals, max_turns validation, and friends."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from genkit import (
    Document,
    Genkit,
    Interrupt,
    Message,
    ModelResponse,
    ModelResponseChunk,
    MultipartToolResponse,
    Part,
    Tool,
    ToolContext,
)
from genkit._ai._agents._client import agent_interrupts_from_message
from genkit._ai._generate import CallScope, augment_with_context, generate_action
from genkit._core._action import ActionRunContext
from genkit._core._error import ErrorReason, GenkitError
from genkit._core._model import GenerateActionOptions, ModelRequest
from genkit._core._typing import FinishReason, GenerationUsage, Role, ToolRequest
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams
from genkit.testing import ScriptedModel, define_scripted_model


def _calls_tool(*, ref: str, usage: GenerationUsage | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part(tool_request=ToolRequest(name='lookup', input={}, ref=ref))]),
        usage=usage,
    )


def _answers(text: str, usage: GenerationUsage | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
        usage=usage,
    )


@pytest.mark.asyncio
async def test_total_usage_sums_every_turn_of_a_tool_loop() -> None:
    """usage is the last call; total_usage is what the whole loop billed."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return 'in stock'

    pm.responses = [
        _calls_tool(
            ref='1',
            usage=GenerationUsage(input_tokens=10, output_tokens=5, thoughts_tokens=3, custom={'calls': 1, 'id': 'a'}),
        ),
        _calls_tool(ref='2', usage=GenerationUsage(input_tokens=20, output_tokens=6, custom={'calls': 1, 'id': 'b'})),
        _answers('done', usage=GenerationUsage(input_tokens=30, output_tokens=7)),
    ]

    response = await ai.generate(prompt='check stock', tools=['lookup'])

    assert response.text == 'done'
    assert response.usage is not None
    assert response.usage.input_tokens == 30
    assert response.total_usage == GenerationUsage(
        input_tokens=60,
        output_tokens=18,
        thoughts_tokens=3,
        custom={'calls': 2, 'id': 'b'},
    )


@pytest.mark.asyncio
async def test_total_usage_matches_usage_on_a_single_turn() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('hi', usage=GenerationUsage(input_tokens=4, output_tokens=2))]

    response = await ai.generate(prompt='hello')

    assert response.total_usage == response.usage


@pytest.mark.asyncio
async def test_total_usage_survives_a_max_turns_abort() -> None:
    """The rounds billed before the cap still show up on the aborted response."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return 'again'

    pm.responses = [
        _calls_tool(ref='1', usage=GenerationUsage(input_tokens=10)),
        _calls_tool(ref='2', usage=GenerationUsage(input_tokens=11)),
    ]

    response = await ai.generate(prompt='loop', tools=['lookup'], max_turns=1)

    assert response.finish_reason == FinishReason.ABORTED
    assert response.total_usage is not None
    assert response.total_usage.input_tokens == 21


@pytest.mark.asyncio
async def test_total_usage_stays_off_when_no_model_reported_usage() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('hi')]

    response = await ai.generate(prompt='hello')

    assert response.total_usage is None


def _transfer_request(amount: int) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name='transfer', input={'days': amount}, ref='t1'))],
        ),
    )


def _hitl_app() -> tuple[Genkit, ScriptedModel, list[object]]:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran_with: list[object] = []

    @ai.tool(name='transfer')
    async def transfer(inp: dict, ctx: ToolContext) -> str:
        if not ctx.is_resumed():
            raise Interrupt({'reason': 'needs approval', 'days': inp['days']})
        ran_with.append(inp)
        return f'booked {inp["days"]} days'

    return ai, pm, ran_with


@pytest.mark.asyncio
async def test_restart_replace_input_is_the_input_history_keeps() -> None:
    """The model's next turn and the audit trail see the approved input, not the original."""
    ai, pm, ran_with = _hitl_app()
    pm.responses = [_transfer_request(7), _answers('booked')]

    paused = await ai.generate(prompt='book 7 days', tools=['transfer'])
    assert paused.finish_reason == FinishReason.INTERRUPTED
    intr = paused.interrupts[0]
    assert intr.metadata is not None
    assert intr.metadata['interrupt'] == {'reason': 'needs approval', 'days': 7}

    done = await ai.generate(
        messages=paused.messages,
        tools=['transfer'],
        resume_restart=intr.restart(replace_input={'days': 5}),
    )

    assert ran_with == [{'days': 5}]
    assert done.text == 'booked'
    model_turn = next(m for m in done.messages if m.role == Role.MODEL and m.content[0].tool_request is not None)
    part = model_turn.content[0]
    assert part.tool_request is not None
    assert part.tool_request.input == {'days': 5}
    assert part.metadata is not None
    assert part.metadata['replacedInput'] == {'days': 7}
    assert part.metadata['resolvedInterrupt'] == {'reason': 'needs approval', 'days': 7}
    assert 'interrupt' not in part.metadata
    assert 'resumed' not in part.metadata
    # The model reads the approved input on the turn after the restart.
    assert pm.last_request is not None
    sent = next(m for m in pm.last_request.messages if m.role == Role.MODEL)
    assert sent.content[0].tool_request is not None
    assert sent.content[0].tool_request.input == {'days': 5}


@pytest.mark.asyncio
async def test_restart_without_replace_input_keeps_the_model_input() -> None:
    ai, pm, ran_with = _hitl_app()
    pm.responses = [_transfer_request(7), _answers('booked')]

    paused = await ai.generate(prompt='book 7 days', tools=['transfer'])
    done = await ai.generate(
        messages=paused.messages,
        tools=['transfer'],
        resume_restart=paused.interrupts[0].restart(),
    )

    assert ran_with == [{'days': 7}]
    model_turn = next(m for m in done.messages if m.role == Role.MODEL and m.content[0].tool_request is not None)
    assert model_turn.content[0].tool_request is not None
    assert model_turn.content[0].tool_request.input == {'days': 7}
    assert 'replacedInput' not in (model_turn.content[0].metadata or {})


@pytest.mark.asyncio
async def test_restart_that_pauses_again_keeps_the_replaced_input() -> None:
    """A second approval step sees the input the restart ran with."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='transfer')
    async def transfer(inp: dict, ctx: ToolContext) -> str:
        raise Interrupt({'reason': 'second approver', 'days': inp['days']})

    pm.responses = [_transfer_request(7)]
    paused = await ai.generate(prompt='book 7 days', tools=['transfer'])
    again = await ai.generate(
        messages=paused.messages,
        tools=['transfer'],
        resume_restart=paused.interrupts[0].restart(replace_input={'days': 5}),
    )

    assert again.finish_reason == FinishReason.INTERRUPTED
    part = again.interrupts[0]
    assert part.tool_request is not None
    assert part.tool_request.input == {'days': 5}
    assert part.metadata is not None
    assert part.metadata['interrupt'] == {'reason': 'second approver', 'days': 5}
    assert part.metadata['replacedInput'] == {'days': 7}


def test_agent_interrupt_exposes_the_interrupt_payload() -> None:
    message = Message(
        role=Role.MODEL,
        content=[
            Part(
                tool_request=ToolRequest(name='transfer', input={'days': 7}, ref='t1'),
                metadata={'interrupt': {'reason': 'needs approval'}},
            )
        ],
    )

    [intr] = agent_interrupts_from_message(message)

    assert intr.metadata == {'reason': 'needs approval'}


@pytest.mark.asyncio
async def test_system_with_saved_history_replaces_the_old_persona() -> None:
    """Handing a saved chat to a new persona sends one system message, the new one."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('Bonjour'), _answers('Bill: $42')]

    first = await ai.generate(system='You are the host.', prompt='Hi')
    await ai.generate(system='You are the cashier.', messages=first.messages, prompt='Check please')

    assert pm.last_request is not None
    roles = [m.role for m in pm.last_request.messages]
    assert roles == [Role.SYSTEM, Role.USER, Role.MODEL, Role.USER]
    assert pm.last_request.messages[0].text == 'You are the cashier.'


@pytest.mark.asyncio
async def test_empty_system_clears_the_saved_persona() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('Bonjour'), _answers('ok')]

    first = await ai.generate(system='You are the host.', prompt='Hi')
    await ai.generate(system='', messages=first.messages, prompt='Thanks')

    assert pm.last_request is not None
    assert [m.role for m in pm.last_request.messages] == [Role.USER, Role.MODEL, Role.USER]


@pytest.mark.asyncio
async def test_omitted_system_keeps_the_saved_persona() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('Bonjour'), _answers('ok')]

    first = await ai.generate(system='You are the host.', prompt='Hi')
    await ai.generate(messages=first.messages, prompt='Thanks')

    assert pm.last_request is not None
    assert pm.last_request.messages[0].role == Role.SYSTEM
    assert pm.last_request.messages[0].text == 'You are the host.'


def test_docs_context_keeps_metadata_and_media() -> None:
    """Citation metadata goes out with the doc, and media parts are not dropped."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('Which dishes are vegan?')])],
        docs=[
            Document(
                content=[
                    Part.from_text('Lentil soup'),
                    Part.from_media('gs://menu/soup.png', content_type='image/png'),
                ],
                metadata={'id': 'menu-7', 'source': 'menu.pdf', 'page': 3},
            )
        ],
    )

    augmented = augment_with_context(request)

    user = augmented.messages[-1]
    context = [p for p in user.content if (p.metadata or {}).get('purpose') == 'context']
    assert context[0].text is not None
    assert '- [menu-7] {"source": "menu.pdf", "page": 3}: Lentil soup' in context[0].text
    assert context[1].media is not None
    assert context[1].media.url == 'gs://menu/soup.png'


def test_docs_context_without_extra_metadata_is_unchanged() -> None:
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        docs=[Document.from_text('Lentil soup', metadata={'ref': 'r1'})],
    )

    augmented = augment_with_context(request)

    assert augmented.messages[-1].content[-1].text is not None
    assert '- [r1]: Lentil soup\n' in augmented.messages[-1].content[-1].text


@pytest.mark.asyncio
async def test_child_flows_get_their_own_copy_of_the_context() -> None:
    """Concurrent child flows writing ctx.context don't clobber each other or the parent."""
    ai = Genkit()
    seen: dict[str, object] = {}

    @ai.flow()
    async def tag_order(table: str, ctx: ActionRunContext) -> str:
        ctx.context['table'] = table
        await asyncio.sleep(0.01)
        seen[table] = ctx.context['table']
        return table

    @ai.flow()
    async def seat_party(n: int, ctx: ActionRunContext) -> list[str]:
        tables = await asyncio.gather(*(tag_order(f't{i}') for i in range(n)))
        seen['parent'] = ctx.context.get('table')
        return list(tables)

    await seat_party(3, context={'auth': {'uid': 'host-1'}})

    assert seen == {'t0': 't0', 't1': 't1', 't2': 't2', 'parent': None}


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [True, False, float('nan'), float('inf'), 2.5])
async def test_max_turns_rejects_non_whole_numbers(bad: object) -> None:
    """max_turns=True used to run as 1; nan leaked a raw ValidationError."""
    ai = Genkit(model='scriptedModel')
    define_scripted_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='hi', max_turns=bad)  # type: ignore[arg-type]

    assert raised.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_max_turns_accepts_a_whole_float_from_dotprompt() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('ok')]

    response = await generate_action(
        CallScope(ai),
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            max_turns=3.0,
        ),
    )

    assert response.text == 'ok'


@pytest.mark.asyncio
async def test_prompt_with_resume_fails_before_the_model_runs() -> None:
    ai, pm, _ = _hitl_app()
    pm.responses = [_transfer_request(7)]
    paused = await ai.generate(prompt='book 7 days', tools=['transfer'])
    calls = pm.request_count

    with pytest.raises(GenkitError) as raised:
        await ai.generate(
            prompt='also add breakfast',
            messages=paused.messages,
            tools=['transfer'],
            resume_restart=paused.interrupts[0].restart(),
        )

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.reason is ErrorReason.INVALID_RESUME
    assert pm.request_count == calls


@pytest.mark.asyncio
async def test_bare_restart_still_reads_as_resumed() -> None:
    """restart() with no metadata resumes with {}; the tool must still see a resume."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    seen: list[object] = []

    @ai.tool(name='transfer')
    async def transfer(inp: dict, ctx: ToolContext) -> str:
        if not ctx.is_resumed():
            raise Interrupt({'reason': 'approve'})
        seen.append(ctx.resumed_metadata)
        return 'ok'

    pm.responses = [_transfer_request(7), _answers('done')]
    paused = await ai.generate(prompt='book', tools=['transfer'])
    await ai.generate(messages=paused.messages, tools=['transfer'], resume_restart=paused.interrupts[0].restart())

    assert len(seen) == 1


def test_tool_accepts_an_async_callable_instance() -> None:
    """A class with `async def __call__` is an async tool; sync callables still fail."""

    class AllergyCheck:
        async def __call__(self, dish: str) -> str:
            return f'{dish}: nut-free'

    class SyncCheck:
        def __call__(self, dish: str) -> str:
            return dish

    tool = Tool(AllergyCheck(), name='allergy_check')
    assert tool.name == 'allergy_check'

    with pytest.raises(TypeError, match='must be async'):
        Tool(SyncCheck(), name='sync_check')


@pytest.mark.asyncio
async def test_middleware_sees_the_context_passed_to_generate() -> None:
    """Genkit.current_context() inside a hook is the generate call's context."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('ok')]
    seen: list[object] = []

    class AuditTenant(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            seen.append(Genkit.current_context())
            return await next_fn(params, ctx)

    await ai.generate(prompt='hi', context={'tenant': 'bistro-9'}, use=[AuditTenant()])

    assert seen == [{'tenant': 'bistro-9'}]
    assert Genkit.current_context() is None


def test_reasoning_accessors_stay_out_of_text() -> None:
    message = Message(
        role=Role.MODEL,
        content=[
            Part.from_reasoning('Check allergens first. '),
            Part.from_reasoning('Nuts: none.'),
            Part.from_text('Safe.'),
        ],
    )
    response = ModelResponse(message=message)
    chunk = ModelResponseChunk(role=Role.MODEL, content=[Part.from_reasoning('Thinking')])

    assert message.reasoning == 'Check allergens first. Nuts: none.'
    assert response.reasoning == message.reasoning
    assert response.text == 'Safe.'
    assert chunk.reasoning == 'Thinking'
    assert ModelResponse().reasoning == ''


def test_from_tool_response_carries_media_content() -> None:
    chart = Part.from_media('data:image/png;base64,iVBOR', content_type='image/png')

    part = Part.from_tool_response(name='sales_chart', ref='c1', output={'total': 42}, content=[chart])

    assert part.tool_response is not None
    assert part.tool_response.output == {'total': 42}
    assert part.tool_response.content == [{'media': {'url': 'data:image/png;base64,iVBOR', 'contentType': 'image/png'}}]


def test_from_tool_response_unwraps_a_multipart_output() -> None:
    chart = Part.from_media('data:image/png;base64,iVBOR', content_type='image/png')
    multipart = MultipartToolResponse(output={'total': 42}, content=[chart], metadata={'cached': True})

    part = Part.from_tool_response(name='sales_chart', ref='c1', output=multipart)

    assert part.tool_response is not None
    assert part.tool_response.output == {'total': 42}
    assert part.tool_response.content is not None
    assert len(part.tool_response.content) == 1
    assert part.metadata == {'cached': True}


@pytest.mark.asyncio
async def test_response_request_is_what_the_model_received() -> None:
    """A wrap_model that strips sensitive docs must not see them echoed on response.request.

    Messages on response.request stay the resendable history by design; scrub
    those in wrap_generate.
    """
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('ok')]

    class DropPatientDocs(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await next_fn(ModelHookParams(request=params.request.model_copy(update={'docs': None})), ctx)

    response = await ai.generate(
        prompt='summarize',
        docs=[Document.from_text('patient SSN 123-45-6789')],
        use=[DropPatientDocs()],
    )

    assert pm.last_request is not None
    assert pm.last_request.docs is None
    assert response.request is not None
    assert response.request.docs is None
    assert [m.text for m in response.request.messages] == ['summarize']
