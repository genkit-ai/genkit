# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What context a subflow sees: the parent's unless the call passes its own."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import pytest
from httpx import ASGITransport, AsyncClient

from genkit import ActionRunContext, FinishReason, Genkit, Message, ModelResponse, Part, Role
from genkit._ai._testing import define_programmable_model
from genkit._core._reflection import create_reflection_asgi_app
from genkit._core._telemetry.http import ActiveSpan
from genkit._core._typing import ToolRequest

CALLER = {'auth': {'uid': 'u_42', 'tier': 'gold'}, 'locale': 'en-US'}


def _define_allergy_check(ai: Genkit):  # noqa: ANN202
    @ai.flow()
    async def allergy_check(dish: str, ctx: ActionRunContext) -> dict[str, object]:
        return dict(ctx.context)

    return allergy_check


def _span(spans: Sequence[ActiveSpan], name: str) -> ActiveSpan:
    matches = [s for s in spans if s.name == name]
    assert matches, f'no span named {name!r} in {[s.name for s in spans]}'
    return matches[-1]


@pytest.mark.asyncio
async def test_subflow_without_context_sees_parent_context() -> None:
    """A subflow called with no `context=` sees the parent's context."""
    ai = Genkit()
    allergy_check = _define_allergy_check(ai)

    @ai.flow()
    async def order_dish(dish: str) -> dict[str, object]:
        return await allergy_check(dish)

    result = await order_dish.run('Smoked Salmon Tartine', context=CALLER)

    assert result.response == CALLER


@pytest.mark.asyncio
async def test_subflow_with_context_sees_only_that_context() -> None:
    """`context=` replaces the parent's context for the subflow; nothing is merged."""
    ai = Genkit()
    allergy_check = _define_allergy_check(ai)

    @ai.flow()
    async def order_dish(dish: str) -> dict[str, object]:
        result = await allergy_check.run(dish, context={'auth': {'uid': 'kitchen'}})
        return result.response

    result = await order_dish.run('Smoked Salmon Tartine', context=CALLER)

    assert result.response == {'auth': {'uid': 'kitchen'}}


@pytest.mark.asyncio
async def test_subflow_with_empty_context_sees_empty_context() -> None:
    """`context={}` is an override too: the subflow sees `{}`, not the parent's."""
    ai = Genkit()
    allergy_check = _define_allergy_check(ai)

    @ai.flow()
    async def order_dish(dish: str) -> dict[str, object]:
        result = await allergy_check.run(dish, context={})
        return result.response

    result = await order_dish.run('Smoked Salmon Tartine', context=CALLER)

    assert result.response == {}


@pytest.mark.asyncio
async def test_parent_context_is_back_after_subflow_override() -> None:
    """After an overriding subflow returns, the parent sees its own context again."""
    ai = Genkit()
    allergy_check = _define_allergy_check(ai)

    @ai.flow()
    async def order_dish(dish: str, ctx: ActionRunContext) -> dict[str, object]:
        await allergy_check.run(dish, context={'auth': {'uid': 'kitchen'}})
        return {'ctx': dict(ctx.context), 'current': ai.current_context()}

    result = await order_dish.run('Smoked Salmon Tartine', context=CALLER)

    assert result.response == {'ctx': CALLER, 'current': CALLER}
    assert ai.current_context() is None


@pytest.mark.asyncio
async def test_parent_context_is_back_after_failing_subflow_override() -> None:
    """A subflow that overrides context and raises still leaves the parent's context in place."""
    ai = Genkit()

    @ai.flow()
    async def charge_card(amount: int) -> str:
        raise ValueError('card declined')

    @ai.flow()
    async def order_dish(dish: str) -> object:
        # Exception, not ValueError: main still wraps it in GenkitError until #6576.
        with pytest.raises(Exception, match='card declined'):
            await charge_card.run(42, context={'auth': {'uid': 'billing'}})
        return ai.current_context()

    result = await order_dish.run('Smoked Salmon Tartine', context=CALLER)

    assert result.response == CALLER
    assert ai.current_context() is None


@pytest.mark.asyncio
async def test_gathered_subflows_see_parent_context() -> None:
    """Subflows run with `asyncio.gather` each see the parent's context."""
    ai = Genkit()
    allergy_check = _define_allergy_check(ai)

    @ai.flow()
    async def order_meal(dishes: list[str]) -> list[dict[str, object]]:
        return list(await asyncio.gather(*(allergy_check(d) for d in dishes)))

    result = await order_meal.run(['Tartine', 'Bisque'], context=CALLER)

    assert result.response == [CALLER, CALLER]


@pytest.mark.asyncio
async def test_task_subflow_sees_parent_context() -> None:
    """A subflow started with `asyncio.create_task` sees the parent's context."""
    ai = Genkit()
    allergy_check = _define_allergy_check(ai)

    @ai.flow()
    async def order_dish(dish: str) -> dict[str, object]:
        return await asyncio.create_task(allergy_check(dish))

    result = await order_dish.run('Smoked Salmon Tartine', context=CALLER)

    assert result.response == CALLER


@pytest.mark.asyncio
async def test_streamed_subflow_sees_parent_context() -> None:
    """A subflow run with `.stream()` sees the parent's context."""
    ai = Genkit()
    allergy_check = _define_allergy_check(ai)

    @ai.flow()
    async def order_dish(dish: str) -> dict[str, object]:
        streamed = allergy_check.stream(dish)
        async for _ in streamed.stream:
            pass
        return await streamed.response

    result = await order_dish.run('Smoked Salmon Tartine', context=CALLER)

    assert result.response == CALLER


@pytest.mark.asyncio
async def test_subflow_called_from_tool_during_generate_sees_flow_context() -> None:
    """flow → generate → tool → subflow: the subflow sees the outer flow's context."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    allergy_check = _define_allergy_check(ai)
    seen: list[dict[str, object]] = []

    @ai.tool()
    async def check_allergies(dish: str) -> str:
        """Check a dish against the diner's allergies."""
        seen.append(await allergy_check(dish))
        return 'no allergens'

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='check_allergies', input='Tartine', ref='r1'))],
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Tartine is safe.')]),
        ),
    ]

    @ai.flow()
    async def concierge(question: str) -> str:
        response = await ai.generate(model='programmableModel', prompt=question, tools=['check_allergies'])
        return response.text

    result = await concierge.run('Is the tartine safe?', context=CALLER)

    assert result.response == 'Tartine is safe.'
    assert seen == [CALLER]


@pytest.mark.asyncio
async def test_dev_ui_run_context_reaches_flow_and_subflow_and_span_hides_auth(exporter) -> None:
    """A Dev UI runAction with `context` reaches the flow and its subflow; both spans show `auth` redacted."""
    ai = Genkit()
    allergy_check = _define_allergy_check(ai)

    @ai.flow()
    async def order_dish(dish: str, ctx: ActionRunContext) -> dict[str, object]:
        return {'flow': dict(ctx.context), 'subflow': await allergy_check(dish)}

    app = create_reflection_asgi_app(ai.registry)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post(
            '/api/runAction',
            json={'key': '/flow/order_dish', 'input': 'Smoked Salmon Tartine', 'context': CALLER},
        )

    assert response.json()['result'] == {'flow': CALLER, 'subflow': CALLER}
    spans = exporter.get_finished_spans()
    for name in ('order_dish', 'allergy_check'):
        attrs = dict(_span(spans, name).attributes or {})
        context_attr = attrs['genkit:metadata:context']
        assert isinstance(context_attr, str)
        assert json.loads(context_attr) == {'auth': '<redacted>', 'locale': 'en-US'}
