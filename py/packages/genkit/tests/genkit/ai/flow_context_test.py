# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What context a subflow sees: the parent's unless the call passes its own."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from genkit import ActionRunContext, FinishReason, Genkit, Message, ModelResponse, Part, Role
from genkit._ai._testing import define_programmable_model
from genkit._core._telemetry.http import ActiveSpan
from genkit._core._typing import ToolRequest

CALLER = {'auth': {'uid': 'u_42', 'tier': 'gold'}, 'locale': 'en-US'}


class CardDeclinedError(Exception):
    """Raised by the failing subflow so the test can tell its error from any other."""


def _define_allergy_check(ai: Genkit):  # noqa: ANN202
    @ai.flow()
    async def allergy_check(dish: str, ctx: ActionRunContext) -> dict[str, object]:
        return dict(ctx.context)

    return allergy_check


def _span(spans: Sequence[ActiveSpan], name: str) -> ActiveSpan:
    matches = [s for s in spans if s.name == name]
    assert matches, f'no span named {name!r} in {[s.name for s in spans]}'
    return matches[-1]


def _raised_by(error: BaseException | None, cls: type[BaseException]) -> bool:
    """True if `error` is `cls` or was raised from one."""
    while error is not None:
        if isinstance(error, cls):
            return True
        error = error.__cause__
    return False


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
    async def order_dish(dish: str) -> object:
        await allergy_check.run(dish, context={'auth': {'uid': 'kitchen'}})
        return ai.current_context()

    result = await order_dish.run('Smoked Salmon Tartine', context=CALLER)

    assert result.response == CALLER
    assert ai.current_context() is None


@pytest.mark.asyncio
async def test_parent_context_is_back_after_failing_subflow_override() -> None:
    """A subflow that overrides context and raises still leaves the parent's context in place."""
    ai = Genkit()
    errors: list[BaseException] = []

    @ai.flow()
    async def charge_card(amount: int) -> str:
        raise CardDeclinedError('card declined')

    @ai.flow()
    async def order_dish(dish: str) -> object:
        try:
            await charge_card.run(42, context={'auth': {'uid': 'billing'}})
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        return ai.current_context()

    result = await order_dish.run('Smoked Salmon Tartine', context=CALLER)

    assert len(errors) == 1
    assert _raised_by(errors[0], CardDeclinedError), repr(errors[0])
    assert result.response == CALLER
    assert ai.current_context() is None


@pytest.mark.asyncio
async def test_streamed_subflow_sees_parent_context() -> None:
    """A subflow run with `.stream()`, which Genkit runs as its own task, sees the parent's context."""
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
async def test_subflow_span_records_inherited_context(exporter) -> None:
    """A subflow called without `context=` records the parent's context on its own span."""
    ai = Genkit()
    allergy_check = _define_allergy_check(ai)

    @ai.flow()
    async def order_dish(dish: str) -> dict[str, object]:
        return await allergy_check(dish)

    await order_dish.run('Smoked Salmon Tartine', context=CALLER)

    attrs = dict(_span(exporter.get_finished_spans(), 'allergy_check').attributes or {})
    context_attr = attrs['genkit:metadata:context']
    assert isinstance(context_attr, str)
    assert json.loads(context_attr) == {'auth': '<redacted>', 'locale': 'en-US'}
