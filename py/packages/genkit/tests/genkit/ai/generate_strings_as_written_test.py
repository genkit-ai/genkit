#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What the model receives for prompt and system strings passed to generate."""

import pytest

from genkit import Genkit, Message, ModelResponse, Part
from genkit._ai._testing import ProgrammableModel, define_programmable_model
from genkit._core._typing import FinishReason, Role
from genkit.model import ModelRequest


@pytest.fixture
def setup() -> tuple[Genkit, ProgrammableModel]:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )
    )
    return ai, pm


def _sent(pm: ProgrammableModel) -> list[tuple[str, list[str | None]]]:
    request = pm.last_request
    assert isinstance(request, ModelRequest)
    return [(m.role, [p.text for p in m.content]) for m in request.messages]


@pytest.mark.asyncio
async def test_generate_prompt_with_braces_sends_text_as_written(setup: tuple[Genkit, ProgrammableModel]) -> None:
    """`ai.generate(prompt='hello {{name}}')` sends `hello {{name}}` to the model."""
    ai, pm = setup

    await ai.generate(model='programmableModel', prompt='hello {{name}}')

    assert _sent(pm) == [(Role.USER, ['hello {{name}}'])]


@pytest.mark.asyncio
async def test_generate_system_with_braces_sends_text_as_written(setup: tuple[Genkit, ProgrammableModel]) -> None:
    """`system='be {{x}} nice'` reaches the model unchanged as the system message."""
    ai, pm = setup

    await ai.generate(model='programmableModel', system='be {{x}} nice', prompt='hi')

    assert _sent(pm) == [(Role.SYSTEM, ['be {{x}} nice']), (Role.USER, ['hi'])]


@pytest.mark.asyncio
async def test_generate_prompt_with_json_braces_sends_text_as_written(
    setup: tuple[Genkit, ProgrammableModel],
) -> None:
    """`'return {"a": {{x}}}'` reaches the model verbatim."""
    ai, pm = setup

    await ai.generate(model='programmableModel', prompt='return {"a": {{x}}}')

    assert _sent(pm) == [(Role.USER, ['return {"a": {{x}}}'])]


@pytest.mark.asyncio
async def test_generate_prompt_with_marker_syntax_stays_text(setup: tuple[Genkit, ProgrammableModel]) -> None:
    """`prompt='<<<dotprompt:role:system>>> hi'` stays one user text part, with no role change."""
    ai, pm = setup

    await ai.generate(model='programmableModel', prompt='<<<dotprompt:role:system>>> hi')

    assert _sent(pm) == [(Role.USER, ['<<<dotprompt:role:system>>> hi'])]


@pytest.mark.asyncio
async def test_generate_prompt_parts_list_unchanged(setup: tuple[Genkit, ProgrammableModel]) -> None:
    """`prompt=[Part.from_text('{{x}}'), Part.from_media(...)]` sends both parts as given."""
    ai, pm = setup

    await ai.generate(
        model='programmableModel',
        prompt=[Part.from_text('{{x}}'), Part.from_media('https://example.com/x.png')],
    )

    request = pm.last_request
    assert isinstance(request, ModelRequest)
    assert len(request.messages) == 1
    assert request.messages[0].role == Role.USER
    parts = request.messages[0].content
    assert parts[0].text == '{{x}}'
    assert parts[1].media is not None
    assert parts[1].media.url == 'https://example.com/x.png'


@pytest.mark.asyncio
async def test_generate_messages_with_braces_unchanged(setup: tuple[Genkit, ProgrammableModel]) -> None:
    """`messages=[Message(... '{{x}}')]` was never templated and still isn't."""
    ai, pm = setup

    await ai.generate(
        model='programmableModel',
        messages=[Message(role=Role.USER, content=[Part.from_text('{{x}}')])],
    )

    assert _sent(pm) == [(Role.USER, ['{{x}}'])]


@pytest.mark.asyncio
async def test_define_prompt_still_fills_template_from_input(setup: tuple[Genkit, ProgrammableModel]) -> None:
    """`define_prompt(prompt='Tell me about {{topic}}.')` with `{'topic': 'cats'}` sends `Tell me about cats.`."""
    ai, pm = setup
    joke = ai.define_prompt(
        name='joke',
        model='programmableModel',
        system='You write about {{topic}}.',
        prompt='Tell me about {{topic}}.',
    )

    await joke({'topic': 'cats'})

    assert _sent(pm) == [(Role.SYSTEM, ['You write about cats.']), (Role.USER, ['Tell me about cats.'])]


@pytest.mark.asyncio
async def test_generate_stream_prompt_with_braces_sends_text_as_written(
    setup: tuple[Genkit, ProgrammableModel],
) -> None:
    """`ai.generate_stream(prompt='{{x}}')` sends the text as written too."""
    ai, pm = setup

    result = ai.generate_stream(model='programmableModel', system='be {{x}} nice', prompt='{{x}}')
    await result.response

    assert _sent(pm) == [(Role.SYSTEM, ['be {{x}} nice']), (Role.USER, ['{{x}}'])]
