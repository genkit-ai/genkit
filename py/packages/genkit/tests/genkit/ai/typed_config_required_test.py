#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A required field on the annotation's config class is checked after every layer merges."""

from typing import Literal

import pytest
from pydantic import BaseModel

from genkit import Genkit, Part
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import Message, ModelRequest, ModelResponse
from genkit._core._typing import Role


class VoiceConfig(BaseModel):
    """A text-to-speech config whose voice has no default."""

    voice: Literal['alloy', 'echo']
    speed: float = 1.0


def define_tts(ai: Genkit) -> list[object]:
    """Register a TTS model typed with VoiceConfig; return the configs it saw."""
    configs: list[object] = []

    async def tts(request: ModelRequest[VoiceConfig], ctx: ActionRunContext) -> ModelResponse:
        configs.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('audio')]))

    ai.define_model('acme/tts', tts)
    return configs


@pytest.mark.asyncio
async def test_missing_required_field_raises_before_the_turn() -> None:
    """Leaving out a field with no default raises INVALID_ARGUMENT; the model never runs."""
    ai = Genkit()
    configs = define_tts(ai)

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model='acme/tts', prompt='Your table is ready.', config={'speed': 1.2})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert "config 'voice'" in str(exc_info.value)
    assert configs == []


@pytest.mark.asyncio
async def test_omitted_config_with_required_field_raises_before_the_turn() -> None:
    """No config at all is still checked against the required field."""
    ai = Genkit()
    configs = define_tts(ai)

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model='acme/tts', prompt='Your table is ready.')

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert configs == []


@pytest.mark.asyncio
async def test_required_field_from_prompt_config_satisfies_call_override() -> None:
    """The prompt supplies voice, the call overrides speed: the merged bag is complete."""
    ai = Genkit()
    configs = define_tts(ai)
    announce = ai.define_prompt(model='acme/tts', prompt='Your table is ready.', config={'voice': 'echo'})

    await announce(config={'speed': 1.2})

    assert configs == [VoiceConfig(voice='echo', speed=1.2)]


@pytest.mark.asyncio
async def test_call_override_clearing_required_field_raises() -> None:
    """A call that clears the prompt's voice with None leaves it missing."""
    ai = Genkit()
    configs = define_tts(ai)
    announce = ai.define_prompt(model='acme/tts', prompt='Your table is ready.', config={'voice': 'echo'})

    with pytest.raises(GenkitError) as exc_info:
        await announce(config={'voice': None})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert configs == []
