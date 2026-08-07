# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Live tests against the real Anthropic API.

Requests go through a ``Genkit`` instance with the plugin registered, so
plugin resolution and the framework config path are exercised end to end.
Skipped unless ``ANTHROPIC_API_KEY`` is set.

Run from ``py/`` with:

    ANTHROPIC_API_KEY=your-key uv run pytest packages/genkit-anthropic/tests/anthropic_live_test.py -ra --no-cov
"""

import os
from typing import Any

import pytest
from anthropic import BadRequestError
from genkit_anthropic import Anthropic

from genkit import Genkit, GenkitError, Message, ReasoningPart

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get('ANTHROPIC_API_KEY'),
        reason='ANTHROPIC_API_KEY not found in the environment',
    ),
]

_ENABLED_THINKING_CONFIG: dict[str, Any] = {
    'thinking': {'enabled': True, 'budgetTokens': 1024},
    'maxOutputTokens': 2048,
}


@pytest.fixture
def ai() -> Genkit:
    """Genkit instance with the Anthropic plugin registered."""
    return Genkit(plugins=[Anthropic()])


def _reasoning_of(message: Message) -> list[ReasoningPart]:
    return [part.root for part in message.content if isinstance(part.root, ReasoningPart)]


async def test_thinking_enabled_budget(ai: Genkit) -> None:
    """A manual thinking budget returns reasoning with a signature."""
    response = await ai.generate(
        model='anthropic/claude-haiku-4-5',
        prompt='What is 15 + 27? Think it through, then answer with just the number.',
        config=_ENABLED_THINKING_CONFIG,
    )

    assert response.message is not None
    assert response.text.strip()
    reasoning = _reasoning_of(response.message)
    assert ''.join(part.reasoning for part in reasoning)
    assert any(part.metadata and part.metadata.get('thoughtSignature') for part in reasoning)


async def test_thinking_enabled_budget_streaming(ai: Genkit) -> None:
    """Thinking deltas stream as reasoning chunks and match the final reasoning."""
    stream_response = ai.generate_stream(
        model='anthropic/claude-haiku-4-5',
        prompt='What is 12 * 12? Think it through, then answer with just the number.',
        config=_ENABLED_THINKING_CONFIG,
    )

    streamed_reasoning: list[str] = []
    streamed_text: list[str] = []
    async for chunk in stream_response.stream:
        for part in chunk.content:
            if isinstance(part.root, ReasoningPart):
                streamed_reasoning.append(part.root.reasoning)
        if chunk.text:
            streamed_text.append(chunk.text)
    response = await stream_response.response

    assert ''.join(streamed_reasoning)
    assert ''.join(streamed_text).strip()

    assert response.message is not None
    final_reasoning = ''.join(part.reasoning for part in _reasoning_of(response.message))
    assert ''.join(streamed_reasoning) == final_reasoning
    assert ''.join(streamed_text) == response.text
    assert response.usage is not None
    assert (response.usage.output_tokens or 0) > 0


async def test_thinking_adaptive(ai: Genkit) -> None:
    """Adaptive thinking with display is accepted by Opus 4.7+ models."""
    response = await ai.generate(
        model='anthropic/claude-opus-4-8',
        prompt='Write a one-sentence story about a robot.',
        config={'thinking': {'adaptive': True, 'display': 'summarized'}},
    )

    assert response.message is not None
    assert response.text.strip()


async def test_thinking_disabled(ai: Genkit) -> None:
    """Disabled thinking is accepted and yields no reasoning parts."""
    response = await ai.generate(
        model='anthropic/claude-haiku-4-5',
        prompt='What is 2 + 2? Answer with just the number.',
        config={'thinking': {'enabled': False}},
    )

    assert response.message is not None
    assert response.text.strip()
    assert not _reasoning_of(response.message)


async def test_thinking_budget_rejected_on_adaptive_only_model(ai: Genkit) -> None:
    """Models that only support adaptive thinking reject a manual budget with a 400."""
    with pytest.raises(GenkitError) as excinfo:
        await ai.generate(
            model='anthropic/claude-opus-4-8',
            prompt='What is 2 + 2? Answer with just the number.',
            config=_ENABLED_THINKING_CONFIG,
        )

    cause: BaseException | None = excinfo.value
    while isinstance(cause, GenkitError):
        cause = cause.cause
    assert isinstance(cause, BadRequestError)
