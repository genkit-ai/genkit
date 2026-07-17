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

Skipped unless ``ANTHROPIC_API_KEY`` is set.

Run from ``py/`` with:

    ANTHROPIC_API_KEY=your-key uv run pytest packages/genkit-anthropic/tests/anthropic_live_test.py -ra --no-cov
"""

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from anthropic import AsyncAnthropic
from genkit_anthropic.models import AnthropicModel

from genkit import (
    Message,
    ModelRequest,
    ModelResponseChunk,
    Part,
    ReasoningPart,
    Role,
    TextPart,
)

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


def _request(prompt: str, config: Any) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text=prompt))])],
        config=config,
    )


def _text_of(message: Message) -> str:
    return ''.join(part.root.text for part in message.content if isinstance(part.root, TextPart))


def _reasoning_of(message: Message) -> list[ReasoningPart]:
    return [part.root for part in message.content if isinstance(part.root, ReasoningPart)]


async def test_thinking_enabled_budget() -> None:
    """A manual thinking budget returns reasoning with a signature."""
    model = AnthropicModel(model_name='claude-haiku-4-5', client=AsyncAnthropic())

    response = await model.generate(
        _request(
            'What is 15 + 27? Think it through, then answer with just the number.',
            _ENABLED_THINKING_CONFIG,
        )
    )

    assert response.message is not None
    assert _text_of(response.message).strip()
    reasoning = _reasoning_of(response.message)
    assert ''.join(part.reasoning for part in reasoning)
    assert any(part.metadata and part.metadata.get('thoughtSignature') for part in reasoning)


async def test_thinking_enabled_budget_streaming() -> None:
    """Thinking deltas stream as reasoning chunks and match the final reasoning."""
    model = AnthropicModel(model_name='claude-haiku-4-5', client=AsyncAnthropic())
    ctx = MagicMock()
    ctx.is_streaming = True
    chunks: list[ModelResponseChunk] = []
    ctx.send_chunk = chunks.append

    response = await model.generate(
        _request(
            'What is 12 * 12? Think it through, then answer with just the number.',
            _ENABLED_THINKING_CONFIG,
        ),
        ctx,
    )

    streamed_reasoning = ''.join(
        part.root.reasoning for chunk in chunks for part in chunk.content if isinstance(part.root, ReasoningPart)
    )
    streamed_text = ''.join(
        part.root.text for chunk in chunks for part in chunk.content if isinstance(part.root, TextPart)
    )
    assert streamed_reasoning
    assert streamed_text.strip()

    assert response.message is not None
    final_reasoning = ''.join(part.reasoning for part in _reasoning_of(response.message))
    assert streamed_reasoning == final_reasoning
    assert streamed_text == _text_of(response.message)
    assert response.usage is not None
    assert (response.usage.output_tokens or 0) > 0


async def test_thinking_adaptive() -> None:
    """Adaptive thinking with display is accepted by Opus 4.7+ models."""
    model = AnthropicModel(model_name='claude-opus-4-8', client=AsyncAnthropic())

    response = await model.generate(
        _request(
            'Write a one-sentence story about a robot.',
            {'thinking': {'adaptive': True, 'display': 'summarized'}},
        )
    )

    assert response.message is not None
    assert _text_of(response.message).strip()


async def test_thinking_disabled() -> None:
    """Disabled thinking is accepted and yields no reasoning parts."""
    model = AnthropicModel(model_name='claude-haiku-4-5', client=AsyncAnthropic())

    response = await model.generate(
        _request('What is 2 + 2? Answer with just the number.', {'thinking': {'enabled': False}})
    )

    assert response.message is not None
    assert _text_of(response.message).strip()
    assert not _reasoning_of(response.message)
