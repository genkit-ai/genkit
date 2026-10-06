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

"""What a Bedrock thinking part carries, and what a saved chat sends back next turn."""

from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from genkit_amazon_bedrock import Bedrock
from genkit_amazon_bedrock.transport import BedrockTransport

from genkit import Genkit, Message, Part, Role

CLAUDE = 'bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0'


class FakeTransport:
    """Stands in for BedrockTransport; records what would go on the wire."""

    def __init__(
        self,
        content: list[dict[str, Any]] | None = None,
        stream_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.content = content or [{'text': 'ok'}]
        self.stream_events = stream_events or []
        self.kwargs: dict[str, Any] | None = None

    async def ensure_client(self) -> None:
        return None

    async def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return {
            'output': {'message': {'role': 'assistant', 'content': self.content}},
            'stopReason': 'end_turn',
            'usage': {'inputTokens': 1, 'outputTokens': 1, 'totalTokens': 2},
        }

    async def converse_stream(self, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        self.kwargs = kwargs
        for event in self.stream_events:
            yield event


def _app(transport: FakeTransport) -> Genkit:
    plugin = Bedrock(region='us-east-1')
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001
    return Genkit(plugins=[plugin])


def _saved_chat(reasoning: Part) -> list[Message]:
    return [
        Message(role=Role.USER, content=[Part.from_text('what is 17 * 23?')]),
        Message(role=Role.MODEL, content=[reasoning, Part.from_text('391')]),
    ]


def _sent_model_turn(transport: FakeTransport) -> list[dict[str, Any]]:
    assert transport.kwargs is not None
    return transport.kwargs['messages'][1]['content']


@pytest.mark.asyncio
async def test_generate_bedrock_reasoning_part_has_only_thought_signature() -> None:
    """A Bedrock reasoning reply's part metadata is exactly ``{'thoughtSignature': 'sig'}``."""
    transport = FakeTransport(
        content=[
            {'reasoningContent': {'reasoningText': {'text': 'because', 'signature': 'sig'}}},
            {'text': '391'},
        ]
    )
    ai = _app(transport)

    response = await ai.generate(model=CLAUDE, prompt='what is 17 * 23?')

    assert response.message is not None
    reasoning, answer = response.message.content
    assert reasoning.reasoning == 'because'
    assert reasoning.metadata == {'thoughtSignature': 'sig'}
    assert answer.text == '391'


@pytest.mark.asyncio
async def test_generate_stream_bedrock_reasoning_part_has_only_thought_signature() -> None:
    """The streamed final message's reasoning part carries only ``thoughtSignature``."""
    transport = FakeTransport(
        stream_events=[
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'reasoningContent': {'text': 'because'}}}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'reasoningContent': {'signature': 'sig'}}}},
            {'contentBlockDelta': {'contentBlockIndex': 1, 'delta': {'text': '391'}}},
            {'messageStop': {'stopReason': 'end_turn'}},
        ]
    )
    ai = _app(transport)

    stream = ai.generate_stream(model=CLAUDE, prompt='what is 17 * 23?')
    async for _ in stream.stream:
        pass
    response = await stream.response

    assert response.message is not None
    reasoning, answer = response.message.content
    assert reasoning.reasoning == 'because'
    assert reasoning.metadata == {'thoughtSignature': 'sig'}
    assert answer.text == '391'


@pytest.mark.asyncio
async def test_generate_bedrock_replays_thought_signature_on_next_turn() -> None:
    """Sending a saved reasoning part back puts its ``thoughtSignature`` in the Converse reasoning block."""
    transport = FakeTransport()
    ai = _app(transport)

    await ai.generate(
        model=CLAUDE,
        messages=_saved_chat(Part.from_reasoning('because', metadata={'thoughtSignature': 'sig'})),
        prompt='now add 100',
    )

    assert _sent_model_turn(transport) == [
        {'reasoningContent': {'reasoningText': {'text': 'because', 'signature': 'sig'}}},
        {'text': '391'},
    ]


@pytest.mark.asyncio
async def test_generate_bedrock_reasoning_part_saved_before_1_0_is_sent_without_reasoning() -> None:
    """A part saved with the old ``bedrockReasoningSignature``/``signature`` keys sends no reasoning block."""
    transport = FakeTransport()
    ai = _app(transport)

    old_part = Part.from_reasoning('because', metadata={'bedrockReasoningSignature': 'sig', 'signature': 'sig'})
    await ai.generate(model=CLAUDE, messages=_saved_chat(old_part), prompt='now add 100')

    assert _sent_model_turn(transport) == [{'text': '391'}]
