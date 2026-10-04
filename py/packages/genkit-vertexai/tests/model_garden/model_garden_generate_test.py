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

"""ai.generate against Model Garden models, with the provider client stubbed."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_vertexai.model_garden import ModelGarden
from openai.types.chat import ChatCompletion

from genkit import Genkit


def _chat_completion(text: str) -> ChatCompletion:
    return ChatCompletion.model_validate({
        'id': 'chatcmpl-1',
        'object': 'chat.completion',
        'created': 0,
        'model': 'meta/llama-3.1-405b-instruct-maas',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': text},
            }
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    })


@pytest.mark.asyncio
async def test_model_garden_openai_compatible_model_resolves_and_generates() -> None:
    """ai.generate on a catalog Llama model sends the prompt to the OpenAI client and returns its reply."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_chat_completion('hello from llama'))

    with patch(
        'genkit_vertexai.model_garden.model_garden.OpenAIClient.create',
        new=AsyncMock(return_value=client),
    ):
        ai = Genkit(plugins=[ModelGarden(project_id='my-project', location='us-central1')])
        response = await ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', prompt='hi')

    assert response.text == 'hello from llama'
    client.chat.completions.create.assert_awaited_once()
    sent = client.chat.completions.create.call_args.kwargs
    assert sent['model'] == 'meta/llama-3.1-405b-instruct-maas'
    assert sent['messages'] == [{'role': 'user', 'content': 'hi'}]


@pytest.mark.asyncio
async def test_model_garden_claude_model_still_generates() -> None:
    """ai.generate on a Model Garden Claude model sends the prompt to the Anthropic client and returns its reply."""
    anthropic_response = MagicMock()
    anthropic_response.content = [MagicMock(type='text', text='hello from claude')]
    anthropic_response.usage = MagicMock(input_tokens=1, output_tokens=1)
    anthropic_response.stop_reason = 'end_turn'
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=anthropic_response)

    with patch('genkit_vertexai.model_garden.anthropic.AsyncAnthropicVertex', return_value=client):
        ai = Genkit(plugins=[ModelGarden(project_id='my-project', location='us-east5')])
        response = await ai.generate(model='modelgarden/anthropic/claude-sonnet-4@20250514', prompt='hi')

    assert response.text == 'hello from claude'
    client.messages.create.assert_awaited_once()
    sent = client.messages.create.call_args.kwargs
    assert sent['model'] == 'claude-sonnet-4@20250514'
    assert sent['messages'][0]['role'] == 'user'
