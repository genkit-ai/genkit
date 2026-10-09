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

"""What ai.generate sends to OpenAI as tool_choice, on the first turn and after a tool turn."""

import json
from typing import Any

import httpx
import pytest
from genkit_openai import OpenAI

from genkit import Genkit


def _completion(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    return {
        'id': 'chatcmpl-1',
        'object': 'chat.completion',
        'created': 0,
        'model': 'gpt-4o',
        'choices': [{'index': 0, 'finish_reason': finish_reason, 'message': message}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }


def _tool_call(call_id: str, name: str, arguments: object) -> dict[str, Any]:
    return _completion(
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}
            ],
        },
        'tool_calls',
    )


class _ScriptedServer:
    """A fake OpenAI endpoint that plays scripted replies in order and records each body."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self._replies = replies
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=self._replies[len(self.bodies) - 1])


_ANSWER = _completion({'role': 'assistant', 'content': 'The pho is in stock and nut-free.'}, 'stop')


def _genkit(server: _ScriptedServer) -> Genkit:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    ai = Genkit(plugins=[OpenAI(api_key='sk-plugin', http_client=http_client, max_retries=0)])

    @ai.tool(name='lookup_menu')
    async def lookup_menu(dish: str) -> str:
        return f'{dish}: in stock'

    @ai.tool(name='check_allergens')
    async def check_allergens(dish: str) -> str:
        return f'{dish}: nut-free'

    return ai


@pytest.mark.asyncio
@pytest.mark.parametrize('tool_choice', ['auto', 'required', 'none', None])
async def test_tool_choice_sent_as_is_with_tools(tool_choice: Any) -> None:  # noqa: ANN401
    """auto, required and none go out unchanged; None sends no tool_choice."""
    server = _ScriptedServer([_ANSWER])
    ai = _genkit(server)

    await ai.generate(
        model='openai/gpt-4o', prompt='Is the pho in stock?', tools=['lookup_menu'], tool_choice=tool_choice
    )

    body = server.bodies[0]
    assert [t['function']['name'] for t in body['tools']] == ['lookup_menu']
    if tool_choice is None:
        assert 'tool_choice' not in body
    else:
        assert body['tool_choice'] == tool_choice


@pytest.mark.asyncio
@pytest.mark.parametrize('tool_choice', ['required', None])
async def test_tool_choice_kept_after_a_tool_turn(tool_choice: Any) -> None:  # noqa: ANN401
    """A tool result no longer forces tool_choice='none', so the model can chain a second tool."""
    # 1. Script a chain: lookup_menu, then check_allergens, then the answer
    server = _ScriptedServer([
        _tool_call('call_1', 'lookup_menu', 'pho'),
        _tool_call('call_2', 'check_allergens', 'pho'),
        _ANSWER,
    ])
    ai = _genkit(server)

    # 2. Generate with both tools
    response = await ai.generate(
        model='openai/gpt-4o',
        prompt='Is the pho in stock, and is it nut-free?',
        tools=['lookup_menu', 'check_allergens'],
        tool_choice=tool_choice,
    )

    # 3. Each turn after a tool result carries the caller's tool_choice, not 'none'
    assert response.text == 'The pho is in stock and nut-free.'
    assert len(server.bodies) == 3
    for body in server.bodies[1:]:
        assert any(m['role'] == 'tool' for m in body['messages'])
        assert body.get('tool_choice') == tool_choice


@pytest.mark.asyncio
@pytest.mark.parametrize('tool_choice', ['auto', 'required', 'none'])
async def test_tool_choice_dropped_without_tools(tool_choice: str) -> None:
    """OpenAI rejects tool_choice without tools, so a request with no tools sends neither."""
    server = _ScriptedServer([_ANSWER])
    ai = _genkit(server)

    await ai.generate(model='openai/gpt-4o', prompt='Suggest a dish.', tool_choice=tool_choice)

    assert 'tools' not in server.bodies[0]
    assert 'tool_choice' not in server.bodies[0]
