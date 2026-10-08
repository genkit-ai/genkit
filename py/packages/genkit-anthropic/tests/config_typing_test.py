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

"""Typed construction of AnthropicConfig and its nested settings, imported from the package root.

pyright, pyrefly and ty must accept these snake_case kwargs with no
suppressions. The runtime asserts pin the config dump and the Messages
request body.
"""

import json
from typing import Any

import httpx
import pytest
from genkit_anthropic import (
    Anthropic,
    AnthropicConfig,
    OutputConfig,
    RequestMetadata,
    TaskBudget,
    ThinkingConfig,
)

from genkit import Genkit


def test_anthropic_config_snake_case_kwargs() -> None:
    """Snake_case kwargs type-check; inherited fields dump camelCase, Anthropic ones keep their wire names."""
    config = AnthropicConfig(
        temperature=0.5,
        max_output_tokens=2000,
        thinking=ThinkingConfig(enabled=True, budget_tokens=2048),
        output_config=OutputConfig(effort='high', task_budget=TaskBudget(total=20000)),
        disable_parallel_tool_use=True,
        metadata=RequestMetadata(user_id='diner-42'),
        api_version='beta',
        betas=['extended-cache-ttl-2025-04-11'],
    )

    assert config.model_dump(by_alias=True, exclude_none=True, mode='json') == {
        'temperature': 0.5,
        'maxOutputTokens': 2000,
        'thinking': {'enabled': True, 'budgetTokens': 2048},
        'output_config': {'effort': 'high', 'task_budget': {'type': 'tokens', 'total': 20000}},
        'disableParallelToolUse': True,
        'metadata': {'user_id': 'diner-42'},
        'apiVersion': 'beta',
        'betas': ['extended-cache-ttl-2025-04-11'],
    }


def test_thinking_config_accepts_both_spellings() -> None:
    """budget_tokens= and the budgetTokens wire key build the same ThinkingConfig."""
    typed = ThinkingConfig(enabled=True, budget_tokens=2048)

    assert ThinkingConfig.model_validate({'enabled': True, 'budgetTokens': 2048}) == typed
    assert ThinkingConfig.model_validate({'enabled': True, 'budget_tokens': 2048}) == typed


def _claude_reply(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            'id': 'msg_1',
            'type': 'message',
            'role': 'assistant',
            'model': 'claude-sonnet-4-5',
            'content': [{'type': 'text', 'text': 'ok'}],
            'stop_reason': 'end_turn',
            'stop_sequence': None,
            'usage': {'input_tokens': 1, 'output_tokens': 1},
        },
        request=request,
    )


@pytest.mark.asyncio
async def test_typed_nested_config_request_body() -> None:
    """A typed nested AnthropicConfig reaches the Messages body with Anthropic's snake_case keys."""
    # 1. Point the plugin at a fake Messages API that records each body
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _claude_reply(request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ai = Genkit(plugins=[Anthropic(api_key='fake-key', http_client=http_client)])

    @ai.tool(name='lookup_menu')
    async def lookup_menu(dish: str) -> str:
        return f'{dish}: in stock'

    # 2. Generate with a config built from typed nested models
    config = AnthropicConfig(
        max_output_tokens=3000,
        thinking=ThinkingConfig(type='enabled', budget_tokens=2048),
        output_config=OutputConfig(task_budget=TaskBudget(total=20000)),
        disable_parallel_tool_use=True,
        metadata=RequestMetadata(user_id='diner-42'),
    )
    await ai.generate(
        model='anthropic/claude-sonnet-4-5',
        prompt='Is the pho in stock?',
        config=config,
        tools=['lookup_menu'],
        tool_choice='required',
    )

    # 3. Check the body the plugin sent
    body = bodies[-1]
    assert body['thinking'] == {'type': 'enabled', 'budget_tokens': 2048}
    assert body['output_config'] == {'task_budget': {'type': 'tokens', 'total': 20000}}
    assert body['tool_choice'] == {'type': 'any', 'disable_parallel_tool_use': True}
    assert body['metadata'] == {'user_id': 'diner-42'}
    assert body['max_tokens'] == 3000
