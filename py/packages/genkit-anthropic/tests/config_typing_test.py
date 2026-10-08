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

"""Typed construction of the flat AnthropicConfig, imported from the package root.

pyright, pyrefly and ty must accept these snake_case kwargs with no
suppressions, and the assert_type calls pin the exported Literals. The
runtime asserts pin the config dump and the Messages request body.
"""

import json
from typing import Any

import httpx
import pytest
from genkit_anthropic import Anthropic, AnthropicConfig, Effort, ThinkingDisplay, ThinkingMode
from typing_extensions import assert_type

from genkit import Genkit


def test_anthropic_config_snake_case_kwargs() -> None:
    """Snake_case kwargs type-check and dump camelCase."""
    config = AnthropicConfig(
        temperature=0.5,
        max_output_tokens=2000,
        thinking='enabled',
        thinking_budget=2048,
        thinking_display='summarized',
        effort='high',
        task_budget=20000,
        disable_parallel_tool_use=True,
        user_id='diner-42',
        api_version='beta',
        betas=['extended-cache-ttl-2025-04-11'],
    )

    assert config.model_dump(by_alias=True, exclude_none=True, mode='json') == {
        'temperature': 0.5,
        'maxOutputTokens': 2000,
        'thinking': 'enabled',
        'thinkingBudget': 2048,
        'thinkingDisplay': 'summarized',
        'effort': 'high',
        'taskBudget': 20000,
        'disableParallelToolUse': True,
        'userId': 'diner-42',
        'apiVersion': 'beta',
        'betas': ['extended-cache-ttl-2025-04-11'],
    }


def test_choice_fields_are_the_exported_literals() -> None:
    """The choice fields read back as the root-exported Literals."""
    mode: ThinkingMode = 'adaptive'
    display: ThinkingDisplay = 'omitted'
    effort: Effort = 'max'
    config = AnthropicConfig(thinking=mode, thinking_display=display, effort=effort)

    assert_type(config.thinking, ThinkingMode | None)
    assert_type(config.thinking_display, ThinkingDisplay | None)
    assert_type(config.effort, Effort | None)
    assert (config.thinking, config.thinking_display, config.effort) == ('adaptive', 'omitted', 'max')


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
async def test_typed_config_request_body() -> None:
    """The plugin builds Anthropic's nested thinking, output_config and metadata objects from the flat fields."""
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

    # 2. Generate with a typed flat config
    config = AnthropicConfig(
        max_output_tokens=3000,
        thinking_budget=2048,
        effort='medium',
        task_budget=20000,
        disable_parallel_tool_use=True,
        user_id='diner-42',
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
    assert body['output_config'] == {'effort': 'medium', 'task_budget': {'type': 'tokens', 'total': 20000}}
    assert body['tool_choice'] == {'type': 'any', 'disable_parallel_tool_use': True}
    assert body['metadata'] == {'user_id': 'diner-42'}
    assert body['max_tokens'] == 3000


@pytest.mark.asyncio
async def test_extra_replaces_the_built_thinking_object_on_the_wire() -> None:
    """extra={'thinking': {...}} goes out whole instead of the object built from the flat fields."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _claude_reply(request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ai = Genkit(plugins=[Anthropic(api_key='fake-key', http_client=http_client)])

    await ai.generate(
        model='anthropic/claude-sonnet-4-5',
        prompt='Plan a nut-free tasting menu.',
        config=AnthropicConfig(
            thinking_budget=2048,
            user_id='diner-42',
            extra={'thinking': {'type': 'enabled', 'budget_tokens': 4096}, 'metadata': {'user_id': 'table-7'}},
        ),
    )

    assert bodies[-1]['thinking'] == {'type': 'enabled', 'budget_tokens': 4096}
    assert bodies[-1]['metadata'] == {'user_id': 'table-7'}
