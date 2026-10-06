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

"""What a Claude call sends over the wire, driven through ``ai.generate``.

Each test runs the real Anthropic SDK against a fake HTTP transport and checks
the request body and headers the API would have received.
"""

import json
from typing import Any, cast

import httpx
import pytest
from genkit_anthropic import Anthropic

from genkit import FinishReason, Genkit
from genkit.plugin_api import ActionKind

MODEL = 'anthropic/claude-sonnet-4-6'
TENANT_KEY = 'tenant-key'
PLUGIN_KEY = 'plugin-key'


def _message(*, text: str = 'ok', stop_reason: str = 'end_turn') -> dict[str, Any]:
    return {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-6',
        'content': [{'type': 'text', 'text': text}],
        'stop_reason': stop_reason,
        'stop_sequence': None,
        'usage': {'input_tokens': 1, 'output_tokens': 1},
    }


def _sse(*, text: str, stop_reason: str) -> str:
    start = {**_message(text='', stop_reason=stop_reason), 'content': [], 'stop_reason': None}
    events = [
        ('message_start', {'type': 'message_start', 'message': start}),
        (
            'content_block_start',
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        ),
        (
            'content_block_delta',
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}},
        ),
        ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
        (
            'message_delta',
            {
                'type': 'message_delta',
                'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
                'usage': {'output_tokens': 1},
            },
        ),
        ('message_stop', {'type': 'message_stop'}),
    ]
    return ''.join(f'event: {name}\ndata: {json.dumps(data)}\n\n' for name, data in events)


class FakeClaudeApi:
    """Records every request and answers like the Messages API."""

    def __init__(self, *, text: str = 'ok', stop_reason: str = 'end_turn') -> None:
        self.requests: list[httpx.Request] = []
        self._text = text
        self._stop_reason = stop_reason

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.body(-1).get('stream'):
            return httpx.Response(
                200,
                text=_sse(text=self._text, stop_reason=self._stop_reason),
                headers={'content-type': 'text/event-stream'},
            )
        return httpx.Response(200, json=_message(text=self._text, stop_reason=self._stop_reason))

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    def api_key(self, index: int = -1) -> str | None:
        return self.requests[index].headers.get('x-api-key')


def _genkit(api: FakeClaudeApi, **client_params: Any) -> Genkit:
    client_params.setdefault('api_key', PLUGIN_KEY)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(api.handler))
    return Genkit(plugins=[Anthropic(http_client=http_client, **client_params)])


# --- config keys --------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_claude_extra_is_sent_as_extra_body() -> None:
    """`extra={'service_tier': 'auto'}` lands at the top level of the request body."""
    api = FakeClaudeApi()
    ai = _genkit(api)

    response = await ai.generate(
        model=MODEL, prompt='hi', config={'temperature': 0.2, 'extra': {'service_tier': 'auto'}}
    )

    assert response.text == 'ok'
    body = api.body()
    assert body['service_tier'] == 'auto'
    assert body['temperature'] == 0.2
    assert 'extra' not in body


@pytest.mark.parametrize(
    ('config', 'field', 'sent'),
    [
        ({'max_output_tokens': 50, 'extra': {'max_tokens': 100}}, 'max_tokens', 100),
        (
            {
                'thinking': {'enabled': True, 'budgetTokens': 2048, 'display': 'summarized'},
                'extra': {'thinking': {'type': 'enabled', 'budget_tokens': 4096}},
            },
            'thinking',
            {'type': 'enabled', 'budget_tokens': 4096},
        ),
    ],
    ids=['scalar', 'nested-replaces-whole-setting'],
)
@pytest.mark.asyncio
async def test_generate_claude_extra_key_wins_over_declared_setting(
    config: dict[str, Any], field: str, sent: object
) -> None:
    """A key in `extra` replaces the declared setting it collides with; nested values aren't merged."""
    api = FakeClaudeApi()
    ai = _genkit(api)

    await ai.generate(model=MODEL, prompt='hi', config=config)

    assert api.body()[field] == sent


@pytest.mark.asyncio
async def test_anthropic_model_advertises_config_without_additional_properties() -> None:
    """The Dev UI config schema says `additionalProperties: false`, at the top and inside `thinking`."""
    action = await Anthropic(api_key=PLUGIN_KEY).resolve(ActionKind.MODEL, MODEL)

    assert action is not None
    options = cast(dict[str, Any], action.metadata['model'])['customOptions']
    assert options['additionalProperties'] is False
    assert options['properties']['thinking']['additionalProperties'] is False
    assert 'apiKey' not in options['properties']


# --- per-request key ----------------------------------------------------------


@pytest.fixture
def no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_AUTH_TOKEN', raising=False)


@pytest.mark.asyncio
@pytest.mark.usefixtures('no_env_key')
async def test_plugin_without_key_uses_tenant_key_from_context() -> None:
    """`Anthropic()` with no key and no ANTHROPIC_API_KEY, plus a secrets key, runs the call on that key."""
    api = FakeClaudeApi()
    ai = _genkit(api, api_key=None)

    response = await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': TENANT_KEY}})

    assert response.text == 'ok'
    assert api.api_key() == TENANT_KEY


@pytest.mark.asyncio
@pytest.mark.usefixtures('no_env_key')
async def test_plugin_without_key_and_no_tenant_key_raises_naming_both() -> None:
    """No plugin key and no secrets key fails FAILED_PRECONDITION naming both places a key can go; nothing is sent."""
    api = FakeClaudeApi()
    ai = _genkit(api, api_key=None)

    response = await ai.generate(model=MODEL, prompt='hi')

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert 'ANTHROPIC_API_KEY' in response.error.message
    assert 'Anthropic(api_key=...)' in response.error.message
    assert "context={'secrets': {'api_key': ...}}" in response.error.message
    assert api.requests == []


@pytest.mark.parametrize('streaming', [False, True], ids=['generate', 'generate_stream'])
@pytest.mark.asyncio
async def test_tenant_key_wins_over_plugin_key(streaming: bool) -> None:
    """`context={'secrets': {'api_key': k}}` sends `x-api-key: k` instead of the plugin's key."""
    api = FakeClaudeApi()
    ai = _genkit(api)
    context: dict[str, object] = {'secrets': {'api_key': TENANT_KEY}}

    if streaming:
        response = await ai.generate_stream(model=MODEL, prompt='hi', context=context).response
    else:
        response = await ai.generate(model=MODEL, prompt='hi', context=context)

    assert response.text == 'ok'
    assert api.api_key() == TENANT_KEY


@pytest.mark.asyncio
async def test_generate_claude_secrets_key_does_not_leak_to_next_call() -> None:
    """A call with no secrets after a tenant call uses the plugin key again."""
    api = FakeClaudeApi()
    ai = _genkit(api)

    await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': TENANT_KEY}})
    await ai.generate(model=MODEL, prompt='hi')

    assert [api.api_key(0), api.api_key(1)] == [TENANT_KEY, PLUGIN_KEY]


@pytest.mark.asyncio
async def test_generate_claude_secrets_without_api_key_uses_plugin_key() -> None:
    """`context={'secrets': {'crm_token': ...}}` holds other app secrets, so the call runs on the plugin key."""
    api = FakeClaudeApi()
    ai = _genkit(api)

    response = await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'crm_token': 'crm-secret'}})

    assert response.text == 'ok'
    assert api.api_key() == PLUGIN_KEY


@pytest.mark.parametrize(
    'secrets',
    [
        {'api_key': ''},
        {'api_key': '   '},
        {'apiKey': ''},
        {'apiKey': '   '},
        {'api_key': '', 'apiKey': TENANT_KEY},
    ],
    ids=['api_key-empty', 'api_key-whitespace', 'apiKey-empty', 'apiKey-whitespace', 'blank-api_key-shadows-apiKey'],
)
@pytest.mark.parametrize('plugin_key', [PLUGIN_KEY, None], ids=['plugin-key', 'no-plugin-key'])
@pytest.mark.asyncio
@pytest.mark.usefixtures('no_env_key')
async def test_generate_claude_blank_secrets_key_fails_instead_of_using_plugin_key(
    secrets: dict[str, str], plugin_key: str | None
) -> None:
    """A set but blank secrets key fails INVALID_ARGUMENT instead of running on the plugin key; nothing is sent."""
    api = FakeClaudeApi()
    ai = _genkit(api, api_key=plugin_key)

    response = await ai.generate(model=MODEL, prompt='hi', context={'secrets': secrets})

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert 'is blank' in response.error.message
    assert "context={'secrets': {'api_key': ...}}" in response.error.message
    assert TENANT_KEY not in response.error.message
    assert api.requests == []


@pytest.mark.parametrize(
    ('client_params', 'reason'),
    [
        ({'api_key': None, 'auth_token': 'corp-bearer'}, 'auth token'),
        ({'default_headers': {'X-Api-Key': 'pinned'}}, 'x-api-key'),
    ],
    ids=['auth-token-client', 'fixed-x-api-key-header-client'],
)
@pytest.mark.asyncio
async def test_generate_claude_secrets_key_on_client_that_cannot_swap_key_fails(
    client_params: dict[str, Any], reason: str
) -> None:
    """A plugin client whose credential can't be swapped fails the tenant call instead of billing its own key."""
    api = FakeClaudeApi()
    ai = _genkit(api, **client_params)

    response = await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': TENANT_KEY}})

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert reason in response.error.message
    assert TENANT_KEY not in response.error.message
    assert api.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'tool_choice',
    [
        {'type': 'auto', 'disable_parallel_tool_use': True},
        {'type': 'any', 'disable_parallel_tool_use': True},
        {'type': 'tool', 'name': 'lookup_menu', 'disable_parallel_tool_use': True},
    ],
    ids=['auto', 'any', 'tool'],
)
async def test_generate_claude_tool_choice_disable_parallel_tool_use_reaches_request(
    tool_choice: dict[str, Any],
) -> None:
    """`tool_choice.disable_parallel_tool_use` is a Claude field, so it validates and goes out as written."""
    api = FakeClaudeApi()
    ai = _genkit(api)

    @ai.tool(name='lookup_menu')
    async def lookup_menu(dish: str) -> str:
        return 'ramen'

    response = await ai.generate(model=MODEL, prompt='hi', tools=['lookup_menu'], config={'tool_choice': tool_choice})

    assert response.text == 'ok'
    assert api.body()['tool_choice'] == tool_choice
