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

"""What ai.generate sends to OpenAI: config keys, extra, the length cap, and whose key."""

import asyncio
import base64
import json
from typing import Any, cast

import httpx
import pytest
from genkit_openai import OpenAI, OpenAIConfig
from genkit_openai._models import OpenAIModel
from openai import AsyncOpenAI

from genkit import ActionRunContext, FinishReason, Genkit, GenkitError, Message, Part, Role
from genkit.model import ModelConfig, ModelRequest
from genkit.plugin_api import ActionKind

PLUGIN_KEY = 'sk-plugin'


def _completion(text: str = 'hi back') -> dict[str, Any]:
    return {
        'id': 'chatcmpl-1',
        'object': 'chat.completion',
        'created': 0,
        'model': 'gpt-4o',
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': text}}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }


def _stream_body(text: str = 'hi back') -> bytes:
    chunk = {
        'id': 'chatcmpl-1',
        'object': 'chat.completion.chunk',
        'created': 0,
        'model': 'gpt-4o',
        'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
    }
    return f'data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n'.encode()


class _OpenAIServer:
    """A fake OpenAI endpoint that records every request it gets."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith('/images/generations'):
            return httpx.Response(200, json={'created': 0, 'data': [{'b64_json': 'aW1n'}]})
        if path.endswith('/audio/speech'):
            return httpx.Response(200, content=b'mp3-bytes', headers={'content-type': 'audio/mpeg'})
        if path.endswith('/audio/transcriptions'):
            return httpx.Response(200, text='heard you', headers={'content-type': 'text/plain'})
        body = json.loads(request.content)
        if body.get('stream'):
            return httpx.Response(200, content=_stream_body(), headers={'content-type': 'text/event-stream'})
        return httpx.Response(200, json=_completion())

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def keys(self) -> list[str]:
        return [r.headers['authorization'] for r in self.requests]


@pytest.fixture
def server() -> _OpenAIServer:
    return _OpenAIServer()


@pytest.fixture
def plugin(server: _OpenAIServer) -> OpenAI:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return OpenAI(api_key=PLUGIN_KEY, http_client=http_client, max_retries=0)


@pytest.fixture
def ai(plugin: OpenAI) -> Genkit:
    return Genkit(plugins=[plugin])


# Unknown keys and extra


@pytest.mark.asyncio
async def test_generate_openai_unknown_config_key_raises_naming_it(ai: Genkit, server: _OpenAIServer) -> None:
    """`config={'temprature': 0.2}` raises INVALID_ARGUMENT naming 'temprature', and nothing is sent."""
    with pytest.raises(GenkitError) as raised:
        await ai.generate(model='openai/gpt-4o', prompt='hi', config={'temprature': 0.2})

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert 'temprature' in str(raised.value)
    assert server.requests == []


@pytest.mark.asyncio
async def test_generate_openai_extra_is_sent_as_body_fields(ai: Genkit, server: _OpenAIServer) -> None:
    """`extra={'reasoning': {'effort': 'low'}}` arrives as a top-level `reasoning` field in the request body."""
    response = await ai.generate(
        model='openai/gpt-4o',
        prompt='hi',
        config={'temperature': 0.2, 'extra': {'reasoning': {'effort': 'low'}}},
    )

    assert response.text == 'hi back'
    [body] = server.bodies()
    assert body['reasoning'] == {'effort': 'low'}
    assert body['temperature'] == 0.2
    assert 'extra' not in body


@pytest.mark.asyncio
async def test_generate_openai_extra_key_overrides_declared_setting(ai: Genkit, server: _OpenAIServer) -> None:
    """`extra={'temperature': 0.9}` with `temperature=0.2` sends 0.9: extra replaces a top-level field."""
    await ai.generate(
        model='openai/gpt-4o',
        prompt='hi',
        config={'temperature': 0.2, 'extra': {'temperature': 0.9}},
    )

    [body] = server.bodies()
    assert body['temperature'] == 0.9


@pytest.mark.asyncio
async def test_openai_model_advertises_config_without_additional_properties(plugin: OpenAI) -> None:
    """The config form the Dev UI shows for an OpenAI chat model says `additionalProperties: false`."""
    action = await plugin.resolve(ActionKind.MODEL, 'openai/gpt-4o')

    assert action is not None
    options = cast(dict[str, Any], action.metadata['model'])['customOptions']
    assert options['additionalProperties'] is False
    assert 'extra' in options['properties']


# Which key caps reply length


@pytest.mark.parametrize(
    'model,cap',
    [
        pytest.param('gpt-4o', 'max_tokens', id='chat'),
        pytest.param('o3-mini', 'max_completion_tokens', id='reasoning'),
    ],
)
@pytest.mark.asyncio
async def test_openai_model_with_model_config_max_output_tokens_caps_reply(
    server: _OpenAIServer, model: str, cap: str
) -> None:
    """`ModelConfig(max_output_tokens=50)` caps the reply as `max_tokens`, or `max_completion_tokens` on o-series."""
    client = AsyncOpenAI(
        api_key=PLUGIN_KEY, http_client=httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    )
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=ModelConfig(max_output_tokens=50, temperature=0.2),
    )

    response = await OpenAIModel(model, client).generate(request, ActionRunContext())

    assert response.text == 'hi back'
    [body] = server.bodies()
    assert body['temperature'] == 0.2
    assert body[cap] == 50
    caps_sent = {
        k for k in ('max_tokens', 'max_completion_tokens', 'max_output_tokens', 'maxOutputTokens') if k in body
    }
    assert caps_sent == {cap}


# Whose key a call runs on


@pytest.fixture
def keyless_ai(server: _OpenAIServer, monkeypatch: pytest.MonkeyPatch) -> Genkit:
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return Genkit(plugins=[OpenAI(http_client=http_client, max_retries=0)])


@pytest.mark.asyncio
async def test_plugin_without_key_uses_tenant_key_from_context(keyless_ai: Genkit, server: _OpenAIServer) -> None:
    """`OpenAI()` with no key and no OPENAI_API_KEY, plus a secrets key, runs the call on that key."""
    response = await keyless_ai.generate(
        model='openai/gpt-4o', prompt='hi', context={'secrets': {'api_key': 'sk-tenant'}}
    )

    assert response.text == 'hi back'
    assert server.keys() == ['Bearer sk-tenant']


@pytest.mark.asyncio
async def test_plugin_without_key_and_no_tenant_key_raises_naming_both(
    keyless_ai: Genkit, server: _OpenAIServer
) -> None:
    """No plugin key and no secrets key fails FAILED_PRECONDITION naming both places a key can go; nothing is sent."""
    response = await keyless_ai.generate(model='openai/gpt-4o', prompt='hi')

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    message = str(response.finish_message)
    assert 'OPENAI_API_KEY' in message
    assert 'OpenAI(api_key=...)' in message
    assert "context={'secrets': {'api_key': ...}}" in message
    assert server.requests == []


@pytest.mark.asyncio
async def test_tenant_key_wins_over_plugin_key(ai: Genkit, server: _OpenAIServer) -> None:
    """`context={'secrets': {'api_key': 'sk-tenant'}}` on a plugin with its own key sends `Bearer sk-tenant`."""
    response = await ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': {'api_key': 'sk-tenant'}})

    assert response.text == 'hi back'
    assert server.keys() == ['Bearer sk-tenant']


@pytest.mark.asyncio
async def test_generate_openai_secrets_key_does_not_leak_to_next_call(ai: Genkit, server: _OpenAIServer) -> None:
    """A call with a tenant key followed by one without: the second runs on the plugin key."""
    await ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': {'api_key': 'sk-tenant'}})
    await ai.generate(model='openai/gpt-4o', prompt='hi')

    assert server.keys() == ['Bearer sk-tenant', f'Bearer {PLUGIN_KEY}']


@pytest.mark.asyncio
async def test_generate_openai_concurrent_tenants_each_use_their_key(ai: Genkit, server: _OpenAIServer) -> None:
    """Two calls in flight at once with different secrets keys each send their own key."""
    await asyncio.gather(
        ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': {'api_key': 'sk-tenant-a'}}),
        ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': {'api_key': 'sk-tenant-b'}}),
    )

    assert sorted(server.keys()) == ['Bearer sk-tenant-a', 'Bearer sk-tenant-b']


@pytest.mark.asyncio
async def test_generate_openai_secrets_without_api_key_runs_on_plugin_key(ai: Genkit, server: _OpenAIServer) -> None:
    """`context.secrets` holding only other app secrets runs the call on the plugin key."""
    response = await ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': {'db_password': 'x'}})

    assert response.text == 'hi back'
    assert server.keys() == [f'Bearer {PLUGIN_KEY}']


@pytest.mark.parametrize(
    'api_key,reason',
    [
        pytest.param(42, 'must be a string', id='not-a-string'),
        pytest.param('   ', 'is blank', id='blank'),
        pytest.param('sk-ten ant', 'invalid whitespace', id='inner-space'),
    ],
)
@pytest.mark.asyncio
async def test_generate_openai_invalid_secrets_api_key_raises_invalid_argument(
    ai: Genkit, server: _OpenAIServer, api_key: object, reason: str
) -> None:
    """A secrets key that is set but unusable fails INVALID_ARGUMENT instead of falling back to the plugin key."""
    response = await ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': {'api_key': api_key}})

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert reason in str(response.finish_message)
    assert server.requests == []


@pytest.mark.parametrize(
    'model,kwargs',
    [
        pytest.param('openai/gpt-4o', {'config': {'api_key': 'sk-tenant-secret'}}, id='config-dict'),
        pytest.param('openai/gpt-4o', {'config': {'apiKey': 'sk-tenant-secret'}}, id='config-camel-case'),
        pytest.param('openai/gpt-4o', {'config': OpenAIConfig(api_key='sk-tenant-secret')}, id='openai-config'),
        pytest.param('openai/gpt-4o', {'context': {'api_key': 'sk-tenant-secret'}}, id='top-level-context'),
        pytest.param('openai/gpt-image-1', {'config': {'api_key': 'sk-tenant-secret'}}, id='image-config-dict'),
    ],
)
@pytest.mark.asyncio
async def test_generate_openai_misplaced_api_key_raises_naming_context_secrets(
    ai: Genkit, server: _OpenAIServer, model: str, kwargs: dict[str, Any]
) -> None:
    """A key on config or the top-level context fails INVALID_ARGUMENT pointing at context.secrets.

    The plugin has its own key, so dropping the misplaced one would bill the
    wrong account. The key is never echoed and nothing is sent.
    """
    response = await ai.generate(model=model, prompt='hi', **kwargs)

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    message = str(response.finish_message)
    assert "context={'secrets': {'api_key': ...}}" in message
    assert 'sk-tenant-secret' not in message
    assert 'sk-tenant-secret' not in str(response.error)
    assert server.requests == []


_AUDIO = 'data:audio/mpeg;base64,' + base64.b64encode(b'mp3-bytes').decode()


@pytest.mark.parametrize(
    'model,prompt',
    [
        pytest.param('openai/gpt-image-1', 'a cat', id='image'),
        pytest.param('openai/tts-1', 'say hi', id='tts'),
        pytest.param('openai/whisper-1', [Part.from_media(_AUDIO, content_type='audio/mpeg')], id='stt'),
    ],
)
@pytest.mark.asyncio
async def test_generate_openai_media_model_runs_on_tenant_key(
    ai: Genkit, server: _OpenAIServer, model: str, prompt: str | list[Part]
) -> None:
    """Image, text-to-speech and transcription calls with a secrets key send that key."""
    response = await ai.generate(model=model, prompt=prompt, context={'secrets': {'api_key': 'sk-tenant'}})

    assert response.error is None
    assert response.message is not None
    assert server.keys() == ['Bearer sk-tenant']
