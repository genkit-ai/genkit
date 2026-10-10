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
from genkit_openai import OpenAI

from genkit import FinishReason, Genkit, GenkitError, Part
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
async def test_generate_openai_max_output_tokens_caps_reply(
    ai: Genkit, server: _OpenAIServer, model: str, cap: str
) -> None:
    """`config={'max_output_tokens': 50}` caps the reply as `max_tokens`, or `max_completion_tokens` on o-series."""
    response = await ai.generate(
        model=f'openai/{model}', prompt='hi', config={'max_output_tokens': 50, 'temperature': 0.2}
    )

    assert response.text == 'hi back'
    [body] = server.bodies()
    assert body['temperature'] == 0.2
    caps_sent = {
        k: body[k] for k in ('max_tokens', 'max_completion_tokens', 'max_output_tokens', 'maxOutputTokens') if k in body
    }
    assert caps_sent == {cap: 50}


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


@pytest.mark.asyncio
async def test_generate_openai_secrets_js_spelling_api_key_runs_on_tenant_key(
    ai: Genkit, server: _OpenAIServer
) -> None:
    """`context={'secrets': {'apiKey': 'sk-tenant'}}` sends `Bearer sk-tenant`."""
    await ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': {'apiKey': 'sk-tenant'}})

    assert server.keys() == ['Bearer sk-tenant']


@pytest.mark.asyncio
async def test_generate_openai_top_level_context_api_key_is_ignored(ai: Genkit, server: _OpenAIServer) -> None:
    """An `api_key` an app's own auth context provider set on the top-level context doesn't reach OpenAI."""
    response = await ai.generate(model='openai/gpt-4o', prompt='hi', context={'api_key': 'app-caller-key'})

    assert response.text == 'hi back'
    assert server.keys() == [f'Bearer {PLUGIN_KEY}']


@pytest.mark.asyncio
async def test_generate_openai_non_dict_secrets_raises_invalid_argument(ai: Genkit, server: _OpenAIServer) -> None:
    """`context={'secrets': 'sk-tenant'}` fails INVALID_ARGUMENT instead of running on the plugin key."""
    response = await ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': 'sk-tenant'})

    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert 'context.secrets must be a dict' in str(response.finish_message)
    assert server.requests == []


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
    'model,config',
    [
        pytest.param('openai/gpt-4o', {'api_key': 'sk-tenant-secret'}, id='config-dict'),
        pytest.param('openai/gpt-4o', {'apiKey': 'sk-tenant-secret'}, id='config-camel-case'),
        pytest.param('openai/gpt-4o', {'extra': {'api_key': 'sk-tenant-secret'}}, id='config-extra'),
        pytest.param('openai/gpt-image-1', {'api_key': 'sk-tenant-secret'}, id='image-config-dict'),
    ],
)
@pytest.mark.asyncio
async def test_generate_openai_config_api_key_raises_naming_context_secrets(
    ai: Genkit, server: _OpenAIServer, model: str, config: dict[str, Any]
) -> None:
    """A key on config or inside config.extra raises INVALID_ARGUMENT from generate, pointing at context.secrets.

    Genkit rejects it before the plugin runs. The key is never echoed and
    nothing is sent.
    """
    with pytest.raises(GenkitError) as raised:
        await ai.generate(model=model, prompt='hi', config=config)

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "context={'secrets': {'api_key': ...}}" in str(raised.value)
    assert 'sk-tenant-secret' not in str(raised.value)
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


# What a tenant call carries besides the key


@pytest.mark.asyncio
async def test_generate_openai_tenant_call_drops_plugin_org_and_project(
    server: _OpenAIServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant call sends no plugin organization or project; other default headers and the plugin call keep theirs."""
    monkeypatch.setenv('OPENAI_ORG_ID', 'org-env')
    monkeypatch.setenv('OPENAI_PROJECT_ID', 'proj-env')
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    ai = Genkit(
        plugins=[
            OpenAI(
                api_key=PLUGIN_KEY,
                organization='org-plugin',
                default_headers={'OpenAI-Project': 'proj-pinned', 'X-Gateway-Route': 'eu'},
                http_client=http_client,
                max_retries=0,
            )
        ]
    )

    await ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': {'api_key': 'sk-tenant'}})
    await ai.generate(model='openai/gpt-4o', prompt='hi')

    tenant, plugin = server.requests
    assert tenant.headers['authorization'] == 'Bearer sk-tenant'
    assert 'openai-organization' not in tenant.headers
    assert 'openai-project' not in tenant.headers
    assert tenant.headers['x-gateway-route'] == 'eu'
    assert plugin.headers['openai-organization'] == 'org-plugin'
    assert plugin.headers['openai-project'] == 'proj-pinned'


@pytest.mark.asyncio
async def test_plugin_without_key_tenant_call_drops_env_org_and_project(
    server: _OpenAIServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`OpenAI()` with no key and OPENAI_ORG_ID/OPENAI_PROJECT_ID set sends neither on a tenant call."""
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_ORG_ID', 'org-env')
    monkeypatch.setenv('OPENAI_PROJECT_ID', 'proj-env')
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    ai = Genkit(plugins=[OpenAI(http_client=http_client, max_retries=0)])

    await ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': {'api_key': 'sk-tenant'}})

    [tenant] = server.requests
    assert tenant.headers['authorization'] == 'Bearer sk-tenant'
    assert 'openai-organization' not in tenant.headers
    assert 'openai-project' not in tenant.headers


@pytest.mark.asyncio
async def test_generate_openai_pinned_authorization_header_refuses_tenant_key(server: _OpenAIServer) -> None:
    """`default_headers={'Authorization': ...}` would replace a tenant key, so the call fails FAILED_PRECONDITION."""
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    ai = Genkit(
        plugins=[
            OpenAI(
                api_key=PLUGIN_KEY,
                default_headers={'Authorization': 'Bearer corp-gateway'},
                http_client=http_client,
                max_retries=0,
            )
        ]
    )

    response = await ai.generate(model='openai/gpt-4o', prompt='hi', context={'secrets': {'api_key': 'sk-tenant'}})

    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert 'Authorization' in str(response.finish_message)
    assert server.requests == []


# Plugin without a key: embedders and the model list


@pytest.mark.asyncio
async def test_plugin_without_key_embed_fails_failed_precondition(keyless_ai: Genkit, server: _OpenAIServer) -> None:
    """`embed()` on `OpenAI()` with no key fails FAILED_PRECONDITION naming the plugin key; nothing is sent."""
    with pytest.raises(GenkitError) as raised:
        await keyless_ai.embed(embedder='openai/text-embedding-3-small', content='hi')

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert 'OPENAI_API_KEY' in str(raised.value)
    assert server.requests == []


@pytest.mark.asyncio
async def test_plugin_without_key_lists_built_in_catalog_without_calling_openai(
    keyless_ai: Genkit, server: _OpenAIServer
) -> None:
    """The Dev UI catalog for `OpenAI()` with no key has the built-in models and makes no request."""
    catalog = await keyless_ai._registry.list_actions()

    assert '/model/openai/gpt-4o' in catalog
    assert server.requests == []
