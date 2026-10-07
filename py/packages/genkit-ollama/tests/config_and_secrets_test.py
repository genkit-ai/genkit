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

"""What an Ollama call sends: config keys, ``extra``, and the per-request key.

Every test drives ``ai.generate`` through the real ``ollama`` client and reads
the HTTP request that would have reached the server.
"""

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast, get_args

import httpx
import ollama as ollama_api
import pytest
from genkit_ollama import Ollama, RequestHeaderParams, ollama_name
from genkit_ollama.constants import OllamaAPITypes
from genkit_ollama.models import ModelDefinition, OllamaConfig
from pydantic import ValidationError

from genkit import FinishReason, Genkit, GenkitError

MODEL = 'ollama/llama3.2'


@dataclass
class Server:
    """Captures the requests the ollama client sends."""

    requests: list[httpx.Request] = field(default_factory=list)

    @property
    def last_body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)

    @property
    def last_options(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.last_body.get('options') or {})

    @property
    def last_authorization(self) -> str | None:
        return self.requests[-1].headers.get('authorization')

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        if request.url.path == '/api/generate':
            reply = {'model': body['model'], 'response': 'ok', 'done': True}
        else:
            reply = {'model': body['model'], 'message': {'role': 'assistant', 'content': 'ok'}, 'done': True}
        if body.get('stream'):
            return httpx.Response(200, content=json.dumps(reply) + '\n')
        return httpx.Response(200, json=reply)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Server]:
    srv = Server()
    real_client = ollama_api.AsyncClient

    def client_with_fake_server(**kwargs: Any) -> ollama_api.AsyncClient:
        return real_client(transport=httpx.MockTransport(srv.handle), **kwargs)

    monkeypatch.setattr(ollama_api, 'AsyncClient', client_with_fake_server)
    monkeypatch.delenv('OLLAMA_API_KEY', raising=False)
    yield srv


def make_ai(**plugin_kwargs: Any) -> Genkit:
    return Genkit(plugins=[Ollama(**plugin_kwargs)])


def test_ollama_config_with_unknown_key_raises_validation_error() -> None:
    """`OllamaConfig(temprature=0.2)` fails naming `temprature`."""
    with pytest.raises(ValidationError, match='temprature'):
        OllamaConfig.model_validate({'temprature': 0.2})


@pytest.mark.asyncio
async def test_generate_ollama_unknown_config_key_raises_before_sending(server: Server) -> None:
    """`config={'temprature': 0.2}` raises INVALID_ARGUMENT naming the key, and nothing is sent."""
    ai = make_ai()

    with pytest.raises(GenkitError, match='temprature') as excinfo:
        await ai.generate(model=MODEL, prompt='hi', config={'temprature': 0.2})

    assert excinfo.value.status == 'INVALID_ARGUMENT'
    assert "config['extra']" in str(excinfo.value)
    assert server.requests == []


@pytest.mark.asyncio
async def test_generate_ollama_declared_sampler_option_reaches_options(server: Server) -> None:
    """`repeat_penalty=1.1` arrives in the request's `options`."""
    ai = make_ai()

    await ai.generate(model=MODEL, prompt='hi', config={'repeat_penalty': 1.1})

    assert server.last_options == {'repeat_penalty': 1.1}


@pytest.mark.asyncio
async def test_generate_ollama_camel_case_sampler_option_reaches_options(server: Server) -> None:
    """`repeatPenalty=1.1` arrives the same way."""
    ai = make_ai()

    await ai.generate(model=MODEL, prompt='hi', config={'repeatPenalty': 1.1})

    assert server.last_options == {'repeat_penalty': 1.1}


def _sample_value(annotation: object) -> object:
    kinds = set(get_args(annotation)) - {type(None)}
    if bool in kinds:
        return True
    if int in kinds:
        return 3
    if float in kinds:
        return 0.5
    return ['###']


@pytest.mark.asyncio
@pytest.mark.parametrize('option', sorted(ollama_api.Options.model_fields))
async def test_generate_ollama_every_client_sampler_option_reaches_options(server: Server, option: str) -> None:
    """Each sampler option the installed `ollama` client knows is accepted and lands in `options`."""
    value = _sample_value(ollama_api.Options.model_fields[option].annotation)
    ai = make_ai()

    await ai.generate(model=MODEL, prompt='hi', config={option: value})

    assert server.last_options == {option: value}


@pytest.mark.asyncio
async def test_generate_ollama_extra_lands_in_options(server: Server) -> None:
    """`extra={'mirostat': 2, 'new_option': 'x'}` is merged into `options` as-is, not the top of the body."""
    ai = make_ai()

    await ai.generate(
        model=MODEL, prompt='hi', config={'temperature': 0.2, 'extra': {'mirostat': 2, 'new_option': 'x'}}
    )

    assert server.last_options == {'temperature': 0.2, 'mirostat': 2, 'new_option': 'x'}
    assert 'new_option' not in server.last_body
    assert 'extra' not in server.last_body


@pytest.mark.asyncio
async def test_generate_ollama_extra_key_overrides_declared_setting(server: Server) -> None:
    """`extra={'num_ctx': 8192}` with `num_ctx=2048` sends 8192."""
    ai = make_ai()

    await ai.generate(model=MODEL, prompt='hi', config={'num_ctx': 2048, 'extra': {'num_ctx': 8192}})

    assert server.last_options == {'num_ctx': 8192}


@pytest.mark.asyncio
async def test_generate_ollama_extra_on_generate_api_model_lands_in_options(server: Server) -> None:
    """A model on Ollama's `/api/generate` endpoint gets `extra` in `options` too."""
    ai = make_ai(models=[ModelDefinition(name='llama3.2', api_type=OllamaAPITypes.GENERATE)])

    await ai.generate(model=MODEL, prompt='hi', config={'extra': {'mirostat': 2}})

    assert server.requests[-1].url.path == '/api/generate'
    assert server.last_options == {'mirostat': 2}


@pytest.mark.asyncio
async def test_generate_ollama_think_and_keep_alive_stay_top_level(server: Server) -> None:
    """`think` and `keep_alive` are still request fields, not options."""
    ai = make_ai()

    await ai.generate(model=MODEL, prompt='hi', config={'think': True, 'keepAlive': '5m', 'num_ctx': 2048})

    assert server.last_body['think'] is True
    assert server.last_body['keep_alive'] == '5m'
    assert server.last_options == {'num_ctx': 2048}


def test_ollama_model_advertises_config_without_additional_properties() -> None:
    """The Dev UI config schema says `additionalProperties: false`."""
    action = Ollama()._create_model_action(ollama_name('llama3.2'))

    model_meta = cast(dict[str, Any], cast(dict[str, Any], action.metadata)['model'])
    assert model_meta['customOptions']['additionalProperties'] is False


@pytest.mark.asyncio
async def test_generate_ollama_secrets_api_key_sends_bearer_header(server: Server) -> None:
    """`context={'secrets': {'api_key': 'tenant'}}` sends `Authorization: Bearer tenant`."""
    ai = make_ai()

    await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': 'tenant'}})

    assert server.last_authorization == 'Bearer tenant'


@pytest.mark.asyncio
async def test_generate_ollama_secrets_camel_case_api_key_sends_bearer_header(server: Server) -> None:
    """`context={'secrets': {'apiKey': 'tenant'}}` sends the same bearer header."""
    ai = make_ai()

    await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'apiKey': 'tenant'}})

    assert server.last_authorization == 'Bearer tenant'


@pytest.mark.asyncio
async def test_generate_stream_ollama_secrets_api_key_sends_bearer_header(server: Server) -> None:
    """The streaming call sends the same bearer header."""
    ai = make_ai()

    result = ai.generate_stream(model=MODEL, prompt='hi', context={'secrets': {'api_key': 'tenant'}})
    async for _ in result.stream:
        pass
    await result.response

    assert server.last_body['stream'] is True
    assert server.last_authorization == 'Bearer tenant'


@pytest.mark.asyncio
async def test_generate_ollama_generate_api_model_secrets_api_key_sends_bearer_header(server: Server) -> None:
    """A model on `/api/generate` sends the same bearer header."""
    ai = make_ai(models=[ModelDefinition(name='llama3.2', api_type=OllamaAPITypes.GENERATE)])

    await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': 'tenant'}})

    assert server.requests[-1].url.path == '/api/generate'
    assert server.last_authorization == 'Bearer tenant'


@pytest.mark.asyncio
async def test_generate_ollama_without_secrets_sends_configured_headers(server: Server) -> None:
    """No secrets means the plugin's `request_headers` only."""
    ai = make_ai(request_headers={'Authorization': 'Bearer plugin', 'X-Org': 'acme'})

    await ai.generate(model=MODEL, prompt='hi')

    assert server.last_authorization == 'Bearer plugin'
    assert server.requests[-1].headers['x-org'] == 'acme'


@pytest.mark.asyncio
@pytest.mark.parametrize('header_name', ['Authorization', 'authorization'])
async def test_generate_ollama_secrets_key_replaces_configured_authorization_header(
    server: Server, header_name: str
) -> None:
    """A secrets key wins over a static `Authorization` header; the plugin's other headers still go."""
    ai = make_ai(request_headers={header_name: 'Bearer plugin', 'X-Org': 'acme'})

    await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': 'tenant'}})

    assert server.requests[-1].headers.get_list('authorization') == ['Bearer tenant']
    assert server.requests[-1].headers['x-org'] == 'acme'


@pytest.mark.asyncio
async def test_generate_ollama_secrets_key_replaces_header_callable_authorization(server: Server) -> None:
    """A secrets key also wins over an `Authorization` header from a `request_headers` callable."""

    def headers(_: RequestHeaderParams) -> dict[str, str]:
        return {'Authorization': 'Bearer minted', 'X-Org': 'acme'}

    ai = make_ai(request_headers=headers)

    await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': 'tenant'}})

    assert server.last_authorization == 'Bearer tenant'
    assert server.requests[-1].headers['x-org'] == 'acme'


@pytest.mark.asyncio
async def test_generate_ollama_secrets_key_wins_over_ollama_api_key_env(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `OLLAMA_API_KEY` set, a secrets key is still the one sent."""
    monkeypatch.setenv('OLLAMA_API_KEY', 'from-env')
    ai = make_ai()

    await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': 'tenant'}})

    assert server.last_authorization == 'Bearer tenant'


@pytest.mark.asyncio
async def test_generate_ollama_secrets_key_does_not_leak_to_next_call(server: Server) -> None:
    """A following call with no secrets sends the plugin's header again."""
    ai = make_ai(request_headers={'Authorization': 'Bearer plugin'})

    await ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': 'tenant'}})
    await ai.generate(model=MODEL, prompt='hi')

    assert [r.headers.get('authorization') for r in server.requests] == ['Bearer tenant', 'Bearer plugin']


@pytest.mark.asyncio
async def test_generate_ollama_concurrent_tenants_each_use_their_key(server: Server) -> None:
    """Two calls in flight at once with different secrets keys each send their own key."""
    ai = make_ai()

    await asyncio.gather(
        ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': 'tenant-a'}}),
        ai.generate(model=MODEL, prompt='hi', context={'secrets': {'api_key': 'tenant-b'}}),
    )

    assert sorted(r.headers.get('authorization') for r in server.requests) == ['Bearer tenant-a', 'Bearer tenant-b']


@pytest.mark.asyncio
async def test_generate_ollama_secrets_without_api_key_raises_invalid_argument(server: Server) -> None:
    """`context={'secrets': {}}` fails the call with INVALID_ARGUMENT on `response.error`, and nothing is sent."""
    ai = make_ai()

    response = await ai.generate(model=MODEL, prompt='hi', context={'secrets': {}})

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert 'context.secrets' in response.error.message
    assert response.message is None
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['api_key', 'apiKey'])
async def test_generate_ollama_config_api_key_raises_naming_context_secrets(server: Server, field: str) -> None:
    """`config={'api_key': ...}` fails INVALID_ARGUMENT pointing at context.secrets, without echoing the key."""
    ai = make_ai()

    response = await ai.generate(model=MODEL, prompt='hi', config={field: 'tenant-secret'})

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    message = str(response.error.message)
    assert 'context.secrets' in message
    assert 'tenant-secret' not in message
    assert server.requests == []
