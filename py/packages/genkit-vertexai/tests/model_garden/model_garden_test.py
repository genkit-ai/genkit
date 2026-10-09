# Copyright 2025 Google LLC
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

"""Unittests for VertexAI Model Garden Models."""

import json
import subprocess  # noqa: S404
import sys
import textwrap
from types import ModuleType
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from anthropic import AsyncAnthropicVertex
from genkit_anthropic import AnthropicConfig
from genkit_openai import OpenAIConfig
from genkit_vertexai.model_garden import ModelGarden
from genkit_vertexai.model_garden._model_info import DEFAULT_SUPPORTS, SUPPORTED_OPENAI_COMPAT_MODELS
from genkit_vertexai.model_garden.anthropic import AnthropicModelGarden
from genkit_vertexai.model_garden.model_garden import ModelGardenModel
from google.auth.exceptions import DefaultCredentialsError
from openai.types.chat import ChatCompletion

from genkit import ActionRunContext, Genkit, GenkitError, Message, Part, Role
from genkit._ai._formats import built_in_formats
from genkit.model import ModelRequest, OutputConfig
from genkit.plugin_api import ActionKind

CLAUDE = 'modelgarden/anthropic/claude-sonnet-4@20250514'
_ADC = 'genkit_vertexai.model_garden.model_garden.google_auth_default'
LLAMA = 'modelgarden/meta/llama-3.1-405b-instruct-maas'
MISTRAL = 'modelgarden/mistralai/mistral-small-2503'

CLAUDE_EXTRA_MISSING = "Model Garden Claude models need the anthropic extra: uv add 'genkit-vertexai[anthropic]'"
OPENAI_EXTRA_MISSING = (
    'Model Garden Llama, Mistral, and other OpenAI-compatible models need the openai extra: '
    "uv add 'genkit-vertexai[openai]'"
)


def test_catalog_output_names_are_known_formats() -> None:
    """supports.output lists Genkit output formats, not OpenAI request options like json_mode."""
    known = {f.name for f in built_in_formats}
    entries = {name: info.supports for name, info in SUPPORTED_OPENAI_COMPAT_MODELS.items()}
    entries['<default>'] = DEFAULT_SUPPORTS
    for name, supports in entries.items():
        unknown = set((supports.output if supports else None) or []) - known
        assert not unknown, f'{name}: {sorted(unknown)}'


@pytest.fixture
def model_garden_instance() -> ModelGardenModel:
    """Model Garden fixture."""
    return ModelGardenModel(model='test', location='us-central1', project='project')


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


@pytest.mark.parametrize(
    'model_name, expected',
    [
        (
            'meta/llama-3.1-405b-instruct-maas',
            {
                'name': 'ModelGarden - Meta - llama-3.1',
                'supports': {
                    'constrained': None,
                    'content_type': None,
                    'long_running': False,
                    'multiturn': True,
                    'media': False,
                    'tools': True,
                    'system_role': True,
                    'output': [
                        'json',
                        'text',
                    ],
                    'tool_choice': None,
                },
            },
        ),
        (
            'meta/lazaro-model-pro-max',
            {
                'name': 'ModelGarden - meta/lazaro-model-pro-max',
                'supports': {
                    'constrained': None,
                    'content_type': None,
                    'long_running': None,
                    'multiturn': True,
                    'media': True,
                    'tools': True,
                    'system_role': True,
                    'output': [
                        'json',
                        'text',
                    ],
                    'tool_choice': None,
                },
            },
        ),
    ],
)
def test_get_model_info(model_name: str, expected: dict[str, Any], model_garden_instance: ModelGardenModel) -> None:
    """Unittest for get_model_info."""
    model_garden_instance.name = model_name

    result = model_garden_instance.get_model_info()

    assert result == expected


def test_anthropic_model_garden_uses_anthropic_config_schema() -> None:
    """Anthropic Model Garden advertises the schema enforced by its handler."""
    assert AnthropicModelGarden.get_config_schema() is AnthropicConfig


def test_anthropic_model_garden_does_not_advertise_api_key() -> None:
    """The advertised schema has no apiKey; a per-request key goes in context.secrets."""
    properties = AnthropicModelGarden.get_config_schema().model_json_schema()['properties']
    assert 'apiKey' not in properties
    assert 'apiVersion' in properties


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'model_name', ['modelgarden/meta/llama-3.2-90b-vision-instruct-maas', 'modelgarden/anthropic/claude-sonnet-4']
)
@pytest.mark.parametrize(
    'adc',
    [
        pytest.param({'return_value': (MagicMock(), None)}, id='adc-without-project'),
        pytest.param({'side_effect': DefaultCredentialsError('no ADC')}, id='no-adc'),
    ],
)
async def test_resolve_without_project_is_failed_precondition(model_name: str, adc: dict[str, Any]) -> None:
    """No project passed, in the environment, or in ADC is a local setup problem, not a bad request."""
    with patch.dict('os.environ', {}, clear=True):
        plugin = ModelGarden(location='us-central1')

    with (
        patch(_ADC, **adc),
        pytest.raises(GenkitError, match=r'ModelGarden\(project=\.\.\.\) or set GOOGLE_CLOUD_PROJECT') as raised,
    ):
        await plugin.resolve(ActionKind.MODEL, model_name)

    assert raised.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_model_garden_llama_json_request_sends_json_object() -> None:
    """ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', output_format='json') sends json_object."""
    captured: dict[str, Any] = {}
    client = MagicMock()

    async def create(**kwargs: Any) -> ChatCompletion:
        captured.update(kwargs)
        return ChatCompletion.construct(
            id='1',
            object='chat.completion',
            created=1,
            model='llama',
            choices=[
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': '{"a": 1}'},
                    'finish_reason': 'stop',
                }
            ],
        )

    client.chat.completions.create = AsyncMock(side_effect=create)
    garden = ModelGardenModel(
        model='meta/llama-3.1-405b-instruct-maas',
        location='us-central1',
        project='p',
    )
    ctx = MagicMock(spec=ActionRunContext)
    type(ctx).is_streaming = PropertyMock(return_value=False)
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('give me json')])],
        output=OutputConfig(format='json'),
        config=OpenAIConfig(),
    )

    with patch.object(garden, 'create_client', AsyncMock(return_value=client)):
        await garden.to_openai_compatible_model()(request, ctx)

    assert captured['response_format'] == {'type': 'json_object'}


@pytest.mark.asyncio
async def test_model_garden_openai_compatible_model_resolves_and_generates() -> None:
    """ai.generate on a catalog Llama model sends the prompt to the OpenAI client and returns its reply."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_chat_completion('hello from llama'))

    with patch(
        'genkit_vertexai.model_garden.model_garden.ModelGardenModel.create_client',
        new=AsyncMock(return_value=client),
    ):
        ai = Genkit(plugins=[ModelGarden(project='my-project', location='us-central1')])
        response = await ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', prompt='hi')

    assert response.text == 'hello from llama'
    client.chat.completions.create.assert_awaited_once()
    sent = client.chat.completions.create.call_args.kwargs
    assert sent['model'] == 'meta/llama-3.1-405b-instruct-maas'
    assert sent['messages'] == [{'role': 'user', 'content': 'hi'}]


class _FakeCredentials:
    """Google credentials stand-in with a token that can expire."""

    def __init__(self) -> None:
        self.token = 'tok-1'
        self.valid = True
        self.refresh_count = 0

    def refresh(self, _request: object) -> None:
        self.refresh_count += 1
        if self.refresh_count > 1:
            self.token = 'tok-2'
        self.valid = True


@pytest.mark.asyncio
async def test_model_garden_repeated_generate_refreshes_credentials_once() -> None:
    """Three ai.generate calls on a Llama model refresh Google credentials once and reuse one OpenAI client."""
    creds = _FakeCredentials()
    clients: list[MagicMock] = []

    def make_client(**kwargs: object) -> MagicMock:
        client = MagicMock()
        client.api_key = kwargs.get('api_key')
        client.chat.completions.create = AsyncMock(return_value=_chat_completion('ok'))
        clients.append(client)
        return client

    with (
        patch('genkit_vertexai.model_garden.client.auth.default', return_value=(creds, 'my-project')),
        patch('genkit_vertexai.model_garden.client.google.auth.transport.requests.Request'),
        patch('genkit_vertexai.model_garden.client._AsyncOpenAI', side_effect=make_client),
    ):
        ai = Genkit(plugins=[ModelGarden(project='my-project', location='us-central1')])
        for _ in range(3):
            response = await ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', prompt='hi')
            assert response.text == 'ok'

    assert creds.refresh_count == 1
    assert len(clients) == 1


@pytest.mark.asyncio
async def test_model_garden_generate_after_token_expiry_sends_fresh_token() -> None:
    """After cached credentials expire, the next generate refreshes and sends the new token."""
    creds = _FakeCredentials()
    clients: list[MagicMock] = []

    def make_client(**kwargs: object) -> MagicMock:
        client = MagicMock()
        client.api_key = kwargs.get('api_key')
        client.chat.completions.create = AsyncMock(return_value=_chat_completion('ok'))
        clients.append(client)
        return client

    with (
        patch('genkit_vertexai.model_garden.client.auth.default', return_value=(creds, 'my-project')),
        patch('genkit_vertexai.model_garden.client.google.auth.transport.requests.Request'),
        patch('genkit_vertexai.model_garden.client._AsyncOpenAI', side_effect=make_client),
    ):
        ai = Genkit(plugins=[ModelGarden(project='my-project', location='us-central1')])
        first = await ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', prompt='hi')
        assert first.text == 'ok'
        assert clients[0].api_key == 'tok-1'

        creds.valid = False
        second = await ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', prompt='hi')
        assert second.text == 'ok'

    assert creds.refresh_count == 2
    assert len(clients) == 1
    assert clients[0].api_key == 'tok-2'


@pytest.mark.asyncio
async def test_generate_model_garden_claude_registers_full_publisher_path() -> None:
    """`modelgarden/anthropic/claude-…` registers under that name; Claude gets the id without the publisher."""
    claude = 'modelgarden/anthropic/claude-sonnet-4-5'
    reply = MagicMock()
    reply.content = [MagicMock(type='text', text='hello')]
    reply.usage = MagicMock(input_tokens=1, output_tokens=1)
    reply.stop_reason = 'end_turn'
    client = MagicMock(spec=AsyncAnthropicVertex)
    client.messages = MagicMock()
    client.beta = MagicMock()
    client.messages.create = AsyncMock(return_value=reply)
    client.beta.messages.create = AsyncMock(return_value=reply)
    with patch('genkit_vertexai.model_garden.anthropic.AsyncAnthropicVertex', return_value=client):
        ai = Genkit(plugins=[ModelGarden(project='p', location='us-central1')])
        response = await ai.generate(model=claude, prompt='hi')
        action = await ai.lookup_model(claude)

    assert response.text == 'hello'
    sent = client.messages.create.await_args or client.beta.messages.create.await_args
    assert sent is not None
    assert sent.kwargs['model'].startswith('claude-sonnet-4-5')
    assert action is not None
    assert action.name == 'modelgarden/anthropic/claude-sonnet-4-5'


def _uninstall(monkeypatch: pytest.MonkeyPatch, *modules: str) -> None:
    """Makes `modules` import as if absent, and drops the Claude worker module so it imports again."""
    # A None entry in sys.modules makes `import` raise ModuleNotFoundError with `name` set.
    module_table = cast(dict[str, ModuleType | None], sys.modules)
    for module in modules:
        monkeypatch.setitem(module_table, module, None)
    monkeypatch.delitem(sys.modules, 'genkit_vertexai.model_garden.anthropic', raising=False)


def test_import_genkit_vertexai_does_not_load_publisher_sdks() -> None:
    """`import genkit_vertexai` and `ModelGarden()` leave the Anthropic and OpenAI SDKs and plugins unloaded."""
    code = textwrap.dedent("""
        import json, sys
        import genkit_vertexai
        from genkit_vertexai.model_garden import ModelGarden
        ModelGarden(project='my-project')
        publishers = ('anthropic', 'genkit_anthropic', 'openai', 'genkit_openai')
        print(json.dumps(sorted(m for m in publishers if m in sys.modules)))
    """)
    proc = subprocess.run(  # noqa: S603
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=120
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'missing', 'project', 'message'),
    [
        pytest.param(CLAUDE, ('genkit_anthropic', 'genkit_openai'), 'p', CLAUDE_EXTRA_MISSING, id='claude-no-extras'),
        pytest.param(CLAUDE, ('genkit_anthropic',), 'p', CLAUDE_EXTRA_MISSING, id='claude-openai-extra-only'),
        pytest.param(CLAUDE, ('anthropic',), 'p', CLAUDE_EXTRA_MISSING, id='claude-sdk-missing'),
        pytest.param(CLAUDE, ('genkit_anthropic',), None, CLAUDE_EXTRA_MISSING, id='claude-no-project-extra-first'),
        pytest.param(LLAMA, ('genkit_anthropic', 'genkit_openai'), 'p', OPENAI_EXTRA_MISSING, id='llama-no-extras'),
        pytest.param(LLAMA, ('genkit_openai',), None, OPENAI_EXTRA_MISSING, id='llama-no-project-extra-first'),
        pytest.param(MISTRAL, ('genkit_openai',), 'p', OPENAI_EXTRA_MISSING, id='mistral-uncataloged'),
    ],
)
async def test_generate_model_garden_model_without_its_extra_raises_install_command(
    monkeypatch: pytest.MonkeyPatch, model: str, missing: tuple[str, ...], project: str | None, message: str
) -> None:
    """Generating with a Model Garden model whose extra is missing raises FAILED_PRECONDITION naming `uv add`."""
    for var in ('GCLOUD_PROJECT', 'GOOGLE_CLOUD_PROJECT'):
        monkeypatch.delenv(var, raising=False)
    _uninstall(monkeypatch, *missing)
    ai = Genkit(plugins=[ModelGarden(project=project)])

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model=model, prompt='hi')

    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert exc_info.value.original_message == message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'missing', 'forget'),
    [
        pytest.param(CLAUDE, 'genkit_anthropic._models', (), id='claude'),
        pytest.param(LLAMA, 'genkit_openai._openai_plugin', ('genkit_openai',), id='llama'),
    ],
)
async def test_generate_model_garden_broken_publisher_install_raises_import_error(
    monkeypatch: pytest.MonkeyPatch, model: str, missing: str, forget: tuple[str, ...]
) -> None:
    """A publisher package that is installed but fails to import raises that import error, not the extra hint."""
    for module in forget:
        monkeypatch.delitem(sys.modules, module)
    _uninstall(monkeypatch, missing)
    ai = Genkit(plugins=[ModelGarden(project='p')])

    with pytest.raises(ModuleNotFoundError) as exc_info:
        await ai.generate(model=model, prompt='hi')

    assert exc_info.value.name == missing


@pytest.mark.asyncio
async def test_model_garden_list_actions_without_openai_extra_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without `[openai]`, the model list the Dev UI reads is empty."""
    _uninstall(monkeypatch, 'genkit_openai')

    assert await ModelGarden(project='p').list_actions() == []


@pytest.mark.asyncio
async def test_model_garden_list_actions_broken_openai_install_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """`list_actions()` raises when genkit-openai is installed but can't import, instead of listing nothing."""
    monkeypatch.delitem(sys.modules, 'genkit_openai')
    _uninstall(monkeypatch, 'genkit_openai._openai_plugin')

    with pytest.raises(ModuleNotFoundError) as exc_info:
        await ModelGarden(project='p').list_actions()

    assert exc_info.value.name == 'genkit_openai._openai_plugin'


@pytest.mark.asyncio
async def test_model_garden_list_actions_with_openai_extra_lists_llama_models() -> None:
    """With `[openai]`, the Dev UI lists every built-in OpenAI-compatible model, Llama included."""
    actions = await ModelGarden(project='p').list_actions()

    names = [a.name for a in actions]
    assert names == [f'modelgarden/{model}' for model in SUPPORTED_OPENAI_COMPAT_MODELS]
    assert LLAMA in names
    metadata = actions[names.index(LLAMA)].metadata
    assert metadata is not None
    llama = metadata['model']
    assert isinstance(llama, dict)
    catalog = SUPPORTED_OPENAI_COMPAT_MODELS['meta/llama-3.1-405b-instruct-maas'].supports
    assert catalog is not None
    assert llama['supports'] == catalog.model_dump(by_alias=True, exclude_none=True)


@pytest.mark.asyncio
async def test_resolve_uncataloged_openai_compat_model_advertises_default_supports() -> None:
    """An uncataloged OpenAI-compatible model advertises, in camelCase, the label and supports its handler runs with."""
    ai = Genkit(plugins=[ModelGarden(project='p')])

    action = await ai.lookup_model(MISTRAL)

    assert action is not None
    info = action.metadata['model']
    assert isinstance(info, dict)
    assert info['label'] == 'ModelGarden - mistralai/mistral-small-2503'
    assert info['supports'] == DEFAULT_SUPPORTS.model_dump(by_alias=True, exclude_none=True)


def _project_in(request: httpx.Request) -> str:
    """The project segment of a Vertex rawPredict URL."""
    parts = request.url.path.split('/')
    return parts[parts.index('projects') + 1]


async def _generate_claude(plugin: ModelGarden) -> None:
    ai = Genkit(plugins=[plugin])
    response = await ai.generate(model='modelgarden/anthropic/claude-sonnet-4-5', prompt='hi')
    assert response.text == 'ok'


@pytest.mark.asyncio
async def test_model_garden_project_reaches_vertex_over_environment(vertex_requests: list[httpx.Request]) -> None:
    """ModelGarden(project='p') sends generate calls to p even when GCLOUD_PROJECT is set."""
    with patch.dict('os.environ', {'GCLOUD_PROJECT': 'env-proj', 'GOOGLE_CLOUD_PROJECT': 'env-proj'}):
        await _generate_claude(ModelGarden(project='p', location='us-central1'))

    assert [_project_in(r) for r in vertex_requests] == ['p']


@pytest.mark.asyncio
async def test_model_garden_without_project_uses_environment(vertex_requests: list[httpx.Request]) -> None:
    """ModelGarden() with no project sends generate calls to GCLOUD_PROJECT."""
    with patch.dict('os.environ', {'GCLOUD_PROJECT': 'env-proj', 'GOOGLE_CLOUD_PROJECT': ''}):
        await _generate_claude(ModelGarden(location='us-central1'))

    assert [_project_in(r) for r in vertex_requests] == ['env-proj']


@pytest.mark.asyncio
async def test_model_garden_without_project_or_environment_uses_adc_project(
    vertex_requests: list[httpx.Request],
) -> None:
    """On Cloud Run (no project env var) ModelGarden() sends generate calls to the ADC project, probing ADC once."""
    with (
        patch.dict('os.environ', {'GCLOUD_PROJECT': '', 'GOOGLE_CLOUD_PROJECT': ''}),
        patch(_ADC, return_value=(MagicMock(), 'adc-proj')) as adc,
    ):
        ai = Genkit(plugins=[ModelGarden(location='us-central1')])
        await ai.generate(model='modelgarden/anthropic/claude-sonnet-4-5', prompt='hi')
        await ai.generate(model='modelgarden/anthropic/claude-sonnet-4-6', prompt='hi')

    assert [_project_in(r) for r in vertex_requests] == ['adc-proj', 'adc-proj']
    adc.assert_called_once()


@pytest.mark.asyncio
async def test_model_garden_environment_project_skips_adc(vertex_requests: list[httpx.Request]) -> None:
    """A project from GCLOUD_PROJECT means ADC is never probed, so nothing waits on the metadata server."""
    with (
        patch.dict('os.environ', {'GCLOUD_PROJECT': 'env-proj', 'GOOGLE_CLOUD_PROJECT': ''}),
        patch(_ADC) as adc,
    ):
        await _generate_claude(ModelGarden(location='us-central1'))

    adc.assert_not_called()
    assert [_project_in(r) for r in vertex_requests] == ['env-proj']
