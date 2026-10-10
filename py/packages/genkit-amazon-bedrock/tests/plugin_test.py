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

"""Tests for the Amazon Bedrock plugin wiring."""

import json
from types import SimpleNamespace
from typing import Any, cast

import boto3.session
import pytest
from genkit_amazon_bedrock import Bedrock, BedrockConfig
from genkit_amazon_bedrock._transport import BedrockTransport
from pydantic import ValidationError

from genkit import Document, Genkit, GenkitError, ModelResponse
from genkit.embedder import EmbedRequest
from genkit.model import ModelRequest
from genkit.plugin_api import ActionKind


def test_plugin_name() -> None:
    plugin = Bedrock()
    assert plugin.name == 'bedrock'


def test_constructor_defaults() -> None:
    plugin = Bedrock()
    # No default region: resolution falls to the SDK chain and fails loudly.
    assert plugin.region is None
    # The AWS client knobs stay unset so the caller's own AWS configuration is
    # what the transport defers to; package defaults apply below that.
    assert plugin.max_retries is None
    assert plugin.read_timeout is None
    assert plugin.connect_timeout is None
    assert plugin.max_pool_connections is None
    assert plugin.total_timeout == 3600.0
    assert plugin.models == []
    assert plugin.embedders == []


def test_config_accepts_camel_case_and_rejects_unknown_fields() -> None:
    config = BedrockConfig.model_validate({
        'toolChoice': 'auto',
        'maxOutputTokens': 128,
        'additionalModelRequestFields': {'thinking': {'type': 'enabled'}},
    })
    assert config.tool_choice == 'auto'
    assert config.max_output_tokens == 128
    assert config.additional_model_request_fields == {'thinking': {'type': 'enabled'}}
    # An unknown key would validate and then silently never reach the wire;
    # rejection turns that typo into an error at the call site.
    with pytest.raises(ValidationError):
        BedrockConfig.model_validate({'maxTokens': 128})


@pytest.mark.asyncio
async def test_init_returns_no_eager_actions() -> None:
    plugin = Bedrock(region='us-east-1')
    assert await plugin.init() == []


@pytest.mark.asyncio
async def test_init_fails_loudly_without_region() -> None:
    # A stub session isolates the test from ambient AWS env/config.
    stub_session = cast(boto3.session.Session, SimpleNamespace(region_name=None))
    plugin = Bedrock(session=stub_session)
    with pytest.raises(GenkitError, match='no AWS region resolved') as excinfo:
        await plugin.init()
    assert excinfo.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_resolve_returns_model_action_for_any_model_id() -> None:
    plugin = Bedrock(region='us-east-1')
    action = await plugin.resolve(ActionKind.MODEL, 'amazon.nova-lite-v1:0')
    assert action is not None
    assert action.name == 'bedrock/amazon.nova-lite-v1:0'
    assert action.metadata is not None
    model_metadata = cast(dict[str, Any], action.metadata['model'])
    assert model_metadata['supports']['tools'] is True
    assert model_metadata['customOptions']['properties'].get('toolChoice') is not None


@pytest.mark.asyncio
async def test_resolve_ignores_non_model_kinds() -> None:
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.FLOW, 'whatever') is None


@pytest.mark.asyncio
async def test_resolve_returns_an_image_action_for_a_listed_image_model() -> None:
    plugin = Bedrock(region='us-east-1', models=['amazon.titan-image-generator-v1'])
    action = await plugin.resolve(ActionKind.MODEL, 'amazon.titan-image-generator-v1')

    assert action is not None
    assert action.metadata is not None
    model_metadata = cast(dict[str, Any], action.metadata['model'])
    # The open image schema: a strict one would reject every family-specific
    # override at Genkit's request validation.
    assert model_metadata['customOptions']['properties'] == {}
    assert model_metadata['customOptions'].get('additionalProperties') is not False


@pytest.mark.asyncio
async def test_resolve_classifies_an_undeclared_image_model_id() -> None:
    # Lazy resolution means an unlisted image ID would otherwise take the
    # Converse path and only fail at call time.
    plugin = Bedrock(region='us-east-1')
    action = await plugin.resolve(ActionKind.MODEL, 'amazon.nova-canvas-v1:0')

    assert action is not None
    assert action.metadata is not None
    model_metadata = cast(dict[str, Any], action.metadata['model'])
    assert model_metadata['supports']['output'] == ['media']
    assert model_metadata['customOptions']['properties'] == {}


@pytest.mark.asyncio
async def test_list_actions_lists_configured_chat_and_image_models() -> None:
    plugin = Bedrock(
        region='us-east-1',
        models=['amazon.nova-lite-v1:0', 'amazon.titan-image-generator-v1'],
    )
    actions = await plugin.list_actions()

    assert [a.name for a in actions] == [
        'bedrock/amazon.nova-lite-v1:0',
        'bedrock/amazon.titan-image-generator-v1',
    ]
    assert [a.action_type for a in actions] == [ActionKind.MODEL, ActionKind.MODEL]
    assert actions[1].metadata is not None
    image_metadata = cast(dict[str, Any], actions[1].metadata['model'])
    assert image_metadata['supports']['output'] == ['media']
    assert image_metadata['customOptions']['properties'] == {}


@pytest.mark.asyncio
async def test_list_and_resolve_route_a_model_id_the_same_way() -> None:
    # models= carries bare IDs, so both paths infer the route from the ID and
    # the Dev UI row must match the action it resolves to.
    plugin = Bedrock(region='us-east-1', models=['amazon.nova-lite-v1:0', 'amazon.nova-canvas-v1:0'])
    listed = await plugin.list_actions()
    assert [m.name for m in listed] == ['bedrock/amazon.nova-lite-v1:0', 'bedrock/amazon.nova-canvas-v1:0']

    for metadata in listed:
        action = await plugin.resolve(ActionKind.MODEL, metadata.name.removeprefix('bedrock/'))
        assert action is not None and action.metadata is not None and metadata.metadata is not None
        listed_model = cast(dict[str, Any], metadata.metadata['model'])
        resolved_model = cast(dict[str, Any], action.metadata['model'])
        assert listed_model['supports'] == resolved_model['supports']
        assert listed_model['customOptions'] == resolved_model['customOptions']


@pytest.mark.asyncio
async def test_resolve_returns_embedder_action_with_registry_metadata() -> None:
    plugin = Bedrock(region='us-east-1')
    action = await plugin.resolve(ActionKind.EMBEDDER, 'amazon.titan-embed-text-v2:0')

    assert action is not None
    assert action.kind == ActionKind.EMBEDDER
    assert action.name == 'bedrock/amazon.titan-embed-text-v2:0'
    assert action.metadata is not None
    embedder_metadata = cast(dict[str, Any], action.metadata['embedder'])
    assert embedder_metadata['dimensions'] == 1024
    assert embedder_metadata['supports'] == {'input': ['text']}


@pytest.mark.asyncio
async def test_resolve_rejects_embedder_requests_for_chat_models() -> None:
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.EMBEDDER, 'amazon.nova-lite-v1:0') is None


@pytest.mark.asyncio
async def test_resolve_accepts_a_profile_prefixed_embedder_id() -> None:
    plugin = Bedrock(region='us-east-1')
    action = await plugin.resolve(ActionKind.EMBEDDER, 'us.amazon.titan-embed-text-v2:0')
    assert action is not None
    assert action.name == 'bedrock/us.amazon.titan-embed-text-v2:0'


@pytest.mark.asyncio
async def test_embedding_models_do_not_resolve_as_chat_models() -> None:
    # Without the guard this returns a Converse action that only fails when called.
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.MODEL, 'amazon.titan-embed-text-v2:0') is None
    assert await plugin.resolve(ActionKind.MODEL, 'cohere.embed-english-v3') is None


@pytest.mark.asyncio
async def test_the_cross_guard_leaves_cohere_chat_models_alone() -> None:
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.MODEL, 'cohere.command-r-v1:0') is not None


@pytest.mark.asyncio
async def test_rerank_models_do_not_resolve_as_chat_models() -> None:
    # Neither rerank family has a Converse path, so resolving one as a chat
    # model would only defer the failure to call time.
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.MODEL, 'cohere.rerank-v3-5:0') is None
    assert await plugin.resolve(ActionKind.MODEL, 'amazon.rerank-v1:0') is None
    assert await plugin.resolve(ActionKind.EMBEDDER, 'cohere.rerank-v3-5:0') is None
    assert await plugin.resolve(ActionKind.EMBEDDER, 'amazon.rerank-v1:0') is None


@pytest.mark.asyncio
async def test_cohere_v4_resolves_but_fails_when_called() -> None:
    plugin = Bedrock(region='us-east-1')
    action = await plugin.resolve(ActionKind.EMBEDDER, 'cohere.embed-v4:0')

    assert action is not None
    with pytest.raises(GenkitError, match='Cohere Embed v4') as excinfo:
        await action.run(EmbedRequest(input=[Document.from_text('hi')]))
    assert excinfo.value.status == 'UNIMPLEMENTED'


@pytest.mark.asyncio
async def test_an_unknown_embedding_family_resolves_but_fails_when_called() -> None:
    # A family with no request shape here is still an embedder, not a chat model.
    plugin = Bedrock(region='us-east-1')
    model_id = 'amazon.some-new-embed-v9:0'

    assert await plugin.resolve(ActionKind.MODEL, model_id) is None
    action = await plugin.resolve(ActionKind.EMBEDDER, model_id)

    assert action is not None
    with pytest.raises(GenkitError, match='unsupported embedding model') as excinfo:
        await action.run(EmbedRequest(input=[Document.from_text('hi')]))
    assert excinfo.value.status == 'UNIMPLEMENTED'


@pytest.mark.asyncio
async def test_list_actions_includes_configured_embedders() -> None:
    plugin = Bedrock(
        region='us-east-1',
        models=['amazon.nova-lite-v1:0'],
        embedders=['amazon.titan-embed-text-v2:0', 'cohere.embed-english-v3'],
    )
    actions = await plugin.list_actions()

    assert [a.name for a in actions] == [
        'bedrock/amazon.nova-lite-v1:0',
        'bedrock/amazon.titan-embed-text-v2:0',
        'bedrock/cohere.embed-english-v3',
    ]
    assert [a.action_type for a in actions[1:]] == [ActionKind.EMBEDDER, ActionKind.EMBEDDER]


@pytest.mark.asyncio
async def test_everything_listed_can_actually_resolve() -> None:
    # An ID in the wrong list used to be advertised anyway, leaving a Dev UI row
    # that answers 404 in both directions.
    plugin = Bedrock(
        region='us-east-1',
        models=['amazon.nova-lite-v1:0', 'amazon.nova-canvas-v1:0', 'amazon.titan-embed-text-v2:0'],
        embedders=['cohere.embed-english-v3', 'amazon.nova-lite-v1:0'],
    )
    listed = await plugin.list_actions()

    assert [a.name for a in listed] == [
        'bedrock/amazon.nova-lite-v1:0',
        'bedrock/amazon.nova-canvas-v1:0',
        'bedrock/cohere.embed-english-v3',
    ]
    for metadata in listed:
        assert metadata.action_type is not None
        assert (
            await plugin.resolve(ActionKind(metadata.action_type), metadata.name.removeprefix('bedrock/')) is not None
        )


@pytest.mark.asyncio
async def test_a_rerank_id_in_models_is_not_listed() -> None:
    # The other side of the listed-can-resolve invariant: the plugin has no
    # rerank action, so a rerank ID has nothing to advertise.
    plugin = Bedrock(
        region='us-east-1',
        models=['amazon.nova-lite-v1:0', 'cohere.rerank-v3-5:0', 'amazon.rerank-v1:0'],
    )
    listed = await plugin.list_actions()
    assert [a.name for a in listed] == ['bedrock/amazon.nova-lite-v1:0']


def test_a_bare_string_id_list_is_rejected() -> None:
    # A missing bracket would iterate the string and list one model per character.
    with pytest.raises(TypeError, match=r"models=\['amazon.nova-lite-v1:0'\]"):
        Bedrock(region='us-east-1', models=cast(Any, 'amazon.nova-lite-v1:0'))
    with pytest.raises(TypeError, match=r"embedders=\['amazon.titan-embed-text-v2:0'\]"):
        Bedrock(region='us-east-1', embedders=cast(Any, 'amazon.titan-embed-text-v2:0'))


@pytest.mark.asyncio
async def test_resolve_and_list_publish_the_same_embedder_metadata() -> None:
    # One helper feeds both paths, so the Dev UI listing can never disagree
    # with what the resolved action reports.
    plugin = Bedrock(region='us-east-1', embedders=['cohere.embed-multilingual-v3'])
    action = await plugin.resolve(ActionKind.EMBEDDER, 'cohere.embed-multilingual-v3')
    listed = await plugin.list_actions()

    assert action is not None
    assert action.metadata is not None and listed[0].metadata is not None
    assert action.metadata['embedder'] == listed[0].metadata['embedder']


class FakeImageTransport:
    """Stands in for BedrockTransport; records the InvokeModel kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {'images': ['modern-image'], 'finish_reasons': ['SUCCESS']}


@pytest.mark.asyncio
async def test_image_config_survives_the_action_boundary() -> None:
    # The unit tests call the image model directly; only this path proves a
    # family-specific key survives Genkit's config handling on the way in.
    plugin = Bedrock(region='us-east-1', models=['stability.sd3-5-large-v1:0'])
    transport = FakeImageTransport()
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001
    action = await plugin.resolve(ActionKind.MODEL, 'stability.sd3-5-large-v1:0')
    assert action is not None

    # Validated from the wire shape, so the config goes through the same
    # coercion a Dev UI or flow call would put it through.
    request = ModelRequest.model_validate({
        'messages': [{'role': 'user', 'content': [{'text': 'a coral reef'}]}],
        'config': {'aspect_ratio': '16:9'},
    })
    response = cast(ModelResponse, (await action.run(request)).response)

    assert json.loads(transport.calls[0]['body'])['aspect_ratio'] == '16:9'
    assert response.message is not None
    part = response.message.content[0]
    assert part.media is not None
    assert part.media.url == 'data:image/png;base64,modern-image'


@pytest.mark.asyncio
async def test_genkit_generic_config_never_reaches_the_image_body() -> None:
    # The framework coerces a raw mapping into ModelConfig, so its own knobs
    # arrive as declared fields; only this path proves they are dropped.
    plugin = Bedrock(region='us-east-1', models=['stability.sd3-5-large-v1:0'])
    transport = FakeImageTransport()
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001
    action = await plugin.resolve(ActionKind.MODEL, 'stability.sd3-5-large-v1:0')
    assert action is not None

    request = ModelRequest.model_validate({
        'messages': [{'role': 'user', 'content': [{'text': 'a coral reef'}]}],
        'config': {'aspect_ratio': '16:9', 'temperature': 0.9},
    })
    await action.run(request)

    assert json.loads(transport.calls[0]['body']) == {
        'prompt': 'a coral reef',
        'output_format': 'png',
        'aspect_ratio': '16:9',
    }


@pytest.mark.parametrize('spelling', ['api_key', 'apiKey'])
@pytest.mark.asyncio
async def test_image_action_config_api_key_raises_and_sends_nothing(spelling: str) -> None:
    """A key in image config raises INVALID_ARGUMENT naming context.secrets, like every other model."""
    plugin = Bedrock(region='us-east-1', models=['stability.sd3-5-large-v1:0'])
    transport = FakeImageTransport()
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001
    action = await plugin.resolve(ActionKind.MODEL, 'stability.sd3-5-large-v1:0')
    assert action is not None

    with pytest.raises(GenkitError) as err:
        await action.run({
            'messages': [{'role': 'user', 'content': [{'text': 'a coral reef'}]}],
            'config': {'aspect_ratio': '16:9', spelling: 'SECRET-VALUE'},
        })

    assert err.value.status == 'INVALID_ARGUMENT'
    assert "context={'secrets': {'api_key': ...}}" in str(err.value)
    assert 'SECRET-VALUE' not in str(err.value)
    assert transport.calls == []


class FakeConverseTransport:
    """Stands in for BedrockTransport; records the Converse kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ensure_client(self) -> None:
        return None

    async def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            'output': {'message': {'role': 'assistant', 'content': [{'text': 'hello'}]}},
            'stopReason': 'end_turn',
        }


def _genkit_with_fake_converse() -> tuple[Genkit, FakeConverseTransport]:
    plugin = Bedrock(region='us-east-1')
    transport = FakeConverseTransport()
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001
    return Genkit(plugins=[plugin]), transport


@pytest.mark.asyncio
async def test_generate_bedrock_model_resolves_with_bare_id() -> None:
    """`bedrock/anthropic.claude-…` still resolves, and Converse gets the id without `bedrock/`."""
    ai, transport = _genkit_with_fake_converse()

    response = await ai.generate(model='bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0', prompt='hi')

    assert response.text == 'hello'
    assert transport.calls[0]['modelId'] == 'anthropic.claude-sonnet-4-5-20250929-v1:0'


@pytest.mark.asyncio
async def test_generate_bedrock_arn_model_keeps_full_path() -> None:
    """A Bedrock inference-profile ARN with `/` keeps the whole ARN."""
    arn = 'arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0'
    ai, transport = _genkit_with_fake_converse()

    response = await ai.generate(model=f'bedrock/{arn}', prompt='hi')

    assert response.text == 'hello'
    assert transport.calls[0]['modelId'] == arn
    action = await ai.registry.resolve_action(ActionKind.MODEL, f'bedrock/{arn}')
    assert action is not None
    assert action.name == f'bedrock/{arn}'


@pytest.mark.asyncio
async def test_generate_with_a_rerank_id_raises_not_found_and_sends_nothing() -> None:
    """The plugin has no rerank action, so a rerank ID fails as a GenkitError before any AWS call."""
    ai, transport = _genkit_with_fake_converse()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='bedrock/cohere.rerank-v3-5:0', prompt='hi')

    assert err.value.status == 'NOT_FOUND'
    assert transport.calls == []
