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
from genkit_amazon_bedrock import Bedrock, BedrockConfig, ModelDefinition, bedrock_name
from genkit_amazon_bedrock.transport import BedrockTransport
from pydantic import ValidationError

from genkit import Document, MediaPart, ModelRequest, ModelResponse
from genkit.embedder import EmbedRequest
from genkit.plugin_api import ActionKind, GenkitError


def test_plugin_name() -> None:
    plugin = Bedrock()
    assert plugin.name == 'bedrock'


def test_bedrock_name_prefixes_model_id() -> None:
    assert bedrock_name('anthropic.claude-sonnet-4-5-20250929-v1:0') == (
        'bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0'
    )


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


def test_model_definition_defaults_to_chat() -> None:
    model = ModelDefinition(name='amazon.nova-lite-v1:0')
    assert model.type == 'chat'


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
    action = await plugin.resolve(ActionKind.MODEL, bedrock_name('amazon.nova-lite-v1:0'))
    assert action is not None
    assert action.name == 'bedrock/amazon.nova-lite-v1:0'
    assert action.metadata is not None
    model_metadata = cast(dict[str, Any], action.metadata['model'])
    assert model_metadata['supports']['tools'] is True
    assert model_metadata['customOptions']['properties'].get('toolChoice') is not None


@pytest.mark.asyncio
async def test_resolve_ignores_non_model_kinds() -> None:
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.FLOW, 'bedrock/whatever') is None


@pytest.mark.asyncio
async def test_resolve_ignores_other_plugin_namespaces() -> None:
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.MODEL, 'openai/gpt-4') is None


@pytest.mark.asyncio
async def test_resolve_returns_an_image_action_for_a_declared_image_model() -> None:
    plugin = Bedrock(
        region='us-east-1',
        models=[ModelDefinition(name='amazon.titan-image-generator-v1', type='image')],
    )
    action = await plugin.resolve(ActionKind.MODEL, bedrock_name('amazon.titan-image-generator-v1'))

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
    action = await plugin.resolve(ActionKind.MODEL, bedrock_name('amazon.nova-canvas-v1:0'))

    assert action is not None
    assert action.metadata is not None
    model_metadata = cast(dict[str, Any], action.metadata['model'])
    assert model_metadata['supports']['output'] == ['media']
    assert model_metadata['customOptions']['properties'] == {}


@pytest.mark.asyncio
async def test_a_declared_chat_type_wins_over_id_classification() -> None:
    plugin = Bedrock(
        region='us-east-1',
        models=[ModelDefinition(name='amazon.nova-canvas-v1:0', type='chat')],
    )
    action = await plugin.resolve(ActionKind.MODEL, bedrock_name('amazon.nova-canvas-v1:0'))

    assert action is not None
    assert action.metadata is not None
    model_metadata = cast(dict[str, Any], action.metadata['model'])
    assert model_metadata['customOptions']['properties'].get('toolChoice') is not None


@pytest.mark.asyncio
async def test_text_type_routes_like_chat() -> None:
    plugin = Bedrock(
        region='us-east-1',
        models=[ModelDefinition(name='meta.llama3-8b-instruct-v1:0', type='text')],
    )
    action = await plugin.resolve(ActionKind.MODEL, bedrock_name('meta.llama3-8b-instruct-v1:0'))
    assert action is not None
    actions = await plugin.list_actions()
    assert [a.name for a in actions] == ['bedrock/meta.llama3-8b-instruct-v1:0']


@pytest.mark.asyncio
async def test_list_actions_lists_configured_chat_and_image_models() -> None:
    plugin = Bedrock(
        region='us-east-1',
        models=[
            ModelDefinition(name='amazon.nova-lite-v1:0'),
            ModelDefinition(name='amazon.titan-image-generator-v1', type='image'),
        ],
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
async def test_a_chat_model_declared_as_an_image_model_is_not_listed() -> None:
    plugin = Bedrock(
        region='us-east-1',
        models=[
            ModelDefinition(name='amazon.nova-lite-v1:0'),
            ModelDefinition(name='anthropic.claude-sonnet-4-5-20250929-v1:0', type='image'),
        ],
    )
    listed = await plugin.list_actions()

    # Listing it would advertise image metadata this model can never honour.
    assert [a.name for a in listed] == ['bedrock/amazon.nova-lite-v1:0']

    # It still resolves, so the caller reads why it cannot work, not a 404.
    action = await plugin.resolve(ActionKind.MODEL, bedrock_name('anthropic.claude-sonnet-4-5-20250929-v1:0'))
    assert action is not None
    request = ModelRequest.model_validate({'messages': [{'role': 'user', 'content': [{'text': 'a reef'}]}]})
    with pytest.raises(GenkitError, match='unsupported image generation model') as excinfo:
        await action.run(request)
    assert excinfo.value.status == 'UNIMPLEMENTED'


@pytest.mark.asyncio
async def test_resolve_returns_embedder_action_with_registry_metadata() -> None:
    plugin = Bedrock(region='us-east-1')
    action = await plugin.resolve(ActionKind.EMBEDDER, bedrock_name('amazon.titan-embed-text-v2:0'))

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
    assert await plugin.resolve(ActionKind.EMBEDDER, bedrock_name('amazon.nova-lite-v1:0')) is None


@pytest.mark.asyncio
async def test_resolve_ignores_foreign_namespaces_for_embedders() -> None:
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.EMBEDDER, 'openai/text-embedding-3-small') is None


@pytest.mark.asyncio
async def test_resolve_accepts_a_profile_prefixed_embedder_id() -> None:
    plugin = Bedrock(region='us-east-1')
    action = await plugin.resolve(ActionKind.EMBEDDER, bedrock_name('us.amazon.titan-embed-text-v2:0'))
    assert action is not None
    assert action.name == 'bedrock/us.amazon.titan-embed-text-v2:0'


@pytest.mark.asyncio
async def test_embedding_models_do_not_resolve_as_chat_models() -> None:
    # Without the guard this returns a Converse action that only fails when called.
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.MODEL, bedrock_name('amazon.titan-embed-text-v2:0')) is None
    assert await plugin.resolve(ActionKind.MODEL, bedrock_name('cohere.embed-english-v3')) is None


@pytest.mark.asyncio
async def test_the_cross_guard_leaves_cohere_chat_models_alone() -> None:
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.MODEL, bedrock_name('cohere.command-r-v1:0')) is not None


@pytest.mark.asyncio
async def test_rerank_models_do_not_resolve_as_chat_models() -> None:
    # Neither rerank family has a Converse path, so resolving one as a chat
    # model would only defer the failure to call time.
    plugin = Bedrock(region='us-east-1')
    assert await plugin.resolve(ActionKind.MODEL, bedrock_name('cohere.rerank-v3-5:0')) is None
    assert await plugin.resolve(ActionKind.MODEL, bedrock_name('amazon.rerank-v1:0')) is None
    assert await plugin.resolve(ActionKind.EMBEDDER, bedrock_name('cohere.rerank-v3-5:0')) is None
    assert await plugin.resolve(ActionKind.EMBEDDER, bedrock_name('amazon.rerank-v1:0')) is None


@pytest.mark.asyncio
async def test_cohere_v4_resolves_but_fails_when_called() -> None:
    plugin = Bedrock(region='us-east-1')
    action = await plugin.resolve(ActionKind.EMBEDDER, bedrock_name('cohere.embed-v4:0'))

    assert action is not None
    with pytest.raises(GenkitError, match='Cohere Embed v4') as excinfo:
        await action.run(EmbedRequest(input=[Document.from_text('hi')]))
    assert excinfo.value.status == 'UNIMPLEMENTED'


@pytest.mark.asyncio
async def test_an_unknown_embedding_family_resolves_but_fails_when_called() -> None:
    # A family with no request shape here is still an embedder, not a chat model.
    plugin = Bedrock(region='us-east-1')
    model_id = 'amazon.some-new-embed-v9:0'

    assert await plugin.resolve(ActionKind.MODEL, bedrock_name(model_id)) is None
    action = await plugin.resolve(ActionKind.EMBEDDER, bedrock_name(model_id))

    assert action is not None
    with pytest.raises(GenkitError, match='unsupported embedding model') as excinfo:
        await action.run(EmbedRequest(input=[Document.from_text('hi')]))
    assert excinfo.value.status == 'UNIMPLEMENTED'


@pytest.mark.asyncio
async def test_list_actions_includes_configured_embedders() -> None:
    plugin = Bedrock(
        region='us-east-1',
        models=[ModelDefinition(name='amazon.nova-lite-v1:0')],
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
        models=[
            ModelDefinition(name='amazon.nova-lite-v1:0'),
            ModelDefinition(name='amazon.nova-canvas-v1:0', type='image'),
            ModelDefinition(name='amazon.titan-embed-text-v2:0'),
        ],
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
        assert await plugin.resolve(ActionKind(metadata.action_type), metadata.name) is not None


@pytest.mark.asyncio
async def test_a_declared_rerank_model_is_not_listed() -> None:
    # The other side of the listed-can-resolve invariant: reranking is the
    # Bedrock.rerank helper, so a rerank ID has no action to advertise.
    plugin = Bedrock(
        region='us-east-1',
        models=[
            ModelDefinition(name='amazon.nova-lite-v1:0'),
            ModelDefinition(name='cohere.rerank-v3-5:0'),
            ModelDefinition(name='amazon.rerank-v1:0'),
        ],
    )
    listed = await plugin.list_actions()
    assert [a.name for a in listed] == ['bedrock/amazon.nova-lite-v1:0']


@pytest.mark.asyncio
async def test_resolve_and_list_publish_the_same_embedder_metadata() -> None:
    # One helper feeds both paths, so the Dev UI listing can never disagree
    # with what the resolved action reports.
    plugin = Bedrock(region='us-east-1', embedders=['cohere.embed-multilingual-v3'])
    action = await plugin.resolve(ActionKind.EMBEDDER, bedrock_name('cohere.embed-multilingual-v3'))
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
    plugin = Bedrock(
        region='us-east-1',
        models=[ModelDefinition(name='stability.sd3-5-large-v1:0', type='image')],
    )
    transport = FakeImageTransport()
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001
    action = await plugin.resolve(ActionKind.MODEL, bedrock_name('stability.sd3-5-large-v1:0'))
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
    part = response.message.content[0].root
    assert isinstance(part, MediaPart)
    assert part.media.url == 'data:image/png;base64,modern-image'


@pytest.mark.asyncio
async def test_genkit_generic_config_never_reaches_the_image_body() -> None:
    # The framework coerces a raw mapping into ModelConfig, so its own knobs
    # arrive as declared fields; only this path proves they are dropped, and an
    # api_key on an InvokeModel body would be a credential on the wire.
    plugin = Bedrock(
        region='us-east-1',
        models=[ModelDefinition(name='stability.sd3-5-large-v1:0', type='image')],
    )
    transport = FakeImageTransport()
    plugin._transport = cast(BedrockTransport, transport)  # noqa: SLF001
    action = await plugin.resolve(ActionKind.MODEL, bedrock_name('stability.sd3-5-large-v1:0'))
    assert action is not None

    request = ModelRequest.model_validate({
        'messages': [{'role': 'user', 'content': [{'text': 'a coral reef'}]}],
        'config': {'aspect_ratio': '16:9', 'temperature': 0.9, 'api_key': 'SECRET-VALUE'},
    })
    await action.run(request)

    body = transport.calls[0]['body']
    assert json.loads(body) == {'prompt': 'a coral reef', 'output_format': 'png', 'aspect_ratio': '16:9'}
    assert 'SECRET-VALUE' not in body
