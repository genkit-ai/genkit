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

from types import SimpleNamespace
from typing import Any, cast

import boto3.session
import pytest
from genkit_amazon_bedrock import Bedrock, BedrockConfig, ModelDefinition, bedrock_name

from genkit import Document
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


def test_config_accepts_camel_case_and_extra_fields() -> None:
    config = BedrockConfig.model_validate({
        'toolChoice': 'auto',
        'maxOutputTokens': 128,
        'additionalModelRequestFields': {'thinking': {'type': 'enabled'}},
        'someFutureKnob': True,
    })
    assert config.tool_choice == 'auto'
    assert config.max_output_tokens == 128
    assert config.additional_model_request_fields == {'thinking': {'type': 'enabled'}}


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
async def test_resolve_skips_non_chat_model_definitions() -> None:
    plugin = Bedrock(
        region='us-east-1',
        models=[ModelDefinition(name='amazon.titan-image-generator-v1', type='image')],
    )
    assert await plugin.resolve(ActionKind.MODEL, bedrock_name('amazon.titan-image-generator-v1')) is None


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
async def test_list_actions_lists_configured_chat_models() -> None:
    plugin = Bedrock(
        region='us-east-1',
        models=[
            ModelDefinition(name='amazon.nova-lite-v1:0'),
            ModelDefinition(name='amazon.titan-image-generator-v1', type='image'),
        ],
    )
    actions = await plugin.list_actions()
    assert [a.name for a in actions] == ['bedrock/amazon.nova-lite-v1:0']
    assert actions[0].action_type == ActionKind.MODEL


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
    assert await plugin.resolve(ActionKind.MODEL, bedrock_name('cohere.rerank-v3-5:0')) is not None


@pytest.mark.asyncio
async def test_cohere_v4_resolves_but_fails_when_called() -> None:
    plugin = Bedrock(region='us-east-1')
    action = await plugin.resolve(ActionKind.EMBEDDER, bedrock_name('cohere.embed-v4:0'))

    assert action is not None
    with pytest.raises(GenkitError, match='Cohere Embed v4') as excinfo:
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
            ModelDefinition(name='amazon.titan-embed-text-v2:0'),
        ],
        embedders=['cohere.embed-english-v3', 'amazon.nova-lite-v1:0'],
    )
    listed = await plugin.list_actions()

    assert [a.name for a in listed] == ['bedrock/amazon.nova-lite-v1:0', 'bedrock/cohere.embed-english-v3']
    for metadata in listed:
        assert metadata.action_type is not None
        assert await plugin.resolve(ActionKind(metadata.action_type), metadata.name) is not None


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
