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
    assert plugin.max_retries == 3
    assert plugin.read_timeout == 3600.0
    assert plugin.connect_timeout == 60.0
    assert plugin.max_pool_connections == 50
    assert plugin.models == []
    assert plugin.embedders == []


def test_model_definition_defaults_to_chat() -> None:
    model = ModelDefinition(name='amazon.nova-lite-v1:0')
    assert model.type == 'chat'


def test_config_accepts_camel_case_and_extra_fields() -> None:
    config = BedrockConfig.model_validate({
        'toolChoice': 'auto',
        'maxTokens': 128,
        'additionalModelRequestFields': {'thinking': {'type': 'enabled'}},
        'someFutureKnob': True,
    })
    assert config.tool_choice == 'auto'
    assert config.max_tokens == 128
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
