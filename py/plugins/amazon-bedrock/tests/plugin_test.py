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

"""Tests for the Amazon Bedrock plugin scaffold."""

import pytest

from genkit.plugins.amazon_bedrock import Bedrock, BedrockConfig, ModelDefinition, bedrock_name


def test_plugin_name() -> None:
    plugin = Bedrock()
    assert plugin.name == 'bedrock'


def test_bedrock_name_prefixes_model_id() -> None:
    assert bedrock_name('anthropic.claude-sonnet-4-5-20250929-v1:0') == (
        'bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0'
    )


def test_constructor_defaults() -> None:
    plugin = Bedrock()
    assert plugin.region == 'us-east-1'
    assert plugin.max_retries == 3
    assert plugin.request_timeout == 30.0
    assert plugin.models == []
    assert plugin.embedders == []


def test_model_definition_defaults_to_chat() -> None:
    model = ModelDefinition(name='amazon.nova-lite-v1:0')
    assert model.type == 'chat'


def test_config_accepts_camel_case_and_extra_fields() -> None:
    config = BedrockConfig.model_validate({
        'toolChoice': 'auto',
        'additionalModelRequestFields': {'thinking': {'type': 'enabled'}},
        'someFutureKnob': True,
    })
    assert config.tool_choice == 'auto'
    assert config.additional_model_request_fields == {'thinking': {'type': 'enabled'}}


@pytest.mark.asyncio
async def test_init_returns_no_eager_actions() -> None:
    plugin = Bedrock()
    assert await plugin.init() == []


@pytest.mark.asyncio
async def test_resolve_returns_none_until_actions_land() -> None:
    from genkit.plugin_api import ActionKind

    plugin = Bedrock()
    assert await plugin.resolve(ActionKind.MODEL, bedrock_name('amazon.nova-lite-v1:0')) is None
    assert await plugin.resolve(ActionKind.FLOW, 'bedrock/whatever') is None
