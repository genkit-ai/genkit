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


"""Integration tests for Genkit resources."""

from typing import cast

import pytest

from genkit.blocks.generate import generate_action
from genkit.blocks.resource import ResourceInput, ResourceOutput, define_resource, resource
from genkit.core.action import ActionRunContext
from genkit.core.registry import ActionKind, Registry
from genkit.core.typing import (
    GenerateActionOptions,
    GenerateRequest,
    GenerateResponse,
    Message,
    Part,
    Resource1,
    ResourcePart,
    Role,
    TextPart,
)


@pytest.mark.asyncio
async def test_generate_with_resources() -> None:
    """Test calling generate with resources."""
    registry = Registry()

    # 1. Register a resource
    async def my_resource(input: ResourceInput, ctx: ActionRunContext) -> ResourceOutput:
        return ResourceOutput(content=[Part(root=TextPart(text=f'Resource content for {input.uri}'))])

    define_resource(registry, {'uri': 'test://foo'}, my_resource)

    # 2. Register a mock model
    async def mock_model(input: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
        # Verify docs are EMPTY (not auto-populated)
        assert not input.docs
        # Access via root because DocumentPart is a RootModel
        # Verify the message content was hydrated (replaced resource part with text part)
        assert input.messages[0].content[0].root.text == 'Resource content for test://foo'
        return GenerateResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Done'))]))

    registry.register_action(ActionKind.MODEL, 'mock-model', mock_model)

    options = GenerateActionOptions(
        model='mock-model',
        messages=[Message(role=Role.USER, content=[Part(root=ResourcePart(resource=Resource1(uri='test://foo')))])],
        resources=['test://foo'],
    )

    response = await generate_action(registry, options)
    # Part also uses RootModel, access via root
    assert response.message is not None
    assert response.message.content[0].root.text == 'Done'


@pytest.mark.asyncio
async def test_dynamic_action_provider_resource() -> None:
    """Test dynamic action provider with resources."""
    registry = Registry()

    # Register a dynamic provider that handles any "dynamic://*" uri
    def provider_fn(input: dict[str, object], ctx: ActionRunContext) -> object:
        kind = cast(ActionKind, input['kind'])
        name = cast(str, input['name'])
        if kind == ActionKind.RESOURCE and name.startswith('dynamic://'):

            async def dyn_res_fn(input: ResourceInput, ctx: ActionRunContext) -> ResourceOutput:
                return ResourceOutput(content=[Part(root=TextPart(text=f'Dynamic content for {input.uri}'))])

            return resource({'uri': name}, dyn_res_fn)
        return None

    # Register the provider as an action (it effectively acts as a factory)
    # Note: Accessing internal structure for test setup as register_action expects specific signature
    # But we want to register it under DYNAMIC_ACTION_PROVIDER kind.
    registry.register_action(kind=ActionKind.DYNAMIC_ACTION_PROVIDER, name='test-provider', fn=provider_fn)

    # Register mock model
    # Register mock model
    async def mock_model(input: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
        # Verify docs are empty
        assert not input.docs
        # Verify dynamic hydration
        assert input.messages[0].content[0].root.text == 'Dynamic content for dynamic://bar'
        return GenerateResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Done'))]))

    registry.register_action(ActionKind.MODEL, 'mock-model', mock_model)

    options = GenerateActionOptions(
        model='mock-model',
        messages=[Message(role=Role.USER, content=[Part(root=ResourcePart(resource=Resource1(uri='dynamic://bar')))])],
        resources=['dynamic://bar'],
    )

    response = await generate_action(registry, options)
    assert response.message is not None
    assert response.message.content[0].root.text == 'Done'
