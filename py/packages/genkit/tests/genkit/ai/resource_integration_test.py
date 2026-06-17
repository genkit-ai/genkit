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

import pytest

from genkit import Message, ModelResponse
from genkit._ai._generate import generate_action
from genkit._ai._resource import ResourceInput, ResourceOutput, define_resource
from genkit._core._action import ActionRunContext
from genkit._core._model import GenerateActionOptions, ModelRequest
from genkit._core._registry import ActionKind, Registry
from genkit._core._typing import (
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
    async def mock_model(input: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        # Verify docs are EMPTY (not auto-populated)
        assert not input.docs
        # Access via root because DocumentPart is a RootModel
        # Verify the message content was hydrated (replaced resource part with text part)
        assert input.messages[0].content[0].root.text == 'Resource content for test://foo'
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Done'))]))

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
