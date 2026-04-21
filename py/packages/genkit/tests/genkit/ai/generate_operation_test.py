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

"""Tests for the generate_operation method.

This module tests the generate_operation method which is used for long-running
model operations (like video generation with Veo). The method:

- Only works with models that support long-running operations
- Returns an Operation that can be polled with check_operation()
- Throws errors if the model doesn't support long-running ops

Test Coverage
=============

┌─────────────────────────────────────────────────────────────────────────────┐
│ Test Case                              │ Description                        │
├────────────────────────────────────────┼────────────────────────────────────┤
│ test_no_model_specified                │ Error when no model provided       │
│ test_model_not_found                   │ Error when model doesn't exist     │
│ test_model_no_long_running_support     │ Error when model lacks LRO support │
│ test_model_no_operation_returned       │ Error when no operation returned   │
│ test_long_running_model_success        │ Success path for LRO models        │
└────────────────────────────────────────┴────────────────────────────────────┘

Cross-Language Parity:
    - JavaScript: js/ai/src/generate.ts (generateOperation function)

Note:
    This is a beta feature matching the JS implementation. Only models that
    explicitly support long-running operations (model.supports.longRunning=True)
    can be used with generate_operation().
"""

import pytest

from genkit import Genkit, Message, ModelResponse
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import ModelRequest
from genkit._core._typing import (
    ModelInfo,
    Operation,
    Part,
    Role,
    Supports,
    TextPart,
)


@pytest.fixture
def ai() -> Genkit:
    """Create a fresh Genkit instance for each test."""
    return Genkit()


@pytest.mark.asyncio
async def test_generate_operation_no_model_specified(ai: Genkit) -> None:
    """Test that generate_operation raises error when no model specified."""
    with pytest.raises(GenkitError) as exc_info:
        await ai.generate_operation(prompt='Hi')

    assert 'No model specified' in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_operation_model_not_found(ai: Genkit) -> None:
    """Test that generate_operation raises error when model not found."""
    with pytest.raises(GenkitError) as exc_info:
        await ai.generate_operation(model='nonexistent/model', prompt='Hi')

    assert 'not found' in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_generate_operation_model_no_long_running_support(ai: Genkit) -> None:
    """Test that generate_operation raises error when model doesn't support long-running.

    This matches the JS behavior where models must have supports.longRunning=True.
    """

    # Define a standard model without long_running support
    async def model_fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='Hello'))],
            ),
        )

    ai.define_model(
        name='standard-model',
        fn=model_fn,
        info=ModelInfo(
            supports=Supports(
                multiturn=True,
                tools=True,
                media=False,
                long_running=False,  # Not a long-running model
            ),
        ),
    )

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate_operation(model='standard-model', prompt='Hi')

    assert 'does not support long running operations' in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_operation_model_no_supports_info(ai: Genkit) -> None:
    """Test that models without supports info are rejected."""

    # Define a model without any ModelInfo
    async def model_fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='Hello'))],
            ),
        )

    ai.define_model(name='no-info-model', fn=model_fn)

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate_operation(model='no-info-model', prompt='Hi')

    assert 'does not support long running operations' in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_operation_no_operation_returned(ai: Genkit) -> None:
    """Test error when model supports LRO but doesn't return an operation.

    This matches the JS FAILED_PRECONDITION error case.
    """

    # Define a model that claims to support long_running but doesn't return an operation
    async def model_fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        # Return a normal response without an operation
        return ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='Hello'))],
            ),
        )

    ai.define_model(
        name='fake-lro-model',
        fn=model_fn,
        info=ModelInfo(
            supports=Supports(
                long_running=True,  # Claims to support LRO
            ),
        ),
    )

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate_operation(model='fake-lro-model', prompt='Hi')

    assert 'did not return an operation' in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_operation_success_with_lro_model(ai: Genkit) -> None:
    """Test successful generate_operation with a proper long-running model."""
    expected_operation = Operation(
        id='test-operation-123',
        done=False,
        action='/background-model/lro-model',
    )

    # Define a model that supports long_running and returns an operation
    async def model_fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='Started'))],
            ),
            operation=expected_operation,
        )

    ai.define_model(
        name='lro-model',
        fn=model_fn,
        info=ModelInfo(
            supports=Supports(
                long_running=True,
            ),
        ),
    )

    operation = await ai.generate_operation(model='lro-model', prompt='Generate video')

    assert isinstance(operation, Operation)
    assert operation.id == 'test-operation-123'
    assert operation.done is False
    assert operation.action == '/background-model/lro-model'


@pytest.mark.asyncio
async def test_generate_operation_with_default_model(ai: Genkit) -> None:
    """Test generate_operation uses default model when set."""
    expected_operation = Operation(
        id='default-op-456',
        done=False,
    )

    async def model_fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='Started'))],
            ),
            operation=expected_operation,
        )

    ai.define_model(
        name='default-lro-model',
        fn=model_fn,
        info=ModelInfo(
            supports=Supports(
                long_running=True,
            ),
        ),
    )

    # Create a new Genkit instance with the default model set
    ai_with_default = Genkit(model='default-lro-model')
    # Re-register the model on the new instance
    ai_with_default.define_model(
        name='default-lro-model',
        fn=model_fn,
        info=ModelInfo(
            supports=Supports(
                long_running=True,
            ),
        ),
    )

    operation = await ai_with_default.generate_operation(prompt='Generate video')

    assert isinstance(operation, Operation)
    assert operation.id == 'default-op-456'


@pytest.mark.asyncio
async def test_generate_operation_passes_all_options(ai: Genkit) -> None:
    """Test that generate_operation passes all options to generate()."""
    captured_request: ModelRequest | None = None
    expected_operation = Operation(id='opt-test-789', done=False)

    async def model_fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        nonlocal captured_request
        captured_request = request
        return ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='Started'))],
            ),
            operation=expected_operation,
        )

    ai.define_model(
        name='options-test-model',
        fn=model_fn,
        info=ModelInfo(
            supports=Supports(
                long_running=True,
            ),
        ),
    )

    await ai.generate_operation(
        model='options-test-model',
        prompt='Test prompt',
        system='You are a test assistant',
        config={'temperature': 0.7},
    )

    assert captured_request is not None
    # Verify config was passed
    assert captured_request.config is not None
