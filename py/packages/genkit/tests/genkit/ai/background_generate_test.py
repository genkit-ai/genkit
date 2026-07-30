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

"""Tests for background model generate() and generate_operation() plumbing."""

from typing import Any

import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, Part, Role, TextPart
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import ModelRequest
from genkit._core._typing import ModelInfo, Operation, Supports


@pytest.fixture
def ai() -> Genkit:
    """Create a fresh Genkit instance for each test."""
    return Genkit()


async def register_bg_model(ai: Genkit, *, op_id: str = 'bg-op-123') -> None:
    async def start(request: ModelRequest, _: ActionRunContext) -> Operation:
        return Operation(id=op_id, done=False)

    async def check(op: Operation) -> Operation:
        return op

    ai.define_background_model(
        name='bg-model',
        start=start,
        check=check,
        info=ModelInfo(supports=Supports(long_running=True)),
    )


class PluginConfig(BaseModel):
    thinking_summaries: str | None = None
    google_search: bool | None = None


@pytest.mark.asyncio
async def test_background_start_receives_plugin_config_schema(ai: Genkit) -> None:
    """Bare ModelRequest configs are re-validated as the plugin config schema."""
    seen: dict[str, Any] = {}

    async def start(request: ModelRequest[PluginConfig], _: ActionRunContext) -> Operation:
        seen['config'] = request.config
        return Operation(id='cfg-op', done=False)

    async def check(op: Operation) -> Operation:
        return op

    action = ai.define_background_model(
        name='cfg-model',
        start=start,
        check=check,
        config_schema=PluginConfig,
        info=ModelInfo(supports=Supports(long_running=True)),
    )

    # Mimic generate: bare ModelRequest keeps a dict; Action coerces to the plugin schema.
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(TextPart(text='go'))])],
        config={'thinking_summaries': 'auto', 'google_search': True},
    )
    assert request.config == {'thinking_summaries': 'auto', 'google_search': True}

    await action.start(request)

    config = seen['config']
    assert isinstance(config, PluginConfig)
    assert config.thinking_summaries == 'auto'
    assert config.google_search is True


@pytest.mark.asyncio
async def test_generate_returns_operation_for_background_model(ai: Genkit) -> None:
    """generate() wraps a background model Operation in ModelResponse."""
    await register_bg_model(ai)

    response = await ai.generate(model='bg-model', prompt='a cat surfing')

    assert response.operation is not None
    assert response.operation.id == 'bg-op-123'
    assert response.operation.done is False
    assert response.operation.action == '/background-model/bg-model'
    assert response.message is None


@pytest.mark.asyncio
async def test_generate_operation_with_background_model(ai: Genkit) -> None:
    """generate_operation resolves background models via resolve_model()."""
    await register_bg_model(ai, op_id='bg-op-456')

    operation = await ai.generate_operation(model='bg-model', prompt='a cat surfing')

    assert isinstance(operation, Operation)
    assert operation.id == 'bg-op-456'
    assert operation.action == '/background-model/bg-model'


@pytest.mark.asyncio
async def test_generate_operation_rejects_foreground_model_without_lro(ai: Genkit) -> None:
    """generate_operation rejects standard foreground models."""

    async def model_fn(request: ModelRequest, _: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='Hello'))],
            ),
        )

    ai.define_model(name='fg-model', fn=model_fn)

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate_operation(model='fg-model', prompt='Hi')

    assert 'does not support long running operations' in str(exc_info.value)
