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

"""Models and embedders whose request type is only a name still run."""

from __future__ import annotations

import types
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from genkit import Document, Genkit
from genkit._core._action import Action
from genkit._core._registry import ActionKind
from genkit._core._typing import ActionMetadata
from genkit.plugin_api import Plugin

Handler = Callable[..., Awaitable[Any]]


def _load(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    exec(source, module.__dict__)  # noqa: S102
    return module


_TYPE_CHECKING_MODEL = """
from __future__ import annotations

from typing import TYPE_CHECKING

from genkit import Message, ModelResponse, Part, Role
from genkit._core._typing import FinishReason

if TYPE_CHECKING:
    from genkit import ActionRunContext
    from genkit.model import ModelRequest


async def reply(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
    return ModelResponse(
        message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        finish_reason=FinishReason.STOP,
    )
"""


_TYPE_CHECKING_EMBEDDER = """
from __future__ import annotations

from typing import TYPE_CHECKING

from genkit._core._typing import Embedding, EmbedResponse

if TYPE_CHECKING:
    from genkit._core._typing import EmbedRequest


async def embed(request: EmbedRequest) -> EmbedResponse:
    return EmbedResponse(embeddings=[Embedding(embedding=[1.0, 2.0])])
"""


_TYPE_CHECKING_FLOW = """
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    class StepInput:
        pass


async def run(step: StepInput) -> str:
    return 'x'
"""


class _GardenPlugin(Plugin):
    name = 'garden'

    def __init__(self, fn: Handler) -> None:
        self._fn = fn

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        if action_type != ActionKind.MODEL or name != f'{self.name}/reply':
            return None
        return Action(kind=ActionKind.MODEL, name=name, fn=self._fn)

    async def list_actions(self) -> list[ActionMetadata]:
        return [ActionMetadata(action_type=ActionKind.MODEL, name=f'{self.name}/reply')]


class _EmbedPlugin(Plugin):
    name = 'garden-embed'

    def __init__(self, fn: Handler) -> None:
        self._fn = fn

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        if action_type != ActionKind.EMBEDDER or name != f'{self.name}/vectors':
            return None
        return Action(kind=ActionKind.EMBEDDER, name=name, fn=self._fn)

    async def list_actions(self) -> list[ActionMetadata]:
        return [ActionMetadata(action_type=ActionKind.EMBEDDER, name=f'{self.name}/vectors')]


@pytest.mark.asyncio
async def test_generate_plugin_model_with_type_checking_request_annotation_returns_reply() -> None:
    fn = _load('garden_model', _TYPE_CHECKING_MODEL).reply
    ai = Genkit(plugins=[_GardenPlugin(fn)])
    resp = await ai.generate(model='garden/reply', prompt='hi')
    assert resp.text == 'ok'


def test_define_model_with_type_checking_request_annotation_registers() -> None:
    fn = _load('defined_model', _TYPE_CHECKING_MODEL).reply
    ai = Genkit()
    action = ai.define_model(name='local-reply', fn=fn)
    assert action.name == 'local-reply'


@pytest.mark.asyncio
async def test_embed_plugin_embedder_with_type_checking_request_annotation_returns_embeddings() -> None:
    fn = _load('garden_embedder', _TYPE_CHECKING_EMBEDDER).embed
    ai = Genkit(plugins=[_EmbedPlugin(fn)])
    embeddings = await ai.embed(embedder='garden-embed/vectors', content=Document.from_text('hello'))
    assert embeddings[0].embedding == [1.0, 2.0]


def test_define_flow_with_type_checking_request_annotation_still_raises() -> None:
    fn = _load('typed_flow', _TYPE_CHECKING_FLOW).run
    ai = Genkit()
    with pytest.raises(TypeError, match='StepInput'):
        ai.flow()(fn)
