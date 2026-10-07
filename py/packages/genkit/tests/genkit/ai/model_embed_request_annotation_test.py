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

import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic import TypeAdapter

from genkit import Document, Genkit, GenkitError
from genkit._core._action import Action
from genkit._core._registry import ActionKind
from genkit._core._typing import ActionMetadata
from genkit.embedder import EmbedRequest
from genkit.model import ModelRequest
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
    from genkit.embedder import EmbedRequest

received: list[object] = []


async def embed(request: EmbedRequest) -> EmbedResponse:
    received.extend(request.input)
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


# Every request type the fallback knows, imported only for type checkers.
_TYPE_CHECKING_HANDLERS = """
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genkit.embedder import EmbedRequest
    from genkit.model import ModelRequest

received: list[object] = []


async def model_handler(request: ModelRequest) -> object:
    received.append(request)
    return {}


async def embed_handler(request: EmbedRequest) -> object:
    return {}


async def dict_handler(request: dict[str, object]) -> object:
    return {}
"""


class _GardenPlugin(Plugin):
    name = 'garden'

    def __init__(self, fn: Handler) -> None:
        self._fn = fn

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        if action_type != ActionKind.MODEL or name != 'reply':
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
        if action_type != ActionKind.EMBEDDER or name != 'vectors':
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
    module = _load('garden_embedder', _TYPE_CHECKING_EMBEDDER)
    ai = Genkit(plugins=[_EmbedPlugin(module.embed)])
    embeddings = await ai.embed(embedder='garden-embed/vectors', content=Document.from_text('hello'))
    assert embeddings[0].embedding == [1.0, 2.0]
    # genkit.embedder.EmbedRequest declares list[Document], so that's what the handler gets.
    assert [type(d) for d in module.received] == [Document]


@pytest.mark.parametrize('define', ['flow', 'tool'])
def test_define_flow_or_tool_with_type_checking_input_annotation_still_raises(define: str) -> None:
    fn = _load(f'typed_{define}', _TYPE_CHECKING_FLOW).run
    ai = Genkit()
    with pytest.raises(TypeError, match='StepInput'):
        if define == 'flow':
            ai.flow()(fn)
        else:
            ai.tool()(fn)


@pytest.mark.parametrize(
    ('kind', 'handler', 'expected'),
    [
        (ActionKind.MODEL, 'model_handler', ModelRequest),
        (ActionKind.BACKGROUND_MODEL, 'model_handler', ModelRequest),
        (ActionKind.EMBEDDER, 'embed_handler', EmbedRequest),
    ],
)
def test_type_checking_request_annotation_publishes_request_type_schema(
    kind: ActionKind, handler: str, expected: type
) -> None:
    """The Dev UI sees the real request schema, not an empty or string one."""
    fn = getattr(_load(f'schema_{kind}', _TYPE_CHECKING_HANDLERS), handler)
    action = Action(kind=kind, name='garden/x', fn=fn)
    assert action.input_class is expected
    assert action.input_schema == TypeAdapter(expected).json_schema()


@pytest.mark.asyncio
async def test_type_checking_request_annotation_validates_raw_json_as_model_request() -> None:
    """JSON from the Dev UI or reflection API is parsed into a ModelRequest, and bad JSON is rejected."""
    module = _load('raw_json_model', _TYPE_CHECKING_HANDLERS)
    action = Action(kind=ActionKind.MODEL, name='garden/x', fn=module.model_handler)

    await action.run({'messages': [{'role': 'user', 'content': [{'text': 'hi'}]}]})
    assert [type(r) for r in module.received] == [ModelRequest]

    with pytest.raises(GenkitError, match='INVALID_ARGUMENT'):
        await action.run({'messages': 'hi'})


def test_model_with_runtime_resolvable_request_annotation_keeps_its_own_type() -> None:
    """Only a name that can't be found falls back; a type that resolves is used as written."""
    fn = _load('dict_model', _TYPE_CHECKING_HANDLERS).dict_handler
    action = Action(kind=ActionKind.MODEL, name='garden/x', fn=fn)
    assert action.input_schema == TypeAdapter(dict[str, object]).json_schema()


@pytest.mark.skipif(sys.version_info < (3, 14), reason='annotations are evaluated lazily from Python 3.14')
@pytest.mark.asyncio
async def test_generate_type_checking_request_annotation_runs_without_future_import() -> None:
    """On 3.14 the name arrives as a ForwardRef, not a string, and still falls back to ModelRequest."""
    source = _TYPE_CHECKING_MODEL.replace('from __future__ import annotations\n', '')
    fn = _load('lazy_model', source).reply
    ai = Genkit(plugins=[_GardenPlugin(fn)])
    resp = await ai.generate(model='garden/reply', prompt='hi')
    assert resp.text == 'ok'
