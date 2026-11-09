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

"""Plugin-author surface: unannotated models and DAPs returned from ``init()``."""

from typing import Any

import pytest

from genkit import Genkit, Message, ModelResponse, Part, Role
from genkit._core._action import GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR  # noqa: TID251
from genkit.model import Model, ModelContext, ModelRequest
from genkit.plugin_api import Action, ActionKind, DynamicActionProvider, Plugin


@pytest.mark.asyncio
async def test_unannotated_model_gets_model_request_from_raw_json() -> None:
    """A Dev UI / CLI run passes JSON; an unannotated handler still sees ModelRequest."""
    seen: list[object] = []

    async def fn(request, ctx: ModelContext) -> ModelResponse:  # noqa: ANN001  # pyright: ignore[reportMissingParameterType]
        seen.append(request)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(request.messages[0].text)]))

    model = Model('echo', fn)
    out = await model.run({'messages': [{'role': 'user', 'content': [{'text': 'hi'}]}]})

    assert isinstance(seen[0], ModelRequest)
    assert out.response.text == 'hi'


class _DapPlugin(Plugin):
    name = 'catalog'

    def __init__(self) -> None:
        async def tool_fn(city: str) -> str:
            return f'sunny in {city}'

        self._tool = Action(kind=ActionKind.TOOL, name='weather', fn=tool_fn)

        async def provide() -> dict[str, list[Action[Any, Any]]]:
            return {'tool': [self._tool]}

        async def dap_fn(input: dict[str, Any]) -> dict[str, Any]:
            return input

        self.dap = DynamicActionProvider(
            Action(kind=ActionKind.DYNAMIC_ACTION_PROVIDER, name='live', fn=dap_fn),
            provide,
        )

    async def init(self) -> list[Action]:
        return [self.dap.action]

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        return None

    async def list_actions(self) -> list[Action]:
        return []


def test_dynamic_action_provider_stamps_its_action() -> None:
    """A directly constructed provider is recoverable from its action."""
    plugin = _DapPlugin()

    assert getattr(plugin.dap.action, GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR) is plugin.dap


@pytest.mark.asyncio
async def test_dap_returned_from_plugin_init_is_listed_as_provider() -> None:
    """``init()`` registers it under the plugin namespace and the catalog finds the provider.

    Children stay out of the catalog: ``catalog/live:tool/weather`` has a slash in
    the provider segment, which neither the Python nor the JS key grammar accepts.
    """
    plugin = _DapPlugin()
    ai = Genkit(plugins=[plugin])

    catalog = await ai._registry.list_actions()

    assert '/dynamic-action-provider/catalog/live' in catalog
    action = await ai._registry.resolve_action(ActionKind.DYNAMIC_ACTION_PROVIDER, 'catalog/live')
    assert getattr(action, GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR) is plugin.dap
