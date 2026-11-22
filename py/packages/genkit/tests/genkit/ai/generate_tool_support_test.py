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

"""generate warns when tools or toolChoice are set on a model whose info says it doesn't support them."""

from collections.abc import Iterator, MutableMapping
from typing import Any, Literal

import pytest
import structlog
from pydantic import BaseModel
from structlog.testing import capture_logs

from genkit import Genkit, Message, ModelResponse, Part
from genkit._core._typing import Role
from genkit.model import ModelContext, ModelInfo, ModelRequest, ModelSupports

TOOLS_WARNING = 'model does not support tools, it may not behave the way you expect'
TOOL_CHOICE_WARNING = 'model does not support toolChoice, it may not behave the way you expect'


@pytest.fixture(autouse=True)
def _restore_structlog() -> Iterator[None]:
    """Restore structlog's global configuration around each test."""
    saved = structlog.get_config().copy()
    was_configured = structlog.is_configured()
    yield
    structlog.reset_defaults()
    if was_configured:
        structlog.configure(**saved)


class _Input(BaseModel):
    value: str


def _ai_with_model(info: ModelInfo | None) -> Genkit:
    """A Genkit with one tool and a model 'm' that answers with text and never calls a tool."""
    ai = Genkit()

    async def model_fn(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='m', fn=model_fn, info=info)

    @ai.tool(name='lookup')
    async def lookup(input: _Input) -> str:
        """Look a value up."""
        return input.value

    return ai


async def _generate(
    ai: Genkit,
    *,
    tools: list[str] | None = None,
    tool_choice: Literal['auto', 'required', 'none'] | None = None,
) -> list[MutableMapping[str, Any]]:
    with capture_logs() as entries:
        response = await ai.generate(model='m', prompt='hi', tools=tools, tool_choice=tool_choice)
    assert response.text == 'ok'
    return entries


def _warnings(entries: list[MutableMapping[str, Any]], event: str) -> list[MutableMapping[str, Any]]:
    return [e for e in entries if e['event'] == event]


@pytest.mark.asyncio
async def test_tools_on_a_model_that_does_not_support_them_warns() -> None:
    ai = _ai_with_model(ModelInfo(supports=ModelSupports(tools=False)))

    entries = await _generate(ai, tools=['lookup'])

    warned = _warnings(entries, TOOLS_WARNING)
    assert len(warned) == 1
    assert warned[0]['log_level'] == 'warning'
    assert warned[0]['model'] == 'm'
    assert warned[0]['tools'] == 1


@pytest.mark.asyncio
async def test_tools_on_a_model_with_supports_that_omits_tools_warns() -> None:
    """Declaring supports without tools means tools are not supported, as in JS."""
    ai = _ai_with_model(ModelInfo(supports=ModelSupports(multiturn=True)))

    entries = await _generate(ai, tools=['lookup'])

    assert len(_warnings(entries, TOOLS_WARNING)) == 1


@pytest.mark.asyncio
async def test_tools_on_a_model_that_supports_them_do_not_warn() -> None:
    ai = _ai_with_model(ModelInfo(supports=ModelSupports(tools=True)))

    entries = await _generate(ai, tools=['lookup'])

    assert _warnings(entries, TOOLS_WARNING) == []


@pytest.mark.asyncio
async def test_tool_choice_on_a_model_that_does_not_support_it_warns() -> None:
    ai = _ai_with_model(ModelInfo(supports=ModelSupports(tools=True, tool_choice=False)))

    entries = await _generate(ai, tools=['lookup'], tool_choice='required')

    warned = _warnings(entries, TOOL_CHOICE_WARNING)
    assert len(warned) == 1
    assert warned[0]['log_level'] == 'warning'
    assert warned[0]['model'] == 'm'
    assert warned[0]['toolChoice'] == 'required'
    assert _warnings(entries, TOOLS_WARNING) == []


@pytest.mark.asyncio
async def test_tool_choice_on_a_model_that_supports_it_does_not_warn() -> None:
    ai = _ai_with_model(ModelInfo(supports=ModelSupports(tools=True, tool_choice=True)))

    entries = await _generate(ai, tools=['lookup'], tool_choice='required')

    assert _warnings(entries, TOOL_CHOICE_WARNING) == []


@pytest.mark.asyncio
async def test_tools_and_tool_choice_on_a_model_that_supports_neither_warn_once_each() -> None:
    ai = _ai_with_model(ModelInfo(supports=ModelSupports(tools=False, tool_choice=False)))

    entries = await _generate(ai, tools=['lookup'], tool_choice='required')

    assert len(_warnings(entries, TOOLS_WARNING)) == 1
    assert len(_warnings(entries, TOOL_CHOICE_WARNING)) == 1


@pytest.mark.asyncio
async def test_model_without_supports_info_does_not_warn() -> None:
    """A model that declares no supports says nothing either way."""
    ai = _ai_with_model(None)

    entries = await _generate(ai, tools=['lookup'], tool_choice='required')

    assert _warnings(entries, TOOLS_WARNING) == []
    assert _warnings(entries, TOOL_CHOICE_WARNING) == []


@pytest.mark.asyncio
async def test_no_tools_and_no_tool_choice_do_not_warn() -> None:
    ai = _ai_with_model(ModelInfo(supports=ModelSupports(tools=False, tool_choice=False)))

    entries = await _generate(ai)

    assert _warnings(entries, TOOLS_WARNING) == []
    assert _warnings(entries, TOOL_CHOICE_WARNING) == []
