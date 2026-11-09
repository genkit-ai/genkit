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

"""toolConfig on the wire: built-in tools next to function tools, and tool_choice='required'."""

import json
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from genkit_google_genai import GeminiConfig, GoogleAI
from genkit_google_genai._models._gemini import GeminiModel
from genkit_google_genai._models._utils import PartConverter
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel

from genkit import Genkit, Message, Part, Role
from genkit.model import ModelRequest, OutputConfig, ToolDefinition, ToolRequest, ToolResponse


def _fake_gemini_api(bodies: list[dict[str, Any]]) -> Callable[..., Awaitable[httpx.Response]]:
    """Stand-in for httpx.AsyncClient.send that answers generateContent and records each POST body."""

    async def send(_client: httpx.AsyncClient, request: httpx.Request, **_: object) -> httpx.Response:
        if request.method == 'GET':
            return httpx.Response(200, json={'models': []}, request=request)
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={'candidates': [{'content': {'role': 'model', 'parts': [{'text': '{"dish": "pho"}'}]}, 'finishReason': 'STOP'}]},
            request=request,
        )

    return send


class Dish(BaseModel):
    dish: str


def _app(monkeypatch: pytest.MonkeyPatch, bodies: list[dict[str, Any]]) -> Genkit:
    monkeypatch.setattr(httpx.AsyncClient, 'send', _fake_gemini_api(bodies))
    ai = Genkit(plugins=[GoogleAI(api_key='fake-key')])

    @ai.tool(name='lookup_menu')
    async def lookup_menu(dish: str) -> str:
        """Check whether a dish is on today's menu."""
        return 'in stock'

    return ai


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'builtin',
    [
        GeminiConfig(google_search=True),
        GeminiConfig(url_context=True),
        GeminiConfig(code_execution=True),
        GeminiConfig(file_search_store_names=['fileSearchStores/menu']),
    ],
    ids=['google_search', 'url_context', 'code_execution', 'file_search'],
)
async def test_builtin_tool_with_function_tool_opts_in_to_server_side_invocations(
    monkeypatch: pytest.MonkeyPatch, builtin: GeminiConfig
) -> None:
    """Gemini 3 rejects built-in + function tools unless toolConfig.includeServerSideToolInvocations is set."""
    bodies: list[dict[str, Any]] = []
    ai = _app(monkeypatch, bodies)

    await ai.generate(model='googleai/gemini-flash-latest', prompt='Is the pho in stock?', tools=['lookup_menu'], config=builtin)

    assert bodies[-1]['toolConfig'] == {'includeServerSideToolInvocations': True}


@pytest.mark.asyncio
async def test_server_side_flag_merges_with_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag rides next to the function calling mode."""
    bodies: list[dict[str, Any]] = []
    ai = _app(monkeypatch, bodies)

    await ai.generate(
        model='googleai/gemini-flash-latest',
        prompt='Is the pho in stock?',
        tools=['lookup_menu'],
        tool_choice='auto',
        config=GeminiConfig(google_search=True),
    )

    assert bodies[-1]['toolConfig'] == {
        'functionCallingConfig': {'mode': 'AUTO'},
        'includeServerSideToolInvocations': True,
    }


@pytest.mark.asyncio
async def test_function_tools_alone_send_no_tool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only function tools: nothing to opt in to."""
    bodies: list[dict[str, Any]] = []
    ai = _app(monkeypatch, bodies)

    await ai.generate(model='googleai/gemini-flash-latest', prompt='Is the pho in stock?', tools=['lookup_menu'])

    assert 'toolConfig' not in bodies[-1]


@pytest.mark.asyncio
async def test_builtin_tool_alone_sends_no_tool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a built-in tool: nothing to opt in to."""
    bodies: list[dict[str, Any]] = []
    ai = _app(monkeypatch, bodies)

    await ai.generate(model='googleai/gemini-flash-latest', prompt='Pho history?', config=GeminiConfig(google_search=True))

    assert 'toolConfig' not in bodies[-1]


def _tool_request(**kwargs: Any) -> ModelRequest:  # noqa: ANN401
    return ModelRequest(
        messages=kwargs.pop('messages', [Message(role=Role.USER, content=[Part.from_text('Is the pho in stock?')])]),
        tools=[ToolDefinition(name='lookup_menu', description='d', input_schema={'type': 'object'})],
        **kwargs,
    )


@pytest.mark.asyncio
async def test_vertex_never_sends_server_side_flag() -> None:
    """Vertex AI has no includeServerSideToolInvocations; google-genai raises if it is set."""
    client = MagicMock(spec=genai.Client)
    client.vertexai = True
    model = GeminiModel('gemini-3.5-flash', client)

    cfg = await model._genkit_to_googleai_cfg(_tool_request(config=GeminiConfig(google_search=True)))

    assert cfg is not None
    assert cfg.tool_config is None


@pytest.mark.asyncio
async def test_function_tools_disable_sdk_afc() -> None:
    """Genkit runs the tools; google-genai AFC is off so it doesn't warn about FunctionDeclarations."""
    client = MagicMock(spec=genai.Client)
    client.vertexai = False
    model = GeminiModel('gemini-flash-latest', client)

    cfg = await model._genkit_to_googleai_cfg(_tool_request(config=GeminiConfig(google_search=True)))

    assert cfg is not None
    assert cfg.automatic_function_calling == genai_types.AutomaticFunctionCallingConfig(disable=True)


@pytest.mark.asyncio
async def test_server_side_tool_parts_round_trip() -> None:
    """toolCall / toolResponse parts survive Genkit history and go back unchanged, signature included."""
    tool_call = genai_types.Part(
        tool_call=genai_types.ToolCall(id='c1', tool_type=genai_types.ToolType.GOOGLE_SEARCH_WEB, args={'q': 'pho'}),
        thought_signature=b'sig-1',
    )
    tool_response = genai_types.Part(
        tool_response=genai_types.ToolResponse(
            id='c1', tool_type=genai_types.ToolType.GOOGLE_SEARCH_WEB, response={'results': ['pho is soup']}
        )
    )

    genkit_call = PartConverter.from_gemini(tool_call)
    genkit_response = PartConverter.from_gemini(tool_response)

    assert genkit_call.custom == {'geminiToolCall': {'id': 'c1', 'toolType': 'GOOGLE_SEARCH_WEB', 'args': {'q': 'pho'}}}
    assert genkit_call.metadata is not None and 'thoughtSignature' in genkit_call.metadata
    rehydrated = Part.model_validate_json(genkit_call.model_dump_json(by_alias=True))
    assert await PartConverter.to_gemini(rehydrated) == tool_call
    assert await PartConverter.to_gemini(genkit_response) == tool_response


# -- tool_choice='required' (PR3-09) --


@pytest.mark.asyncio
async def test_required_with_output_schema_omits_json_mode_on_first_turn() -> None:
    """ANY mode plus responseMimeType=application/json is a Gemini 400; the schema waits for the answer turn."""
    client = MagicMock(spec=genai.Client)
    client.vertexai = False
    model = GeminiModel('gemini-flash-latest', client)
    request = _tool_request(
        tool_choice='required',
        output=OutputConfig(format='json', json_schema=Dish.model_json_schema(), constrained=True),
    )

    cfg = await model._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.tool_config is not None and cfg.tool_config.function_calling_config is not None
    assert cfg.tool_config.function_calling_config.mode == genai_types.FunctionCallingConfigMode.ANY
    assert cfg.response_mime_type is None
    assert cfg.response_json_schema is None


@pytest.mark.asyncio
async def test_required_relaxes_to_auto_after_tool_responses() -> None:
    """After the tools answer, ANY would only allow another call and loop to max_turns; the turn goes AUTO."""
    client = MagicMock(spec=genai.Client)
    client.vertexai = False
    model = GeminiModel('gemini-flash-latest', client)
    messages = [
        Message(role=Role.USER, content=[Part.from_text('Is the pho in stock?')]),
        Message(role=Role.MODEL, content=[Part(tool_request=ToolRequest(name='lookup_menu', ref='1', input={}))]),
        Message(role=Role.TOOL, content=[Part(tool_response=ToolResponse(name='lookup_menu', ref='1', output='in stock'))]),
    ]
    request = _tool_request(
        messages=messages,
        tool_choice='required',
        output=OutputConfig(format='json', json_schema=Dish.model_json_schema(), constrained=True),
    )

    cfg = await model._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.tool_config is not None and cfg.tool_config.function_calling_config is not None
    assert cfg.tool_config.function_calling_config.mode == genai_types.FunctionCallingConfigMode.AUTO
    assert cfg.tool_config.function_calling_config.allowed_function_names is None
    assert cfg.response_mime_type == 'application/json'
    assert cfg.response_json_schema == Dish.model_json_schema()


@pytest.mark.asyncio
async def test_required_end_to_end_returns_structured_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """ai.generate with required + output_schema: turn 1 sends ANY without JSON mode."""
    bodies: list[dict[str, Any]] = []
    ai = _app(monkeypatch, bodies)

    await ai.generate(
        model='googleai/gemini-flash-latest',
        prompt='Is the pho in stock?',
        tools=['lookup_menu'],
        tool_choice='required',
        output_schema=Dish,
    )

    first = bodies[0]
    assert first['toolConfig']['functionCallingConfig']['mode'] == 'ANY'
    assert 'responseMimeType' not in first.get('generationConfig', {})
