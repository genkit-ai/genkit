# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The sample's own wiring, driven by a fake model instead of Gemini.

The bookshop server really is launched as a child process, so these tests cover
the sample end to end apart from the model itself.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from genkit import Message, ModelRequest, ModelResponse
from genkit._ai._testing import define_programmable_model
from genkit._core._typing import (
    FinishReason,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)

SAMPLE_MAIN = Path(__file__).resolve().parents[1] / 'src' / 'main.py'


def load_sample(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the sample's entry module, which registers the MCP client."""
    # The sample builds its model plugin at import time and that plugin refuses
    # to construct without a key. Nothing here ever reaches Gemini.
    monkeypatch.setenv('GEMINI_API_KEY', 'unused-by-these-tests')
    spec = importlib.util.spec_from_file_location('mcp_hello_main', SAMPLE_MAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text_response(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=text))]),
        finish_reason=FinishReason.STOP,
    )


def tool_call_response(tool_name: str, input: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        message=Message(
            role=Role.MODEL,
            content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name=tool_name, input=input, ref='1')))],
        ),
        finish_reason=FinishReason.STOP,
    )


def tool_output(request: ModelRequest) -> Any:
    """Read what the tool returned, as the model saw it on the follow-up turn."""
    tool_message = request.messages[-1]
    assert tool_message.role == Role.TOOL
    assert tool_message.content[0].root.tool_response is not None
    return tool_message.content[0].root.tool_response.output


@pytest.mark.asyncio
async def test_the_client_reports_its_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool names carry the name the client was registered under, not the server's own."""
    sample = load_sample(monkeypatch)

    try:
        tools = await sample.client.get_active_tools()
    finally:
        await sample.client.close()

    assert sorted(tool.name for tool in tools) == [
        'bookshop_check_stock',
        'bookshop_opening_hours',
        'bookshop_search_books',
    ]
    descriptions = {tool.name: tool.description for tool in tools}
    assert descriptions['bookshop_opening_hours'] == "Report the shop's opening hours for a day of the week."


def test_tool_name_namespaces_a_server_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Naming a tool starts nothing: the prefix never came from the server."""
    sample = load_sample(monkeypatch)

    assert sample.client.tool_name('opening_hours') == 'bookshop_opening_hours'


@pytest.mark.asyncio
async def test_a_single_tool_selector_runs_that_tool_and_returns_its_structured_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = load_sample(monkeypatch)
    pm, _ = define_programmable_model(sample.ai)
    pm.responses = [
        tool_call_response('bookshop_check_stock', {'title': 'Piranesi'}),
        text_response('two copies are on the shelf'),
    ]

    try:
        selector = f'{sample.client.name}:tool/{sample.client.tool_name("check_stock")}'
        response = await sample.ai.generate(
            model='programmableModel',
            prompt='is Piranesi in stock?',
            tools=[selector],
        )
    finally:
        await sample.client.close()

    assert response.text == 'two copies are on the shelf'
    assert pm.request_count == 2
    assert pm.last_request is not None
    assert [tool.name for tool in pm.last_request.tools or []] == ['bookshop_check_stock']
    assert tool_output(pm.last_request) == {'title': 'Piranesi', 'stocked': True, 'copies': 2}


@pytest.mark.asyncio
async def test_the_wildcard_selector_passes_an_argument_to_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = load_sample(monkeypatch)
    pm, _ = define_programmable_model(sample.ai)
    pm.responses = [
        tool_call_response('bookshop_search_books', {'topic': 'history', 'limit': 1}),
        text_response('one history book'),
    ]

    try:
        await sample.ai.generate(
            model='programmableModel',
            prompt='any history books?',
            tools=[f'{sample.client.name}:tool/*'],
        )
    finally:
        await sample.client.close()

    assert pm.last_request is not None
    assert tool_output(pm.last_request) == {
        'result': [
            {'title': 'The Making of the Atomic Bomb', 'author': 'Richard Rhodes', 'topic': 'history'},
        ]
    }
