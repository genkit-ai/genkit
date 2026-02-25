#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import json
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field

from genkit.ai import Genkit, Output, ToolRunContext, tool_response
from genkit.blocks.document import Document
from genkit.blocks.formats.types import FormatDef, Formatter, FormatterConfig
from genkit.blocks.model import MessageWrapper, ModelMiddlewareNext, text_from_message
from genkit.core.action import ActionRunContext
from genkit.core.action.types import ActionKind
from genkit.core.typing import (
    BaseDataPoint,
    Details,
    DocumentData,
    DocumentPart,
    EvalFnResponse,
    EvalRequest,
    EvalResponse,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    Message,
    Metadata,
    ModelInfo,
    OutputConfig,
    Part,
    RetrieverResponse,
    Role,
    Score,
    Supports,
    TextPart,
    ToolChoice,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)
from genkit.testing import (
    EchoModel,
    ProgrammableModel,
    define_echo_model,
    define_programmable_model,
)

# type SetupFixture = tuple[Genkit, EchoModel, ProgrammableModel]
SetupFixture = tuple[Genkit, EchoModel, ProgrammableModel]


@pytest.fixture
def setup_test() -> SetupFixture:
    """Setup a test fixture for the veneer tests."""
    ai = Genkit(model='echoModel')

    pm, _ = define_programmable_model(ai)
    echo, _ = define_echo_model(ai)

    return (ai, echo, pm)


@pytest.mark.asyncio
async def test_generate_uses_default_model(setup_test: SetupFixture) -> None:
    """Test that the generate function uses the default model."""
    ai, *_ = setup_test

    want_txt = '[ECHO] user: "hi" {"temperature":11.0}'

    response = await ai.generate(prompt='hi', config={'temperature': 11})

    assert response.text == want_txt

    _, response = ai.generate_stream(prompt='hi', config={'temperature': 11})

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_populates_latency_ms(setup_test: SetupFixture) -> None:
    """Test that the generate function populates latency_ms in the response."""
    ai, *_ = setup_test

    response = await ai.generate(prompt='hi')

    # Verify latency_ms is set and is a positive number
    assert response.latency_ms is not None
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_generate_latency_ms_in_serialized_json(setup_test: SetupFixture) -> None:
    """Test that latencyMs appears in the serialized JSON output.

    This is critical for DevUI trace viewer which expects the camelCase alias
    'latencyMs' to be present in the span output JSON.
    """
    ai, *_ = setup_test

    response = await ai.generate(prompt='hi')

    # Serialize using the same method used in span output recording
    serialized = response.model_dump_json(by_alias=True, exclude_none=True)
    parsed = json.loads(serialized)

    # Verify latencyMs (camelCase) is in the serialized output
    assert 'latencyMs' in parsed, f'latencyMs not found in serialized JSON. Keys: {list(parsed.keys())}'
    assert parsed['latencyMs'] > 0


@pytest.mark.asyncio
async def test_generate_with_explicit_model(setup_test: SetupFixture) -> None:
    """Test that the generate function uses the explicit model."""
    ai, *_ = setup_test

    response = await ai.generate(model='echoModel', prompt='hi', config={'temperature': 11})

    assert response.text == '[ECHO] user: "hi" {"temperature":11.0}'

    _, response = ai.generate_stream(model='echoModel', prompt='hi', config={'temperature': 11})

    assert (await response).text == '[ECHO] user: "hi" {"temperature":11.0}'


@pytest.mark.asyncio
async def test_generate_with_str_prompt(setup_test: SetupFixture) -> None:
    """Test that the generate function with a string prompt works."""
    ai, *_ = setup_test

    response = await ai.generate(prompt='hi', config={'temperature': 11})

    assert response.text == '[ECHO] user: "hi" {"temperature":11.0}'


@pytest.mark.asyncio
async def test_generate_with_part_prompt(setup_test: SetupFixture) -> None:
    """Test that the generate function with a part prompt works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] user: "hi" {"temperature":11.0}'

    response = await ai.generate(prompt=Part(root=TextPart(text='hi')), config={'temperature': 11})

    assert response.text == want_txt

    _, response = ai.generate_stream(prompt=Part(root=TextPart(text='hi')), config={'temperature': 11})

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_part_list_prompt(setup_test: SetupFixture) -> None:
    """Test that the generate function with a list of parts prompt works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] user: "hello","world" {"temperature":11.0}'

    response = await ai.generate(
        prompt=[Part(root=TextPart(text='hello')), Part(root=TextPart(text='world'))],
        config={'temperature': 11},
    )

    assert response.text == want_txt

    _, response = ai.generate_stream(
        prompt=[Part(root=TextPart(text='hello')), Part(root=TextPart(text='world'))],
        config={'temperature': 11},
    )

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_str_system(setup_test: SetupFixture) -> None:
    """Test that the generate function with a string system works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] system: "talk like pirate" user: "hi" {"temperature":11.0}'

    response = await ai.generate(system='talk like pirate', prompt='hi', config={'temperature': 11})

    assert response.text == want_txt

    _, response = ai.generate_stream(system='talk like pirate', prompt='hi', config={'temperature': 11})

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_part_system(setup_test: SetupFixture) -> None:
    """Test that the generate function with a part system works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] system: "talk like pirate" user: "hi" {"temperature":11.0}'

    response = await ai.generate(
        system=Part(root=TextPart(text='talk like pirate')),
        prompt='hi',
        config={'temperature': 11},
    )

    assert response.text == want_txt

    _, response = ai.generate_stream(
        system=Part(root=TextPart(text='talk like pirate')),
        prompt='hi',
        config={'temperature': 11},
    )

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_part_list_system(setup_test: SetupFixture) -> None:
    """Test that the generate function with a list of parts system works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] system: "talk","like pirate" user: "hi" {"temperature":11.0}'

    response = await ai.generate(
        system=[Part(root=TextPart(text='talk')), Part(root=TextPart(text='like pirate'))],
        prompt='hi',
        config={'temperature': 11},
    )

    assert response.text == want_txt

    _, response = ai.generate_stream(
        system=[Part(root=TextPart(text='talk')), Part(root=TextPart(text='like pirate'))],
        prompt='hi',
        config={'temperature': 11},
    )

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_messages(setup_test: SetupFixture) -> None:
    """Test that the generate function with a list of messages works."""
    ai, *_ = setup_test

    response = await ai.generate(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='hi'))],
            ),
        ],
        config={'temperature': 11},
    )

    assert response.text == '[ECHO] user: "hi" {"temperature":11.0}'

    _, response = ai.generate_stream(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='hi'))],
            ),
        ],
        config={'temperature': 11},
    )

    assert (await response).text == '[ECHO] user: "hi" {"temperature":11.0}'


@pytest.mark.asyncio
async def test_generate_with_system_prompt_messages(
    setup_test: SetupFixture,
) -> None:
    """Generate function with a system prompt and messages works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] system: "talk like pirate" user: "hi" model: "bye" user: "hi again"'

    response = await ai.generate(
        system='talk like pirate',
        prompt='hi again',
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='hi'))],
            ),
            Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='bye'))],
            ),
        ],
    )

    assert response.text == want_txt

    _, response = ai.generate_stream(
        system='talk like pirate',
        prompt='hi again',
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='hi'))],
            ),
            Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='bye'))],
            ),
        ],
    )

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_tools(setup_test: SetupFixture) -> None:
    """Test that the generate function with tools works."""
    ai, echo, *_ = setup_test

    class ToolInput(BaseModel):
        value: int | None = Field(None, description='value field')

    @ai.tool(name='testTool')
    def test_tool(input: ToolInput) -> int:
        """The tool."""
        return input.value or 0

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        tool_choice=ToolChoice.REQUIRED,
        tools=['testTool'],
    )

    want_txt = f'[ECHO] user: "hi" tools=testTool tool_choice={ToolChoice.REQUIRED}'

    want_request = [
        ToolDefinition(
            name='testTool',
            description='The tool.',
            input_schema={
                'properties': {
                    'value': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                        'default': None,
                        'description': 'value field',
                        'title': 'Value',
                    }
                },
                'title': 'ToolInput',
                'type': 'object',
            },
            output_schema={'type': 'integer'},
        )
    ]

    assert response.text == want_txt
    assert echo.last_request is not None
    assert echo.last_request.tools == want_request

    _, response = ai.generate_stream(
        model='echoModel',
        prompt='hi',
        tool_choice=ToolChoice.REQUIRED,
        tools=['testTool'],
    )

    assert (await response).text == want_txt
    assert echo.last_request is not None
    assert echo.last_request.tools == want_request


@pytest.mark.asyncio
async def test_generate_with_interrupting_tools(
    setup_test: SetupFixture,
) -> None:
    """Test that the generate function with tools works."""
    ai, _, pm, *_ = setup_test

    class ToolInput(BaseModel):
        value: int | None = Field(None, description='value field')

    @ai.tool(name='test_tool')
    def test_tool(input: ToolInput) -> int:
        """The tool."""
        return (input.value or 0) + 7

    @ai.tool(name='test_interrupt')
    def test_interrupt(input: ToolInput, ctx: ToolRunContext) -> None:
        """The interrupt."""
        ctx.interrupt({'banana': 'yes please'})

    tool_request_msg = MessageWrapper(
        Message(
            role=Role.MODEL,
            content=[
                Part(root=TextPart(text='call these tools')),
                Part(
                    root=ToolRequestPart(tool_request=ToolRequest(input={'value': 5}, name='test_interrupt', ref='123'))
                ),
                Part(root=ToolRequestPart(tool_request=ToolRequest(input={'value': 5}, name='test_tool', ref='234'))),
            ],
        )
    )
    pm.responses.append(
        GenerateResponse(
            finish_reason=FinishReason.STOP,
            message=tool_request_msg,
        )
    )
    pm.responses.append(
        GenerateResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='tool called'))]),
        )
    )

    response = await ai.generate(
        model='programmableModel',
        prompt='hi',
        tools=['test_tool', 'test_interrupt'],
    )

    want_request = [
        ToolDefinition(
            name='test_tool',
            description='The tool.',
            input_schema={
                'properties': {
                    'value': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                        'default': None,
                        'description': 'value field',
                        'title': 'Value',
                    }
                },
                'title': 'ToolInput',
                'type': 'object',
            },
            output_schema={'type': 'integer'},
        ),
        ToolDefinition(
            name='test_interrupt',
            description='The interrupt.',
            input_schema={
                'properties': {
                    'value': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                        'default': None,
                        'description': 'value field',
                        'title': 'Value',
                    }
                },
                'title': 'ToolInput',
                'type': 'object',
            },
            output_schema={'type': 'null'},
        ),
    ]

    assert response.text == 'call these tools'
    assert response.message == MessageWrapper(
        Message(
            role=Role.MODEL,
            content=[
                Part(root=TextPart(text='call these tools')),
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(ref='123', name='test_interrupt', input={'value': 5}),
                        metadata=Metadata(root={'interrupt': {'banana': 'yes please'}}),
                    )
                ),
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(ref='234', name='test_tool', input={'value': 5}),
                        metadata=Metadata(root={'pendingOutput': 12}),
                    )
                ),
            ],
        )
    )
    assert pm.last_request is not None
    assert pm.last_request.tools == want_request


@pytest.mark.asyncio
async def test_generate_with_interrupt_respond(
    setup_test: SetupFixture,
) -> None:
    """Test that the generate function with tools works."""
    ai, _, pm, *_ = setup_test

    class ToolInput(BaseModel):
        value: int | None = Field(None, description='value field')

    @ai.tool(name='test_tool')
    def test_tool(input: ToolInput) -> int:
        """The tool."""
        return (input.value or 0) + 7

    @ai.tool(name='test_interrupt')
    def test_interrupt(input: ToolInput, ctx: ToolRunContext) -> None:
        """The interrupt."""
        ctx.interrupt({'banana': 'yes please'})

    tool_request_msg = MessageWrapper(
        Message(
            role=Role.MODEL,
            content=[
                Part(root=TextPart(text='call these tools')),
                Part(
                    root=ToolRequestPart(tool_request=ToolRequest(input={'value': 5}, name='test_interrupt', ref='123'))
                ),
                Part(root=ToolRequestPart(tool_request=ToolRequest(input={'value': 5}, name='test_tool', ref='234'))),
            ],
        )
    )
    pm.responses.append(
        GenerateResponse(
            finish_reason=FinishReason.STOP,
            message=tool_request_msg,
        )
    )
    pm.responses.append(
        GenerateResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='tool called'))]),
        )
    )

    interrupted_response = await ai.generate(
        model='programmableModel',
        prompt='hi',
        tools=['test_tool', 'test_interrupt'],
    )

    assert interrupted_response.finish_reason == 'interrupted'
    assert interrupted_response.tool_requests == [
        Part(
            root=ToolRequestPart(
                tool_request=ToolRequest(ref='123', name='test_interrupt', input={'value': 5}),
                metadata=Metadata(root={'interrupt': {'banana': 'yes please'}}),
            )
        ).root,
        Part(
            root=ToolRequestPart(
                tool_request=ToolRequest(ref='234', name='test_tool', input={'value': 5}),
                metadata=Metadata(root={'pendingOutput': 12}),
            )
        ).root,
    ]

    assert interrupted_response.messages == [
        Message(
            role='user',
            content=[Part(root=TextPart(text='hi'))],
        ),
        Message(
            role='model',
            content=[
                Part(root=TextPart(text='call these tools')),
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(ref='123', name='test_interrupt', input={'value': 5}),
                        metadata=Metadata(root={'interrupt': {'banana': 'yes please'}}),
                    )
                ),
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(ref='234', name='test_tool', input={'value': 5}),
                        metadata=Metadata(root={'pendingOutput': 12}),
                    )
                ),
            ],
        ),
    ]

    response = await ai.generate(
        model='programmableModel',
        messages=interrupted_response.messages,
        tool_responses=[tool_response(interrupted_response.interrupts[0], {'bar': 2})],
        tools=['test_tool', 'test_interrupt'],
    )

    assert response.text == 'tool called'

    assert response.messages == [
        Message(
            role='user',
            content=[Part(root=TextPart(text='hi'))],
        ),
        Message(
            role='model',
            content=[
                Part(root=TextPart(text='call these tools')),
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(ref='123', name='test_interrupt', input={'value': 5}),
                        metadata=Metadata(root={'resolvedInterrupt': {'banana': 'yes please'}}),
                    )
                ),
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(ref='234', name='test_tool', input={'value': 5}),
                        metadata=Metadata(root={'pendingOutput': 12}),
                    )
                ),
            ],
            metadata=None,
        ),
        Message(
            role='tool',
            content=[
                Part(
                    root=ToolResponsePart(
                        tool_response=ToolResponse(ref='123', name='test_interrupt', output={'bar': 2}),
                        metadata=Metadata(root={'interruptResponse': True}),
                    )
                ),
                Part(
                    root=ToolResponsePart(
                        tool_response=ToolResponse(ref='234', name='test_tool', output=12),
                        metadata=Metadata(root={'source': 'pending'}),
                    )
                ),
            ],
            metadata={'resumed': True},
        ),
        Message(
            role='model',
            content=[Part(root=TextPart(text='tool called'))],
            metadata=None,
        ),
    ]


@pytest.mark.asyncio
async def test_generate_with_tools_and_output(setup_test: SetupFixture) -> None:
    """Test that the generate function with tools and output works."""
    ai, _, pm, *_ = setup_test

    class ToolInput(BaseModel):
        value: int | None = Field(None, description='value field')

    @ai.tool(name='testTool')
    def test_tool(input: ToolInput) -> str:
        """The tool."""
        return 'abc'

    tool_request_msg = MessageWrapper(
        Message(
            role=Role.MODEL,
            content=[
                Part(root=ToolRequestPart(tool_request=ToolRequest(input={'value': 5}, name='testTool', ref='123')))
            ],
        )
    )
    pm.responses.append(
        GenerateResponse(
            finish_reason=FinishReason.STOP,
            message=tool_request_msg,
        )
    )
    pm.responses.append(
        GenerateResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='tool called'))]),
        )
    )

    response = await ai.generate(
        model='programmableModel',
        prompt='hi',
        tool_choice=ToolChoice.REQUIRED,
        tools=['testTool'],
    )

    assert response.text == 'tool called'
    assert response.request is not None
    assert response.request.messages is not None
    assert response.request.messages[0] == Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])
    assert response.request.messages[1] == tool_request_msg
    assert response.request.messages[2] == Message(
        role=Role.TOOL,
        content=[Part(root=ToolResponsePart(tool_response=ToolResponse(ref='123', name='testTool', output='abc')))],
    )
    assert pm.last_request is not None
    assert pm.last_request.tools == [
        ToolDefinition(
            name='testTool',
            description='The tool.',
            input_schema={
                'properties': {
                    'value': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                        'default': None,
                        'description': 'value field',
                        'title': 'Value',
                    }
                },
                'title': 'ToolInput',
                'type': 'object',
            },
            output_schema={'type': 'string'},
        )
    ]


@pytest.mark.asyncio
async def test_generate_stream_with_tools(setup_test: SetupFixture) -> None:
    """Test that the generate stream function with tools works."""
    ai, _, pm, *_ = setup_test

    class ToolInput(BaseModel):
        value: int | None = Field(None, description='value field')

    @ai.tool(name='testTool')
    def test_tool(input: ToolInput) -> str:
        """The tool."""
        return 'abc'

    tool_request_msg = MessageWrapper(
        Message(
            role=Role.MODEL,
            content=[
                Part(root=ToolRequestPart(tool_request=ToolRequest(input={'value': 5}, name='testTool', ref='123')))
            ],
        )
    )
    pm.responses.append(
        GenerateResponse(
            finish_reason=FinishReason.STOP,
            message=tool_request_msg,
        )
    )
    pm.responses.append(
        GenerateResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='tool called'))]),
        )
    )
    pm.chunks = [
        [
            GenerateResponseChunk(
                role=Role(tool_request_msg.role),
                content=tool_request_msg.content,
            )
        ],
        [GenerateResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='tool called'))])],
    ]

    stream, aresponse = ai.generate_stream(
        model='programmableModel',
        prompt='hi',
        tool_choice=ToolChoice.REQUIRED,
        tools=['testTool'],
    )

    chunks = []
    async for chunk in stream:
        summary = ''
        if chunk.role:
            summary += f'{chunk.role} '
        for p in chunk.content:
            summary += str(type(p.root).__name__)
            if isinstance(p.root, TextPart):
                summary += f' {p.root.text}'
        chunks.append(summary)

    response = await aresponse

    assert response.text == 'tool called'
    assert response.request is not None
    assert response.request.messages is not None
    assert response.request.messages[0] == Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])
    assert response.request.messages[1] == tool_request_msg
    assert response.request.messages[2] == Message(
        role=Role.TOOL,
        content=[Part(root=ToolResponsePart(tool_response=ToolResponse(ref='123', name='testTool', output='abc')))],
    )
    assert chunks == [
        'model ToolRequestPart',
        'tool ToolResponsePart',
        'model TextPart tool called',
    ]


@pytest.mark.asyncio
async def test_generate_stream_no_need_to_await_response(
    setup_test: SetupFixture,
) -> None:
    """Test that the generate stream function no need to await response."""
    ai, _, pm, *_ = setup_test

    pm.responses.append(
        GenerateResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='something else'))]),
        )
    )
    pm.chunks = [
        [
            GenerateResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='h'))]),
            GenerateResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='i'))]),
        ],
    ]

    stream, _ = ai.generate_stream(model='programmableModel', prompt='do it')
    chunks = ''
    async for chunk in stream:
        chunks += chunk.text
    assert chunks == 'hi'


@pytest.mark.asyncio
async def test_generate_with_output(setup_test: SetupFixture) -> None:
    """Test that the generate function with output works."""
    ai, *_ = setup_test

    class TestSchema(BaseModel):
        foo: int | None = Field(None, description='foo field')
        bar: str | None = Field(None, description='bar field')

    want = GenerateRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
        ],
        config={},
        tools=[],
        output=OutputConfig(
            format='json',
            schema={
                'properties': {
                    'foo': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                        'default': None,
                        'description': 'foo field',
                        'title': 'Foo',
                    },
                    'bar': {
                        'anyOf': [{'type': 'string'}, {'type': 'null'}],
                        'default': None,
                        'description': 'bar field',
                        'title': 'Bar',
                    },
                },
                'title': 'TestSchema',
                'type': 'object',
            },
            constrained=True,
            content_type='application/json',
        ),
    )

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        output=Output(schema=TestSchema, format='json', content_type='application/json', constrained=True),
        output_instructions=False,
    )

    assert response.request == want

    _, response = ai.generate_stream(
        model='echoModel',
        prompt='hi',
        output=Output(schema=TestSchema, format='json', content_type='application/json', constrained=True),
        output_instructions=False,
    )

    assert (await response).request == want


@pytest.mark.asyncio
async def test_generate_defaults_to_json_format(
    setup_test: SetupFixture,
) -> None:
    """When Output is provided, format will default to json."""
    ai, *_ = setup_test

    class TestSchema(BaseModel):
        foo: int | None = Field(None, description='foo field')
        bar: str | None = Field(None, description='bar field')

    want = GenerateRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
        ],
        config={},
        tools=[],
        output=OutputConfig(
            format='json',
            schema={
                'properties': {
                    'foo': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                        'default': None,
                        'description': 'foo field',
                        'title': 'Foo',
                    },
                    'bar': {
                        'anyOf': [{'type': 'string'}, {'type': 'null'}],
                        'default': None,
                        'description': 'bar field',
                        'title': 'Bar',
                    },
                },
                'title': 'TestSchema',
                'type': 'object',
            },
            # these get populated by the format
            constrained=True,
            content_type='application/json',
        ),
    )

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        output=Output(schema=TestSchema),
    )

    assert response.request == want

    _, response = ai.generate_stream(
        model='echoModel',
        prompt='hi',
        output=Output(schema=TestSchema),
    )

    assert (await response).request == want


@pytest.mark.asyncio
async def test_generate_json_format_unconstrained(
    setup_test: SetupFixture,
) -> None:
    """When Output is provided, format will default to json."""
    ai, *_ = setup_test

    class TestSchema(BaseModel):
        foo: int | None = Field(None, description='foo field')
        bar: str | None = Field(None, description='bar field')

    want = GenerateRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
        ],
        config={},
        tools=[],
        output=OutputConfig(
            format='json',
            schema={
                'properties': {
                    'foo': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                        'default': None,
                        'description': 'foo field',
                        'title': 'Foo',
                    },
                    'bar': {
                        'anyOf': [{'type': 'string'}, {'type': 'null'}],
                        'default': None,
                        'description': 'bar field',
                        'title': 'Bar',
                    },
                },
                'title': 'TestSchema',
                'type': 'object',
            },
            constrained=False,
            content_type='application/json',
        ),
    )

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        output=Output(schema=TestSchema, constrained=False),
    )

    assert response.request == want

    _, response = ai.generate_stream(
        model='echoModel',
        prompt='hi',
        output=Output(schema=TestSchema, constrained=False),
    )

    assert (await response).request == want


@pytest.mark.asyncio
async def test_generate_with_middleware(
    setup_test: SetupFixture,
) -> None:
    """When middleware is provided, applies it."""
    ai, *_ = setup_test

    async def pre_middle(req: GenerateRequest, ctx: ActionRunContext, next: ModelMiddlewareNext) -> GenerateResponse:
        txt = ''.join(text_from_message(m) for m in req.messages)
        return await next(
            GenerateRequest(
                messages=[
                    Message(role=Role.USER, content=[Part(root=TextPart(text=f'PRE {txt}'))]),
                ],
            ),
            ctx,
        )

    async def post_middle(req: GenerateRequest, ctx: ActionRunContext, next: ModelMiddlewareNext) -> GenerateResponse:
        resp: GenerateResponse = await next(req, ctx)
        assert resp.message is not None
        txt = text_from_message(resp.message)
        return GenerateResponse(
            finish_reason=resp.finish_reason,
            message=Message(role=Role.USER, content=[Part(root=TextPart(text=f'{txt} POST'))]),
        )

    want = '[ECHO] user: "PRE hi" POST'

    response = await ai.generate(model='echoModel', prompt='hi', use=[pre_middle, post_middle])

    assert response.text == want

    _, response = ai.generate_stream(model='echoModel', prompt='hi', use=[pre_middle, post_middle])

    assert (await response).text == want


@pytest.mark.asyncio
async def test_generate_passes_through_current_action_context(
    setup_test: SetupFixture,
) -> None:
    """Test that generate uses current action context by default."""
    ai, *_ = setup_test

    async def inject_context(
        req: GenerateRequest, ctx: ActionRunContext, next: ModelMiddlewareNext
    ) -> GenerateResponse:
        txt = ''.join(text_from_message(m) for m in req.messages)
        return await next(
            GenerateRequest(
                messages=[
                    Message(
                        role=Role.USER,
                        content=[Part(root=TextPart(text=f'{txt} {ctx.context}'))],
                    ),
                ],
            ),
            ctx,
        )

    async def action_fn() -> GenerateResponse:
        return await ai.generate(model='echoModel', prompt='hi', use=[inject_context])

    action = ai.registry.register_action(name='test_action', kind=ActionKind.CUSTOM, fn=action_fn)
    action_response = await action.arun(context={'foo': 'bar'})

    assert action_response.response.text == '''[ECHO] user: "hi {'foo': 'bar'}"'''


@pytest.mark.asyncio
async def test_generate_uses_explicitly_passed_in_context(
    setup_test: SetupFixture,
) -> None:
    """Generate uses specific context instead of current action context."""
    ai, *_ = setup_test

    async def inject_context(
        req: GenerateRequest, ctx: ActionRunContext, next: ModelMiddlewareNext
    ) -> GenerateResponse:
        txt = ''.join(text_from_message(m) for m in req.messages)
        return await next(
            GenerateRequest(
                messages=[
                    Message(
                        role=Role.USER,
                        content=[Part(root=TextPart(text=f'{txt} {ctx.context}'))],
                    ),
                ],
            ),
            ctx,
        )

    async def action_fn() -> GenerateResponse:
        return await ai.generate(
            model='echoModel',
            prompt='hi',
            use=[inject_context],
            context={'bar': 'baz'},
        )

    action = ai.registry.register_action(name='test_action', kind=ActionKind.CUSTOM, fn=action_fn)
    action_response = await action.arun(context={'foo': 'bar'})

    assert action_response.response.text == '''[ECHO] user: "hi {'bar': 'baz'}"'''


@pytest.mark.asyncio
async def test_generate_json_format_unconstrained_with_instructions(
    setup_test: SetupFixture,
) -> None:
    """When Output is provided, format will default to json."""
    ai, *_ = setup_test

    class TestSchema(BaseModel):
        foo: int | None = Field(None, description='foo field')
        bar: str | None = Field(None, description='bar field')

    want = GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='hi')),
                    Part(
                        root=TextPart(
                            text=(
                                'Output should be in JSON format and conform to the '
                                'following schema:\n\n```\n{\n  "properties": {\n    '
                                '"foo": {\n      "anyOf": [\n        {\n          '
                                '"type": "integer"\n        },\n        {\n          '
                                '"type": "null"\n        }\n      ],\n      '
                                '"default": null,\n      "description": "foo field",\n      '
                                '"title": "Foo"\n    },\n    "bar": {\n      '
                                '"anyOf": [\n        {\n          "type": "string"\n        },\n        '
                                '{\n          "type": "null"\n        }\n      ],\n      '
                                '"default": null,\n      "description": "bar field",\n      '
                                '"title": "Bar"\n    }\n  },\n  "title": "TestSchema",\n  '
                                '"type": "object"\n}\n```\n'
                            ),
                            metadata=Metadata(root={'purpose': 'output'}),
                        )
                    ),
                ],
            )
        ],
        config={},
        tools=[],
        output=OutputConfig(
            format='json',
            schema={
                'properties': {
                    'foo': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                        'default': None,
                        'description': 'foo field',
                        'title': 'Foo',
                    },
                    'bar': {
                        'anyOf': [{'type': 'string'}, {'type': 'null'}],
                        'default': None,
                        'description': 'bar field',
                        'title': 'Bar',
                    },
                },
                'title': 'TestSchema',
                'type': 'object',
            },
            constrained=False,
            content_type='application/json',
        ),
    )

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        output=Output(schema=TestSchema, constrained=False),
        output_instructions=True,
    )

    assert response.request == want

    _, response = ai.generate_stream(
        model='echoModel',
        prompt='hi',
        output=Output(schema=TestSchema, constrained=False),
        output_instructions=True,
    )

    assert (await response).request == want


@pytest.mark.asyncio
async def test_generate_simulates_doc_grounding(
    setup_test: SetupFixture,
) -> None:
    """Test that generate simulates doc grounding."""
    ai, *_ = setup_test

    want_msg = Message(
        role=Role.USER,
        content=[
            Part(root=TextPart(text='hi')),
            Part(
                root=TextPart(
                    text='\n\nUse the following information to complete your task:' + '\n\n- [0]: doc content 1\n\n',
                    metadata=Metadata(root={'purpose': 'context'}),
                )
            ),
        ],
    )

    response = await ai.generate(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='hi'))],
            ),
        ],
        docs=[DocumentData(content=[DocumentPart(root=TextPart(text='doc content 1'))])],
    )

    assert response.request is not None
    assert response.request.messages is not None
    assert response.request.messages[0] == want_msg

    _, response = ai.generate_stream(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='hi'))],
            ),
        ],
        docs=[DocumentData(content=[DocumentPart(root=TextPart(text='doc content 1'))])],
    )

    resp = await response
    assert resp.request is not None
    assert resp.request.messages is not None
    assert resp.request.messages[0] == want_msg


class MockBananaFormat(FormatDef):
    """Mock format for testing the format."""

    def __init__(self) -> None:
        """Initialize the format."""
        super().__init__(
            'banana',
            FormatterConfig(
                format='json',
                content_type='application/banana',
                constrained=True,
            ),
        )

    def handle(self, schema: dict[str, Any] | None) -> Formatter:
        """Handle the format."""

        def message_parser(msg: Message) -> str:
            """Parse the message."""
            parts = [p.root.text or '' for p in msg.content if hasattr(p.root, 'text') and p.root.text]
            return f'banana {"".join(parts)}'  # type: ignore[arg-type]

        def chunk_parser(chunk: GenerateResponseChunk) -> str:
            """Parse the chunk."""
            parts = [p.root.text or '' for p in chunk.content if hasattr(p.root, 'text') and p.root.text]
            return f'banana chunk {"".join(parts)}'  # type: ignore[arg-type]

        instructions: str | None = None

        if schema:
            instructions = f'schema: {json.dumps(schema)}'

        return Formatter(
            chunk_parser=chunk_parser,
            message_parser=message_parser,
            instructions=instructions,
        )


@pytest.mark.asyncio
async def test_define_format(setup_test: SetupFixture) -> None:
    """Test that the define format function works."""
    ai, _, pm, *_ = setup_test

    ai.define_format(MockBananaFormat())

    class TestSchema(BaseModel):
        foo: int | None = Field(None, description='foo field')
        bar: str | None = Field(None, description='bar field')

    pm.responses = [
        (
            GenerateResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='model says'))]),
            )
        )
    ]
    pm.chunks = [
        [
            GenerateResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='1'))]),
            GenerateResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='2'))]),
            GenerateResponseChunk(role=Role.MODEL, content=[Part(root=TextPart(text='3'))]),
        ]
    ]

    chunks = []

    stream, aresponse = ai.generate_stream(
        model='programmableModel',
        prompt='hi',
        output=Output(schema=TestSchema, format='banana'),
    )

    async for chunk in stream:
        chunks.append(chunk.output)

    response = await aresponse

    assert response.output == 'banana model says'
    assert chunks == ['banana chunk 1', 'banana chunk 2', 'banana chunk 3']

    assert response.request == GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='hi')),
                    Part(
                        root=TextPart(
                            text=(
                                'schema: {"properties": {"foo": {"anyOf": [{"type": "integer"}, '
                                '{"type": "null"}], "default": null, "description": "foo field", '
                                '"title": "Foo"}, "bar": {"anyOf": [{"type": "string"}, '
                                '{"type": "null"}], "default": null, "description": "bar field", '
                                '"title": "Bar"}}, "title": "TestSchema", "type": "object"}'
                            ),
                            metadata=Metadata(root={'purpose': 'output'}),
                        )
                    ),
                ],
            ),
        ],
        config={},
        tools=[],
        output=OutputConfig(
            format='json',
            schema={
                'properties': {
                    'foo': {
                        'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                        'default': None,
                        'description': 'foo field',
                        'title': 'Foo',
                    },
                    'bar': {
                        'anyOf': [{'type': 'string'}, {'type': 'null'}],
                        'default': None,
                        'description': 'bar field',
                        'title': 'Bar',
                    },
                },
                'title': 'TestSchema',
                'type': 'object',
            },
            # these get populated by the format
            constrained=True,
            content_type='application/banana',
        ),
    )


def test_define_model_default_metadata(setup_test: SetupFixture) -> None:
    """Test that the define model function works."""
    ai, _, _, *_ = setup_test

    def foo_model_fn(request: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
        return GenerateResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='banana!'))]))

    action = ai.define_model(
        name='foo',
        fn=foo_model_fn,
    )

    assert action.metadata['model'] == {
        'label': 'foo',
    }


def test_define_model_with_schema(setup_test: SetupFixture) -> None:
    """Test that the define model function with schema works."""
    ai, _, _, *_ = setup_test

    class Config(BaseModel):
        field_a: str = Field(description='a field')
        field_b: str = Field(description='b field')

    def foo_model_fn(request: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
        return GenerateResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='banana!'))]))

    action = ai.define_model(
        name='foo',
        fn=foo_model_fn,
        config_schema=Config,
    )
    assert action.metadata['model'] == {
        'customOptions': {
            'properties': {
                'field_a': {
                    'description': 'a field',
                    'title': 'Field A',
                    'type': 'string',
                },
                'field_b': {
                    'description': 'b field',
                    'title': 'Field B',
                    'type': 'string',
                },
            },
            'required': [
                'field_a',
                'field_b',
            ],
            'title': 'Config',
            'type': 'object',
        },
        'label': 'foo',
    }


def test_define_model_with_info(setup_test: SetupFixture) -> None:
    """Test that the define model function with info works."""
    ai, _, _, *_ = setup_test

    def foo_model_fn(request: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
        return GenerateResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='banana!'))]))

    action = ai.define_model(
        name='foo',
        fn=foo_model_fn,
        info=ModelInfo(label='Foo Bar', supports=Supports(multiturn=True, tools=True)),
    )
    assert action.metadata['model'] == {
        'label': 'Foo Bar',
        'supports': {
            'multiturn': True,
            'tools': True,
        },
    }


def test_define_retriever_default_metadata(setup_test: SetupFixture) -> None:
    """Test that the define retriever function works."""
    ai, _, _, *_ = setup_test

    def my_retriever(doc: Document, config: dict[str, Any] | None = None) -> RetrieverResponse:
        return RetrieverResponse(documents=[Document.from_text('Hello'), Document.from_text('World')])

    action = ai.define_retriever(
        name='fooRetriever',
        fn=my_retriever,
    )

    assert action.metadata['retriever'] == {
        'label': 'fooRetriever',
    }


def test_define_retriever_with_schema(setup_test: SetupFixture) -> None:
    """Test that the define retriever function with schema works."""
    ai, _, _, *_ = setup_test

    class Config(BaseModel):
        field_a: str = Field(description='a field')
        field_b: str = Field(description='b field')

    def my_retriever(doc: Document, config: dict[str, Any] | None = None) -> RetrieverResponse:
        return RetrieverResponse(documents=[Document.from_text('Hello'), Document.from_text('World')])

    action = ai.define_retriever(
        name='fooRetriever',
        fn=my_retriever,
        config_schema=Config,
    )

    assert action.metadata['retriever'] == {
        'customOptions': {
            'properties': {
                'field_a': {
                    'description': 'a field',
                    'title': 'Field A',
                    'type': 'string',
                },
                'field_b': {
                    'description': 'b field',
                    'title': 'Field B',
                    'type': 'string',
                },
            },
            'required': [
                'field_a',
                'field_b',
            ],
            'title': 'Config',
            'type': 'object',
        },
        'label': 'fooRetriever',
    }


def test_define_evaluator_simple(setup_test: SetupFixture) -> None:
    """Test that the define evaluator function works."""
    ai, _, _, *_ = setup_test

    async def my_eval_fn(datapoint: BaseDataPoint, options: dict[str, Any] | None = None) -> EvalFnResponse:
        return EvalFnResponse(
            test_case_id=datapoint.test_case_id or '',
            evaluation=Score(score=True, details=Details(reasoning='I think it is true')),
        )

    action = ai.define_evaluator(
        name='my_eval',
        display_name='Test evaluator',
        definition='Test evaluator that always returns True',
        fn=my_eval_fn,
    )

    assert action.metadata['evaluator'] == {
        'label': 'my_eval',
        'evaluatorDefinition': 'Test evaluator that always returns True',
        'evaluatorDisplayName': 'Test evaluator',
        'evaluatorIsBilled': False,
    }


def test_define_evaluator_custom_config(setup_test: SetupFixture) -> None:
    """Test that the define evaluator function works."""
    ai, _, _, *_ = setup_test

    class CustomOption(BaseModel):
        foo_bar: str = Field('baz', description='foo_bar field')

    async def my_eval_fn(datapoint: BaseDataPoint, options: dict[str, Any] | None = None) -> EvalFnResponse:
        return EvalFnResponse(
            test_case_id=datapoint.test_case_id or '',
            evaluation=Score(
                score=True, details=Details(reasoning=options.get('foo_bar', 'baz') if options else 'baz')
            ),
        )

    action = ai.define_evaluator(
        name='my_eval',
        display_name='Test evaluator',
        definition='Test evaluator that always returns True',
        fn=my_eval_fn,
        config_schema=CustomOption,
    )

    assert action.metadata['evaluator'] == {
        'label': 'my_eval',
        'evaluatorDefinition': 'Test evaluator that always returns True',
        'evaluatorDisplayName': 'Test evaluator',
        'evaluatorIsBilled': False,
        'customOptions': {
            'properties': {
                'foo_bar': {
                    'default': 'baz',
                    'description': 'foo_bar field',
                    'title': 'Foo Bar',
                    'type': 'string',
                }
            },
            'title': 'CustomOption',
            'type': 'object',
        },
    }


def test_define_batch_evaluator(setup_test: SetupFixture) -> None:
    """Test that the define batch evaluator function works."""
    ai, _, _, *_ = setup_test

    async def my_eval_fn(req: EvalRequest, options: object | None) -> list[EvalFnResponse]:
        eval_responses: list[EvalFnResponse] = []
        for index in range(len(req.dataset)):
            datapoint = req.dataset[index]
            eval_responses.append(
                EvalFnResponse(
                    test_case_id=f'testCase{index}',
                    evaluation=Score(
                        score=True,
                        details=Details(reasoning=f'I think {datapoint.input} is true'),
                    ),
                )
            )

        return eval_responses

    action = ai.define_batch_evaluator(
        name='my_eval',
        display_name='Test evaluator',
        definition='Test evaluator that always returns True',
        fn=my_eval_fn,
    )

    assert action.metadata['evaluator'] == {
        'label': 'my_eval',
        'evaluatorDefinition': 'Test evaluator that always returns True',
        'evaluatorDisplayName': 'Test evaluator',
        'evaluatorIsBilled': False,
    }


@pytest.mark.asyncio
async def test_define_sync_flow(setup_test: SetupFixture) -> None:
    """Test defining a synchronous flow."""
    ai, _, _, *_ = setup_test

    @cast(Any, ai.flow())
    def my_flow(input: str, ctx: ActionRunContext | None = None) -> str:
        if ctx:
            ctx.send_chunk(1)
            ctx.send_chunk(2)
            ctx.send_chunk(3)
        return input

    assert my_flow('banana') == 'banana'

    stream, response = my_flow.stream('banana2')

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == [1, 2, 3]
    assert (await response).response == 'banana2'


@pytest.mark.asyncio
async def test_define_async_flow(setup_test: SetupFixture) -> None:
    """Test defining an asynchronous flow."""
    ai, _, _, *_ = setup_test

    @ai.flow()
    async def my_flow(input: str, ctx: ActionRunContext | None = None) -> str:
        if ctx:
            ctx.send_chunk(1)
            ctx.send_chunk(2)
            ctx.send_chunk(3)
        return input

    assert (await my_flow('banana')) == 'banana'

    stream, response = my_flow.stream('banana2')

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == [1, 2, 3]
    assert (await response).response == 'banana2'


@pytest.mark.asyncio
async def test_evaluate(setup_test: SetupFixture) -> None:
    """Test that the evaluate function works."""
    ai, _, _, *_ = setup_test

    async def my_eval_fn(datapoint: BaseDataPoint, options: object | None) -> EvalFnResponse:
        return EvalFnResponse(
            test_case_id=datapoint.test_case_id or '',
            evaluation=Score(score=True, details=Details(reasoning='I think it is true')),
        )

    ai.define_evaluator(
        name='my_eval',
        display_name='Test evaluator',
        definition='Test evaluator that always returns True',
        fn=my_eval_fn,
    )

    dataset = [
        BaseDataPoint(input='hi', output='hi', test_case_id='case1'),
        BaseDataPoint(input='bye', output='bye', test_case_id='case2'),
    ]

    response = await ai.evaluate(evaluator='my_eval', dataset=dataset)

    assert isinstance(response, EvalResponse)
    assert len(response.root) == 2
    assert response.root[0].test_case_id == 'case1'
    assert isinstance(response.root[0].evaluation, Score)
    assert response.root[0].evaluation.score is True
    assert response.root[1].test_case_id == 'case2'
    assert isinstance(response.root[1].evaluation, Score)
    assert response.root[1].evaluation.score is True
