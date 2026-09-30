# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MCP tool result translation."""

from typing import Any

import pytest
from genkit_mcp._errors import McpConnectionFailedError, McpToolResultError
from genkit_mcp._tools import (
    call_result_to_multipart,
    mcp_tool_name,
    mcp_tool_to_action,
    mcp_tools_to_actions,
    validate_provider_name,
    validate_tool_prefix,
)
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
    Tool,
)
from pydantic import AnyUrl
from structlog.testing import capture_logs

from genkit import Action, Part
from genkit._core._error import GenkitError
from genkit._core._typing import Resource


def text(value: str) -> TextContent:
    return TextContent(type='text', text=value)


def server_tool_of(action: Action) -> object:
    """Read the server's own tool name out of an action's mcp metadata."""
    mcp = action.metadata['mcp']
    assert isinstance(mcp, dict)
    return mcp['tool']


@pytest.mark.parametrize(
    ('result', 'output', 'content', 'metadata'),
    [
        (CallToolResult(content=[text('hello'), text(' world')]), 'hello world', None, None),
        (CallToolResult(content=[text('{"answer": 42}')]), {'answer': 42}, None, None),
        (CallToolResult(content=[text('[1, 2]')]), [1, 2], None, None),
        (CallToolResult(content=[text('{not json')]), '{not json', None, None),
        (CallToolResult(content=[text('42')]), '42', None, None),
        (CallToolResult(content=[text('true')]), 'true', None, None),
        (CallToolResult(content=[text('null')]), 'null', None, None),
        (
            CallToolResult(content=[text('ignored')], structuredContent={'answer': 'structured'}),
            {'answer': 'structured'},
            None,
            None,
        ),
        (
            CallToolResult(
                content=[text('bad'), ImageContent(type='image', mimeType='image/png', data='abc')],
                isError=True,
                _meta={'requestId': 'r1'},  # ty: ignore[unknown-argument]
            ),
            {'error': 'bad'},
            None,
            {'requestId': 'r1', 'mcp': {'isError': True}},
        ),
    ],
)
def test_call_result_to_multipart_text_and_errors(
    result: CallToolResult,
    output: object,
    content: object,
    metadata: object,
) -> None:
    """Text, structured content, and error results keep their documented shape."""
    response = call_result_to_multipart(result)
    assert response.output == output
    assert response.content == content
    assert response.metadata == metadata


def test_call_result_to_multipart_maps_multipart_content_and_meta() -> None:
    """Media, resource links, and embedded resources use Genkit multipart parts."""
    result = CallToolResult(
        content=[
            text('before'),
            ImageContent(type='image', mimeType='image/png', data='image-data'),
            AudioContent(type='audio', mimeType='audio/mpeg', data='audio-data'),
            EmbeddedResource(
                type='resource',
                resource=TextResourceContents(uri=AnyUrl('file:///note.txt'), mimeType='text/plain', text='note'),
            ),
            EmbeddedResource(
                type='resource',
                resource=BlobResourceContents(
                    uri=AnyUrl('file:///data.bin'),
                    mimeType='application/octet-stream',
                    blob='blob-data',
                ),
            ),
            ResourceLink(
                type='resource_link',
                name='documentation',
                uri=AnyUrl('https://example.test/docs'),
            ),
        ],
        _meta={'trace': 'mcp-1'},  # ty: ignore[unknown-argument]
    )

    response = call_result_to_multipart(result)

    assert response.output == 'before\n\nResource (file:///note.txt):\nnote'
    assert response.metadata == {'trace': 'mcp-1'}
    assert response.content is not None
    assert response.content == [
        Part.from_media(url='data:image/png;base64,image-data', content_type='image/png'),
        Part.from_media(url='data:audio/mpeg;base64,audio-data', content_type='audio/mpeg'),
        Part.from_media(url='data:application/octet-stream;base64,blob-data', content_type='application/octet-stream'),
        Part(resource=Resource(uri='https://example.test/docs')),
    ]


@pytest.mark.asyncio
async def test_mcp_tool_to_action_preserves_schema_metadata_and_context_meta() -> None:
    """The action retains an MCP schema verbatim and forwards per-call metadata."""
    awkward_schema: dict[str, Any] = {
        '$ref': '#/$defs/Lookup',
        '$defs': {'Lookup': {'properties': {'q': {'type': 'string'}}}},
    }
    tool = Tool(
        name='lookup',
        description='Find a record.',
        inputSchema=awkward_schema,
        _meta={'audience': 'internal'},  # ty: ignore[unknown-argument]
    )
    calls: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []

    async def call(name: str, arguments: dict[str, Any], meta: dict[str, Any] | None) -> CallToolResult:
        calls.append((name, arguments, meta))
        return CallToolResult(content=[text('ok')])

    action = mcp_tool_to_action(tool, 'catalog', call)
    response = (await action.run({'q': 'coffee'}, context={'mcp': {'_meta': {'progressToken': 7}}})).response

    assert mcp_tool_name('catalog', 'lookup') == 'catalog_lookup'
    assert action.name == 'catalog_lookup'
    assert action.input_schema == awkward_schema
    assert action.metadata['name'] == 'catalog_lookup'
    assert action.metadata['mcp'] == {'prefix': 'catalog', 'tool': 'lookup', '_meta': {'audience': 'internal'}}
    assert calls == [('lookup', {'q': 'coffee'}, {'progressToken': 7})]
    assert response.output == 'ok'


@pytest.mark.asyncio
async def test_mcp_tool_to_action_folds_a_tool_result_error_into_an_error_envelope() -> None:
    """A result which violates the tool's output schema is a tool error the model can retry."""
    tool = Tool(name='lookup', inputSchema={'type': 'object'})

    async def call(name: str, arguments: dict[str, Any], meta: dict[str, Any] | None) -> CallToolResult:  # noqa: ARG001
        raise McpToolResultError('MCP output did not match the declared schema')

    action = mcp_tool_to_action(tool, 'catalog', call)
    response = (await action.run({})).response

    assert response.output == {'error': 'MCP output did not match the declared schema'}
    assert response.content is None
    assert response.metadata == {'mcp': {'isError': True}}


@pytest.mark.asyncio
async def test_mcp_tool_to_action_lets_a_runtime_error_through() -> None:
    """A bare RuntimeError is infrastructure failing, not a result the model should retry."""
    tool = Tool(name='lookup', inputSchema={'type': 'object'})

    async def call(name: str, arguments: dict[str, Any], meta: dict[str, Any] | None) -> CallToolResult:  # noqa: ARG001
        raise RuntimeError('Event loop is closed')

    action = mcp_tool_to_action(tool, 'catalog', call)

    with pytest.raises(GenkitError) as raised:
        await action.run({})

    assert type(raised.value.cause) is RuntimeError
    assert str(raised.value.cause) == 'Event loop is closed'


@pytest.mark.asyncio
async def test_mcp_tool_to_action_lets_a_connection_failure_through() -> None:
    """A dead connection is not a tool error: the model must not retry against it."""
    tool = Tool(name='lookup', inputSchema={'type': 'object'})

    async def call(name: str, arguments: dict[str, Any], meta: dict[str, Any] | None) -> CallToolResult:  # noqa: ARG001
        raise McpConnectionFailedError("MCP connection to 'npx' failed")

    action = mcp_tool_to_action(tool, 'catalog', call)

    with pytest.raises(GenkitError) as raised:
        await action.run({})

    assert isinstance(raised.value.cause, McpConnectionFailedError)


@pytest.mark.parametrize(
    'prefix',
    ['everything', 'mcp_everything', 'fake-server', 'A1', '_private'],
)
def test_validate_tool_prefix_accepts_a_usable_prefix(prefix: str) -> None:
    validate_tool_prefix(prefix)


@pytest.mark.parametrize(
    'prefix',
    [
        'mcp-servers/everything',
        'My App',
        '',
        '9lives',
        'caf\u00e9',
        'tools.everything',
    ],
)
def test_validate_tool_prefix_rejects_a_prefix_a_model_would_reject(prefix: str) -> None:
    """A prefix which cannot appear in a function declaration name fails loudly."""
    with pytest.raises(ValueError, match='cannot prefix an MCP tool name'):
        validate_tool_prefix(prefix)


@pytest.mark.parametrize(
    'name',
    ['everything', 'my server', 'tools.everything', 'caf\u00e9', '9lives'],
)
def test_validate_provider_name_accepts_a_name_a_selector_can_carry(name: str) -> None:
    """A provider name reaches no model, so only the selector constrains it."""
    validate_provider_name(name)


@pytest.mark.parametrize('name', ['mcp-servers/everything', 'fake:everything', ':', '/', ''])
def test_validate_provider_name_rejects_a_name_no_selector_could_carry(name: str) -> None:
    """A name holding a selector separator could never be picked out again."""
    with pytest.raises(ValueError, match='cannot name an MCP provider'):
        validate_provider_name(name)


async def no_content(name: str, arguments: dict[str, Any], meta: dict[str, Any] | None) -> CallToolResult:  # noqa: ARG001
    return CallToolResult(content=[])


@pytest.mark.parametrize(
    ('tool_name', 'expected'),
    [
        ('lookup', 'catalog_lookup'),
        ('look.up', 'catalog_look_up'),
        ('fs/read_file', 'catalog_fs_read_file'),
        ('caf\u00e9 menu', 'catalog_caf__menu'),
    ],
)
def test_mcp_tool_name_rewrites_what_a_model_would_reject(tool_name: str, expected: str) -> None:
    assert mcp_tool_name('catalog', tool_name) == expected


@pytest.mark.parametrize('tool_name', ['look.up', 'fs/read_file'])
@pytest.mark.asyncio
async def test_mcp_tool_to_action_sanitises_the_name_and_keeps_the_server_name(tool_name: str) -> None:
    """The model sees a name it accepts while the server is still called by its own name."""
    calls: list[str] = []

    async def call(name: str, arguments: dict[str, Any], meta: dict[str, Any] | None) -> CallToolResult:  # noqa: ARG001
        calls.append(name)
        return CallToolResult(content=[text('ok')])

    with capture_logs() as logs:
        action = mcp_tool_to_action(Tool(name=tool_name, inputSchema={'type': 'object'}), 'catalog', call)
    await action.run({})

    assert action.name == 'catalog_' + tool_name.replace('.', '_').replace('/', '_')
    assert action.metadata['name'] == action.name
    assert server_tool_of(action) == tool_name
    assert calls == [tool_name]
    assert [entry['log_level'] for entry in logs] == ['warning']
    assert logs[0]['tool'] == action.name
    assert logs[0]['server_tool'] == tool_name


def test_mcp_tool_to_action_warns_about_a_name_longer_than_a_model_accepts() -> None:
    """Length cannot be rewritten, so an over-long name is reported rather than refused."""
    tool_name = 'a' * 64

    with capture_logs() as logs:
        action = mcp_tool_to_action(Tool(name=tool_name, inputSchema={'type': 'object'}), 'catalog', no_content)

    assert action.name == f'catalog_{tool_name}'
    assert [entry['log_level'] for entry in logs] == ['warning']
    assert logs[0]['tool'] == f'catalog_{tool_name}'


def test_mcp_tools_to_actions_maps_a_listing_in_order() -> None:
    tools = [Tool(name='lookup', inputSchema={'type': 'object'}), Tool(name='look.up', inputSchema={'type': 'object'})]

    actions = mcp_tools_to_actions(tools, 'catalog', no_content)

    assert [action.name for action in actions] == ['catalog_lookup', 'catalog_look_up']
    assert [server_tool_of(action) for action in actions] == ['lookup', 'look.up']


def test_mcp_tools_to_actions_refuses_a_listing_whose_sanitised_names_collide() -> None:
    """Two server tools which would answer to one Genkit name cannot both be handed out."""
    tools = [Tool(name='look.up', inputSchema={'type': 'object'}), Tool(name='look/up', inputSchema={'type': 'object'})]

    with pytest.raises(ValueError, match="'look.up' and 'look/up' both map to the Genkit tool name 'catalog_look_up'"):
        mcp_tools_to_actions(tools, 'catalog', no_content)
