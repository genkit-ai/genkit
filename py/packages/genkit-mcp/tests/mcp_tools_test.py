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
from genkit_mcp._errors import McpConnectionFailedError
from genkit_mcp._tools import (
    call_result_to_multipart,
    mcp_tool_name,
    mcp_tool_to_action,
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

from genkit._core._error import GenkitError
from genkit._core._typing import MediaPart, ResourcePart


def text(value: str) -> TextContent:
    return TextContent(type='text', text=value)


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
    assert [type(part.root) for part in response.content] == [MediaPart, MediaPart, MediaPart, ResourcePart]
    assert response.content[0].root.media.url == 'data:image/png;base64,image-data'  # type: ignore[union-attr]
    assert response.content[1].root.media.url == 'data:audio/mpeg;base64,audio-data'  # type: ignore[union-attr]
    assert response.content[2].root.media.url == 'data:application/octet-stream;base64,blob-data'  # type: ignore[union-attr]
    assert response.content[3].root.resource.uri == 'https://example.test/docs'  # type: ignore[union-attr]


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
    assert action.metadata['mcp'] == {'prefix': 'catalog', '_meta': {'audience': 'internal'}}
    assert calls == [('lookup', {'q': 'coffee'}, {'progressToken': 7})]
    assert response.output == 'ok'


@pytest.mark.asyncio
async def test_mcp_tool_to_action_folds_call_validation_error_into_mcp_error() -> None:
    """MCP output-schema validation errors become retryable tool error envelopes."""
    tool = Tool(name='lookup', inputSchema={'type': 'object'})

    async def call(name: str, arguments: dict[str, Any], meta: dict[str, Any] | None) -> CallToolResult:  # noqa: ARG001
        raise RuntimeError('MCP output did not match the declared schema')

    action = mcp_tool_to_action(tool, 'catalog', call)
    response = (await action.run({})).response

    assert response.output == {'error': 'MCP output did not match the declared schema'}
    assert response.content is None
    assert response.metadata == {'mcp': {'isError': True}}


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


@pytest.mark.parametrize(
    ('prefix', 'tool_name'),
    [('catalog', 'look/up'), ('catalog', 'a' * 64)],
)
def test_mcp_tool_to_action_warns_about_a_name_a_model_would_reject(prefix: str, tool_name: str) -> None:
    """The tool half of the name is the server's, so it is reported rather than refused."""

    async def call(name: str, arguments: dict[str, Any], meta: dict[str, Any] | None) -> CallToolResult:  # noqa: ARG001
        return CallToolResult(content=[])

    with capture_logs() as logs:
        action = mcp_tool_to_action(Tool(name=tool_name, inputSchema={'type': 'object'}), prefix, call)

    assert action.name == f'{prefix}_{tool_name}'
    assert [entry['log_level'] for entry in logs] == ['warning']
    assert logs[0]['tool'] == f'{prefix}_{tool_name}'
