#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Genkit extra API methods."""

from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from genkit.ai import Genkit
from genkit.ai._registry import SimpleRetrieverOptions
from genkit.core.action import Action
from genkit.core.action._action import _action_context
from genkit.core.action.types import ActionKind
from genkit.core.typing import DocumentPart, Operation
from genkit.types import DocumentData, RetrieverRequest, RetrieverResponse, TextPart


@pytest.mark.asyncio
async def test_genkit_run() -> None:
    """Test Genkit.run method."""
    ai = Genkit()

    def sync_fn() -> str:
        return 'hello'

    async def async_fn() -> str:
        return 'world'

    res1 = await ai.run('test1', sync_fn)
    assert res1 == 'hello'

    res2 = await ai.run('test2', async_fn)
    assert res2 == 'world'

    # Test with metadata
    res3 = await ai.run('test3', sync_fn, metadata={'foo': 'bar'})
    assert res3 == 'hello'

    # Test with input overload
    async def multiply(x: int) -> int:
        return x * 2

    res4 = await ai.run('multiply', 10, multiply)
    assert res4 == 20


@pytest.mark.asyncio
async def test_genkit_dynamic_tool() -> None:
    """Test Genkit.dynamic_tool method."""
    ai = Genkit()

    def my_tool(x: int) -> int:
        return x + 1

    tool = ai.dynamic_tool('my_tool', my_tool, description='increment x')

    assert isinstance(tool, Action)
    assert tool.kind == ActionKind.TOOL
    assert tool.name == 'my_tool'
    assert tool.description == 'increment x'
    assert tool.metadata.get('type') == 'tool'
    assert tool.metadata.get('dynamic') is True

    # Execution
    resp = await tool.arun(5)
    assert resp.response == 6


@pytest.mark.asyncio
async def test_genkit_check_operation() -> None:
    """Test Genkit.check_operation method."""
    ai = Genkit()

    op = Operation(id='123', done=False, action='/background-model/test_action')

    # Create mock background action with check method
    mock_background_action = MagicMock()
    mock_background_action.check = AsyncMock(return_value=Operation(id='123', done=True, output='result'))

    # Patch lookup_background_action to return our mock
    with mock.patch(
        'genkit.blocks.background_model.lookup_background_action',
        new=AsyncMock(return_value=mock_background_action),
    ) as mock_lookup:
        updated_op = await ai.check_operation(op)

        assert updated_op.done is True
        assert updated_op.output == 'result'
        mock_lookup.assert_called_once()


@pytest.mark.asyncio
async def test_genkit_check_operation_no_action() -> None:
    """Test Genkit.check_operation method with no action."""
    ai = Genkit()
    op = Operation(id='123', done=False)  # action is None

    with pytest.raises(ValueError, match='Provided operation is missing original request information'):
        await ai.check_operation(op)


@pytest.mark.asyncio
async def test_genkit_check_operation_not_found() -> None:
    """Test Genkit.check_operation method with action not found."""
    ai = Genkit()
    op = Operation(id='123', done=False, action='missing')
    ai.registry.resolve_action_by_key = AsyncMock(return_value=None)  # type: ignore[assignment]

    with pytest.raises(ValueError, match='Failed to resolve background action from original request: missing'):
        await ai.check_operation(op)


@pytest.mark.asyncio
async def test_define_simple_retriever_legacy() -> None:
    """Test define_simple_retriever with legacy handler signature."""
    ai = Genkit()

    def simple_retriever(query: DocumentData, options: Any) -> list[DocumentData]:  # noqa: ANN401
        # Returns list[DocumentData] directly

        text_part: DocumentPart = DocumentPart(root=TextPart(text='doc1'))
        return [DocumentData(content=[text_part])]

    retriever_action = ai.define_simple_retriever('simple', simple_retriever)

    assert retriever_action.kind == ActionKind.RETRIEVER

    # Test execution
    req = RetrieverRequest(query=DocumentData(content=[]))
    resp_wrapper = await retriever_action.arun(req)
    response = resp_wrapper.response

    assert isinstance(response, RetrieverResponse)
    assert len(response.documents) == 1
    assert response.documents[0].content[0].root.text == 'doc1'


@pytest.mark.asyncio
async def test_define_simple_retriever_mapped() -> None:
    """Test define_simple_retriever with mapping options."""
    ai = Genkit()

    def db_handler(query: DocumentData, options: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        return [
            {'id': '1', 'text': 'hello', 'extra': 'data'},
            {'id': '2', 'text': 'world', 'extra': 'more'},
        ]

    options = SimpleRetrieverOptions(name='mapped', content='text', metadata=['extra'])

    retriever_action = ai.define_simple_retriever(options, db_handler)

    req = RetrieverRequest(query=DocumentData(content=[]))
    resp_wrapper = await retriever_action.arun(req)
    response = resp_wrapper.response

    assert len(response.documents) == 2
    assert response.documents[0].content[0].root.text == 'hello'
    assert response.documents[0].metadata == {'extra': 'data'}
    assert 'id' not in response.documents[0].metadata


@pytest.mark.asyncio
async def test_current_context() -> None:
    """Test Genkit.current_context method."""
    # current_context is a static method
    assert Genkit.current_context() is None

    context: dict[str, object] = {'auth': {'uid': '123'}}

    # Simulate being inside an action run using ActionRunContext internal mechanism
    token = _action_context.set(context)
    try:
        assert Genkit.current_context() == context
    finally:
        _action_context.reset(token)

    assert Genkit.current_context() is None


@pytest.mark.asyncio
async def test_flush_tracing() -> None:
    """Test Genkit.flush_tracing method."""
    ai = Genkit()

    mock_provider = MagicMock(spec=TracerProvider)
    mock_provider.force_flush = MagicMock()

    with mock.patch.object(trace_api, 'get_tracer_provider', return_value=mock_provider):
        await ai.flush_tracing()
        mock_provider.force_flush.assert_called_once()
