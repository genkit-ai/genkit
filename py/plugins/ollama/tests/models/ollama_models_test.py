# Copyright 2025 Google LLC
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

"""Unit tests for Ollama models package."""

import unittest
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
import ollama as ollama_api
import pytest

from genkit import (
    ActionRunContext,
    Media,
    MediaPart,
    Message,
    ModelConfig,
    ModelRequest,
    ModelResponseChunk,
    ModelUsage,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolRequestPart,
)
from genkit.plugins.ollama.constants import OllamaAPITypes
from genkit.plugins.ollama.models import ModelDefinition, OllamaConfig, OllamaModel, _convert_parameters


class TestOllamaModelGenerate(unittest.IsolatedAsyncioTestCase):
    """Tests for Generate method of OllamaModel."""

    async def asyncSetUp(self) -> None:
        """Common setup for all async tests."""
        self.mock_client = MagicMock()
        self.request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])])
        self.ctx = ActionRunContext()
        cast(Any, self.ctx).send_chunk = MagicMock()

    @patch(
        'genkit.model.get_basic_usage_stats',
        return_value=ModelUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        ),
    )
    async def test_generate_chat_non_streaming(self, mock_get_basic_usage_stats: MagicMock) -> None:
        """Test generate method with CHAT API type in non-streaming mode."""
        model_def = ModelDefinition(
            name='chat-model',
            api_type=OllamaAPITypes.CHAT,
        )
        ollama_model = OllamaModel(
            client=self.mock_client,
            model_definition=model_def,
        )

        # Mock internal methods
        mock_chat_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='Generated chat text',
            ),
        )
        cast(Any, ollama_model)._chat_with_ollama = AsyncMock(
            return_value=mock_chat_response,
        )
        cast(Any, ollama_model)._generate_ollama_response = AsyncMock()
        cast(Any, ollama_model)._build_multimodal_chat_response = MagicMock(
            return_value=[Part(root=TextPart(text='Parsed chat content'))],
        )
        cast(Any, ollama_model).get_usage_info = MagicMock(
            return_value=ModelUsage(
                input_tokens=5,
                output_tokens=10,
                total_tokens=15,
            ),
        )
        cast(Any, ollama_model).is_streaming_request = MagicMock(return_value=False)

        response = await ollama_model.generate(self.request, self.ctx)

        # Assertions
        cast(AsyncMock, ollama_model._chat_with_ollama).assert_awaited_once_with(request=self.request, ctx=self.ctx)
        cast(AsyncMock, ollama_model._generate_ollama_response).assert_not_awaited()
        cast(MagicMock, self.ctx.send_chunk).assert_not_called()
        cast(MagicMock, ollama_model._build_multimodal_chat_response).assert_called_once_with(
            chat_response=mock_chat_response, thinking_enabled=False
        )
        cast(MagicMock, ollama_model.is_streaming_request).assert_called_with(ctx=self.ctx)
        cast(MagicMock, ollama_model.get_usage_info).assert_called_once()

        self.assertIsNotNone(response.message)
        self.assertEqual(cast(Message, response.message).role, Role.MODEL)
        self.assertEqual(len(cast(Message, response.message).content), 1)
        self.assertEqual(cast(Message, response.message).content[0].root.text, 'Parsed chat content')
        self.assertIsNotNone(response.usage)
        self.assertEqual(cast(ModelUsage, response.usage).input_tokens, 5)
        self.assertEqual(cast(ModelUsage, response.usage).output_tokens, 10)

    @patch(
        'genkit.model.get_basic_usage_stats',
        return_value=ModelUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        ),
    )
    async def test_generate_generate_non_streaming(self, mock_get_basic_usage_stats: MagicMock) -> None:
        """Test generate method with GENERATE API type in non-streaming mode."""
        model_def = ModelDefinition(
            name='generate-model',
            api_type=OllamaAPITypes.GENERATE,
        )
        ollama_model = OllamaModel(
            client=self.mock_client,
            model_definition=model_def,
        )

        # Mock internal methods
        mock_generate_response = ollama_api.GenerateResponse(
            response='Generated text',
        )
        cast(Any, ollama_model)._generate_ollama_response = AsyncMock(
            return_value=mock_generate_response,
        )
        cast(Any, ollama_model)._chat_with_ollama = AsyncMock()
        cast(Any, ollama_model).is_streaming_request = MagicMock(return_value=False)
        cast(Any, ollama_model).get_usage_info = MagicMock(
            return_value=ModelUsage(
                input_tokens=7,
                output_tokens=14,
                total_tokens=21,
            ),
        )

        response = await ollama_model.generate(self.request, self.ctx)

        # Assertions
        cast(AsyncMock, ollama_model._generate_ollama_response).assert_awaited_once_with(
            request=self.request, ctx=self.ctx
        )
        cast(AsyncMock, ollama_model._chat_with_ollama).assert_not_called()
        cast(MagicMock, ollama_model.is_streaming_request).assert_called_with(ctx=self.ctx)
        cast(MagicMock, ollama_model.get_usage_info).assert_called_once()

        self.assertIsNotNone(response.message)
        self.assertIsNotNone(response.message)
        self.assertEqual(cast(Message, response.message).role, Role.MODEL)
        self.assertEqual(len(cast(Message, response.message).content), 1)
        self.assertEqual(cast(Message, response.message).content[0].root.text, 'Generated text')
        self.assertIsNotNone(response.usage)
        self.assertEqual(cast(ModelUsage, response.usage).input_tokens, 7)
        self.assertEqual(cast(ModelUsage, response.usage).output_tokens, 14)

    @patch(
        'genkit.model.get_basic_usage_stats',
        return_value=ModelUsage(),
    )
    async def test_generate_chat_streaming(self, mock_get_basic_usage_stats: MagicMock) -> None:
        """Test generate method with CHAT API type in streaming mode."""
        model_def = ModelDefinition(name='chat-model', api_type=OllamaAPITypes.CHAT)
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)
        streaming_ctx = ActionRunContext(streaming_callback=MagicMock())

        # Mock internal methods
        mock_chat_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='Generated chat text',
            ),
        )
        cast(Any, ollama_model)._chat_with_ollama = AsyncMock(
            return_value=mock_chat_response,
        )
        cast(Any, ollama_model)._build_multimodal_chat_response = MagicMock(
            return_value=[Part(root=TextPart(text='Parsed chat content'))],
        )
        cast(Any, ollama_model).is_streaming_request = MagicMock(return_value=True)
        cast(Any, ollama_model).get_usage_info = MagicMock(
            return_value=ModelUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
        )

        response = await ollama_model.generate(self.request, streaming_ctx)

        # Assertions for streaming behavior
        cast(AsyncMock, ollama_model._chat_with_ollama).assert_awaited_once_with(
            request=self.request,
            ctx=streaming_ctx,
        )
        cast(MagicMock, ollama_model.is_streaming_request).assert_called_with(
            ctx=streaming_ctx,
        )
        self.assertIsNotNone(response.message)
        self.assertEqual(cast(Message, response.message).content, [])

    @patch(
        'genkit.model.get_basic_usage_stats',
        return_value=ModelUsage(),
    )
    async def test_generate_generate_streaming(self, mock_get_basic_usage_stats: MagicMock) -> None:
        """Test generate method with GENERATE API type in streaming mode."""
        model_def = ModelDefinition(
            name='generate-model',
            api_type=OllamaAPITypes.GENERATE,
        )
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)
        streaming_ctx = ActionRunContext(streaming_callback=MagicMock())

        # Mock internal methods
        mock_generate_response = ollama_api.GenerateResponse(
            response='Generated text',
        )
        cast(Any, ollama_model)._generate_ollama_response = AsyncMock(
            return_value=mock_generate_response,
        )
        cast(Any, ollama_model).is_streaming_request = MagicMock(return_value=True)
        cast(Any, ollama_model).get_usage_info = MagicMock(
            return_value=ModelUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
        )

        response = await ollama_model.generate(self.request, streaming_ctx)

        # Assertions for streaming behavior
        cast(AsyncMock, ollama_model._generate_ollama_response).assert_awaited_once_with(
            request=self.request,
            ctx=streaming_ctx,
        )
        cast(MagicMock, ollama_model.is_streaming_request).assert_called_with(
            ctx=streaming_ctx,
        )
        self.assertIsNotNone(response.message)
        self.assertEqual(cast(Message, response.message).content, [])

    @patch(
        'genkit.model.get_basic_usage_stats',
        return_value=ModelUsage(),
    )
    async def test_generate_chat_api_response_none(self, mock_get_basic_usage_stats: MagicMock) -> None:
        """Test generate method when _chat_with_ollama returns None."""
        model_def = ModelDefinition(name='chat-model', api_type=OllamaAPITypes.CHAT)
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)

        cast(Any, ollama_model)._chat_with_ollama = AsyncMock(return_value=None)
        cast(Any, ollama_model)._build_multimodal_chat_response = MagicMock()
        cast(Any, ollama_model).is_streaming_request = MagicMock(return_value=False)
        cast(Any, ollama_model).get_usage_info = MagicMock(return_value=ModelUsage())

        response = await ollama_model.generate(self.request, self.ctx)

        cast(AsyncMock, ollama_model._chat_with_ollama).assert_awaited_once()
        cast(MagicMock, ollama_model._build_multimodal_chat_response).assert_not_called()
        self.assertIsNotNone(response.message)
        self.assertEqual(cast(Message, response.message).content[0].root.text, 'Failed to get response from Ollama API')
        self.assertIsNotNone(response.usage)
        self.assertEqual(cast(ModelUsage, response.usage).input_tokens, None)
        self.assertEqual(cast(ModelUsage, response.usage).output_tokens, None)

    @patch(
        'genkit.model.get_basic_usage_stats',
        return_value=ModelUsage(),
    )
    async def test_generate_generate_api_response_none(self, mock_get_basic_usage_stats: MagicMock) -> None:
        """Test generate method when _generate_ollama_response returns None."""
        model_def = ModelDefinition(name='generate-model', api_type=OllamaAPITypes.GENERATE)
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)

        cast(Any, ollama_model)._generate_ollama_response = AsyncMock(return_value=None)
        cast(Any, ollama_model).is_streaming_request = MagicMock(return_value=False)
        cast(Any, ollama_model).get_usage_info = MagicMock(return_value=ModelUsage())

        response = await ollama_model.generate(self.request, self.ctx)

        cast(AsyncMock, ollama_model._generate_ollama_response).assert_awaited_once()
        self.assertIsNotNone(response.message)
        self.assertEqual(cast(Message, response.message).content[0].root.text, 'Failed to get response from Ollama API')
        self.assertIsNotNone(response.usage)
        self.assertEqual(cast(ModelUsage, response.usage).input_tokens, None)
        self.assertEqual(cast(ModelUsage, response.usage).output_tokens, None)


class TestOllamaModelChatWithOllama(unittest.IsolatedAsyncioTestCase):
    """Unit tests for OllamaModel._chat_with_ollama method."""

    async def asyncSetUp(self) -> None:
        """Common setup."""
        self.mock_ollama_client_instance = AsyncMock()
        self.mock_ollama_client_factory = MagicMock(return_value=self.mock_ollama_client_instance)
        self.model_definition = ModelDefinition(name='test-chat-model', api_type=OllamaAPITypes.CHAT)
        self.ollama_model = OllamaModel(client=self.mock_ollama_client_factory, model_definition=self.model_definition)
        self.request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])])
        self.ctx = ActionRunContext()
        cast(Any, self.ctx).send_chunk = MagicMock()

        # Properly mock methods of ollama_model using patch.object
        self.patcher_build_chat_messages = patch.object(
            self.ollama_model, 'build_chat_messages', new_callable=AsyncMock, return_value=[{}]
        )
        self.patcher_is_streaming_request = patch.object(self.ollama_model, 'is_streaming_request', return_value=False)
        self.patcher_build_request_options = patch.object(
            self.ollama_model, 'build_request_options', return_value={'temperature': 0.7}
        )
        self.patcher_build_multimodal_response = patch.object(
            self.ollama_model,
            '_build_multimodal_chat_response',
            return_value=[Part(root=TextPart(text='mocked content'))],
        )

        self.mock_build_chat_messages = self.patcher_build_chat_messages.start()
        self.mock_is_streaming_request = self.patcher_is_streaming_request.start()
        self.mock_build_request_options = self.patcher_build_request_options.start()
        self.mock_build_multimodal_response = self.patcher_build_multimodal_response.start()

        self.mock_convert_parameters = MagicMock(return_value={'type': 'string'})

    async def asyncTearDown(self) -> None:
        """Cleanup patches."""
        self.patcher_build_chat_messages.stop()
        self.patcher_is_streaming_request.stop()
        self.patcher_build_request_options.stop()
        self.patcher_build_multimodal_response.stop()

    async def test_non_streaming_chat_success(self) -> None:
        """Test _chat_with_ollama in non-streaming mode with successful response."""
        expected_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='Ollama non-stream response',
            ),
        )
        self.mock_ollama_client_instance.chat.return_value = expected_response

        response = await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        self.assertIsNotNone(response)
        self.assertEqual(cast(ollama_api.ChatResponse, response).message.content, 'Ollama non-stream response')
        self.mock_build_chat_messages.assert_called_once_with(self.request)
        self.mock_is_streaming_request.assert_called_once_with(ctx=self.ctx)
        cast(MagicMock, self.ctx.send_chunk).assert_not_called()
        self.mock_ollama_client_instance.chat.assert_awaited_once_with(
            model=self.model_definition.name,
            messages=self.mock_build_chat_messages.return_value,
            tools=[],
            options=self.mock_build_request_options.return_value,
            format='',
            stream=False,
        )

        self.mock_build_multimodal_response.assert_not_called()

    async def test_think_and_keep_alive_forwarded_as_top_level_kwargs(self) -> None:
        """think/keep_alive reach the chat call as top-level kwargs, not sampler options.

        Guards the JS/Go parity wiring: build_request_kwargs feeds ``**extra_kwargs``
        into client.chat(), so dropping that spread would silently regress reasoning
        and model keep-alive.
        """
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])],
            config={'think': 'low', 'keepAlive': '5m'},
        )
        self.mock_ollama_client_instance.chat.return_value = ollama_api.ChatResponse(
            message=ollama_api.Message(role='', content='ok')
        )

        await self.ollama_model._chat_with_ollama(request, self.ctx)

        call_kwargs = self.mock_ollama_client_instance.chat.await_args.kwargs
        assert call_kwargs['think'] == 'low'
        assert call_kwargs['keep_alive'] == '5m'
        # Sampler options stay separate from the top-level kwargs.
        assert call_kwargs['options'] == self.mock_build_request_options.return_value

    async def test_streaming_chat_success(self) -> None:
        """Test _chat_with_ollama in streaming mode with multiple chunks."""
        self.mock_is_streaming_request.return_value = True
        # Create a streaming context with a callback
        self.ctx = ActionRunContext(streaming_callback=MagicMock())
        cast(Any, self.ctx).send_chunk = MagicMock()

        # Simulate an async iterator of chunks
        async def mock_streaming_chunks() -> AsyncIterator[ollama_api.ChatResponse]:
            yield ollama_api.ChatResponse(
                message=ollama_api.Message(
                    role='',
                    content='chunk1',
                ),
            )
            yield ollama_api.ChatResponse(
                message=ollama_api.Message(
                    role='',
                    content='chunk2',
                ),
            )

        self.mock_ollama_client_instance.chat.return_value = mock_streaming_chunks()

        response = await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        # For streaming requests, the method returns None because response chunks
        # are sent incrementally via ctx.send_chunk() rather than returned at the end.
        # This is the expected behavior for streaming APIs.
        self.assertIsNone(response)
        self.mock_build_chat_messages.assert_called_once_with(self.request)
        self.mock_is_streaming_request.assert_called_once_with(ctx=self.ctx)
        self.mock_ollama_client_instance.chat.assert_awaited_once_with(
            model=self.model_definition.name,
            messages=self.mock_build_chat_messages.return_value,
            tools=[],
            options=self.mock_build_request_options.return_value,
            format='',
            stream=True,
        )
        self.assertEqual(cast(MagicMock, self.ctx.send_chunk).call_count, 2)
        self.assertEqual(self.mock_build_multimodal_response.call_count, 2)
        cast(MagicMock, self.ctx.send_chunk).assert_any_call(chunk=ANY)
        self.mock_build_multimodal_response.assert_any_call(chat_response=ANY)

    async def test_streaming_chat_empty_role_maps_to_model(self) -> None:
        """Streamed chunks with an empty role should be labeled MODEL, not TOOL."""
        self.mock_is_streaming_request.return_value = True
        self.ctx = ActionRunContext(streaming_callback=MagicMock())
        cast(Any, self.ctx).send_chunk = MagicMock()

        async def mock_streaming_chunks() -> AsyncIterator[ollama_api.ChatResponse]:
            # Ollama commonly sends an empty role on streamed deltas.
            yield ollama_api.ChatResponse(message=ollama_api.Message(role='', content='delta'))

        self.mock_ollama_client_instance.chat.return_value = mock_streaming_chunks()

        await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        send_chunk = cast(MagicMock, self.ctx.send_chunk)
        send_chunk.assert_called_once()
        sent_chunk = send_chunk.call_args.kwargs['chunk']
        self.assertEqual(cast(ModelResponseChunk, sent_chunk).role, Role.MODEL)

    async def test_streaming_chat_tool_role_maps_to_tool(self) -> None:
        """Streamed chunks explicitly labeled as tools should remain TOOL."""
        self.mock_is_streaming_request.return_value = True
        self.ctx = ActionRunContext(streaming_callback=MagicMock())
        cast(Any, self.ctx).send_chunk = MagicMock()

        async def mock_streaming_chunks() -> AsyncIterator[ollama_api.ChatResponse]:
            yield ollama_api.ChatResponse(message=ollama_api.Message(role='tool', content='delta'))

        self.mock_ollama_client_instance.chat.return_value = mock_streaming_chunks()

        await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        send_chunk = cast(MagicMock, self.ctx.send_chunk)
        send_chunk.assert_called_once()
        sent_chunk = send_chunk.call_args.kwargs['chunk']
        self.assertEqual(cast(ModelResponseChunk, sent_chunk).role, Role.TOOL)

    async def test_chat_with_output_format_string(self) -> None:
        """Test _chat_with_ollama with request.output.format string."""
        self.request.output_format = 'json'

        expected_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='json output',
            ),
        )
        self.mock_ollama_client_instance.chat.return_value = expected_response

        await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        _call_args, call_kwargs = self.mock_ollama_client_instance.chat.call_args
        self.assertIn('format', call_kwargs)
        self.assertEqual(call_kwargs['format'], 'json')

    async def test_chat_with_output_format_schema(self) -> None:
        """Test _chat_with_ollama with request.output.schema dictionary."""
        schema_dict = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        self.request.output_schema = schema_dict

        expected_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='schema output',
            ),
        )
        self.mock_ollama_client_instance.chat.return_value = expected_response

        await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        _call_args, call_kwargs = self.mock_ollama_client_instance.chat.call_args
        self.assertIn('format', call_kwargs)
        self.assertEqual(call_kwargs['format'], schema_dict)

    async def test_chat_with_no_output_format(self) -> None:
        """Test _chat_with_ollama with no output format specified."""
        self.request.output_format = None
        self.request.output_schema = None

        expected_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='normal output',
            ),
        )
        self.mock_ollama_client_instance.chat.return_value = expected_response

        await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        _call_args, call_kwargs = self.mock_ollama_client_instance.chat.call_args
        self.assertIn('format', call_kwargs)
        self.assertEqual(call_kwargs['format'], '')

    async def test_chat_api_raises_exception(self) -> None:
        """Test _chat_with_ollama handles exception from client.chat."""
        self.mock_ollama_client_instance.chat.side_effect = Exception('Ollama API Error')

        with self.assertRaisesRegex(Exception, 'Ollama API Error'):
            await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        self.mock_ollama_client_instance.chat.assert_awaited_once()
        cast(MagicMock, self.ctx.send_chunk).assert_not_called()


class TestOllamaModelGenerateOllamaResponse(unittest.IsolatedAsyncioTestCase):
    """Unit tests for OllamaModel._generate_ollama_response."""

    async def asyncSetUp(self) -> None:
        """Common setup."""
        self.mock_ollama_client_instance = AsyncMock()
        self.mock_ollama_client_factory = MagicMock(return_value=self.mock_ollama_client_instance)

        self.model_definition = ModelDefinition(name='test-generate-model', api_type=OllamaAPITypes.GENERATE)
        self.ollama_model = OllamaModel(client=self.mock_ollama_client_factory, model_definition=self.model_definition)
        self.request = ModelRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='Test generate message'))],
                )
            ],
            config={'temperature': 0.8},
        )
        self.ctx = ActionRunContext()
        cast(Any, self.ctx).send_chunk = MagicMock()

        # Properly mock methods of ollama_model using patch.object
        self.patcher_build_prompt = patch.object(
            self.ollama_model, 'build_prompt', return_value='Mocked prompt from build_prompt'
        )
        self.patcher_is_streaming_request = patch.object(self.ollama_model, 'is_streaming_request', return_value=False)
        self.patcher_build_request_options = patch.object(
            self.ollama_model, 'build_request_options', return_value={'temperature': 0.8}
        )

        self.mock_build_prompt = self.patcher_build_prompt.start()
        self.mock_is_streaming_request = self.patcher_is_streaming_request.start()
        self.mock_build_request_options = self.patcher_build_request_options.start()

    async def asyncTearDown(self) -> None:
        """Cleanup patches."""
        self.patcher_build_prompt.stop()
        self.patcher_is_streaming_request.stop()
        self.patcher_build_request_options.stop()

    async def test_think_and_keep_alive_forwarded_as_top_level_kwargs(self) -> None:
        """think/keep_alive reach the generate call as top-level kwargs, matching chat."""
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])],
            config={'think': True, 'keepAlive': '10m'},
        )
        self.mock_ollama_client_instance.generate.return_value = ollama_api.GenerateResponse(response='ok')

        await self.ollama_model._generate_ollama_response(request, self.ctx)

        call_kwargs = self.mock_ollama_client_instance.generate.await_args.kwargs
        assert call_kwargs['think'] is True
        assert call_kwargs['keep_alive'] == '10m'
        assert call_kwargs['options'] == self.mock_build_request_options.return_value

    async def test_non_streaming_generate_success(self) -> None:
        """Test _generate_ollama_response in non-streaming mode with successful response."""
        expected_response = ollama_api.GenerateResponse(response='Full generated text')
        self.mock_ollama_client_instance.generate.return_value = expected_response

        response = await self.ollama_model._generate_ollama_response(self.request, self.ctx)

        self.assertIsNotNone(response)
        self.assertEqual(cast(ollama_api.GenerateResponse, response).response, 'Full generated text')

        self.mock_build_prompt.assert_called_once_with(self.request)
        self.mock_is_streaming_request.assert_called_once_with(ctx=self.ctx)
        self.mock_build_request_options.assert_called_once_with(config=self.request.config)
        self.mock_ollama_client_instance.generate.assert_awaited_once_with(
            model=self.model_definition.name,
            prompt=self.mock_build_prompt.return_value,
            options=self.mock_build_request_options.return_value,
            stream=False,
        )
        cast(MagicMock, self.ctx.send_chunk).assert_not_called()

    async def test_streaming_generate_success(self) -> None:
        """Test _generate_ollama_response in streaming mode with multiple chunks."""
        self.mock_is_streaming_request.return_value = True

        # Simulate an async iterator of chunks
        async def mock_streaming_chunks() -> AsyncIterator[ollama_api.GenerateResponse]:
            yield ollama_api.GenerateResponse(response='chunk1 ')
            yield ollama_api.GenerateResponse(response='chunk2')

        self.mock_ollama_client_instance.generate.return_value = mock_streaming_chunks()

        response = await self.ollama_model._generate_ollama_response(self.request, self.ctx)

        # For streaming requests, the method returns None because response chunks
        # are sent incrementally via ctx.send_chunk() rather than returned at the end.
        # This is the expected behavior for streaming APIs.
        self.assertIsNone(response)

        self.mock_build_prompt.assert_called_once_with(self.request)
        self.mock_is_streaming_request.assert_called_once_with(ctx=self.ctx)
        self.mock_ollama_client_instance.generate.assert_awaited_once_with(
            model=self.model_definition.name,
            prompt=self.mock_build_prompt.return_value,
            options=self.mock_build_request_options.return_value,
            stream=True,
        )
        self.assertEqual(cast(MagicMock, self.ctx.send_chunk).call_count, 2)
        cast(MagicMock, self.ctx.send_chunk).assert_any_call(
            chunk=ModelResponseChunk(role=Role.MODEL, index=1, content=[Part(root=TextPart(text='chunk1 '))])
        )
        cast(MagicMock, self.ctx.send_chunk).assert_any_call(
            chunk=ModelResponseChunk(role=Role.MODEL, index=2, content=[Part(root=TextPart(text='chunk2'))])
        )

    async def test_generate_api_raises_exception(self) -> None:
        """Test _generate_ollama_response handles exception from client.generate."""
        self.mock_ollama_client_instance.generate.side_effect = Exception('Ollama generate API Error')

        with self.assertRaisesRegex(Exception, 'Ollama generate API Error'):
            await self.ollama_model._generate_ollama_response(self.request, self.ctx)

        self.mock_ollama_client_instance.generate.assert_awaited_once()
        cast(MagicMock, self.ctx.send_chunk).assert_not_called()


def test_convert_parameters_empty_schema_returns_none() -> None:
    """An empty schema produces no parameters (the no-tool-params case)."""
    assert _convert_parameters({}) is None


def test_convert_parameters_non_object_raises() -> None:
    """JS parity: Ollama only supports object-typed tool inputs."""
    with pytest.raises(ValueError):
        _convert_parameters({'type': 'string'})


def test_convert_parameters_object_with_properties() -> None:
    """An object schema maps properties and required without pinning the whole Parameters object."""
    result = _convert_parameters({
        'type': 'object',
        'properties': {
            'name': {'type': 'string', 'description': 'User name'},
            'age': {'type': 'integer', 'description': 'User age'},
        },
        'required': ['name'],
    })

    assert result is not None
    assert result.type == 'object'
    assert result.required == ['name']
    assert result.properties is not None
    assert result.properties['name'].type == 'string'
    assert result.properties['name'].description == 'User name'
    assert result.properties['age'].type == 'integer'


def test_convert_parameters_object_without_properties() -> None:
    """An object schema with no properties yields an empty properties dict."""
    result = _convert_parameters({'type': 'object'})

    assert result is not None
    assert result.type == 'object'
    assert result.required is None
    assert result.properties == {}


def test_convert_parameters_infers_object_from_properties() -> None:
    """A schema with properties but no explicit type is treated as an object."""
    result = _convert_parameters({'properties': {'name': {'type': 'string'}}})

    assert result is not None
    assert result.type == 'object'
    assert result.required is None
    assert result.properties is not None
    assert result.properties['name'].type == 'string'
    assert result.properties['name'].description == ''


def test_convert_parameters_maps_anyof_to_type_list() -> None:
    """Optional fields serialize as anyOf with no top-level type; map to the list form."""
    result = _convert_parameters({
        'type': 'object',
        'properties': {
            'units': {
                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                'description': 'Temperature units',
            },
        },
    })

    assert result is not None
    assert result.properties is not None
    assert result.properties['units'].type == ['string', 'null']
    assert result.properties['units'].description == 'Temperature units'


def test_convert_parameters_keeps_optional_required_property() -> None:
    """A property without a top-level type is kept, so required stays consistent."""
    result = _convert_parameters({
        'type': 'object',
        'properties': {
            'city': {'type': 'string'},
            'units': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
        },
        'required': ['city', 'units'],
    })

    assert result is not None
    assert result.required == ['city', 'units']
    assert result.properties is not None
    # Every required name still resolves to a real property (no dangling reference).
    assert set(result.required).issubset(result.properties.keys())


def test_convert_parameters_untyped_property_falls_back_to_none() -> None:
    """A typeless schema (e.g. an `Any` field) stays present with no type rather than crashing."""
    result = _convert_parameters({'type': 'object', 'properties': {'payload': {}}})

    assert result is not None
    assert result.properties is not None
    assert result.properties['payload'].type is None


class TestBuildRequestOptions:
    """Tests for OllamaModel.build_request_options.

    The method returns a plain mapping (not ``ollama_api.Options``) so that
    sampler knobs the installed ``Options`` model doesn't field — e.g.
    ``min_p`` on ollama 0.6.1 — still reach the server.
    """

    def test_none_returns_empty_mapping(self) -> None:
        """None config produces an empty options mapping."""
        assert OllamaModel.build_request_options(None) == {}

    def test_options_input_is_normalised_to_dict(self) -> None:
        """A raw Options input is returned as a plain mapping."""
        options = OllamaModel.build_request_options(ollama_api.Options(num_ctx=512))
        assert options == {'num_ctx': 512}

    def test_model_config_top_p(self) -> None:
        """ModelConfig.top_p maps to the top_p server field."""
        options = OllamaModel.build_request_options(ModelConfig(top_p=0.9))
        assert options['top_p'] == 0.9

    def test_model_config_max_output_tokens_maps_to_int_num_predict(self) -> None:
        """max_output_tokens (float in genkit) maps to an int num_predict."""
        options = OllamaModel.build_request_options(ModelConfig(max_output_tokens=128))
        assert options['num_predict'] == 128
        assert isinstance(options['num_predict'], int)

    def test_raw_dict_camel_case_top_p(self) -> None:
        """A camelCase ``topP`` knob is snake-cased onto top_p."""
        options = OllamaModel.build_request_options({'topP': 0.9})
        assert options['top_p'] == 0.9

    def test_ollama_config_instance_forwards_knobs(self) -> None:
        """An OllamaConfig instance forwards Ollama-only sampler knobs."""
        options = OllamaModel.build_request_options(OllamaConfig(num_ctx=4096, seed=42, temperature=0.5))
        assert options['num_ctx'] == 4096
        assert options['seed'] == 42
        assert options['temperature'] == 0.5

    def test_min_p_is_preserved(self) -> None:
        """min_p survives even though ollama 0.6.1's Options model drops it."""
        options = OllamaModel.build_request_options(OllamaConfig(min_p=0.05))
        assert options['min_p'] == 0.05

    def test_ollama_config_extras_snake_cased(self) -> None:
        """Unknown OllamaConfig knobs are forwarded snake-cased (instance + camel)."""
        snake = OllamaModel.build_request_options(OllamaConfig(repeat_penalty=1.1))
        assert snake['repeat_penalty'] == 1.1
        camel = OllamaModel.build_request_options(OllamaConfig(repeatPenalty=1.2))
        assert camel['repeat_penalty'] == 1.2

    def test_num_predict_wins_over_max_output_tokens(self) -> None:
        """An explicit num_predict beats the inherited max_output_tokens."""
        options = OllamaModel.build_request_options(OllamaConfig(num_predict=10, max_output_tokens=99))
        assert options['num_predict'] == 10

    def test_think_and_keep_alive_excluded_from_options(self) -> None:
        """think/keep_alive are request kwargs, never sampler options."""
        options = OllamaModel.build_request_options(OllamaConfig(think=True, keep_alive='5m', num_ctx=2048))
        assert 'think' not in options
        assert 'keep_alive' not in options
        assert options['num_ctx'] == 2048

    def test_dumped_dict_path_matches_instance(self) -> None:
        """A dumped OllamaConfig (dict) yields the same options as the instance."""
        cfg = OllamaConfig(think=True, keep_alive='5m', num_ctx=4096, temperature=0.5, max_output_tokens=100)
        dumped = cfg.model_dump(exclude_none=True, mode='json')

        from_dict = OllamaModel.build_request_options(dumped)
        from_instance = OllamaModel.build_request_options(cfg)

        assert from_dict == from_instance
        assert from_dict == {'num_ctx': 4096, 'num_predict': 100, 'temperature': 0.5}

    def test_stop_sequences_map_to_stop_list(self) -> None:
        """stop_sequences maps to Ollama's stop field, preserved as a list.

        Go parity: stop is sent as a list, unlike the JS plugin which joins it.
        """
        options = OllamaModel.build_request_options(ModelConfig(stop_sequences=['STOP', '###']))
        assert options['stop'] == ['STOP', '###']
        assert 'stop_sequences' not in options


class TestBuildRequestKwargs:
    """Tests for OllamaModel.build_request_kwargs."""

    def test_instance_surfaces_think_and_keep_alive(self) -> None:
        """An OllamaConfig instance surfaces think/keep_alive as top-level kwargs."""
        kwargs = OllamaModel.build_request_kwargs(OllamaConfig(think=True, keep_alive='5m'))
        assert kwargs == {'think': True, 'keep_alive': '5m'}

    def test_dumped_dict_surfaces_think_and_keep_alive(self) -> None:
        """A dumped OllamaConfig (camelCased keys) still surfaces think/keep_alive."""
        dumped = OllamaConfig(think='low', keep_alive='10m').model_dump(exclude_none=True, mode='json')
        # The dumped dict carries the camelCase alias for keep_alive.
        assert 'keepAlive' in dumped
        kwargs = OllamaModel.build_request_kwargs(dumped)
        assert kwargs == {'think': 'low', 'keep_alive': '10m'}

    def test_plain_model_config_extras_surface_think_and_keep_alive(self) -> None:
        """A plain ModelConfig carrying think/keep_alive as extras surfaces them.

        ModelConfig has ``extra='allow'``, so the knobs can ride on a base
        ModelConfig instance (including camelCased), not just OllamaConfig.
        """
        config = ModelConfig.model_validate({'think': True, 'keepAlive': '5m'})
        kwargs = OllamaModel.build_request_kwargs(config)
        assert kwargs == {'think': True, 'keep_alive': '5m'}

    def test_absent_knobs_yield_empty_kwargs(self) -> None:
        """Configs without think/keep_alive produce no extra kwargs."""
        assert OllamaModel.build_request_kwargs(OllamaConfig(num_ctx=2048)) == {}
        assert OllamaModel.build_request_kwargs(None) == {}
        assert OllamaModel.build_request_kwargs(ModelConfig(top_p=0.9)) == {}


class TestFromOllamaRole:
    """Tests for OllamaModel._from_ollama_role."""

    def test_known_roles(self) -> None:
        """Each known Ollama role maps to its Genkit counterpart."""
        assert OllamaModel._from_ollama_role('assistant') == Role.MODEL
        assert OllamaModel._from_ollama_role('tool') == Role.TOOL
        assert OllamaModel._from_ollama_role('user') == Role.USER
        assert OllamaModel._from_ollama_role('system') == Role.SYSTEM

    def test_empty_role_defaults_to_model(self) -> None:
        """An empty/None role (common on streamed deltas) defaults to MODEL."""
        assert OllamaModel._from_ollama_role('') == Role.MODEL
        assert OllamaModel._from_ollama_role(None) == Role.MODEL

    def test_unknown_role_warns_and_defaults_to_model(self) -> None:
        """An unrecognized role warns and falls back to MODEL."""
        with patch('genkit.plugins.ollama.models.logger') as mock_logger:
            assert OllamaModel._from_ollama_role('wizard') == Role.MODEL
            cast(MagicMock, mock_logger.warning).assert_called_once()


class TestReasoning:
    """Tests for surfacing message.thinking as a leading ReasoningPart."""

    def test_thinking_yields_leading_reasoning_part(self) -> None:
        """A message with ``thinking`` prepends a ReasoningPart before text."""
        response = ollama_api.ChatResponse(
            message=ollama_api.Message(role='assistant', content='The answer is 4.', thinking='2+2 is 4'),
        )
        content = OllamaModel._build_multimodal_chat_response(chat_response=response)

        assert isinstance(content[0].root, ReasoningPart)
        assert content[0].root.reasoning == '2+2 is 4'
        assert isinstance(content[1].root, TextPart)
        assert content[1].root.text == 'The answer is 4.'

    def test_no_thinking_has_no_reasoning_part(self) -> None:
        """Without ``thinking`` no ReasoningPart is emitted."""
        response = ollama_api.ChatResponse(
            message=ollama_api.Message(role='assistant', content='Hi'),
        )
        content = OllamaModel._build_multimodal_chat_response(chat_response=response)

        assert all(not isinstance(part.root, ReasoningPart) for part in content)

    def test_think_tag_fallback_extracts_reasoning(self) -> None:
        """With thinking requested and no dedicated field, inline <think> tags are
        surfaced as reasoning and stripped from the text (Go parseThinking parity)."""
        response = ollama_api.ChatResponse(
            message=ollama_api.Message(role='assistant', content='<think>2+2 is 4</think>The answer is 4.'),
        )
        content = OllamaModel._build_multimodal_chat_response(chat_response=response, thinking_enabled=True)

        assert isinstance(content[0].root, ReasoningPart)
        assert content[0].root.reasoning == '2+2 is 4'
        assert isinstance(content[1].root, TextPart)
        assert content[1].root.text == 'The answer is 4.'

    def test_think_tag_not_parsed_when_thinking_disabled(self) -> None:
        """Without an explicit think request, <think> tags stay verbatim in the text."""
        response = ollama_api.ChatResponse(
            message=ollama_api.Message(role='assistant', content='<think>hidden</think>visible'),
        )
        content = OllamaModel._build_multimodal_chat_response(chat_response=response, thinking_enabled=False)

        assert all(not isinstance(part.root, ReasoningPart) for part in content)
        assert content[0].root.text == '<think>hidden</think>visible'

    def test_dedicated_thinking_field_wins_over_tags(self) -> None:
        """The dedicated thinking field takes precedence; content tags are left intact."""
        response = ollama_api.ChatResponse(
            message=ollama_api.Message(role='assistant', content='<think>inline</think>answer', thinking='structured'),
        )
        content = OllamaModel._build_multimodal_chat_response(chat_response=response, thinking_enabled=True)

        assert isinstance(content[0].root, ReasoningPart)
        assert content[0].root.reasoning == 'structured'
        # The content tags are not double-processed when the dedicated field exists.
        assert content[1].root.text == '<think>inline</think>answer'

    def test_multiple_think_blocks_joined_and_stripped(self) -> None:
        """Multiple <think>/<thinking> blocks are joined with blank lines and removed."""
        response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='assistant',
                content='<think>first</think>mid<thinking>second</thinking>end',
            ),
        )
        content = OllamaModel._build_multimodal_chat_response(chat_response=response, thinking_enabled=True)

        assert content[0].root.reasoning == 'first\n\nsecond'
        assert content[1].root.text == 'midend'

    def test_think_only_content_yields_no_text_part(self) -> None:
        """Content that is entirely a think block produces reasoning but no text."""
        response = ollama_api.ChatResponse(
            message=ollama_api.Message(role='assistant', content='<think>just reasoning</think>'),
        )
        content = OllamaModel._build_multimodal_chat_response(chat_response=response, thinking_enabled=True)

        assert len(content) == 1
        assert isinstance(content[0].root, ReasoningPart)
        assert content[0].root.reasoning == 'just reasoning'


class TestReasoningStreaming(unittest.IsolatedAsyncioTestCase):
    """Reasoning is also surfaced on streamed chunks (same builder path)."""

    async def test_streaming_chunk_yields_reasoning_part(self) -> None:
        """A streamed chunk carrying ``thinking`` emits a leading ReasoningPart."""
        client_instance = AsyncMock()
        factory = MagicMock(return_value=client_instance)
        model = OllamaModel(
            client=factory,
            model_definition=ModelDefinition(name='m', api_type=OllamaAPITypes.CHAT),
        )

        async def chunks() -> AsyncIterator[ollama_api.ChatResponse]:
            yield ollama_api.ChatResponse(
                message=ollama_api.Message(role='assistant', content='4', thinking='2+2'),
            )

        client_instance.chat.return_value = chunks()

        ctx = ActionRunContext(streaming_callback=MagicMock())
        sent: list[ModelResponseChunk] = []
        cast(Any, ctx).send_chunk = MagicMock(side_effect=lambda chunk: sent.append(chunk))

        request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])])
        with patch.object(model, 'build_chat_messages', new_callable=AsyncMock, return_value=[]):
            await model._chat_with_ollama(request=request, ctx=ctx)

        assert len(sent) == 1
        first_part = sent[0].content[0]
        assert isinstance(first_part.root, ReasoningPart)
        assert first_part.root.reasoning == '2+2'

    async def test_streaming_chunk_does_not_parse_think_tags(self) -> None:
        """Inline <think> tags in a streamed chunk are left untouched even when think
        is enabled — a tag may be split across chunks, so only the dedicated field is
        surfaced mid-stream. Matches the Go plugin's translateChatChunk."""
        client_instance = AsyncMock()
        factory = MagicMock(return_value=client_instance)
        model = OllamaModel(
            client=factory,
            model_definition=ModelDefinition(name='m', api_type=OllamaAPITypes.CHAT),
        )

        async def chunks() -> AsyncIterator[ollama_api.ChatResponse]:
            yield ollama_api.ChatResponse(
                message=ollama_api.Message(role='assistant', content='<think>partial'),
            )

        client_instance.chat.return_value = chunks()

        ctx = ActionRunContext(streaming_callback=MagicMock())
        sent: list[ModelResponseChunk] = []
        cast(Any, ctx).send_chunk = MagicMock(side_effect=lambda chunk: sent.append(chunk))

        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
            config=OllamaConfig(think=True),
        )
        with patch.object(model, 'build_chat_messages', new_callable=AsyncMock, return_value=[]):
            await model._chat_with_ollama(request=request, ctx=ctx)

        assert len(sent) == 1
        parts = sent[0].content
        assert all(not isinstance(part.root, ReasoningPart) for part in parts)
        assert parts[0].root.text == '<think>partial'


class TestReasoningGenerate:
    """Tests for surfacing GenerateResponse.thinking as a leading ReasoningPart."""

    def test_thinking_yields_leading_reasoning_part(self) -> None:
        """A generate response with ``thinking`` prepends a ReasoningPart before text."""
        response = ollama_api.GenerateResponse(response='The answer is 4.', thinking='2+2 is 4')
        content = OllamaModel._build_generate_response(generate_response=response)

        assert isinstance(content[0].root, ReasoningPart)
        assert content[0].root.reasoning == '2+2 is 4'
        assert isinstance(content[1].root, TextPart)
        assert content[1].root.text == 'The answer is 4.'

    def test_no_thinking_has_no_reasoning_part(self) -> None:
        """Without ``thinking`` no ReasoningPart is emitted."""
        response = ollama_api.GenerateResponse(response='Hi')
        content = OllamaModel._build_generate_response(generate_response=response)

        assert all(not isinstance(part.root, ReasoningPart) for part in content)

    def test_think_tag_fallback_extracts_reasoning(self) -> None:
        """With thinking requested and no dedicated field, inline <think> tags are
        surfaced as reasoning and stripped from the text (Go parseThinking parity)."""
        response = ollama_api.GenerateResponse(response='<think>2+2 is 4</think>The answer is 4.')
        content = OllamaModel._build_generate_response(generate_response=response, thinking_enabled=True)

        assert isinstance(content[0].root, ReasoningPart)
        assert content[0].root.reasoning == '2+2 is 4'
        assert isinstance(content[1].root, TextPart)
        assert content[1].root.text == 'The answer is 4.'

    def test_think_tag_not_parsed_when_thinking_disabled(self) -> None:
        """Without an explicit think request, <think> tags stay verbatim in the text."""
        response = ollama_api.GenerateResponse(response='<think>hidden</think>visible')
        content = OllamaModel._build_generate_response(generate_response=response, thinking_enabled=False)

        assert all(not isinstance(part.root, ReasoningPart) for part in content)
        assert content[0].root.text == '<think>hidden</think>visible'

    def test_dedicated_thinking_field_wins_over_tags(self) -> None:
        """The dedicated thinking field takes precedence; content tags are left intact."""
        response = ollama_api.GenerateResponse(response='<think>inline</think>answer', thinking='structured')
        content = OllamaModel._build_generate_response(generate_response=response, thinking_enabled=True)

        assert isinstance(content[0].root, ReasoningPart)
        assert content[0].root.reasoning == 'structured'
        assert content[1].root.text == '<think>inline</think>answer'


class TestReasoningGenerateStreaming(unittest.IsolatedAsyncioTestCase):
    """Reasoning is also surfaced on streamed generate chunks (same builder path)."""

    async def test_streaming_chunk_yields_reasoning_part(self) -> None:
        """A streamed generate chunk carrying ``thinking`` emits a leading ReasoningPart."""
        client_instance = AsyncMock()
        factory = MagicMock(return_value=client_instance)
        model = OllamaModel(
            client=factory,
            model_definition=ModelDefinition(name='m', api_type=OllamaAPITypes.GENERATE),
        )

        async def chunks() -> AsyncIterator[ollama_api.GenerateResponse]:
            yield ollama_api.GenerateResponse(response='4', thinking='2+2')

        client_instance.generate.return_value = chunks()

        ctx = ActionRunContext(streaming_callback=MagicMock())
        sent: list[ModelResponseChunk] = []
        cast(Any, ctx).send_chunk = MagicMock(side_effect=lambda chunk: sent.append(chunk))

        request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])])
        with patch.object(model, 'build_prompt', return_value='hi'):
            await model._generate_ollama_response(request=request, ctx=ctx)

        assert len(sent) == 1
        first_part = sent[0].content[0]
        assert isinstance(first_part.root, ReasoningPart)
        assert first_part.root.reasoning == '2+2'

    async def test_streaming_chunk_does_not_parse_think_tags(self) -> None:
        """Inline <think> tags in a streamed generate chunk are left untouched even when
        think is enabled — a tag may be split across chunks, so only the dedicated field
        is surfaced mid-stream. Matches the Go plugin's translateChatChunk."""
        client_instance = AsyncMock()
        factory = MagicMock(return_value=client_instance)
        model = OllamaModel(
            client=factory,
            model_definition=ModelDefinition(name='m', api_type=OllamaAPITypes.GENERATE),
        )

        async def chunks() -> AsyncIterator[ollama_api.GenerateResponse]:
            yield ollama_api.GenerateResponse(response='<think>partial')

        client_instance.generate.return_value = chunks()

        ctx = ActionRunContext(streaming_callback=MagicMock())
        sent: list[ModelResponseChunk] = []
        cast(Any, ctx).send_chunk = MagicMock(side_effect=lambda chunk: sent.append(chunk))

        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
            config=OllamaConfig(think=True),
        )
        with patch.object(model, 'build_prompt', return_value='hi'):
            await model._generate_ollama_response(request=request, ctx=ctx)

        assert len(sent) == 1
        parts = sent[0].content
        assert all(not isinstance(part.root, ReasoningPart) for part in parts)
        assert parts[0].root.text == '<think>partial'


class TestThinkingRequested:
    """Tests for OllamaModel._thinking_requested (mirrors Go ThinkOption.IsEnabled)."""

    def test_bool_true_enables(self) -> None:
        """think=True enables the fallback."""
        assert OllamaModel._thinking_requested(OllamaConfig(think=True)) is True

    def test_bool_false_disables(self) -> None:
        """think=False disables the fallback."""
        assert OllamaModel._thinking_requested(OllamaConfig(think=False)) is False

    def test_effort_string_enables(self) -> None:
        """A non-empty effort string (low/medium/high) enables the fallback."""
        assert OllamaModel._thinking_requested(OllamaConfig(think='high')) is True

    def test_absent_or_none_disables(self) -> None:
        """Configs without think, and None, do not enable the fallback."""
        assert OllamaModel._thinking_requested(OllamaConfig(num_ctx=8)) is False
        assert OllamaModel._thinking_requested(None) is False
        assert OllamaModel._thinking_requested({'think': 'low'}) is True


class TestResolveImage(unittest.IsolatedAsyncioTestCase):
    """Tests for OllamaModel._resolve_image."""

    async def test_data_uri_strips_prefix(self) -> None:
        """Data URIs should have their prefix stripped, returning raw base64."""
        data_uri = 'data:image/jpeg;base64,/9j/4AAQSkZJRg=='
        result = await OllamaModel._resolve_image(data_uri)
        assert result == '/9j/4AAQSkZJRg=='

    async def test_data_uri_png(self) -> None:
        """PNG data URI should also be stripped correctly."""
        data_uri = 'data:image/png;base64,iVBORw0KGgo='
        result = await OllamaModel._resolve_image(data_uri)
        assert result == 'iVBORw0KGgo='

    async def test_data_uri_without_comma_raises(self) -> None:
        """A malformed data URI with no comma separator should raise ValueError."""
        with self.assertRaises(ValueError):
            await OllamaModel._resolve_image('data:image/png;base64')

    async def test_raw_base64_passthrough(self) -> None:
        """Raw base64 strings (not data URIs, not URLs) pass through unchanged."""
        raw_b64 = '/9j/4AAQSkZJRgABAQ=='
        result = await OllamaModel._resolve_image(raw_b64)
        assert result == raw_b64

    async def test_local_file_path_passthrough(self) -> None:
        """Local file paths pass through unchanged for Image to handle."""
        path = './test_images/cat.jpg'
        result = await OllamaModel._resolve_image(path)
        assert result == path

    @patch('genkit.plugins.ollama.models.get_cached_client')
    async def test_http_url_downloads_image(self, mock_get_client: MagicMock) -> None:
        """HTTP URLs should be downloaded and returned as bytes."""
        mock_response = MagicMock()
        mock_response.content = b'\x89PNG\r\n\x1a\n'
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = await OllamaModel._resolve_image('https://example.com/cat.jpg')

        assert result == b'\x89PNG\r\n\x1a\n'
        mock_get_client.assert_called_once_with(
            cache_key='ollama/image-fetch',
            timeout=60.0,
            headers={
                'User-Agent': 'Genkit/1.0 (https://github.com/genkit-ai/genkit; genkit@google.com)',
            },
            follow_redirects=True,
        )
        mock_client.get.assert_awaited_once_with('https://example.com/cat.jpg')
        mock_response.raise_for_status.assert_called_once()

    @patch('genkit.plugins.ollama.models.get_cached_client')
    async def test_http_url_raises_on_failure(self, mock_get_client: MagicMock) -> None:
        """HTTP errors during image download should propagate."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            '403 Forbidden', request=MagicMock(), response=MagicMock()
        )
        mock_client.get.return_value = mock_response
        mock_get_client.return_value = mock_client

        with self.assertRaises(httpx.HTTPStatusError):
            await OllamaModel._resolve_image('https://example.com/secret.jpg')


class TestBuildChatMessagesWithMedia(unittest.IsolatedAsyncioTestCase):
    """Tests for build_chat_messages with MediaPart content."""

    async def test_text_and_media_message(self) -> None:
        """Messages with text + media should produce text content and images."""
        request = ModelRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(root=TextPart(text='Describe this image')),
                        Part(root=MediaPart(media=Media(url='data:image/jpeg;base64,AAAA', content_type='image/jpeg'))),
                    ],
                )
            ]
        )

        with patch.object(OllamaModel, '_resolve_image', new_callable=AsyncMock, return_value='AAAA'):
            messages = await OllamaModel.build_chat_messages(request)

        assert len(messages) == 1
        assert messages[0].content == 'Describe this image'
        assert len(messages[0]['images']) == 1

    async def test_media_only_message(self) -> None:
        """Messages with only media should have empty text content."""
        request = ModelRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(root=MediaPart(media=Media(url='data:image/png;base64,BBB', content_type='image/png'))),
                    ],
                )
            ]
        )

        with patch.object(OllamaModel, '_resolve_image', new_callable=AsyncMock, return_value='BBB'):
            messages = await OllamaModel.build_chat_messages(request)

        assert len(messages) == 1
        assert messages[0].content == ''
        assert len(messages[0]['images']) == 1


class TestToOllamaRole:
    """Tests for OllamaModel._to_ollama_role.

    Ported from the former converters ``to_ollama_role`` tests — this logic now
    lives only as the model's static role-mapping helper.
    """

    def test_user(self) -> None:
        """USER maps to 'user'."""
        assert OllamaModel._to_ollama_role(Role.USER) == 'user'

    def test_model(self) -> None:
        """MODEL maps to 'assistant'."""
        assert OllamaModel._to_ollama_role(Role.MODEL) == 'assistant'

    def test_system(self) -> None:
        """SYSTEM maps to 'system'."""
        assert OllamaModel._to_ollama_role(Role.SYSTEM) == 'system'

    def test_tool(self) -> None:
        """TOOL maps to 'tool'."""
        assert OllamaModel._to_ollama_role(Role.TOOL) == 'tool'

    def test_unknown_raises(self) -> None:
        """An unrecognized role raises ValueError."""
        with pytest.raises(ValueError):
            OllamaModel._to_ollama_role(cast(Role, 'not-a-role'))


class TestBuildPrompt:
    """Tests for OllamaModel.build_prompt.

    Ported from the former converters ``build_prompt`` tests. Unlike that copy,
    the model method takes a ``ModelRequest`` (not ``list[Message]``) and logs
    when it skips a non-text part.
    """

    def test_single_message(self) -> None:
        """A single text message is returned verbatim."""
        request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])])
        assert OllamaModel.build_prompt(request) == 'Hello'

    def test_multiple_messages(self) -> None:
        """Text across messages is concatenated in order."""
        request = ModelRequest(
            messages=[
                Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='System. '))]),
                Message(role=Role.USER, content=[Part(root=TextPart(text='User.'))]),
            ]
        )
        assert OllamaModel.build_prompt(request) == 'System. User.'

    def test_empty_messages(self) -> None:
        """No messages yields an empty prompt."""
        assert OllamaModel.build_prompt(ModelRequest(messages=[])) == ''

    def test_non_text_part_skipped_and_logged(self) -> None:
        """Non-text parts are skipped (and logged), keeping only text content."""
        request = ModelRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part(root=TextPart(text='see ')),
                        Part(root=MediaPart(media=Media(url='data:image/png;base64,AAAA', content_type='image/png'))),
                    ],
                )
            ]
        )

        with patch('genkit.plugins.ollama.models.logger') as mock_logger:
            result = OllamaModel.build_prompt(request)

        assert result == 'see '
        mock_logger.error.assert_called_once()


class TestGetUsageInfo:
    """Tests for OllamaModel.get_usage_info.

    Ported from the former converters ``get_usage_info`` tests. The model method
    reads token counts off the Ollama API response object rather than from raw
    integer arguments.
    """

    def test_with_counts(self) -> None:
        """Token counts are taken from the API response and summed."""
        basic = ModelUsage(input_characters=100)
        api_response = ollama_api.GenerateResponse(response='x', prompt_eval_count=10, eval_count=20)

        got = OllamaModel.get_usage_info(basic_generation_usage=basic, api_response=api_response)

        assert got.input_tokens == 10
        assert got.output_tokens == 20
        assert got.total_tokens == 30
        assert got.input_characters == 100, 'Lost input_characters'

    def test_none_counts_default_to_zero(self) -> None:
        """Missing counts on the response default to zero."""
        api_response = ollama_api.GenerateResponse(response='x')

        got = OllamaModel.get_usage_info(basic_generation_usage=ModelUsage(), api_response=api_response)

        assert got.input_tokens == 0
        assert got.output_tokens == 0
        assert got.total_tokens == 0

    def test_none_response_passthrough(self) -> None:
        """A missing API response leaves the basic usage untouched."""
        basic = ModelUsage(input_characters=5)

        got = OllamaModel.get_usage_info(basic_generation_usage=basic, api_response=None)

        assert got.input_characters == 5


class TestBuildMultimodalChatResponse:
    """Tests for OllamaModel._build_multimodal_chat_response.

    Ported from the former converters ``build_response_parts`` tests. There is no
    1:1 model method: this is the surviving home for response-part building (text
    and tool calls), though it consumes an Ollama ``ChatResponse`` rather than
    raw content/tool-call arguments, and additionally handles image parts.
    """

    @staticmethod
    def _response(message: ollama_api.Message) -> ollama_api.ChatResponse:
        return ollama_api.ChatResponse(message=message)

    def test_text_only(self) -> None:
        """Text content becomes a single TextPart."""
        parts = OllamaModel._build_multimodal_chat_response(
            self._response(ollama_api.Message(role='assistant', content='Hello'))
        )

        assert len(parts) == 1
        assert isinstance(parts[0].root, TextPart)
        assert parts[0].root.text == 'Hello'

    def test_tool_calls(self) -> None:
        """Tool calls become ToolRequestParts carrying name and input."""
        message = ollama_api.Message(
            role='assistant',
            content='',
            tool_calls=[
                ollama_api.Message.ToolCall(
                    function=ollama_api.Message.ToolCall.Function(name='search', arguments={'q': 'test'})
                )
            ],
        )

        parts = OllamaModel._build_multimodal_chat_response(self._response(message))

        assert len(parts) == 1
        root = parts[0].root
        assert isinstance(root, ToolRequestPart)
        assert root.tool_request.name == 'search'
        assert root.tool_request.input == {'q': 'test'}

    def test_text_and_tool_calls(self) -> None:
        """Text plus a tool call yields two parts."""
        message = ollama_api.Message(
            role='assistant',
            content='Thinking...',
            tool_calls=[
                ollama_api.Message.ToolCall(function=ollama_api.Message.ToolCall.Function(name='calc', arguments={}))
            ],
        )

        parts = OllamaModel._build_multimodal_chat_response(self._response(message))

        assert len(parts) == 2, f'Expected 2 parts, got {len(parts)}'

    def test_empty_content_yields_no_text_part(self) -> None:
        """Empty content produces no parts."""
        parts = OllamaModel._build_multimodal_chat_response(
            self._response(ollama_api.Message(role='assistant', content=''))
        )

        assert parts == []

    def test_images_become_media_parts(self) -> None:
        """Image content becomes MediaParts."""
        message = ollama_api.Message(
            role='assistant',
            content='',
            images=[ollama_api.Image(value='iVBORw0KGgo=')],
        )

        parts = OllamaModel._build_multimodal_chat_response(self._response(message))

        assert len(parts) == 1
        assert isinstance(parts[0].root, MediaPart)
