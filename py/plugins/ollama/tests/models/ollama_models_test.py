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

from genkit.plugins.ollama.constants import OllamaAPITypes
from genkit.plugins.ollama.models import ModelDefinition, OllamaModel, _convert_parameters
from genkit.types import (
    ActionRunContext,
    GenerateRequest,
    GenerateResponseChunk,
    GenerationUsage,
    Media,
    MediaPart,
    Message,
    OutputConfig,
    Part,
    Role,
    TextPart,
)


class TestOllamaModelGenerate(unittest.IsolatedAsyncioTestCase):
    """Tests for Generate method of OllamaModel."""

    async def asyncSetUp(self) -> None:
        """Common setup for all async tests."""
        self.mock_client = MagicMock()
        self.request = GenerateRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])])
        self.ctx = ActionRunContext()
        cast(Any, self.ctx).send_chunk = MagicMock()

    @patch(
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(
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
            return_value=GenerationUsage(
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
            chat_response=mock_chat_response
        )
        cast(MagicMock, ollama_model.is_streaming_request).assert_called_with(ctx=self.ctx)
        cast(MagicMock, ollama_model.get_usage_info).assert_called_once()

        self.assertIsNotNone(response.message)
        self.assertEqual(cast(Message, response.message).role, Role.MODEL)
        self.assertEqual(len(cast(Message, response.message).content), 1)
        self.assertEqual(cast(Message, response.message).content[0].root.text, 'Parsed chat content')
        self.assertIsNotNone(response.usage)
        self.assertEqual(cast(GenerationUsage, response.usage).input_tokens, 5)
        self.assertEqual(cast(GenerationUsage, response.usage).output_tokens, 10)

    @patch(
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(
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
            return_value=GenerationUsage(
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
        self.assertEqual(cast(GenerationUsage, response.usage).input_tokens, 7)
        self.assertEqual(cast(GenerationUsage, response.usage).output_tokens, 14)

    @patch(
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(),
    )
    async def test_generate_chat_streaming(self, mock_get_basic_usage_stats: MagicMock) -> None:
        """Test generate method with CHAT API type in streaming mode."""
        model_def = ModelDefinition(name='chat-model', api_type=OllamaAPITypes.CHAT)
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)
        streaming_ctx = ActionRunContext(on_chunk=MagicMock())

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
            return_value=GenerationUsage(
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
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(),
    )
    async def test_generate_generate_streaming(self, mock_get_basic_usage_stats: MagicMock) -> None:
        """Test generate method with GENERATE API type in streaming mode."""
        model_def = ModelDefinition(
            name='generate-model',
            api_type=OllamaAPITypes.GENERATE,
        )
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)
        streaming_ctx = ActionRunContext(on_chunk=MagicMock())

        # Mock internal methods
        mock_generate_response = ollama_api.GenerateResponse(
            response='Generated text',
        )
        cast(Any, ollama_model)._generate_ollama_response = AsyncMock(
            return_value=mock_generate_response,
        )
        cast(Any, ollama_model).is_streaming_request = MagicMock(return_value=True)
        cast(Any, ollama_model).get_usage_info = MagicMock(
            return_value=GenerationUsage(
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
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(),
    )
    async def test_generate_chat_api_response_none(self, mock_get_basic_usage_stats: MagicMock) -> None:
        """Test generate method when _chat_with_ollama returns None."""
        model_def = ModelDefinition(name='chat-model', api_type=OllamaAPITypes.CHAT)
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)

        cast(Any, ollama_model)._chat_with_ollama = AsyncMock(return_value=None)
        cast(Any, ollama_model)._build_multimodal_chat_response = MagicMock()
        cast(Any, ollama_model).is_streaming_request = MagicMock(return_value=False)
        cast(Any, ollama_model).get_usage_info = MagicMock(return_value=GenerationUsage())

        response = await ollama_model.generate(self.request, self.ctx)

        cast(AsyncMock, ollama_model._chat_with_ollama).assert_awaited_once()
        cast(MagicMock, ollama_model._build_multimodal_chat_response).assert_not_called()
        self.assertIsNotNone(response.message)
        self.assertEqual(cast(Message, response.message).content[0].root.text, 'Failed to get response from Ollama API')
        self.assertIsNotNone(response.usage)
        self.assertEqual(cast(GenerationUsage, response.usage).input_tokens, None)
        self.assertEqual(cast(GenerationUsage, response.usage).output_tokens, None)

    @patch(
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(),
    )
    async def test_generate_generate_api_response_none(self, mock_get_basic_usage_stats: MagicMock) -> None:
        """Test generate method when _generate_ollama_response returns None."""
        model_def = ModelDefinition(name='generate-model', api_type=OllamaAPITypes.GENERATE)
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)

        cast(Any, ollama_model)._generate_ollama_response = AsyncMock(return_value=None)
        cast(Any, ollama_model).is_streaming_request = MagicMock(return_value=False)
        cast(Any, ollama_model).get_usage_info = MagicMock(return_value=GenerationUsage())

        response = await ollama_model.generate(self.request, self.ctx)

        cast(AsyncMock, ollama_model._generate_ollama_response).assert_awaited_once()
        self.assertIsNotNone(response.message)
        self.assertEqual(cast(Message, response.message).content[0].root.text, 'Failed to get response from Ollama API')
        self.assertIsNotNone(response.usage)
        self.assertEqual(cast(GenerationUsage, response.usage).input_tokens, None)
        self.assertEqual(cast(GenerationUsage, response.usage).output_tokens, None)


class TestOllamaModelChatWithOllama(unittest.IsolatedAsyncioTestCase):
    """Unit tests for OllamaModel._chat_with_ollama method."""

    async def asyncSetUp(self) -> None:
        """Common setup."""
        self.mock_ollama_client_instance = AsyncMock()
        self.mock_ollama_client_factory = MagicMock(return_value=self.mock_ollama_client_instance)
        self.model_definition = ModelDefinition(name='test-chat-model', api_type=OllamaAPITypes.CHAT)
        self.ollama_model = OllamaModel(client=self.mock_ollama_client_factory, model_definition=self.model_definition)
        self.request = GenerateRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])])
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

    async def test_streaming_chat_success(self) -> None:
        """Test _chat_with_ollama in streaming mode with multiple chunks."""
        self.mock_is_streaming_request.return_value = True
        self.ctx.is_streaming = True

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

    async def test_chat_with_output_format_string(self) -> None:
        """Test _chat_with_ollama with request.output.format string."""
        self.request.output = OutputConfig(format='json')

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
        self.request.output = OutputConfig(schema=schema_dict)

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
        self.request.output = OutputConfig(format=None, schema=None)

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
        self.request = GenerateRequest(
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
            chunk=GenerateResponseChunk(role=Role.MODEL, index=1, content=[Part(root=TextPart(text='chunk1 '))])
        )
        cast(MagicMock, self.ctx.send_chunk).assert_any_call(
            chunk=GenerateResponseChunk(role=Role.MODEL, index=2, content=[Part(root=TextPart(text='chunk2'))])
        )

    async def test_generate_api_raises_exception(self) -> None:
        """Test _generate_ollama_response handles exception from client.generate."""
        self.mock_ollama_client_instance.generate.side_effect = Exception('Ollama generate API Error')

        with self.assertRaisesRegex(Exception, 'Ollama generate API Error'):
            await self.ollama_model._generate_ollama_response(self.request, self.ctx)

        self.mock_ollama_client_instance.generate.assert_awaited_once()
        cast(MagicMock, self.ctx.send_chunk).assert_not_called()


@pytest.mark.parametrize(
    'input_schema, expected_output',
    [
        ({}, None),
        ({'properties': {'name': {'type': 'string'}}}, None),
        (
            {'type': 'object'},
            ollama_api.Tool.Function.Parameters(type='object', properties={}),
        ),
        (
            {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'User name'},
                    'age': {'type': 'integer', 'description': 'User age'},
                },
                'required': ['name'],
            },
            ollama_api.Tool.Function.Parameters(
                type='object',
                required=['name'],
                properties={
                    'name': ollama_api.Tool.Function.Parameters.Property(type='string', description='User name'),
                    'age': ollama_api.Tool.Function.Parameters.Property(type='integer', description='User age'),
                },
            ),
        ),
        (
            {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string', 'description': 'City name'},
                },
            },
            ollama_api.Tool.Function.Parameters(
                type='object',
                required=None,
                properties={
                    'city': ollama_api.Tool.Function.Parameters.Property(
                        type='string',
                        description='City name',
                    ),
                },
            ),
        ),
        (
            {
                'type': 'object',
                'properties': {},
            },
            ollama_api.Tool.Function.Parameters(
                type='object',
                required=None,
                properties={},
            ),
        ),
        # Test 8: Object schema with nested properties
        (
            {
                'type': 'object',
                'properties': {
                    'address': {'type': 'object', 'properties': {'street': {'type': 'string'}}},
                    'zip': {'type': 'string'},
                },
            },
            ollama_api.Tool.Function.Parameters(
                type='object',
                required=None,
                properties={
                    'address': ollama_api.Tool.Function.Parameters.Property(type='object', description=''),
                    'zip': ollama_api.Tool.Function.Parameters.Property(type='string', description=''),
                },
            ),
        ),
        (
            {'type': 'object', 'description': 'A general description'},
            ollama_api.Tool.Function.Parameters(
                type='object',
                required=None,
                properties={},
            ),
        ),
    ],
)
def test_convert_parameters(input_schema: dict[str, Any], expected_output: object) -> None:
    """Unit Tests for _convert_parameters function with various input schemas."""
    result = _convert_parameters(input_schema)
    assert result == expected_output


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
                'User-Agent': 'Genkit/1.0 (https://github.com/firebase/genkit; genkit@google.com)',
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
        request = GenerateRequest(
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
        request = GenerateRequest(
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
