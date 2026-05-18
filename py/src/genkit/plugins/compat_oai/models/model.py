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

"""OpenAI Compatible Models for Genkit."""

import json
from collections.abc import Callable
from typing import Any, cast

import structlog
from openai import AsyncOpenAI
from openai.lib._pydantic import _ensure_strict_json_schema

from genkit import (
    Message,
    ModelConfig,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolDefinition,
)
from genkit.plugin_api import ActionRunContext
from genkit.plugins.compat_oai.models.model_info import SUPPORTED_OPENAI_MODELS
from genkit.plugins.compat_oai.models.utils import (
    DictMessageAdapter,
    MessageAdapter,
    MessageConverter,
    strip_markdown_fences,
)
from genkit.plugins.compat_oai.typing import OpenAIConfig, SupportedOutputFormat

logger = structlog.get_logger(__name__)


class OpenAIModel:
    """Handles OpenAI API interactions for the Genkit plugin."""

    def __init__(self, model: str, client: AsyncOpenAI) -> None:
        """Initializes the OpenAIModel instance with the specified model and OpenAI client parameters.

        Args:
            model: The OpenAI model to use for generating responses.
            client: Async OpenAI client instance.
        """
        self._model = model
        self._openai_client = client

    @property
    def name(self) -> str:
        """The name of the OpenAI model."""
        return self._model

    def _get_messages(self, messages: list[Message]) -> list[dict]:
        """Converts the request messages into the format required by OpenAI's API.

        Args:
            messages: A list of the user messages.

        Returns:
            A list of dictionaries, where each dictionary represents a message
            with 'role' and 'content' fields.

        Raises:
            ValueError: If no messages are provided in the request.
        """
        openai_messages = []
        for message in messages:
            openai_messages.extend(MessageConverter.to_openai(message=message))
        return openai_messages

    async def _get_tools_definition(self, tools: list[ToolDefinition]) -> list[dict]:
        """Converts the provided tools into OpenAI-compatible function call format.

        OpenAI's strict mode requires ``additionalProperties: false`` and a
        ``required`` array listing **every** property key at each level of the
        schema.  Rather than adding these fields manually, we delegate to
        ``_ensure_strict_json_schema`` — the same helper already used for
        structured-output response schemas — which handles all strict-mode
        constraints recursively.

        Args:
            tools: A list of tool definitions.

        Returns:
            A list of dictionaries representing the formatted tools.
        """
        result = []
        for tool_definition in tools:
            parameters = tool_definition.input_schema or {}
            if parameters:
                parameters = _ensure_strict_json_schema(parameters, path=(), root=parameters)

            function_call = {
                'type': 'function',
                'function': {
                    'name': tool_definition.name,
                    'description': tool_definition.description or '',
                    'parameters': parameters,
                    'strict': True,
                },
            }
            result.append(function_call)
        return result

    def _needs_schema_in_prompt(self, request: ModelRequest) -> bool:
        """Check whether the schema must be injected into the prompt.

        Models that only support ``json_object`` mode (e.g. DeepSeek) never
        receive the schema via ``response_format``.  When a schema is present
        in the request we must include it in the system message so the model
        knows what structure to produce.

        Args:
            request: The model request with output_format and output_schema.

        Returns:
            True when the schema should be injected into the messages.
        """
        if request.output_format != 'json' or not request.output_schema:
            return False
        # DeepSeek models use json_object mode — schema never reaches the API.
        return self._model.startswith('deepseek')

    def _get_response_format(self, request: ModelRequest) -> dict | None:
        """Determines the response format configuration based on the output settings.

        Args:
            request: The model request with output_format and output_schema.

        Returns:
            A dictionary representing the response format, which may include:
            - 'type': 'json_schema' and a validated JSON Schema if a schema is provided.
            - 'type': 'json_object' if the model supports JSON mode and no schema is provided.
            - 'type': 'text' as the default fallback.
        """
        if request.output_format == 'json':
            # DeepSeek models: always use 'json_object' (schema is injected
            # into the prompt by _get_openai_request_config instead).
            if self._model.startswith('deepseek'):
                return {'type': 'json_object'}
            if request.output_schema:
                return {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': request.output_schema.get('title', 'Response'),
                        'schema': _ensure_strict_json_schema(
                            request.output_schema, path=(), root=request.output_schema
                        ),
                        'strict': True,
                    },
                }

            model = SUPPORTED_OPENAI_MODELS[self._model]
            if model.supports and model.supports.output and SupportedOutputFormat.JSON_MODE in model.supports.output:
                return {'type': 'json_object'}

        return {'type': 'text'}

    def _clean_json_response(self, response: ModelResponse, request: ModelRequest) -> ModelResponse:
        """Strip markdown fences from JSON responses for json_object-mode models.

        Only applies when the model uses ``json_object`` mode (e.g. DeepSeek)
        and the request asked for JSON output.

        Args:
            response: The generate response.
            request: The original request.

        Returns:
            The response with cleaned text parts, or the original response.
        """
        if request.output_format != 'json' or not self._model.startswith('deepseek') or response.message is None:
            return response

        cleaned_parts: list[Part] = []
        changed = False
        for part in response.message.content:
            if isinstance(part.root, TextPart) and part.root.text:
                cleaned_text = strip_markdown_fences(part.root.text)
                if cleaned_text != part.root.text:
                    cleaned_parts.append(Part(root=TextPart(text=cleaned_text)))
                    changed = True
                else:
                    cleaned_parts.append(part)
            else:
                cleaned_parts.append(part)

        if changed:
            return ModelResponse(
                request=request,
                message=Message(role=response.message.role, content=cleaned_parts),
                finish_reason=response.finish_reason,
                finish_message=response.finish_message,
                latency_ms=response.latency_ms,
                usage=response.usage,
                custom=response.custom,
            )
        return response

    @staticmethod
    def _build_schema_instruction(schema: dict[str, Any]) -> dict[str, str]:
        """Build a system message instructing the model to follow a JSON schema.

        Used for models that only support ``json_object`` mode (e.g. DeepSeek)
        where the API does not accept a ``json_schema`` response format.

        Args:
            schema: The JSON schema dictionary.

        Returns:
            A dict representing an OpenAI system message.
        """
        formatted = json.dumps(schema, indent=2)
        return {
            'role': 'system',
            'content': (
                'You must respond with a JSON object that conforms '
                'EXACTLY to the following JSON schema. Do not include '
                'any additional fields beyond those specified in the '
                'schema. Use the exact field names shown.\n\n'
                f'```json\n{formatted}\n```'
            ),
        }

    async def _get_openai_request_config(self, request: ModelRequest) -> dict:
        """Get the OpenAI request configuration.

        Args:
            request: The request containing messages and configurations.

        Returns:
            A dictionary representing the OpenAI request configuration.
        """
        messages = self._get_messages(request.messages)

        # For models that only support json_object mode, inject the schema
        # into the messages so the model knows the expected output structure.
        if self._needs_schema_in_prompt(request) and request.output_schema:
            schema_msg = self._build_schema_instruction(request.output_schema)
            messages = [schema_msg, *messages]

        openai_config: dict[str, Any] = {
            'messages': messages,
            'model': self._model,
        }
        if request.tools:
            openai_config['tools'] = await self._get_tools_definition(request.tools)
        if any(msg.role == Role.TOOL for msg in request.messages):
            # After a tool response, stop forcing additional tool calls.
            openai_config['tool_choice'] = 'none'
        elif request.tool_choice:
            openai_config['tool_choice'] = request.tool_choice
        if request.output_format:
            response_format = self._get_response_format(request)
            if response_format:
                # pyrefly: ignore[bad-typed-dict-key] - response_format dict is valid for OpenAI API
                openai_config['response_format'] = response_format
        if request.config:
            openai_config.update(**request.config.model_dump(exclude_none=True))
        return openai_config

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        """Processes the request using OpenAI's chat completion API and returns the generated response.

        Args:
            request: The ModelRequest object containing the input text and configuration.

        Returns:
            A ModelResponse object containing the generated message.
        """
        openai_config = await self._get_openai_request_config(request=request)
        logger.debug('OpenAI generate request', model=self._model, streaming=False)
        response = await self._openai_client.chat.completions.create(**openai_config)
        logger.debug(
            'OpenAI raw API response',
            model=self._model,
            finish_reason=str(response.choices[0].finish_reason) if response.choices else None,
        )

        result = ModelResponse(
            request=request,
            message=MessageConverter.to_genkit(MessageAdapter(response.choices[0].message)),
        )
        return self._clean_json_response(result, request)

    async def _generate_stream(
        self, request: ModelRequest, callback: Callable[[ModelResponseChunk], None]
    ) -> ModelResponse:
        """Streams responses from the OpenAI client and sends chunks to a callback.

        Args:
            request: The ModelRequest object containing generation parameters.
            callback: A function to receive streamed ModelResponseChunk objects.

        Returns:
            ModelResponse: A final message with accumulated content after streaming is complete.
        """
        openai_config = await self._get_openai_request_config(request=request)
        openai_config['stream'] = True

        stream = await self._openai_client.chat.completions.create(**openai_config)

        tool_calls: dict[int, Any] = {}
        accumulated_content: list[Part] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta

            # Text content chunk
            if delta.content:
                message = MessageConverter.to_genkit(MessageAdapter(delta))
                accumulated_content.extend(message.content)
                callback(
                    ModelResponseChunk(
                        role=Role.MODEL,
                        content=message.content,
                    )
                )

            # Reasoning content chunk (DeepSeek R1 / reasoner models).
            # Note: Pydantic models raise AttributeError for unknown fields,
            # so getattr() with a default doesn't work. Use try-except.
            elif reasoning_text := MessageAdapter(delta).reasoning_content:
                reasoning_part = Part(root=ReasoningPart(reasoning=reasoning_text))
                accumulated_content.append(reasoning_part)
                callback(
                    ModelResponseChunk(
                        role=Role.MODEL,
                        content=[reasoning_part],
                    )
                )

            # Tool call chunk (partial function call)
            elif delta.tool_calls:
                for tool_call in delta.tool_calls:
                    # Accumulate fragmented tool call arguments
                    if tool_call.index not in tool_calls:
                        tool_calls[tool_call.index] = tool_call
                    else:
                        existing = tool_calls[tool_call.index]
                        if hasattr(existing, 'function') and existing.function and tool_call.function:
                            existing.function.arguments += tool_call.function.arguments
                content = [
                    MessageConverter.tool_call_to_genkit(
                        tool_calls[tool_call.index],
                        args_segment=tool_call.function.arguments if tool_call.function else None,
                    )
                    for tool_call in delta.tool_calls
                ]
                callback(ModelResponseChunk(role=Role.MODEL, content=content))

        if tool_calls:
            message = MessageConverter.to_genkit(
                DictMessageAdapter({'tool_calls': tool_calls.values(), 'role': Role.MODEL})
            )
            accumulated_content.extend(message.content)

        result = ModelResponse(
            request=request,
            message=Message(role=Role.MODEL, content=accumulated_content),
        )
        return self._clean_json_response(result, request)

    async def generate(self, request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        """Processes the request using OpenAI's chat completion API.

        Args:
            request: The request containing messages and configurations.
            ctx: The context of the action run.

        Returns:
            A ModelResponse containing the model's response.
        """
        request.config = self.normalize_config(request.config)

        if ctx.is_streaming:
            logger.debug('OpenAI generate request', model=self._model, streaming=True)
            return await self._generate_stream(request, ctx.send_chunk)
        else:
            return await self._generate(request)

    @staticmethod
    def normalize_config(config: object) -> OpenAIConfig:
        """Ensures the config is an OpenAIConfig instance."""
        if isinstance(config, OpenAIConfig):
            return config

        if isinstance(config, (ModelConfig, ModelConfig)):
            return OpenAIConfig(
                temperature=config.temperature,
                max_tokens=int(config.max_output_tokens) if config.max_output_tokens is not None else None,
                top_p=config.top_p,
                stop=config.stop_sequences,
            )

        if isinstance(config, dict):
            config_dict = cast(dict[str, Any], config)
            if config_dict.get('top_k'):
                del config_dict['top_k']
            return OpenAIConfig(**config_dict)

        raise ValueError(f'Expected request.config to be a dict or OpenAIConfig, got {type(config).__name__}.')
