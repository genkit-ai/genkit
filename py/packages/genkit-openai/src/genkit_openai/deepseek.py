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

"""DeepSeek model provider for Genkit."""

import os
from typing import Any, cast

from openai import AsyncOpenAI
from pydantic import Field

from genkit import ModelInfo, ModelRequest, ModelResponse, Supports
from genkit.model import model_action_metadata
from genkit.plugin_api import (
    Action,
    ActionKind,
    ActionMetadata,
    ActionRunContext,
    ModelConfig,
    Plugin,
    loop_local_client,
    to_json_schema,
)
from genkit_openai.models import OpenAIModel
from genkit_openai.typing import OpenAIConfig

DEEPSEEK_PLUGIN_NAME = 'deepseek'
DEEPSEEK_API_BASE_URL = 'https://api.deepseek.com'
SUPPORTED_DEEPSEEK_MODELS = ('deepseek-chat', 'deepseek-reasoner')

DEEPSEEK_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)


class DeepSeekConfig(OpenAIConfig):
    """Configuration for DeepSeek chat-completion models."""

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias='maxTokens',
        serialization_alias='max_tokens',
    )


def _normalize_config(config: object) -> DeepSeekConfig:
    """Normalize common and provider-specific fields for DeepSeek."""
    if isinstance(config, DeepSeekConfig):
        config_data = config.model_dump(exclude_none=True)
    else:
        normalized = OpenAIModel.normalize_config(config)
        config_data = normalized.model_dump(exclude_none=True)
        if isinstance(config, ModelConfig) and config.model_extra:
            config_data.update(config.model_extra)

    max_tokens = config_data.pop('max_tokens', None)
    if max_tokens is None:
        max_tokens = config_data.pop('maxTokens', None)
    else:
        config_data.pop('maxTokens', None)

    max_completion_tokens = config_data.pop('max_completion_tokens', None)
    if max_completion_tokens is None:
        max_completion_tokens = config_data.pop('maxCompletionTokens', None)
    else:
        config_data.pop('maxCompletionTokens', None)

    resolved_max_tokens = max_tokens if max_tokens is not None else max_completion_tokens
    if resolved_max_tokens is not None:
        config_data['max_tokens'] = resolved_max_tokens

    return DeepSeekConfig.model_validate(config_data)


def deepseek_name(name: str) -> str:
    """Return a model name in the canonical DeepSeek namespace."""
    return f'{DEEPSEEK_PLUGIN_NAME}/{name}'


def deepseek_model(name: str) -> str:
    """Return a DeepSeek model reference for use with Genkit."""
    return deepseek_name(name)


def _model_info(name: str) -> ModelInfo:
    return ModelInfo(label=f'DeepSeek - {name}', supports=DEEPSEEK_MODEL_SUPPORTS)


class DeepSeek(Plugin):
    """Plugin registering DeepSeek's OpenAI-compatible chat models."""

    name = DEEPSEEK_PLUGIN_NAME

    def __init__(self, api_key: str | None = None, **openai_params: object) -> None:
        """Initialize the DeepSeek plugin.

        Args:
            api_key: DeepSeek API key. Defaults to ``DEEPSEEK_API_KEY``.
            **openai_params: Additional parameters passed to ``AsyncOpenAI``.

        Raises:
            ValueError: If neither ``api_key`` nor ``DEEPSEEK_API_KEY`` is set.
        """
        resolved_api_key = api_key if api_key is not None else os.getenv('DEEPSEEK_API_KEY')
        if not resolved_api_key:
            raise ValueError('Pass api_key or set the DEEPSEEK_API_KEY environment variable.')

        client_params = dict(openai_params)
        client_params.update(api_key=resolved_api_key, base_url=DEEPSEEK_API_BASE_URL)
        self._openai_params = client_params
        self._runtime_client = loop_local_client(
            lambda: AsyncOpenAI(**cast(dict[str, Any], self._openai_params)),
        )

    async def init(self) -> list[Action]:
        """Register the canonical DeepSeek chat and reasoning models."""
        return [self._create_model_action(deepseek_name(name)) for name in SUPPORTED_DEEPSEEK_MODELS]

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve a DeepSeek model action."""
        if action_type != ActionKind.MODEL or not name.startswith(f'{DEEPSEEK_PLUGIN_NAME}/'):
            return None
        return self._create_model_action(name)

    async def list_actions(self) -> list[ActionMetadata]:
        """List the statically supported DeepSeek models without network I/O."""
        return [
            model_action_metadata(
                name=deepseek_name(name),
                config_schema=DeepSeekConfig,
                info=_model_info(name).model_dump(by_alias=True, exclude_none=True),
            )
            for name in SUPPORTED_DEEPSEEK_MODELS
        ]

    def _create_model_action(self, name: str) -> Action:
        clean_name = name.removeprefix(f'{DEEPSEEK_PLUGIN_NAME}/')
        model_info = _model_info(clean_name)

        async def _generate(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            request.config = _normalize_config(request.config)
            model = OpenAIModel(clean_name, self._runtime_client())
            return await model.generate(request, ctx)

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=_generate,
            metadata={
                'model': {
                    **model_info.model_dump(by_alias=True, exclude_none=True),
                    'customOptions': to_json_schema(DeepSeekConfig),
                },
            },
        )


__all__ = [
    'DEEPSEEK_API_BASE_URL',
    'DEEPSEEK_PLUGIN_NAME',
    'SUPPORTED_DEEPSEEK_MODELS',
    'DeepSeek',
    'DeepSeekConfig',
    'deepseek_model',
    'deepseek_name',
]
