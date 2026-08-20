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

"""xAI model provider for Genkit."""

import os
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import ConfigDict, Field

from genkit import ModelConfig, ModelInfo, ModelRequest, ModelResponse, Supports
from genkit.model import model_action_metadata
from genkit.plugin_api import (
    Action,
    ActionKind,
    ActionMetadata,
    ActionRunContext,
    Plugin,
    loop_local_client,
    to_json_schema,
)
from genkit_openai.models import OpenAIImageModel, OpenAIModel, OpenAIModelHandler
from genkit_openai.typing import OpenAIConfig

XAI_PLUGIN_NAME = 'xai'
XAI_API_BASE_URL = 'https://api.x.ai/v1'

SUPPORTED_XAI_CHAT_MODELS = (
    'grok-3',
    'grok-3-fast',
    'grok-3-mini',
    'grok-3-mini-fast',
    'grok-2-vision-1212',
)
SUPPORTED_XAI_IMAGE_MODELS = ('grok-2-image-1212',)

XAI_CHAT_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)
XAI_VISION_SUPPORTS = Supports(
    multiturn=False,
    media=True,
    tools=True,
    system_role=False,
    output=['text', 'json'],
)
XAI_IMAGE_SUPPORTS = Supports(
    multiturn=False,
    media=False,
    tools=False,
    system_role=False,
    output=['media'],
)


class XAIConfig(OpenAIConfig):
    """Configuration for xAI chat-completion models."""

    deferred: bool | None = None
    reasoning_effort: Literal['low', 'medium', 'high'] | None = Field(  # pyrefly: ignore[bad-override]
        default=None,
        validation_alias='reasoningEffort',
        serialization_alias='reasoning_effort',
    )
    web_search_options: dict[str, object] | None = Field(
        default=None,
        validation_alias='webSearchOptions',
        serialization_alias='web_search_options',
    )


class XAIImageConfig(ModelConfig):
    """Configuration for xAI image-generation models."""

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    size: Literal['1024x1024', '1792x1024', '1024x1792'] | None = None
    style: Literal['vivid', 'natural'] | None = None
    user: str | None = None
    n: int = Field(default=1, ge=1, le=10)
    quality: Literal['standard', 'hd'] | None = None
    response_format: Literal['b64_json', 'url'] | None = Field(
        default='b64_json',
        validation_alias='responseFormat',
        serialization_alias='response_format',
    )


def xai_name(name: str) -> str:
    """Return a model name in the canonical xAI namespace."""
    return f'{XAI_PLUGIN_NAME}/{name}'


def xai_model(name: str) -> str:
    """Return an xAI model reference for use with Genkit."""
    return xai_name(name)


def _is_image_model(name: str) -> bool:
    return 'image' in name


def _model_info(name: str) -> ModelInfo:
    if _is_image_model(name):
        supports = XAI_IMAGE_SUPPORTS
    elif 'vision' in name:
        supports = XAI_VISION_SUPPORTS
    else:
        supports = XAI_CHAT_SUPPORTS
    return ModelInfo(label=f'xAI - {name}', supports=supports)


def _config_schema(name: str) -> type[ModelConfig]:
    return XAIImageConfig if _is_image_model(name) else XAIConfig


class _XAIChatModel(OpenAIModel):
    """OpenAI-compatible chat model with xAI request extensions."""

    @staticmethod
    def normalize_config(config: object) -> XAIConfig:
        """Normalize generic Genkit configuration as xAI configuration."""
        normalized = OpenAIModel.normalize_config(dict(config) if isinstance(config, dict) else config)
        return XAIConfig.model_validate(normalized.model_dump(exclude_unset=True))

    async def _get_openai_request_config(self, request: ModelRequest) -> dict[str, Any]:
        config = await super()._get_openai_request_config(request)
        deferred = config.pop('deferred', None)
        has_snake_case_extra_body = 'extra_body' in config
        extra_body = config.pop('extra_body', None)
        camel_case_extra_body = config.pop('extraBody', None)
        if not has_snake_case_extra_body:
            extra_body = camel_case_extra_body
        if extra_body is not None and not isinstance(extra_body, dict):
            raise ValueError('XAIConfig extra_body must be a dictionary.')
        if deferred is not None:
            extra_body = {**(extra_body or {}), 'deferred': deferred}
        if extra_body is not None:
            config['extra_body'] = extra_body
        return config


class XAI(Plugin):
    """Plugin registering xAI's OpenAI-compatible Grok models."""

    name = XAI_PLUGIN_NAME

    def __init__(self, api_key: str | None = None, **openai_params: object) -> None:
        """Initialize the xAI plugin.

        Args:
            api_key: xAI API key. Defaults to ``XAI_API_KEY``.
            **openai_params: Additional parameters passed to ``AsyncOpenAI``.

        Raises:
            ValueError: If neither ``api_key`` nor ``XAI_API_KEY`` is set.
        """
        resolved_api_key = api_key if api_key is not None else os.getenv('XAI_API_KEY')
        if not resolved_api_key:
            raise ValueError('Pass api_key or set the XAI_API_KEY environment variable.')

        client_params: dict[str, Any] = dict(openai_params)
        client_params.update(api_key=resolved_api_key, base_url=XAI_API_BASE_URL)
        self._openai_params = client_params
        self._runtime_client = loop_local_client(
            lambda: AsyncOpenAI(**self._openai_params),
        )

    async def init(self) -> list[Action]:
        """Register the canonical xAI chat, vision, and image models."""
        chat_actions = [self._create_chat_action(xai_name(name)) for name in SUPPORTED_XAI_CHAT_MODELS]
        image_actions = [self._create_image_action(xai_name(name)) for name in SUPPORTED_XAI_IMAGE_MODELS]
        return [*chat_actions, *image_actions]

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an xAI model action."""
        if action_type != ActionKind.MODEL or not name.startswith(f'{XAI_PLUGIN_NAME}/'):
            return None
        if _is_image_model(name):
            return self._create_image_action(name)
        return self._create_chat_action(name)

    async def list_actions(self) -> list[ActionMetadata]:
        """List the statically supported xAI models without network I/O."""
        return [
            model_action_metadata(
                name=xai_name(name),
                config_schema=_config_schema(name),
                info=_model_info(name).model_dump(by_alias=True, exclude_none=True),
            )
            for name in (*SUPPORTED_XAI_CHAT_MODELS, *SUPPORTED_XAI_IMAGE_MODELS)
        ]

    def _create_chat_action(self, name: str) -> Action:
        clean_name = name.removeprefix(f'{XAI_PLUGIN_NAME}/')

        async def _generate(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            model = OpenAIModelHandler(_XAIChatModel(clean_name, self._runtime_client()))
            return await model.generate(request, ctx)

        return self._action(name, clean_name, _generate)

    def _create_image_action(self, name: str) -> Action:
        clean_name = name.removeprefix(f'{XAI_PLUGIN_NAME}/')

        async def _generate(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            model = OpenAIImageModel(clean_name, self._runtime_client())
            return await model.generate(request, ctx)

        return self._action(name, clean_name, _generate)

    @staticmethod
    def _action(
        name: str,
        clean_name: str,
        generate: Callable[[ModelRequest, ActionRunContext], Awaitable[ModelResponse]],
    ) -> Action:
        model_info = _model_info(clean_name)
        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=generate,
            metadata={
                'model': {
                    **model_info.model_dump(by_alias=True, exclude_none=True),
                    'customOptions': to_json_schema(_config_schema(clean_name)),
                },
            },
        )


__all__ = [
    'SUPPORTED_XAI_CHAT_MODELS',
    'SUPPORTED_XAI_IMAGE_MODELS',
    'XAI',
    'XAI_API_BASE_URL',
    'XAI_PLUGIN_NAME',
    'XAIConfig',
    'XAIImageConfig',
    'xai_model',
    'xai_name',
]
