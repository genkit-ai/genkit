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

"""OpenAI Compatible Model handlers for Genkit."""

from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from genkit.ai import ActionRunContext
from genkit.plugins.compat_oai.models.model import OpenAIModel
from genkit.plugins.compat_oai.models.model_info import (
    SUPPORTED_OPENAI_COMPAT_MODELS,
    SUPPORTED_OPENAI_MODELS,
    PluginSource,
)
from genkit.types import (
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
)


class OpenAIModelHandler:
    """Handles OpenAI API interactions for the Genkit plugin."""

    def __init__(self, model: OpenAIModel, source: PluginSource = PluginSource.OPENAI) -> None:
        """Initializes the OpenAIModelHandler with a specified model.

        Args:
            model: An instance of a Model subclass representing the OpenAI model.
            source: Helps distinguish if model handler is called from model-garden plugin.
                    Default source is openai.
        """
        self._model = model
        self._source = source

    @staticmethod
    def _get_supported_models(source: PluginSource) -> dict[str, ModelInfo]:
        """Returns the supported models based on the plugin source.

        Args:
            source: Helps distinguish if model handler is called from model-garden plugin.
                    Default source is openai.

        Returns:
            Openai models if source is openai. Merges supported openai models
            with openai-compat models if source is model-garden.

        """
        return SUPPORTED_OPENAI_COMPAT_MODELS if source == PluginSource.MODEL_GARDEN else SUPPORTED_OPENAI_MODELS

    @classmethod
    def get_model_handler(
        cls, model: str, client: AsyncOpenAI, source: PluginSource = PluginSource.OPENAI
    ) -> Callable[[GenerateRequest, ActionRunContext], Awaitable[GenerateResponse]]:
        """Factory method to initialize the model handler for the specified OpenAI model.

        OpenAI models in this context are not instantiated as traditional
        classes but rather as Actions. This method returns a callable that
        serves as an action handler, conforming to the structure of:

            Action[GenerateRequest, GenerateResponse, GenerateResponseChunk]

        Args:
            model: The OpenAI model name.
            client: OpenAI client instance.
            source: Helps distinguish if model handler is called from model-garden plugin.
                    Default source is openai.

        Returns:
            A callable function that acts as an action handler.

        Raises:
            ValueError: If the specified model is not supported.
        """
        supported_models = cls._get_supported_models(source)

        if model not in supported_models:
            raise ValueError(f"Model '{model}' is not supported.")

        openai_model = OpenAIModel(model, client)
        return cls(openai_model, source).generate

    def _validate_version(self, version: str) -> None:
        """Validates whether the specified model version is supported.

        Args:
            version: The version of the model to be validated.

        Raises:
            ValueError: If the specified model version is not supported.
        """
        supported_models = self._get_supported_models(self._source)
        model_info = supported_models[self._model.name]
        if model_info.versions is not None and version not in model_info.versions:
            raise ValueError(f"Model version '{version}' is not supported.")

    async def generate(self, request: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
        """Processes the request using OpenAI's chat completion API.

        Args:
            request: The request containing messages and configurations.
            ctx: The context of the action run.

        Returns:
            A GenerateResponse containing the model's response.

        Raises:
            ValueError: If the specified model version is not supported.
        """
        request.config = self._model.normalize_config(request.config)

        if request.config and hasattr(request.config, 'model') and request.config.model:
            self._validate_version(request.config.model)

        return await self._model.generate(request, ctx)
