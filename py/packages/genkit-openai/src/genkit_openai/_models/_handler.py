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

from collections.abc import Awaitable, Callable, Mapping
from typing import cast

from openai import AsyncOpenAI

from genkit import ActionRunContext, ModelResponse
from genkit.model import ModelInfo, ModelRequest
from genkit_openai._models._model import OpenAIModel
from genkit_openai._models._model_info import SUPPORTED_OPENAI_MODELS

_SUPPORTED_MODELS = cast(Mapping[str, ModelInfo], SUPPORTED_OPENAI_MODELS)


class OpenAIModelHandler:
    """Handles OpenAI API interactions for the Genkit plugin."""

    def __init__(self, model: OpenAIModel) -> None:
        """Initializes the OpenAIModelHandler with a specified model.

        Args:
            model: An instance of a Model subclass representing the OpenAI model.
        """
        self._model = model

    @classmethod
    def get_model_handler(
        cls, model: str, client: AsyncOpenAI
    ) -> Callable[[ModelRequest, ActionRunContext], Awaitable[ModelResponse]]:
        """Factory method to initialize the model handler for the specified OpenAI model.

        OpenAI models in this context are not instantiated as traditional
        classes but rather as Actions. This method returns a callable that
        serves as an action handler, conforming to the structure of:

            Action[ModelRequest, ModelResponse, ModelResponseChunk]

        Args:
            model: The OpenAI model name.
            client: OpenAI client instance.

        Returns:
            A callable function that acts as an action handler.

        Raises:
            ValueError: If the specified model is not supported.
        """
        if model not in _SUPPORTED_MODELS:
            raise ValueError(f"Model '{model}' is not supported.")

        openai_model = OpenAIModel(model, client)
        return cls(openai_model).generate

    def _validate_version(self, version: str) -> None:
        """Validates whether the specified model version is supported.

        Args:
            version: The version of the model to be validated.

        Raises:
            ValueError: If the specified model version is not supported.
        """
        model_info = _SUPPORTED_MODELS[self._model.name]
        if model_info.versions is not None and version not in model_info.versions:
            raise ValueError(f"Model version '{version}' is not supported.")

    async def generate(self, request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        """Processes the request using OpenAI's chat completion API.

        Args:
            request: The request containing messages and configurations.
            ctx: The context of the action run.

        Returns:
            A ModelResponse containing the model's response.

        Raises:
            ValueError: If the specified model version is not supported.
        """
        request.config = self._model._normalize_config(request.config)

        if request.config and hasattr(request.config, 'model') and request.config.model:
            self._validate_version(request.config.model)

        return await self._model.generate(request, ctx)
