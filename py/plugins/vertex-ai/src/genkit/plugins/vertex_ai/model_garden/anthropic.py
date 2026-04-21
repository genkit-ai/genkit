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


"""Anthropic models."""

from collections.abc import Awaitable, Callable
from typing import cast

from anthropic import AsyncAnthropic, AsyncAnthropicVertex
from pydantic import ConfigDict

from genkit import ModelConfig, ModelInfo, ModelRequest, ModelResponse, Supports
from genkit.plugin_api import ActionRunContext, loop_local_client
from genkit.plugins.anthropic.models import AnthropicModel


class AnthropicConfigSchema(ModelConfig):
    """Configuration for Anthropic models."""

    model_config = ConfigDict(extra='allow')


class AnthropicModelGarden:
    """Manages integration with Anthropic models on Vertex AI Model Garden."""

    def __init__(
        self,
        model: str,
        location: str,
        project_id: str,
    ) -> None:
        """Initializes the AnthropicModelGarden instance.

        Args:
            model: The name of the specific model to be used from Model Garden
                in the way <publisher>/<model> (e.g., 'anthropic/claude-3-5-sonnet-v2@20241022').
            location: The Google Cloud region where the Model Garden service
                is hosted (e.g., 'us-central1').
            project_id: The Google Cloud project ID where the Model Garden
                model is deployed.
        """
        self.name = model
        self._runtime_client = loop_local_client(lambda: AsyncAnthropicVertex(region=location, project_id=project_id))
        # Strip 'anthropic/' prefix for the model passed to Anthropic SDK
        clean_model_name = model.removeprefix('anthropic/')
        self._model_name = clean_model_name

    def get_handler(self) -> Callable[[ModelRequest, ActionRunContext], Awaitable[ModelResponse]]:
        """Returns the generate handler function for this model.

        Returns:
            The handler function that can be used as an Action's fn parameter.
        """

        async def _generate(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            model = AnthropicModel(
                model_name=self._model_name,
                client=cast(AsyncAnthropic, self._runtime_client()),
            )
            return await model.generate(request, ctx)

        return _generate

    def get_model_info(self) -> ModelInfo:
        """Returns the model information/metadata for this model.

        Returns:
            ModelInfo with the model's capabilities.
        """
        return ModelInfo(
            label=f'ModelGarden - {self.name}',
            supports=Supports(
                multiturn=True,
                media=True,
                tools=True,
                system_role=True,
                output=['text', 'json'],
            ),
        )

    @staticmethod
    def get_config_schema() -> type[ModelConfig]:
        """Returns the config schema for this model type."""
        return AnthropicConfigSchema
