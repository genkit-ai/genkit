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

"""ModelGarden API Compatible Plugin for Genkit."""

import os
from typing import cast

from genkit.ai import Plugin
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind
from genkit.core.schema import to_json_schema
from genkit.plugins.compat_oai.models import SUPPORTED_OPENAI_COMPAT_MODELS
from genkit.plugins.compat_oai.typing import OpenAIConfig
from genkit.plugins.vertex_ai import constants as const

from .model_garden import MODELGARDEN_PLUGIN_NAME, ModelGarden, model_garden_name


class ModelGardenPlugin(Plugin):
    """Model Garden plugin for Genkit.

    This plugin provides integration with Google Cloud's Vertex AI platform,
    enabling the use of Vertex AI models and services within the Genkit
    framework. It handles initialization of the Model Garden client and
    registration of model actions.
    """

    name = MODELGARDEN_PLUGIN_NAME

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        models: list[str] | None = None,
        model_locations: dict[str, str] | None = None,
    ) -> None:
        """Initializes the plugin and sets up its configuration.

        This constructor prepares the plugin by assigning the Google Cloud project ID,
        location, and a list of models to be used.

        Args:
            project_id: The Google Cloud project ID to use. If not provided, it attempts
                to load from the `GCLOUD_PROJECT` environment variable.
            location: The Google Cloud region to use for services. If not provided,
                it defaults to `DEFAULT_REGION`.
            models: An optional list of model names to register with the plugin.
            model_locations: An optional dictionary mapping model names to their specific
                Google Cloud regions. This overrides the default `location` for the
                specified models.
        """
        self.project_id = (
            project_id
            if project_id is not None
            else os.getenv(const.GCLOUD_PROJECT) or os.getenv('GOOGLE_CLOUD_PROJECT')
        )

        self.location = (
            location or os.getenv('GOOGLE_CLOUD_LOCATION') or os.getenv('GOOGLE_CLOUD_REGION') or const.DEFAULT_REGION
        )

        self.models = models
        self.model_locations = model_locations or {}

    async def init(self) -> list[Action]:
        """Initialize plugin.

        Returns:
            Empty list (using lazy loading via resolve).
        """
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced name of the action to resolve.

        Returns:
            Action object if found, None otherwise.
        """
        if action_type != ActionKind.MODEL:
            return None

        return await self._create_model_action(name)

    async def _create_model_action(self, name: str) -> Action:
        """Create an Action object for a Model Garden Vertex AI model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        clean_name = (
            name.replace(MODELGARDEN_PLUGIN_NAME + '/', '') if name.startswith(MODELGARDEN_PLUGIN_NAME) else name
        )

        if clean_name.startswith('anthropic/'):
            from .anthropic import AnthropicModelGarden as AnthropicWorker

            location = self.model_locations.get(clean_name, self.location)
            if not self.project_id:
                raise ValueError('project_id must be provided')
            model_proxy = AnthropicWorker(
                model=clean_name,
                location=location,
                project_id=self.project_id,
            )

            handler = model_proxy.get_handler()
            model_info = model_proxy.get_model_info()

            return Action(
                kind=ActionKind.MODEL,
                name=name,
                fn=handler,
                metadata={
                    'model': {
                        **model_info.model_dump(),
                        'customOptions': to_json_schema(model_proxy.get_config_schema()),
                    },
                },
            )

        location = self.model_locations.get(clean_name, self.location)
        if not self.project_id:
            raise ValueError('project_id must be provided')
        model_proxy = ModelGarden(
            model=clean_name,
            location=location,
            project_id=self.project_id,
        )

        # Create the async client (offloads credential refresh to thread)
        await model_proxy.create_client()

        # Get model info and handler
        model_info = SUPPORTED_OPENAI_COMPAT_MODELS.get(clean_name, {})
        handler = model_proxy.to_openai_compatible_model()

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=handler,
            metadata={
                'model': {
                    **(
                        model_info.model_dump()  # type: ignore[union-attr]
                        if hasattr(model_info, 'model_dump')
                        else cast(dict[str, object], model_info)
                    ),
                    'customOptions': to_json_schema(OpenAIConfig),
                },
            },
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """Generate a list of available actions or models.

        Returns:
            list[ActionMetadata]: A list of ActionMetadata objects, each with the following attributes:
                - name (str): The name of the action or model.
                - kind (ActionKind): The type or category of the action.
                - info (dict): The metadata dictionary describing the model configuration and properties.
                - config_schema (type): The schema class used for validating the model's configuration.
        """
        actions_list = []
        for model, model_info in SUPPORTED_OPENAI_COMPAT_MODELS.items():
            actions_list.append(
                model_action_metadata(
                    name=model_garden_name(model), info=model_info.model_dump(), config_schema=OpenAIConfig
                )
            )

        return actions_list
