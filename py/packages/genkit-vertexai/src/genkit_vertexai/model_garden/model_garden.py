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


"""Model Garden plugin for Genkit.

The publisher SDKs are extras, so this module imports them only when a model
from that publisher is resolved or called.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from genkit_vertexai import constants as const
from genkit_vertexai.model_garden._model_info import (
    SUPPORTED_OPENAI_COMPAT_MODELS,
    get_default_model_info,
)

from genkit import ActionRunContext, GenkitError, ModelResponse
from genkit.model import ModelRequest, model as create_model, model_action_metadata
from genkit.plugin_api import Action, ActionKind, ActionMetadata, Plugin, to_json_schema

if TYPE_CHECKING:
    from openai import AsyncOpenAI

MODELGARDEN_PLUGIN_NAME = 'modelgarden'

# claude and the openai-compatible publishers are separate extras, so an app
# only installs the SDK for the models it actually calls.
_CLAUDE_EXTRA_MISSING = "Model Garden Claude models need the anthropic extra: uv add 'genkit-vertexai[anthropic]'"
_OPENAI_COMPAT_EXTRA_MISSING = (
    'Model Garden Llama, Mistral, and other OpenAI-compatible models need the openai extra: '
    "uv add 'genkit-vertexai[openai]'"
)


def model_garden_name(name: str) -> str:
    """Create a Model Garden action name.

    Args:
        name: Base name for the action.

    Returns:
        The fully qualified Model Garden action name.
    """
    return f'{MODELGARDEN_PLUGIN_NAME}/{name}'


class ModelGardenModel:
    """Manages integration with Google's Model Garden service for Genkit.

    This class provides a convenient way to interact with models hosted on
    Google's Model Garden, allowing them to be exposed as Genkit models
    with OpenAI compatibility. It handles client initialization, model
    information retrieval, and dynamic model definition within the Genkit
    registry.
    """

    def __init__(
        self,
        model: str,
        location: str,
        project_id: str,
    ) -> None:
        """Initialize the ModelGardenModel instance.

        Client creation is deferred to ``create_client()`` (async) so the
        blocking credential refresh never runs on the event loop.

        Args:
            model: The name of the specific model to be used from Model Garden
                in the way <publisher>/<model> (e.g., 'meta/llama3.2-pro-max').
            location: The Google Cloud region where the Model Garden service
                is hosted (e.g., 'us-central1').
            project_id: The Google Cloud project ID where the Model Garden
                model is deployed.
        """
        self.name = model
        self._openai_params = {'location': location, 'project_id': project_id}

    async def create_client(self) -> 'AsyncOpenAI':
        """Create the AsyncOpenAI client with refreshed credentials.

        This offloads the blocking ``credentials.refresh()`` call to a
        thread via ``OpenAIClient.create()``.

        Returns:
            The authenticated AsyncOpenAI client.
        """
        from genkit_vertexai.model_garden.client import OpenAIClient

        return await OpenAIClient.create(**self._openai_params)

    def get_model_info(self) -> dict[str, object] | None:
        """Retrieve metadata and supported features for the specified model.

        This method looks up the model's information from a predefined list
        of supported OpenAI-compatible models or provides default information.

        Returns:
            A dictionary containing the model's 'name' and 'supports' features,
            or None if no information can be found (though typically, a default
            is provided). The 'supports' key contains a dictionary representing
            the model's capabilities (e.g., tools, streaming).
        """
        model_info = SUPPORTED_OPENAI_COMPAT_MODELS.get(self.name, get_default_model_info(self.name))
        supports = model_info.supports
        return {
            'name': model_info.label,
            'supports': (
                supports.model_dump(by_alias=False, exclude_none=False)
                if supports and hasattr(supports, 'model_dump')
                else {}
            ),
        }

    def to_openai_compatible_model(self) -> Callable:
        """Convert the Model Garden model into an OpenAI-compatible Genkit model function.

        Returns:
            A callable function (specifically, the ``generate`` method of an
            ``OpenAIModel`` instance) that can be used by Genkit.
        """

        async def _generate(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            # Private import across packages, on purpose. genkit-openai and
            # genkit-vertexai release in lockstep, so Model Garden reuses the
            # OpenAI-compatible model class instead of copying it.
            from genkit_openai._models import OpenAIModel

            client = await self.create_client()
            openai_model = OpenAIModel(self.name, client)
            return await openai_model.generate(request, ctx)

        return _generate


class ModelGarden(Plugin):
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
            try:
                from .anthropic import AnthropicModelGarden as AnthropicWorker
            except ModuleNotFoundError as e:
                raise GenkitError(status='FAILED_PRECONDITION', message=_CLAUDE_EXTRA_MISSING) from e

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

            return create_model(
                name,
                handler,
                config_schema=model_proxy.get_config_schema(),
                metadata={
                    'model': {
                        **model_info.model_dump(),
                        'customOptions': to_json_schema(model_proxy.get_config_schema()),
                    },
                },
            )

        try:
            from genkit_openai import OpenAIConfig
        except ModuleNotFoundError as e:
            raise GenkitError(status='FAILED_PRECONDITION', message=_OPENAI_COMPAT_EXTRA_MISSING) from e

        location = self.model_locations.get(clean_name, self.location)
        if not self.project_id:
            raise ValueError('project_id must be provided')
        model_proxy = ModelGardenModel(
            model=clean_name,
            location=location,
            project_id=self.project_id,
        )

        # Get model info and handler
        model_info = SUPPORTED_OPENAI_COMPAT_MODELS.get(clean_name, {})
        handler = model_proxy.to_openai_compatible_model()

        return create_model(
            name,
            handler,
            config_schema=OpenAIConfig,
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
            Empty when the ``openai`` extra isn't installed, so the Dev UI only lists models that can run.
        """
        try:
            from genkit_openai import OpenAIConfig
        except ModuleNotFoundError:
            return []

        actions_list = []
        for model, model_info in SUPPORTED_OPENAI_COMPAT_MODELS.items():
            actions_list.append(
                model_action_metadata(
                    name=model_garden_name(model), info=model_info.model_dump(), config_schema=OpenAIConfig
                )
            )

        return actions_list
