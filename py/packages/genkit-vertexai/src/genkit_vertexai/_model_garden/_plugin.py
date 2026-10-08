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
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from genkit import ActionRunContext, GenkitError, ModelResponse
from genkit.model import ModelInfo, ModelRequest, model as create_model, model_action_metadata
from genkit.plugin_api import Action, ActionKind, ActionMetadata, Plugin, loop_local_client
from genkit_vertexai import _constants as const
from genkit_vertexai._model_garden._model_info import (
    SUPPORTED_OPENAI_COMPAT_MODELS,
    get_default_model_info,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from genkit_vertexai._model_garden._client import CachedOpenAI

MODELGARDEN_PLUGIN_NAME = 'modelgarden'


@dataclass(frozen=True)
class _Extra:
    """An optional publisher dependency of genkit-vertexai.

    Attributes:
        packages: Top-level packages the extra installs. A missing extra fails
            with ``ModuleNotFoundError.name`` set to one of these. Any other
            name (a broken transitive dependency or a submodule) re-raises as-is.
        missing_message: The error message naming the install command.
    """

    packages: frozenset[str]
    missing_message: str


# claude and the openai-compatible publishers are separate extras, so an app
# only installs the SDK for the models it actually calls.
_ANTHROPIC_EXTRA = _Extra(
    packages=frozenset({'anthropic', 'genkit_anthropic'}),
    missing_message="Model Garden Claude models need the anthropic extra: uv add 'genkit-vertexai[anthropic]'",
)
_OPENAI_EXTRA = _Extra(
    packages=frozenset({'openai', 'genkit_openai'}),
    missing_message=(
        'Model Garden Llama, Mistral, and other OpenAI-compatible models need the openai extra: '
        "uv add 'genkit-vertexai[openai]'"
    ),
)


@contextmanager
def _requires_extra(extra: _Extra) -> Iterator[None]:
    """Turns a missing extra inside the block into FAILED_PRECONDITION naming `uv add`.

    Raises:
        GenkitError: FAILED_PRECONDITION when one of ``extra.packages`` isn't installed.
    """
    try:
        yield
    except ModuleNotFoundError as e:
        if e.name not in extra.packages:
            raise
        raise GenkitError(status='FAILED_PRECONDITION', message=extra.missing_message) from e


def _openai_compat_model_info(name: str) -> ModelInfo:
    """Catalog info for an OpenAI-compatible model, or the defaults for an uncataloged one."""
    return SUPPORTED_OPENAI_COMPAT_MODELS.get(name) or get_default_model_info(name)


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

        def _new_cached_client() -> 'CachedOpenAI':
            # client.py imports openai, which is an extra; load it on first generate.
            from genkit_vertexai._model_garden._client import CachedOpenAI

            return CachedOpenAI(location=location, project_id=project_id)

        self._runtime_client = loop_local_client(_new_cached_client)

    async def create_client(self) -> 'AsyncOpenAI':
        """Return the per-loop AsyncOpenAI client, refreshing the token only when expired.

        Returns:
            The authenticated AsyncOpenAI client.
        """
        return await self._runtime_client().get()

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
        model_info = _openai_compat_model_info(self.name)
        supports = model_info.supports
        return {
            'name': model_info.label,
            'supports': (
                supports.model_dump(by_alias=False, exclude_none=False)
                if supports and hasattr(supports, 'model_dump')
                else {}
            ),
        }

    def to_openai_compatible_model(self) -> Callable[[ModelRequest, ActionRunContext], Awaitable[ModelResponse]]:
        """Convert the Model Garden model into an OpenAI-compatible Genkit model function.

        Returns:
            A callable function (specifically, the ``generate`` method of an
            ``OpenAIModel`` instance) that can be used by Genkit.
        """

        async def _generate(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            # Private import across packages, on purpose. The [openai] extra
            # pins genkit-openai to this package's exact version, so Model
            # Garden reuses the OpenAI-compatible model class instead of
            # copying it.
            from genkit_openai._models import OpenAIModel

            client = await self.create_client()
            info = _openai_compat_model_info(self.name)
            openai_model = OpenAIModel(self.name, client, supports=info.supports)
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
            name: The model id without the ``modelgarden/`` prefix.

        Returns:
            Action object if found, None otherwise.
        """
        if action_type != ActionKind.MODEL:
            return None

        return await self._create_model_action(name)

    def _location_and_project(self, name: str) -> tuple[str, str]:
        """Region and project the model ``name`` runs in.

        Raises:
            GenkitError: FAILED_PRECONDITION when no project ID was passed or found in the environment.
        """
        if not self.project_id:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message='project_id must be provided',
            )
        return self.model_locations.get(name, self.location), self.project_id

    async def _create_model_action(self, name: str) -> Action:
        """Create an Action object for a Model Garden Vertex AI model.

        The publisher's extra is checked before the project ID, so a missing
        extra is the first error reported.

        Args:
            name: The model id without the ``modelgarden/`` prefix, publisher
                included (``anthropic/claude-sonnet-4-6``).

        Returns:
            Action object for the model.
        """
        full_name = model_garden_name(name)

        if name.startswith('anthropic/'):
            with _requires_extra(_ANTHROPIC_EXTRA):
                from ._anthropic import AnthropicModelGarden

            location, project_id = self._location_and_project(name)
            claude = AnthropicModelGarden(model=name, location=location, project_id=project_id)
            return create_model(
                full_name,
                claude.get_handler(),
                config_schema=claude.get_config_schema(),
                info=claude.get_model_info(),
            )

        with _requires_extra(_OPENAI_EXTRA):
            from genkit_openai import OpenAIConfig

        location, project_id = self._location_and_project(name)
        openai_compat = ModelGardenModel(model=name, location=location, project_id=project_id)
        return create_model(
            full_name,
            openai_compat.to_openai_compatible_model(),
            config_schema=OpenAIConfig,
            info=_openai_compat_model_info(name),
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
            with _requires_extra(_OPENAI_EXTRA):
                from genkit_openai import OpenAIConfig
        except GenkitError:
            return []

        return [
            model_action_metadata(
                name=model_garden_name(model),
                info=model_info.model_dump(by_alias=True, exclude_none=True),
                config_schema=OpenAIConfig,
            )
            for model, model_info in SUPPORTED_OPENAI_COMPAT_MODELS.items()
        ]
