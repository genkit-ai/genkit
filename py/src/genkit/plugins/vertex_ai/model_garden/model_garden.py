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


"""Model Garden implementation."""

from __future__ import annotations

import typing
from collections.abc import Callable

if typing.TYPE_CHECKING:
    from openai import AsyncOpenAI

    from genkit import ModelRequest, ModelResponse
    from genkit.plugin_api import ActionRunContext

from genkit.plugins.compat_oai.models import (
    SUPPORTED_OPENAI_COMPAT_MODELS,
    get_default_model_info,
)
from genkit.plugins.compat_oai.models.model import OpenAIModel
from genkit.plugins.vertex_ai.model_garden.client import OpenAIClient

MODELGARDEN_PLUGIN_NAME = 'modelgarden'


def model_garden_name(name: str) -> str:
    """Create a Model Garden action name.

    Args:
        name: Base name for the action.

    Returns:
        The fully qualified Model Garden action name.
    """
    return f'{MODELGARDEN_PLUGIN_NAME}/{name}'


class ModelGarden:
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
        """Initialize the ModelGarden instance.

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

    async def create_client(self) -> AsyncOpenAI:
        """Create the AsyncOpenAI client with refreshed credentials.

        This offloads the blocking ``credentials.refresh()`` call to a
        thread via ``OpenAIClient.create()``.

        Returns:
            The authenticated AsyncOpenAI client.
        """
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
            client = await self.create_client()
            openai_model = OpenAIModel(self.name, client)
            return await openai_model.generate(request, ctx)

        return _generate
