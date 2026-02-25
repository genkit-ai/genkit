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

"""Hugging Face Plugin for Genkit."""

import os
from typing import Any

from genkit.ai import Plugin
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.core.schema import to_json_schema
from genkit.plugins.huggingface.model_info import POPULAR_HUGGINGFACE_MODELS
from genkit.plugins.huggingface.models import (
    HUGGINGFACE_PLUGIN_NAME,
    HuggingFaceConfig,
    HuggingFaceModel,
    huggingface_name,
)


class HuggingFace(Plugin):
    """Hugging Face plugin for Genkit.

    This plugin provides integration with Hugging Face's Inference API,
    enabling the use of 1,000,000+ models within the Genkit framework.

    Example:
        >>> from genkit import Genkit
        >>> from genkit.plugins.huggingface import HuggingFace
        >>>
        >>> ai = Genkit(
        ...     plugins=[HuggingFace()],
        ...     model='huggingface/meta-llama/Llama-3.3-70B-Instruct',
        ... )
        >>>
        >>> response = await ai.generate(prompt='Hello!')

    Using Inference Providers for faster inference:

        >>> ai = Genkit(
        ...     plugins=[HuggingFace(provider='groq')],  # Use Groq for speed
        ...     model='huggingface/meta-llama/Llama-3.3-70B-Instruct',
        ... )
    """

    name = HUGGINGFACE_PLUGIN_NAME

    def __init__(
        self,
        token: str | None = None,
        provider: str | None = None,
        models: list[str] | None = None,
        **hf_params: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the plugin and set up its configuration.

        Args:
            token: The Hugging Face API token. If not provided, it attempts to load
                from the HF_TOKEN environment variable.
            provider: Default inference provider to use (e.g., 'cerebras', 'groq',
                'together'). Can be overridden per-request via config.
            models: An optional list of model IDs to register with the plugin.
                If None, popular models will be listed but any model can be used.
            **hf_params: Additional parameters for the InferenceClient.

        Raises:
            GenkitError: If no token is provided via parameter or environment.
        """
        self.token = token if token is not None else os.getenv('HF_TOKEN')

        if not self.token:
            raise GenkitError(message='Please provide token or set HF_TOKEN environment variable.')

        self.provider = provider
        self.models = models
        self.hf_params = hf_params

    async def init(self) -> list[Action]:
        """Initialize the plugin.

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

        return self._create_model_action(name)

    def _create_model_action(self, name: str) -> Action:
        """Create an Action object for a Hugging Face model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        clean_name = (
            name.replace(HUGGINGFACE_PLUGIN_NAME + '/', '') if name.startswith(HUGGINGFACE_PLUGIN_NAME) else name
        )

        # Create the HuggingFace model instance
        hf_model = HuggingFaceModel(
            model=clean_name,
            token=str(self.token),
            provider=self.provider,
            **self.hf_params,
        )

        model_info = hf_model.get_model_info() or {}
        generate_fn = hf_model.to_generate_fn()

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=generate_fn,
            metadata={
                'model': {
                    **model_info,
                    'customOptions': to_json_schema(HuggingFaceConfig),
                },
            },
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """Generate a list of popular Hugging Face models.

        Note: This returns a curated list of popular models. Any model ID
        from huggingface.co can be used with this plugin.

        Returns:
            list[ActionMetadata]: A list of ActionMetadata objects for popular
                Hugging Face models.
        """
        actions_list = []
        for model, model_info in POPULAR_HUGGINGFACE_MODELS.items():
            actions_list.append(
                model_action_metadata(
                    name=huggingface_name(model), info=model_info.model_dump(), config_schema=HuggingFaceConfig
                )
            )

        return actions_list
