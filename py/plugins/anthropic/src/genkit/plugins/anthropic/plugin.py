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

"""Anthropic plugin for Genkit."""

from typing import Any, cast

from anthropic import AsyncAnthropic
from genkit.ai import Plugin
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.registry import ActionKind
from genkit.core.schema import to_json_schema
from genkit.plugins.anthropic.model_info import SUPPORTED_ANTHROPIC_MODELS, get_model_info
from genkit.plugins.anthropic.models import AnthropicModel
from genkit.types import GenerationCommonConfig

ANTHROPIC_PLUGIN_NAME = 'anthropic'


def anthropic_name(name: str) -> str:
    """Get Anthropic model name.

    Args:
        name: The name of Anthropic model.

    Returns:
        Fully qualified Anthropic model name.
    """
    return f'{ANTHROPIC_PLUGIN_NAME}/{name}'


class Anthropic(Plugin):
    """Anthropic plugin for Genkit.

    This plugin adds Anthropic models to Genkit for generative AI applications.
    """

    name = ANTHROPIC_PLUGIN_NAME

    def __init__(
        self,
        models: list[str] | None = None,
        **anthropic_params: object,
    ) -> None:
        """Initializes Anthropic plugin with given configuration.

        Args:
            models: List of model names to register. Defaults to all supported models.
            **anthropic_params: Additional parameters passed to the AsyncAnthropic client.
                This may include api_key, base_url, timeout, and other configuration
                settings required by Anthropic's API.
        """
        self.models = models or list(SUPPORTED_ANTHROPIC_MODELS.keys())
        self._anthropic_params = anthropic_params
        self._anthropic_client = AsyncAnthropic(**cast(dict[str, Any], anthropic_params))

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

        return self._create_model_action(name)

    def _create_model_action(self, name: str) -> Action:
        """Create an Action object for an Anthropic model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace(f'{ANTHROPIC_PLUGIN_NAME}/', '') if name.startswith(ANTHROPIC_PLUGIN_NAME) else name

        model = AnthropicModel(model_name=clean_name, client=self._anthropic_client)
        model_info = get_model_info(clean_name)

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=model.generate,
            metadata={
                'model': {
                    'supports': (
                        model_info.supports.model_dump(by_alias=True, exclude_none=True) if model_info.supports else {}
                    ),
                    'customOptions': to_json_schema(GenerationCommonConfig),
                },
            },
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """List available Anthropic models.

        Returns:
            List of ActionMetadata for all supported models.
        """
        actions = []
        for model_name, model_info in SUPPORTED_ANTHROPIC_MODELS.items():
            actions.append(
                model_action_metadata(
                    name=anthropic_name(model_name),
                    info=model_info.model_dump(by_alias=True, exclude_none=True),
                    config_schema=GenerationCommonConfig,
                )
            )
        return actions
