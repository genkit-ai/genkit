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

"""DeepSeek Plugin for Genkit."""

import os
from typing import Any

from genkit.ai import Plugin
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.core.schema import to_json_schema
from genkit.plugins.compat_oai.typing import OpenAIConfig
from genkit.plugins.deepseek.model_info import SUPPORTED_DEEPSEEK_MODELS
from genkit.plugins.deepseek.models import DEEPSEEK_PLUGIN_NAME, DeepSeekModel, deepseek_name


class DeepSeek(Plugin):
    """DeepSeek plugin for Genkit.

    This plugin provides integration with DeepSeek's OpenAI-compatible API,
    enabling the use of DeepSeek models within the Genkit framework.
    """

    name = DEEPSEEK_PLUGIN_NAME

    def __init__(
        self,
        api_key: str | None = None,
        models: list[str] | None = None,
        **deepseek_params: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the plugin and set up its configuration.

        Args:
            api_key: The DeepSeek API key. If not provided, it attempts to load
                from the DEEPSEEK_API_KEY environment variable.
            models: An optional list of model names to register with the plugin.
                If None, all supported models will be registered.
            **deepseek_params: Additional parameters for the DeepSeek client.

        Raises:
            GenkitError: If no API key is provided via parameter or environment.
        """
        self.api_key = api_key if api_key is not None else os.getenv('DEEPSEEK_API_KEY')

        if not self.api_key:
            raise GenkitError(message='Please provide api_key or set DEEPSEEK_API_KEY environment variable.')

        self.models = models
        self.deepseek_params = deepseek_params

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
        """Create an Action object for a DeepSeek model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace(DEEPSEEK_PLUGIN_NAME + '/', '') if name.startswith(DEEPSEEK_PLUGIN_NAME) else name

        # Create the DeepSeek model instance
        deepseek_model = DeepSeekModel(
            model=clean_name,
            api_key=str(self.api_key),
            **self.deepseek_params,
        )

        model_info = deepseek_model.get_model_info() or {}
        generate_fn = deepseek_model.to_deepseek_model()

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=generate_fn,
            metadata={
                'model': {
                    **model_info,
                    'customOptions': to_json_schema(OpenAIConfig),
                },
            },
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """Generate a list of available DeepSeek models.

        Returns:
            list[ActionMetadata]: A list of ActionMetadata objects for each
                supported DeepSeek model, including name, metadata, and config schema.
        """
        actions_list = []
        for model, model_info in SUPPORTED_DEEPSEEK_MODELS.items():
            actions_list.append(
                model_action_metadata(
                    name=deepseek_name(model),
                    info=model_info.model_dump(by_alias=True, exclude_none=True),
                    config_schema=OpenAIConfig,
                )
            )

        return actions_list
