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

"""xAI plugin for Genkit."""

import os
from typing import Any, cast

from xai_sdk import Client as XAIClient

from genkit.ai import Plugin
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action
from genkit.core.error import GenkitError
from genkit.core.registry import ActionKind
from genkit.core.schema import to_json_schema
from genkit.plugins.xai.model_info import SUPPORTED_XAI_MODELS, get_model_info
from genkit.plugins.xai.models import XAIConfig, XAIModel

__all__ = ['XAI', 'xai_name']

XAI_PLUGIN_NAME = 'xai'


def xai_name(name: str) -> str:
    """Create a fully qualified xAI model name."""
    return f'{XAI_PLUGIN_NAME}/{name}'


class XAI(Plugin):
    """xAI plugin for Genkit."""

    name = XAI_PLUGIN_NAME

    def __init__(
        self,
        api_key: str | None = None,
        models: list[str] | None = None,
        **xai_params: object,
    ) -> None:
        """Initialize the XAI plugin.

        Args:
            api_key: The xAI API key.
            models: List of models to register.
            xai_params: Additional parameters for xAI client.
        """
        api_key = api_key or os.getenv('XAI_API_KEY')

        if not api_key:
            raise GenkitError(message='Please provide api_key or set XAI_API_KEY environment variable.')

        self.models = models or list(SUPPORTED_XAI_MODELS.keys())
        self._xai_params = xai_params
        self._xai_client = XAIClient(api_key=api_key, **cast(dict[str, Any], xai_params))

    async def init(self) -> list:
        """Initialize plugin.

        Returns:
            List of Action objects for pre-configured models.
        """
        actions = []

        # Register pre-configured models
        for model_name in self.models:
            name = xai_name(model_name)
            action = self._create_model_action(name)
            actions.append(action)

        return actions

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
        """Create an Action object for an XAI model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace(f'{XAI_PLUGIN_NAME}/', '') if name.startswith(XAI_PLUGIN_NAME) else name

        model = XAIModel(model_name=clean_name, client=self._xai_client)
        model_info = get_model_info(clean_name)

        model_metadata = model_info.model_dump(by_alias=True, exclude_none=True)
        model_metadata['customOptions'] = to_json_schema(XAIConfig)

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=model.generate,
            metadata={'model': model_metadata},
        )

    async def list_actions(self) -> list:
        """List available XAI models.

        Returns:
            List of ActionMetadata for all supported models.
        """
        actions = []
        for model_name, model_info in SUPPORTED_XAI_MODELS.items():
            actions.append(
                model_action_metadata(
                    name=xai_name(model_name),
                    info=model_info.model_dump(by_alias=True, exclude_none=True),
                    config_schema=XAIConfig,
                )
            )
        return actions
