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

import structlog
from anthropic import AsyncAnthropic

from genkit import ModelConfig, ModelRequest, ModelResponse
from genkit.model import model_action_metadata
from genkit.plugin_api import (
    Action,
    ActionKind,
    ActionMetadata,
    ActionRunContext,
    Plugin,
    loop_local_client,
    to_json_schema,
)
from genkit_anthropic.model_info import SUPPORTED_ANTHROPIC_MODELS, get_model_info
from genkit_anthropic.models import AnthropicModel

logger = structlog.get_logger(__name__)

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
        self._runtime_client = loop_local_client(lambda: AsyncAnthropic(**cast(dict[str, Any], self._anthropic_params)))
        self._list_actions_cache: list[ActionMetadata] | None = None

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

        model_info = get_model_info(clean_name)

        async def _generate(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            model = AnthropicModel(model_name=clean_name, client=self._runtime_client())
            return await model.generate(request, ctx)

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=_generate,
            metadata={
                'model': {
                    'supports': (
                        model_info.supports.model_dump(by_alias=True, exclude_none=True) if model_info.supports else {}
                    ),
                    'customOptions': to_json_schema(ModelConfig),
                },
            },
        )

    def _model_metadata(self, model_id: str) -> ActionMetadata:
        """Build ActionMetadata for a single (bare, unprefixed) Anthropic model id.

        Args:
            model_id: Bare model id (no ``anthropic/`` prefix).

        Returns:
            ActionMetadata for the model, using curated info if known, else a
            generic fallback.
        """
        return model_action_metadata(
            name=anthropic_name(model_id),
            info=get_model_info(model_id).model_dump(by_alias=True, exclude_none=True),
            config_schema=ModelConfig,
        )

    async def _fetch_dynamic_model_ids(self) -> list[str]:
        """Fetch all available model ids from the Anthropic API.

        Uses the beta models endpoint (matching JS, so both stable and beta
        models are discovered) and fully paginates via ``async for`` (matching
        Go's completeness, rather than JS's first-page-only read).

        Returns:
            Model ids in API order.
        """
        model_ids: list[str] = []
        async for model in self._runtime_client().beta.models.list():
            if model.id:
                model_ids.append(model.id)
        return model_ids

    async def list_actions(self) -> list[ActionMetadata]:
        """List available Anthropic models.

        Queries the Anthropic API for currently available models and returns
        the union of API-discovered models and any statically known models
        not already covered by the API response (API ids first, in API
        order, then remaining static ids, deduplicated by bare id). The
        successful result is cached for the lifetime of the plugin instance.
        If the API call fails, logs a warning and returns the static model
        list only, without caching the failure, so the next call retries the
        API.

        Returns:
            List of ActionMetadata for all discovered/supported models.
        """
        if self._list_actions_cache is not None:
            return self._list_actions_cache

        try:
            model_ids = await self._fetch_dynamic_model_ids()
        except Exception as e:
            logger.warning('Failed to list Anthropic models from API, using static model list', error=str(e))
            return [self._model_metadata(model_id) for model_id in SUPPORTED_ANTHROPIC_MODELS]

        seen: set[str] = set()
        ordered_ids: list[str] = []
        for model_id in model_ids:
            if model_id and model_id not in seen:
                seen.add(model_id)
                ordered_ids.append(model_id)
        for model_id in SUPPORTED_ANTHROPIC_MODELS:
            if model_id not in seen:
                seen.add(model_id)
                ordered_ids.append(model_id)

        actions = [self._model_metadata(model_id) for model_id in ordered_ids]
        self._list_actions_cache = actions
        return actions
