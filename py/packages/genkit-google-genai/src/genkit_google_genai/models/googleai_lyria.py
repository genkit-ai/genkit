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

"""Google AI Interactions Lyria audio model (not the legacy Vertex Lyria)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from genkit import ModelInfo, ModelRequest, ModelResponse, Supports
from genkit.plugin_api import Action, ActionKind, ActionRunContext, model_action_metadata
from genkit_google_genai._interactions.client import InteractionsClient
from genkit_google_genai._interactions.converters import (
    ensure_tool_ids,
    from_interaction_sync,
    to_interaction_steps,
)
from genkit_google_genai._interactions.types import ClientOptions, CreateInteractionRequest, ResponseModality
from genkit_google_genai.models.interactions_utils import (
    calculate_api_key,
    config_as_dict,
    extract_version,
    merge_client_options,
    remove_client_option_overrides,
    response_modalities_from_config,
)

GENERIC_GOOGLEAI_LYRIA_INFO = ModelInfo(
    label='Google AI - lyria-3',
    supports=Supports(
        multiturn=False,
        media=True,
        tools=False,
        tool_choice=False,
        system_role=False,
        output=['media', 'text'],
    ),
)

KNOWN_GOOGLEAI_LYRIA_MODELS: dict[str, ModelInfo] = {
    'lyria-3-clip-preview': GENERIC_GOOGLEAI_LYRIA_INFO,
    'lyria-3-pro-preview': GENERIC_GOOGLEAI_LYRIA_INFO,
}


class GoogleAILyriaConfigSchema(BaseModel):
    """Google AI Interactions Lyria model configuration."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    api_key: str | None = Field(default=None, alias='apiKey')
    base_url: str | None = Field(default=None, alias='baseUrl')
    api_version: str | None = Field(default=None, alias='apiVersion')
    response_modalities: list[Literal['TEXT', 'IMAGE', 'AUDIO']] | None = Field(
        default=None,
        alias='responseModalities',
    )


def is_googleai_lyria_model_name(name: str | None) -> bool:
    """Return True for Google AI Interactions Lyria models (lyria-3-*)."""
    return bool(name and name.startswith('lyria-3'))


def googleai_lyria_model_info(version: str) -> ModelInfo:
    """Return capability metadata for an Interactions Lyria model."""
    known = KNOWN_GOOGLEAI_LYRIA_MODELS.get(version)
    if known is not None:
        return ModelInfo(label=f'Google AI - {version}', supports=known.supports)
    return ModelInfo(label=f'Google AI - {version}', supports=GENERIC_GOOGLEAI_LYRIA_INFO.supports)


def list_known_googleai_lyria_models() -> list[str]:
    """Return statically known Interactions Lyria model names."""
    return list(KNOWN_GOOGLEAI_LYRIA_MODELS.keys())


class GoogleAILyriaModel:
    """Interactions Lyria model for Google AI."""

    def __init__(
        self,
        version: str,
        *,
        plugin_api_key: str | None,
        client_options: ClientOptions,
    ) -> None:
        """Initialize Interactions Lyria model."""
        self._version = version
        self._plugin_api_key = plugin_api_key
        self._client_options = client_options

    async def generate(self, request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        """Run a synchronous Interactions Lyria generation."""
        config = config_as_dict(request.config)
        api_key = calculate_api_key(
            self._plugin_api_key,
            config.get('apiKey') or config.get('api_key'),
        )
        client_options = merge_client_options(self._client_options, config)
        passthrough = remove_client_option_overrides(config)
        passthrough.pop('responseModalities', None)
        passthrough.pop('response_modalities', None)

        messages = request.messages or []
        default_modalities: list[ResponseModality] = ['audio', 'text']
        req: CreateInteractionRequest = {
            'model': extract_version(self._version),
            'input': to_interaction_steps(ensure_tool_ids(messages)),
            'response_modalities': response_modalities_from_config(config, default=default_modalities)
            or default_modalities,
            **passthrough,
        }

        client = InteractionsClient(api_key=api_key, client_options=client_options)
        try:
            interaction = await client.create_interaction(req)
        finally:
            await client.aclose()
        return from_interaction_sync(interaction)


def create_googleai_lyria_action(
    name: str,
    *,
    plugin_api_key: str | None,
    client_options: ClientOptions,
) -> Action:
    """Build a foreground model action for Interactions Lyria."""
    clean_name = extract_version(name)
    model = GoogleAILyriaModel(clean_name, plugin_api_key=plugin_api_key, client_options=client_options)
    info = googleai_lyria_model_info(clean_name)

    async def _run(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return await model.generate(request, ctx)

    return Action(
        kind=ActionKind.MODEL,
        name=name,
        fn=_run,
        metadata=model_action_metadata(
            name=name,
            info=info.model_dump(by_alias=True),
            config_schema=GoogleAILyriaConfigSchema,
        ).metadata,
    )
