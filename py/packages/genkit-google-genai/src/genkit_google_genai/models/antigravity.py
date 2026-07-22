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

"""Antigravity foreground model via the Google AI Interactions API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from genkit import ModelInfo, ModelRequest, ModelResponse, Supports
from genkit.plugin_api import Action, ActionKind, ActionRunContext, model_action_metadata
from genkit_google_genai._interactions.client import InteractionsClient
from genkit_google_genai._interactions.converters import (
    ensure_tool_ids,
    from_interaction_sync,
    to_interaction_steps,
)
from genkit_google_genai._interactions.types import ClientOptions, CreateInteractionRequest
from genkit_google_genai.models.interactions_utils import (
    calculate_api_key,
    config_as_dict,
    downgrade_system_messages,
    extract_version,
    merge_client_options,
    remove_client_option_overrides,
    response_modalities_from_config,
)

GENERIC_ANTIGRAVITY_INFO = ModelInfo(
    label='Google AI - antigravity',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=False,
        tool_choice=False,
        system_role=False,
        output=['text'],
    ),
)

KNOWN_ANTIGRAVITY_MODELS: dict[str, ModelInfo] = {
    'antigravity-preview-05-2026': GENERIC_ANTIGRAVITY_INFO,
}


class AntigravityConfigSchema(BaseModel):
    """Antigravity model configuration."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    api_key: str | None = Field(default=None, alias='apiKey')
    base_url: str | None = Field(default=None, alias='baseUrl')
    api_version: str | None = Field(default=None, alias='apiVersion')
    previous_interaction_id: str | None = Field(default=None, alias='previousInteractionId')
    store: bool | None = None
    environment: str | dict[str, Any] | None = None
    response_modalities: list[Literal['TEXT', 'IMAGE']] | None = Field(
        default=None,
        alias='responseModalities',
    )


def is_antigravity_model_name(name: str | None) -> bool:
    """Return True when the model name belongs to the Antigravity family."""
    return bool(name and name.startswith('antigravity-'))


def antigravity_model_info(version: str) -> ModelInfo:
    """Return capability metadata for an Antigravity model."""
    known = KNOWN_ANTIGRAVITY_MODELS.get(version)
    if known is not None:
        return ModelInfo(label=f'Google AI - {version}', supports=known.supports)
    return ModelInfo(label=f'Google AI - {version}', supports=GENERIC_ANTIGRAVITY_INFO.supports)


def list_known_antigravity_models() -> list[str]:
    """Return statically known Antigravity model names."""
    return list(KNOWN_ANTIGRAVITY_MODELS.keys())


class AntigravityModel:
    """Antigravity model backed by the Interactions API."""

    def __init__(
        self,
        version: str,
        *,
        plugin_api_key: str | None,
        client_options: ClientOptions,
    ) -> None:
        """Initialize Antigravity model."""
        self._version = version
        self._plugin_api_key = plugin_api_key
        self._client_options = client_options

    async def generate(self, request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        """Run a synchronous Antigravity interaction."""
        config = config_as_dict(request.config)
        api_key = calculate_api_key(
            self._plugin_api_key,
            config.get('apiKey') or config.get('api_key'),
        )
        client_options = merge_client_options(self._client_options, config)
        request_options = remove_client_option_overrides(config)

        previous_interaction_id = request_options.pop('previousInteractionId', None) or request_options.pop(
            'previous_interaction_id', None
        )
        store = request_options.pop('store', None)
        environment = request_options.pop('environment', None)
        request_options.pop('responseModalities', None)
        request_options.pop('response_modalities', None)

        messages = downgrade_system_messages(request.messages or [])
        req: CreateInteractionRequest = {
            'agent': extract_version(self._version),
            'input': to_interaction_steps(ensure_tool_ids(messages)),
            **({'previous_interaction_id': previous_interaction_id} if previous_interaction_id else {}),
            **({'store': store} if store is not None else {}),
            **({'environment': environment} if environment is not None else {}),
            **request_options,
        }

        response_modalities = response_modalities_from_config(config)
        if response_modalities is not None:
            req['response_modalities'] = response_modalities

        client = InteractionsClient(api_key=api_key, client_options=client_options)
        try:
            interaction = await client.create_interaction(req)
        finally:
            await client.aclose()
        return from_interaction_sync(interaction)


def create_antigravity_action(
    name: str,
    *,
    plugin_api_key: str | None,
    client_options: ClientOptions,
) -> Action:
    """Build a foreground model action for Antigravity."""
    clean_name = extract_version(name)
    model = AntigravityModel(clean_name, plugin_api_key=plugin_api_key, client_options=client_options)
    info = antigravity_model_info(clean_name)

    async def _run(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return await model.generate(request, ctx)

    return Action(
        kind=ActionKind.MODEL,
        name=name,
        fn=_run,
        metadata=model_action_metadata(
            name=name,
            info=info.model_dump(by_alias=True),
            config_schema=AntigravityConfigSchema,
        ).metadata,
    )
