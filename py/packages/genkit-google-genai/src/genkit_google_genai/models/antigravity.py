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

"""Google AI Interactions Antigravity model action."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing_extensions import Never

from genkit import ModelRequest, ModelResponse
from genkit.plugin_api import Action, ActionKind, ActionRunContext, model_action_metadata
from genkit_google_genai._interactions.client import create_interaction
from genkit_google_genai._interactions.converters import (
    ensure_tool_ids,
    from_interaction_sync,
    split_system_instruction,
    to_interaction_steps,
)
from genkit_google_genai._interactions.options import ClientOptions
from genkit_google_genai.models.interactions_registry import antigravity_model_info
from genkit_google_genai.models.interactions_utils import (
    calculate_api_key,
    client_overrides_from_config,
    extract_version,
    partition_keys,
    remove_client_option_overrides,
    require_interaction_steps,
)

DEFAULT_ENVIRONMENT: dict[str, str] = {'type': 'remote'}

CREATE_OPTION_KEYS = (
    'previous_interaction_id',
    'store',
    'environment',
    'response_modalities',
)


class AntigravityConfig(BaseModel):
    """Antigravity model configuration."""

    # Per-request options arrive camelCased (apiKey).
    model_config = ConfigDict(extra='allow', populate_by_name=True, alias_generator=to_camel)
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    # Milliseconds — applied to the HTTP call, not the create body.
    timeout: float | None = None
    custom_headers: dict[str, str] | None = None
    previous_interaction_id: str | None = None
    store: bool | None = None
    environment: str | dict[str, Any] | None = None
    response_modalities: list[Literal['text', 'image']] | None = None


def create_antigravity_action(
    name: str,
    *,
    plugin_api_key: str | None,
    client_options: ClientOptions,
) -> Action[ModelRequest[AntigravityConfig], ModelResponse, Never]:
    """Build a foreground model action for Antigravity."""
    version = extract_version(name)
    info = antigravity_model_info(version)

    async def run(request: ModelRequest[AntigravityConfig], _: ActionRunContext) -> ModelResponse:
        config = request.config or AntigravityConfig()
        api_key = calculate_api_key(plugin_api_key, config.api_key)
        merged_options = client_options.merge(
            client_overrides_from_config(
                base_url=config.base_url,
                api_version=config.api_version,
                timeout=config.timeout,
                custom_headers=config.custom_headers,
            )
        )

        # Known create kwargs vs undocumented passthrough — non-mutating split.
        dumped = remove_client_option_overrides(config.model_dump(exclude_none=True))
        create_options, passthrough = partition_keys(dumped, CREATE_OPTION_KEYS)
        # Antigravity doesn't take system_instruction; fold system text into the
        # leading user turn so guidance still reaches the model.
        system_instruction, turns = split_system_instruction(request.messages or [])
        steps = to_interaction_steps(ensure_tool_ids(turns))
        if system_instruction:
            steps.insert(
                0,
                {
                    'type': 'user_input',
                    'content': [{'type': 'text', 'text': system_instruction}],
                },
            )
        require_interaction_steps(steps)
        create_kwargs: dict[str, Any] = {
            'agent': version,
            'input': steps,
            **create_options,
            **passthrough,
        }
        # Default missing environment to remote; the API rejects unsupported values.
        create_kwargs.setdefault('environment', DEFAULT_ENVIRONMENT)

        created = await create_interaction(api_key, create_kwargs, merged_options)
        return from_interaction_sync(created)

    return Action(
        kind=ActionKind.MODEL,
        name=name,
        fn=run,
        metadata=model_action_metadata(
            name=name,
            info=info.model_dump(by_alias=True),
            config_schema=AntigravityConfig,
        ).metadata,
    )
