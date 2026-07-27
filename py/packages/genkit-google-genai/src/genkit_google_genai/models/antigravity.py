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

from collections.abc import Callable
from typing import Any, Literal, cast

from google import genai
from google.genai.interactions import Interaction
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing_extensions import Never

from genkit import GenkitError, ModelRequest, ModelResponse
from genkit.plugin_api import Action, ActionKind, ActionRunContext, model_action_metadata
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
    extract_version,
    map_genai_error,
    remove_client_option_overrides,
    require_interaction_steps,
    resolve_interactions_client,
    take_keys,
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
    previous_interaction_id: str | None = None
    store: bool | None = None
    environment: str | dict[str, Any] | None = None
    response_modalities: list[Literal['text', 'image']] | None = None


def _is_remote_environment(environment: object) -> bool:
    """Return True when environment is the remote sandbox Antigravity accepts."""
    if environment == 'remote':
        return True
    if not isinstance(environment, dict):
        return False
    env = cast(dict[str, Any], environment)
    return env.get('type') == 'remote'


def create_antigravity_action(
    name: str,
    *,
    plugin_api_key: str | None,
    client_options: ClientOptions,
    client_getter: Callable[[], genai.Client] | None = None,
) -> Action[ModelRequest[AntigravityConfig], ModelResponse, Never]:
    """Build a foreground model action for Antigravity."""
    version = extract_version(name)
    info = antigravity_model_info(version)

    async def _run(request: ModelRequest[AntigravityConfig], _: ActionRunContext) -> ModelResponse:
        config = request.config or AntigravityConfig()
        request_api_key = config.api_key
        api_key = calculate_api_key(plugin_api_key, request_api_key)
        merged_options = client_options.merge(
            ClientOptions(
                base_url=config.base_url,
                api_version=config.api_version,
            )
        )

        # Peel known create kwargs (previous_interaction_id, store, …) out of the
        # dumped config; anything left in wire is an undocumented passthrough.
        wire = remove_client_option_overrides(config.model_dump(exclude_none=True))
        create_options = take_keys(wire, CREATE_OPTION_KEYS)
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
            **wire,
        }
        # Default after merge so a missing environment still becomes remote, and
        # a caller override can't silently clobber it into something the API rejects.
        create_kwargs.setdefault('environment', DEFAULT_ENVIRONMENT)
        if not _is_remote_environment(create_kwargs.get('environment')):
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message="Antigravity only supports environment {'type': 'remote'}.",
            )

        async with resolve_interactions_client(
            client_getter=client_getter,
            plugin_api_key=plugin_api_key,
            api_key=api_key,
            request_api_key=request_api_key,
            plugin_client_options=client_options,
            client_options=merged_options,
        ) as client:
            try:
                created = await client.aio.interactions.create(**create_kwargs)
            except Exception as error:
                raise map_genai_error(error) from error
        if not isinstance(created, Interaction):
            raise GenkitError(
                status='INTERNAL',
                message='Expected a non-streaming Interaction response from Antigravity',
            )
        return from_interaction_sync(created)

    return Action(
        kind=ActionKind.MODEL,
        name=name,
        fn=_run,
        metadata=model_action_metadata(
            name=name,
            info=info.model_dump(by_alias=True),
            config_schema=AntigravityConfig,
        ).metadata,
    )
