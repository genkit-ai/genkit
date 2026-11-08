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

from collections.abc import Mapping
from typing import Any, Literal

from genkit_google_genai._interactions._client import create_interaction
from genkit_google_genai._interactions._converters import (
    ensure_tool_ids,
    from_interaction_sync,
    to_interaction_tool,
)
from genkit_google_genai._interactions._options import ClientOptions
from genkit_google_genai._models._interactions_registry import antigravity_model_info
from genkit_google_genai._models._interactions_utils import (
    INPUT_HINT,
    MODEL_ID_HINT,
    TIMEOUT_DESCRIPTION,
    api_key_for_context,
    client_overrides_from_config,
    extract_version,
    lowercase_choice_list,
    partition_keys,
    pop_extra,
    reject_managed_extra,
    remove_client_option_overrides,
    steps_with_folded_system_instruction,
)
from genkit_google_genai._models._sdk_config import check_timeout_seconds, deep_merge, reject_bool_timeout
from genkit_google_genai._models._secrets import reject_request_config_api_key
from pydantic import Field, field_validator

from genkit import GenkitError, Message, ModelResponse
from genkit.model import Model, ModelContext, ModelRequest
from genkit.plugin_api import GenkitConfig

DEFAULT_ENVIRONMENT: dict[str, str] = {'type': 'remote'}

CREATE_OPTION_KEYS = (
    'previous_interaction_id',
    'store',
    'environment',
    'response_modalities',
)

ANTIGRAVITY_EXTRA_DESCRIPTION = (
    "Interactions API fields this class doesn't declare, in the API's snake_case wire names "
    "(including `tools`, for example `[{'type': 'google_search'}, {'type': 'url_context'}]`). "
    'Merged into the top level of the create body after everything else, and nested objects merge key by key, '
    'so a key named here wins. Keys match ignoring case and underscores. Fields Genkit builds from the request '
    '(`agent` and `input`) and an empty `tools` list (`[]`) raise; `tools` from `extra` are sent ahead of '
    'any function tools passed to `generate(tools=...)`. A `None` value means unset and is skipped. '
    'Not validated, except that a top-level API key raises; do not put API keys here.'
)


class AntigravityConfig(GenkitConfig):
    """Antigravity model configuration.

    Unknown keys are rejected. An API field this class doesn't declare goes in
    ``extra``, which is deep-merged into the create body. Fields Genkit builds
    from the request (``agent``, ``input``) raise there.

    The agent enables ``google_search``, ``url_context``, and
    ``code_execution`` by default when ``tools`` is omitted on the wire. Client
    function tools go on ``generate(tools=[...])``. To restrict built-in tools
    or attach Interactions server tools (such as ``mcp_server``), pass wire
    tool dicts in ``extra={'tools': [...]}``; any ``generate(tools=[...])``
    function tools are appended after ``extra['tools']``.
    ``extra={'tools': []}`` raises ``INVALID_ARGUMENT`` because the API reads
    an empty list as unset and turns all three defaults back on.

    ``timeout`` is the HTTP timeout for this request, in seconds.

    Example:
        ```python
        # 1. Default: omit `extra['tools']` to keep search, URL context, and sandbox bash on
        response = await ai.generate(
            model=GoogleAI.antigravity('antigravity-preview-09-2026'),
            prompt='Reconcile invoice subtotal $224.49 with 8.25% tax.',
        )
        # => create body omits `tools`

        # 2. Escape hatch: restrict built-ins via `extra['tools']` and attach a Genkit function tool
        response = await ai.generate(
            model=GoogleAI.antigravity('antigravity-preview-09-2026'),
            prompt='Cross-check the allergen policy at three Austin ramen shops against our menu.',
            tools=[lookup_supplier_allergens],
            config=AntigravityConfig(
                extra={'tools': [{'type': 'google_search'}, {'type': 'url_context'}]},
            ),
        )
        # => create body tools: [{'type': 'google_search'}, {'type': 'url_context'},
        #                        {'type': 'function', 'name': 'lookup_supplier_allergens', ...}]
        ```
    """

    base_url: str | None = None
    api_version: str | None = None
    timeout: float | None = Field(default=None, description=TIMEOUT_DESCRIPTION)
    custom_headers: dict[str, str] | None = None
    previous_interaction_id: str | None = None
    store: bool | None = None
    environment: str | dict[str, Any] | None = None
    response_modalities: list[Literal['text', 'image']] | None = None
    extra: dict[str, Any] | None = Field(default=None, description=ANTIGRAVITY_EXTRA_DESCRIPTION)

    @field_validator('timeout', mode='before')
    @classmethod
    def no_bool_timeout(cls, value: object) -> object:
        """``True`` is not a number of seconds."""
        return reject_bool_timeout(value)

    @field_validator('timeout')
    @classmethod
    def check_timeout(cls, value: float | None) -> float | None:
        """Reject 0, negative, inf and NaN; ``timeout`` is seconds."""
        return check_timeout_seconds(value)

    @field_validator('response_modalities', mode='before')
    @classmethod
    def fold_modalities_case(cls, value: object) -> object:
        """Accept TEXT/IMAGE the same as lowercase wire values."""
        return lowercase_choice_list(value)


# Fields Genkit builds from the request; extra can't replace them.
ANTIGRAVITY_MANAGED_EXTRA = {
    'agent': MODEL_ID_HINT,
    'input': INPUT_HINT,
}


def pop_antigravity_extra_tools(name: str, extra: dict[str, Any]) -> list[Any]:
    """Pop and validate ``extra['tools']`` before merging ``extra`` onto the create body.

    The Interactions API reads ``tools: []`` as unset on the wire and turns all
    three built-in tools back on, so an empty list fails closed instead of
    looking like a sandbox lockdown. Popping ``tools`` here also ensures
    ``deep_merge`` does not overwrite ``request.tools`` appended onto the
    create body.
    """
    tools_from_extra: list[Any] = []
    for key in list(extra):
        if key.replace('_', '').lower() != 'tools':
            continue
        value = extra.pop(key)
        if not isinstance(value, list):
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{name}: extra[{key!r}] must be a non-empty list of Interactions tool dicts.',
            )
        if not value:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f'{name}: extra[{key!r}] cannot be empty ([]). '
                    'The Interactions API reads an empty tools list as unset and '
                    'turns google_search, url_context, and code_execution back on; '
                    'pass at least one tool or omit tools to keep the defaults.'
                ),
            )
        tools_from_extra.extend(value)
    return tools_from_extra


def select_antigravity_continuation(
    messages: list[Message] | None,
    create_options: dict[str, Any],
) -> tuple[list[Message], str | None, str | dict[str, Any]]:
    """Slice multi-turn history to continue the prior stored Interaction and sandbox.

    Antigravity rejects ``function_call`` / ``code_execution_call`` steps
    replayed in ``input``, and a fresh ``remote`` sandbox would lose files from
    prior turns. Continue from the last ``model`` message carrying
    ``metadata['interactionId']``, sending only newer messages and reusing its
    ``environmentId`` unless the caller pinned a specific environment.
    """
    paired = ensure_tool_ids(messages or [])
    configured_prev_id: str | None = create_options.get('previous_interaction_id')
    configured_env: str | dict[str, Any] | None = create_options.get('environment')

    previous_id = configured_prev_id
    new_messages = paired
    environment: str | dict[str, Any] = configured_env if configured_env is not None else dict(DEFAULT_ENVIRONMENT)

    if not previous_id:
        for idx in range(len(paired) - 1, -1, -1):
            msg = paired[idx]
            meta = msg.metadata if isinstance(msg.metadata, Mapping) else None
            prev_id = meta.get('interactionId') if meta else None
            if msg.role == 'model' and isinstance(prev_id, str) and prev_id:
                previous_id = prev_id
                new_messages = paired[idx + 1 :]
                prev_env = meta.get('environmentId') if meta else None
                if configured_env in (None, 'remote', DEFAULT_ENVIRONMENT) and isinstance(prev_env, str) and prev_env:
                    environment = prev_env
                break
    else:
        # Explicit previous_interaction_id wins; server already holds history through the last model turn.
        for idx in range(len(paired) - 1, -1, -1):
            if paired[idx].role == 'model':
                new_messages = paired[idx + 1 :]
                break

    return new_messages, previous_id, environment


def create_antigravity_action(
    name: str,
    *,
    plugin_api_key: str | None,
    client_options: ClientOptions,
) -> Model:
    """Build a foreground model action for Antigravity."""
    version = extract_version(name)
    info = antigravity_model_info(version)

    async def run(request: ModelRequest[AntigravityConfig], ctx: ModelContext) -> ModelResponse:
        reject_request_config_api_key(request.config)
        config = request.config or AntigravityConfig()
        api_key = api_key_for_context(ctx.context, plugin_api_key)
        merged_options = client_options.merge(client_overrides_from_config(config))

        dumped = remove_client_option_overrides(config.model_dump(exclude_none=True))
        extra = pop_extra(dumped)
        reject_managed_extra(name, extra, ANTIGRAVITY_MANAGED_EXTRA)
        extra_tools = pop_antigravity_extra_tools(name, extra)
        # GenkitConfig rejects undeclared keys, so nothing is left over.
        create_options, _ = partition_keys(dumped, CREATE_OPTION_KEYS)
        new_messages, previous_id, environment = select_antigravity_continuation(
            request.messages,
            create_options,
        )

        # Built-in / server tools from extra['tools'] first, followed by client function tools.
        tools = [
            *extra_tools,
            *(to_interaction_tool(tool) for tool in (request.tools or [])),
        ]

        create_kwargs: dict[str, Any] = {
            **create_options,
            'agent': version,
            'input': steps_with_folded_system_instruction(new_messages),
            'environment': environment,
        }
        if previous_id:
            create_kwargs['previous_interaction_id'] = previous_id
        if tools:
            create_kwargs['tools'] = tools

        created = await create_interaction(api_key, deep_merge(create_kwargs, extra), merged_options)
        return from_interaction_sync(created)

    return Model(
        name,
        run,
        info=info,
        config_schema=AntigravityConfig,
    )
