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

"""Google AI Interactions Deep Research background model action."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit import ModelInfo, ModelRequest
from genkit._core._background import define_background_model
from genkit._core._registry import Registry
from genkit.model import BackgroundAction, ModelRef, Operation, model_ref
from genkit.plugin_api import ActionRunContext
from genkit_google_genai._interactions.client import (
    cancel_interaction,
    create_interaction,
    get_interaction,
)
from genkit_google_genai._interactions.converters import (
    clean_schema,
    ensure_tool_ids,
    from_interaction,
    split_system_instruction,
    to_interaction_steps,
    to_interaction_tool,
)
from genkit_google_genai._interactions.options import ClientOptions
from genkit_google_genai.models.interactions_registry import (
    DEEP_RESEARCH_INFO,
    KNOWN_DEEP_RESEARCH_MODELS,
    deep_research_model_info,
)
from genkit_google_genai.models.interactions_utils import (
    calculate_api_key,
    client_options_for_operation,
    client_overrides_from_config,
    extract_version,
    partition_keys,
    remove_client_option_overrides,
    require_interaction_steps,
)

AGENT_CONFIG_KEYS = (
    'thinking_summaries',
    'visualization',
    'collaborative_planning',
)
TOOL_CONFIG_KEYS = (
    'google_search',
    'url_context',
    'code_execution',
    'file_search',
    'mcp_servers',
)
CREATE_OPTION_KEYS = (
    'previous_interaction_id',
    'store',
    'response_modalities',
)


class McpServerConfig(BaseModel):
    """MCP server configuration for Deep Research."""

    # Request config shows up camelCased on the wire (allowedTools, …).
    model_config = ConfigDict(extra='allow', populate_by_name=True, alias_generator=to_camel)
    name: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: list[str] | None = None


class FileSearchConfig(BaseModel):
    """File search store configuration for Deep Research."""

    model_config = ConfigDict(extra='allow', populate_by_name=True, alias_generator=to_camel)
    file_search_store_names: list[str]


class DeepResearchConfig(BaseModel):
    """Deep Research model configuration."""

    # Per-request options arrive camelCased (apiKey). Without accepting those
    # names, the override silently disappears before check/cancel can reuse it.
    model_config = ConfigDict(extra='allow', populate_by_name=True, alias_generator=to_camel)
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    # Milliseconds — applied to the HTTP call, not the create body.
    timeout: float | None = None
    custom_headers: dict[str, str] | None = None
    thinking_summaries: Literal['auto', 'none'] | None = None
    previous_interaction_id: str | None = None
    store: bool | None = None
    response_modalities: list[Literal['text', 'image', 'audio']] | None = None
    visualization: Literal['auto', 'off'] | None = None
    collaborative_planning: bool | None = None
    google_search: bool | None = None
    url_context: bool | None = None
    code_execution: bool | None = None
    file_search: FileSearchConfig | None = None
    mcp_servers: list[McpServerConfig] | None = None


def deep_research_model(version: str) -> ModelRef:
    """Return a ModelRef for a Deep Research version (namespaced or bare)."""
    clean = extract_version(version)
    info = KNOWN_DEEP_RESEARCH_MODELS.get(clean) or DEEP_RESEARCH_INFO
    # Prefer a version-specific label over the shared catalog default.
    if info.label is None or info.label == DEEP_RESEARCH_INFO.label:
        info = ModelInfo(label=f'Google AI - {clean}', supports=info.supports)
    return model_ref(
        name=clean,
        namespace='googleai',
        info=info,
        config_schema=DeepResearchConfig,
    )


def build_tools(request: ModelRequest[DeepResearchConfig], config: DeepResearchConfig) -> list[dict[str, Any]]:
    """Build Interactions API tool configurations for Deep Research."""
    tools: list[dict[str, Any]] = []
    if request.tools:
        tools.extend(dict(to_interaction_tool(tool_def)) for tool_def in request.tools)

    if config.google_search:
        tools.append({'type': 'google_search'})
    if config.url_context:
        tools.append({'type': 'url_context'})
    if config.code_execution:
        tools.append({'type': 'code_execution'})
    if config.file_search is not None:
        tools.append({'type': 'file_search', **config.file_search.model_dump(exclude_none=True)})
    for mcp_server in config.mcp_servers or []:
        tools.append({'type': 'mcp_server', **mcp_server.model_dump(exclude_none=True)})

    return tools


def response_format_from_request(
    request: ModelRequest[DeepResearchConfig],
) -> dict[str, Any] | None:
    """Build response_format when the caller asked for JSON output."""
    if request.output_format != 'json' and request.output_content_type != 'application/json':
        return None
    response_format: dict[str, Any] = {'type': 'text', 'mime_type': 'application/json'}
    if request.output_schema:
        response_format['schema'] = clean_schema(request.output_schema)
    return response_format


def api_key_from_operation(
    stored: ClientOptions,
    *,
    plugin_api_key: str | None,
) -> str:
    """Resolve the api key for check/cancel.

    Prefer the key persisted on the operation at start. If metadata omitted it,
    calculate_api_key falls through plugin → env (and errors if nothing is set).
    """
    return stored.api_key or calculate_api_key(plugin_api_key, None)


def create_deep_research_background_action(
    target: str | ModelRef,
    *,
    plugin_api_key: str | None,
    client_options: ClientOptions,
) -> BackgroundAction:
    """Wire Deep Research Interactions start/check/cancel through define_background_model."""
    name = target.name if isinstance(target, ModelRef) else target
    version = extract_version(name)
    info = deep_research_model_info(version)

    async def start(request: ModelRequest[DeepResearchConfig], _: ActionRunContext) -> Operation:
        config = request.config or DeepResearchConfig()
        api_key = calculate_api_key(plugin_api_key, config.api_key)
        options = client_options.merge(
            client_overrides_from_config(
                base_url=config.base_url,
                api_version=config.api_version,
                timeout=config.timeout,
                custom_headers=config.custom_headers,
            )
        )

        # Map dumped config into the buckets interactions.create expects without
        # mutating the dump: agent fields, tool flags (via build_tools), create
        # options, then undocumented passthrough leftovers.
        dumped = remove_client_option_overrides(config.model_dump(exclude_none=True))
        agent_fields, _tool_fields, create_options, passthrough = partition_keys(
            dumped,
            AGENT_CONFIG_KEYS,
            TOOL_CONFIG_KEYS,
            CREATE_OPTION_KEYS,
        )
        agent_config: dict[str, Any] = {'type': 'deep-research', **agent_fields}

        tools = build_tools(request, config)
        response_format = response_format_from_request(request)
        # Deep Research rejects system_instruction and asks for guidance in the
        # input prompt instead, so system turns become a leading user_input step.
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
            'background': True,
            'agent_config': agent_config,
            **create_options,
            **passthrough,
        }
        if tools:
            create_kwargs['tools'] = tools
        if response_format is not None:
            create_kwargs['response_format'] = response_format

        interaction = await create_interaction(api_key, create_kwargs, options)
        return from_interaction(
            interaction,
            client_options_for_operation(options, api_key=api_key),
        )

    async def follow_up(operation: Operation, *, cancel: bool = False) -> Operation:
        stored = ClientOptions.from_metadata(operation.metadata)
        options = client_options.merge(stored)
        api_key = api_key_from_operation(stored, plugin_api_key=plugin_api_key)
        if cancel:
            interaction = await cancel_interaction(api_key, operation.id, options)
        else:
            interaction = await get_interaction(api_key, operation.id, options)
        return from_interaction(
            interaction,
            client_options_for_operation(options, api_key=api_key),
        )

    async def check(operation: Operation) -> Operation:
        return await follow_up(operation)

    async def cancel(operation: Operation) -> Operation:
        return await follow_up(operation, cancel=True)

    # Throwaway registry: plugin init re-registers the returned actions on the app registry.
    # define_background_model stamps Operation.action so check_operation/cancel_operation work.
    return define_background_model(
        registry=Registry(),
        name=name,
        start=start,
        check=check,
        cancel=cancel,
        label=info.label or name,
        info=info,
        config_schema=DeepResearchConfig,
    )
