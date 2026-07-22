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

"""Deep Research background model via the Google AI Interactions API."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from genkit import ModelInfo, ModelRequest, Supports
from genkit._core._background import define_background_model
from genkit._core._registry import Registry
from genkit.model import BackgroundAction, ModelRef, Operation, model_ref
from genkit.plugin_api import ActionRunContext
from genkit_google_genai._interactions.client import InteractionsClient
from genkit_google_genai._interactions.converters import (
    clean_schema,
    ensure_tool_ids,
    from_interaction,
    to_interaction_steps,
    to_interaction_tool,
)
from genkit_google_genai._interactions.types import (
    ClientOptions,
    CreateInteractionRequest,
    DeepResearchAgentConfig,
    InteractionTool,
)
from genkit_google_genai.models.interactions_utils import (
    calculate_api_key,
    client_options_for_operation,
    config_as_dict,
    downgrade_system_messages,
    extract_version,
    merge_client_options,
    remove_client_option_overrides,
    response_modalities_from_config,
)

GENERIC_DEEP_RESEARCH_INFO = ModelInfo(
    label='Google AI - deep-research',
    supports=Supports(
        multiturn=True,
        media=False,
        tools=False,
        tool_choice=False,
        system_role=False,
        output=['text'],
        long_running=True,
    ),
)

ADVANCED_DEEP_RESEARCH_INFO = ModelInfo(
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=False,
        system_role=False,
        output=['text', 'media'],
        long_running=True,
    ),
)


class McpServerConfig(BaseModel):
    """MCP server configuration for Deep Research."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    name: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: list[str] | None = Field(default=None, alias='allowedTools')


class FileSearchConfig(BaseModel):
    """File search store configuration for Deep Research."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    file_search_store_names: list[str] = Field(alias='fileSearchStoreNames')


class DeepResearchConfigSchema(BaseModel):
    """Deep Research model configuration."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    api_key: str | None = Field(default=None, alias='apiKey')
    base_url: str | None = Field(default=None, alias='baseUrl')
    api_version: str | None = Field(default=None, alias='apiVersion')
    thinking_summaries: Literal['AUTO', 'NONE'] | None = Field(default=None, alias='thinkingSummaries')
    previous_interaction_id: str | None = Field(default=None, alias='previousInteractionId')
    store: bool | None = None
    response_modalities: list[Literal['TEXT', 'IMAGE', 'AUDIO']] | None = Field(
        default=None,
        alias='responseModalities',
    )
    visualization: Literal['AUTO', 'OFF'] | None = None
    collaborative_planning: bool | None = Field(default=None, alias='collaborativePlanning')
    google_search: bool | dict[str, Any] | None = Field(default=None, alias='googleSearch')
    url_context: bool | dict[str, Any] | None = Field(default=None, alias='urlContext')
    code_execution: bool | dict[str, Any] | None = Field(default=None, alias='codeExecution')
    file_search: FileSearchConfig | dict[str, Any] | None = Field(default=None, alias='fileSearch')
    mcp_servers: list[McpServerConfig | dict[str, Any]] | None = Field(default=None, alias='mcpServers')


def _common_ref(name: str, info: ModelInfo | None = None) -> ModelRef:
    """Build a googleai/ ModelRef for a Deep Research version."""
    resolved = info or GENERIC_DEEP_RESEARCH_INFO
    # Prefer a version-specific label when callers pass ADVANCED/GENERIC without one.
    if resolved.label is None or resolved.label == GENERIC_DEEP_RESEARCH_INFO.label:
        resolved = ModelInfo(label=f'Google AI - {name}', supports=resolved.supports)
    return model_ref(
        name=name,
        namespace='googleai',
        info=resolved,
    ).model_copy(update={'config_schema': DeepResearchConfigSchema})


KNOWN_DEEP_RESEARCH_MODELS: dict[str, ModelRef] = {
    'deep-research-pro-preview-12-2025': _common_ref('deep-research-pro-preview-12-2025'),
    'deep-research-preview-04-2026': _common_ref(
        'deep-research-preview-04-2026',
        ADVANCED_DEEP_RESEARCH_INFO,
    ),
    'deep-research-max-preview-04-2026': _common_ref(
        'deep-research-max-preview-04-2026',
        ADVANCED_DEEP_RESEARCH_INFO,
    ),
}


def is_deep_research_model_name(name: str | None) -> bool:
    """Return True when the model name belongs to the Deep Research family."""
    return bool(name and name.startswith('deep-research-'))


def deep_research_model(version: str) -> ModelRef:
    """Return a ModelRef for a Deep Research version (namespaced or bare)."""
    clean = extract_version(version)
    known = KNOWN_DEEP_RESEARCH_MODELS.get(clean)
    if known is not None:
        return known
    return _common_ref(clean)


def deep_research_model_info(version: str) -> ModelInfo:
    """Return capability metadata for a Deep Research model."""
    ref = deep_research_model(version)
    info = ref.info
    if isinstance(info, ModelInfo):
        return info
    if isinstance(info, dict):
        return ModelInfo.model_validate(info)
    return ModelInfo(
        label=f'Google AI - {extract_version(version)}',
        supports=GENERIC_DEEP_RESEARCH_INFO.supports,
    )


def list_known_deep_research_models() -> list[ModelRef]:
    """Return statically known Deep Research ModelRefs."""
    return list(KNOWN_DEEP_RESEARCH_MODELS.values())


def _build_tools(request: ModelRequest, config: dict[str, Any]) -> list[InteractionTool]:
    tools: list[InteractionTool] = []
    if request.tools:
        for tool in request.tools:
            tools.append(to_interaction_tool(tool))

    google_search = config.get('googleSearch') or config.get('google_search')
    if google_search:
        tool: InteractionTool = {'type': 'google_search'}
        if isinstance(google_search, dict):
            tool = cast(InteractionTool, {'type': 'google_search', **google_search})
        tools.append(tool)

    url_context = config.get('urlContext') or config.get('url_context')
    if url_context:
        tool = {'type': 'url_context'}
        if isinstance(url_context, dict):
            tool = cast(InteractionTool, {'type': 'url_context', **url_context})
        tools.append(tool)

    code_execution = config.get('codeExecution') or config.get('code_execution')
    if code_execution:
        tool = {'type': 'code_execution'}
        if isinstance(code_execution, dict):
            tool = cast(InteractionTool, {'type': 'code_execution', **code_execution})
        tools.append(tool)

    file_search = config.get('fileSearch') or config.get('file_search')
    if isinstance(file_search, dict):
        store_names = file_search.get('fileSearchStoreNames') or file_search.get('file_search_store_names')
        rest = {
            key: value
            for key, value in file_search.items()
            if key not in {'fileSearchStoreNames', 'file_search_store_names'}
        }
        tool = cast(
            InteractionTool,
            {
                'type': 'file_search',
                'file_search_store_names': store_names,
                **rest,
            },
        )
        tools.append(tool)

    mcp_servers = config.get('mcpServers') or config.get('mcp_servers') or []
    for mcp_server in mcp_servers:
        if not isinstance(mcp_server, dict):
            continue
        allowed_tools = mcp_server.get('allowedTools') or mcp_server.get('allowed_tools')
        rest_mcp = {key: value for key, value in mcp_server.items() if key not in {'allowedTools', 'allowed_tools'}}
        tool = cast(
            InteractionTool,
            {
                'type': 'mcp_server',
                **rest_mcp,
                **({'allowed_tools': allowed_tools} if allowed_tools else {}),
            },
        )
        tools.append(tool)

    return tools


def _build_create_request(request: ModelRequest, version: str, config: dict[str, Any]) -> CreateInteractionRequest:
    thinking_summaries = config.get('thinkingSummaries') or config.get('thinking_summaries')
    visualization = config.get('visualization')
    collaborative_planning = config.get('collaborativePlanning') or config.get('collaborative_planning')
    previous_interaction_id = config.get('previousInteractionId') or config.get('previous_interaction_id')
    store = config.get('store')

    passthrough = remove_client_option_overrides(config)
    for key in (
        'thinkingSummaries',
        'thinking_summaries',
        'visualization',
        'collaborativePlanning',
        'collaborative_planning',
        'previousInteractionId',
        'previous_interaction_id',
        'store',
        'responseModalities',
        'response_modalities',
        'googleSearch',
        'google_search',
        'urlContext',
        'url_context',
        'codeExecution',
        'code_execution',
        'fileSearch',
        'file_search',
        'mcpServers',
        'mcp_servers',
    ):
        passthrough.pop(key, None)

    agent_config: DeepResearchAgentConfig = {'type': 'deep-research'}
    if isinstance(thinking_summaries, str):
        agent_config['thinking_summaries'] = thinking_summaries.lower()  # type: ignore[typeddict-item]
    if isinstance(visualization, str):
        agent_config['visualization'] = visualization.lower()  # type: ignore[typeddict-item]
    if collaborative_planning is not None:
        agent_config['collaborative_planning'] = bool(collaborative_planning)

    response_format: dict[str, Any] | None = None
    if request.output_format == 'json' or request.output_content_type == 'application/json':
        response_format = {'type': 'text', 'mime_type': 'application/json'}
        if request.output_schema:
            response_format['schema'] = clean_schema(request.output_schema)

    tools = _build_tools(request, config)
    messages = downgrade_system_messages(request.messages or [])

    req: CreateInteractionRequest = {
        'agent': extract_version(version),
        'input': to_interaction_steps(ensure_tool_ids(messages)),
        'background': True,
        'agent_config': agent_config,
        **({'previous_interaction_id': previous_interaction_id} if previous_interaction_id else {}),
        **({'store': store} if store is not None else {}),
        **({'tools': tools} if tools else {}),
        **({'response_format': response_format} if response_format else {}),
        **passthrough,
    }

    response_modalities = response_modalities_from_config(config)
    if response_modalities is not None:
        req['response_modalities'] = response_modalities

    return req


class DeepResearchModel:
    """Deep Research model backed by the Interactions API."""

    def __init__(
        self,
        version: str,
        *,
        plugin_api_key: str | None,
        client_options: ClientOptions,
    ) -> None:
        """Initialize Deep Research model."""
        self._version = version
        self._plugin_api_key = plugin_api_key
        self._client_options = client_options

    async def start(self, request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        """Start a background Deep Research interaction."""
        config = config_as_dict(request.config)
        api_key = calculate_api_key(
            self._plugin_api_key,
            config.get('apiKey') or config.get('api_key'),
        )
        client_options = merge_client_options(self._client_options, config)
        client = InteractionsClient(api_key=api_key, client_options=client_options)
        try:
            interaction = await client.create_interaction(_build_create_request(request, self._version, config))
        finally:
            await client.aclose()
        return from_interaction(interaction, client_options_for_operation(client_options))

    async def check(self, operation: Operation) -> Operation:
        """Poll a Deep Research interaction for completion."""
        stored_options = cast(ClientOptions | None, (operation.metadata or {}).get('clientOptions'))
        check_options = merge_client_options(self._client_options, dict(stored_options or {}))
        api_key = calculate_api_key(self._plugin_api_key, None)
        client = InteractionsClient(api_key=api_key, client_options=check_options)
        try:
            interaction = await client.get_interaction(operation.id)
        finally:
            await client.aclose()
        return from_interaction(interaction, client_options_for_operation(check_options))

    async def cancel(self, operation: Operation) -> Operation:
        """Cancel an in-progress Deep Research interaction."""
        stored_options = cast(ClientOptions | None, (operation.metadata or {}).get('clientOptions'))
        cancel_options = merge_client_options(self._client_options, dict(stored_options or {}))
        api_key = calculate_api_key(self._plugin_api_key, None)
        client = InteractionsClient(api_key=api_key, client_options=cancel_options)
        try:
            interaction = await client.cancel_interaction(operation.id)
        finally:
            await client.aclose()
        return from_interaction(interaction, client_options_for_operation(cancel_options))


def create_deep_research_background_action(
    ref: ModelRef,
    *,
    plugin_api_key: str | None,
    client_options: ClientOptions,
) -> BackgroundAction:
    """Wire Deep Research Interactions start/check/cancel through define_background_model."""
    version = extract_version(ref.name)
    model = DeepResearchModel(version, plugin_api_key=plugin_api_key, client_options=client_options)
    info = deep_research_model_info(version)
    label = info.label or ref.name

    # Throwaway registry: plugin init re-registers the returned actions on the app registry.
    # define_background_model stamps Operation.action so check_operation/cancel_operation work.
    return define_background_model(
        registry=Registry(),
        name=ref.name,
        start=model.start,
        check=model.check,
        cancel=model.cancel,
        label=label,
        info=info,
        config_schema=DeepResearchConfigSchema,
    )
