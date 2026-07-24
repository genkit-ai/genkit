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

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, cast

from google import genai
from pydantic import BaseModel, ConfigDict

from genkit import ModelInfo, ModelRequest, Supports
from genkit._core._background import define_background_model
from genkit._core._registry import Registry
from genkit.model import BackgroundAction, ModelRef, Operation, model_ref
from genkit.plugin_api import ActionRunContext
from genkit_google_genai._interactions.converters import (
    clean_schema,
    ensure_tool_ids,
    from_interaction,
    to_interaction_steps,
    to_interaction_tool,
)
from genkit_google_genai.models.interactions_utils import (
    ClientOptions,
    calculate_api_key,
    client_options_for_operation,
    config_as_dict,
    downgrade_system_messages,
    extract_version,
    map_genai_error,
    merge_client_options,
    remove_client_option_overrides,
    resolve_interactions_client,
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

    model_config = ConfigDict(extra='allow')
    name: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: list[str] | None = None


class FileSearchConfig(BaseModel):
    """File search store configuration for Deep Research."""

    model_config = ConfigDict(extra='allow')
    file_search_store_names: list[str]


class DeepResearchConfigSchema(BaseModel):
    """Deep Research model configuration."""

    model_config = ConfigDict(extra='allow')
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    thinking_summaries: Literal['auto', 'none'] | None = None
    previous_interaction_id: str | None = None
    store: bool | None = None
    response_modalities: list[Literal['text', 'image', 'audio']] | None = None
    visualization: Literal['auto', 'off'] | None = None
    collaborative_planning: bool | None = None
    google_search: bool | dict[str, Any] | None = None
    url_context: bool | dict[str, Any] | None = None
    code_execution: bool | dict[str, Any] | None = None
    file_search: FileSearchConfig | dict[str, Any] | None = None
    mcp_servers: list[McpServerConfig | dict[str, Any]] | None = None


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


def _build_tools(request: ModelRequest, config: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if request.tools:
        for tool_def in request.tools:
            tools.append(to_interaction_tool(tool_def))

    google_search = config.get('google_search')
    if google_search:
        builtin: dict[str, Any] = {'type': 'google_search'}
        if isinstance(google_search, dict):
            builtin = {'type': 'google_search', **google_search}
        tools.append(builtin)

    url_context = config.get('url_context')
    if url_context:
        builtin = {'type': 'url_context'}
        if isinstance(url_context, dict):
            builtin = {'type': 'url_context', **url_context}
        tools.append(builtin)

    code_execution = config.get('code_execution')
    if code_execution:
        builtin = {'type': 'code_execution'}
        if isinstance(code_execution, dict):
            builtin = {'type': 'code_execution', **code_execution}
        tools.append(builtin)

    file_search = config.get('file_search')
    if isinstance(file_search, dict):
        store_names = file_search.get('file_search_store_names')
        rest = {key: value for key, value in file_search.items() if key != 'file_search_store_names'}
        tools.append({
            'type': 'file_search',
            'file_search_store_names': store_names,
            **rest,
        })

    for mcp_server in config.get('mcp_servers') or []:
        if not isinstance(mcp_server, dict):
            continue
        allowed_tools = mcp_server.get('allowed_tools')
        rest_mcp = {key: value for key, value in mcp_server.items() if key != 'allowed_tools'}
        tools.append({
            'type': 'mcp_server',
            **rest_mcp,
            **({'allowed_tools': allowed_tools} if allowed_tools else {}),
        })

    return tools


def _build_create_request(request: ModelRequest, version: str, config: dict[str, Any]) -> dict[str, Any]:
    thinking_summaries = config.get('thinking_summaries')
    visualization = config.get('visualization')
    collaborative_planning = config.get('collaborative_planning')
    previous_interaction_id = config.get('previous_interaction_id')
    store = config.get('store')

    passthrough = remove_client_option_overrides(config)
    for key in (
        'thinking_summaries',
        'visualization',
        'collaborative_planning',
        'previous_interaction_id',
        'store',
        'response_modalities',
        'google_search',
        'url_context',
        'code_execution',
        'file_search',
        'mcp_servers',
    ):
        passthrough.pop(key, None)

    agent_config: dict[str, Any] = {'type': 'deep-research'}
    if isinstance(thinking_summaries, str):
        agent_config['thinking_summaries'] = thinking_summaries
    if isinstance(visualization, str):
        agent_config['visualization'] = visualization
    if collaborative_planning is not None:
        agent_config['collaborative_planning'] = bool(collaborative_planning)

    response_format: dict[str, Any] | None = None
    if request.output_format == 'json' or request.output_content_type == 'application/json':
        response_format = {'type': 'text', 'mime_type': 'application/json'}
        if request.output_schema:
            response_format['schema'] = clean_schema(request.output_schema)

    tools = _build_tools(request, config)
    messages = downgrade_system_messages(request.messages or [])

    req_dict: dict[str, Any] = {
        'agent': extract_version(version),
        'input': to_interaction_steps(ensure_tool_ids(messages)),
        'background': True,
        'agent_config': agent_config,
    }
    if previous_interaction_id:
        req_dict['previous_interaction_id'] = previous_interaction_id
    if store is not None:
        req_dict['store'] = store
    if tools:
        req_dict['tools'] = tools
    if response_format is not None:
        req_dict['response_format'] = response_format
    req_dict.update(passthrough)

    response_modalities = response_modalities_from_config(config)
    if response_modalities is not None:
        req_dict['response_modalities'] = response_modalities

    return req_dict


class DeepResearchModel:
    """Deep Research model backed by the Interactions API."""

    def __init__(
        self,
        version: str,
        *,
        plugin_api_key: str | None,
        client_options: ClientOptions,
        client_getter: Callable[[], genai.Client] | None = None,
    ) -> None:
        """Initialize Deep Research model."""
        self._version = version
        self._plugin_api_key = plugin_api_key
        self._client_options = client_options
        self._client_getter = client_getter

    def _client_scope(
        self,
        *,
        api_key: str,
        request_api_key: str | None,
        client_options: ClientOptions,
    ) -> AbstractAsyncContextManager[genai.Client]:
        return resolve_interactions_client(
            client_getter=self._client_getter,
            plugin_api_key=self._plugin_api_key,
            api_key=api_key,
            request_api_key=request_api_key,
            plugin_client_options=self._client_options,
            client_options=client_options,
        )

    async def start(self, request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        """Start a background Deep Research interaction."""
        config = config_as_dict(request.config)
        request_api_key = config.get('api_key')
        if request_api_key is not None:
            request_api_key = str(request_api_key)
        api_key = calculate_api_key(self._plugin_api_key, request_api_key)
        client_options = merge_client_options(self._client_options, config)
        async with self._client_scope(
            api_key=api_key,
            request_api_key=request_api_key,
            client_options=client_options,
        ) as client:
            try:
                interaction = cast(
                    BaseModel,
                    await client.aio.interactions.create(**_build_create_request(request, self._version, config)),
                )
            except Exception as error:
                raise map_genai_error(error) from error
        return from_interaction(
            interaction,
            client_options_for_operation(client_options, api_key=api_key),
        )

    def _api_key_from_operation(self, stored: dict[str, Any]) -> tuple[str, str | None]:
        """Resolve api key for check/cancel, preferring the key stored at start."""
        stored_api_key = stored.get('api_key')
        if isinstance(stored_api_key, str) and stored_api_key:
            # Treat a stored override as request_api_key so we don't reuse the
            # plugin client when the start call used a different key.
            request_api_key = (
                stored_api_key if self._plugin_api_key is not None and stored_api_key != self._plugin_api_key else None
            )
            return stored_api_key, request_api_key
        return calculate_api_key(self._plugin_api_key, None), None

    async def check(self, operation: Operation) -> Operation:
        """Poll a Deep Research interaction for completion."""
        stored_raw = (operation.metadata or {}).get('clientOptions')
        stored_dict: dict[str, Any] = cast(dict[str, Any], stored_raw) if isinstance(stored_raw, dict) else {}
        check_options = merge_client_options(self._client_options, stored_dict)
        api_key, request_api_key = self._api_key_from_operation(stored_dict)
        async with self._client_scope(
            api_key=api_key,
            request_api_key=request_api_key,
            client_options=check_options,
        ) as client:
            try:
                interaction = await client.aio.interactions.get(operation.id)
            except Exception as error:
                raise map_genai_error(error) from error
        return from_interaction(
            interaction,
            client_options_for_operation(check_options, api_key=api_key),
        )

    async def cancel(self, operation: Operation) -> Operation:
        """Cancel an in-progress Deep Research interaction."""
        stored_raw = (operation.metadata or {}).get('clientOptions')
        stored_dict: dict[str, Any] = cast(dict[str, Any], stored_raw) if isinstance(stored_raw, dict) else {}
        cancel_options = merge_client_options(self._client_options, stored_dict)
        api_key, request_api_key = self._api_key_from_operation(stored_dict)
        async with self._client_scope(
            api_key=api_key,
            request_api_key=request_api_key,
            client_options=cancel_options,
        ) as client:
            try:
                interaction = await client.aio.interactions.cancel(operation.id)
            except Exception as error:
                raise map_genai_error(error) from error
        return from_interaction(
            interaction,
            client_options_for_operation(cancel_options, api_key=api_key),
        )


def create_deep_research_background_action(
    ref: ModelRef,
    *,
    plugin_api_key: str | None,
    client_options: ClientOptions,
    client_getter: Callable[[], genai.Client] | None = None,
) -> BackgroundAction:
    """Wire Deep Research Interactions start/check/cancel through define_background_model."""
    version = extract_version(ref.name)
    model = DeepResearchModel(
        version,
        plugin_api_key=plugin_api_key,
        client_options=client_options,
        client_getter=client_getter,
    )
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
