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

"""Ollama Plugin for Genkit."""

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

import ollama as ollama_api
import structlog

from genkit import ActionRunContext, ModelResponse
from genkit.embedder import (
    EmbedderInfo,
    EmbedderSupports,
    EmbedRequest,
    EmbedResponse,
    embedder as create_embedder,
    embedder_action_metadata,
)
from genkit.model import Constrained, ModelInfo, ModelRequest, Supports, model as create_model, model_action_metadata
from genkit.plugin_api import (
    Action,
    ActionKind,
    ActionMetadata,
    Plugin,
    loop_local_client,
    to_json_schema,
    wrap_http_error,
)
from genkit_ollama._constants import DEFAULT_OLLAMA_SERVER_URL
from genkit_ollama._embedders import OllamaEmbedder
from genkit_ollama._errors import wrap_connection_errors
from genkit_ollama._models import (
    OllamaConfig,
    OllamaModel,
    _ResolvedModel,
)

OLLAMA_PLUGIN_NAME = 'ollama'
logger = structlog.get_logger(__name__)

# Capability probing (POST /api/show), bounded the way the Go plugin bounds it:
# a probe never waits longer than this, or the plugin timeout if that is
# shorter; listing runs at most this many probes at once; and a failed probe
# is remembered only briefly, so a server that comes up later gets asked again.
_PROBE_TIMEOUT_SECONDS = 5.0
_MAX_CONCURRENT_PROBES = 4
_PROBE_FAILURE_TTL_SECONDS = 30.0

# Templates that pass the prompt through untouched. Ollama reports
# ``{{ .Prompt }}`` for a model that ships no chat template.
_RAW_TEMPLATES = frozenset({'', '{{ .Prompt }}'})
# Capabilities only a chat-formatted model has. A model that renders chat
# natively (no Go template) still reports these, so it stays on /api/chat.
_CHAT_ONLY_CAPABILITIES = frozenset({'tools', 'vision', 'thinking'})


@dataclass(frozen=True)
class _ProbeResult:
    definition: _ResolvedModel
    digest: str
    expires_at: float | None
    """Monotonic deadline for a failed probe; None keeps a success for good."""


def _cache_key(name: str) -> str:
    # ``llama3.2`` and ``llama3.2:latest`` are the same model on the server.
    return name.removesuffix(':latest')


def _resolved_from_show(name: str, show: object) -> _ResolvedModel | None:
    """Reads an ``/api/show`` answer into a model definition.

    Returns None when the answer carries no ``capabilities`` list, which is
    what an Ollama server older than the field sends.
    """
    capabilities = getattr(show, 'capabilities', None)
    if not isinstance(capabilities, list):
        return None
    caps = {c for c in cast(list[object], capabilities) if isinstance(c, str)}
    template = getattr(show, 'template', None)
    raw = not isinstance(template, str) or template.strip() in _RAW_TEMPLATES
    return _ResolvedModel(
        name=name,
        # /api/chat on a template-less model flattens the turns into one
        # prompt anyway; /api/generate says so honestly.
        api_type='generate' if raw and not caps & _CHAT_ONLY_CAPABILITIES else 'chat',
        tools='tools' in caps,
        media='vision' in caps,
    )


def ollama_name(name: str) -> str:
    """Get the name of the Ollama model.

    Args:
        name: The name of the Ollama model.

    Returns:
        The name of the Ollama model.
    """
    return f'{OLLAMA_PLUGIN_NAME}/{name}'


def ollama_model_info(model_ref: _ResolvedModel, label: str) -> dict[str, object]:
    """Build Dev UI capability metadata for an Ollama model.

    Capabilities are gated on the model's API type so the Dev UI advertises
    only what the endpoint actually supports: the ``chat`` endpoint is
    multiturn and can use tools/media, whereas the ``generate`` endpoint is
    single-turn text-in/text-out.

    Args:
        model_ref: The resolved model, with its API type and capabilities.
        label: The human-readable label to show in the Dev UI.

    Returns:
        The serialized :class:`ModelInfo` metadata (camelCase aliases, no
        ``None`` values) ready to embed under ``metadata['model']``.
    """
    is_chat = model_ref.api_type == 'chat'
    return ModelInfo(
        label=label,
        supports=Supports(
            multiturn=is_chat,
            media=is_chat and model_ref.media,
            tools=is_chat and model_ref.tools,
            system_role=True,
            # Deliberate JS/Go deviation. we match other Python plugins for Dev UI consistency.
            output=['text', 'json'],
            constrained=Constrained.ALL,
        ),
    ).model_dump(by_alias=True, exclude_none=True)


@dataclass(frozen=True)
class RequestHeaderParams:
    """Context passed to a ``request_headers`` callable.

    Mirrors the JS plugin's ``RequestHeaderFunction`` params so a callback can
    tailor headers to the server, the model, or the specific request — e.g. a
    freshly minted, per-request auth token. ``model`` is the model or embedder
    name. ``model_request`` is set for model actions and ``embed_request`` for
    embedder actions; both are ``None`` for the ``list_actions`` discovery call
    and for the ``/api/show`` capability probe, which sets only ``model``.
    """

    server_address: str
    model: str | None = None
    model_request: ModelRequest | None = None
    embed_request: EmbedRequest | None = None


# A request_headers callable receives the per-request context and returns the
# headers to merge (or ``None`` for no extra headers), optionally as an awaitable.
RequestHeaderFunction = Callable[
    [RequestHeaderParams],
    dict[str, str] | None | Awaitable[dict[str, str] | None],
]
# request_headers may be a static dict or a (sync/async) callable.
RequestHeaders = dict[str, str] | RequestHeaderFunction


def _require_name_list(arg: str, value: list[str] | None) -> list[str]:
    # A missing bracket (models='llama3.2') would otherwise iterate the string
    # and list one model per character.
    if isinstance(value, str):
        raise TypeError(f'{arg}= takes a list of Ollama model names, got a str. Did you mean {arg}=[{value!r}]?')
    return list(value or [])


class Ollama(Plugin):
    """Ollama plugin for Genkit.

    This plugin integrates Ollama models and embedding capabilities into Genkit
    for local or custom server-based generative AI applications.
    """

    name = OLLAMA_PLUGIN_NAME

    def __init__(
        self,
        models: list[str] | None = None,
        embedders: list[str] | None = None,
        server_address: str | None = None,
        request_headers: RequestHeaders | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the Ollama plugin.

        Args:
            models: Ollama model names to list in the Dev UI, e.g.
                ``['llama3.2', 'llava']``. Capabilities come from the server:
                each model is probed once via ``/api/show`` for tools, vision
                and its chat template. Unlisted names still resolve on demand.
            embedders: Ollama embedding model names to register, e.g.
                ``['nomic-embed-text']``.
            server_address: The URL of the Ollama server. Defaults to a predefined
                Ollama server URL if not provided.
            request_headers: Optional HTTP headers to include with requests to the
                Ollama server. May be a static dict, or a sync/async callable that
                takes a :class:`RequestHeaderParams` (server address plus model/request
                context) and returns a dict (or ``None``). A callable is resolved per
                request — matching the JS plugin — so expiring auth tokens and
                request-specific headers take effect; a static dict is applied once to a
                cached client.
            timeout: Optional request timeout (seconds) forwarded to the underlying
                httpx client.
        """
        self.models = _require_name_list('models', models)
        self.embedders = _require_name_list('embedders', embedders)
        self.server_address = server_address or DEFAULT_OLLAMA_SERVER_URL

        self._request_headers_source = request_headers
        # Static dicts are baked into the cached client; callables resolve per request.
        self.request_headers = dict(request_headers) if isinstance(request_headers, dict) else {}
        self.timeout = timeout
        self.client = loop_local_client(self._make_client)
        self._probes: dict[str, _ProbeResult] = {}

    def _make_client(self, headers: dict[str, str] | None = None) -> ollama_api.AsyncClient:
        """Build an Ollama AsyncClient with the given (or static) headers and timeout.

        Args:
            headers: Per-request headers to use instead of the static ``request_headers``
                (e.g. resolved from a callable). Defaults to the static headers, which is
                what the per-event-loop cached client is built with.

        Returns:
            A new ``ollama.AsyncClient`` targeting the configured server.
        """
        kwargs: dict[str, Any] = {
            'host': self.server_address,
            'headers': self.request_headers if headers is None else headers,
        }
        if self.timeout is not None:
            kwargs['timeout'] = self.timeout
        return ollama_api.AsyncClient(**kwargs)

    @asynccontextmanager
    async def _client_for_request(
        self,
        *,
        model: str | None = None,
        model_request: ModelRequest | None = None,
        embed_request: EmbedRequest | None = None,
    ) -> AsyncIterator[ollama_api.AsyncClient]:
        """Yield the Ollama client to use for a single request.

        Static (or absent) headers are baked into a per-event-loop cached client that
        is shared across requests and left open. A header *callable* is resolved on
        every call — receiving the server address plus any model/request context —
        and applied to a *fresh* client, so expiring auth tokens or
        request-specific headers take effect. Because the Ollama SDK bakes headers in
        at construction (it has no per-request header hook), that fresh client owns
        its own httpx connection pool; it is closed on exit so long-running callers
        don't accumulate pools.

        Args:
            model: The model or embedder name this request targets, if any.
            model_request: The generate request, when resolving for a model action.
            embed_request: The embed request, when resolving for an embedder action.

        Yields:
            The Ollama client for this request.
        """
        source = self._request_headers_source
        if not callable(source):
            # Shared per-event-loop cached client — reused across requests, not closed.
            yield self.client()
            return

        params = RequestHeaderParams(
            server_address=self.server_address,
            model=model,
            model_request=model_request,
            embed_request=embed_request,
        )
        result = source(params)
        if inspect.isawaitable(result):
            result = await result
        headers = dict(cast(dict[str, str], result)) if result else {}
        client = self._make_client(headers=headers)
        try:
            yield client
        finally:
            # ollama.AsyncClient exposes no public close, so close the wrapped httpx
            # client to release this request's connection pool. aclose() is idempotent.
            inner = getattr(client, '_client', None)
            if inner is not None:
                await inner.aclose()
            else:
                # Defensive: if a future ollama SDK renames/drops ``_client`` this
                # would silently leak a connection pool per request. Surface it.
                logger.warning('ollama client exposes no _client; per-request connection pool was not closed')

    async def init(self) -> list:
        """Initialize the Ollama plugin.

        Registers the configured embedders. Models are not registered here:
        their capabilities come from ``/api/show``, which ``resolve`` and
        ``list_actions`` ask for, and an eagerly registered action would mask
        the probed listing with generic metadata.

        Returns:
            Embedder actions for the configured embedders.
        """
        # Header callables are resolved per request (see _client_for_request), so
        # there is nothing to resolve eagerly here; static headers are already set.
        return [self._create_embedder_action(name) for name in self.embedders]

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        A model is probed once via ``/api/show`` before its action is built,
        whether or not it is listed in ``models``. A failed probe falls back
        to ``/api/chat`` with every capability advertised.

        Args:
            action_type: The kind of action to resolve.
            name: The id without the ``ollama/`` prefix.

        Returns:
            Action object if found, None otherwise.
        """
        if action_type == ActionKind.MODEL:
            return self._create_model_action(await self._resolve_model(name))
        elif action_type == ActionKind.EMBEDDER:
            return self._create_embedder_action(name)
        return None

    async def _resolve_model(self, name: str, digest: str = '') -> _ResolvedModel:
        """Returns the model's capabilities, probing the server at most once.

        Args:
            name: The Ollama model name.
            digest: The digest ``/api/tags`` reported, if known. A cached
                result for a different digest is stale: the model was re-pulled.

        Returns:
            The probed definition, or the generic fallback when the probe fails.
        """
        key = _cache_key(name)
        cached = self._probes.get(key)
        if cached is not None and (not digest or cached.digest == digest):
            if cached.expires_at is None or time.monotonic() < cached.expires_at:
                return cached.definition

        definition = await self._probe(name)
        detected = definition is not None
        if definition is None:
            definition = _ResolvedModel(name=name)
        self._probes[key] = _ProbeResult(
            definition=definition,
            digest=digest,
            expires_at=None if detected else time.monotonic() + _PROBE_FAILURE_TTL_SECONDS,
        )
        return definition

    async def _probe(self, name: str) -> _ResolvedModel | None:
        """Asks ``/api/show`` what the model can do.

        Any failure (server down, model not pulled, an old server without
        ``capabilities``, a header callable that raises) returns None so the
        caller falls back; plugin init and resolve never fail on a probe.
        """
        timeout = _PROBE_TIMEOUT_SECONDS
        if self.timeout is not None and 0 < self.timeout < timeout:
            timeout = self.timeout
        try:
            async with self._client_for_request(model=name) as client:
                show = await asyncio.wait_for(client.show(name), timeout=timeout)
        except Exception as e:  # noqa: BLE001 - every probe failure means the same fallback.
            logger.debug('Ollama capability probe failed', model=name, error=type(e).__name__)
            return None
        definition = _resolved_from_show(name, show)
        if definition is None:
            logger.debug('Ollama capability probe failed', model=name, error='no_capabilities')
            return None
        logger.debug(
            'Ollama capabilities probed',
            model=name,
            api_type=definition.api_type,
            tools=definition.tools,
            media=definition.media,
        )
        return definition

    def _create_model_action(self, model_ref: _ResolvedModel) -> Action:
        """Create an Action object for an Ollama model.

        Args:
            model_ref: The resolved model; its name is used as received (no
                plugin-prefix stripping).

        Returns:
            Action object for the model.
        """
        name = model_ref.name
        model = OllamaModel(
            client=self.client,
            model_definition=model_ref,
            server_address=self.server_address,
        )

        action_metadata = model_action_metadata(
            name=ollama_name(name),
            config_schema=OllamaConfig,
            info=ollama_model_info(model_ref, f'Ollama - {name}'),
        )

        async def _run(request: ModelRequest, ctx: ActionRunContext | None = None) -> ModelResponse:
            # Resolve per-request headers (no-op for static headers), passing the model
            # and request context to a header callable (JS parity). OllamaModel wraps
            # connection errors at the SDK boundary, so a failed media-URL fetch isn't
            # misreported as an Ollama server outage.
            async with self._client_for_request(model=name, model_request=request) as client:
                return await model.generate(request, ctx, client=client)

        action = create_model(
            ollama_name(name),
            _run,
            config_schema=OllamaConfig,
            metadata=action_metadata.metadata,
        )

        # Explicitly set schemas (always present in the action metadata).
        action.input_schema = action_metadata.input_json_schema  # type: ignore[invalid-assignment]
        action.output_schema = action_metadata.output_json_schema  # type: ignore[invalid-assignment]

        return action

    def _create_embedder_action(self, name: str) -> Action:
        """Create an Action object for an Ollama embedder.

        Args:
            name: The embedder id as received (no plugin-prefix stripping).

        Returns:
            Action object for the embedder.
        """
        embedder = OllamaEmbedder(client=self.client, model=name)

        server_address = self.server_address

        async def _run(request: EmbedRequest) -> EmbedResponse:
            # Pass the embedder and embed request to a header callable (JS parity).
            # Embedding requests never fetch media, so the whole SDK call is wrapped.
            async with self._client_for_request(model=name, embed_request=request) as client:
                async with wrap_connection_errors(server_address):
                    return await embedder.embed(request, client=client)

        return create_embedder(
            ollama_name(name),
            _run,
            info=EmbedderInfo(
                label=f'Ollama Embedding - {name}',
                supports=EmbedderSupports(input=['text']),
                config_schema=to_json_schema(ollama_api.Options),
            ),
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """List the server's models and embedders, plus any configured model it lacks.

        Reads ``/api/tags``, then probes each model's ``/api/show``
        concurrently (at most four at once, five seconds each) so the Dev UI
        shows real capabilities. Results are cached per model and digest; a
        failed probe falls back to the generic set and is retried after 30s.
        Names containing ``embed`` are listed as embedders, as in JS and Go.

        Returns:
            ActionMetadata for each model and embedder.
        """
        async with self._client_for_request() as client:
            async with wrap_connection_errors(self.server_address):
                try:
                    response = await client.list()
                except ollama_api.ResponseError as e:
                    raise wrap_http_error(e, status_code=e.status_code) from e

        embedder_names: list[str] = []
        # (name, digest) per model row, in server order.
        model_rows: list[tuple[str, str]] = []
        for model in response.models:
            name = model.model
            if not name:
                continue
            if 'embed' in name:
                embedder_names.append(name)
            else:
                model_rows.append((name, getattr(model, 'digest', None) or ''))
        listed = {_cache_key(name) for name, _ in model_rows}
        # A configured model the server lacks is listed anyway, as before; its
        # probe 404s and it carries the generic capabilities.
        model_rows.extend((name, '') for name in self.models if _cache_key(name) not in listed)

        slots = asyncio.Semaphore(_MAX_CONCURRENT_PROBES)

        async def describe(name: str, digest: str) -> _ResolvedModel:
            async with slots:
                return await self._resolve_model(name, digest)

        definitions = await asyncio.gather(*(describe(name, digest) for name, digest in model_rows))

        actions: list[ActionMetadata] = [
            model_action_metadata(
                name=ollama_name(definition.name),
                config_schema=OllamaConfig,
                info=ollama_model_info(definition, f'Ollama - {definition.name}'),
            )
            for definition in definitions
        ]
        actions.extend(
            embedder_action_metadata(
                name=ollama_name(name),
                info=EmbedderInfo(
                    config_schema=to_json_schema(ollama_api.Options),
                    label=f'Ollama Embedding - {name}',
                    supports=EmbedderSupports(input=['text']),
                ),
            )
            for name in embedder_names
        )
        return actions
