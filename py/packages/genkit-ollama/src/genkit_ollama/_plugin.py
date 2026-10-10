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
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import httpx
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
    return _ResolvedModel(name=name, tools='tools' in caps, media='vision' in caps)


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

    Every model goes through ``/api/chat``, as in Go; the probe only decides
    whether tools and media are advertised.

    Args:
        model_ref: The resolved model and its capabilities.
        label: The human-readable label to show in the Dev UI.

    Returns:
        The serialized :class:`ModelInfo` metadata (camelCase aliases, no
        ``None`` values) ready to embed under ``metadata['model']``.
    """
    return ModelInfo(
        label=label,
        supports=Supports(
            multiturn=True,
            media=model_ref.media,
            tools=model_ref.tools,
            system_role=True,
            # Deliberate JS/Go deviation. we match other Python plugins for Dev UI consistency.
            output=['text', 'json'],
            constrained=Constrained.ALL,
        ),
    ).model_dump(by_alias=True, exclude_none=True)


# A request_headers callable takes no arguments and returns the headers to
# merge (or None), optionally as an awaitable.
_HeaderSource = Callable[[], dict[str, str] | None | Awaitable[dict[str, str] | None]]


def _require_no_arg_callable(source: _HeaderSource) -> None:
    # A one-argument callable from the RequestHeaderParams era would otherwise
    # fail inside the httpx auth hook on the first request, where the TypeError
    # surfaces as a bare INTERNAL error.
    try:
        signature = inspect.signature(source)
    except (TypeError, ValueError):  # Builtins and C callables have no signature to check.
        return
    try:
        signature.bind()
    except TypeError:
        raise TypeError(
            'request_headers callables take no arguments; change `def headers(params)` to `def headers()`.'
        ) from None


class _CallableHeaders(httpx.Auth):
    """Runs a ``request_headers`` callable before every HTTP request.

    The Ollama SDK fixes headers when its client is built. Hooking httpx auth
    instead keeps one pooled client per event loop and still lets an expiring
    token refresh on every call.
    """

    def __init__(self, source: _HeaderSource) -> None:
        self._source = source

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        result = self._source()
        if inspect.isawaitable(result):
            result = await result
        if result:
            request.headers.update(result)
        yield request


class Ollama(Plugin):
    """Ollama plugin for Genkit.

    This plugin integrates Ollama models and embedding capabilities into Genkit
    for local or custom server-based generative AI applications.
    """

    name = OLLAMA_PLUGIN_NAME

    def __init__(
        self,
        server_address: str | None = None,
        request_headers: dict[str, str] | _HeaderSource | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the Ollama plugin.

        Models and embedders are not configured here. The Dev UI lists what the
        server has pulled (``/api/tags``), any ``ollama/<name>`` resolves on
        demand, and each model's tools and vision support come from one
        ``/api/show`` probe.

        Args:
            server_address: The URL of the Ollama server. Defaults to a predefined
                Ollama server URL if not provided.
            request_headers: Extra HTTP headers for every request to the Ollama
                server, typically auth for a proxy in front of it. A dict is
                sent as-is. A callable (sync or async, no arguments) runs before
                every request, so an expiring token can be refreshed; it should
                cache the token itself.
            timeout: Optional request timeout (seconds) forwarded to the underlying
                httpx client.
        """
        self.server_address = server_address or DEFAULT_OLLAMA_SERVER_URL

        # A dict is baked into the cached client; a callable runs per request via httpx auth.
        self.request_headers: dict[str, str] = {}
        self._auth: _CallableHeaders | None = None
        if isinstance(request_headers, dict):
            self.request_headers = dict(request_headers)
        elif request_headers is not None:
            _require_no_arg_callable(request_headers)
            self._auth = _CallableHeaders(request_headers)
        self.timeout = timeout
        self.client = loop_local_client(self._make_client)
        self._probes: dict[str, _ProbeResult] = {}
        # One probe per model at a time; concurrent first uses await the same task.
        self._inflight: dict[str, asyncio.Task[_ResolvedModel]] = {}

    def _make_client(self) -> ollama_api.AsyncClient:
        """Build the Ollama AsyncClient for the current event loop.

        Returns:
            A new ``ollama.AsyncClient`` targeting the configured server.
        """
        kwargs: dict[str, Any] = {'host': self.server_address, 'headers': self.request_headers}
        if self._auth is not None:
            # Extra kwargs go straight to httpx.AsyncClient.
            kwargs['auth'] = self._auth
        if self.timeout is not None:
            kwargs['timeout'] = self.timeout
        return ollama_api.AsyncClient(**kwargs)

    async def init(self) -> list:
        """Initialize the Ollama plugin.

        Registers nothing: models and embedders resolve on demand, and an
        eagerly registered action would mask the probed listing with generic
        metadata.

        Returns:
            An empty list.
        """
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        A model is probed once via ``/api/show`` before its action is built. A
        failed probe advertises the generic capabilities; the request still
        goes to ``/api/chat`` either way.

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
            name: The Ollama model name. The result always carries this name,
                even when the cache entry came from ``name:latest``.
            digest: The digest ``/api/tags`` reported, if known. A cached
                result for a different digest is stale: the model was re-pulled.

        Returns:
            The probed definition, or the generic fallback when the probe fails.
        """
        key = _cache_key(name)
        cached = self._probes.get(key)
        if cached is not None and (not digest or cached.digest == digest):
            if cached.expires_at is None or time.monotonic() < cached.expires_at:
                return replace(cached.definition, name=name)

        loop = asyncio.get_running_loop()
        task = self._inflight.get(key)
        if task is None or task.get_loop() is not loop:
            task = loop.create_task(self._probe_and_cache(key, name, digest))
            self._inflight[key] = task
            task.add_done_callback(lambda done: self._forget_inflight(key, done))
        # Shielded so one cancelled caller does not cancel the probe the others await.
        return replace(await asyncio.shield(task), name=name)

    def _forget_inflight(self, key: str, task: 'asyncio.Task[_ResolvedModel]') -> None:
        if self._inflight.get(key) is task:
            del self._inflight[key]

    async def _probe_and_cache(self, key: str, name: str, digest: str) -> _ResolvedModel:
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
        caller falls back; plugin init and resolve never fail on a probe. The
        header callable runs inside the request, so the timeout covers it too.
        """
        timeout = _PROBE_TIMEOUT_SECONDS
        if self.timeout is not None and 0 < self.timeout < timeout:
            timeout = self.timeout

        try:
            show = await asyncio.wait_for(self.client().show(name), timeout=timeout)
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
            # OllamaModel wraps connection errors at the SDK boundary, so a failed
            # media-URL fetch isn't misreported as an Ollama server outage.
            return await model.generate(request, ctx)

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
            # Embedding requests never fetch media, so the whole SDK call is wrapped.
            async with wrap_connection_errors(server_address):
                return await embedder.embed(request)

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
        """List the models and embedders the server has pulled.

        Reads ``/api/tags``, then probes each model's ``/api/show``
        concurrently (at most four at once, five seconds each) so the Dev UI
        shows real capabilities. Results are cached per model and digest; a
        failed probe falls back to the generic set and is retried after 30s.
        Names containing ``embed`` are listed as embedders, as in JS and Go.

        Returns:
            ActionMetadata for each model and embedder.
        """
        async with wrap_connection_errors(self.server_address):
            try:
                response = await self.client().list()
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
