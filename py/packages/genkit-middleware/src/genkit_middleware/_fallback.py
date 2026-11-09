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

"""Fallback middleware for Genkit model calls."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import BaseModel, Field, field_validator

from genkit import GenkitError, ModelResponse, ModelResponseChunk
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams
from genkit.model import ModelRef, ModelRequest
from genkit.plugin_api import Action, GenkitConfig
from genkit_middleware._statuses import TRANSIENT_STATUSES
from genkit_middleware._streaming import watch_stream

# Everything Retry would retry, plus failures another model may not have:
# this model doesn't exist or can't do what the request asks.
_DEFAULT_FALLBACK_STATUSES: list[str] = [*TRANSIENT_STATUSES, 'NOT_FOUND', 'UNIMPLEMENTED']


class FallbackModelEntry(GenkitConfig):
    """A backup model and the config that model runs with."""

    name: str
    config: dict[str, Any] | None = None


class FallbackConfig(GenkitConfig):
    """Models and statuses that trigger fallback."""

    models: list[str | FallbackModelEntry] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=lambda: list(_DEFAULT_FALLBACK_STATUSES))

    @field_validator('models', mode='before')
    @classmethod
    def coerce_models(cls, value: object) -> object:
        """A string is the model name; a ref carries that model's own config."""
        if not isinstance(value, list):
            return value
        entries: list[str | FallbackModelEntry | dict[str, Any]] = []
        for item in value:
            if isinstance(item, str | FallbackModelEntry):
                entries.append(item)
            elif isinstance(item, ModelRef):
                entries.append(FallbackModelEntry(name=item.name, config=config_from_ref(item)))
            elif isinstance(item, dict):
                entries.append(cast(dict[str, Any], item))
            elif isinstance(item, Action):
                raise ValueError(f"Fallback models are names or a ModelRef(...); pass '{item.name}', not the action")
            else:
                raise ValueError('each Fallback model must be a model name or a ModelRef(...)')
        return entries


def config_from_ref(model: ModelRef[Any]) -> dict[str, Any] | None:
    """The config this backup runs with: the ref's version and config only.

    Stored as wire keys (``by_alias=True``) because ``FallbackConfig`` is a
    JSON middleware config, with ``ref.version < ref.config`` and an explicit
    ``None`` kept. The backup model's own defaults and validation apply when
    it runs.
    """
    bag: dict[str, Any] = {}
    if model.version is not None:
        bag['version'] = model.version
    if model.config is not None:
        bag.update(model.config.model_dump(by_alias=True, exclude_unset=True, exclude_none=False))
    return bag or None


# Generation settings every provider understands (GenerationCommonConfig).
_PORTABLE_CONFIG_KEYS: dict[str, str] = {
    'temperature': 'temperature',
    'max_output_tokens': 'max_output_tokens',
    'maxOutputTokens': 'max_output_tokens',
    'top_k': 'top_k',
    'topK': 'top_k',
    'top_p': 'top_p',
    'topP': 'top_p',
    'stop_sequences': 'stop_sequences',
    'stopSequences': 'stop_sequences',
}


def _config_dict(config: object) -> dict[str, Any]:
    if isinstance(config, BaseModel):
        return config.model_dump(exclude_none=True)
    if isinstance(config, dict):
        return {k: v for k, v in cast(dict[str, Any], config).items() if v is not None}
    return {}


def _accepted_keys(schema: object) -> set[str] | None:
    """Keys a backup model's config class validates, or None when it has none."""
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        return None
    keys: set[str] = set()
    for name, info in schema.model_fields.items():
        keys.add(name)
        for alias in (info.alias, info.validation_alias, info.serialization_alias):
            if isinstance(alias, str):
                keys.add(alias)
    return keys


def inherited_config(primary: object, backup_schema: object) -> dict[str, Any]:
    """The caller's config, narrowed to what the backup model accepts.

    Keys the backup's config class declares carry over as-is, so
    ``max_output_tokens`` and ``temperature`` survive a failover. A backup
    with no config class gets only the portable generation keys, so a
    provider-specific key (``safety_settings``) does not fail validation.
    ``version`` and ``extra`` belong to the primary's provider, so they never
    carry over.
    """
    source = _config_dict(primary)
    # version names the primary's model; extra is that provider's passthrough.
    source.pop('version', None)
    source.pop('extra', None)
    accepted = _accepted_keys(backup_schema)
    if accepted is not None:
        return {k: v for k, v in source.items() if k in accepted}
    return {_PORTABLE_CONFIG_KEYS[k]: v for k, v in source.items() if k in _PORTABLE_CONFIG_KEYS}


def fallback_request_config(
    entry: str | FallbackModelEntry,
    *,
    primary: object = None,
    backup_schema: object = None,
) -> dict[str, Any] | None:
    """A ref entry uses only its own config; a string entry inherits the caller's.

    Matches JS: ``normalizedModel.config ?? req.config``, narrowed so the
    backup model can validate it.
    """
    if isinstance(entry, FallbackModelEntry) and entry.config is not None:
        return entry.config
    return inherited_config(primary, backup_schema) or None


class Fallback(Middleware[FallbackConfig]):
    """Fallback middleware to try alternative models on failure.

    A backup named by string inherits the caller's config, narrowed to keys
    the backup accepts; a ``ModelRef`` backup uses only its own config. Once
    an attempt has streamed chunks, its failure is raised rather than failed
    over. Backups run directly, not through middleware listed after
    ``Fallback``, so ``use=[Fallback(...), Retry(...)]`` retries only the
    primary; ``use=[Retry(...), Fallback(...)]`` retries the whole sequence.
    """

    name = 'fallback'
    description = 'Falls back to alternative models on failure'

    async def _resolve_fallback_model(
        self,
        ctx: MiddlewareContext,
        model_name: str,
    ) -> Action[ModelRequest, ModelResponse, ModelResponseChunk]:
        """Look up a fallback model among the app's models."""
        action = await ctx.ai.lookup_model(model_name)
        if action is None:
            raise GenkitError(
                status='NOT_FOUND',
                message=f'No model named "{model_name}" is registered on this app.',
            )
        return action

    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Try the primary model, then fall back to alternates on retryable errors."""
        last_error: Exception | None = None
        try:
            with watch_stream(ctx) as stream:
                return await next_fn(params, ctx)
        except Exception as exc:
            # Once chunks reached the caller, a backup's answer would be
            # spliced onto the failed one; surface the failure instead.
            if stream.committed or not self._should_fall_back(exc):
                raise
            last_error = exc

        assert last_error is not None  # noqa: S101
        on_chunk = ctx.on_chunk
        for entry in self.config.models:
            model_name = entry if isinstance(entry, str) else entry.name
            fallback_action = await self._resolve_fallback_model(ctx, model_name)
            # A plain ModelRequest with a dict config, so the backup model parses
            # it into its own class like any other call. A copy of the primary's
            # typed request would pass a dict straight through when both models
            # share a config class.
            config = fallback_request_config(
                entry, primary=params.request.config, backup_schema=fallback_action.config_schema
            )
            fallback_request = ModelRequest.model_validate({**dict(params.request), 'config': config or {}})
            try:
                with watch_stream(ctx) as stream:
                    result = await fallback_action._run(
                        input=fallback_request,
                        context=ctx.context,
                        on_chunk=ctx.on_chunk if on_chunk is not None else None,
                    )
                return result.response  # type: ignore[return-value]
            except Exception as e2:
                last_error = e2
                if stream.committed or not self._should_fall_back(e2):
                    raise

        raise last_error

    def _should_fall_back(self, exc: Exception) -> bool:
        # A raw exception has no status, so it stays on this model. Only a
        # listed GenkitError status sends the request on.
        curr = asyncio.current_task()
        if curr is not None and sys.version_info >= (3, 11) and curr.cancelling() > 0:
            return False
        return isinstance(exc, GenkitError) and exc.status in self.config.statuses
