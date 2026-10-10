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

from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import BaseModel, Field, field_validator

from genkit import GenkitError, ModelResponse, ModelResponseChunk
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ModelHookParams, is_streaming_callback_error
from genkit.model import ModelRef, ModelRequest
from genkit.plugin_api import Action
from genkit_middleware._statuses import TRANSIENT_STATUSES

# Everything Retry would retry, plus failures another model may not have:
# this model doesn't exist or can't do what the request asks.
_DEFAULT_FALLBACK_STATUSES: list[str] = [*TRANSIENT_STATUSES, 'NOT_FOUND', 'UNIMPLEMENTED']


class FallbackModelEntry(BaseModel):
    """A backup model and the config that model runs with."""

    name: str
    config: dict[str, Any] | None = None


class FallbackConfig(BaseModel):
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
                raise ValueError(f"Fallback models are names or model_ref(...); pass '{item.name}', not the action")
            else:
                raise ValueError('each Fallback model must be a model name or a model_ref(...)')
        return entries


def config_from_ref(model: ModelRef[Any]) -> dict[str, Any] | None:
    """The config this backup runs with: the ref's version and config only."""
    bag: dict[str, Any] = {}
    if model.version is not None:
        bag['version'] = model.version
    if model.config is not None:
        bag.update(model.config.model_dump(exclude_unset=True, exclude_none=True))
    return bag or None


def fallback_request_config(entry: str | FallbackModelEntry) -> dict[str, Any] | None:
    """A string entry uses the model's defaults; a ref entry uses only its config."""
    if isinstance(entry, str):
        return None
    return entry.config


class Fallback(BaseMiddleware[FallbackConfig]):
    """Fallback middleware to try alternative models on failure."""

    async def _resolve_fallback_model(
        self,
        ctx: GenerateMiddlewareContext,
        model_name: str,
    ) -> Action[ModelRequest, ModelResponse, ModelResponseChunk]:
        """Look up a fallback model on the per-call registry."""
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
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Try the primary model, then fall back to alternates on retryable errors."""
        last_error: Exception | None = None
        try:
            return await next_fn(params, ctx)
        except Exception as exc:
            if not self._should_fall_back(exc):
                raise
            last_error = exc

        assert last_error is not None  # noqa: S101
        on_chunk = ctx.on_chunk
        for entry in self.config.models:
            if ctx.abort_signal.is_set():
                raise last_error
            model_name = entry if isinstance(entry, str) else entry.name
            fallback_action = await self._resolve_fallback_model(ctx, model_name)
            # A plain ModelRequest with a dict config, so the backup model parses
            # it into its own class like any other call. A copy of the primary's
            # typed request would pass a dict straight through when both models
            # share a config class.
            fallback_request = ModelRequest.model_validate(
                {**dict(params.request), 'config': fallback_request_config(entry) or {}},
            )
            try:
                result = await fallback_action.run(
                    input=fallback_request,
                    context=ctx.custom_context,
                    on_chunk=on_chunk,
                    abort_signal=ctx.abort_signal,
                )
                return result.response  # type: ignore[return-value]
            except Exception as e2:
                last_error = e2
                if not self._should_fall_back(e2):
                    raise

        raise last_error

    def _should_fall_back(self, exc: Exception) -> bool:
        # The caller's own on_chunk failure never switches models. A raw
        # exception has no status, so it also stays on this model. Only a
        # listed GenkitError status sends the request on.
        if is_streaming_callback_error(exc):
            return False
        return isinstance(exc, GenkitError) and exc.status in self.config.statuses
