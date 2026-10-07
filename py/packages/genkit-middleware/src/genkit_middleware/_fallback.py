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

from genkit import GenkitError, ModelResponse
from genkit._core._model import ModelRef
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ModelHookParams
from genkit.plugin_api import Action, ActionKind

_DEFAULT_FALLBACK_STATUSES: list[str] = [
    'UNAVAILABLE',
    'DEADLINE_EXCEEDED',
    'RESOURCE_EXHAUSTED',
    'ABORTED',
    'INTERNAL',
    'NOT_FOUND',
    'UNIMPLEMENTED',
]


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
    ) -> Action[Any, Any, Any]:
        """Look up a fallback model on the per-call registry."""
        action = await ctx.ai.registry.resolve_action(ActionKind.MODEL, model_name)
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
            if not isinstance(exc, GenkitError) or exc.status not in self.config.statuses:
                raise
            last_error = exc

        assert last_error is not None  # noqa: S101
        on_chunk = ctx.on_chunk
        for entry in self.config.models:
            if ctx.abort_signal.is_set():
                raise last_error
            model_name = entry if isinstance(entry, str) else entry.name
            fallback_action = await self._resolve_fallback_model(ctx, model_name)
            fallback_request = params.request.model_copy(update={'config': fallback_request_config(entry)})
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
                if not isinstance(e2, GenkitError) or e2.status not in self.config.statuses:
                    raise

        raise last_error
