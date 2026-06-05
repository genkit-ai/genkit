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
from typing import Any

from pydantic import BaseModel, Field

from genkit import GenkitError
from genkit._core._action import Action, ActionKind
from genkit._core._model import ModelResponse
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ModelHookParams

_DEFAULT_FALLBACK_STATUSES: list[str] = [
    'UNAVAILABLE',
    'DEADLINE_EXCEEDED',
    'RESOURCE_EXHAUSTED',
    'ABORTED',
    'INTERNAL',
    'NOT_FOUND',
    'UNIMPLEMENTED',
]


class FallbackConfig(BaseModel):
    """Models and statuses that trigger fallback."""

    models: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=lambda: list(_DEFAULT_FALLBACK_STATUSES))


class Fallback(BaseMiddleware[FallbackConfig]):
    """Fallback middleware to try alternative models on failure."""

    async def _resolve_fallback_model(
        self,
        ctx: GenerateMiddlewareContext,
        model_name: str,
    ) -> Action[Any, Any, Any]:
        """Look up a fallback model on the per-call registry."""
        action = await ctx.registry.resolve_action(ActionKind.MODEL, model_name)
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
        for model_name in self.config.models:
            fallback_action = await self._resolve_fallback_model(ctx, model_name)
            try:
                result = await fallback_action.run(
                    input=params.request,
                    context=ctx.custom_context,
                    on_chunk=on_chunk,
                )
                return result.response  # type: ignore[return-value]
            except Exception as e2:
                last_error = e2
                if not isinstance(e2, GenkitError) or e2.status not in self.config.statuses:
                    raise

        raise last_error
