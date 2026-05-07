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

"""Fallback middleware for Genkit model calls.

Automatically falls back to alternative models when the primary model fails with
retryable errors. Useful for handling rate limits, service outages, or unsupported
features by seamlessly switching to backup models.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import ClassVar, cast

from pydantic import Field

from genkit import GenkitError
from genkit._core._action import ActionKind
from genkit._core._model import ModelResponse
from genkit.middleware import BaseMiddleware, ModelHookParams

_DEFAULT_FALLBACK_STATUSES: list[str] = [
    'UNAVAILABLE',
    'DEADLINE_EXCEEDED',
    'RESOURCE_EXHAUSTED',
    'ABORTED',
    'INTERNAL',
    'NOT_FOUND',
    'UNIMPLEMENTED',
]


class Fallback(BaseMiddleware):
    """Fallback middleware to try alternative models on failure.

    When the primary model call fails with a retryable ``GenkitError`` status (one of
    the ``statuses`` list), each model in ``models`` is tried in order until one
    succeeds or all are exhausted.

    Only ``GenkitError`` failures with a matching status trigger fallback — raw network
    errors, ``TimeoutError``, or other non-``GenkitError`` exceptions propagate immediately
    without trying any fallback model. (Use ``Retry`` for transient non-API errors.)

    If ``models`` is empty and the primary fails with a retryable status, the original
    error is re-raised unchanged.

    ``self._registry`` is injected at resolve time so model lookup works whether
    Fallback is passed inline or registered via ``middleware_plugin``.
    """

    name: ClassVar[str] = 'fallback'
    description: ClassVar[str | None] = 'Falls back to alternative models on failure'

    models: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=lambda: list(_DEFAULT_FALLBACK_STATUSES))

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn: Callable[[ModelHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Try the primary model, then fall back to alternates on retryable errors."""
        last_error: Exception | None = None
        try:
            return await next_fn(params)
        except Exception as exc:
            if not isinstance(exc, GenkitError) or exc.status not in self.statuses:
                raise
            last_error = exc

        if not self._registry:
            raise GenkitError(
                status='INTERNAL',
                message=(
                    'Fallback middleware requires registry access. '
                    'Ensure it is resolved via generate(use=[...]) or registered with middleware_plugin.'
                ),
            )

        assert last_error is not None  # noqa: S101
        # Pass the streaming callback through so fallback models can stream
        # to the same caller as the primary model would have.
        on_chunk = (
            cast(Callable[[object], None], params.on_chunk) if params.on_chunk else None
        )
        for model_name in self.models:
            fallback_action = await self._registry.resolve_action(ActionKind.MODEL, model_name)
            if fallback_action is None:
                raise GenkitError(
                    status='NOT_FOUND',
                    message=f'Fallback model "{model_name}" not found in registry.',
                )
            try:
                result = await fallback_action.run(
                    input=params.request,
                    context=params.context,
                    on_chunk=on_chunk,
                )
                return result.response  # type: ignore[return-value]
            except Exception as e2:
                last_error = e2
                if not isinstance(e2, GenkitError) or e2.status not in self.statuses:
                    raise

        raise last_error
