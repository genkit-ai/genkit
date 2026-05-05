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

from typing import ClassVar

from pydantic import Field

from genkit._core._error import GenkitError
from genkit._core._model import ModelHookParams, ModelResponse
from genkit.middleware import BaseMiddleware


class Fallback(BaseMiddleware):
    """Fallback middleware to try alternative models on failure.

    When the primary model call fails with a retryable error (UNAVAILABLE, DEADLINE_EXCEEDED,
    RESOURCE_EXHAUSTED, ABORTED, INTERNAL, NOT_FOUND, UNIMPLEMENTED), automatically tries
    each model in the fallback list in order. Only GenkitError exceptions with matching
    status codes trigger fallback; other errors fail immediately.

    Note: Currently limited by wrap_model hook scope. Full implementation requires
    registry access to resolve and invoke fallback models at the wrap_generate level.
    """

    name: ClassVar[str] = 'middleware/fallback'
    description: ClassVar[str | None] = 'Falls back to alternative models on failure'

    models: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(
        default_factory=lambda: [
            'UNAVAILABLE',
            'DEADLINE_EXCEEDED',
            'RESOURCE_EXHAUSTED',
            'ABORTED',
            'INTERNAL',
            'NOT_FOUND',
            'UNIMPLEMENTED',
        ]
    )

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn,
    ) -> ModelResponse:
        """Try the primary model, then fall back to alternates on retryable errors."""
        last_error: Exception | None = None

        # Try the primary model first
        try:
            return await next_fn(params)
        except Exception as e:
            last_error = e

            # Only GenkitError with matching status triggers fallback
            if not isinstance(e, GenkitError):
                raise
            if e.status not in self.statuses:
                raise

        # TODO: Fallback model switching requires registry access to resolve model actions.
        # The wrap_model hook operates on a single model invocation and cannot directly
        # switch to a different model action. Full fallback implementation needs to happen
        # at the wrap_generate level where the generate loop can resolve different model
        # names from the registry and invoke them sequentially.
        #
        # Current behavior: validates the error is retryable but cannot invoke fallback models.
        # For now, we re-raise the error after validation.

        if last_error:
            raise last_error
        raise RuntimeError('Fallback logic completed without success or error')
