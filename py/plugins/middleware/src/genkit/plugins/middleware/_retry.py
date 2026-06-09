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

"""Retry middleware for Genkit model calls."""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from genkit import GenkitError
from genkit._core._model import ModelResponse
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ModelHookParams

_DEFAULT_RETRY_STATUSES: list[str] = [
    'UNAVAILABLE',
    'DEADLINE_EXCEEDED',
    'RESOURCE_EXHAUSTED',
    'ABORTED',
    'INTERNAL',
]


class RetryConfig(BaseModel):
    """Knobs for retry backoff and which error statuses are retried."""

    max_retries: int = Field(default=3, ge=0)
    statuses: list[str] = Field(default_factory=lambda: list(_DEFAULT_RETRY_STATUSES))
    initial_delay_ms: int = 1000
    max_delay_ms: int = 60000
    backoff_factor: float = 2.0
    no_jitter: bool = False


class Retry(BaseMiddleware[RetryConfig]):
    """Retry middleware with exponential backoff for transient failures."""

    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Retry the model call up to max_retries times on transient failures."""
        current_delay_ms = float(self.config.initial_delay_ms)

        for attempt in range(self.config.max_retries + 1):
            try:
                return await next_fn(params, ctx)
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise

                if isinstance(e, GenkitError) and e.status not in self.config.statuses:
                    raise

                delay_ms = current_delay_ms
                if not self.config.no_jitter:
                    delay_ms += 1000.0 * math.pow(2, attempt) * random.random()
                delay_ms = min(delay_ms, self.config.max_delay_ms)

                await asyncio.sleep(delay_ms / 1000.0)
                current_delay_ms = min(current_delay_ms * self.config.backoff_factor, self.config.max_delay_ms)

        raise AssertionError('Retry loop exited without returning or raising')  # noqa: EM101
