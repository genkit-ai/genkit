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

"""Retry middleware for Genkit model calls.

Automatically retries model API calls on transient failures with
exponential backoff. Non-retryable errors (e.g. ``INVALID_ARGUMENT``) are
raised immediately, while transient errors (``UNAVAILABLE``,
``DEADLINE_EXCEEDED``, …) trigger retry with a configurable delay.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable

from pydantic import Field

from genkit import GenkitError
from genkit._core._model import ModelResponse
from genkit.middleware import BaseMiddleware, ModelHookParams

_DEFAULT_RETRY_STATUSES: list[str] = [
    'UNAVAILABLE',
    'DEADLINE_EXCEEDED',
    'RESOURCE_EXHAUSTED',
    'ABORTED',
    'INTERNAL',
]


class Retry(BaseMiddleware):
    """Retry middleware with exponential backoff for transient failures.

    Retries model API calls when they fail with retryable status codes:

    * Non-``GenkitError`` exceptions (network failures, etc.) are always
      retried.
    * Non-retryable ``GenkitError`` statuses (``INVALID_ARGUMENT``, etc.)
      fail immediately.

    Jitter grows with the attempt number: ``1s * 2^attempt * random()`` is
    added on top of the base delay, with the total capped at
    ``max_delay_ms``. This prevents thundering-herd retries while
    guaranteeing sleep never exceeds the configured maximum.
    """

    max_retries: int = Field(default=3, ge=0)
    statuses: list[str] = Field(default_factory=lambda: list(_DEFAULT_RETRY_STATUSES))
    initial_delay_ms: int = 1000
    max_delay_ms: int = 60000
    backoff_factor: float = 2.0
    # On by default; set to False for deterministic backoff (useful in tests).
    jitter: bool = True

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn: Callable[[ModelHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Retry the model call up to max_retries times on transient failures."""
        current_delay_ms = float(self.initial_delay_ms)

        for attempt in range(self.max_retries + 1):
            try:
                return await next_fn(params)
            except Exception as e:
                if attempt == self.max_retries:
                    raise

                if isinstance(e, GenkitError) and e.status not in self.statuses:
                    raise  # non-retryable status

                # Additive jitter: 1 s × 2^attempt × random() on top of the
                # base delay, then capped so the total never exceeds max_delay_ms.
                delay_ms = current_delay_ms
                if self.jitter:
                    delay_ms += 1000.0 * math.pow(2, attempt) * random.random()
                delay_ms = min(delay_ms, self.max_delay_ms)

                await asyncio.sleep(delay_ms / 1000.0)
                current_delay_ms = min(current_delay_ms * self.backoff_factor, self.max_delay_ms)

        raise AssertionError('Retry loop exited without returning or raising')  # noqa: EM101
