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

Automatically retries model API calls on transient failures with exponential backoff.
Non-retryable errors (like INVALID_ARGUMENT) are raised immediately, while transient
errors (UNAVAILABLE, DEADLINE_EXCEEDED, etc.) trigger retry with configurable delay.
"""

from __future__ import annotations

import asyncio
import random
from typing import ClassVar

from pydantic import Field

from genkit._core._error import GenkitError
from genkit._core._model import ModelHookParams, ModelResponse
from genkit.middleware import BaseMiddleware


class Retry(BaseMiddleware):
    """Retry middleware with exponential backoff for transient failures.

    Retries model API calls when they fail with retryable status codes (UNAVAILABLE,
    DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED, ABORTED, INTERNAL). Non-GenkitError exceptions
    (like network failures) are always retried. Non-retryable errors (INVALID_ARGUMENT, etc.)
    fail immediately without retry.

    Delays between retries grow exponentially with jitter to avoid thundering herd.
    """

    name: ClassVar[str] = 'middleware/retry'
    description: ClassVar[str | None] = 'Retries model calls on transient failures with exponential backoff'

    max_retries: int = 3
    statuses: list[str] = Field(
        default_factory=lambda: [
            'UNAVAILABLE',
            'DEADLINE_EXCEEDED',
            'RESOURCE_EXHAUSTED',
            'ABORTED',
            'INTERNAL',
        ]
    )
    initial_delay_ms: int = 1000
    max_delay_ms: int = 60000
    backoff_factor: float = 2.0
    no_jitter: bool = False

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn,
    ) -> ModelResponse:
        """Retry the model call up to max_retries times on transient failures."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await next_fn(params)
            except Exception as e:
                last_error = e

                # Last attempt: raise immediately
                if attempt == self.max_retries:
                    raise

                # Check if error is retryable
                if isinstance(e, GenkitError):
                    # GenkitError with non-retryable status: fail immediately
                    if e.status not in self.statuses:
                        raise
                # Non-GenkitError exceptions (network failures, etc.) are always retried

                # Compute delay with exponential backoff
                delay_ms = min(
                    self.initial_delay_ms * (self.backoff_factor**attempt),
                    self.max_delay_ms,
                )

                # Add jitter unless disabled
                if not self.no_jitter:
                    delay_ms += random.uniform(0, delay_ms)

                await asyncio.sleep(delay_ms / 1000.0)

        # Should never reach here, but raise the last error if we do
        if last_error:
            raise last_error
        raise RuntimeError('Retry loop completed without success or error')
