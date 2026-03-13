# Copyright 2026 Google LLC
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

"""Async HTTP client for the Genkit reflection API.

Communicates with a running Genkit reflection server to list actions
and run model actions.  Works with any runtime (Python, JS, Go) that
implements the Genkit reflection protocol.

This replaces the dependency on ``genkit dev:test-model`` by talking
directly to the reflection server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Timeouts.
_HEALTH_TIMEOUT = 5.0
_ACTION_TIMEOUT = 120.0  # LLM calls can be slow.


class ReflectionClient:
    """Async client for the Genkit reflection API.

    Args:
        base_url: The reflection server URL (e.g. ``http://localhost:3100``).
        action_timeout: Timeout in seconds for action calls.
        health_timeout: Timeout in seconds for health checks.
    """

    def __init__(
        self,
        base_url: str,
        *,
        action_timeout: float = _ACTION_TIMEOUT,
        health_timeout: float = _HEALTH_TIMEOUT,
    ) -> None:
        """Initialize the client with the given base URL."""
        self._base_url = base_url.rstrip('/')
        self._action_timeout = action_timeout
        self._health_timeout = health_timeout
        self._client = httpx.AsyncClient(timeout=action_timeout)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def health(self) -> bool:
        """Check if the reflection server is healthy."""
        try:
            resp = await self._client.get(
                f'{self._base_url}/api/__health',
                timeout=self._health_timeout,
            )
            return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    async def list_actions(self) -> dict[str, Any]:
        """List all registered actions.

        Returns:
            Dict mapping action keys to action metadata.

        Raises:
            httpx.HTTPStatusError: On non-2xx response.
        """
        resp = await self._client.get(f'{self._base_url}/api/actions')
        resp.raise_for_status()
        return resp.json()

    async def run_action(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        stream: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run a model action via the reflection API.

        Args:
            key: The action key (e.g. ``/model/googleai/gemini-2.5-flash``).
            input_data: The ``GenerateRequest`` payload.
            stream: Whether to request streaming.

        Returns:
            A tuple of (final_response, chunks).  If ``stream`` is False,
            chunks will be an empty list.

        Raises:
            httpx.HTTPStatusError: On non-2xx response.
            RuntimeError: If the action returns an error payload.
        """
        payload = {'key': key, 'input': input_data}
        chunks: list[dict[str, Any]] = []

        if stream:
            resp = await self._client.post(
                f'{self._base_url}/api/runAction',
                json=payload,
                params={'stream': 'true'},
                timeout=self._action_timeout,
            )
            resp.raise_for_status()
            # The streaming response is newline-delimited JSON.
            # The last line is the final response.
            lines = resp.text.strip().split('\n')
            for line in lines[:-1]:
                if line.strip():
                    chunks.append(json.loads(line))
            final = json.loads(lines[-1]) if lines else {}
        else:
            resp = await self._client.post(
                f'{self._base_url}/api/runAction',
                json=payload,
                timeout=self._action_timeout,
            )
            resp.raise_for_status()
            final = resp.json()

        if 'error' in final:
            error = final['error']
            msg = error.get('message', str(error))
            raise RuntimeError(f'Action error: {msg}')

        result = final.get('result', final)
        return result, chunks

    async def wait_for_health(
        self,
        *,
        max_attempts: int = 60,
        delay_s: float = 0.5,
    ) -> bool:
        """Poll the health endpoint until it responds.

        Returns:
            True if healthy, False if timed out.
        """
        for i in range(max_attempts):
            if await self.health():
                logger.debug('Reflection server healthy after %d polls.', i + 1)
                return True
            await asyncio.sleep(delay_s)
        return False

    async def wait_for_actions(
        self,
        required_keys: set[str],
        *,
        max_attempts: int = 60,
        delay_s: float = 0.5,
    ) -> bool:
        """Poll until all required action keys are registered.

        Args:
            required_keys: Set of action keys to wait for.
            max_attempts: Maximum number of polling attempts.
            delay_s: Delay between attempts in seconds.

        Returns:
            True if all found, False if timed out.
        """
        if not required_keys:
            return True

        for attempt in range(max_attempts):
            try:
                actions = await self.list_actions()
                registered = set(actions.keys())
                missing = required_keys - registered
                if not missing:
                    logger.info(
                        'All %d model actions registered.',
                        len(required_keys),
                    )
                    return True
                if attempt > 0 and attempt % 10 == 0:
                    logger.info(
                        'Waiting for %d action(s): %s',
                        len(missing),
                        ', '.join(sorted(missing)),
                    )
            except (httpx.HTTPError, OSError):
                logger.debug('Polling for actions failed, will retry.')
            await asyncio.sleep(delay_s)

        logger.warning('Not all actions registered after %.0f seconds.', max_attempts * delay_s)
        return False
