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

"""Transport seam for the Amazon Bedrock plugin.

Every boto3 call goes through this module. boto3 is synchronous, so calls are
bridged onto worker threads with ``asyncio.to_thread`` to keep the event loop
unblocked for the seconds-to-minutes an LLM call takes. Keeping the whole SDK
surface behind one seam lets us swap in AWS's official async SDK once it
matures without touching converters or models.
"""

import asyncio
import os
import threading
from typing import TYPE_CHECKING, Any

from genkit.plugin_api import GenkitError

if TYPE_CHECKING:
    import boto3.session

NO_REGION_MESSAGE = (
    'bedrock: no AWS region resolved; set Bedrock(region=...), AWS_REGION, '
    'AWS_DEFAULT_REGION, or a region in ~/.aws/config'
)


class BedrockTransport:
    """Owns the shared bedrock-runtime client and the sync-to-async bridge.

    The sync boto3 client is not bound to an event loop (unlike async SDK
    clients), so one client instance safely serves both the application loop
    and the Dev UI reflection loop; boto3 clients are thread-safe for calls,
    only creation needs the lock.
    """

    def __init__(
        self,
        *,
        region: str | None = None,
        max_retries: int,
        read_timeout: float,
        connect_timeout: float,
        max_pool_connections: int,
        session: 'boto3.session.Session | None' = None,
    ) -> None:
        """Initializes the transport.

        Args:
            region: AWS region; falls back to the SDK resolution chain.
            max_retries: Retry limit for Bedrock API calls.
            read_timeout: Socket read timeout in seconds. Deliberately not a
                whole-call deadline: long generations stream for minutes and
                must not be killed mid-flight.
            connect_timeout: Socket connect timeout in seconds.
            max_pool_connections: HTTP connection pool size, raised off the
                botocore default of 10 so the pool is never the bottleneck.
                Concurrency is bounded first by the event loop's default
                thread-pool executor, which ``asyncio.to_thread`` dispatches to.
            session: Optional pre-configured ``boto3.session.Session`` for
                custom credentials or advanced SDK wiring.
        """
        self._region = region
        self._max_retries = max_retries
        self._read_timeout = read_timeout
        self._connect_timeout = connect_timeout
        self._max_pool_connections = max_pool_connections
        self._session = session
        self._client: Any = None
        self._lock = threading.Lock()

    def client(self) -> Any:  # noqa: ANN401
        """Returns the shared bedrock-runtime client, building it on first use.

        Raises:
            GenkitError: FAILED_PRECONDITION when no region resolves. Matching
                the Go plugin, there is deliberately no default region: a
                silent ``us-east-1`` fallback sends traffic (and data) to a
                region the user never chose.
        """
        with self._lock:
            if self._client is None:
                self._client = self._build_client()
            return self._client

    async def ensure_client(self) -> None:
        """Builds the client off-loop so init fails fast on config errors."""
        await asyncio.to_thread(self.client)

    async def converse(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Calls the Converse API on a worker thread.

        Args:
            kwargs: Keyword arguments passed verbatim to ``converse``.

        Returns:
            The raw Converse response dict.
        """
        return await asyncio.to_thread(self._converse_sync, kwargs)

    def _converse_sync(self, kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
        return self.client().converse(**kwargs)

    def _build_client(self) -> Any:  # noqa: ANN401
        import boto3.session
        from botocore.config import Config

        session = self._session or boto3.session.Session()
        # botocore only began reading AWS_REGION in 1.41, below this package's
        # floor, so resolve it here. A caller-supplied session states its own
        # region first; otherwise env wins over ~/.aws/config, as in the SDKs.
        env_region = os.environ.get('AWS_REGION')
        if self._session is not None:
            region = self._region or session.region_name or env_region
        else:
            region = self._region or env_region or session.region_name
        if not region:
            raise GenkitError(message=NO_REGION_MESSAGE, status='FAILED_PRECONDITION')

        return session.client(
            'bedrock-runtime',
            region_name=region,
            config=Config(
                retries={'max_attempts': self._max_retries, 'mode': 'standard'},
                read_timeout=self._read_timeout,
                connect_timeout=self._connect_timeout,
                max_pool_connections=self._max_pool_connections,
            ),
        )
