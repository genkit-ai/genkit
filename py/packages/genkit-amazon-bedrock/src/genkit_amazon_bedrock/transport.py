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
import json
import os
import threading
from collections.abc import AsyncGenerator, Awaitable
from typing import TYPE_CHECKING, Any

import structlog

from genkit.plugin_api import GenkitError
from genkit_amazon_bedrock.config import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_MAX_POOL_CONNECTIONS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_TOTAL_TIMEOUT,
)

if TYPE_CHECKING:
    import boto3.session

logger = structlog.get_logger(__name__)

NO_REGION_MESSAGE = (
    'bedrock: no AWS region resolved; set Bedrock(region=...), AWS_REGION, '
    'AWS_DEFAULT_REGION, or a region in ~/.aws/config'
)

# Distinguishes stream exhaustion from a legitimately falsy event.
_STREAM_DONE = object()


def _close_abandoned_stream(call: 'asyncio.Future[dict[str, Any]]') -> None:
    """Closes an event stream nobody is left to consume.

    A worker thread cannot be interrupted, so a call cancelled in flight still
    opens a stream; without this it stays open until garbage collection.
    """
    if call.cancelled() or call.exception() is not None:
        return
    stream = call.result().get('stream')
    if stream is not None:
        stream.close()


def _botocore_session(session: Any) -> Any:  # noqa: ANN401
    """Returns the botocore session underneath a boto3 session, if reachable."""
    return getattr(session, '_session', None)


def _has_ambient_setting(session: Any, name: str) -> bool:  # noqa: ANN401
    """True when ``AWS_<NAME>`` or the active ~/.aws/config profile sets ``name``.

    Both sources are read directly rather than through
    ``get_config_variable``, which folds in botocore's own default and so
    reports ``retry_mode`` as configured even when nobody configured it.
    """
    from botocore.exceptions import ProfileNotFound

    if os.environ.get(f'AWS_{name.upper()}'):
        return True
    botocore_session = _botocore_session(session)
    if botocore_session is None:
        return False
    try:
        scoped_config = botocore_session.get_scoped_config()
    except ProfileNotFound:
        return False
    return bool(scoped_config.get(name))


def _ambient_client_config(session: Any) -> Any:  # noqa: ANN401
    """Returns the session's default client config, if one was installed."""
    botocore_session = _botocore_session(session)
    if botocore_session is None:
        return None
    return botocore_session.get_default_client_config()


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
        max_retries: int | None = None,
        read_timeout: float | None = None,
        connect_timeout: float | None = None,
        max_pool_connections: int | None = None,
        total_timeout: float | None = DEFAULT_TOTAL_TIMEOUT,
        session: 'boto3.session.Session | None' = None,
    ) -> None:
        """Initializes the transport.

        Every botocore knob below takes None to mean "whatever the ambient AWS
        configuration says", falling back to the package default only when that
        configuration is silent too.

        Args:
            region: AWS region; falls back to the SDK resolution chain.
            max_retries: Retry limit for Bedrock API calls.
            read_timeout: Socket read timeout in seconds, reset on every byte
                received, so it bounds silence rather than the whole call.
            connect_timeout: Socket connect timeout in seconds.
            max_pool_connections: HTTP connection pool size, raised off the
                botocore default of 10 so the pool is never the bottleneck.
                Concurrency is bounded first by the event loop's default
                thread-pool executor, which ``asyncio.to_thread`` dispatches to.
            total_timeout: Whole-call deadline in seconds; None removes it.
            session: Optional pre-configured ``boto3.session.Session`` for
                custom credentials or advanced SDK wiring.
        """
        self._region = region
        self._max_retries = max_retries
        self._read_timeout = read_timeout
        self._connect_timeout = connect_timeout
        self._max_pool_connections = max_pool_connections
        self._total_timeout = total_timeout
        self._session = session
        self._client: Any = None
        self._lock = threading.Lock()

    def client(self) -> Any:  # noqa: ANN401
        """Returns the shared bedrock-runtime client, building it on first use.

        Raises:
            GenkitError: FAILED_PRECONDITION when no region resolves. There is
                deliberately no default region: a silent ``us-east-1`` fallback
                sends traffic (and data) to a region the user never chose.
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

        Raises:
            GenkitError: DEADLINE_EXCEEDED when the call outruns the total
                timeout. The caller is freed at that point, but boto3 offers no
                way to abort an in-flight call, so the worker thread stays busy
                until the socket timeouts below it fire.
        """
        call = asyncio.to_thread(self._converse_sync, kwargs)
        if self._total_timeout is None:
            return await call
        try:
            return await asyncio.wait_for(call, self._total_timeout)
        except asyncio.TimeoutError as e:
            raise GenkitError(
                message=f'bedrock converse failed: call exceeded the {self._total_timeout}s total timeout',
                status='DEADLINE_EXCEEDED',
            ) from e

    def _converse_sync(self, kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
        return self.client().converse(**kwargs)

    async def converse_stream(self, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:  # noqa: ANN401
        """Calls ConverseStream and yields raw event dicts.

        botocore hands back a blocking ``EventStream``, so both the initial
        call and every ``next()`` cross the thread bridge — a stream idles for
        seconds between events and must not hold the event loop.

        The total timeout is one deadline across the whole stream rather than
        a per-event allowance: the read timeout already resets on every byte,
        so the slow dribble it cannot see is exactly what the deadline ends.

        Args:
            kwargs: Keyword arguments passed verbatim to ``converse_stream``.

        Yields:
            One raw event dict per event, e.g. ``{'contentBlockDelta': {...}}``.

        Raises:
            GenkitError: INTERNAL when the response carries no event stream;
                DEADLINE_EXCEEDED when the stream outruns the total timeout.
            botocore.exceptions.EventStreamError: For mid-stream failures; it
                subclasses ``ClientError``, so callers map it like any other
                AWS error.
        """
        deadline = None if self._total_timeout is None else asyncio.get_running_loop().time() + self._total_timeout
        # Shielded so a cancellation here can still close the stream the
        # uninterruptible worker goes on to open.
        call = asyncio.ensure_future(asyncio.to_thread(self._converse_stream_sync, kwargs))
        try:
            response = await self._before_deadline(asyncio.shield(call), deadline)
        except (asyncio.CancelledError, GenkitError):
            call.add_done_callback(_close_abandoned_stream)
            raise
        stream = response.get('stream')
        if stream is None:
            raise GenkitError(message='bedrock: converse stream response has no stream', status='INTERNAL')
        events = iter(stream)
        try:
            while True:
                event: Any = await self._before_deadline(asyncio.to_thread(next, events, _STREAM_DONE), deadline)
                if event is _STREAM_DONE:
                    return
                yield event
        finally:
            # Called inline rather than through the bridge: it is a socket
            # teardown, and awaiting here would be re-cancelled on cancellation.
            # On a deadline this is also what frees the parked worker thread.
            stream.close()

    async def _before_deadline(self, awaitable: Awaitable[Any], deadline: float | None) -> Any:  # noqa: ANN401
        """Awaits with the budget remaining until ``deadline``; None waits freely.

        Raises:
            GenkitError: DEADLINE_EXCEEDED when the budget runs out first.
        """
        if deadline is None:
            return await awaitable
        remaining = deadline - asyncio.get_running_loop().time()
        try:
            return await asyncio.wait_for(awaitable, remaining)
        except asyncio.TimeoutError as e:
            raise GenkitError(
                message=f'bedrock converse stream failed: stream exceeded the {self._total_timeout}s total timeout',
                status='DEADLINE_EXCEEDED',
            ) from e

    def _converse_stream_sync(self, kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
        return self.client().converse_stream(**kwargs)

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Calls the InvokeModel API on a worker thread.

        Args:
            kwargs: Keyword arguments passed verbatim to ``invoke_model``.

        Returns:
            The parsed JSON response body.

        Raises:
            GenkitError: INTERNAL when the response carries no body, or a body
                that is not a JSON object.
        """
        return await asyncio.to_thread(self._invoke_model_sync, kwargs)

    def _invoke_model_sync(self, kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
        response = self.client().invoke_model(**kwargs)
        body = response.get('body')
        if body is None:
            raise GenkitError(message='bedrock: invoke model response has no body', status='INTERNAL')
        # Read and parse here, not on the loop: the body is a StreamingBody and
        # read() is a blocking socket read.
        raw = body.read()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise GenkitError(message=f'bedrock: invoke model response is not JSON: {e}', status='INTERNAL') from e
        if not isinstance(payload, dict):
            raise GenkitError(message='bedrock: invoke model response body is not a JSON object', status='INTERNAL')
        return payload

    def _resolve_region(self, session: Any) -> str:  # noqa: ANN401
        # botocore only began reading AWS_REGION in 1.41, above this package's
        # floor, so resolve it here. A caller-supplied session states its own
        # region first; otherwise env wins over ~/.aws/config, as in the SDKs.
        env_region = os.environ.get('AWS_REGION')
        if self._session is not None:
            region = self._region or session.region_name or env_region
        else:
            region = self._region or env_region or session.region_name
        if not region:
            raise GenkitError(message=NO_REGION_MESSAGE, status='FAILED_PRECONDITION')
        return region

    def _client_config(self, session: Any) -> Any:  # noqa: ANN401
        """Builds the botocore config, leaving anything the caller stated alone.

        botocore reads AWS_MAX_ATTEMPTS, AWS_RETRY_MODE, ~/.aws/config and a
        session's default client config only where the config passed to
        ``client()`` leaves that key unset; whatever is passed wins outright.
        So the package defaults go on underneath as a floor, the ambient
        configuration on top of them, and the explicit arguments last.
        """
        from botocore.config import Config

        defaults: dict[str, Any] = {
            'read_timeout': DEFAULT_READ_TIMEOUT,
            'connect_timeout': DEFAULT_CONNECT_TIMEOUT,
            'max_pool_connections': DEFAULT_MAX_POOL_CONNECTIONS,
        }
        explicit: dict[str, Any] = {}
        for key, value in (
            ('read_timeout', self._read_timeout),
            ('connect_timeout', self._connect_timeout),
            ('max_pool_connections', self._max_pool_connections),
        ):
            if value is not None:
                explicit[key] = value

        ambient = _ambient_client_config(session)
        retries = self._retry_config(session, ambient)
        if retries:
            explicit['retries'] = retries

        config = Config(**defaults)
        if ambient is not None:
            config = config.merge(ambient)
        return config.merge(Config(**explicit))

    def _retry_config(self, session: Any, ambient: Any) -> dict[str, Any]:  # noqa: ANN401
        """Resolves the retry keys one at a time.

        botocore resolves ``max_attempts`` and ``retry_mode`` independently
        from the env and config file, but a retries block sent to ``client()``
        outranks both for every key it carries, and merging replaces the block
        wholesale rather than key by key. So each key is filled in here only
        when nothing else supplies it: sending ``{'mode': ...}`` alone still
        lets AWS_MAX_ATTEMPTS through, and vice versa.
        """
        retries: dict[str, Any] = {}
        if not _has_ambient_setting(session, 'max_attempts'):
            retries['max_attempts'] = DEFAULT_MAX_RETRIES
        if not _has_ambient_setting(session, 'retry_mode'):
            retries['mode'] = 'standard'
        # Re-apply the session's own block by hand, since the merge below would
        # otherwise drop whichever keys it does not carry.
        ambient_retries = getattr(ambient, 'retries', None)
        if ambient_retries:
            retries.update(ambient_retries)
        if self._max_retries is not None:
            retries['max_attempts'] = self._max_retries
        return retries

    def _build_client(self) -> Any:  # noqa: ANN401
        import boto3.session

        session = self._session or boto3.session.Session()
        region = self._resolve_region(session)
        config = self._client_config(session)
        client = session.client('bedrock-runtime', region_name=region, config=config)
        logger.debug(
            'Bedrock client created',
            region=region,
            caller_session=self._session is not None,
            read_timeout=config.read_timeout,
            connect_timeout=config.connect_timeout,
            max_pool_connections=config.max_pool_connections,
            retries=config.retries,
        )
        return client
