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

"""Asynchronous server gateway interface implementation for Genkit."""

import signal
from collections.abc import Coroutine
from typing import Any, TypeVar

import anyio
import httpx
import uvicorn

from genkit.aio.loop import run_loop
from genkit.blocks.formats import built_in_formats
from genkit.blocks.generate import define_generate_action
from genkit.core.environment import is_dev_environment
from genkit.core.logging import get_logger
from genkit.core.plugin import Plugin
from genkit.core.reflection import create_reflection_asgi_app
from genkit.core.registry import Registry
from genkit.web.manager._ports import find_free_port_sync

from ._registry import GenkitRegistry
from ._runtime import RuntimeManager
from ._server import ServerSpec

logger = get_logger(__name__)

T = TypeVar('T')

_UNSET = object()  # Sentinel to distinguish "not yet set" from None.


class GenkitBase(GenkitRegistry):
    """Base class with shared infra for Genkit instances (sync and async)."""

    def __init__(
        self,
        plugins: list[Plugin] | None = None,
        model: str | None = None,
        reflection_server_spec: ServerSpec | None = None,
    ) -> None:
        """Initialize a new Genkit instance.

        Args:
            plugins: List of plugins to initialize.
            model: Model name to use.
            reflection_server_spec: Server spec for the reflection
                server. If not provided in dev mode, a default will be used.
        """
        super().__init__()
        self._reflection_server_spec: ServerSpec | None = reflection_server_spec
        self._initialize_registry(model, plugins)
        # Ensure the default generate action is registered for async usage.
        define_generate_action(self.registry)

    def _initialize_registry(self, model: str | None, plugins: list[Plugin] | None) -> None:
        """Initialize the registry for the Genkit instance.

        Args:
            model: Model name to use.
            plugins: List of plugins to initialize.

        Raises:
            ValueError: If an invalid plugin is provided.

        Returns:
            None
        """
        self.registry.default_model = model
        for fmt in built_in_formats:
            self.define_format(fmt)

        if not plugins:
            logger.warning('No plugins provided to Genkit')
        else:
            for plugin in plugins:
                if isinstance(plugin, Plugin):  # pyright: ignore[reportUnnecessaryIsInstance]
                    self.registry.register_plugin(plugin)
                else:
                    raise ValueError(f'Invalid {plugin=} provided to Genkit: must be of type `genkit.ai.Plugin`')

    def run_main(self, coro: Coroutine[Any, Any, T]) -> T | None:
        """Run the user's main coroutine.

        In development mode (`GENKIT_ENV=dev`), this starts the Genkit
        reflection server and runs the user's coroutine concurrently within the
        same event loop, blocking until the server is stopped (e.g., via
        Ctrl+C).

        In production mode, this simply runs the user's coroutine to completion
        using `uvloop.run()` for performance if available, otherwise
        `asyncio.run()`.

        Args:
            coro: The main coroutine provided by the user.

        Returns:
            The result of the user's coroutine, or None if the
            development server was shut down gracefully (SIGTERM/Ctrl+C)
            before the coroutine completed.
        """
        if not is_dev_environment():
            logger.info('Running in production mode.')
            return run_loop(coro)

        logger.info('Running in development mode.')

        spec = self._reflection_server_spec
        if not spec:
            spec = ServerSpec(scheme='http', host='127.0.0.1', port=find_free_port_sync(3100, 3999))
        assert spec is not None  # Type narrowing: spec is guaranteed non-None after the above check

        async def dev_runner() -> T | None:
            """Internal async function to run tasks using AnyIO TaskGroup."""
            # Assert for type narrowing inside closure (pyrefly doesn't propagate from outer scope)
            assert spec is not None
            # Capture spec in local var for nested functions (pyrefly doesn't narrow closures)
            server_spec: ServerSpec = spec
            user_result: T | object = _UNSET
            user_task_finished_event = anyio.Event()
            cancelled = False  # Track whether shutdown was intentional (SIGTERM/Ctrl+C)

            async def run_user_coro_wrapper() -> None:
                """Wraps user coroutine to capture result and signal completion."""
                nonlocal user_result
                try:
                    user_result = await coro
                    logger.debug('User coroutine completed successfully.')
                except Exception as err:
                    # Log error but don't necessarily stop the server
                    logger.error(f'User coroutine failed: {err}', exc_info=True)
                    pass  # Continue running server for now
                finally:
                    user_task_finished_event.set()

            reflection_server = _make_reflection_server(self.registry, server_spec)

            # Since anyio/asyncio handles SIGINT well, let's add a task to catch SIGTERM
            async def handle_sigterm(tg_to_cancel: anyio.abc.TaskGroup) -> None:  # type: ignore[name-defined]
                nonlocal cancelled
                with anyio.open_signal_receiver(signal.SIGTERM) as signals:
                    async for _signum in signals:
                        logger.info('Received SIGTERM, cancelling tasks...')
                        cancelled = True
                        tg_to_cancel.cancel_scope.cancel()
                        return

            try:
                # Use lazy_write=True to prevent race condition where file exists before server is up
                async with RuntimeManager(server_spec, lazy_write=True) as runtime_manager:
                    # We use anyio.TaskGroup because it is compatible with
                    # asyncio's event loop and works with Python 3.10
                    # (asyncio.TaskGroup was added in 3.11, and we can switch to
                    # that when we drop support for 3.10).
                    async with anyio.create_task_group() as tg:
                        # Start reflection server in the background.
                        tg.start_soon(reflection_server.serve, name='genkit-reflection-server')
                        await logger.ainfo(f'Started Genkit reflection server at {server_spec.url}')

                        # Start SIGTERM handler
                        tg.start_soon(handle_sigterm, tg, name='genkit-sigterm-handler')

                        # Wait for server to be responsive

                        max_retries = 20  # 2 seconds total roughly
                        for _i in range(max_retries):
                            try:
                                health_url = f'{server_spec.url}/api/__health'
                                async with httpx.AsyncClient(timeout=0.5) as client:
                                    response = await client.get(health_url)
                                    if response.status_code == 200:
                                        break
                            except Exception:
                                await anyio.sleep(0.1)
                        else:
                            logger.warning(f'Reflection server at {server_spec.url} did not become healthy in time.')

                        # Now write the file (or verify it persisted)
                        _ = runtime_manager.write_runtime_file()

                        # Start the (potentially short-lived) user coroutine wrapper
                        tg.start_soon(run_user_coro_wrapper, name='genkit-user-coroutine')
                        await logger.ainfo('Started Genkit user coroutine')

            except anyio.get_cancelled_exc_class():
                cancelled = True
                logger.info('Development server task group cancelled (e.g., Ctrl+C).')
            except Exception:
                logger.exception('Development server task group error')
                raise

            # Graceful shutdown (SIGTERM or Ctrl+C) — not an error.
            if cancelled:
                logger.info('Development server shut down gracefully.')
                return user_result if user_result is not _UNSET else None  # type: ignore[return-value] - _UNSET is object, not T

            # Normal exit (not cancelled) — user coroutine should have finished.
            if user_task_finished_event.is_set():
                await logger.adebug('User coroutine finished before TaskGroup exit.')
                if user_result is _UNSET:
                    raise RuntimeError('User coroutine finished without a result.')
                return user_result  # type: ignore[return-value] - narrowed by _UNSET check above

            await logger.adebug('User coroutine did not finish before TaskGroup exit.')
            raise RuntimeError('User coroutine did not finish before TaskGroup exit.')

        return anyio.run(dev_runner)


def _make_reflection_server(registry: Registry, spec: ServerSpec) -> uvicorn.Server:
    """Make a reflection server for the given registry and spec.

    This is a helper function to make it easier to test the reflection server
    in isolation.

    Args:
        registry: The registry to use for the reflection server.
        spec: The spec to use for the reflection server.

    Returns:
        A uvicorn server instance.
    """
    app = create_reflection_asgi_app(registry=registry)
    # pyrefly: ignore[bad-argument-type] - Starlette app is valid ASGI app for uvicorn
    config = uvicorn.Config(app, host=spec.host, port=spec.port, loop='asyncio')
    return uvicorn.Server(config)
