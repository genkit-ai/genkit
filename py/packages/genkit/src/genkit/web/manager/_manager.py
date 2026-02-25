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

"""ASGI Server Management utilities for running multiple servers concurrently.

This module provides tools for managing multiple ASGI server instances within a
single application. The primary use case is running multiple web services on
different ports that need to share resources and coordinate shutdown.

### Example

```python
import asyncio
import structlog
import signal
from genkit.web.manager import ServersManager
from genkit.web.manager import Server, ServerConfig
from genkit.web.manager import ASGIServerAdapter, ServerType

logger = structlog.get_logger(__name__)

app1 = create_fastapi_app()
app2 = create_starlette_app()


def main():
    # Create a manager with specified server adapter type
    manager = ServersManager()
    # Add server configurations
    manager.add_server(
        Server(
            config=ServerConfig(name='api', ports=[8000]),
            lifecycle=YourLifecycleClass(app=app1),
            adapter=ASGIServerAdapter.create(ServerType.UVICORN),
        )
    )
    manager.add_server(
        Server(
            config=ServerConfig(name='admin', ports=[8001]),
            lifecycle=YourLifecycleClass(app=app2),
            adapter=ASGIServerAdapter.create(ServerType.UVICORN),
        )
    )

    manager.add_shutdown_callback(lambda: logger.info('Shutting down!'))

    # Start all servers and block until shutdown triggered.
    async def run():
        await manager.start_all()

    asyncio.run(run())


if __name__ == '__main__':
    main()
```

The module handles signal management, concurrent server operation, and
coordinated shutdown to ensure clean application termination.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from genkit.core.logging import get_logger

from ._ports import is_port_available
from ._server import Server
from .signals import SignalHandler

logger = get_logger(__name__)


class ServerManagerProtocol(Protocol):
    """Protocol for a server manager."""

    def queue_server(self, server: Server) -> None:
        """Queue a server to be started."""
        ...


class ServerManager:
    """Manages multiple ASGI servers on a single event loop.

    The ServersManager provides a unified interface for starting, managing,
    and stopping multiple ASGI server instances within a single Python process.
    It handles signal management, coordinated shutdown, and resource cleanup.

    ### Key features

    1. Run multiple ASGI web applications concurrently
    2. Share a single event loop across all server instances
    3. Coordinate graceful shutdown across all servers
    4. Handle system signals (SIGINT, SIGTERM, etc.) centrally
    5. Support custom shutdown callbacks for resource cleanup
    6. Switch between different ASGI server implementations using the adapter
       pattern

    The manager uses the adapter pattern to abstract away the differences
    between various ASGI server implementations like
    [uvicorn](https://www.uvicorn.org/). This pattern allows
    ServersManager to operate independently of the specific server
    implementation being used, improving modularity and maintainability.

    Server configurations are defined using the `Server` class, which
    encapsulates the application instance, lifecycle hooks, and server settings.

    ### Adapter pattern benefits

    1. Separation of concerns - ServersManager focuses only on orchestration
    2. Extensibility - New server implementations can be added independently
    3. Maintainability - Implementation-specific code is isolated in adapters
    4. Testing - Server implementations can be mocked for testing

    ### Typical usage

    1. Create a ServersManager instance, optionally specifying a server adapter.
    2. Register server configurations using add_server().
    3. Optionally add shutdown callbacks using add_shutdown_callback().
    4. Call start_all() to concurrently start all servers.
    5. The servers will run until a shutdown signal is received or stop_all() is
       called.

    ### Key operations

    | Method                    | Description                                        |
    |---------------------------|----------------------------------------------------|
    | `add_server()`            | Register an ASGI application to be served          |
    | `add_shutdown_callback()` | Register function to be called during shutdown     |
    | `add_signal_handler()`    | Add callback for specific signal                   |
    | `remove_signal_handler()` | Remove callback for specific signal                |
    | `start_server()`          | Start a single server (normally called internally) |
    | `start_all()`             | Start all registered servers concurrently          |
    | `stop_all()`              | Stop all running servers                           |
    | `queue_server()`          | Queue a server to be started                     |
    """

    def __init__(
        self,
        handle_signals: bool = True,
    ) -> None:
        """Initialize the manager.

        Args:
            handle_signals: Whether to handle system signals automatically.  If
                True, the manager will set up handlers for SIGINT and SIGTERM.
        """
        self._servers: list[Server] = []
        self._server_tasks: list[asyncio.Task[None]] = []
        self._signal_handler: SignalHandler = SignalHandler()
        self._handle_signals: bool = handle_signals
        self._shutdown_callbacks: list[Callable[[], Any]] = []
        self._server_queue: asyncio.Queue[Server] = asyncio.Queue()
        self._is_running: bool = False

    async def _attempt_ports(self, server: Server) -> int:
        """Attempt to use a port from among a list of ports.

        This method will attempt to use a port from the list of ports provided
        in the server configuration. If the port is available, the server
        configuration port will be updated with the new port and the port will
        be returned.

        Args:
            server: The server to attempt to use.

        Returns:
            The port that was successfully used.

        Raises:
            RuntimeError: If no port is available.
        """
        host = server.config.host
        for port in server.config.ports:
            await server.lifecycle.on_port_check(server.config, host, port)
            if await is_port_available(port, host):
                server.config.port = port
                await server.lifecycle.on_port_available(server.config, host, port)
                return port
            else:
                await server.lifecycle.on_port_unavailable(server.config, host, port)

        raise RuntimeError(f'No port available on {host} among {server.config.ports}')

    def add_server(self, server: Server) -> None:
        """Add a server configuration to be started.

        Args:
            server: The server to add

        Returns:
            None
        """
        logger.info(
            'Registering server',
            name=server.config.name,
            ports=server.config.ports,
        )
        self._servers.append(server)
        # If we're already running, add to the queue
        if self._is_running:
            _ = asyncio.create_task(self._server_queue.put(server))

    async def queue_server(self, server: Server) -> None:
        """Queue a server to be started.

        This method allows adding servers dynamically after the manager has
        started running. The server will be started as soon as it's added to
        the queue.

        Args:
            server: The server to queue for starting

        Returns:
            None
        """
        await logger.ainfo(
            'Queueing server for startup',
            name=server.config.name,
            ports=server.config.ports,
        )
        # Add to our list of servers
        if server not in self._servers:
            self._servers.append(server)
        # Add to the queue to be started
        await self._server_queue.put(server)

    def add_shutdown_callback(self, callback: Callable[[], Any]) -> None:
        """Add a callback function to be called during shutdown.

        Args:
            callback: Function to call during shutdown process

        Returns:
            None
        """
        self._shutdown_callbacks.append(callback)

    def add_signal_handler(self, sig: int, callback: Callable[[], Any]) -> None:
        """Add a handler for a specific signal.

        Args:
            sig: Signal number (e.g., signal.SIGINT, signal.SIGTERM)
            callback: Function to call when the signal is received

        Returns:
            None
        """
        self._signal_handler.add_handler(sig, callback)

    def remove_signal_handler(self, sig: int, callback: Callable[[], Any]) -> None:
        """Remove a callback for a specific signal.

        Args:
            sig: Signal number (e.g., signal.SIGINT, signal.SIGTERM)
            callback: Function to remove

        Returns:
            None
        """
        self._signal_handler.remove_handler(sig, callback)

    async def start_server(self, server: Server) -> None:
        """Start a single server in the current event loop.

        Args:
            server: The server to start

        Returns:
            None
        """
        app = server.lifecycle.create(server.config)
        if app is None:
            raise ValueError('app cannot be None')

        port = await self._attempt_ports(server)

        # Record the start time for metrics.
        server.config.start_time = time.time()
        await server.lifecycle.on_start(server.config)

        # Start the server.
        if server.adapter is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError('server_adapter cannot be None')  # pyright: ignore[reportUnreachable]
        await server.adapter.serve(
            app=app,
            host=server.config.host,
            port=port,
            log_level=server.config.log_level,
        )

    async def start_all(self) -> None:
        """Start all configured servers concurrently."""
        # Setup signal handlers if requested.
        if self._handle_signals:
            self._signal_handler.setup_signal_handlers()

        # Mark as running.
        self._is_running = True

        # Create tasks for each server.
        self._server_tasks = [asyncio.create_task(self.start_server(server)) for server in self._servers]

        # Add tasks to monitor:
        # - shutdown event and server errors.
        # - server queue.
        _ = asyncio.create_task(self._monitor_shutdown())
        _ = asyncio.create_task(self._monitor_server_tasks())
        _ = asyncio.create_task(self._monitor_server_queue())

        try:
            # Wait for shutdown.
            _ = await self._signal_handler.shutdown_event.wait()
        except asyncio.CancelledError:
            await logger.adebug('ServersManager.start_all was cancelled')
            raise
        finally:
            # Mark as no longer running.
            self._is_running = False
            await logger.ainfo('Shutting down all servers')
            await self.stop_all()

    async def _monitor_server_queue(self) -> None:
        """Monitor the server queue for new servers."""
        while True:
            try:
                server = await self._server_queue.get()
                if server is None:  # pyright: ignore[reportUnnecessaryComparison]
                    break

                await logger.ainfo(
                    'Adding new server from queue',
                    name=server.config.name,
                    ports=server.config.ports,
                )

                # Start the server.
                task = asyncio.create_task(
                    self.start_server(server),
                    name=f'server-{server.config.name}',
                )
                self._server_tasks.append(task)
                self._server_queue.task_done()
            except Exception as e:
                logger.exception('Error processing server from queue', error=e)

    async def _monitor_server_tasks(self) -> None:
        """Monitor server tasks for completion and log any errors."""
        while self._server_tasks:
            done, pending = await asyncio.wait(
                self._server_tasks,
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                self._server_tasks.remove(task)
                try:
                    task_result = task.result()
                    await logger.awarning(
                        'Server task completed unexpectedly',
                        task_result=task_result,
                    )
                except Exception as e:
                    logger.exception('Server task failed with error', error=e)
                    # If a server task fails, we should trigger shutdown.
                    self._signal_handler.shutdown_event.set()
                    return

            # If all tasks are done, exit the monitoring loop.
            if not pending:
                break

        # If we get here, all server tasks have completed.
        logger.info('All server tasks have completed')
        self._signal_handler.shutdown_event.set()

    async def _monitor_shutdown(self) -> None:
        """Monitor for shutdown events."""
        _ = await self._signal_handler.shutdown_event.wait()
        await logger.ainfo('Shutdown event detected')

    async def stop_all(self) -> None:
        """Stop all running servers."""
        await logger.adebug('Stopping all servers')

        # Execute all shutdown callbacks.
        for callback in self._shutdown_callbacks:
            try:
                result = callback()
                # Handle coroutines returned from callbacks.
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.exception('Error in shutdown callback', error=e)

        # Stop all servers.
        for s in self._servers:
            if s.lifecycle:
                await logger.ainfo('Stopping server', name=s.config.name)
                await s.lifecycle.on_shutdown(s.config)

        # Cancel all server tasks.
        for task in self._server_tasks:
            if not task.done():
                _ = task.cancel()

        # Wait for all tasks to complete.
        if self._server_tasks:
            _ = await asyncio.gather(*self._server_tasks, return_exceptions=True)

        await logger.ainfo('All servers stopped')

    # TODO(#4355): I'm not sure the async context manager is useful, but it's here for
    # now.
    async def __aenter__(self) -> ServerManager:
        """Enter the async context, starting all servers.

        Returns:
            self: The instance for method chaining
        """
        await self.start_all()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,  # noqa: ANN401
    ) -> None:
        """Exit the async context, stopping all servers.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            None
        """
        await self.stop_all()

    async def run_all(
        self,
        servers: list[Server],
        stopping: Awaitable[None] | None = None,
        cleanup: Callable[[], Any] | None = None,
    ) -> None:
        """Runs multiple servers in a single process.

        Each server is defined by a Server, which includes a ServerConfig and a
        lifecycle that defines the server's lifecycle behavior.

        Args:
            servers: List of server definitions to run.
            stopping: Optional function to be called when all servers are
                started.
            cleanup: Optional cleanup function to be called during shutdown.
        """
        if cleanup is not None:
            self.add_shutdown_callback(cleanup)

        for server in servers:
            self.add_server(server)

        try:
            async with self:
                if stopping is not None:
                    await stopping
                # Block until shutdown is triggered.
                _ = await self._signal_handler.shutdown_event.wait()
        except asyncio.CancelledError:
            logger.debug('Servers task was cancelled')
            raise
        except Exception as e:
            logger.exception('Error running servers', error=e)
            raise
