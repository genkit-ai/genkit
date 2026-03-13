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

"""Manages Genkit runtime lifecycle: creation/cleanup of CLI metadata files."""

from __future__ import annotations

import atexit
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import TracebackType

from genkit.core.constants import DEFAULT_GENKIT_VERSION
from genkit.core.logging import get_logger

from ._server import ServerSpec

logger = get_logger(__name__)

DEFAULT_RUNTIME_DIR_NAME = '.genkit/runtimes'


def _remove_file(file_path: Path | None) -> bool:
    """Synchronously attempts to delete the file.

    Returns:
        True if cleanup was successful or file didn't exist, False on error.
    """
    # NOTE: Neither print nor logger appears to work during atexit, so print is intentional here.
    if not file_path:
        return True
    try:
        if file_path.exists():
            print(f'Removing file: {file_path}')  # noqa: T201 - atexit handler, logger unavailable
            file_path.unlink()
            # Consider success if unlink didn't raise error
            return True
        else:
            # Consider success if file already gone
            return True
    except Exception as e:
        print(f'Error deleting {file_path}: {e}')  # noqa: T201 - atexit handler, logger unavailable
        return False


def _register_atexit_cleanup_handler(path_to_remove: Path | None) -> None:
    """Defines and registers the synchronous atexit cleanup handler for a path.

    Args:
        path_to_remove: The path to the file that needs cleanup.
    """
    if not path_to_remove:
        logger.warning('Cannot register atexit cleanup: runtime file path not set.')
        return

    def sync_cleanup() -> None:
        # TODO(#4335): Neither print nor logger appears to work during atexit.
        _ = _remove_file(path_to_remove)

    logger.debug(f'Registering synchronous atexit cleanup for {path_to_remove}')
    _ = atexit.register(sync_cleanup)


def _create_and_write_runtime_file(runtime_dir: Path, spec: ServerSpec) -> Path:
    """Calculates metadata, creates filename, and writes the runtime file.

    Args:
        runtime_dir: The directory to write the file into.
        spec: The ServerSpec containing reflection server details.

    Returns:
        The Path object of the created file.
    """
    current_datetime = datetime.now()
    timestamp_ms = int(current_datetime.timestamp() * 1000)
    pid = os.getpid()

    # Build a unique runtime ID from the process ID and port
    port = spec.port if spec.port else ''
    runtime_id = f'{pid}-{port}' if port else f'{pid}'

    # Include timestamp in filename to avoid collisions across restarts
    runtime_file_name = f'{runtime_id}-{timestamp_ms}.json'
    runtime_file_path = runtime_dir / runtime_file_name

    metadata = json.dumps({
        'reflectionApiSpecVersion': 1,
        'id': runtime_id,
        'pid': pid,
        'genkitVersion': 'py/' + DEFAULT_GENKIT_VERSION,
        'reflectionServerUrl': spec.url,
        'timestamp': current_datetime.isoformat(),
    })

    logger.debug(f'Writing runtime file: {runtime_file_path}')
    with Path(runtime_file_path).open('w', encoding='utf-8') as f:
        _ = f.write(metadata)

    logger.info(f'Initialized runtime file: {runtime_file_path}')
    _ = sys.stdout.flush()
    _ = sys.stderr.flush()
    return runtime_file_path


class RuntimeManager:
    """Asynchronous and synchronous context manager for Genkit runtime.

    This class provides a context manager for Genkit runtime. It ensures that
    the runtime directory and file are created and cleaned up when the context
    is exited.

    The runtime file is a JSON file that contains metadata about the runtime.
    It is used to track the runtime and the reflection server. Example:

    ```json
    {
        "reflectionApiSpecVersion": 1,
        "id": "1234567890",
        "pid": 1234567890,
        "reflectionServerUrl": "http://localhost:3100",
        "timestamp": "2021-01-01T00:00:00Z"
    }
    ```

    The context manager registers a cleanup handler that is called at process
    exit. The cleanup handler removes the runtime file.

    The exit handler for the context manager is a no-op. It is used to ensure
    that the context manager exits cleanly and allows exceptions to propagate.
    """

    def __init__(
        self,
        spec: ServerSpec,
        runtime_dir: str | Path | None = None,
        lazy_write: bool = False,
    ) -> None:
        """Initialize the RuntimeManager.

        Args:
            spec: The server specification for the reflection server.
            runtime_dir: The directory to store the runtime file in.
                         Defaults to .genkit/runtimes in the current directory.
            lazy_write: If True, the runtime file will not be written immediately
                        on context entry. It must be written manually by calling
                        write_runtime_file().
        """
        self.spec: ServerSpec = spec
        if runtime_dir is None:
            self._runtime_dir: Path = Path(Path.cwd()) / DEFAULT_RUNTIME_DIR_NAME
        else:
            self._runtime_dir = Path(runtime_dir)

        self.lazy_write: bool = lazy_write
        self._runtime_file_path: Path | None = None

    async def __aenter__(self) -> RuntimeManager:
        """Create the runtime directory and file."""
        try:
            await logger.adebug(f'Ensuring runtime directory exists: {self._runtime_dir}')
            _ = self._runtime_dir.mkdir(parents=True, exist_ok=True)
            if not self.lazy_write:
                _ = self.write_runtime_file()

        except Exception as e:
            logger.error(f'Failed to initialize runtime file: {e}', exc_info=True)
            _ = sys.stdout.flush()
            _ = sys.stderr.flush()
            raise

        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> bool:
        """Async context manager exit handler.

        Args:
            exc_type: The type of the exception that occurred.
            exc_val: The value of the exception that occurred.
            exc_tb: The traceback of the exception that occurred.

        Returns:
            False to indicate exceptions should propagate.
        """
        self.cleanup()
        await logger.adebug('RuntimeManager async context exited.')
        return False

    def __enter__(self) -> RuntimeManager:
        """Synchronous entry point: Create the runtime directory and file."""
        try:
            logger.debug(f'[sync] Ensuring runtime directory exists: {self._runtime_dir}')
            _ = self._runtime_dir.mkdir(parents=True, exist_ok=True)
            if not self.lazy_write:
                _ = self.write_runtime_file()

        except Exception as e:
            logger.error(f'[sync] Failed to initialize runtime file: {e}', exc_info=True)
            _ = sys.stdout.flush()
            _ = sys.stderr.flush()
            raise

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Synchronous exit handler.

        Cleanup is handled by atexit. This method primarily ensures the context
        exits cleanly and allows exceptions to propagate.

        Returns:
            False to indicate exceptions should propagate.
        """
        self.cleanup()
        logger.debug('RuntimeManager sync context exited.')
        return False

    def write_runtime_file(self) -> Path:
        """Calculates metadata, creates filename, and writes the runtime file.

        Returns:
            The Path object of the created file.
        """
        if self._runtime_file_path:
            return self._runtime_file_path

        self._runtime_file_path = _create_and_write_runtime_file(self._runtime_dir, self.spec)
        _register_atexit_cleanup_handler(self._runtime_file_path)
        return self._runtime_file_path

    def cleanup(self) -> None:
        """Explicitly cleanup the runtime file."""
        if self._runtime_file_path:
            logger.debug(f'Cleaning up runtime file: {self._runtime_file_path}')
            _ = _remove_file(self._runtime_file_path)
