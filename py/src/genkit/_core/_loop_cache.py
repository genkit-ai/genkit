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

"""Per-event-loop resource caching for async HTTP clients."""

import asyncio
import threading
import weakref
from collections.abc import Callable
from typing import TypeVar

T = TypeVar('T')


def _loop_local_client(factory: Callable[[], T]) -> Callable[[], T]:
    """Return a getter that caches one resource instance per event loop."""
    by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, T] = weakref.WeakKeyDictionary()
    lock = threading.Lock()

    def _get() -> T:
        loop = asyncio.get_running_loop()
        with lock:
            existing = by_loop.get(loop)
            if existing is not None:
                return existing
            created = factory()
            by_loop[loop] = created
            return created

    return _get
