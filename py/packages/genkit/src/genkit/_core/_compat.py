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

"""Compatibility layer for asyncio."""

import asyncio
import sys
from typing import TypeVar

T = TypeVar('T')

# StrEnum - use strenum package for cross-version compatibility
# Note: StrEnum was added to stdlib in Python 3.11, but we use strenum for 3.10 compat
# override decorator - use typing_extensions for consistency across Python versions
# Note: override was added to typing in Python 3.12, but typing_extensions has it for all versions
from typing import overload as overload  # noqa: E402

if sys.version_info >= (3, 11):
    from enum import StrEnum as StrEnum  # noqa: E402
else:
    from strenum import StrEnum as StrEnum  # noqa: E402
from typing_extensions import override as override  # noqa: E402


async def wait_for_310(fut: asyncio.Future[T], timeout: float | None = None) -> T:
    """Python 3.10 compat: raises TimeoutError instead of asyncio.TimeoutError."""
    try:
        return await asyncio.wait_for(fut, timeout)
    except asyncio.TimeoutError as e:
        raise TimeoutError() from e


if sys.version_info < (3, 11):
    wait_for = wait_for_310  # pyright: ignore[reportUnreachable]
else:
    wait_for = asyncio.wait_for
