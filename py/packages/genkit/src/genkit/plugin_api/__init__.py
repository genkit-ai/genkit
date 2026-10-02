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

"""Framework primitives for plugin authors."""

# Base class and framework primitives
from genkit._core._action import Action, ActionKind
from genkit._core._compat import StrEnum
from genkit._core._constants import GENKIT_CLIENT_HEADER
from genkit._core._environment import is_dev_environment
from genkit._core._error import (
    ErrorResponseMetadata,
    StatusName,
    from_http_code,
    get_callable_json,
    get_http_status,
    parse_retry_after_ms,
    wrap_http_error,
)
from genkit._core._http_client import get_cached_client
from genkit._core._loop_cache import _loop_local_client as loop_local_client
from genkit._core._middleware import new_middleware
from genkit._core._plugin import MiddlewarePlugin, Plugin
from genkit._core._protocols import GenkitLike
from genkit._core._schema import to_json_schema
from genkit._core._typing import ActionMetadata


async def resolve_action(ai: GenkitLike, kind: ActionKind, name: str) -> Action | None:
    """Look up an action on this app, including per-call registrations on ``ctx.ai``."""
    return await ai._registry.resolve_action(kind, name)


__all__ = [
    # Base class and framework primitives
    'MiddlewarePlugin',
    'Plugin',
    'new_middleware',
    'Action',
    'ActionMetadata',
    'ActionKind',
    'ErrorResponseMetadata',
    'resolve_action',
    'StatusName',
    'from_http_code',
    'parse_retry_after_ms',
    'wrap_http_error',
    # HTTP / version stamping
    'GENKIT_CLIENT_HEADER',
    # Loop-local caching
    'loop_local_client',
    # Environment detection
    'is_dev_environment',
    # Schema utilities
    'to_json_schema',
    # String enums that work the same on every supported Python
    'StrEnum',
    # HTTP client
    'get_cached_client',
    # Error serialization
    'get_callable_json',
    'get_http_status',
]
