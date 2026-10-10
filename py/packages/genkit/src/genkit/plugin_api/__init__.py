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

# StrEnum that works the same on 3.10 through 3.14
from genkit._core._compat import StrEnum
from genkit._core._constants import GENKIT_CLIENT_HEADER
from genkit._core._environment import is_dev_environment
from genkit._core._error import StatusName, provider_error
from genkit._core._loop_cache import loop_local_client
from genkit._core._plugin import MiddlewarePlugin, Plugin
from genkit._core._schema import to_json_schema
from genkit._core._typing import ActionMetadata

__all__ = [
    # Base class and framework primitives
    'MiddlewarePlugin',
    'Plugin',
    'Action',
    'ActionMetadata',
    'ActionKind',
    'StatusName',
    'StrEnum',
    # Provider failures
    'provider_error',
    # HTTP / version stamping
    'GENKIT_CLIENT_HEADER',
    # Loop-local caching
    'loop_local_client',
    # Environment detection
    'is_dev_environment',
    # Schema utilities
    'to_json_schema',
]
