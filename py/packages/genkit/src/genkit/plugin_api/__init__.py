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
from genkit._core._action import Action, ActionKind, ActionMetadata, ActionRunContext
from genkit._core._constants import GENKIT_CLIENT_HEADER, GENKIT_VERSION
from genkit._core._context import ContextProvider, RequestData
from genkit._core._environment import is_dev_environment
from genkit._core._error import GenkitError, StatusCodes, StatusName, get_callable_json
from genkit._core._http_client import get_cached_client
from genkit._core._loop_cache import _loop_local_client as loop_local_client
from genkit._core._plugin import Plugin
from genkit._core._schema import to_json_schema
from genkit._core._trace._adjusting_exporter import AdjustingTraceExporter, RedactedSpan
from genkit._core._trace._path import to_display_path
from genkit._core._tracing import add_custom_exporter, tracer

# Embedder domain re-exports
from genkit.embedder import (
    EmbedderRef,
    embedder_action_metadata,
    embedder_ref,
)

# Evaluator domain re-exports
from genkit.evaluator import (
    EvaluatorRef,
    evaluator_action_metadata,
    evaluator_ref,
)

# Model domain re-exports
from genkit.model import (
    ModelRef,
    model_action_metadata,
    model_ref,
)

__all__ = [
    # Base class and framework primitives
    'Plugin',
    'Action',
    'ActionMetadata',
    'ActionKind',
    'ActionRunContext',
    'StatusCodes',
    'StatusName',
    'GenkitError',
    # HTTP / version stamping
    'GENKIT_CLIENT_HEADER',
    'GENKIT_VERSION',
    # Loop-local caching
    'loop_local_client',
    # Tracing
    'tracer',
    'add_custom_exporter',
    'AdjustingTraceExporter',
    'RedactedSpan',
    'to_display_path',
    # Schema utilities
    'to_json_schema',
    # HTTP client
    'get_cached_client',
    # Error serialization
    'get_callable_json',
    # Environment detection
    'is_dev_environment',
    # Model domain
    'model_action_metadata',
    'model_ref',
    'ModelRef',
    # Embedder domain
    'embedder_action_metadata',
    'embedder_ref',
    'EmbedderRef',
    # Evaluator domain
    'evaluator_action_metadata',
    'evaluator_ref',
    'EvaluatorRef',
    'ContextProvider',
    'RequestData',
]
