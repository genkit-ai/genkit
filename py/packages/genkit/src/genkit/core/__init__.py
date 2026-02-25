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


"""Core foundations for the Genkit framework.

This package provides the fundamental building blocks and abstractions used
throughout the Genkit framework. It includes the action system, plugin
architecture, registry, tracing, and schema types.

Architecture Overview:
    The core package forms the foundation layer upon which all Genkit
    functionality is built. It provides primitives that are used by both
    the framework internals and user-facing APIs.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           User Application                              │
    │                    (flows, prompts, tools, chat)                        │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           genkit.ai Layer                               │
    │                      (Genkit class, GenkitRegistry)                     │
    └─────────────────────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          genkit.core Layer                              │
    │    ┌────────────┬────────────┬────────────┬────────────┬────────────┐   │
    │    │  Action    │   Plugin   │  Registry  │   Trace    │   Schema   │   │
    │    └────────────┴────────────┴────────────┴────────────┴────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

Key Modules:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Module                │ Description                                     │
    ├───────────────────────┼─────────────────────────────────────────────────┤
    │ action                │ Action class and execution context              │
    │ plugin                │ Plugin base class for extending Genkit          │
    │ registry              │ Central repository for actions and resources    │
    │ tracing               │ OpenTelemetry-based tracing infrastructure      │
    │ typing                │ Core type definitions (Part, Message, etc.)     │
    │ schema                │ JSON Schema generation and validation           │
    │ error                 │ GenkitError and error handling utilities        │
    │ logging               │ Structured logging via structlog                │
    └───────────────────────┴─────────────────────────────────────────────────┘

Key Concepts:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Concept             │ Description                                       │
    ├─────────────────────┼───────────────────────────────────────────────────┤
    │ Action              │ Strongly-typed, traceable, callable function.     │
    │                     │ All models, tools, flows, embedders are actions.  │
    │ ActionKind          │ Type of action: MODEL, TOOL, FLOW, EMBEDDER, etc. │
    │ ActionRunContext    │ Execution context with streaming and user context │
    │ Plugin              │ Extension mechanism for adding capabilities       │
    │ Registry            │ Central storage for actions, schemas, and plugins │
    │ Trace               │ OpenTelemetry span for observability              │
    └─────────────────────┴───────────────────────────────────────────────────┘

Usage:
    Most users interact with the core layer indirectly through the ``Genkit``
    class. Direct usage is typically for plugin authors or advanced use cases:

    ```python
    from genkit.core.action import Action
    from genkit.core.plugin import Plugin
    from genkit.core.registry import Registry

    # Create a registry
    registry = Registry()

    # Register an action directly
    action = registry.register_action(
        kind=ActionKind.TOOL,
        name='my_tool',
        fn=my_tool_function,
    )
    ```

See Also:
    - genkit.ai: High-level user-facing API
    - genkit.blocks: Building blocks for models, prompts, embedders
    - genkit.types: Re-exported type definitions from core.typing
"""

from .constants import GENKIT_CLIENT_HEADER, GENKIT_VERSION
from .http_client import clear_client_cache, close_cached_clients, get_cached_client
from .logging import Logger, get_logger


def package_name() -> str:
    """Get the fully qualified package name.

    Returns:
        The string 'genkit.core', which is the fully qualified package name.
    """
    return 'genkit.core'


__all__ = [
    'GENKIT_CLIENT_HEADER',
    'GENKIT_VERSION',
    'Logger',
    'clear_client_cache',
    'close_cached_clients',
    'get_cached_client',
    'get_logger',
    'package_name',
]
