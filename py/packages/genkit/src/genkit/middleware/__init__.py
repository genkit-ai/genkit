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

"""Middleware for Genkit model calls.

This module provides types and helpers to define middleware and register it on
the app. Chain ordering: middleware is applied first-in, outermost.

Example usage:
    from genkit import Genkit, MiddlewareRef
    from genkit.middleware import BaseMiddleware, new_middleware, middleware_plugin

    class MyMiddleware(BaseMiddleware):
        name = "my_middleware"
        ...

    ai = Genkit(plugins=[middleware_plugin([new_middleware(MyMw)])])

    response = await ai.generate(
        model="gemini-pro",
        prompt="Hello",
        use=[MiddlewareRef(name="my_mw")],
    )
"""

from genkit._core._middleware._base import (
    BaseMiddleware,
    MiddlewareDesc,
    new_middleware,
)
from genkit._core._model import GenerateHookParams, ModelHookParams, ToolHookParams
from genkit._core._plugin import middleware_plugin

__all__ = [
    'BaseMiddleware',
    'GenerateHookParams',
    'MiddlewareDesc',
    'ModelHookParams',
    'ToolHookParams',
    'middleware_plugin',
    'new_middleware',
]
