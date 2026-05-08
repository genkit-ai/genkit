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

This module provides types and helpers to define custom middleware.
Chain ordering: middleware is applied first-in, outermost.

Define a middleware class with ``@middleware`` and pass instances inline via ``use=``:

    from genkit import Genkit
    from genkit.middleware import BaseMiddleware, middleware

    @middleware(name='logging')
    class LoggingMiddleware(BaseMiddleware):
        async def wrap_generate(self, params, next_fn):
            print('before')
            result = await next_fn(params)
            print('after')
            return result

    ai = Genkit()

    response = await ai.generate(
        model='your-model-here',
        prompt='Hello',
        use=[LoggingMiddleware()],
    )

To make middleware available to the **Dev UI** and referenceable by name, register it
on the app via ``ai.define_middleware`` or declare it in a plugin:

    ai.define_middleware(LoggingMiddleware)
"""

from genkit._core._middleware import (
    BaseMiddleware,
    GenerateHookParams,
    MiddlewareDesc,
    ModelHookParams,
    MultipartToolResponse,
    ToolHookParams,
    middleware,
)
from genkit._core._plugin import middleware_plugin

__all__ = [
    'BaseMiddleware',
    'GenerateHookParams',
    'MiddlewareDesc',
    'ModelHookParams',
    'MultipartToolResponse',
    'ToolHookParams',
    'middleware',
    'middleware_plugin',
]
