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

This module provides types and helpers to define and register custom middleware.
Chain ordering: middleware is applied first-in, outermost.

Define a middleware class and pass it inline directly in ``use=``:

    from genkit import Genkit
    from genkit.middleware import BaseMiddleware

    class LoggingMiddleware(BaseMiddleware):
        name = 'logging'

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

To reference middleware by name (e.g. from the Dev UI or a config), register it
first via ``ai.define_middleware``:

    from genkit import MiddlewareRef

    # Option A — imperative, after Genkit() is built.
    # Once registered, reference by name with MiddlewareRef and pass config there:
    ai.define_middleware(LoggingMiddleware)
    await ai.generate(model='your-model-here', prompt='Hello',
                      use=[MiddlewareRef(name='logging', config={'prefix': '[span]'})])

    # Option B — pass config directly via the inline instance:
    await ai.generate(model='your-model-here', prompt='Hello',
                      use=[LoggingMiddleware(prefix='[span]')])
"""

from genkit._core._middleware._base import (
    BaseMiddleware,
    MiddlewareDesc,
    new_middleware,
)
from genkit._core._model import GenerateHookParams, ModelHookParams, MultipartToolResponse, ToolHookParams
from genkit._core._plugin import middleware_plugin

__all__ = [
    'BaseMiddleware',
    'GenerateHookParams',
    'MiddlewareDesc',
    'ModelHookParams',
    'MultipartToolResponse',
    'ToolHookParams',
    'middleware_plugin',
    'new_middleware',
]
