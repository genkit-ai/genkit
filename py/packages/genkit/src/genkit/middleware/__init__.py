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

Define a subclass of ``BaseMiddleware`` and register it on your app
with ``@ai.middleware``:

    from genkit import Genkit
    from genkit.middleware import BaseMiddleware

    ai = Genkit()

    @ai.middleware(name='logging')
    class LoggingMiddleware(BaseMiddleware):
        async def wrap_generate(self, params, next_fn, ctx: GenerateMiddlewareContext):
            print('before')
            result = await next_fn(params)
            print('after')
            return result

    response = await ai.generate(
        model='your-model-here',
        prompt='Hello',
        use=[LoggingMiddleware()],
    )

Once registered, the middleware is visible in the Dev UI.  You can play
with the Model Runner and mix-and-match your choice of middleware to see
its impact on generating the next response.

Order of middleware in ``use=[...]`` determines the order in which they are
called. The first middleware in the list is called first, and the last
middleware in the list is called last.

For example, given this call:

```python
use = [A(), B()]
```

The execution sequence is:

```
A_before()
    B_before()
        model_call()
    B_after()
A_after()
```
"""

from genkit._core._middleware import (
    BaseMiddleware,
    GenerateHookParams,
    GenerateMiddleware,
    GenerateMiddlewareContext,
    ModelHookParams,
    ToolHookParams,
)
from genkit._core._typing import MultipartToolResponse

__all__ = [
    'BaseMiddleware',
    'GenerateHookParams',
    'GenerateMiddleware',
    'GenerateMiddlewareContext',
    'ModelHookParams',
    'MultipartToolResponse',
    'ToolHookParams',
]
