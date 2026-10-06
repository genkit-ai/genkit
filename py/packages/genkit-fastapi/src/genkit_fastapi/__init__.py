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

"""Serve Genkit flows as FastAPI routes.

With ``GENKIT_ENV=dev``, the Dev UI reflection server starts on a background
thread. No lifespan wiring needed.

```python
from fastapi import FastAPI
from genkit import Genkit
from genkit_fastapi import serve_flow
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))
app = FastAPI()


@ai.flow()
async def suggest_dish(cuisine: str) -> str:
    response = await ai.generate(prompt=f'Suggest one {cuisine} dish.')
    return response.text


# POST /api/suggest_dish
app.include_router(serve_flow(suggest_dish), prefix='/api')
```

For a custom route, stack ``@genkit_fastapi_handler(ai)`` over ``@ai.flow()``.

Agents are experimental, so their routes come from ``genkit_fastapi.exp``::

    from genkit_fastapi.exp import serve_agent

```bash
genkit start -- uvicorn main:app --reload  # with Dev UI
uvicorn main:app                           # production
```
"""

from .handler import genkit_fastapi_handler, handle_genkit_request, serve_flow


def package_name() -> str:
    """Get the package name for the FastAPI plugin."""
    return 'genkit_fastapi'


__all__ = [
    'genkit_fastapi_handler',
    'handle_genkit_request',
    'package_name',
    'serve_flow',
]
