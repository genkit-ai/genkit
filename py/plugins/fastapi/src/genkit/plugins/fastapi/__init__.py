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

"""FastAPI Plugin for Genkit.

This plugin provides FastAPI integration for Genkit, enabling you to expose
Genkit flows as HTTP endpoints in a FastAPI application.

The Dev UI reflection server starts automatically in a background thread when
``GENKIT_ENV=dev`` is set — no lifespan wiring needed.

Both ``serve_flow`` and ``serve_agent`` return an ``APIRouter`` you mount with
``app.include_router`` — so FastAPI's own ``prefix`` / ``dependencies`` / ``tags``
do the framework-level wiring, and a flow and an agent read the same way.

Serve a flow (one route):
    ```python
    from fastapi import FastAPI
    from genkit import Genkit
    from genkit.plugins.fastapi import serve_flow
    from genkit.plugins.google_genai import GoogleAI

    ai = Genkit(plugins=[GoogleAI()])
    app = FastAPI()


    @ai.flow()
    async def chat_flow(prompt: str) -> str:
        response = await ai.generate(prompt=prompt)
        return response.text


    app.include_router(serve_flow(chat_flow), prefix='/api')  # POST /api/chat_flow
    ```

Serve an agent (run-turn / getSnapshot / abort):
    ```python
    from fastapi import FastAPI
    from genkit.plugins.fastapi import serve_agent

    app = FastAPI()
    app.include_router(serve_agent(my_agent), prefix='/api')  # POST /api/<agent name>
    ```

Running:
    ```bash
    # With Genkit Dev UI
    genkit start -- uvicorn main:app --reload

    # Production (no Dev UI)
    uvicorn main:app
    ```
"""

from .agent_handler import serve_agent
from .handler import serve_flow


def package_name() -> str:
    """Get the package name for the FastAPI plugin."""
    return 'genkit.plugins.fastapi'


__all__ = [
    'package_name',
    'serve_agent',
    'serve_flow',
]
