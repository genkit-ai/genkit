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

Example:
    ```python
    from fastapi import FastAPI
    from genkit import Genkit
    from genkit.plugins.fastapi import genkit_fastapi_handler, genkit_lifespan
    from genkit.plugins.google_genai import GoogleAI

    ai = Genkit(plugins=[GoogleAI()])
    app = FastAPI(lifespan=genkit_lifespan(ai))


    @ai.flow()
    async def chat_flow(prompt: str) -> str:
        response = await ai.generate(prompt=prompt)
        return response.text


    @app.post('/chat')
    @genkit_fastapi_handler(ai)
    async def chat():
        return chat_flow
    ```

Running:
    ```bash
    # With Genkit Dev UI
    genkit start -- uvicorn main:app --reload

    # Production (no Dev UI)
    uvicorn main:app
    ```
"""

from .handler import genkit_fastapi_handler
from .lifespan import genkit_lifespan


def package_name() -> str:
    """Get the package name for the FastAPI plugin."""
    return 'genkit.plugins.fastapi'


__all__ = ['package_name', 'genkit_fastapi_handler', 'genkit_lifespan']
