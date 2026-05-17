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


"""Flask Plugin for Genkit.

This plugin provides Flask integration for Genkit, enabling you to expose
Genkit flows as HTTP endpoints in a Flask application.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Flask               │ A simple Python web framework. Like a waiter      │
    │                     │ that takes HTTP requests and serves responses.    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ HTTP Endpoint       │ A URL that accepts requests. Like a phone number  │
    │                     │ your app answers when called.                     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Flow                │ A Genkit function that does AI work. This plugin  │
    │                     │ lets you call flows via HTTP requests.            │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Route               │ Maps a URL to a function. /api/chat → chat_flow   │
    │                     │                                                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Request Handler     │ Code that processes incoming requests.            │
    │                     │ genkit_flask_handler does this for you.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ POST                │ HTTP method for sending data. Like mailing a      │
    │                     │ letter with your prompt inside.                   │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                HOW FLASK SERVES YOUR GENKIT FLOWS                       │
    │                                                                         │
    │    Client (Browser, curl, etc.)                                         │
    │    POST /api/chat {"prompt": "Hello!"}                                  │
    │         │                                                               │
    │         │  (1) HTTP request arrives                                     │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Flask App      │   Routes request to the right handler            │
    │    │  @app.route()   │                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Handler invoked                                      │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │ genkit_flask_   │   Parses JSON body, validates input              │
    │    │ handler()       │                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) Calls your Genkit flow                               │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Your Flow      │   Does AI magic (generate, tools, etc.)          │
    │    │  async def ...  │                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (4) Response serialized to JSON                          │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Client         │   {"result": "Hello! How can I help?"}           │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         Flask Plugin                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Plugin Entry Point (__init__.py)                                       │
    │  └── genkit_flask_handler() - Create Flask route handler                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  handler.py - Request Handler                                           │
    │  ├── genkit_flask_handler() - Factory for Flask handlers                │
    │  ├── Request parsing and validation                                     │
    │  └── Response serialization                                             │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Request Flow                                     │
    │                                                                         │
    │  HTTP Request ──► Flask Route ──► genkit_flask_handler ──► Genkit Flow  │
    │                                                                         │
    │  HTTP Response ◄── Flask Route ◄── Handler ◄── Flow Result              │
    └─────────────────────────────────────────────────────────────────────────┘

Example:
    ```python
    from flask import Flask
    from genkit import Genkit
    from genkit.plugins.flask import genkit_flask_handler

    app = Flask(__name__)
    ai = Genkit(...)


    @ai.flow()
    async def my_flow(prompt: str) -> str:
        response = await ai.generate(prompt=prompt)
        return response.text


    # Expose flow as HTTP endpoint
    @app.route('/api/flow', methods=['POST'])
    def handle_flow():
        return genkit_flask_handler(ai, my_flow)
    ```

Caveats:
    - Requires Flask to be installed
    - Async flows are run synchronously in Flask (use async frameworks for better performance)
    - For production, consider using the async-native Genkit server

See Also:
    - Flask documentation: https://flask.palletsprojects.com/
    - Genkit documentation: https://genkit.dev/
"""

from .handler import genkit_flask_handler


def package_name() -> str:
    """Get the package name for the Flask plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit.plugins.flask'


__all__ = ['package_name', genkit_flask_handler.__name__]
