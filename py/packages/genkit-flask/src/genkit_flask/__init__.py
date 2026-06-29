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

Example:
    ```python
    from flask import Flask
    from genkit import Genkit
    from genkit_flask import genkit_flask_handler
    from genkit_googleai import GoogleAI

    # 1. Initialize Flask app and Genkit with GoogleAI
    app = Flask(__name__)
    ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')


    # 2. Define an asynchronous Genkit flow
    @ai.flow()
    async def greet_user(name: str) -> str:
        res = await ai.generate(prompt=f'Say hello to {name} in one sentence.')
        return res.text


    # 3. Expose flow as an HTTP endpoint
    @app.route('/api/greet', methods=['POST'])
    def greet_endpoint():
        return genkit_flask_handler(ai, greet_user)


    # POST /api/greet {"data": "Alice"}
    # => {"result": "Hello Alice! Welcome to our AI community."}
    ```

Requirements:
    - Requires Flask 3.0+.
    - Async flows are run via an asyncio event loop within the Flask request handler.

See Also:
    - Flask documentation: https://flask.palletsprojects.com/
"""

from .handler import genkit_flask_handler


def package_name() -> str:
    """Get the package name for the Flask plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit_flask'


# String literals so pyright can see what's public — `Cls.__name__` looks
# right at runtime but type checkers can't trace it back to an exported symbol.
__all__ = ['genkit_flask_handler', 'package_name']
