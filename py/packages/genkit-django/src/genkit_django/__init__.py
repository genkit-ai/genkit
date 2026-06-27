# Copyright 2026 Google LLC
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


"""Django Plugin for Genkit.

This plugin provides Django integration for Genkit, enabling you to expose
Genkit flows as HTTP endpoints in a Django ASGI application.

Example:
    ```python
    # myapp/views.py
    from genkit import Genkit
    from genkit_django import genkit_django_handler

    ai = Genkit(plugins=[...])


    @genkit_django_handler(ai)
    @ai.flow()
    async def chat(prompt: str) -> str:
        response = await ai.generate(prompt=prompt)
        return response.text
    ```

    ```python
    # myproject/urls.py
    from django.urls import path
    from myapp.views import chat

    urlpatterns = [
        path('chat/', chat),
    ]
    ```

Requirements:
    - Django 4.1+ (async views require ASGI)
    - An ASGI server such as ``uvicorn`` or ``daphne``:

      ```bash
      uvicorn myproject.asgi:application
      ```

Wire protocol:
    - Body: ``{"data": <flow_input>}``. Missing ``data`` returns 400.
    - Streaming: ``Accept: text/event-stream`` or ``?stream=true`` returns
      ``text/event-stream`` with ``data: {"message": ...}`` chunks and a
      final ``data: {"result": ...}`` event.
    - Non-stream: ``{"result": <flow_output>}`` on success; 500 with
      ``HttpErrorWireFormat`` JSON on exception.

The returned view is automatically ``csrf_exempt`` because this is a JSON API.

See Also:
    - Django ASGI: https://docs.djangoproject.com/en/stable/howto/deployment/asgi/
    - Genkit documentation: https://genkit.dev/
"""

from .handler import genkit_django_handler


def package_name() -> str:
    """Get the package name for the Django plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit_django'


__all__ = ['package_name', genkit_django_handler.__name__]
