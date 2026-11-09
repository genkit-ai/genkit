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

"""Hooks for plugins that serve OpenAI-compatible endpoints.

App code uses ``genkit_openai.OpenAI``. This module is for other plugins
(for example ``genkit_vertexai.ModelGarden``) that bring their own
``AsyncOpenAI`` client and want genkit-openai's request and response
conversion without importing its private modules.
"""

from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from genkit import ModelResponse
from genkit.model import ModelContext, ModelRequest, ModelSupports
from genkit_openai._models._model import OpenAIModel

__all__ = ['openai_compatible_model']


def openai_compatible_model(
    model: str,
    client: AsyncOpenAI,
    *,
    supports: ModelSupports | None = None,
) -> Callable[[ModelRequest, ModelContext], Awaitable[ModelResponse]]:
    """Returns a Genkit model function that calls ``model`` through ``client``.

    Args:
        model: Model ID sent as ``model`` on each chat completion request.
        client: Client for the endpoint. Its base URL and auth decide where the
            request goes.
        supports: Capabilities of ``model``. ``'json'`` in ``supports.output``
            enables schema-less JSON mode.

    Returns:
        An async ``(request, ctx) -> ModelResponse`` function, usable as the
        ``fn`` of a model action.
    """
    return OpenAIModel(model, client, supports=supports).generate
