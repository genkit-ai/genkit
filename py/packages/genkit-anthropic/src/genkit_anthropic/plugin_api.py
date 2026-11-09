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

"""Hooks for plugins that serve Claude through their own client.

App code uses ``genkit_anthropic.Anthropic``. This module is for other plugins
(for example ``genkit_vertexai.ModelGarden``, which passes an
``AsyncAnthropicVertex``) that want genkit-anthropic's request and response
conversion without importing its private modules.
"""

from collections.abc import Awaitable, Callable

from anthropic import AsyncAnthropic

from genkit import ModelResponse
from genkit.model import ModelContext, ModelInfo, ModelRequest
from genkit_anthropic._model_info import get_model_info
from genkit_anthropic._models import AnthropicModel

__all__ = ['anthropic_model', 'anthropic_model_info']


def anthropic_model(
    model: str,
    client: AsyncAnthropic,
) -> Callable[[ModelRequest, ModelContext], Awaitable[ModelResponse]]:
    """Returns a Genkit model function that calls Claude ``model`` through ``client``.

    Args:
        model: Claude model ID, without a provider prefix (``claude-sonnet-4-5``).
        client: Anthropic client. Subclasses such as ``AsyncAnthropicVertex``
            work; cast them to ``AsyncAnthropic``.

    Returns:
        An async ``(request, ctx) -> ModelResponse`` function, usable as the
        ``fn`` of a model action.
    """
    return AnthropicModel(model_name=model, client=client).generate


def anthropic_model_info(model: str) -> ModelInfo:
    """Returns the catalog entry for ``model``, or the defaults for an unknown ID.

    ``supports.constrained`` here matches what ``anthropic_model`` sends, so
    callers that build their own ``ModelInfo`` agree with it on native JSON mode.
    """
    return get_model_info(model)
