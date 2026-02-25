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

"""DeepSeek model integration for Genkit.

Wraps the compat-oai OpenAIModel for use with DeepSeek's
OpenAI-compatible API. Adds parameter validation warnings for
reasoning models (R1, deepseek-reasoner) which silently ignore
temperature, top_p, and tools parameters.
"""

from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI

from genkit.core.action._action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.compat_oai.models.model import OpenAIModel
from genkit.plugins.deepseek.client import DeepSeekClient
from genkit.plugins.deepseek.model_info import (
    SUPPORTED_DEEPSEEK_MODELS,
    get_default_model_info,
    is_reasoning_model,
)
from genkit.types import GenerateRequest, GenerateResponse

DEEPSEEK_PLUGIN_NAME = 'deepseek'

logger = get_logger(__name__)

# Parameters that reasoning models silently ignore.
_REASONING_IGNORED_PARAMS: frozenset[str] = frozenset({
    'temperature',
    'top_p',
})


def deepseek_name(name: str) -> str:
    """Create a DeepSeek action name.

    Args:
        name: Base name for the action.

    Returns:
        The fully qualified DeepSeek action name.
    """
    return f'{DEEPSEEK_PLUGIN_NAME}/{name}'


def _get_config_value(config: dict[str, Any] | object, param: str) -> Any:  # noqa: ANN401
    """Get a config value by name from either a dict or an object.

    Args:
        config: Dict or Pydantic model config.
        param: Parameter name to look up.

    Returns:
        The parameter value, or None if not found.
    """
    if isinstance(config, dict):
        return config.get(param)  # type: ignore[arg-type]
    return getattr(config, param, None)


def _warn_reasoning_params(model_name: str, config: dict[str, Any] | object | None) -> None:
    """Emit warnings for parameters that reasoning models silently ignore.

    DeepSeek R1 and deepseek-reasoner accept but silently ignore
    temperature, top_p, and tools. We warn so users don't get
    confused by unexpected behavior.

    Args:
        model_name: The model name.
        config: The request config (may be dict or Pydantic model).
    """
    if not is_reasoning_model(model_name) or config is None:
        return

    for param in _REASONING_IGNORED_PARAMS:
        if _get_config_value(config, param) is not None:
            logger.warning(
                'DeepSeek reasoning model silently ignores parameter;'
                ' removing it from your config will silence this warning.',
                model_name=model_name,
                parameter=param,
            )


class DeepSeekModel:
    """Manages DeepSeek model integration for Genkit.

    This class provides integration with DeepSeek's OpenAI-compatible API,
    allowing DeepSeek models to be exposed as Genkit models. It handles
    client initialization, model information retrieval, and dynamic model
    definition within the Genkit registry.

    For reasoning models (R1, deepseek-reasoner), the generate method
    validates request parameters and warns about silently ignored ones.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        **deepseek_params: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the DeepSeek instance.

        Args:
            model: The name of the specific DeepSeek model (e.g., 'deepseek-chat').
            api_key: DeepSeek API key for authentication.
            **deepseek_params: Additional parameters for the DeepSeek client.
        """
        self.name = model
        client_params = {'api_key': api_key, **deepseek_params}
        self.client: AsyncOpenAI = DeepSeekClient(**client_params)

    def get_model_info(self) -> dict[str, Any] | None:
        """Retrieve metadata and supported features for the specified model.

        Returns:
            A dictionary containing the model's 'name' and 'supports' features.
        """
        model_info = SUPPORTED_DEEPSEEK_MODELS.get(self.name, get_default_model_info(self.name))
        supports_dict = model_info.supports.model_dump(by_alias=True, exclude_none=True) if model_info.supports else {}
        return {
            'name': model_info.label,
            'supports': supports_dict,
        }

    def to_deepseek_model(self) -> Callable:
        """Convert the DeepSeek model into a Genkit-compatible model function.

        For reasoning models, wraps the generate method to validate
        parameters before forwarding to the OpenAI model.

        Returns:
            A callable function that can be used by Genkit.
        """
        openai_model = OpenAIModel(self.name, self.client)
        model_name = self.name

        async def _generate_with_validation(request: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
            _warn_reasoning_params(model_name, request.config)
            return await openai_model.generate(request, ctx)

        # Only wrap with validation for reasoning models.
        if is_reasoning_model(self.name):
            return _generate_with_validation
        return openai_model.generate
