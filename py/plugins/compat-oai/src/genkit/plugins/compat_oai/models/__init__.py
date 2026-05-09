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


"""OpenAI Compatible Models for Genkit."""

from .audio import (
    SUPPORTED_STT_MODELS,
    SUPPORTED_TTS_MODELS,
    OpenAISTTModel,
    OpenAITTSModel,
)
from .handler import OpenAIModelHandler
from .image import (
    SUPPORTED_IMAGE_MODELS,
    OpenAIImageModel,
)
from .model import OpenAIModel
from .model_info import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_OPENAI_COMPAT_MODELS,
    SUPPORTED_OPENAI_MODELS,
    PluginSource,
    get_default_model_info,
)

__all__ = [
    'OpenAIImageModel',
    'OpenAIModel',
    'OpenAIModelHandler',
    'OpenAISTTModel',
    'OpenAITTSModel',
    'PluginSource',
    'SUPPORTED_EMBEDDING_MODELS',
    'SUPPORTED_IMAGE_MODELS',
    'SUPPORTED_OPENAI_COMPAT_MODELS',
    'SUPPORTED_OPENAI_MODELS',
    'SUPPORTED_STT_MODELS',
    'SUPPORTED_TTS_MODELS',
    'get_default_model_info',
]
