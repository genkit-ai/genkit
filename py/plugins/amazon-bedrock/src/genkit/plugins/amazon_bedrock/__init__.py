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

"""Amazon Bedrock plugin for Genkit."""

from genkit.plugins.amazon_bedrock.config import BedrockConfig, ModelDefinition
from genkit.plugins.amazon_bedrock.plugin import Bedrock, bedrock_name

__all__ = [
    'Bedrock',
    'BedrockConfig',
    'ModelDefinition',
    'bedrock_name',
]
