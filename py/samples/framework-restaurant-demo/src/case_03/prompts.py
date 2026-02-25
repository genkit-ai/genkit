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

"""Prompts for case 03."""

from menu_ai import ai
from menu_schemas import DataMenuQuestionInputSchema

from genkit.plugins.google_genai.models.gemini import GoogleAIGeminiVersion

s03_chat_preamble_prompt = ai.define_prompt(
    variant='s03_chatPreamble',
    model=f'googleai/{GoogleAIGeminiVersion.GEMINI_3_FLASH_PREVIEW}',
    input_schema=DataMenuQuestionInputSchema,
    config={'temperature': 0.3},
    system="""{{ role "user" }}
  Hi. What's on the menu today?

  {{ role "model" }}
  I am Walt, a helpful AI assistant here at the restaurant.
  I can answer questions about the food on the menu or any other questions
  you have about food in general. I probably can't help you with anything else.
  Here is today's menu:
  {{#each menuData~}}
  - {{this.title}} ${{this.price}}
    {{this.description}}
  {{~/each}}
  Do you have any questions about the menu?
""",
)
