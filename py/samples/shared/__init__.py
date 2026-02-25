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
"""Shared utilities, types, and flow logic for provider samples."""

from .flows import (
    calculation_logic,
    chat_flow_logic,
    convert_currency_logic,
    describe_image_logic,
    generate_character_logic,
    generate_code_logic,
    generate_greeting_logic,
    generate_multi_turn_chat_logic,
    generate_streaming_story_logic,
    generate_streaming_with_tools_logic,
    generate_weather_logic,
    generate_with_config_logic,
    generate_with_system_prompt_logic,
    solve_reasoning_problem_logic,
    translate_text_logic,
)
from .logging import setup_sample
from .tools import (
    calculate,
    convert_currency,
    get_weather,
)
from .types import (
    CalculatorInput,
    CharacterInput,
    CodeInput,
    ConfigInput,
    CurrencyExchangeInput,
    EmbedInput,
    GreetingInput,
    ImageDescribeInput,
    MultiTurnInput,
    ReasoningInput,
    RpgCharacter,
    Skills,
    StreamingToolInput,
    StreamInput,
    SystemPromptInput,
    TranslateInput,
    WeatherInput,
)

__all__ = [
    'setup_sample',
    'calculate',
    'convert_currency',
    'get_weather',
    'calculation_logic',
    'chat_flow_logic',
    'convert_currency_logic',
    'describe_image_logic',
    'generate_character_logic',
    'generate_code_logic',
    'generate_greeting_logic',
    'generate_multi_turn_chat_logic',
    'generate_streaming_story_logic',
    'generate_streaming_with_tools_logic',
    'generate_weather_logic',
    'generate_with_config_logic',
    'generate_with_system_prompt_logic',
    'solve_reasoning_problem_logic',
    'translate_text_logic',
    'CalculatorInput',
    'CharacterInput',
    'CodeInput',
    'ConfigInput',
    'CurrencyExchangeInput',
    'EmbedInput',
    'GreetingInput',
    'ImageDescribeInput',
    'MultiTurnInput',
    'ReasoningInput',
    'RpgCharacter',
    'Skills',
    'StreamInput',
    'StreamingToolInput',
    'SystemPromptInput',
    'TranslateInput',
    'WeatherInput',
]
