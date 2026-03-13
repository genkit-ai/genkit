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

"""Common types for provider samples.

Centralizes Pydantic models that are shared across multiple provider
hello samples. Provider-specific types stay in each sample's main.py.
"""

from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input for the calculator tool."""

    operation: str = Field(description='Math operation: add, subtract, multiply, divide', default='add')
    a: float = Field(description='First number', default=123)
    b: float = Field(description='Second number', default=321)


class CharacterInput(BaseModel):
    """Input for character generation."""

    name: str = Field(default='Whiskers', description='Character name')


class CodeInput(BaseModel):
    """Input for code generation flow."""

    task: str = Field(
        default='Write a Python function to calculate fibonacci numbers',
        description='Coding task description',
    )


class ConfigInput(BaseModel):
    """Input for config flow."""

    name: str = Field(default='Ginger', description='User name for greeting')


class CurrencyExchangeInput(BaseModel):
    """Currency conversion input schema."""

    amount: float = Field(description='Amount to convert', default=100)
    from_currency: str = Field(description='Source currency code (e.g., USD)', default='USD')
    to_currency: str = Field(description='Target currency code (e.g., EUR)', default='EUR')


class EmbedInput(BaseModel):
    """Input for embedding flow."""

    text: str = Field(default='Artificial intelligence is transforming the world.', description='Text to embed')


class ImageDescribeInput(BaseModel):
    """Input for image description flow."""

    image_url: str = Field(
        default='https://upload.wikimedia.org/wikipedia/commons/1/13/Cute_kitten.jpg',
        description='URL of the image to describe',
    )


class MultiTurnInput(BaseModel):
    """Input for multi_turn_chat flow."""

    destination: str = Field(default='Japan', description='Travel destination')


class ReasoningInput(BaseModel):
    """Input for reasoning flow."""

    prompt: str = Field(
        default='What is heavier, one kilo of steel or one kilo of feathers? Explain step by step.',
        description='Reasoning question to solve',
    )


class GreetingInput(BaseModel):
    """Input for generate_greeting flow."""

    name: str = Field(default='Mittens', description='Name to greet')


class Skills(BaseModel):
    """A set of core character skills for an RPG character."""

    strength: int = Field(description='strength (0-100)')
    charisma: int = Field(description='charisma (0-100)')
    endurance: int = Field(description='endurance (0-100)')


class RpgCharacter(BaseModel):
    """An RPG character."""

    name: str = Field(description='name of the character')
    back_story: str = Field(description='back story', alias='backStory')
    abilities: list[str] = Field(description='list of abilities (3-4)')
    skills: Skills


class StreamInput(BaseModel):
    """Input for streaming flow."""

    name: str = Field(default='Shadow', description='Name for streaming story')


class StreamingToolInput(BaseModel):
    """Input for streaming tool flow."""

    location: str = Field(default='London', description='Location to get weather for')


class SystemPromptInput(BaseModel):
    """Input for system_prompt flow."""

    question: str = Field(default='What is your quest?', description='Question to ask')


class TranslateInput(BaseModel):
    """Input for translation flow."""

    text: str = Field(
        default='Artificial intelligence is transforming how we build software.',
        description='Text to translate',
    )
    target_language: str = Field(
        default='French',
        description='Target language for translation',
    )


class WeatherInput(BaseModel):
    """Input for the weather tool."""

    location: str = Field(description='City or location name', default='New York')
