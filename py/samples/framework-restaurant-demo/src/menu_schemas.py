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


"""Schemas for the menu AI sample."""

from constants import DEFAULT_MENU_QUESTION, DEFAULT_MENU_TEXT
from pydantic import BaseModel, Field


class MenuItemSchema(BaseModel):
    """Schema for a menu item."""

    title: str = Field(..., description='The name of the menu item')
    description: str = Field(..., description='Details including ingredients and preparation')
    price: float = Field(..., description='Price in dollars')


class MenuQuestionInputSchema(BaseModel):
    """Input schema for the menu question prompt."""

    question: str = Field(default=DEFAULT_MENU_QUESTION, description='A question about the menu')


class AnswerOutputSchema(BaseModel):
    """Output schema for the answer prompt."""

    answer: str = Field(..., description='An answer to a menu-related question')


class DataMenuQuestionInputSchema(BaseModel):
    """Input schema for the data menu question prompt."""

    menu_data: list[MenuItemSchema] = Field(..., alias='menuData')
    question: str = Field(..., description='A question about the menu')


class TextMenuQuestionInputSchema(BaseModel):
    """Input schema for the text menu question prompt."""

    menu_text: str = Field(
        default=DEFAULT_MENU_TEXT,
        description='The menu text content',
        alias='menuText',
    )
    question: str = Field(default=DEFAULT_MENU_QUESTION, description='A question about the menu')


class MenuToolOutputSchema(BaseModel):
    """Output schema for the menu tool."""

    menu_data: list[MenuItemSchema] = Field(..., description='A list of all the items on the menu')


class ReadMenuImagePromptSchema(BaseModel):
    """Input schema for the read menu image prompt."""

    image_url: str = Field(...)


class ReadMenuPromptOutputSchema(BaseModel):
    """Output schema for the read menu prompt."""

    menu_text: str = Field(...)
