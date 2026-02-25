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


"""Flows for case 05."""

import base64
import os
import pathlib

from constants import DEFAULT_MENU_QUESTION
from menu_ai import ai
from menu_schemas import (
    AnswerOutputSchema,
    MenuQuestionInputSchema,
    TextMenuQuestionInputSchema,
)

from .prompts import s05_read_menu_prompt, s05_text_menu_prompt


@ai.flow(name='s05_read_menu')
async def s05_read_menu_flow(_: None = None) -> str:
    """Read menu from image file.

    Args:
        _: Ignored.

    Returns:
        The text content of the menu image.

    Example:
        >>> await s05_read_menu_flow()
        "Menu content..."
    """
    image_data_url = inline_data_url('menu.jpeg', 'image/jpeg')
    response = await s05_read_menu_prompt({'image_url': image_data_url})
    return response.text


@ai.flow(name='s05_text_menu_question')
async def s05_text_menu_question_flow(
    my_input: TextMenuQuestionInputSchema,
) -> AnswerOutputSchema:
    """Answer a question based on provided menu text.

    Args:
        my_input: Input containing menu text and question.

    Returns:
        The answer.

    Example:
        >>> await s05_text_menu_question_flow(TextMenuQuestionInputSchema(menu_text='Burger: $10', question='Price?'))
        AnswerOutputSchema(answer="It costs $10")
    """
    response = await s05_text_menu_prompt({'menuText': my_input.menu_text, 'question': my_input.question})
    return AnswerOutputSchema(
        answer=response.text,
    )


@ai.flow(name='s05_vision_menu_question')
async def s05_vision_menu_question_flow(
    my_input: MenuQuestionInputSchema,
) -> AnswerOutputSchema:
    """Answer a question by first reading the menu image.

    Args:
        my_input: Input containing the question.

    Returns:
        The answer.

    Example:
        >>> await s05_vision_menu_question_flow(MenuQuestionInputSchema(question='What is on the menu?'))
        AnswerOutputSchema(answer="We have...")
    """
    # If empty question provided (e.g., from Dev UI default), use the default question
    question = my_input.question if my_input.question else DEFAULT_MENU_QUESTION

    menu_text = await s05_read_menu_flow()
    return await s05_text_menu_question_flow(
        TextMenuQuestionInputSchema(
            question=question,
            menu_text=menu_text,
        )
    )


def inline_data_url(image_filename: str, content_type: str) -> str:
    """Create a data URL for an inline image."""
    file_path = os.path.join(pathlib.Path(__file__).parent, '..', '..', 'data', image_filename)
    with pathlib.Path(file_path).open('rb') as image_file:
        image_data = image_file.read()
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f'data:{content_type};base64,{base64_data}'
