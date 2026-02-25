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


"""Flows for case 03."""

import json
import os
import pathlib

from menu_ai import ai

from genkit.core.typing import Message, Part, Role, TextPart
from genkit.plugins.google_genai.models.gemini import GoogleAIGeminiVersion as GeminiVersion

from .chats import (
    ChatHistoryStore,
    ChatSessionInputSchema,
    ChatSessionOutputSchema,
)

menu_json_path = os.path.join(pathlib.Path(__file__).parent, '..', '..', 'data', 'menu.json')
with pathlib.Path(menu_json_path).open() as f:
    menu_data = json.load(f)

formatted_menu_data = '\n'.join([f'- {r["title"]} ${r["price"]}\n{r["description"]}' for r in menu_data])

preamble = [
    Message(
        role=Role.USER,
        content=[
            Part(root=TextPart(text="Hi. What's on the menu today?")),
        ],
    ),
    Message(
        role=Role.MODEL,
        content=[
            Part(
                root=TextPart(
                    text=f"""I am Walt, a helpful AI assistant here at the restaurant.
I can answer questions about the food on the menu or any other questions
you have about food in general. I probably can't help you with anything else.
Here is today's menu:
{formatted_menu_data}
Do you have any questions about the menu?"""
                )
            ),
        ],
    ),
]

chat_history_store = ChatHistoryStore(
    preamble=preamble,
)


@ai.flow(name='s03_multi_turn_chat')
async def s03_multi_turn_chat_flow(
    my_input: ChatSessionInputSchema,
) -> ChatSessionOutputSchema:
    """Run a multi-turn chat session about the menu.

    Args:
        my_input: Input containing session ID and question.

    Returns:
        The chat session history and ID.

    Example:
        >>> await s03_multi_turn_chat_flow(ChatSessionInputSchema(question='Hi', session_id='123'))
        ChatSessionOutputSchema(session_id="123", history=[...])
    """
    history = chat_history_store.read(my_input.session_id)

    llm_response = await ai.generate(
        model=f'googleai/{GeminiVersion.GEMINI_3_FLASH_PREVIEW}',
        messages=history,
        prompt=[Part(root=TextPart(text=my_input.question))],
    )

    history = llm_response.messages

    chat_history_store.write(my_input.session_id, history)

    return ChatSessionOutputSchema(
        session_id=my_input.session_id,
        history=history,
    )
