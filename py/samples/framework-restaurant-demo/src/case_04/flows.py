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


"""Flows for case 04."""

import json
import os
import pathlib

from menu_ai import ai
from menu_schemas import AnswerOutputSchema, MenuItemSchema, MenuQuestionInputSchema
from pydantic import BaseModel, Field

from genkit.blocks.document import Document

from .prompts import s04_rag_data_menu_prompt


class IndexMenuItemsOutputSchema(BaseModel):
    """Output schema for indexing items."""

    rows: int = Field(...)


@ai.flow(name='s04_index_menu_items')
async def s04_index_menu_items_flow(
    menu_items: list[MenuItemSchema],
) -> IndexMenuItemsOutputSchema:
    """Index menu items for retrieval.

    Args:
        menu_items: List of menu items to index.

    Returns:
        Number of items indexed.

    Example:
        >>> await s04_index_menu_items_flow([MenuItemSchema(title='Burger', price=10.0, description='Yum')])
        IndexMenuItemsOutputSchema(rows=1)
    """
    # If empty list provided (e.g., from Dev UI default), load from example file
    # Filter out None values that may come from test input like [null]
    menu_items = [item for item in menu_items if item is not None]

    if not menu_items:
        example_file = os.path.join(pathlib.Path(__file__).parent, 'example.indexMenuItems.json')
        with pathlib.Path(example_file).open() as f:
            menu_data = json.load(f)
        menu_items = [MenuItemSchema(**item) for item in menu_data]

    documents = [
        Document.from_text(f'{item.title} {item.price} \n {item.description}', metadata=item.model_dump())
        for item in menu_items
    ]

    await ai.index(
        indexer='menu-items',
        documents=documents,
    )
    return IndexMenuItemsOutputSchema(rows=len(menu_items))


@ai.flow(name='s04_rag_menu_question')
async def s04_rag_menu_question_flow(
    my_input: MenuQuestionInputSchema,
) -> AnswerOutputSchema:
    """Answer a question using RAG on menu items.

    Args:
        my_input: Input containing the question.

    Returns:
        The answer.

    Example:
        >>> await s04_rag_menu_question_flow(MenuQuestionInputSchema(question='Do you have burgers?'))
        AnswerOutputSchema(answer="Yes, we have...")
    """
    # Retrieve the 3 most relevant menu items for the question
    docs = await ai.retrieve(
        retriever='menu-items',
        query=my_input.question,
        options={'limit': 3},
    )

    menu_data = [MenuItemSchema(**doc.metadata) for doc in docs.documents if doc.metadata]

    # Generate the response
    response = await s04_rag_data_menu_prompt({
        'menuData': [item.model_dump() for item in menu_data],
        'question': my_input.question,
    })
    return AnswerOutputSchema(answer=response.text)
