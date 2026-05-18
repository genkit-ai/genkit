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

"""Middleware for the Genkit framework."""

from collections.abc import Awaitable, Callable

from genkit._ai._model import (
    Message,
    ModelMiddleware,
    ModelRequest,
    ModelResponse,
    text_from_content,
)
from genkit._core._action import ActionRunContext
from genkit._core._model import Document
from genkit._core._typing import (
    Part,
    TextPart,
)

CONTEXT_PREFACE = '\n\nUse the following information to complete your task:\n\n'


def context_item_template(d: Document, index: int) -> str:
    """Render a document as a citation line for context injection."""
    out = '- '
    ref = (d.metadata and (d.metadata.get('ref') or d.metadata.get('id'))) or index
    out += f'[{ref}]: '
    out += text_from_content(d.content) + '\n'
    return out


def augment_with_context() -> ModelMiddleware:
    """Middleware that injects document context into the last user message."""

    async def middleware(
        req: ModelRequest,
        ctx: ActionRunContext,
        next_middleware: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if not req.docs:
            return await next_middleware(req, ctx)

        user_message = last_user_message(req.messages)
        if not user_message:
            return await next_middleware(req, ctx)

        context_part_index = -1
        for i, part in enumerate(user_message.content):
            part_metadata = part.root.metadata
            if isinstance(part_metadata, dict) and part_metadata.get('purpose') == 'context':
                context_part_index = i
                break

        context_part = user_message.content[context_part_index] if context_part_index >= 0 else None

        if context_part:
            metadata = context_part.root.metadata
            if not (isinstance(metadata, dict) and metadata.get('pending')):
                return await next_middleware(req, ctx)

        out = CONTEXT_PREFACE
        for i, doc_data in enumerate(req.docs):
            doc = Document(content=doc_data.content, metadata=doc_data.metadata)
            out += context_item_template(doc, i)
        out += '\n'

        text_part = Part(root=TextPart(text=out, metadata={'purpose': 'context'}))
        if context_part_index >= 0:
            user_message.content[context_part_index] = text_part
        else:
            if not user_message.content:
                user_message.content = []
            user_message.content.append(text_part)

        return await next_middleware(req, ctx)

    return middleware


def last_user_message(messages: list[Message]) -> Message | None:
    """Find the last user message in a list."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == 'user':
            return messages[i]
    return None
