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


"""Tests for the action module."""

import asyncio

import pytest

from genkit import Document, Message, ModelResponse
from genkit._core._middleware._augment_with_context import augment_with_context
from genkit._core._model import ModelHookParams, ModelRequest
from genkit._core._typing import (
    DocumentPart,
    Part,
    Role,
    TextPart,
)


async def run_augmenter(req: ModelRequest) -> ModelRequest:
    """Helper to run the augment_with_context middleware."""
    augmenter = augment_with_context()
    req_future = asyncio.Future()

    async def next_fn(params: ModelHookParams) -> ModelResponse:
        req_future.set_result(params.request)
        return ModelResponse(message=Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]))

    await augmenter.wrap_model(
        ModelHookParams(request=req, on_chunk=None, context={}),
        next_fn,
    )

    return req_future.result()


@pytest.mark.asyncio
async def test_augment_with_context_ignores_no_docs() -> None:
    """Test simple prompt rendering."""
    req = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
        ],
    )

    transformed_req = await run_augmenter(req)

    assert transformed_req == req


@pytest.mark.asyncio
async def test_augment_with_context_adds_docs_as_context() -> None:
    """Test simple prompt rendering."""
    req = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
        ],
        docs=[
            Document(content=[DocumentPart(root=TextPart(text='doc content 1'))]),
            Document(content=[DocumentPart(root=TextPart(text='doc content 2'))]),
        ],
    )

    transformed_req = await run_augmenter(req)

    assert transformed_req == ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='hi')),
                    Part(
                        root=TextPart(
                            text='\n\nUse the following information to complete '
                            + 'your task:\n\n'
                            + '- [0]: doc content 1\n'
                            + '- [1]: doc content 2\n\n',
                            metadata={'purpose': 'context'},
                        )
                    ),
                ],
            )
        ],
        docs=[
            Document(content=[DocumentPart(root=TextPart(text='doc content 1'))]),
            Document(content=[DocumentPart(root=TextPart(text='doc content 2'))]),
        ],
    )


@pytest.mark.asyncio
async def test_augment_with_context_should_not_modify_non_pending_part() -> None:
    """Test simple prompt rendering."""
    req = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(
                        root=TextPart(
                            text='this is already context',
                            metadata={'purpose': 'context'},
                        )
                    ),
                    Part(root=TextPart(text='hi')),
                ],
            ),
        ],
        docs=[
            Document(content=[DocumentPart(root=TextPart(text='doc content 1'))]),
        ],
    )

    transformed_req = await run_augmenter(req)

    assert transformed_req == req


@pytest.mark.asyncio
async def test_augment_with_context_with_purpose_part() -> None:
    """Test simple prompt rendering."""
    req = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(
                        root=TextPart(
                            text='insert context here',
                            metadata={'purpose': 'context', 'pending': True},
                        )
                    ),
                    Part(root=TextPart(text='hi')),
                ],
            ),
        ],
        docs=[
            Document(content=[DocumentPart(root=TextPart(text='doc content 1'))]),
        ],
    )

    transformed_req = await run_augmenter(req)

    assert transformed_req == ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(
                        root=TextPart(
                            text='\n\nUse the following information to complete '
                            + 'your task:\n\n'
                            + '- [0]: doc content 1\n\n',
                            metadata={'purpose': 'context'},
                        )
                    ),
                    Part(root=TextPart(text='hi')),
                ],
            )
        ],
        docs=[
            Document(content=[DocumentPart(root=TextPart(text='doc content 1'))]),
        ],
    )
