#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the message utils."""

from genkit import Message
from genkit._ai._messages import inject_instructions
from genkit._core._typing import (
    Part,
    Role,
    TextPart,
)


def test_inject_instructions_user_message() -> None:
    """Test injecting instructions into a user message."""
    result = inject_instructions(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='hello')), Part(root=TextPart(text='world'))],
            )
        ],
        instructions='injected',
    )

    assert result == [
        Message(
            role=Role.USER,
            content=[
                Part(root=TextPart(text='hello')),
                Part(root=TextPart(text='world')),
                Part(
                    root=TextPart(
                        text='injected',
                        metadata={'purpose': 'output'},
                    )
                ),
            ],
            metadata=None,
        )
    ]


def test_inject_instructions_system_message() -> None:
    """Tests that it injects into the system message."""
    result = inject_instructions(
        messages=[
            Message(
                role=Role.SYSTEM,
                content=[Part(root=TextPart(text='system')), Part(root=TextPart(text='message'))],
            ),
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='hello')), Part(root=TextPart(text='world'))],
            ),
        ],
        instructions='injected',
    )

    assert result == [
        Message(
            role=Role.SYSTEM,
            content=[
                Part(root=TextPart(text='system')),
                Part(root=TextPart(text='message')),
                Part(
                    root=TextPart(
                        text='injected',
                        metadata={'purpose': 'output'},
                    )
                ),
            ],
            metadata=None,
        ),
        Message(
            role=Role.USER,
            content=[
                Part(root=TextPart(text='hello')),
                Part(root=TextPart(text='world')),
            ],
            metadata=None,
        ),
    ]


def test_inject_instructions_purpose() -> None:
    """Tests that it injects into message with purpose metadata."""
    result = inject_instructions(
        messages=[
            Message(
                role=Role.SYSTEM,
                content=[Part(root=TextPart(text='system')), Part(root=TextPart(text='message'))],
            ),
            Message(
                role=Role.USER,
                content=[
                    Part(
                        root=TextPart(
                            text='will be overridden',
                            metadata={'purpose': 'output', 'pending': True},
                        )
                    ),
                    Part(root=TextPart(text='world')),
                ],
            ),
        ],
        instructions='injected',
    )

    assert result == [
        Message(
            role=Role.SYSTEM,
            content=[
                Part(root=TextPart(text='system')),
                Part(root=TextPart(text='message')),
            ],
            metadata=None,
        ),
        Message(
            role=Role.USER,
            content=[
                Part(
                    root=TextPart(
                        text='injected',
                        metadata={'purpose': 'output'},
                    )
                ),
                Part(root=TextPart(text='world')),
            ],
            metadata=None,
        ),
    ]


def test_inject_instructions_short_circuit() -> None:
    """Tests that it slips injection when injected data already present."""
    result = inject_instructions(
        messages=[
            Message(
                role=Role.SYSTEM,
                content=[Part(root=TextPart(text='system')), Part(root=TextPart(text='message'))],
            ),
            Message(
                role=Role.USER,
                content=[
                    Part(
                        root=TextPart(
                            text='previously injected',
                            metadata={'purpose': 'output'},
                        )
                    ),
                    Part(root=TextPart(text='world')),
                ],
            ),
        ],
        instructions='injected',
    )

    assert result == [
        Message(
            role=Role.SYSTEM,
            content=[
                Part(root=TextPart(text='system')),
                Part(root=TextPart(text='message')),
            ],
            metadata=None,
        ),
        Message(
            role=Role.USER,
            content=[
                Part(
                    root=TextPart(
                        text='previously injected',
                        metadata={'purpose': 'output'},
                    )
                ),
                Part(root=TextPart(text='world')),
            ],
            metadata=None,
        ),
    ]
