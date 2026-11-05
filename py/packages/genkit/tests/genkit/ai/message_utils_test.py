#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the message utils."""

from genkit import Message, Part
from genkit._ai._messages import inject_instructions
from genkit._core._typing import (
    Role,
)


def test_inject_instructions_user_message() -> None:
    """Test injecting instructions into a user message."""
    result = inject_instructions(
        messages=[
            Message(
                role=Role.USER,
                content=[Part.from_text('hello'), Part.from_text('world')],
            )
        ],
        instructions='injected',
    )

    assert result == [
        Message(
            role=Role.USER,
            content=[
                Part.from_text('hello'),
                Part.from_text('world'),
                Part.from_text('injected', metadata={'purpose': 'output'}),
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
                content=[Part.from_text('system'), Part.from_text('message')],
            ),
            Message(
                role=Role.USER,
                content=[Part.from_text('hello'), Part.from_text('world')],
            ),
        ],
        instructions='injected',
    )

    assert result == [
        Message(
            role=Role.SYSTEM,
            content=[
                Part.from_text('system'),
                Part.from_text('message'),
                Part.from_text('injected', metadata={'purpose': 'output'}),
            ],
            metadata=None,
        ),
        Message(
            role=Role.USER,
            content=[
                Part.from_text('hello'),
                Part.from_text('world'),
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
                content=[Part.from_text('system'), Part.from_text('message')],
            ),
            Message(
                role=Role.USER,
                content=[
                    Part.from_text('will be overridden', metadata={'purpose': 'output', 'pending': True}),
                    Part.from_text('world'),
                ],
            ),
        ],
        instructions='injected',
    )

    assert result == [
        Message(
            role=Role.SYSTEM,
            content=[
                Part.from_text('system'),
                Part.from_text('message'),
            ],
            metadata=None,
        ),
        Message(
            role=Role.USER,
            content=[
                Part.from_text('injected', metadata={'purpose': 'output'}),
                Part.from_text('world'),
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
                content=[Part.from_text('system'), Part.from_text('message')],
            ),
            Message(
                role=Role.USER,
                content=[
                    Part.from_text('previously injected', metadata={'purpose': 'output'}),
                    Part.from_text('world'),
                ],
            ),
        ],
        instructions='injected',
    )

    assert result == [
        Message(
            role=Role.SYSTEM,
            content=[
                Part.from_text('system'),
                Part.from_text('message'),
            ],
            metadata=None,
        ),
        Message(
            role=Role.USER,
            content=[
                Part.from_text('previously injected', metadata={'purpose': 'output'}),
                Part.from_text('world'),
            ],
            metadata=None,
        ),
    ]
