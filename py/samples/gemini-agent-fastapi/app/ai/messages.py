# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Message helpers for API responses."""

from __future__ import annotations

from genkit import Message
from genkit._ai._model import text_from_message
from genkit._core._typing import MessageData


def message_text(msg: MessageData) -> str:
    """Extract plain text from a Genkit message for JSON API responses."""
    return text_from_message(Message(msg))
