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

"""Prompt preamble tagging for prompt-backed agent turns.

When a prompt agent renders a turn, the rendered messages mix two things: the
caller's conversation history and the prompt template's own output (the
"preamble" — system instructions, few-shot examples, etc). We stamp each with a
metadata marker at render time so the persist step can drop the preamble and
keep only real history, instead of letting template boilerplate accumulate in
the session on every turn.
"""

from __future__ import annotations

from collections.abc import Sequence

from genkit._core._model import Message
from genkit._core._typing import MessageData

# Render-time markers: HISTORY_TAG flags a message as prior conversation,
# PREAMBLE_KEY flags it as prompt-template output that persistence should strip.
HISTORY_TAG = '_genkit_history'
PREAMBLE_KEY = '_genkit_agent_preamble'


def coerce_message(msg: MessageData) -> Message:
    return msg if isinstance(msg, Message) else Message.model_validate(msg.model_dump())


def message_with_metadata(*, msg: MessageData, metadata: dict[str, object]) -> Message:
    base = coerce_message(msg)
    merged = {**(base.metadata or {}), **metadata}
    return base.model_copy(update={'metadata': merged})


def message_without_metadata_key(*, msg: MessageData, key: str) -> Message:
    base = coerce_message(msg)
    if not base.metadata or key not in base.metadata:
        return base
    remaining = {k: v for k, v in base.metadata.items() if k != key}
    return base.model_copy(update={'metadata': remaining or None})


def tag_history_for_render(messages: list[MessageData]) -> list[Message]:
    """Mark session messages so prompt render can tell them apart from template output."""
    return [message_with_metadata(msg=m, metadata={HISTORY_TAG: True}) for m in messages]


def apply_preamble_tags(messages: Sequence[MessageData]) -> list[Message]:
    """After render: tag prompt-template messages and strip the internal history marker."""
    tagged: list[Message] = []
    for msg in messages:
        meta = msg.metadata or {}
        if meta.get(HISTORY_TAG):
            tagged.append(message_without_metadata_key(msg=msg, key=HISTORY_TAG))
        else:
            tagged.append(message_with_metadata(msg=msg, metadata={PREAMBLE_KEY: True}))
    return tagged
