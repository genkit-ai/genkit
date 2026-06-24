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

"""Stateless helper functions for Genkit agents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import Message, ModelResponse
from genkit._core._typing import (
    AgentFinishReason,
    AgentInput,
    AgentStreamChunk,
    FinishReason,
    MessageData,
    ModelResponseChunk,
    Part,
    Role,
    RuntimeError as GenkitRuntimeError,
    ToolRequestPart,
)


class MessagePersister(Protocol):
    async def set_messages(self, messages: list[MessageData]) -> None: ...


HISTORY_TAG = '_genkit_history'
PREAMBLE_KEY = '_genkit_agent_preamble'


def to_error_details(exc: BaseException) -> GenkitRuntimeError:
    status = getattr(exc, 'status', None) or 'INTERNAL'
    message = str(exc) or 'Internal failure'
    details = getattr(exc, 'detail', None) or getattr(exc, 'details', None)
    if details is None and not isinstance(exc, GenkitError):
        details = str(exc)
    return GenkitRuntimeError(status=str(status), message=message, details=details)


def to_agent_finish_reason(fr: FinishReason | None) -> AgentFinishReason | None:
    if fr is None:
        return None
    for reason in AgentFinishReason:
        if reason.value == fr.value:
            return reason
    return AgentFinishReason.UNKNOWN


def tool_request_parts(msg: MessageData | None) -> list[Part]:
    if not msg or not msg.content:
        return []
    parts = []
    for part in msg.content:
        p = part if isinstance(part, Part) else Part.model_validate(part)
        if isinstance(p.root, ToolRequestPart):
            parts.append(p)
    return parts


def emit_interrupt_tool_chunk(ctx: ActionRunContext, response: ModelResponse) -> None:
    parts = tool_request_parts(response.message)
    if not parts:
        return
    ctx.send_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(role=Role.TOOL, content=parts),
        )
    )


def coerce_message(msg: MessageData) -> Message:
    return msg if isinstance(msg, Message) else Message.model_validate(msg.model_dump())


def message_with_metadata(msg: MessageData, metadata: dict[str, object]) -> Message:
    base = coerce_message(msg)
    merged = {**(base.metadata or {}), **metadata}
    return base.model_copy(update={'metadata': merged})


def message_without_metadata_key(msg: MessageData, key: str) -> Message:
    base = coerce_message(msg)
    if not base.metadata or key not in base.metadata:
        return base
    remaining = {k: v for k, v in base.metadata.items() if k != key}
    return base.model_copy(update={'metadata': remaining or None})


def tag_history_for_render(messages: list[MessageData]) -> list[Message]:
    """Mark session messages so prompt render can tell them apart from template output."""
    return [message_with_metadata(m, {HISTORY_TAG: True}) for m in messages]


def apply_preamble_tags(messages: Sequence[MessageData]) -> list[Message]:
    """After render: tag prompt-template messages and strip the internal history marker."""
    tagged: list[Message] = []
    for msg in messages:
        meta = getattr(msg, 'metadata', None) or {}
        if meta.get(HISTORY_TAG):
            tagged.append(message_without_metadata_key(msg, HISTORY_TAG))
        else:
            tagged.append(message_with_metadata(msg, {PREAMBLE_KEY: True}))
    return tagged


async def persist_turn_messages(
    sess: MessagePersister,
    history: list[MessageData],
    response_message: MessageData | Message | None,
    *,
    response: ModelResponse | None = None,
) -> None:
    if response is not None and response.request is not None and response.request.messages:
        clean: list[MessageData] = []
        for m in response.request.messages:
            meta = getattr(m, 'metadata', None) or {}
            if meta.get(PREAMBLE_KEY):
                continue
            clean.append(coerce_message(m))
        if response_message is not None:
            clean.append(coerce_message(response_message))
        await sess.set_messages(clean)
        return

    if response_message is None:
        return

    clean_history: list[MessageData] = [coerce_message(m) for m in history]
    clean_history = [m for m in clean_history if not (m.metadata or {}).get(PREAMBLE_KEY)]
    clean_history.append(coerce_message(response_message))
    await sess.set_messages(clean_history)


def agent_input_has_payload(inp: AgentInput) -> bool:
    """True when ``AgentInput`` carries turn data beyond a detach directive."""
    if inp.message:
        return True
    if inp.resume is not None:
        if inp.resume.restart:
            return True
        if inp.resume.respond:
            return True
    return False
