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


"""Chats for case 03."""

from pydantic import BaseModel, Field

from genkit.core.typing import Message


class ChatSessionInputSchema(BaseModel):
    """Input for starting a chat session."""

    session_id: str = Field(default='demo-session-1', description='Unique session identifier')
    question: str = Field(default='What dishes do you recommend?', description='Question about the menu')


class ChatSessionOutputSchema(BaseModel):
    """Output for a chat session."""

    session_id: str
    history: list[Message]


ChatHistory = list[Message]


class ChatHistoryStore:
    """Store for chat history."""

    def __init__(self, preamble: ChatHistory | None = None) -> None:
        """Initialize the store."""
        self.preamble = preamble if preamble is not None else []
        self.sessions: dict[str, ChatHistory] = {}

    def write(self, session_id: str, history: ChatHistory) -> None:
        """Write history for a session."""
        self.sessions[session_id] = history

    def read(self, session_id: str) -> ChatHistory:
        """Read history for a session."""
        return self.sessions.get(session_id, self.preamble)
