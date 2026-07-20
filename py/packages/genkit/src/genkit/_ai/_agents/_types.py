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

"""Type definitions and transforms for Genkit agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from genkit._core._typing import (
    AgentFinishReason,
    AgentStreamChunk,
    SessionState,
)

StateManagement = Literal['server', 'client']

# state_transform / chunk_transform are the two egress redaction hooks an agent can
# set — they shape only what a client sees, never persisted state. Both fail closed:
# a hook that raises propagates rather than leaking the unredacted value, so keeping
# the two consistent (e.g. redacting the same fields) is the caller's job.
#
# StateTransform — redact or reshape session state before it leaves the server.
# It shapes snapshot reads, client-managed AgentOutput, and the baseline for streamed
# custom patches. It must return a state: to hide the whole thing, return an
# explicitly cleared one — there's no "return None to omit," precisely so a transform
# that forgets to return can't silently wipe the client's entire view.
StateTransform = Callable[[SessionState], SessionState]

# ChunkTransform — reshape or drop a stream chunk before it reaches the client.
# Returning None drops the chunk (the point of the hook, e.g. hide artifact chunks);
# the blast radius is one chunk, so unlike the state hook that's safe.
ChunkTransform = Callable[[AgentStreamChunk], AgentStreamChunk | None]


@dataclass
class TurnResult:
    """What an agent turn function returns to tell the loop how the turn ended."""

    finish_reason: AgentFinishReason | None = None
