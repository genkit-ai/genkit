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

"""Shared payload helpers for serving agents over the reflection APIs.

Both reflection servers (the v1 streaming endpoint and the v2 JSON-RPC channel)
turn a raw JSON payload into the two things an agent turn needs — an
``AgentInit`` (session identity) and an ``AgentInput`` (the turn message).
Sharing that here keeps the Dev UI behaving identically whichever endpoint drives
the agent.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from genkit._core._action import BidiAction
from genkit._core._typing import AgentInit, AgentInput


def agent_has_server_store(action: BidiAction[Any, Any, Any]) -> bool:
    """True when the agent keeps session state on the server rather than the client."""
    agent_meta = (action.metadata or {}).get('agent')
    agent_dict = cast(dict[str, Any], agent_meta) if isinstance(agent_meta, dict) else {}
    return agent_dict.get('stateManagement') == 'server'


def resolve_agent_init(action: BidiAction[Any, Any, Any], init_val: object) -> AgentInit:
    """Validate a raw init payload into an ``AgentInit``, normalized for the agent's store.

    For a server-store agent we mint a session id when the caller didn't supply
    one, and drop any caller-provided state — the store owns state, so a client
    copy could otherwise overwrite the server's history with a stale snapshot.
    """
    init = AgentInit.model_validate(init_val) if isinstance(init_val, dict) else AgentInit()
    if agent_has_server_store(action):
        if not init.session_id and not init.snapshot_id:
            init.session_id = str(uuid4())
        init.state = None
    return init


def resolve_agent_input(input_val: object) -> AgentInput:
    """Validate a raw per-turn input payload into an ``AgentInput`` (empty when absent)."""
    return AgentInput.model_validate(input_val) if input_val is not None else AgentInput()
