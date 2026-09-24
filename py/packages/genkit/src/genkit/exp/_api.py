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

"""Experimental Genkit instance that grows agent methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, overload

from pydantic import BaseModel

from genkit._ai._agents._base import (
    Agent,
    define_agent,
    define_custom_agent,
    define_prompt_agent,
)
from genkit._ai._agents._runtime import AgentFn
from genkit._ai._agents._session import SessionStore, StateT
from genkit._ai._agents._types import ChunkTransform, StateTransform
from genkit._ai._aio import Genkit as StableGenkit
from genkit._ai._tools import Tool
from genkit._core._action import ActionKind
from genkit._core._error import GenkitError
from genkit._core._middleware import BaseMiddleware
from genkit._core._model import ModelConfigDict, ModelRef, ModelRefConfigT
from genkit._core._typing import MiddlewareRef, Part


class Genkit(StableGenkit):
    """Genkit plus the agent APIs, which are still experimental.

    Importing this class is how you opt into agents. The stable
    ``from genkit import Genkit`` does not grow these methods.
    """

    async def agent(self, name: str) -> Agent:
        """Look up a registered agent by name."""
        resolved = await self.registry.resolve_action(ActionKind.AGENT, name)
        if resolved is None:
            raise GenkitError(
                status='NOT_FOUND',
                message=f"Agent '{name}' not found in registry.",
            )
        if not isinstance(resolved, Agent):
            raise GenkitError(
                status='INTERNAL',
                message=f"Registry entry '{name}' is not an Agent.",
            )
        return resolved

    def define_custom_agent(
        self,
        name: str,
        fn: AgentFn,
        *,
        store: SessionStore[StateT] | None = None,
        state_transform: StateTransform | None = None,
        chunk_transform: ChunkTransform | None = None,
        state_schema: type[StateT] | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Agent[StateT]:
        """Define and register an agent with full control over the turn loop.

        fn receives (SessionRunner, ActionRunContext) and must call sess.run(handle_turn)
        to process inputs, then return an AgentResult.

        Pass ``state_schema`` (a Pydantic model) to type the custom state, so the
        chat's ``state``, ``response.state``, and streamed ``chunk.custom`` come
        back as that model instead of a dict.
        """
        return define_custom_agent(
            registry=self.registry,
            name=name,
            fn=fn,
            store=store,
            state_transform=state_transform,
            chunk_transform=chunk_transform,
            state_schema=state_schema,
            description=description,
            metadata=metadata,
        )

    @overload
    def define_agent(
        self,
        name: str,
        *,
        model: ModelRef[ModelRefConfigT] | str | None = None,
        system: str | list[Part] | None = None,
        tools: Sequence[str | Tool] | None = None,
        use: Sequence[BaseMiddleware | MiddlewareRef] | None = None,
        config: ModelConfigDict,
        max_turns: int | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
        store: SessionStore[StateT] | None = None,
        state_transform: StateTransform | None = None,
        chunk_transform: ChunkTransform | None = None,
        state_schema: type[StateT] | None = None,
    ) -> Agent[StateT]: ...

    @overload
    def define_agent(
        self,
        name: str,
        *,
        model: ModelRef[ModelRefConfigT] | str | None = None,
        system: str | list[Part] | None = None,
        tools: Sequence[str | Tool] | None = None,
        use: Sequence[BaseMiddleware | MiddlewareRef] | None = None,
        config: ModelRefConfigT | Mapping[str, Any] | None = None,
        max_turns: int | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
        store: SessionStore[StateT] | None = None,
        state_transform: StateTransform | None = None,
        chunk_transform: ChunkTransform | None = None,
        state_schema: type[StateT] | None = None,
    ) -> Agent[StateT]: ...

    def define_agent(
        self,
        name: str,
        *,
        model: ModelRef[ModelRefConfigT] | str | None = None,
        system: str | list[Part] | None = None,
        tools: Sequence[str | Tool] | None = None,
        use: Sequence[BaseMiddleware | MiddlewareRef] | None = None,
        config: BaseModel | ModelConfigDict | Mapping[str, Any] | None = None,
        max_turns: int | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
        store: SessionStore[StateT] | None = None,
        state_transform: StateTransform | None = None,
        chunk_transform: ChunkTransform | None = None,
        state_schema: type[StateT] | None = None,
    ) -> Agent[StateT]:
        """Define a prompt-backed agent.

        Each turn: attaches session history, calls generate with streaming,
        updates session. Pass resume in AgentInput to resume from an interrupt.

        Pass ``state_schema`` (a Pydantic model) to type the custom state tools
        read and write — the chat's ``state``, ``response.state``, and streamed
        ``chunk.custom`` come back as that model instead of a dict.

        Example:
            from genkit.exp import Genkit, InMemorySessionStore
            from genkit_google_genai import GoogleAI

            ai = Genkit(plugins=[GoogleAI()])
            agent = ai.define_agent(
                name='weatherAgent',
                model=GoogleAI.gemini_model('gemini-flash-latest'),
                system='Weather assistant.',
                tools=[current_weather],
                store=InMemorySessionStore(),
            )
            chat = agent.chat()
            res = await chat.send('Weather in Paris?')
        """
        return define_agent(
            registry=self.registry,
            name=name,
            model=model,
            system=system,
            tools=tools,
            use=use,
            config=config,
            max_turns=max_turns,
            description=description,
            metadata=metadata,
            store=store,
            state_transform=state_transform,
            chunk_transform=chunk_transform,
            state_schema=state_schema,
        )

    def define_prompt_agent(
        self,
        name: str,
        *,
        store: SessionStore[StateT] | None = None,
        state_transform: StateTransform | None = None,
        chunk_transform: ChunkTransform | None = None,
        state_schema: type[StateT] | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Agent[StateT]:
        """Wire an already-registered prompt as an agent.

        Looks up the prompt named `name` from the registry. Use when the prompt
        is defined via ai.define_prompt() or loaded from a .prompt file.
        """
        return define_prompt_agent(
            registry=self.registry,
            name=name,
            store=store,
            state_transform=state_transform,
            chunk_transform=chunk_transform,
            state_schema=state_schema,
            description=description,
            metadata=metadata,
        )
