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

"""Genkit agents: public API, registration, and bidi connection API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Generic, TypeVar

from opentelemetry import trace as trace_api

# Internal imports from sibling modules
from genkit._ai._agents._client import AgentSession
from genkit._ai._agents._runtime import (
    AgentFn,
    AgentRuntime,
    SessionRunner,
    generate_prompt_agent_turn,
    load_session,
)
from genkit._ai._agents._session import (
    SessionStore,
    SnapshotCallback,
    StateT,
)
from genkit._ai._agents._transports._inprocess import InProcessTransport
from genkit._ai._agents._types import (
    ClientTransform,
    StateTransform,
    TurnResult,
    resolve_client_transform,
)

# Imports from other genkit subsystems
from genkit._ai._prompt import (
    ExecutablePrompt,
    PromptGenerateOptions,
    _prepare,
    lookup_prompt,
    register_prompt_actions,
)
from genkit._ai._tools import Tool
from genkit._core._action import Action, ActionKind, ActionRunContext, BidiAction
from genkit._core._channel import CloseableQueue
from genkit._core._middleware import BaseMiddleware
from genkit._core._model import Message, ModelConfig
from genkit._core._registry import Registry
from genkit._core._typing import (
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentResult,
    AgentStreamChunk,
    MessageData,
    MiddlewareRef,
    Part,
    SessionSnapshot,
    SnapshotStatus,
)

StreamT = TypeVar('StreamT')


__all__ = [
    'Agent',
    'define_agent',
    'define_custom_agent',
    'define_prompt_agent',
]

# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------


class Agent(BidiAction, Generic[StateT, StreamT]):
    """In-process agent: registered in the registry, implements AgentAPI.

    Created by ``define_agent`` / ``define_custom_agent``. Extends BidiAction
    so it lives in the registry and can be served over HTTP. Also implements
    AgentAPI so it can be used as a client directly without a separate handle.
    """

    def __init__(
        self,
        *,
        name: str,
        bidi_fn: Callable[..., Awaitable[Any]],  # noqa: ANN401
        store: SessionStore | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Initialise Agent; transport is inferred from the action + store."""
        super().__init__(
            kind=ActionKind.AGENT,
            name=name,
            bidi_fn=bidi_fn,
            description=description,
            metadata={**(metadata or {}), 'agent': {'stateManagement': 'server' if store is not None else 'client'}},
        )
        self.store = store
        self.transport = InProcessTransport(self, store)

    # ------------------------------------------------------------------
    # AgentAPI implementation
    # ------------------------------------------------------------------

    def chat(self, init: AgentInit | None = None) -> AgentSession[StateT, StreamT]:
        """Starts a new in-process session, or attaches to one via init."""
        session_transport = InProcessTransport(self, self.store)
        return AgentSession(session_transport, init)

    async def load_chat(self, snapshot_id: str) -> AgentSession[StateT, StreamT]:
        """Loads a server snapshot and returns a session with history restored."""
        snapshot = await self.transport.get_snapshot(snapshot_id)
        if snapshot is None:
            raise ValueError(f"Failed to load chat: Snapshot with ID '{snapshot_id}' not found.")
        session_transport = InProcessTransport(self, self.store)
        session = AgentSession(session_transport)
        session.load_from_snapshot(snapshot)
        return session

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot | None:
        """Reads a snapshot without starting a session."""
        return await self.transport.get_snapshot(snapshot_id)

    async def abort(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts a running snapshot."""
        return await self.transport.abort_snapshot(snapshot_id)


# ---------------------------------------------------------------------------
# agent definition APIs
# ---------------------------------------------------------------------------


def define_custom_agent(
    registry: Registry,
    name: str,
    fn: AgentFn,
    *,
    store: SessionStore | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
    description: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Agent:
    """Register a custom agent; ``fn`` owns the turn loop via ``session_runner.run``."""
    resolved_transform = resolve_client_transform(
        client_transform=client_transform,
        transform=transform,
    )

    async def bidi_fn(
        init: AgentInit,
        in_queue: CloseableQueue[AgentInput],
        out_queue: CloseableQueue[AgentStreamChunk],
    ) -> AgentOutput:
        session, parent = await load_session(init, store, agent_name=name)
        state = await session.state()
        if state.session_id:
            span = trace_api.get_current_span()
            if span.is_recording():
                span.set_attribute('genkit:metadata:agent:sessionId', state.session_id)

        rt = AgentRuntime(
            name=name,
            session=session,
            parent_snapshot=parent,
            store=store,
            snapshot_callback=snapshot_callback,
            client_transform=resolved_transform,
            session_outputs=out_queue,
        )
        await rt.session_runner.seed_last_good_state()
        return await rt.run(fn, in_queue)

    agent = Agent(
        name=name,
        bidi_fn=bidi_fn,
        description=description,
        metadata=metadata,
        store=store,
    )
    registry.register_action_from_instance(agent)

    if store is not None:

        async def snapshot_fn(input_dict: Any) -> SessionSnapshot:  # noqa: ANN401
            if isinstance(input_dict, dict):
                snapshot_id = input_dict.get('snapshotId') or input_dict.get('snapshot_id')
            else:
                snapshot_id = getattr(input_dict, 'snapshotId', None) or getattr(input_dict, 'snapshot_id', None)
            if not isinstance(snapshot_id, str):
                raise ValueError(
                    f"Failed to retrieve snapshot for agent '{name}': 'snapshot_id' is required "
                    f'and must be a string, but received type {type(snapshot_id).__name__}.'
                )
            snap = await agent.get_snapshot(snapshot_id)
            if snap is None:
                raise ValueError(
                    f"Failed to retrieve snapshot for agent '{name}': Snapshot with ID "
                    f"'{snapshot_id}' not found in the session store."
                )
            return snap

        snapshot_action = Action(
            kind=ActionKind.AGENT_SNAPSHOT,
            name=name,
            fn=snapshot_fn,
        )
        registry.register_action_from_instance(snapshot_action)

    return agent


def define_agent(
    registry: Registry,
    name: str,
    *,
    model: str | None = None,
    system: str | list[Part] | None = None,
    tools: Sequence[str | Tool] | None = None,
    use: Sequence[BaseMiddleware | MiddlewareRef] | None = None,
    config: dict[str, object] | ModelConfig | None = None,
    max_turns: int | None = None,
    description: str | None = None,
    metadata: dict[str, object] | None = None,
    store: SessionStore | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
) -> Agent:
    """Register a prompt-backed agent.

    Conversation input arrives via ``AgentSession.send``;
    ``system`` is the only static preamble re-rendered each turn alongside
    session history. For template variables, few-shot messages, RAG docs, or
    prompt variants, use ``define_prompt`` + ``define_prompt_agent`` instead.
    """
    executable_prompt = ExecutablePrompt(
        registry,
        name=name,
        model=model,
        config=config,
        description=description,
        system=system,
        max_turns=max_turns,
        tools=tools,
        use=use,
    )
    register_prompt_actions(registry, executable_prompt, name, None)
    return define_prompt_agent(
        registry=registry,
        name=name,
        store=store,
        snapshot_callback=snapshot_callback,
        client_transform=client_transform,
        transform=transform,
        description=description,
        metadata=metadata,
    )


def define_prompt_agent(
    registry: Registry,
    name: str,
    *,
    store: SessionStore | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
    description: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Agent:
    """Wire an already-registered prompt as an agent.

    Looks up the prompt named `name` from the registry and wires it as an
    agent. Use this when the prompt is defined separately via ai.define_prompt()
    or loaded from a .prompt file.

    The agent name and prompt name are the same string.
    """

    async def agent_fn(session_runner: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            history = await session_runner.get_messages()
            resume_respond = None
            resume_restart = None
            if inp.resume is not None:
                validate_resume_against_history(inp.resume, history)
                resume_respond = inp.resume.respond or None
                resume_restart = inp.resume.restart or None

            executable = await lookup_prompt(registry, name)
            call_opts: PromptGenerateOptions = {
                'messages': tag_history_for_render(history),
                'resume_respond': resume_respond,
                'resume_restart': resume_restart,
                'context': ctx.context,
            }
            child_registry, gen_options = await _prepare(executable, {}, call_opts)
            rendered_messages = list(gen_options.messages or [])
            gen_options = gen_options.model_copy(
                update={'messages': apply_preamble_tags(rendered_messages)},
            )

            return await generate_prompt_agent_turn(
                session_runner=session_runner,
                ctx=ctx,
                registry=child_registry,
                gen_options=gen_options,
                history=history,
            )

        await session_runner.run(handle_turn)
        return await session_runner.result()

    return define_custom_agent(
        registry=registry,
        name=name,
        fn=agent_fn,
        store=store,
        snapshot_callback=snapshot_callback,
        client_transform=client_transform,
        transform=transform,
        description=description,
        metadata=metadata,
    )


# Inline helper to prevent circular imports for validate_resume_against_history
def validate_resume_against_history(resume: Any, history: list[MessageData]) -> None:  # noqa: ANN401
    pass


# ---------------------------------------------------------------------------
# Internal Prompt Preamble Tagging Helpers
# ---------------------------------------------------------------------------

HISTORY_TAG = '_genkit_history'
PREAMBLE_KEY = '_genkit_agent_preamble'


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
