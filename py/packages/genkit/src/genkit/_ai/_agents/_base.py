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

from collections.abc import Sequence
from typing import Any, Generic

from opentelemetry import trace as trace_api

# Internal imports from sibling modules
from genkit._ai._agents._client import AgentChat, _init_from
from genkit._ai._agents._runtime import (
    AgentFn,
    AgentRuntime,
    SessionRunner,
    generate_prompt_agent_turn,
    load_session,
)
from genkit._ai._agents._session import (
    SessionStore,
    StateT,
)
from genkit._ai._agents._snapshot import (
    abort_snapshot_in_store,
    lookup_label,
    parse_abort_input,
    parse_snapshot_lookup_input,
    resolve_snapshot,
)
from genkit._ai._agents._transports._inprocess import InProcessTransport
from genkit._ai._agents._types import (
    ClientTransform,
    StateManagement,
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
from genkit._core._action import Action, ActionKind, ActionRunContext, BidiAction, BidiFn
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
    Artifact,
    MessageData,
    MiddlewareRef,
    Part,
    SessionSnapshot,
    SnapshotStatus,
)

# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------


class Agent(BidiAction, Generic[StateT]):
    """In-process agent: registered in the registry, implements AgentAPI.

    Created by ``define_agent`` / ``define_custom_agent``. Extends BidiAction
    so it lives in the registry and can be served over HTTP. Also implements
    AgentAPI so it can be used as a client directly without a separate handle.
    """

    def __init__(
        self,
        *,
        name: str,
        bidi_fn: BidiFn[AgentInit, AgentOutput],
        store: SessionStore | None = None,
        client_transform: ClientTransform | None = None,
        state_schema: type[Any] | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Initialise Agent; transport is inferred from the action + store."""
        agent_meta: dict[str, object] = {'stateManagement': 'server' if store is not None else 'client'}
        # Publish the state shape so tooling (e.g. the Dev UI) can inspect and
        # validate custom state the same way it does tool/prompt schemas.
        if state_schema is not None and hasattr(state_schema, 'model_json_schema'):
            agent_meta['stateSchema'] = state_schema.model_json_schema()
        super().__init__(
            kind=ActionKind.AGENT,
            name=name,
            bidi_fn=bidi_fn,
            description=description,
            metadata={**(metadata or {}), 'agent': agent_meta},
        )
        self.store = store
        self._client_transform = client_transform
        self._state_schema = state_schema

    def _in_process_transport(self) -> InProcessTransport:
        state_management: StateManagement = 'server' if self.store is not None else 'client'
        if self.store is None:
            return InProcessTransport(self, state_management=state_management)
        return InProcessTransport(
            self,
            get_snapshot=self.get_snapshot_data,
            abort_snapshot=self.abort_snapshot_data,
            state_management=state_management,
        )

    async def get_snapshot_data(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        """Read a snapshot by id or latest session leaf (client-visible form)."""
        if self.store is None:
            return None
        return await resolve_snapshot(
            self.store,
            snapshot_id=snapshot_id,
            session_id=session_id,
            client_transform=self._client_transform,
        )

    async def abort_snapshot_data(self, snapshot_id: str) -> SnapshotStatus | None:
        """Abort a running snapshot."""
        if self.store is None:
            return None
        return await abort_snapshot_in_store(self.store, snapshot_id)

    # ------------------------------------------------------------------
    # AgentAPI implementation
    # ------------------------------------------------------------------

    def chat(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
        messages: list[MessageData] | None = None,
        artifacts: list[Artifact] | None = None,
        state: StateT | None = None,
    ) -> AgentChat[StateT]:
        """Starts a new in-process session, or attaches to one via a snapshot/session id or saved conversation state."""
        return AgentChat(
            self._in_process_transport(),
            _init_from(snapshot_id, session_id, messages, artifacts, state),
            state_schema=self._state_schema,
        )

    async def load_chat(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> AgentChat[StateT]:
        """Loads a server snapshot and returns a chat with history restored."""
        session_transport = self._in_process_transport()
        snapshot = await session_transport.get_snapshot(snapshot_id=snapshot_id, session_id=session_id)
        if snapshot is None:
            label = lookup_label(snapshot_id=snapshot_id, session_id=session_id)
            raise ValueError(f'Failed to load chat: Snapshot {label!r} not found.')
        session_transport.state_management = 'server'
        chat = AgentChat(session_transport, state_schema=self._state_schema)
        chat._load_from_snapshot(snapshot)
        return chat

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        """Reads a snapshot without starting a session."""
        return await self.get_snapshot_data(snapshot_id=snapshot_id, session_id=session_id)

    async def abort(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts a running snapshot."""
        return await self.abort_snapshot_data(snapshot_id)


# ---------------------------------------------------------------------------
# agent definition APIs
# ---------------------------------------------------------------------------


def define_custom_agent(
    registry: Registry,
    name: str,
    fn: AgentFn,
    *,
    store: SessionStore[StateT] | None = None,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
    state_schema: type[StateT] | None = None,
    description: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Agent[StateT]:
    """Register a custom agent; ``fn`` owns the turn loop via ``session_runner.run``.

    Pass ``state_schema`` (a Pydantic model) to type the custom state: the chat's
    ``state``, each turn's ``response.state``, and streamed ``chunk.custom`` come
    back as that model, validated on the way in, instead of a bare dict.
    """
    resolved_transform = resolve_client_transform(
        client_transform=client_transform,
        transform=transform,
    )

    async def bidi_fn(
        init: AgentInit,
        in_queue: CloseableQueue[AgentInput],
        out_queue: CloseableQueue[AgentStreamChunk],
    ) -> AgentOutput:
        session, parent = await load_session(init, store, agent_name=name, state_schema=state_schema)
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
        client_transform=resolved_transform,
        state_schema=state_schema,
    )
    registry.register_action_from_instance(agent)

    if store is not None:
        _register_snapshot_actions(registry, name, agent)

    return agent


def _register_snapshot_actions(registry: Registry, name: str, agent: Agent) -> None:
    async def snapshot_fn(input_val: Any) -> SessionSnapshot | None:  # noqa: ANN401
        sid, sess_id = parse_snapshot_lookup_input(input_val)
        return await agent.get_snapshot_data(snapshot_id=sid, session_id=sess_id)

    async def abort_fn(input_val: Any) -> dict[str, object]:  # noqa: ANN401
        snapshot_id = parse_abort_input(input_val)
        status = await agent.abort_snapshot_data(snapshot_id)
        return {'snapshotId': snapshot_id, 'status': status}

    registry.register_action_from_instance(
        Action(
            kind=ActionKind.AGENT_SNAPSHOT,
            name=name,
            fn=snapshot_fn,
            description=f'Gets snapshot data for {name} by snapshotId or sessionId',
        )
    )
    registry.register_action_from_instance(
        Action(
            kind=ActionKind.AGENT_ABORT,
            name=name,
            fn=abort_fn,
            description=f'Aborts {name} agent by snapshotId',
        )
    )


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
    store: SessionStore[StateT] | None = None,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
    state_schema: type[StateT] | None = None,
) -> Agent[StateT]:
    """Register a prompt-backed agent.

    Conversation input arrives via ``AgentChat.send``;
    ``system`` is the only static preamble re-rendered each turn alongside
    session history. For template variables, few-shot messages, RAG docs, or
    prompt variants, use ``define_prompt`` + ``define_prompt_agent`` instead.

    Pass ``state_schema`` (a Pydantic model) to type the custom state that tools
    read and write via the session — the chat's ``state``, ``response.state``,
    and streamed ``chunk.custom`` come back as that model instead of a dict.
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
        client_transform=client_transform,
        transform=transform,
        state_schema=state_schema,
        description=description,
        metadata=metadata,
    )


def define_prompt_agent(
    registry: Registry,
    name: str,
    *,
    store: SessionStore[StateT] | None = None,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
    state_schema: type[StateT] | None = None,
    description: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Agent[StateT]:
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
        client_transform=client_transform,
        transform=transform,
        state_schema=state_schema,
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
