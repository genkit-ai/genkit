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

import json
from collections.abc import Sequence
from typing import Any, Generic

from opentelemetry import trace as trace_api

# Internal imports from sibling modules
from genkit._ai._agents._client import AgentClient, part_roots
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
from genkit._core._error import GenkitError
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
    Resume,
    Role,
    SessionSnapshot,
    SnapshotStatus,
    ToolRequest,
)

# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------


class Agent(
    BidiAction[AgentInput, AgentOutput, AgentStreamChunk, AgentInit],
    AgentClient[StateT],
    Generic[StateT],
):
    """The low-level agent primitive: a BidiAction that lives in the registry.

    Created by ``define_agent`` / ``define_custom_agent``. As a BidiAction it's
    the thing that gets registered and served over HTTP. It's *also* an
    ``AgentClient``, so the ergonomic chat surface (``chat``/``load_chat``/
    ``get_snapshot``/``abort``) is inherited rather than reimplemented — it's the
    same client used for remote agents, just pointed at an in-process transport.
    So talking to a local agent and a remote one go through one client, not two.

    The action generics are pinned to the agent turn shape: each turn's input is
    an ``AgentInput``, streamed chunks are ``AgentStreamChunk``, the turn result
    is an ``AgentOutput``, and ``init`` (session identity) is an ``AgentInit``.
    """

    def __init__(
        self,
        *,
        name: str,
        bidi_fn: BidiFn[AgentInit, AgentInput, AgentStreamChunk, AgentOutput],
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
        # BidiAction is inited via super() (not an explicit BidiAction.__init__)
        # so the type checker keeps Agent's bound generics; calling it explicitly
        # makes it re-infer them and the invariant ChunkT collapses to Never. The
        # AgentClient half is inited explicitly just below.
        super().__init__(
            kind=ActionKind.AGENT,
            name=name,
            bidi_fn=bidi_fn,
            description=description,
            # 'agent' is framework-owned metadata (state management + schema), so
            # it always wins over anything the caller put under that key.
            metadata={**(metadata or {}), 'agent': agent_meta},
            # An agent turn always resumes (or starts) a session, so its init is
            # always an AgentInit. Declaring it here validates the session
            # identity up front and lets a bare run() default to a fresh session.
            init_schema=AgentInit,
            # Each turn's input is an AgentInput. Declaring it means a raw payload
            # (e.g. an HTTP JSON body) is coerced into an AgentInput before the
            # turn runs, instead of reaching the runtime as a bare dict.
            input_schema=AgentInput,
        )
        self.store = store
        self._client_transform = client_transform
        # The AgentClient half (chat/load_chat/get_snapshot/abort rides along)
        # runs against an in-process transport that drives this very action — the
        # way remote_agent runs against an HTTP one.
        AgentClient.__init__(self, self._in_process_transport(), state_schema=state_schema)

    def _in_process_transport(self) -> InProcessTransport:
        state_management: StateManagement = 'server' if self.store is not None else 'client'
        if self.store is None:
            return InProcessTransport(action=self, state_management=state_management)
        return InProcessTransport(
            action=self,
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
            store=self.store,
            snapshot_id=snapshot_id,
            session_id=session_id,
            client_transform=self._client_transform,
        )

    async def abort_snapshot_data(self, snapshot_id: str) -> SnapshotStatus | None:
        """Abort a running snapshot."""
        if self.store is None:
            return None
        return await abort_snapshot_in_store(store=self.store, snapshot_id=snapshot_id)


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
        session, parent = await load_session(init=init, store=store, agent_name=name, state_schema=state_schema)
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
        return await rt.run(fn=fn, client_inputs=in_queue)

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
    async def snapshot_fn(input_val: Any) -> SessionSnapshot:  # noqa: ANN401
        sid, sess_id = parse_snapshot_lookup_input(input_val)
        snap = await agent.get_snapshot_data(snapshot_id=sid, session_id=sess_id)
        if snap is None:
            # A poller asking for a snapshot that isn't there is a lookup miss, not
            # an empty-but-successful read, so surface it as NOT_FOUND instead of a
            # null the caller has to re-interpret.
            target = sid or sess_id or 'unknown'
            raise GenkitError(status='NOT_FOUND', message=f'Snapshot {target!r} not found for agent {name!r}.')
        return snap

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


def _tool_input_key(value: object) -> str:
    """Canonical JSON form of a tool input for order-insensitive deep comparison."""
    return json.dumps(value, sort_keys=True, default=str)


def validate_resume_against_history(resume: Resume, history: list[MessageData]) -> None:
    """Reject a resume that doesn't line up with the tool requests in history.

    A resumed turn answers tool requests the model actually made, so every
    ``respond``/``restart`` entry has to point at a tool request recorded in the
    session (searched across the whole history, not just the last message). A
    restart additionally has to carry the *same* inputs as the interrupted
    request — otherwise a client could resume a tool with forged arguments.
    Raises ``INVALID_ARGUMENT`` on the first mismatch.
    """
    tool_requests: list[ToolRequest] = []
    for msg in history:
        if msg.role != Role.MODEL:
            continue
        for root in part_roots(msg.content):
            tr = getattr(root, 'tool_request', None)
            if isinstance(tr, ToolRequest):
                tool_requests.append(tr)

    def find(name: str, ref: str | None) -> ToolRequest | None:
        return next((tr for tr in tool_requests if tr.name == name and tr.ref == ref), None)

    def ref_suffix(ref: str | None) -> str:
        return f' (ref: {ref})' if ref else ''

    for restart_part in resume.restart or []:
        tr = restart_part.tool_request
        match = find(tr.name, tr.ref)
        if match is None:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f"resume.restart references tool '{tr.name}'{ref_suffix(tr.ref)} "
                    'which was not found in session history.'
                ),
            )
        if _tool_input_key(tr.input) != _tool_input_key(match.input):
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f"resume.restart for tool '{tr.name}'{ref_suffix(tr.ref)} has modified inputs that do not "
                    'match the original tool request in session history. Restart inputs must exactly match the '
                    'interrupted tool request.'
                ),
            )

    for respond_part in resume.respond or []:
        resp = respond_part.tool_response
        if find(resp.name, resp.ref) is None:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f"resume.respond references tool '{resp.name}'{ref_suffix(resp.ref)} "
                    'which was not found in session history.'
                ),
            )


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
