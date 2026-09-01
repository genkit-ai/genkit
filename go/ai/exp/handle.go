// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package exp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// AgentHandle is the untyped caller-side view of an agent: one turn, plus the
// snapshot lifecycle behind a detached one (read, follow, abort), addressed
// with the custom-state type fixed to [json.RawMessage]. It is how code that
// does not hold the defining [Agent] value calls an agent it knows only by
// name: orchestrators, middleware, and tools resolve one with [LookupAgent]
// (or the genkit/exp package's LookupAgent), and a typed owner hands one out
// with [Agent.Handle].
//
// A handle adds no capability over the agent it names; it only removes the
// wire plumbing (JSON marshaling, companion-action lookup and dispatch) that
// name-resolved callers would otherwise repeat. Custom state on every surface
// is [json.RawMessage]: unmarshal [SessionState.Custom] into the agent's own
// state type when the caller knows it.
//
// Reaching the agent is an [agentTransport]'s job, so where the agent lives is
// not part of this surface. Both constructors bind the in-process transport
// today; the seam is what will let one bind an agent behind an HTTP endpoint
// without moving a method.
type AgentHandle struct {
	name string
	// meta is the agent's capability metadata, or nil when it is unknown. It
	// is static, so the handle holds it rather than asking the transport for
	// it on every call, and it is resolved on first use rather than at
	// construction: deriving it deep-copies the agent's state schema, which
	// costs more than a lookup itself, and most callers only run the agent.
	// metaSrc is the descriptor it comes from, released once resolved.
	metaOnce  sync.Once
	meta      *AgentMetadata
	metaSrc   api.Action
	transport agentTransport
}

// agentTransport is how an [AgentHandle] reaches the agent it names. The
// handle owns the ergonomic surface and the argument validation that holds
// wherever the agent lives; a transport owns marshaling, dispatch, and the
// refusal an agent that lacks a capability earns.
//
// It is unexported until a second implementation exists. The in-process
// transport below is the only one today, and an HTTP one over the
// /agents/{name}/... routes is what the seam is for; writing that is what will
// settle the shape well enough to publish it.
//
// Two rules every implementation owes its callers:
//
// Errors are matched by status name ([status.Classified]), never by sentinel
// identity. A transport that crosses a wire decodes a status name and nothing
// else, so a sentinel subtype arrives as its parent status: the in-process
// transport's live error chain is a convenience of where it runs, not a
// promise of the seam. A message that would name which subtype it is has to
// hedge instead.
//
// Capability metadata is not here. It is static, [AgentHandle] holds it
// directly, and a transport that had to fetch it could report no failure
// through a getter that returns one value. Whoever builds a transport supplies
// what it knows about the agent, and nil means unknown.
//
// Connect-style sessions are deliberately out of scope. [AgentHandle] exposes
// none, and a long-lived duplex stream is a different problem from four
// request/response calls.
type agentTransport interface {
	// Run delivers one turn and returns its final output. init carries the
	// session source and may be nil; cb receives streamed chunks as raw JSON
	// and may be nil.
	Run(ctx context.Context, input *AgentInput, init *AgentInit[json.RawMessage], cb func(context.Context, json.RawMessage) error) (*AgentOutput[json.RawMessage], error)
	// GetSnapshot reads one snapshot, addressed either by its own ID or as a
	// session's latest.
	GetSnapshot(ctx context.Context, lookup *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error)
	// WaitForSnapshot blocks until the snapshot settles and returns it.
	WaitForSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[json.RawMessage], error)
	// Abort stops the background work behind a pending snapshot and reports
	// the snapshot's status after the attempt.
	Abort(ctx context.Context, snapshotID string) (SnapshotStatus, error)
}

// LookupAgent resolves the agent registered under name and returns its
// [AgentHandle], or nil when name resolves to no agent. It is the caller-side
// counterpart of [DefineAgent] and friends: definition hands back the typed
// [Agent], and a later caller that holds only the name recovers this untyped
// view, companion actions included. Callers holding a genkit instance should
// use the genkit/exp package's LookupAgent instead.
//
// Nil covers both misses, matching every other Lookup in the framework: no
// action under the agent key, and an action registered there that is not an
// agent. Neither is worth distinguishing to a caller, because the answer to
// both is the same and the caller knows the name it asked for. Guard the
// result, or call [AgentHandle.Run] and let its nil check report it.
func LookupAgent(r api.Registry, name string) *AgentHandle {
	run, ok := r.LookupAction(api.KeyFromName(api.ActionTypeAgent, name)).(api.BidiAction)
	if !ok {
		return nil
	}
	return &AgentHandle{
		name:    name,
		metaSrc: run,
		transport: &actionTransport{
			name:        name,
			run:         run,
			getSnapshot: r.LookupAction(api.KeyFromName(api.ActionTypeAgentSnapshot, name)),
			wait:        r.LookupAction(api.KeyFromName(api.ActionTypeAgentWait, name)),
			abort:       r.LookupAction(api.KeyFromName(api.ActionTypeAgentAbort, name)),
		},
	}
}

// Handle returns the agent's untyped caller-side view: the same surface
// [LookupAgent] builds for callers that hold only the agent's name. Use it to
// hand the agent to code written against [AgentHandle] (orchestration helpers,
// middleware) without exposing the typed [Agent]. It performs no registry
// lookup, so it also works for an unregistered agent (see [NewCustomAgent]).
func (a *Agent[State]) Handle() *AgentHandle {
	return &AgentHandle{
		name:    a.Name(),
		metaSrc: a,
		transport: &actionTransport{
			name:        a.Name(),
			run:         a,
			getSnapshot: a.getSnapshot,
			wait:        a.wait,
			abort:       a.abort,
		},
	}
}

// agentMetadataOf extracts the [AgentMetadata] an agent's action descriptor
// carries under its "agent" metadata key, or nil when a is not an agent action
// or carries none. It handles both in-process descriptors (typed metadata) and
// descriptors decoded from JSON (map form), so [AgentHandle.Metadata] reads an
// agent's capabilities without knowing where the action came from.
//
// The returned value is a copy: mutating its fields, [AgentMetadata.StateSchema]
// and anything nested inside it included, does not affect the descriptor.
//
// A wire descriptor that does not decode reports as nil rather than partially.
// Every field here is a capability a caller gates on, and a zero value reads as
// a definite "no": an agent whose abortable field arrived mistyped would be
// refused background work it can do, and told why in terms that are not true.
// Absent metadata is the honest answer, and callers already treat it as
// "unknown" and fall back to asking the runtime.
func agentMetadataOf(a api.Action) *AgentMetadata {
	if a == nil {
		return nil
	}
	switch m := a.Desc().Metadata["agent"].(type) {
	case AgentMetadata:
		m.StateSchema = base.CloneSchema(m.StateSchema)
		return &m
	case *AgentMetadata:
		if m == nil {
			return nil
		}
		copied := *m
		copied.StateSchema = base.CloneSchema(copied.StateSchema)
		return &copied
	case map[string]any:
		meta, err := base.MapToStruct[AgentMetadata](m)
		if err != nil {
			return nil
		}
		return &meta
	}
	return nil
}

// Name returns the agent's registered name.
func (h *AgentHandle) Name() string { return h.name }

// Metadata returns the agent's capability metadata (who manages state, whether
// background work can be aborted), or nil when the agent's descriptor carries
// none or did not decode. The handle holds one copy, detached from the
// descriptor but shared across calls, so treat it as read-only.
func (h *AgentHandle) Metadata() *AgentMetadata {
	if h == nil {
		return nil
	}
	h.metaOnce.Do(func() {
		// Derive only when there is a descriptor to derive from. A handle
		// constructed with eager metadata and no metaSrc (the shape a remote
		// transport takes, having no api.Action to inspect, like JS
		// remoteAgent filling stateManagement) keeps what it was built with;
		// running the derivation over a nil source would clobber it to nil.
		if h.metaSrc != nil {
			h.meta, h.metaSrc = agentMetadataOf(h.metaSrc), nil
		}
	})
	return h.meta
}

// Run starts a single-turn invocation with the given input and returns the
// final output, with custom state as raw JSON. It is [Agent.Run] for callers
// holding only the handle; the [InvocationOption] values instantiate at
// [json.RawMessage] (e.g. WithSessionID[json.RawMessage](id)).
//
// In-band failures (e.g. a failed turn) resolve as a failed [AgentOutput]
// rather than an error, exactly like [Agent.Run]; a non-nil error means the
// invocation never started (a rejected init payload) or could not run to a
// result.
func (h *AgentHandle) Run(ctx context.Context, input *AgentInput, opts ...InvocationOption[json.RawMessage]) (*AgentOutput[json.RawMessage], error) {
	if h == nil {
		return nil, nilHandleError("Run")
	}
	if input == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: input must not be nil", h.name)
	}
	init, err := resolveInvocationInit(h.name, opts)
	if err != nil {
		return nil, err
	}
	// No stream callback: a handle reports a turn by its final output. The
	// transport takes one because a turn is streamed at that level whatever
	// carries it, so a streaming surface on the handle needs no new seam.
	return h.transport.Run(ctx, input, init, nil)
}

// RunText is [AgentHandle.Run] with a user text message as the input, the way
// [Agent.RunText] is for [Agent.Run].
func (h *AgentHandle) RunText(ctx context.Context, text string, opts ...InvocationOption[json.RawMessage]) (*AgentOutput[json.RawMessage], error) {
	if h == nil {
		return nil, nilHandleError("RunText")
	}
	return h.Run(ctx, &AgentInput{Message: ai.NewUserTextMessage(text)}, opts...)
}

// RunDetached launches input as a detached (background) invocation and returns
// the task tracking it. It is [Agent.RunDetached] for callers holding only the
// handle, with the task's snapshots read as raw JSON: the input is delivered
// with [AgentInput.Detach] set whatever the caller set it to, a launch the
// agent cannot support (no session store, or one without [SnapshotSubscriber];
// check [AgentMetadata.Abortable] to pre-flight) fails with
// FAILED_PRECONDITION, and an invocation the agent settled before observing
// the detach yields a task over its committed snapshot.
//
// The rejection is decoded from the invocation's failed output, which keeps
// only the status name, so match it with [status.Of] rather than errors.Is.
func (h *AgentHandle) RunDetached(ctx context.Context, input *AgentInput, opts ...InvocationOption[json.RawMessage]) (*DetachedTask[json.RawMessage], error) {
	if h == nil {
		return nil, nilHandleError("RunDetached")
	}
	if input == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: input must not be nil", h.name)
	}
	out, err := h.Run(ctx, detachedInput(input), opts...)
	if err != nil {
		return nil, err
	}
	return detachedTaskFrom(h.name, h, out)
}

// Task returns the task tracking a snapshot ID recorded earlier (e.g.
// by a prior process that called [AgentHandle.RunDetached]), with custom state
// as raw JSON. It performs no I/O and does not verify the snapshot exists; the
// first [DetachedTask.Poll] or [DetachedTask.Wait] surfaces NOT_FOUND for an
// unknown ID.
func (h *AgentHandle) Task(snapshotID string) *DetachedTask[json.RawMessage] {
	return &DetachedTask[json.RawMessage]{ops: h, snapshotID: snapshotID}
}

// nilHandleError is what a method reports on a nil receiver, so a caller that
// skipped the nil check after [LookupAgent] reads the cause rather than a
// panic. method names the caller.
func nilHandleError(method string) error {
	return status.Errorf(status.ErrInvalidArgument,
		"AgentHandle.%s: called on a nil handle; check that LookupAgent found the agent", method)
}

// GetSnapshot fetches a session snapshot by ID through the agent's getSnapshot
// companion action, so the read is shaped exactly as remote callers see it:
// the configured [WithStateTransform] applies and a stale-heartbeat pending
// row surfaces as [SnapshotStatusExpired]. It is [Agent.GetSnapshot] with
// custom state as raw JSON. Pass [WithMetadataOnly] to read the shaped
// metadata without loading the state.
//
// It returns FAILED_PRECONDITION ([ErrSessionStoreNotConfigured]) when the
// agent has no session store and INVALID_ARGUMENT when snapshotID is empty; a
// missing snapshot is NOT_FOUND.
func (h *AgentHandle) GetSnapshot(ctx context.Context, snapshotID string, opts ...SnapshotReadOption) (*SessionSnapshot[json.RawMessage], error) {
	if snapshotID == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: GetSnapshot: snapshotID is required", h.name)
	}
	return h.transport.GetSnapshot(ctx, resolveSnapshotRead(&GetSnapshotRequest{SnapshotID: snapshotID}, opts))
}

// WaitForSnapshot fetches a session snapshot by ID and blocks until it settles,
// through the agent's waitForSnapshot companion action, returning the terminal
// snapshot shaped exactly as [AgentHandle.GetSnapshot] shapes a read. An
// already-terminal snapshot returns at once. It is [Agent.WaitForSnapshot] with
// custom state as raw JSON, and it is how a caller that holds only actions
// follows a detached invocation: one dispatch for the whole wait, rather than a
// read per tick.
//
// A snapshot that failed, aborted, or expired is returned like any other, so a
// non-nil error means the wait itself could not proceed: reads failed past the
// wait's transient-retry budget, or ctx ended and its error is returned. Bound
// the wait with [context.WithTimeout], then call [AgentHandle.GetSnapshot] to
// learn where the task stands.
//
// It returns FAILED_PRECONDITION ([ErrSessionStoreNotConfigured]) when the
// agent has no session store and INVALID_ARGUMENT when snapshotID is empty; a
// missing snapshot is NOT_FOUND.
func (h *AgentHandle) WaitForSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[json.RawMessage], error) {
	if snapshotID == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: WaitForSnapshot: snapshotID is required", h.name)
	}
	return h.transport.WaitForSnapshot(ctx, snapshotID)
}

// GetLatestSnapshot fetches a session's most recently created snapshot
// (whatever its status) through the agent's getSnapshot companion action, with
// the same shaping and the same [SnapshotReadOption] projections as
// [AgentHandle.GetSnapshot]. It is [Agent.GetLatestSnapshot] with custom state
// as raw JSON.
//
// It returns FAILED_PRECONDITION ([ErrSessionStoreNotConfigured]) when the
// agent has no session store and INVALID_ARGUMENT when sessionID is empty; an
// unknown session is NOT_FOUND.
func (h *AgentHandle) GetLatestSnapshot(ctx context.Context, sessionID string, opts ...SnapshotReadOption) (*SessionSnapshot[json.RawMessage], error) {
	if sessionID == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: GetLatestSnapshot: sessionID is required", h.name)
	}
	return h.transport.GetSnapshot(ctx, resolveSnapshotRead(&GetSnapshotRequest{SessionID: sessionID}, opts))
}

// Abort asks the background work behind a pending snapshot to stop, through
// the agent's abort companion action, and returns the snapshot's status after
// the attempt: [SnapshotStatusAborting] when the row was pending (the stop
// landed, and the row settles as [SnapshotStatusAborted] once the worker's
// finalize stamps the state on) or already aborting, or the existing terminal
// status (the abort was a no-op) when it had already settled. A missing
// snapshot is NOT_FOUND, matching the companion action remote callers use
// (unlike [Agent.Abort], which reports it as "").
//
// It returns FAILED_PRECONDITION when the agent has no session store
// ([ErrSessionStoreNotConfigured]) or the store cannot observe aborts (no
// [SnapshotSubscriber]; see [AgentMetadata.Abortable]), and INVALID_ARGUMENT
// when snapshotID is empty.
func (h *AgentHandle) Abort(ctx context.Context, snapshotID string) (SnapshotStatus, error) {
	if snapshotID == "" {
		return "", status.Errorf(status.ErrInvalidArgument, "agent %q: Abort: snapshotID is required", h.name)
	}
	return h.transport.Abort(ctx, snapshotID)
}

// --- In-process transport ---

// actionTransport reaches an agent through its registered actions, in this
// process. It is what [LookupAgent] and [Agent.Handle] build, and the only
// [agentTransport] today.
//
// A companion action is nil when the agent does not publish it, which an agent
// does only when it has a session store supporting that capability. Turning
// that into a refusal is the transport's job rather than the handle's: the
// handle cannot see an action, and a transport reaching a remote agent learns
// the same fact from a status the server sends back.
//
// Errors from here keep a live chain, so sentinel matching with errors.Is
// works, subtypes included ([ErrSnapshotNotFound] is a [status.ErrNotFound]).
// That is a property of running in-process, not of the seam. Callers written
// against [AgentHandle] match on status name, per [agentTransport].
type actionTransport struct {
	name        string
	run         api.BidiAction
	getSnapshot api.Action
	wait        api.Action
	abort       api.Action
}

func (t *actionTransport) Run(ctx context.Context, input *AgentInput, init *AgentInit[json.RawMessage], cb func(context.Context, json.RawMessage) error) (*AgentOutput[json.RawMessage], error) {
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("agent %q: marshal input: %w", t.name, err)
	}
	var initJSON json.RawMessage
	if init != nil {
		initJSON, err = json.Marshal(init)
		if err != nil {
			return nil, fmt.Errorf("agent %q: marshal init: %w", t.name, err)
		}
	}
	res, err := t.run.RunBidiJSON(ctx, inputJSON, cb, &api.BidiJSONOptions{Init: initJSON})
	if err != nil {
		return nil, err
	}
	var out AgentOutput[json.RawMessage]
	if err := json.Unmarshal(res.Result, &out); err != nil {
		return nil, fmt.Errorf("agent %q: unmarshal output: %w", t.name, err)
	}
	return &out, nil
}

func (t *actionTransport) GetSnapshot(ctx context.Context, lookup *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
	return t.snapshotVia(ctx, t.getSnapshot, "read", lookup)
}

func (t *actionTransport) WaitForSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[json.RawMessage], error) {
	return t.snapshotVia(ctx, t.wait, "follow", &GetSnapshotRequest{SnapshotID: snapshotID})
}

// snapshotVia dispatches req to act, one of the two companion actions that
// answer a [GetSnapshotRequest] with a [SessionSnapshot], and decodes the
// result. verb names what the caller wanted to do with the snapshot, for the
// message reporting an agent that keeps none.
func (t *actionTransport) snapshotVia(ctx context.Context, act api.Action, verb string, req *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
	if act == nil {
		// State what is known. A transport built by name binds each companion
		// by its own registry lookup, so a missing one means only that the
		// agent does not publish it; no session store is the usual cause, not
		// a fact this can check. The middleware relays this text to a model.
		return nil, status.Errorf(ErrSessionStoreNotConfigured,
			"agent %q publishes no action to %s a snapshot; an agent publishes one only when it has a session store",
			t.name, verb)
	}
	return callJSON[SessionSnapshot[json.RawMessage]](ctx, t.name, act, "snapshot", req)
}

func (t *actionTransport) Abort(ctx context.Context, snapshotID string) (SnapshotStatus, error) {
	if t.abort == nil {
		// Two refusals, because the caller can act on the difference: no
		// snapshot action at all means no session store, while a store that
		// reads but cannot abort is one without SnapshotSubscriber.
		if t.getSnapshot == nil {
			return "", status.Errorf(ErrSessionStoreNotConfigured,
				"agent %q publishes no action to abort background work; an agent publishes one only when it has a session store that can observe aborts",
				t.name)
		}
		return "", status.Errorf(status.ErrFailedPrecondition,
			"agent %q: the session store does not support abort (it does not implement SnapshotSubscriber)", t.name)
	}
	resp, err := callJSON[AgentAbortResponse](ctx, t.name, t.abort, "abort", &AgentAbortRequest{SnapshotID: snapshotID})
	if err != nil {
		return "", err
	}
	return resp.Status, nil
}

// callJSON dispatches req to act, one of the agent's companion actions, and
// decodes the result into Resp. what names the exchange in the marshal and
// unmarshal errors; a dispatch error is returned as is, so its status stays
// matchable.
func callJSON[Resp any](ctx context.Context, agentName string, act api.Action, what string, req any) (*Resp, error) {
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("agent %q: marshal %s request: %w", agentName, what, err)
	}
	raw, err := act.RunJSON(ctx, reqJSON, nil)
	if err != nil {
		return nil, err
	}
	var resp Resp
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("agent %q: unmarshal %s response: %w", agentName, what, err)
	}
	return &resp, nil
}
