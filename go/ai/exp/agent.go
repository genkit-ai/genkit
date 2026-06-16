// Copyright 2025 Google LLC
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

// Package exp provides experimental AI primitives for Genkit.
//
// APIs in this package are under active development and may change in any
// minor version release.
package exp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"runtime/debug"
	"sync"
	"sync/atomic"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/google/uuid"
)

// --- SessionRunner ---

// SessionRunner extends Session with agent-runtime functionality:
// turn management, snapshot persistence, and input channel handling.
type SessionRunner[State any] struct {
	*Session[State]

	// InputCh is the channel that delivers per-turn inputs from the client.
	// It is consumed automatically by [SessionRunner.Run], but is exposed
	// for advanced use cases that need direct access to the input stream
	// (e.g., custom turn loops or fan-out patterns).
	InputCh <-chan *AgentInput
	// TurnIndex is the zero-based index of the current conversation turn.
	// It is incremented automatically by [SessionRunner.Run], but is exposed
	// for advanced use cases that need to track or manipulate turn ordering
	// directly.
	TurnIndex int

	snapshotCallback  SnapshotCallback[State]
	onEndTurn         func(ctx context.Context)
	collectTurnOutput func() any

	// snapMu serializes snapshot persistence with the detach handler's
	// suspend-and-capture. lastSnapshot and lastSnapshotVersion are
	// written under it; the terminal paths that read them without it
	// (handleFnDone, failedOutput) run after fn completes, with a
	// happens-before edge through the fnDone channel.
	snapMu              sync.Mutex
	snapshotsSuspended  bool
	lastSnapshot        *SessionSnapshot[State]
	lastSnapshotVersion uint64

	// lastTurnFinishReason is the finish reason reported by the most recent
	// turn (via the [TurnResult] its callback returned), or "" if the turn
	// reported none. It is written by endTurn before [SessionRunner.onEndTurn]
	// and read by the runtime when emitting the turn-end signal and when
	// defaulting the invocation's finish reason. All accesses are confined
	// to the fn goroutine (Run and its synchronous onEndTurn callback) until
	// fn completes, after which the terminal paths read it with a
	// happens-before edge through the fnDone channel, so no lock is needed.
	// The same confinement applies to lastTurnFailed and the lastGood*
	// fields below; the terminal paths that read them (handleFnDone and
	// the detach-failure paths) all wait on fnDone first.
	lastTurnFinishReason AgentFinishReason

	// lastTurnFailed reports whether the most recent turn ended in error.
	// Set by endTurn each turn.
	lastTurnFailed bool

	// lastGoodState is a deep copy of the session state as of the most
	// recent successful turn (or the initial state when no turn has
	// completed yet), captured regardless of whether the snapshot callback
	// persisted that turn. lastGoodVersion is the session version at that
	// capture and lastGoodFinishReason that turn's reported reason. The
	// failure path returns (client-managed) or persists (server-managed
	// recovery snapshot) this state.
	lastGoodState        *SessionState[State]
	lastGoodVersion      uint64
	lastGoodFinishReason AgentFinishReason
}

// parentSnapshotID returns the ID of the most recent snapshot in this
// invocation (used to chain new snapshots via ParentID), or "" if no
// snapshot has been written yet.
func (s *SessionRunner[State]) parentSnapshotID() string {
	if s.lastSnapshot == nil {
		return ""
	}
	return s.lastSnapshot.SnapshotID
}

// suspendSnapshots stops all further snapshot persistence for this
// invocation and returns the ID of the newest persisted snapshot. Taking
// snapMu makes the two steps atomic with respect to an in-flight turn-end
// write: a write already inside maybeSnapshot completes first (so the
// returned parent is current, not stale), and any later turn end observes
// the suspension and skips its write. Called by the detach handler, after
// which the queued inputs roll into a single finalize rewrite of the
// pending row.
func (s *SessionRunner[State]) suspendSnapshots() (parentID string) {
	s.snapMu.Lock()
	defer s.snapMu.Unlock()
	s.snapshotsSuspended = true
	return s.parentSnapshotID()
}

// TurnResult is the optional return value of a [SessionRunner.Run] per-turn
// callback. It lets a custom agent report how the turn ended; the framework
// forwards the reason on the turn's [TurnEnd] chunk, persists it on the
// turn-end snapshot, and uses it to default the invocation's finish reason.
//
// Returning nil (or a zero TurnResult) omits the reason: the framework
// performs no implicit inference. A prompt-backed agent populates it
// automatically from the underlying generate response.
type TurnResult struct {
	// FinishReason is why this turn ended (e.g. [AgentFinishReasonStop],
	// [AgentFinishReasonInterrupted]). Empty to report no reason.
	FinishReason AgentFinishReason
}

// Run loops over the input channel, calling fn for each turn. Each turn is
// wrapped in a trace span for observability. Input messages are automatically
// added to the session before fn is called. After fn returns successfully, a
// snapshot check is triggered and a [TurnEnd] chunk (carrying any new
// snapshot's ID) is sent.
//
// fn may return a [TurnResult] to report how the turn ended (e.g. its finish
// reason); returning nil reports nothing. The reason rides the turn's
// [TurnEnd] chunk and is persisted on the turn-end snapshot.
//
// When fn returns an error, Run records the failure ([TurnEnd] is emitted
// with [AgentFinishReasonFailed] and no snapshot is taken of the turn's
// partial state), stops looping, and returns the error. A custom agent may
// recover (e.g. call Run again to keep processing inputs) or propagate the
// error out of the agent function, which resolves the invocation with a
// failed [AgentOutput] carrying the error and the last-good state rather
// than failing the action.
func (s *SessionRunner[State]) Run(ctx context.Context, fn func(ctx context.Context, input *AgentInput) (*TurnResult, error)) error {
	for input := range s.InputCh {
		// Deep-copy at the framework boundary: an in-process caller
		// retains the pointers it sent (message, resume parts) and may
		// mutate them after Send returns, so everything past this point
		// (trace marshaling, session state, snapshot writes) must work
		// on private memory rather than race the caller.
		input = jsonClone(input)
		spanMeta := &tracing.SpanMetadata{
			Name:    fmt.Sprintf("agent/turn/%d", s.TurnIndex),
			Type:    "flowStep",
			Subtype: "flowStep",
		}
		_, err := tracing.RunInNewSpan(ctx, spanMeta, input,
			func(ctx context.Context, input *AgentInput) (any, error) {
				if input.Message != nil {
					// The session owns its history: store a copy so fn's
					// view of the input stays independent of session state.
					s.AddMessages(jsonClone(input.Message))
				}
				tr, err := fn(ctx, input)
				if err != nil {
					return nil, err
				}
				// A returned TurnResult sets the reason, nil reports none.
				var reason AgentFinishReason
				if tr != nil {
					reason = tr.FinishReason
				}
				s.endTurn(ctx, reason, false)
				if s.collectTurnOutput != nil {
					return s.collectTurnOutput(), nil
				}
				return nil, nil
			},
		)
		if err != nil {
			s.endTurn(ctx, AgentFinishReasonFailed, true)
			return err
		}
	}
	return nil
}

// endTurn records how the turn ended and runs the shared turn-end tail:
// the turn-end emit, the last-good capture on success, and the turn
// advance.
func (s *SessionRunner[State]) endTurn(ctx context.Context, reason AgentFinishReason, failed bool) {
	s.lastTurnFinishReason = reason
	s.lastTurnFailed = failed
	s.onEndTurn(ctx)
	if !failed {
		s.recordLastGood()
	}
	s.TurnIndex++
}

// recordLastGood captures the current session state as the last-good
// recovery point. Called once at session start and after every successful
// turn, whether or not the snapshot callback persisted that turn. Runs
// after the turn-end snapshot check so that when the newest snapshot
// already captures this exact version, the deep copy is skipped;
// recoverySnapshotID then resolves to that snapshot's ID without reading
// lastGoodState.
func (s *SessionRunner[State]) recordLastGood() {
	s.mu.RLock()
	version := s.version
	persisted := s.lastSnapshot != nil && version == s.lastSnapshotVersion
	if !persisted {
		state := s.copyStateLocked()
		s.lastGoodState = &state
	}
	s.mu.RUnlock()
	s.lastGoodVersion = version
	s.lastGoodFinishReason = s.lastTurnFinishReason
}

// Result returns an [AgentResult] populated from the current session state:
// the last message in the conversation history and all artifacts. The
// returned value is independent of the session; callers may mutate it
// without affecting session state. An artifact sent through the
// [Responder] is visible here as soon as the Send call returns.
//
// It is a convenience for custom agents that don't need to construct the
// result manually.
func (s *SessionRunner[State]) Result() *AgentResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := &AgentResult{}
	if msgs := s.state.Messages; len(msgs) > 0 {
		result.Message = jsonClone(msgs[len(msgs)-1])
	}
	if len(s.state.Artifacts) > 0 {
		result.Artifacts = cloneArtifacts(s.state.Artifacts)
	}
	return result
}

// invocationReason resolves the finish reason for the whole invocation: the
// last turn's reason, unless the agent's result overrides it with a non-empty
// one. Shared by the synchronous-completion and detach-finalize paths.
func (s *SessionRunner[State]) invocationReason(result *AgentResult) AgentFinishReason {
	if result != nil && result.FinishReason != "" {
		return result.FinishReason
	}
	return s.lastTurnFinishReason
}

// maybeSnapshot creates a snapshot if conditions are met (store configured,
// snapshots not suspended by detach, callback approves, state changed).
// Returns the snapshot ID or empty string. finishReason is recorded on the
// snapshot so a resumed or background task can report how the captured turn
// or invocation ended.
//
// The body runs under snapMu so the detach handler's suspend-and-capture
// (suspendSnapshots) cannot interleave with a write: it either waits for
// this write to commit or suspends before it starts.
func (s *SessionRunner[State]) maybeSnapshot(ctx context.Context, event SnapshotEvent, finishReason AgentFinishReason) string {
	if s.store == nil {
		return ""
	}

	s.snapMu.Lock()
	defer s.snapMu.Unlock()
	if s.snapshotsSuspended {
		return ""
	}

	s.mu.RLock()
	currentVersion := s.version
	currentState := s.copyStateLocked()
	s.mu.RUnlock()

	// Skip only if this snapshot would be identical to the last one: same
	// state AND same finish reason. This dedups the common invocation-end
	// snapshot after a single-turn Run (the turn-end snapshot already
	// captured the same state and reason), but still writes when the
	// invocation reports a different reason than the last turn (e.g. a
	// custom agent overrode it on its AgentResult) — that snapshot is not
	// redundant, it carries a new reason.
	if s.lastSnapshot != nil &&
		currentVersion == s.lastSnapshotVersion &&
		finishReason == s.lastSnapshot.FinishReason {
		return ""
	}

	if s.snapshotCallback != nil {
		var prevState *SessionState[State]
		if s.lastSnapshot != nil {
			prevState = s.lastSnapshot.State
		}
		if !s.snapshotCallback(ctx, &SnapshotContext[State]{
			State:     &currentState,
			PrevState: prevState,
			TurnIndex: s.TurnIndex,
			Event:     event,
		}) {
			return ""
		}
	}

	return s.persistSnapshotLocked(ctx, event, finishReason, &currentState, currentVersion)
}

// persistSnapshotLocked writes a succeeded snapshot row capturing state (at
// the given session version), chained to the newest persisted snapshot, and
// advances the lastSnapshot bookkeeping. Both the routine cadence
// (maybeSnapshot) and the failure path (recoverySnapshotID) funnel through
// here so the row shape and bookkeeping live in one place. Caller must hold
// snapMu. Persistence is best-effort: a store failure must not kill the
// in-flight turn, so it is logged and "" is returned.
func (s *SessionRunner[State]) persistSnapshotLocked(ctx context.Context, event SnapshotEvent, finishReason AgentFinishReason, state *SessionState[State], version uint64) string {
	parentID := s.parentSnapshotID()
	sessionID := s.SessionID()

	saved, err := s.store.SaveSnapshot(ctx, "",
		func(_ *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			return &SessionSnapshot[State]{
				SessionID:    sessionID,
				ParentID:     parentID,
				Event:        event,
				Status:       SnapshotStatusSucceeded,
				FinishReason: finishReason,
				State:        state,
			}, nil
		})
	if err != nil {
		logger.FromContext(ctx).Error("agent: failed to save snapshot",
			"parentId", parentID,
			"event", event,
			"err", err)
		return ""
	}

	s.lastSnapshot = saved
	s.lastSnapshotVersion = version
	return saved.SnapshotID
}

// recoverySnapshotID returns the ID of a snapshot holding the last-good
// state, writing one (event [SnapshotEventRecovery]) when the newest
// persisted snapshot is behind it. The write uses the captured
// lastGoodState, never the live state (which may hold the failed turn's
// partial mutations), and intentionally bypasses both the snapshot
// callback and the post-detach suspension, so neither a selective cadence
// nor a dying detach can lose the conversation. If the write fails, the
// newest persisted snapshot's ID is returned instead.
//
// Returns "" when no store is configured or there is nothing to recover
// (no snapshot exists and no turn ever changed state).
func (s *SessionRunner[State]) recoverySnapshotID(ctx context.Context) string {
	if s.store == nil {
		return ""
	}
	s.snapMu.Lock()
	defer s.snapMu.Unlock()
	// The newest snapshot already captures exactly the last-good state.
	if s.lastSnapshot != nil && s.lastGoodVersion == s.lastSnapshotVersion {
		return s.lastSnapshot.SnapshotID
	}
	if s.lastSnapshot == nil && s.lastGoodVersion == 0 {
		return ""
	}

	if id := s.persistSnapshotLocked(ctx, SnapshotEventRecovery, s.lastGoodFinishReason, s.lastGoodState, s.lastGoodVersion); id != "" {
		return id
	}
	return s.parentSnapshotID()
}

// --- Responder ---

// Responder is the output channel for an agent. Artifacts sent through
// it are added to the session synchronously: by the time a Send method
// returns, the chunk's session-level side effects have been applied, so
// a state read ([SessionRunner.Result], [Session.Artifacts]) or a
// turn-end snapshot that follows the call observes them. Only the wire
// forward to the client is asynchronous.
//
// All Send methods are ctx-aware: if the agent's work context is
// cancelled (typically client disconnect, abort during detach, or fn
// completion), Send returns promptly with the chunk dropped from the
// wire; the session-level side effects still apply. Send itself remains
// fire-and-forget and returns no error; the user fn is expected to
// observe cancellation through its own ctx check and stop producing.
type Responder[Stream any] struct {
	in  chan<- *AgentStreamChunk[Stream]
	ctx context.Context
	// effects applies the chunk's in-process side effects (session
	// artifact add, turn-chunk accumulation) synchronously in send, in
	// the sender's goroutine, so reads and snapshots that follow a Send
	// cannot miss the chunk.
	effects func(*AgentStreamChunk[Stream])
}

// SendModelChunk sends a generation chunk (token-level streaming).
func (r Responder[Stream]) SendModelChunk(chunk *ai.ModelResponseChunk) {
	r.send(&AgentStreamChunk[Stream]{ModelChunk: chunk})
}

// SendStatus sends a user-defined status update.
func (r Responder[Stream]) SendStatus(status Stream) {
	r.send(&AgentStreamChunk[Stream]{Status: status})
}

// SendArtifact sends an artifact to the stream and adds it to the session.
// If an artifact with the same name already exists in the session, it is
// replaced. The artifact is in the session by the time SendArtifact
// returns, and the session stores a deep copy captured at the call, so
// later mutations of the caller's artifact do not affect session state.
// The session-level side effect happens whether or not detach has landed;
// only the wire forward to the client is suppressed post-detach, when
// there is no longer a client to receive it.
func (r Responder[Stream]) SendArtifact(artifact *Artifact) {
	r.send(&AgentStreamChunk[Stream]{Artifact: artifact})
}

// send applies chunk's in-process side effects, then delivers it to the
// router for the wire forward, returning promptly if r.ctx is cancelled.
// Applying side effects synchronously (in the sender's goroutine, before
// the channel send) orders them before everything the caller does after
// Send, so a state read or a turn-end snapshot cannot miss a chunk whose
// Send already returned. Dropping the wire forward on cancel decouples
// fn liveness from the runtime's shutdown choreography: a Send issued
// after workCtx cancellation completes immediately rather than blocking
// on a router that has not yet been put into drain mode by a terminal
// path.
func (r Responder[Stream]) send(chunk *AgentStreamChunk[Stream]) {
	if r.effects != nil {
		r.effects(chunk)
	}
	select {
	case r.in <- chunk:
	case <-r.ctx.Done():
	}
}

// --- Agent ---

// AgentFunc is the function signature for custom agents.
// Type parameters:
//   - Stream: Type for status updates sent via the responder
//   - State: Type for user-defined state in snapshots
type AgentFunc[Stream, State any] = func(ctx context.Context, resp Responder[Stream], sess *SessionRunner[State]) (*AgentResult, error)

// Agent is a bidirectional streaming agent with automatic snapshot management.
//
// Agent implements [api.BidiAction], so generic transports accept it
// directly (e.g. pass it to genkit.Handler to serve it over HTTP, one turn
// per request). The [Agent.Run], [Agent.RunText], and [Agent.StreamBidi]
// methods are typed conveniences over the same underlying action; both
// surfaces run the identical per-invocation runtime.
//
// Server-managed agents (those with a [SessionStore] configured) also
// register companion actions for the snapshot lifecycle, available via
// [Agent.GetSnapshotAction] and [Agent.AbortSnapshotAction] for serving
// alongside the agent, and expose the store itself via [Agent.Store].
type Agent[Stream, State any] struct {
	action *core.BidiAction[*AgentInput, *AgentOutput[State], *AgentStreamChunk[Stream], *AgentInit[State]]
	// Companion actions, retained so transports can serve them without a
	// registry lookup. Nil when the corresponding capability is absent;
	// see newSnapshotActions.
	getSnapshot   api.Action
	abortSnapshot api.Action
	// store is the configured session store, or nil for a client-managed
	// agent. Retained so callers can reach it via Store without threading
	// a separate reference.
	store SessionStore[State]
}

// Name returns the agent's registered name. This is also the name under
// which any inline-defined prompt and companion actions (getSnapshot,
// abortSnapshot) are registered.
func (a *Agent[Stream, State]) Name() string {
	return a.action.Name()
}

// GetSnapshotAction returns the agent's getSnapshot companion action,
// which fetches a session snapshot by ID (input [GetSnapshotRequest],
// output [SessionSnapshot]). It returns nil when the agent is
// client-managed (no [SessionStore] configured): there is no server-side
// snapshot to fetch.
//
// Use it to expose snapshot polling over a transport (e.g. mount it with
// genkit.Handler next to the agent itself); local Go code should read
// from the store directly.
func (a *Agent[Stream, State]) GetSnapshotAction() api.Action {
	return a.getSnapshot
}

// AbortSnapshotAction returns the agent's abortSnapshot companion action,
// which asks the background work behind a pending snapshot (e.g. a
// detached invocation) to stop (input [AbortSnapshotRequest], output
// [AbortSnapshotResponse]). It returns nil when the agent has no
// [SessionStore] or the store does not implement [SnapshotAborter].
//
// Use it to expose aborting over a transport (e.g. mount it with
// genkit.Handler next to the agent itself); local Go code should call the
// store's [SnapshotAborter.AbortSnapshot] directly.
func (a *Agent[Stream, State]) AbortSnapshotAction() api.Action {
	return a.abortSnapshot
}

// Store returns the [SessionStore] the agent was configured with via
// [WithSessionStore], or nil when the agent is client-managed (no store).
// It lets local Go code read and write snapshots directly given an agent
// reference, without threading a separate store variable.
//
// The store is returned as the [SessionStore] interface, not its concrete
// type; a caller needing a store-specific capability (e.g.
// [SnapshotAborter]) type-asserts for it.
func (a *Agent[Stream, State]) Store() SessionStore[State] {
	return a.store
}

// --- api.BidiAction implementation ---

// Agent is itself an [api.BidiAction]: transports that accept an
// [api.Action] (or [api.BidiAction]) take an Agent directly. The bidi
// methods matter beyond mere interface completeness: generic transports
// type-assert to [api.BidiAction] to route session init (the wire
// counterpart of [WithSessionID], [WithSnapshotID], and [WithState]), so
// satisfying only [api.Action] would silently break session resume.
var _ api.BidiAction = (*Agent[any, any])(nil)

// Register registers the agent's run action and any companion actions
// (getSnapshot, abortSnapshot) with the registry. Agents defined via
// [DefineAgent] or [DefineCustomAgent] are already registered; this
// exists so an agent can travel to another registry as a unit. An
// inline-defined prompt does not travel: the agent holds it directly, so
// execution is unaffected, but the prompt action stays in the registry it
// was defined in.
func (a *Agent[Stream, State]) Register(r api.Registry) {
	// Register the wrapped bidi action under the agent key, the same way
	// every other action registers itself; the registry holds a uniform
	// api.BidiAction that the reflection servers, ListAgents, and the route
	// builders consume without knowing about the Agent type.
	//
	// The companion actions register independently under their own keys, so
	// registry consumers recover them by key (genkit.LookupAction) rather
	// than by reaching through the agent action; see newSnapshotActions.
	a.action.Register(r)
	if a.getSnapshot != nil {
		a.getSnapshot.Register(r)
	}
	if a.abortSnapshot != nil {
		a.abortSnapshot.Register(r)
	}
}

// Desc returns the descriptor of the agent's run action.
func (a *Agent[Stream, State]) Desc() api.ActionDesc {
	return a.action.Desc()
}

// RunJSON runs a one-shot invocation with no init (a fresh session):
// input is the turn's [AgentInput] and the result is the final
// [AgentOutput]. To supply a session source, use [Agent.RunBidiJSON].
func (a *Agent[Stream, State]) RunJSON(ctx context.Context, input json.RawMessage, cb func(context.Context, json.RawMessage) error) (json.RawMessage, error) {
	return a.action.RunJSON(ctx, input, cb)
}

// RunJSONWithTelemetry is [Agent.RunJSON] with trace information on the
// result.
func (a *Agent[Stream, State]) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb func(context.Context, json.RawMessage) error) (*api.ActionRunResult[json.RawMessage], error) {
	return a.action.RunJSONWithTelemetry(ctx, input, cb)
}

// RunBidiJSON runs a one-shot invocation whose session init (the wire
// counterpart of the [InvocationOption] values) rides in opts: input is
// delivered as the only chunk on the input stream and outgoing chunks are
// forwarded to cb.
func (a *Agent[Stream, State]) RunBidiJSON(ctx context.Context, input json.RawMessage, cb func(context.Context, json.RawMessage) error, opts *api.BidiSessionOptions) (*api.ActionRunResult[json.RawMessage], error) {
	return a.action.RunBidiJSON(ctx, input, cb, opts)
}

// StreamBidiJSON starts a bidirectional streaming session using
// JSON-encoded messages. Local Go callers should prefer the typed
// [Agent.StreamBidi].
func (a *Agent[Stream, State]) StreamBidiJSON(ctx context.Context, opts *api.BidiSessionOptions) (api.BidiJSONConnection, error) {
	return a.action.StreamBidiJSON(ctx, opts)
}

// DefineAgent defines a prompt-backed agent and registers it. Each turn
// renders the agent's prompt, appends conversation history, calls the
// model with streaming, and updates session state.
//
// source selects how the prompt is backed:
//
//   - [FromInline] defines the prompt inline from a set of
//     [ai.PromptOption] values; the prompt is registered under name.
//   - [FromPrompt] references an existing prompt registered with the
//     registry under name (e.g. one defined via [ai.DefinePrompt] or
//     loaded from a .prompt file).
//
// State is inferred from the typed agent options (e.g.
// [WithSessionStore], [WithSnapshotOn]); pass an explicit [State] only
// when no typed option is provided. A typed option that disagrees with
// the inferred State fails at compile time.
//
// For full control over the per-turn loop, use [DefineCustomAgent].
func DefineAgent[State any](
	r api.Registry,
	name string,
	source AgentSource,
	opts ...AgentOption[State],
) *Agent[any, State] {
	switch s := source.(type) {
	case inlineSource:
		prompt := ai.DefinePrompt(r, name, s.opts...)
		return DefineCustomAgent(r, name, agentLoop[State](r, prompt, nil), opts...)
	case promptSource:
		prompt := ai.LookupPrompt(r, name)
		if prompt == nil {
			panic(fmt.Sprintf("DefineAgent %q: prompt %q not found", name, name))
		}
		if _, err := prompt.Render(context.Background(), s.defaultInput); err != nil {
			panic(fmt.Sprintf("DefineAgent %q: defaultInput does not satisfy prompt schema: %v", name, err))
		}
		return DefineCustomAgent(r, name, agentLoop[State](r, prompt, s.defaultInput), opts...)
	default:
		panic(fmt.Sprintf("DefineAgent %q: unknown source type %T", name, source))
	}
}

// NewCustomAgent creates an agent with full control over the conversation
// loop without registering it. Register it later with the registry (e.g.
// genkit.RegisterAction), which also registers its companion actions; see
// [Agent.Register]. fn receives a [Responder] for streaming output and a
// [SessionRunner] for turn and state management; call [SessionRunner.Run]
// to enter the per-turn loop.
//
// This is the agent counterpart of [core.NewStreamingAction]: use it when
// the agent must outlive or precede a registry (e.g. built in a library,
// registered conditionally, or moved between registries). For the common
// case, [DefineCustomAgent] creates and registers in one step.
//
// There is no NewAgent counterpart for prompt-backed agents: a prompt is
// bound to the registry it renders and generates against, so a
// prompt-backed agent cannot be built before it has one. To get
// prompt-like behavior without registration, write a custom agent that
// renders and generates with your own [genkit.Genkit] inside fn.
func NewCustomAgent[Stream, State any](
	name string,
	fn AgentFunc[Stream, State],
	opts ...AgentOption[State],
) *Agent[Stream, State] {
	cfg := &agentOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyAgent(cfg); err != nil {
			panic(fmt.Errorf("NewCustomAgent %q: %w", name, err))
		}
	}

	// Typed under ActionTypeAgent so agents surface as their own action
	// kind rather than as flows (genkit.ListAgents vs ListFlows). Built on
	// NewBidiAction so the agent capability metadata is set at construction
	// time; actions must be immutable once registered. WithFlowContext
	// below preserves the flow-context wrapping that makes core.Run work
	// inside fn.
	//
	// metadata["agent"] carries the capability info for the Dev UI;
	// metadata["description"], if set, is lifted to the descriptor's
	// top-level Description by core (see core.newAction), the standard
	// place reflective tooling reads an action's description.
	metadata := map[string]any{"agent": agentMetadataFor(cfg.store)}
	if cfg.description != "" {
		metadata["description"] = cfg.description
	}
	action := core.NewBidiAction(name, api.ActionTypeAgent,
		&core.BidiActionOptions{Metadata: metadata},
		func(
			ctx context.Context,
			in *AgentInit[State],
			inCh <-chan *AgentInput,
			outCh chan<- *AgentStreamChunk[Stream],
		) (*AgentOutput[State], error) {
			ctx = core.WithFlowContext(ctx, name)
			rt, err := newAgentRuntime(ctx, name, cfg, in, inCh, outCh)
			if err != nil {
				// Init failures (a rejected init payload, a failed
				// snapshot load) fail the action outright rather than
				// resolving as a failed output: the invocation never
				// reached the input phase of its lifecycle, so there is
				// no conversation state to hand back and nothing to
				// snapshot.
				return nil, err
			}
			return rt.run(ctx, fn)
		})

	getSnapshot, abortSnapshot := newSnapshotActions(name, cfg.store, cfg.transform)

	return &Agent[Stream, State]{
		action:        action,
		getSnapshot:   getSnapshot,
		abortSnapshot: abortSnapshot,
		store:         cfg.store,
	}
}

// DefineCustomAgent defines an agent with full control over the
// conversation loop and registers it (and any companion actions) with the
// registry. fn receives a [Responder] for streaming output and a
// [SessionRunner] for turn and state management; call [SessionRunner.Run]
// to enter the per-turn loop.
//
// It is [NewCustomAgent] followed by [Agent.Register]. To build an agent
// without registering it, use [NewCustomAgent] directly. For agents backed
// by a prompt, use [DefineAgent] instead.
func DefineCustomAgent[Stream, State any](
	r api.Registry,
	name string,
	fn AgentFunc[Stream, State],
	opts ...AgentOption[State],
) *Agent[Stream, State] {
	a := NewCustomAgent(name, fn, opts...)
	a.Register(r)
	return a
}

// agentMetadataFor derives the [AgentMetadata] value attached to the
// agent's action descriptor under the "agent" key. [AgentMetadata]
// itself is generated from agent.ts; this constructor is hand-written
// because it inspects the configured store's optional capabilities.
func agentMetadataFor[State any](store SessionStore[State]) AgentMetadata {
	mgmt := AgentStateManagementClient
	abortable := false
	if store != nil {
		mgmt = AgentStateManagementServer
		_, abortable = store.(SnapshotAborter)
	}
	return AgentMetadata{
		StateManagement: mgmt,
		Abortable:       abortable,
	}
}

// --- agentRuntime ---

// agentRuntime owns the per-invocation wiring of an agent:
// session, runner, output router, input intake, and the goroutine that runs
// the user fn. Its methods implement the three terminal paths the agent can
// take: detach, fn-completion, and client-cancel.
type agentRuntime[Stream, State any] struct {
	name string
	cfg  *agentOptions[State]

	session *Session[State]
	sess    *SessionRunner[State]
	router  *chunkRouter[Stream, State]
	intake  *detachIntake

	fnDone chan fnDoneResult[State]
}

// fnDoneResult carries the user fn's return values across the goroutine
// boundary that runs it. A named type keeps the channel signatures readable.
type fnDoneResult[State any] struct {
	result *AgentResult
	err    error
}

func newAgentRuntime[Stream, State any](
	ctx context.Context,
	name string,
	cfg *agentOptions[State],
	in *AgentInit[State],
	inCh <-chan *AgentInput,
	outCh chan<- *AgentStreamChunk[Stream],
) (*agentRuntime[Stream, State], error) {
	session, parent, err := loadSession(ctx, in, cfg.store)
	if err != nil {
		return nil, err
	}

	// The session ID is settled up front, before the agent function runs,
	// so it exists for the whole invocation regardless of when (or
	// whether) the first snapshot is written. It lives inside the session
	// state ([SessionState.SessionID]), so it rides along wherever the
	// state goes: every persisted snapshot's state, and the state
	// returned to (and resent by) client-managed callers.
	if cfg.store != nil {
		// Server-managed: the store row is canonical. Inherit the resumed
		// chain's ID, overriding whatever the loaded state blob claims (a
		// third-party writer could have let them drift), or mint one for a
		// fresh conversation (including one resumed from a snapshot that
		// predates session IDs).
		if parent != nil && parent.SessionID != "" {
			session.state.SessionID = parent.SessionID
		} else {
			session.state.SessionID = uuid.New().String()
		}
	} else if session.state.SessionID == "" {
		// Client-managed: the state object is canonical; keep the ID it
		// carried. Mint one when absent (a fresh conversation) so the
		// output state is self-describing from the first turn and the
		// client can round-trip it without tracking a separate field.
		session.state.SessionID = uuid.New().String()
	}

	rt := &agentRuntime[Stream, State]{
		name:    name,
		cfg:     cfg,
		session: session,
		router:  startChunkRouter(ctx, session, outCh),
		intake:  startDetachIntake(inCh),
		fnDone:  make(chan fnDoneResult[State], 1),
	}

	rt.sess = &SessionRunner[State]{
		Session:          session,
		InputCh:          rt.intake.out(),
		snapshotCallback: cfg.callback,
		lastSnapshot:     parent,
	}
	rt.sess.collectTurnOutput = func() any { return rt.router.collectTurnChunks() }
	rt.sess.onEndTurn = rt.emitTurnEnd
	// The initial state (fresh, client-provided, or loaded from a
	// snapshot) is the last-good recovery point until a turn completes.
	rt.sess.recordLastGood()

	return rt, nil
}

// emitTurnEnd is called by the session after each turn. It paces the
// intake (releasing the forwarder for the next input), writes a turn-end
// snapshot (if applicable), and forwards the resulting [TurnEnd] chunk
// through the router so clients see it on the output stream.
//
// The snapshot sees everything the turn produced: the side effects of
// the turn's Send calls (e.g. artifacts) are applied synchronously in
// [Responder] before each Send returns, and fn returned before this
// runs, so there is no in-flight router work to wait out.
//
// The snapshot is skipped when the turn failed (the live state holds the
// turn's partial mutations) and when detach has landed (maybeSnapshot
// observes the suspension under snapMu; the pending row already captures
// the invocation and a single finalize rewrite records the cumulative
// state once the queued inputs drain).
func (rt *agentRuntime[Stream, State]) emitTurnEnd(ctx context.Context) {
	rt.intake.releaseForward()
	reason := rt.sess.lastTurnFinishReason
	var snapshotID string
	if !rt.sess.lastTurnFailed {
		snapshotID = rt.sess.maybeSnapshot(ctx, SnapshotEventTurnEnd, reason)
	}
	rt.router.sendChunk(ctx, &AgentStreamChunk[Stream]{TurnEnd: &TurnEnd{
		SnapshotID:   snapshotID,
		FinishReason: reason,
	}})
}

// run drives the user fn to completion and returns the agent output.
//
// workCtx carries the session and is decoupled from clientCtx: pre-detach a
// watcher mirrors clientCtx so a disconnect cancels the work; on detach the
// watcher exits and the finalizer goroutine owns workCtx until fn returns.
func (rt *agentRuntime[Stream, State]) run(
	clientCtx context.Context,
	fn AgentFunc[Stream, State],
) (*AgentOutput[State], error) {
	workCtx, cancelWork := context.WithCancel(context.WithoutCancel(clientCtx))
	workCtx = NewSessionContext(workCtx, rt.session)

	var detachOnce sync.Once
	detached := make(chan struct{})
	markDetached := func() { detachOnce.Do(func() { close(detached) }) }
	defer markDetached() // ensure the watcher exits on every return path

	go func() {
		select {
		case <-clientCtx.Done():
			// Arbitrate atomically against markDetached: whichever claims
			// the Once first wins. clientCtx ends not only on a true
			// disconnect but also when the framework releases the action
			// context right after run returns the detached output; by then
			// both select arms are ready and this arm may be picked, so
			// claiming the Once (rather than cancelling outright) is what
			// keeps an already-landed detach's background work alive.
			detachOnce.Do(func() {
				close(detached)
				cancelWork()
			})
		case <-detached:
		}
	}()

	go func() {
		// Run fn under deferred panic recovery so a panic surfaces as
		// an error rather than crashing the process or leaking the
		// fnDone channel.
		var (
			result *AgentResult
			fnErr  error
		)
		func() {
			defer func() {
				if r := recover(); r != nil {
					logger.FromContext(workCtx).Error("agent fn panicked", "panic", r, "stack", string(debug.Stack()))
					fnErr = core.NewError(core.INTERNAL, "agent fn panicked: %v", r)
				}
			}()
			result, fnErr = fn(workCtx, rt.router.responder(workCtx), rt.sess)
		}()
		rt.fnDone <- fnDoneResult[State]{result: result, err: fnErr}
	}()

	select {
	case <-rt.intake.detachSignal():
		if err := rt.checkDetachCapabilities(); err != nil {
			rt.drainAndWait(cancelWork)
			return rt.failedOutput(clientCtx, err), nil
		}
		return rt.handleDetach(clientCtx, workCtx, cancelWork, markDetached)

	case res := <-rt.fnDone:
		return rt.handleFnDone(clientCtx, cancelWork, res)

	case <-clientCtx.Done():
		res := rt.drainAndWait(cancelWork)
		if res.err != nil {
			return nil, res.err
		}
		return nil, clientCtx.Err()
	}
}

// checkDetachCapabilities reports whether the configured store is capable
// of supporting detach. Detach requires a writable store (to persist the
// pending snapshot) and a [SnapshotAborter] (which bundles both abort
// triggering and status-change subscription so the runtime can react to
// the abort without polling).
func (rt *agentRuntime[Stream, State]) checkDetachCapabilities() error {
	if rt.cfg.store == nil {
		return core.NewError(core.FAILED_PRECONDITION,
			"agent %q: detach requires a session store", rt.name)
	}
	if _, ok := rt.cfg.store.(SnapshotAborter); !ok {
		return core.NewError(core.FAILED_PRECONDITION,
			"agent %q: detach requires a session store implementing SnapshotAborter", rt.name)
	}
	return nil
}

// drainAndWait performs a synchronous shutdown: cancel work, stop router
// writes (so a fn mid-send doesn't deadlock once outCh's consumer is
// gone), wait for the intake reader/forwarder to finish, drain fnDone,
// and close the router. Returns the fn's result for callers that need
// to surface its error.
func (rt *agentRuntime[Stream, State]) drainAndWait(cancelWork context.CancelFunc) fnDoneResult[State] {
	cancelWork()
	// Switch the router to discard mode before waiting on fn. Without
	// this, a fn mid-SendStatus blocks on the router's r.in receive while
	// the router blocks on r.out send (consumer is gone), so fn never
	// observes ctx and we deadlock waiting on fnDone.
	rt.router.stopAndWait()
	rt.intake.stopAndWait()
	res := <-rt.fnDone
	rt.router.close()
	return res
}

// handleFnDone is the synchronous-completion path: fn returned before any
// detach signal. Capture an invocation-end snapshot if state advanced past
// the last turn-end snapshot, then assemble the output. When fn returned
// an error, the invocation resolves gracefully as a failed output instead
// (see failedOutput).
//
// router.close blocks on the forward goroutine exiting, and fn returning
// does not imply the router is idle: fn's last accepted chunk may still be
// in the router's hands, parked on the send to a full out buffer. On the
// error path stopAndWait closes stopWriting first, deliberately dropping
// the failed turn's in-flight chunks so close cannot wedge behind a
// slow/gone consumer. On the success path those chunks are wanted, so
// close relies on the parked send being released instead: a consuming
// client drains it, a disconnected client's ctx cancellation trips
// forward's ctx arm, and a client that stopped receiving unparks it when
// its Output call drains the stream.
func (rt *agentRuntime[Stream, State]) handleFnDone(
	ctx context.Context,
	cancelWork context.CancelFunc,
	res fnDoneResult[State],
) (*AgentOutput[State], error) {
	cancelWork()
	rt.intake.stopAndWait()
	if res.err != nil {
		rt.router.stopAndWait()
	}
	rt.router.close()

	if res.err != nil {
		// A disconnect-driven failure keeps its error semantics: the
		// client is gone, so there is no one to hand a graceful failed
		// output to. The clientCtx.Done arm of the run select handles the
		// common ordering; this guards the race where fn observes the
		// cancellation first and its result wins the select.
		if ctx.Err() != nil {
			return nil, res.err
		}
		return rt.failedOutput(ctx, res.err), nil
	}

	invocationReason := rt.sess.invocationReason(res.result)
	snapshotID := rt.sess.maybeSnapshot(ctx, SnapshotEventInvocationEnd, invocationReason)
	if snapshotID == "" && rt.sess.lastSnapshot != nil {
		// No new row was written; reuse the last snapshot so the response
		// always carries an ID when a store is configured. On the dedup path
		// the reused row is genuinely identical (same state and reason). If
		// the snapshot callback declined the write or the save failed, the
		// reused row is the last turn-end snapshot, whose reason (and state)
		// may lag what this output reports.
		snapshotID = rt.sess.lastSnapshot.SnapshotID
	}

	out := &AgentOutput[State]{
		SessionID:    rt.session.SessionID(),
		SnapshotID:   snapshotID,
		FinishReason: invocationReason,
	}
	if res.result != nil {
		// Deep-copy at the framework boundary so the caller cannot
		// mutate session contents through the returned output, even
		// if a custom fn constructed AgentResult with raw session
		// pointers rather than going through [SessionRunner.Result].
		out.Message = jsonClone(res.result.Message)
		out.Artifacts = cloneArtifacts(res.result.Artifacts)
	}
	if rt.cfg.store == nil {
		out.State = rt.outboundState(ctx, rt.session.State())
	}
	return out, nil
}

// outboundState applies the configured state transform and re-stamps the
// framework-owned SessionID, so the state handed to a client-managed
// caller always carries the conversation's identity even if a transform
// rewrote or dropped it. Returns nil if state is nil.
func (rt *agentRuntime[Stream, State]) outboundState(ctx context.Context, state *SessionState[State]) *SessionState[State] {
	out := applyTransform(ctx, rt.cfg.transform, state)
	if out != nil {
		out.SessionID = rt.session.SessionID()
	}
	return out
}

// failedOutput assembles the output for an invocation that ended in
// failure: [AgentFinishReasonFailed], the error with its original status,
// and the last-good state (inline when client-managed, behind a recovery
// snapshot ID when server-managed). Message and Artifacts are left empty;
// they describe the result of a completed run.
func (rt *agentRuntime[Stream, State]) failedOutput(ctx context.Context, cause error) *AgentOutput[State] {
	out := &AgentOutput[State]{
		FinishReason: AgentFinishReasonFailed,
		Error:        core.AsGenkitError(cause),
	}
	if rt.cfg.store == nil {
		out.State = rt.outboundState(ctx, rt.sess.lastGoodState)
	} else {
		out.SnapshotID = rt.sess.recoverySnapshotID(ctx)
	}
	out.SessionID = rt.session.SessionID()
	return out
}

// handleDetach commits the pending snapshot, returns its ID, and spawns the
// status-subscriber and finalizer goroutines that own the rest of the
// invocation. Per-turn snapshots are suspended for the remainder so the
// queued inputs roll into a single finalize rewrite; the chunk router
// stops writing to outCh and discards further chunks, whose in-process
// side effects (e.g. artifacts added via Responder.SendArtifact) still
// apply at Send time, so user code does not have to branch on detach.
func (rt *agentRuntime[Stream, State]) handleDetach(
	clientCtx, workCtx context.Context,
	cancelWork context.CancelFunc,
	markDetached func(),
) (*AgentOutput[State], error) {
	// Stop mirroring clientCtx. From here, only the abort subscription or
	// fn completion can cancel workCtx.
	markDetached()

	// Atomically suspend per-turn snapshots and capture the chain tip: a
	// turn-end write already in flight commits first (so the pending row
	// chains off the real tip instead of becoming its sibling), and any
	// later turn end skips its write.
	parentID := rt.sess.suspendSnapshots()
	sessionID := rt.session.SessionID()

	// Detach intends to outlive the client connection. If clientCtx was
	// already cancelled (or cancels mid-write), we still want the pending
	// row durable so observers can find it later. Decouple this write.
	pending, err := rt.cfg.store.SaveSnapshot(context.WithoutCancel(clientCtx), "",
		func(_ *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			return &SessionSnapshot[State]{
				SessionID: sessionID,
				ParentID:  parentID,
				Event:     SnapshotEventDetach,
				Status:    SnapshotStatusPending,
			}, nil
		})
	if err != nil {
		rt.drainAndWait(cancelWork)
		return rt.failedOutput(clientCtx, core.NewError(core.INTERNAL,
			"agent %q: detach: save pending snapshot: %v", rt.name, err)), nil
	}
	// The router can no longer write to outCh once we return; the bidi
	// framework closes it shortly after. The router stops writing and
	// trashes any further chunks.
	rt.router.stopAndWait()

	abortedByUser := &atomic.Bool{}
	subCtx, stopSub := context.WithCancel(workCtx)
	aborter := rt.cfg.store.(SnapshotAborter) // safe: checkDetachCapabilities ran already
	statusCh := aborter.OnSnapshotStatusChange(subCtx, pending.SnapshotID)
	go func() {
		for status := range statusCh {
			if status == SnapshotStatusAborted {
				abortedByUser.Store(true)
				cancelWork()
				return
			}
		}
	}()

	finalizeCtx := context.WithoutCancel(clientCtx)
	go func() {
		res := <-rt.fnDone
		stopSub()
		rt.intake.stopAndWait()
		rt.router.close()
		rt.finalizePendingSnapshot(finalizeCtx, pending, res.result, res.err, abortedByUser.Load())
		cancelWork()
	}()

	// The invocation, from the client's perspective, ended by detaching. The
	// pending snapshot is finalized later with how the background work
	// actually ended (see finalizePendingSnapshot).
	return &AgentOutput[State]{
		SessionID:    pending.SessionID,
		SnapshotID:   pending.SnapshotID,
		FinishReason: AgentFinishReasonDetached,
	}, nil
}

// finalizePendingSnapshot rewrites the pending snapshot row with the
// terminal state and status. abortedByUser distinguishes a context
// cancellation from abortSnapshot (status=aborted) from an internal
// failure (status=failed). The write is funneled through SaveSnapshot
// so the read-and-rewrite is one atomic step: if the row has already
// transitioned to aborted (a late abort racing this finalize),
// SaveSnapshot sees it inside fn and we leave the row untouched.
func (rt *agentRuntime[Stream, State]) finalizePendingSnapshot(
	ctx context.Context,
	pending *SessionSnapshot[State],
	result *AgentResult,
	fnErr error,
	abortedByUser bool,
) {
	finalState := *rt.session.State()
	// Captured outside the SaveSnapshot callback (which must stay pure): the
	// finalizer runs after fn returned, so this is stable. The abort/error
	// branches below own their reasons and ignore this clean-success default.
	succeededReason := rt.sess.invocationReason(result)

	_, err := rt.cfg.store.SaveSnapshot(ctx, pending.SnapshotID,
		func(existing *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			// Late abort wins over the terminal we were about to land: keep
			// the aborted status and whatever state the abort left, but
			// stamp the aborted finish reason so the snapshot is
			// self-describing. (AbortSnapshot only flips status; the runtime
			// owns the semantic reason.) Skip the write once already stamped.
			if existing != nil && existing.Status == SnapshotStatusAborted {
				if existing.FinishReason == AgentFinishReasonAborted {
					return nil, nil
				}
				annotated := *existing
				annotated.FinishReason = AgentFinishReasonAborted
				return &annotated, nil
			}

			status := SnapshotStatusSucceeded
			// The persisted finish reason records how the background work
			// actually ended, distinct from the detached reason the client
			// already saw on AgentOutput.
			finishReason := succeededReason
			var snapErr *core.GenkitError
			switch {
			case abortedByUser:
				status = SnapshotStatusAborted
				finishReason = AgentFinishReasonAborted
				if fnErr != nil {
					snapErr = core.AsGenkitError(fnErr) // aborted wins, preserve text
				}
			case fnErr != nil:
				status = SnapshotStatusFailed
				finishReason = AgentFinishReasonFailed
				snapErr = core.AsGenkitError(fnErr)
			}

			return &SessionSnapshot[State]{
				SessionID:    pending.SessionID,
				ParentID:     pending.ParentID,
				Event:        SnapshotEventDetach,
				Status:       status,
				FinishReason: finishReason,
				Error:        snapErr,
				State:        &finalState,
			}, nil
		})
	if err != nil {
		logger.FromContext(ctx).Error("agent: failed to finalize pending snapshot",
			"snapshotId", pending.SnapshotID, "err", err)
	}
}

// loadSession constructs a Session from the invocation's init payload,
// loading from the store when a snapshot or session ID is provided.
// Returns the loaded snapshot too so the runtime can chain ParentID (and
// carry the session ID) off it.
//
// State is mutually exclusive with both SessionID and SnapshotID: it is
// the client-managed conversation source and carries its own identity
// ([SessionState.SessionID]), while the two IDs resolve against a store.
// SessionID and SnapshotID compose: the snapshot picks the exact resume
// point and the session ID is asserted against it.
func loadSession[State any](
	ctx context.Context,
	init *AgentInit[State],
	store SessionStore[State],
) (*Session[State], *SessionSnapshot[State], error) {
	s := &Session[State]{store: store}
	if init == nil {
		return s, nil, nil
	}

	if init.State != nil && (init.SessionID != "" || init.SnapshotID != "") {
		return nil, nil, core.NewError(core.INVALID_ARGUMENT,
			"state is mutually exclusive with session ID and snapshot ID; a client-managed conversation's identity rides inside the state (SessionState.SessionID)")
	}

	switch {
	case init.State != nil:
		if store != nil {
			return nil, nil, core.NewError(core.FAILED_PRECONDITION,
				"state provided but agent has a session store configured (server-managed state); use snapshot ID instead")
		}
		// Deep-copy at the entry boundary: an in-process caller retains
		// its state object ([WithState] documents resending it), so the
		// session must own private memory. Without this, AddArtifacts'
		// in-place replace writes into the caller's array and the
		// caller's later mutations race snapshot marshaling.
		s.state = *jsonClone(init.State)
		return s, nil, nil

	case init.SnapshotID != "":
		if store == nil {
			return nil, nil, core.NewError(core.FAILED_PRECONDITION,
				"snapshot ID %q provided but agent has no session store configured (client-managed state); use state instead", init.SnapshotID)
		}
		snap, err := store.GetSnapshot(ctx, init.SnapshotID)
		if err != nil {
			return nil, nil, core.NewError(core.INTERNAL, "failed to load snapshot %q: %v", init.SnapshotID, err)
		}
		if snap == nil {
			return nil, nil, core.NewError(core.NOT_FOUND, "snapshot %q not found", init.SnapshotID)
		}
		// A session ID sent alongside the snapshot ID asserts which
		// conversation the snapshot belongs to; a mismatch means the
		// caller would silently continue the wrong conversation.
		if init.SessionID != "" && snap.SessionID != init.SessionID {
			return nil, nil, core.NewError(core.INVALID_ARGUMENT,
				"snapshot %q does not belong to session %q (snapshot's session: %q)", init.SnapshotID, init.SessionID, snap.SessionID)
		}
		return resumeSessionFrom(s, snap)

	case init.SessionID != "":
		if store == nil {
			return nil, nil, core.NewError(core.FAILED_PRECONDITION,
				"session ID %q provided but agent has no session store configured (client-managed state); the conversation's identity rides inside the state object (SessionState.SessionID)", init.SessionID)
		}
		snap, err := store.GetLatestSnapshot(ctx, init.SessionID)
		if err != nil {
			return nil, nil, core.NewError(core.INTERNAL, "failed to resolve latest snapshot for session %q: %v", init.SessionID, err)
		}
		if snap == nil {
			return nil, nil, core.NewError(core.NOT_FOUND, "no resumable snapshot found for session %q", init.SessionID)
		}
		if snap.SessionID != init.SessionID {
			return nil, nil, core.NewError(core.INTERNAL,
				"store resolved session %q to snapshot %q, which belongs to session %q; the store violates the GetLatestSnapshot contract", init.SessionID, snap.SnapshotID, snap.SessionID)
		}
		return resumeSessionFrom(s, snap)
	}
	return s, nil, nil
}

// resumeSessionFrom validates that snap is in a resumable status and loads
// its state into s. Shared by the snapshot-ID and session-ID init paths;
// the session-ID path can only hit the pending case (a conforming store's
// GetLatestSnapshot never resolves to failed/aborted dead ends), but the
// full switch stays as a defense against non-conforming stores.
func resumeSessionFrom[State any](s *Session[State], snap *SessionSnapshot[State]) (*Session[State], *SessionSnapshot[State], error) {
	switch snap.Status {
	case SnapshotStatusFailed:
		msg := "snapshot recorded an error"
		if snap.Error != nil && snap.Error.Message != "" {
			msg = snap.Error.Message
		}
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot %q terminated with error: %s", snap.SnapshotID, msg)
	case SnapshotStatusPending:
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot %q is still pending: its detached invocation is still running; wait for it to finalize or abort it before resuming", snap.SnapshotID)
	case SnapshotStatusAborted:
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot %q was aborted", snap.SnapshotID)
	}
	if snap.State != nil {
		// Stores may return rows sharing memory with their internal
		// copies (the [SnapshotReader] contract does not require fresh
		// memory), so the session takes a private copy; otherwise two
		// invocations resumed from the same snapshot would cross-corrupt
		// through the shared backing arrays.
		s.state = *jsonClone(snap.State)
	}
	return s, snap, nil
}

// --- chunkRouter ---
//
// chunkRouter owns the intermediate stream channel that all chunks flow
// through on their way to outCh. A chunk's in-process side effects
// (adding artifacts to the session, accumulating turn chunks for span
// output) are applied synchronously by Responder.send before the chunk
// enters the router, so every chunk gets them in its sender's goroutine
// regardless of whether detach has landed; the router owns only the wire
// forward to outCh, which is the one thing detach suppresses, since the
// bidi framework closes outCh shortly after bidiFn returns. The router
// commits to not writing before we return so that close is safe, and
// keeps draining its input so the user fn never blocks on a responder
// send.

type chunkRouter[Stream, State any] struct {
	ctx     context.Context // action context; ends on client disconnect (or completion)
	in      chan *AgentStreamChunk[Stream]
	out     chan<- *AgentStreamChunk[Stream]
	session *Session[State]

	turnMu     sync.Mutex
	turnChunks []*AgentStreamChunk[Stream]

	done          chan struct{}
	stopWriting   chan struct{}
	writerStopped chan struct{}
}

func startChunkRouter[Stream, State any](
	ctx context.Context,
	session *Session[State],
	out chan<- *AgentStreamChunk[Stream],
) *chunkRouter[Stream, State] {
	r := &chunkRouter[Stream, State]{
		ctx:           ctx,
		in:            make(chan *AgentStreamChunk[Stream]),
		out:           out,
		session:       session,
		done:          make(chan struct{}),
		stopWriting:   make(chan struct{}),
		writerStopped: make(chan struct{}),
	}
	go r.run()
	return r
}

func (r *chunkRouter[Stream, State]) run() {
	defer close(r.done)
	if !r.forward() {
		// r.in closed while writes were still allowed; nothing left to do.
		return
	}
	close(r.writerStopped)
	// Writes stopped (detach, shutdown, or client disconnect): keep
	// draining so a producer mid-send never blocks. The chunks' side
	// effects already happened at Send time; only the wire forward to
	// outCh is suppressed, so the chunks are simply discarded.
	for range r.in {
	}
}

// applySideEffects records the chunk's effect on session state and turn
// span output. Invoked synchronously from Responder.send, in the
// sender's goroutine, so the effects are ordered before everything the
// sender does after Send: a state read, a turn-end snapshot, or
// [SessionRunner.Result] immediately after SendArtifact observes the
// artifact. The artifact is deep-copied on its way into the session so
// the sender's retained pointer (which also rides the wire chunk) cannot
// alias live session state.
func (r *chunkRouter[Stream, State]) applySideEffects(chunk *AgentStreamChunk[Stream]) {
	if chunk.Artifact != nil {
		r.session.AddArtifacts(jsonClone(chunk.Artifact))
	}
	if chunk.TurnEnd == nil {
		r.turnMu.Lock()
		r.turnChunks = append(r.turnChunks, chunk)
		r.turnMu.Unlock()
	}
}

// forward delivers chunks to outCh until told to stop writing, the
// action context ends, or r.in closes. Returns true if the router must
// keep draining (writes stopped), false if r.in closed.
func (r *chunkRouter[Stream, State]) forward() bool {
	for {
		select {
		case chunk, ok := <-r.in:
			if !ok {
				return false
			}
			select {
			case r.out <- chunk:
			case <-r.stopWriting:
				return true
			case <-r.ctx.Done():
				// The client is gone (disconnect cancels the action
				// context), so nothing will drain out again and a blocked
				// forward would wedge close. Drop the chunk and switch to
				// side-effects-only mode.
				return true
			}
		case <-r.stopWriting:
			return true
		}
	}
}

// responder returns a [Responder] that applies chunk side effects
// synchronously and sends chunks into the router for the wire forward.
// The returned Responder's Send methods drop the forward (returning
// promptly) when ctx is cancelled.
func (r *chunkRouter[Stream, State]) responder(ctx context.Context) Responder[Stream] {
	return Responder[Stream]{in: r.in, ctx: ctx, effects: r.applySideEffects}
}

// sendChunk delivers chunk to the router for producers other than the
// user agent function (e.g. the runtime's emitTurnEnd). It skips the
// in-process side effects (the only runtime-produced chunk is TurnEnd,
// which has none: no artifact, and TurnEnd is excluded from turn-chunk
// accumulation) and returns promptly if ctx is cancelled, dropping the
// chunk.
func (r *chunkRouter[Stream, State]) sendChunk(ctx context.Context, chunk *AgentStreamChunk[Stream]) {
	select {
	case r.in <- chunk:
	case <-ctx.Done():
	}
}

// collectTurnChunks returns and resets accumulated turn chunks.
func (r *chunkRouter[Stream, State]) collectTurnChunks() []*AgentStreamChunk[Stream] {
	r.turnMu.Lock()
	defer r.turnMu.Unlock()
	result := r.turnChunks
	r.turnChunks = nil
	return result
}

// stopAndWait tells the router to stop writing to out and blocks until it
// has committed. After it returns, it is safe for the framework to close
// out without risking a write-to-closed-channel panic.
func (r *chunkRouter[Stream, State]) stopAndWait() {
	close(r.stopWriting)
	<-r.writerStopped
}

// close signals end-of-input and waits for the router to drain.
func (r *chunkRouter[Stream, State]) close() {
	close(r.in)
	<-r.done
}

// --- detachIntake ---
//
// detachIntake separates eager src reading from runner-paced forwarding,
// and owns the queue and suspend state.
//
// The reader goroutine pulls from the bidi framework's inCh as soon as
// inputs arrive and appends them to an internal queue. This is what makes
// detach detection immediate: the moment an input with [AgentInput.Detach]
// lands in src, the reader sees it without waiting for the runner to
// finish whatever it's processing.
//
// The forwarder goroutine pops the queue and writes to dst, blocking on
// the runner via turnDone so it stays in step with turn pacing.
//
// Snapshot suspension after detach is not the intake's concern: the
// runner gates writes itself (see SessionRunner.suspendSnapshots), so a
// detach can atomically wait out an in-flight turn-end write. The intake
// only owns input pacing.

type detachIntake struct {
	src    <-chan *AgentInput
	dst    chan *AgentInput
	notify chan struct{} // buffered size 1; wakes forwarder when queue grows

	// turnDone is signaled at each turn end to release the forwarder so
	// it may pop the next input. Initialized with one token so the very
	// first turn can start without a preceding turn end.
	turnDone chan struct{}

	mu    sync.Mutex
	queue []*AgentInput

	readDone atomic.Bool
	detachCh chan struct{} // signaled by reader when detach observed

	stop     chan struct{}
	stopOnce sync.Once
	done     chan struct{}
}

func startDetachIntake(src <-chan *AgentInput) *detachIntake {
	i := &detachIntake{
		src:      src,
		dst:      make(chan *AgentInput),
		notify:   make(chan struct{}, 1),
		turnDone: make(chan struct{}, 1),
		detachCh: make(chan struct{}, 1),
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
	}
	i.turnDone <- struct{}{} // initial credit for the first turn
	go i.run()
	return i
}

func (i *detachIntake) run() {
	defer close(i.done)

	forwarderDone := make(chan struct{})
	go func() {
		defer close(forwarderDone)
		defer close(i.dst)
		i.forward()
	}()

	i.read()
	<-forwarderDone
}

// signal wakes the forwarder. Non-blocking: the channel is buffered size
// 1, so a pending signal is enough.
func (i *detachIntake) signal() {
	select {
	case i.notify <- struct{}{}:
	default:
	}
}

// read pulls eagerly from src into the internal queue and detects detach
// the moment it lands. When detach is observed, it drains any remaining
// buffered src non-blockingly (so all pre-detach inputs are accounted
// for), signals the detach handler, and exits.
func (i *detachIntake) read() {
	defer func() {
		i.readDone.Store(true)
		i.signal()
	}()

	for {
		select {
		case input, ok := <-i.src:
			if !ok {
				return
			}
			if input == nil {
				// A nil input (e.g. a JSON null decoded by a transport)
				// carries nothing to process; dropping it here also keeps
				// nil out of the queue, where the forwarder would read it
				// as end-of-input.
				continue
			}
			if input.Detach {
				i.handleDetach(input)
				return
			}
			i.enqueue(input)
		case <-i.stop:
			return
		}
	}
}

func (i *detachIntake) enqueue(input *AgentInput) {
	i.mu.Lock()
	i.queue = append(i.queue, input)
	i.mu.Unlock()
	i.signal()
}

// handleDetach drains any buffered src inputs into the queue and signals
// the detach handler. The detach handler then suspends turn-end snapshots
// (via the runner) while the queued inputs finish processing.
//
// A pure detach signal (no Messages, no Resume payload) is dropped
// rather than enqueued: it carries no payload to process, so it would
// just trigger a no-op turn. Callers that want to ride a final input
// on the detach signal can do so by calling
// Send(&AgentInput{Detach: true, Message: ...}) explicitly.
func (i *detachIntake) handleDetach(first *AgentInput) {
	var drained []*AgentInput
	if hasInputPayload(first) {
		drained = append(drained, first)
	}
drainLoop:
	for {
		select {
		case more, ok := <-i.src:
			if !ok {
				break drainLoop
			}
			if more != nil {
				drained = append(drained, more)
			}
		default:
			break drainLoop
		}
	}

	if len(drained) > 0 {
		i.mu.Lock()
		i.queue = append(i.queue, drained...)
		i.mu.Unlock()
		i.signal()
	}

	select {
	case i.detachCh <- struct{}{}:
	case <-i.stop:
	}
}

// hasInputPayload reports whether the input carries data the runner would
// otherwise process. Used to filter pure detach signals out of the
// queue so they don't trigger no-op turns.
func hasInputPayload(in *AgentInput) bool {
	if in == nil {
		return false
	}
	if in.Message != nil {
		return true
	}
	if in.Resume != nil && (len(in.Resume.Respond) > 0 || len(in.Resume.Restart) > 0) {
		return true
	}
	return false
}

// forward pops the queue and writes to dst at the runner's pace. The
// runtime signals turnDone via releaseForward when it's ready for the
// next input; until then the forwarder waits, so it never gets ahead of
// the runner.
func (i *detachIntake) forward() {
	for {
		// Wait for the previous turn to release us (initial credit lets
		// the first turn through immediately).
		select {
		case <-i.turnDone:
		case <-i.stop:
			return
		}
		input := i.awaitInput()
		if input == nil {
			return // reader done with empty queue, or stop signaled
		}
		forwarded := *input
		forwarded.Detach = false
		select {
		case i.dst <- &forwarded:
		case <-i.stop:
			return
		}
	}
}

// awaitInput blocks until the queue has an input, the reader is done, or
// stop is signaled. Returns the popped input or nil if no further inputs
// will arrive.
func (i *detachIntake) awaitInput() *AgentInput {
	for {
		i.mu.Lock()
		if len(i.queue) > 0 {
			input := i.queue[0]
			i.queue = i.queue[1:]
			i.mu.Unlock()
			return input
		}
		done := i.readDone.Load()
		i.mu.Unlock()
		if done {
			return nil
		}
		select {
		case <-i.notify:
		case <-i.stop:
			return nil
		}
	}
}

// releaseForward releases the forwarder so it can pop the next input.
// Called by the runtime's emitTurnEnd at each turn end (and only there)
// so the forwarder stays in step with the runner's turn pacing.
func (i *detachIntake) releaseForward() {
	select {
	case i.turnDone <- struct{}{}:
	default:
	}
}

func (i *detachIntake) out() <-chan *AgentInput {
	return i.dst
}

func (i *detachIntake) detachSignal() <-chan struct{} {
	return i.detachCh
}

// stopAndWait forces the intake to exit and waits for both reader and
// forwarder goroutines.
func (i *detachIntake) stopAndWait() {
	i.stopOnce.Do(func() { close(i.stop) })
	<-i.done
}

// promptMessageKey is the metadata key used to tag base messages from the
// agent config (system prompt, prompt template output, etc.) so they can be
// excluded from session history after generation.
const promptMessageKey = "_genkit_prompt"

// validateUserMessage rejects inputs the prompt-backed agent loop can't
// safely consume: a non-user role would be appended to history under the
// wrong speaker, and tool request / response parts belong on the
// [AgentInput.Resume] payload, not on a turn message.
func validateUserMessage(m *ai.Message) error {
	if m == nil {
		return nil
	}
	if m.Role != "" && m.Role != ai.RoleUser {
		return core.NewError(core.INVALID_ARGUMENT,
			"agent input message must have role %q, got %q", ai.RoleUser, m.Role)
	}
	for _, p := range m.Content {
		if p == nil {
			continue
		}
		if p.IsToolRequest() || p.IsToolResponse() {
			return core.NewError(core.INVALID_ARGUMENT,
				"agent input message must not contain tool request or response parts; use AgentInput.Resume instead")
		}
	}
	return nil
}

// agentLoop returns the per-turn function for a prompt-backed agent. Each
// turn renders the prompt, appends conversation history, calls the model
// with streaming, and updates the session.
//
// defaultInput is the prompt input passed to Render on every turn. It is
// nil for inline-defined prompts ([FromInline]), which take no per-turn
// input.
func agentLoop[State any](r api.Registry, prompt ai.Prompt, defaultInput any) AgentFunc[any, State] {
	return func(ctx context.Context, resp Responder[any], sess *SessionRunner[State]) (*AgentResult, error) {
		if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
			if err := validateUserMessage(input.Message); err != nil {
				return nil, err
			}

			actionOpts, err := prompt.Render(ctx, defaultInput)
			if err != nil {
				return nil, fmt.Errorf("prompt render: %w", err)
			}

			// Tag base messages so they can be filtered out of session
			// history after generation. Tag copies rather than the
			// rendered messages themselves: Render can alias message
			// metadata from shared prompt config (e.g. messages
			// registered via [ai.WithMessages]), so tagging in place
			// would leak the tag into the registered prompt and race
			// with concurrent invocations.
			base := make([]*ai.Message, 0, len(actionOpts.Messages))
			for _, m := range actionOpts.Messages {
				if m == nil {
					continue
				}
				tagged := *m
				tagged.Metadata = maps.Clone(tagged.Metadata)
				if tagged.Metadata == nil {
					tagged.Metadata = make(map[string]any, 1)
				}
				tagged.Metadata[promptMessageKey] = true
				base = append(base, &tagged)
			}

			// Append conversation history after the base messages.
			actionOpts.Messages = append(base, sess.Messages()...)

			// If a resume payload was provided, forward it to the
			// generate call so handleResumeOption re-executes the
			// interrupted tools and / or applies the responses.
			if input.Resume != nil {
				actionOpts.Resume = &ai.GenerateActionResume{
					Respond: input.Resume.Respond,
					Restart: input.Resume.Restart,
				}
			}

			modelResp, err := ai.GenerateWithRequest(ctx, r, actionOpts, nil,
				func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					resp.SendModelChunk(chunk)
					return nil
				},
			)
			if err != nil {
				return nil, fmt.Errorf("generate: %w", err)
			}

			// Replace session messages with the full history minus base
			// messages. This captures intermediate tool call/response
			// messages from the tool loop, not just the final response.
			if modelResp.Request != nil {
				history := modelResp.History()
				msgs := make([]*ai.Message, 0, len(history))
				for _, m := range history {
					if m.Metadata != nil && m.Metadata[promptMessageKey] == true {
						continue
					}
					msgs = append(msgs, m)
				}
				sess.SetMessages(msgs)
			} else if modelResp.Message != nil {
				sess.AddMessages(modelResp.Message)
			}

			// Stream interrupt parts so the client can detect and
			// handle them (e.g. prompt the user for confirmation).
			if modelResp.FinishReason == ai.FinishReasonInterrupted {
				if parts := modelResp.Interrupts(); len(parts) > 0 {
					resp.SendModelChunk(&ai.ModelResponseChunk{
						Role:    ai.RoleTool,
						Content: parts,
					})
				}
			}

			// Forward the generate response's finish reason verbatim: the
			// agent enum is a superset of the model enum for these values,
			// so the turn (and a single-turn invocation) reports e.g.
			// "interrupted" without the client scanning message content.
			return &TurnResult{FinishReason: AgentFinishReason(modelResp.FinishReason)}, nil
		}); err != nil {
			return nil, err
		}
		return sess.Result(), nil
	}
}

// --- Agent client API ---

// StreamBidi starts a new agent invocation with bidirectional streaming.
// Use this for multi-turn interactions where you need to send multiple inputs
// and receive streaming chunks. For single-turn usage, see Run and RunText.
func (a *Agent[Stream, State]) StreamBidi(
	ctx context.Context,
	opts ...InvocationOption[State],
) (*AgentConnection[Stream, State], error) {
	init, err := a.resolveOptions(opts)
	if err != nil {
		return nil, err
	}
	conn, err := a.action.StreamBidi(ctx, init)
	if err != nil {
		return nil, err
	}
	return &AgentConnection[Stream, State]{conn: conn}, nil
}

// Run starts a single-turn agent invocation with the given input.
// It sends the input, waits for the agent to complete, and returns the output.
// For multi-turn interactions or streaming, use StreamBidi instead.
//
// In-band failures (e.g. a failed turn) resolve as a failed [AgentOutput]
// rather than an error; a rejected init payload fails with an error, since
// the invocation never starts. See [AgentConnection.Output].
func (a *Agent[Stream, State]) Run(
	ctx context.Context,
	input *AgentInput,
	opts ...InvocationOption[State],
) (*AgentOutput[State], error) {
	conn, err := a.StreamBidi(ctx, opts...)
	if err != nil {
		return nil, err
	}
	// The invocation may resolve before consuming the input (e.g. an init
	// validation failure errors out before the first turn); the outcome,
	// whether output or error, is on Output regardless.
	if err := conn.Send(input); err != nil && !errors.Is(err, core.ErrActionCompleted) {
		return nil, err
	}
	return conn.Output()
}

// RunText is a convenience method that starts a single-turn agent invocation
// with a user text message. It is equivalent to calling Run with an
// AgentInput whose Message is a user text message.
func (a *Agent[Stream, State]) RunText(
	ctx context.Context,
	text string,
	opts ...InvocationOption[State],
) (*AgentOutput[State], error) {
	return a.Run(ctx, &AgentInput{
		Message: ai.NewUserTextMessage(text),
	}, opts...)
}

// resolveOptions applies invocation options and returns the init struct.
// Mutual exclusivity is checked here, once, after all options are merged:
// WithState excludes both WithSessionID and WithSnapshotID (a
// client-managed conversation's identity rides inside the state itself),
// while WithSessionID and WithSnapshotID compose as an assertion.
// Per-option duplicate checks live in applyInvocation.
func (a *Agent[Stream, State]) resolveOptions(opts []InvocationOption[State]) (*AgentInit[State], error) {
	invOpts := &invocationOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyInvocation(invOpts); err != nil {
			return nil, fmt.Errorf("Agent %q: %w", a.action.Name(), err)
		}
	}

	if invOpts.state != nil && invOpts.snapshotID != "" {
		return nil, fmt.Errorf("Agent %q: WithState and WithSnapshotID are mutually exclusive", a.action.Name())
	}
	if invOpts.state != nil && invOpts.sessionIDSet {
		return nil, fmt.Errorf("Agent %q: WithState and WithSessionID are mutually exclusive; the conversation's identity rides inside the state (SessionState.SessionID)", a.action.Name())
	}

	return &AgentInit[State]{
		SessionID:  invOpts.sessionID,
		SnapshotID: invOpts.snapshotID,
		State:      invOpts.state,
	}, nil
}

// --- AgentConnection ---

// AgentConnection wraps BidiConnection with agent-specific Send helpers
// (SendMessage / SendText / SendResume / Detach) and an Output that
// always waits for finalization (so detached invocations see the
// pending snapshot ID rather than a context-cancellation error).
type AgentConnection[Stream, State any] struct {
	conn *core.BidiConnection[*AgentInput, *AgentOutput[State], *AgentStreamChunk[Stream]]
}

// Send sends an AgentInput to the agent. The input must not be nil.
//
// Once the invocation has resolved (e.g. a failed turn ended it), Send
// fails with an error matching [core.ErrActionCompleted]; the outcome is
// on [AgentConnection.Output]. The same applies to the SendMessage,
// SendText, SendResume, and Detach helpers.
func (c *AgentConnection[Stream, State]) Send(input *AgentInput) error {
	if input == nil {
		return core.NewError(core.INVALID_ARGUMENT, "agent input must not be nil")
	}
	return c.conn.Send(input)
}

// SendMessage sends a message to the agent for one turn.
func (c *AgentConnection[Stream, State]) SendMessage(message *ai.Message) error {
	return c.conn.Send(&AgentInput{Message: message})
}

// SendText sends a user text message to the agent.
func (c *AgentConnection[Stream, State]) SendText(text string) error {
	return c.conn.Send(&AgentInput{
		Message: ai.NewUserTextMessage(text),
	})
}

// SendResume sends a resume payload to continue an interrupted generation.
// Construct the payload with [ai.ToolDef.RestartWith] or
// [ai.ToolDef.RespondWith] parts.
func (c *AgentConnection[Stream, State]) SendResume(resume *ToolResume) error {
	return c.conn.Send(&AgentInput{Resume: resume})
}

// Detach asks the server to write a pending snapshot, close the
// connection, and continue processing any already-buffered inputs in
// the background. Output() returns the pending snapshot ID; the client
// can later call AbortSnapshot to stop the background work or
// GetSnapshot to observe its progression. The pending snapshot is
// finalized with the cumulative final state once the queued inputs
// are processed.
//
// Streamed chunks emitted after detach are not forwarded over the wire
// (the connection is gone), but their session-level side effects still
// apply: artifacts sent via [Responder.SendArtifact] land in the
// session and end up in the final snapshot's state.
//
// To send a final input as part of the same wire message, use
// Send(&AgentInput{Detach: true, Message: ...}) directly.
func (c *AgentConnection[Stream, State]) Detach() error {
	return c.conn.Send(&AgentInput{Detach: true})
}

// Close signals that no more inputs will be sent.
func (c *AgentConnection[Stream, State]) Close() error {
	return c.conn.Close()
}

// Receive returns an iterator for receiving stream chunks. Breaking out
// of the iterator does not cancel the connection; multi-turn callers
// routinely break on [TurnEnd], send the next input, then call Receive
// again to consume the next batch. Call [AgentConnection.Output] to
// finish the invocation, or cancel the ctx passed to StreamBidi to
// abort it.
func (c *AgentConnection[Stream, State]) Receive() iter.Seq2[*AgentStreamChunk[Stream], error] {
	return c.conn.Receive()
}

// Output finalizes the connection and returns the agent's result.
//
// Output is the single "I'm done" call: it implicitly closes the input
// side, drains any chunks the caller did not consume via Receive, and
// blocks until the agent finalizes. Calling Close first is allowed but
// redundant. Output is idempotent: subsequent calls return the same
// (*AgentOutput, error); the returned pointer is shared across calls,
// so treat it as read-only.
//
// In-band failures resolve rather than error: a failed turn returns an
// [AgentOutput] with [AgentFinishReasonFailed], the error on
// [AgentOutput.Error] (original status intact), and the last-good state
// on [AgentOutput.State] (client-managed) or behind
// [AgentOutput.SnapshotID] (server-managed), so a failure costs the
// caller only the failed turn, never the session. A detached invocation
// resolves with the pending snapshot ID rather than a cancellation
// error. A non-nil error here means the invocation never started (a
// rejected init payload) or could not run to a result (e.g. the
// connection's context was cancelled).
//
// Do not call Output concurrently with a goroutine iterating Receive;
// both consume from the same stream and chunks would be split between
// them. Finish Receive first, then call Output.
func (c *AgentConnection[Stream, State]) Output() (*AgentOutput[State], error) {
	_ = c.conn.Close()
	// The core connection applies backpressure and its Output does not
	// consume the stream, so drain the chunks the caller did not Receive;
	// the agent must never wedge publishing to a stream nobody reads.
	// Receive ends on completion or cancellation either way, and the core
	// Output prefers the finalized result when both are ready.
	for range c.conn.Receive() {
	}
	return c.conn.Output()
}

// Done returns a channel closed when the connection completes.
func (c *AgentConnection[Stream, State]) Done() <-chan struct{} {
	return c.conn.Done()
}
