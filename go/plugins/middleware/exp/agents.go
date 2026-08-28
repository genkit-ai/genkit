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
	"maps"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
)

// agentsMarker tags the system prompt part injected by this middleware. The
// listing is constant for a given configuration, so it is injected once and
// matched (no-op) on later tool-loop iterations.
const agentsMarker = "agents-instructions"

// defaultToolPrefix is the prefix applied to generated delegation tool names
// when [Agents.ToolPrefix] is unset (tools become delegate_to_<agent>).
const defaultToolPrefix = "delegate_to"

// ArtifactStrategy controls how a sub-agent's artifacts are surfaced back to the
// orchestrator by the [Agents] middleware.
type ArtifactStrategy string

const (
	// ArtifactStrategyInline includes artifact content in the delegation tool
	// result so the orchestrator model can see it, and also merges artifacts
	// into the parent session. This is the default.
	ArtifactStrategyInline ArtifactStrategy = "inline"
	// ArtifactStrategySession merges artifacts into the parent session only; the
	// tool result names the artifacts but omits their content. Pair it with the
	// [Artifacts] middleware so the model can read/write session artifacts.
	ArtifactStrategySession ArtifactStrategy = "session"
)

// resolveAgent looks the agent up by name through g and returns its handle.
// Resolution goes through the Genkit instance (the sanctioned path for
// third-party middleware) rather than the registry directly; the handle
// carries the agent's companion actions and capability metadata along with
// the run surface. Both delegation and background-task reads resolve through
// it.
// The lookup answers a miss with nil, as every Lookup in the framework does,
// so the message a miss deserves is written here: this middleware knows the
// agent was configured on it, which makes an absent registration a deployment
// mistake worth naming rather than a lookup that came up empty.
func resolveAgent(g *genkit.Genkit, ref aix.AgentRef) (*aix.AgentHandle, error) {
	if g == nil {
		// A failed precondition: a wiring gap in how the middleware runs, not
		// anything the caller sent.
		return nil, status.Errorf(status.ErrFailedPrecondition,
			"no Genkit instance on the context (the agents middleware must run within genkit.Generate or a genkit-defined agent)")
	}
	agent := genkitx.LookupAgent(g, ref.Name)
	if agent == nil {
		return nil, status.Errorf(status.ErrNotFound,
			"agent %q is configured on the agents middleware but is not registered in this process", ref.Name)
	}
	return agent, nil
}

// Agents is a middleware that enables sub-agent delegation.
//
// For every configured agent it injects a dedicated delegation tool (e.g.
// delegate_to_researcher) whose description is the agent's configured
// description or, in the system prompt, the description auto-discovered from the
// registry. A <sub-agents> block listing the available agents is appended to the
// system prompt.
//
// When the model calls a delegation tool the middleware resolves the target
// agent from the registry (via the [github.com/firebase/genkit/go/genkit.Genkit]
// instance carried on the context), optionally forwards recent conversation
// history, runs the sub-agent with the task, and returns its response as the
// tool result.
//
// Artifact handling follows [Agents.ArtifactStrategy]: ArtifactStrategyInline
// (default) returns artifact content in the tool result and merges artifacts
// into the parent session; ArtifactStrategySession merges into the session only
// and returns names. Merged artifacts are namespaced by an invocation ID
// (<agent>_<n>/<name>) and tagged with the source agent.
//
// If a sub-agent interrupts (e.g. for human input) it is reported back to the
// orchestrator as a normal tool response, not propagated as an interrupt: there
// is no stateful sub-agent runtime to resume into, so interactive sub-agent
// interaction is a future feature.
//
// With [Agents.Async] set, delegation tools additionally accept a "background"
// flag. A background delegation starts the sub-agent through its detach support
// ([aix.AgentInput.Detach]): the sub-agent's runtime persists a pending
// snapshot, hands back a task ID immediately, and keeps working in the
// background, so the orchestrator can continue calling tools and collect the
// result later through the added check_background_tasks and
// wait_for_background_tasks tools, or drop it with abort_background_tasks. The
// pending snapshot (heartbeated while the worker lives, finalized in place
// with the cumulative state when the work settles) is the durable record, and
// task IDs are self-contained
// ("<agent>:<snapshotId>"), so a re-instantiated orchestrator can pick results
// up using nothing but the IDs recorded in its conversation history. Background
// delegation requires server-managed sub-agents whose stores implement
// [aix.SnapshotSubscriber] (e.g. the localstore stores); launches on other
// agents are rejected by the sub-agent runtime and reported as tool text.
//
// Task handles are not access-scoped: the background-task tools read any
// snapshot ID belonging to a configured sub-agent, whether or not this
// conversation launched it (mirroring the sub-agent's getSnapshot companion
// action, which is itself unscoped). In multi-tenant deployments treat
// snapshot IDs as capability-like secrets: text that reaches the orchestrator
// model can steer these tools at any ID it names.
//
// The middleware resolves agents through genkit.FromContext, which is seeded by
// genkit.Generate and by agents defined via the genkit/exp constructors
// (genkitx.DefineAgent and friends). It is therefore typically attached to an
// orchestrator agent (or a genkit.Generate call).
//
// Usage:
//
//	orchestrator := genkitx.DefineAgent(g, "orchestrator",
//	    aix.InlinePrompt{
//	        ai.WithModelName("googleai/gemini-flash-latest"),
//	        ai.WithSystem("You are a helpful project assistant."),
//	        ai.WithUse(
//	            &middlewarex.Agents{
//	                Agents: []aix.AgentRef{
//	                    {Name: "researcher"}, // by name
//	                    coderAgent.Ref(),     // by instance (carries its description)
//	                },
//	                MaxDelegations:   5,
//	                HistoryLength:    4,
//	                ArtifactStrategy: middlewarex.ArtifactStrategySession,
//	            },
//	            &middlewarex.Artifacts{},
//	        ),
//	    },
//	)
type Agents struct {
	// Agents lists the sub-agents available for delegation: by name
	// (aix.AgentRef{Name: ...}) or as a captured instance (agentValue.Ref()).
	// At least one is required.
	Agents []aix.AgentRef `json:"agents,omitempty" jsonschema_description:"Sub-agents available for delegation. At least one is required."`
	// ToolPrefix is the prefix for generated delegation tool names. A nil value
	// defaults to "delegate_to" (tools become delegate_to_<agent>); a pointer to
	// the empty string uses bare agent names. An explicitly set, non-empty
	// prefix also namespaces the [Agents.Async] background-task tools (e.g.
	// research_check_background_tasks). Two Async instances in one generate
	// call therefore need distinct, explicitly set prefixes: left at the
	// default they both emit the bare background-task tool names, and the
	// generate call is rejected for duplicate tools. New rejects colliding
	// names within one instance; it cannot see a sibling.
	ToolPrefix *string `json:"toolPrefix,omitempty" jsonschema_description:"Prefix for generated delegation tool names. Defaults to \"delegate_to\", so tools become delegate_to_<agent>. Set it to the empty string to use bare agent names. A non-empty prefix also namespaces the async background-task tools."`
	// MaxDelegations caps the number of sub-agent delegations per generate call,
	// preventing runaway delegation loops. 0 means unlimited.
	MaxDelegations int `json:"maxDelegations,omitempty" jsonschema_description:"Caps the number of sub-agent delegations per generate call, preventing runaway delegation loops. Defaults to 0, which means unlimited."`
	// HistoryLength is the number of recent user/model messages forwarded to a
	// sub-agent as context. 0 means only the task description is sent. History is
	// forwarded only to client-managed sub-agents (those without a session
	// store); server-managed sub-agents receive only the task.
	HistoryLength int `json:"historyLength,omitempty" jsonschema_description:"Number of recent user/model messages forwarded to a sub-agent as context. Defaults to 0, which sends only the task description. History reaches client-managed sub-agents only."`
	// ArtifactStrategy controls how sub-agent artifacts are surfaced. Defaults to
	// ArtifactStrategyInline.
	ArtifactStrategy ArtifactStrategy `json:"artifactStrategy,omitempty" jsonschema_description:"How sub-agent artifacts are surfaced. \"inline\" adds artifact content to the delegation tool result and merges it into the parent session. \"session\" merges into the session only. Defaults to \"inline\"."`
	// Async enables background delegation. Delegation tools gain a "background"
	// input flag that starts the sub-agent through its detach support and
	// returns a task ID immediately, and three shared tools are added, one per
	// control the orchestrator has over a launched task:
	// check_background_tasks (non-blocking status and results),
	// wait_for_background_tasks (blocks until the listed tasks settle), and
	// abort_background_tasks (stops tasks whose results are no longer needed).
	// Background delegation requires server-managed sub-agents whose stores
	// implement [aix.SnapshotSubscriber].
	Async bool `json:"async,omitempty" jsonschema_description:"Enables background delegation: delegation tools accept a \"background\" flag, and the check_background_tasks / wait_for_background_tasks / abort_background_tasks tools are added. Background delegation requires server-managed sub-agents whose session stores support detach."`
}

func (a Agents) Name() string { return provider + "/agents" }

// agentsState is the per-generate mutable state shared by the delegation tools
// and the generate hook. New allocates a fresh one per call, and a mutex guards
// it because delegation tools can run concurrently (parallel tool calls).
type agentsState struct {
	mu sync.Mutex
	// delegations counts delegations made so far, enforcing MaxDelegations and
	// providing the per-invocation number used to namespace artifacts.
	delegations int
	// seq allocates invocation numbers and never decreases, unlike the
	// delegations cap counter above: a refunded delegation's number must not
	// be reissued, because under parallel tool calls the reissued number can
	// belong to a still-running delegation, and two delegations sharing an
	// invocation ID overwrite each other's artifacts (AddArtifacts replaces
	// by name).
	seq int
	// conversation is the latest request message list, captured each turn for
	// optional history forwarding.
	conversation []*ai.Message
	// settledReports caches terminal background-task reports by task ID for
	// the rest of the generate call: completed, failed, and aborted rows
	// never change, so later re-checks skip the snapshot fetch and artifact
	// re-merge (and cannot clobber a merged artifact the orchestrator has
	// since edited). Pending, expired, and unresolvable reports are never
	// cached; those can still change.
	settledReports map[string]backgroundTaskReport
}

// New validates the configuration and returns the hooks: a delegation tool per
// agent plus a generate hook that injects the <sub-agents> system prompt and
// captures conversation history.
func (a Agents) New(ctx context.Context) (*ai.Hooks, error) {
	if len(a.Agents) == 0 {
		return nil, status.Errorf(status.ErrInvalidArgument,
			"agents middleware requires at least one agent in the \"agents\" option")
	}
	for _, ref := range a.Agents {
		if ref.Name == "" {
			return nil, status.Errorf(status.ErrInvalidArgument,
				"agents middleware: every agent reference must have a name")
		}
	}

	prefix := a.prefix()
	st := &agentsState{settledReports: make(map[string]backgroundTaskReport)}

	// Every generated tool name is validated against the set as it is built:
	// a collision (two agents mapping to one delegation tool name, or a
	// delegation tool landing on a background-task tool's name) would
	// otherwise surface only at generate time as a duplicate-tool rejection
	// of the whole request.
	names := make(map[string]string, len(a.Agents)+3)
	claimName := func(name, owner string) error {
		if prev, ok := names[name]; ok {
			return status.Errorf(status.ErrInvalidArgument,
				"agents middleware: tool name %q for %s collides with %s; use a different ToolPrefix or agent name", name, owner, prev)
		}
		names[name] = owner
		return nil
	}

	tools := make([]ai.Tool, 0, len(a.Agents)+3)
	for _, ref := range a.Agents {
		desc := ref.Description
		if desc == "" {
			desc = fmt.Sprintf("Delegates a task to the %q sub-agent.", ref.Name)
		}
		name := makeToolName(prefix, ref.Name)
		if err := claimName(name, fmt.Sprintf("agent %q", ref.Name)); err != nil {
			return nil, err
		}
		// The async variant carries the extra "background" input flag, so the
		// two modes need distinct input schemas (tool schemas are static).
		if a.Async {
			tools = append(tools, aix.NewTool(name, desc, a.delegateAsync(ref, st)))
		} else {
			tools = append(tools, aix.NewTool(name, desc, a.delegate(ref, st)))
		}
	}
	if a.Async {
		for _, n := range a.backgroundToolNames().all() {
			if err := claimName(n, "the background-task tools"); err != nil {
				return nil, err
			}
		}
		tools = append(tools, a.backgroundTaskTools(st)...)
	}

	// The <sub-agents> block depends only on the configuration and the
	// registry, both fixed for this call, so it is built here rather than per
	// tool-loop turn: New already runs exactly once per generate call, and
	// re-rendering cost a registry lookup and a descriptor copy per agent per
	// turn to produce a string the injector then dropped as identical.
	instructions := a.buildInstructions(genkit.FromContext(ctx))

	wrapGenerate := func(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
		// Capture the latest messages for optional history forwarding. The
		// delegation count is intentionally not reset here: this hook runs on
		// every tool-loop turn, but the count must accumulate across the whole
		// generate call (it starts at 0 when New allocates st).
		st.mu.Lock()
		st.conversation = params.Request.Messages
		st.mu.Unlock()

		params.Request = injectSystemText(params.Request, agentsMarker, instructions)
		return next(ctx, params)
	}

	return &ai.Hooks{
		Tools:        tools,
		WrapGenerate: wrapGenerate,
	}, nil
}

// delegateInput is the input schema for a delegation tool.
type delegateInput struct {
	Task string `json:"task" jsonschema_description:"A clear, self-contained description of the task to delegate."`
}

// delegationResult is the output of a delegation tool.
type delegationResult struct {
	// Response is the sub-agent's text response. For a background delegation it
	// describes the launch instead; the sub-agent's response arrives later via
	// the background-task tools.
	Response string `json:"response"`
	// Artifacts are the sub-agent's artifacts. Content is populated only under
	// ArtifactStrategyInline.
	Artifacts []delegatedArtifact `json:"artifacts,omitempty"`
	// TaskID is the delegation's handle ("<agent>:<snapshotId>"). For a
	// background delegation it names the pending task; for a synchronous
	// delegation to a server-managed sub-agent it names the run's last
	// committed snapshot, whatever the outcome. It is the input to the
	// background-task tools (check, wait, abort). Empty when there is nothing
	// addressable behind the result: a client-managed sub-agent, a run that
	// committed no turn, or an interrupt.
	TaskID string `json:"taskId,omitempty"`
	// Status is the outcome behind TaskID: "pending" when a background
	// delegation was started, and the settled outcome ("completed", "failed",
	// "aborted", or the finish reason for the rest) for a synchronous
	// delegation that carries a handle. Empty whenever TaskID is.
	Status string `json:"status,omitempty"`
}

type delegatedArtifact struct {
	Name    string `json:"name,omitempty"`
	Content string `json:"content,omitempty"`
}

// delegate builds the delegation tool function for one sub-agent. The function
// uses the experimental [aix.NewTool] signature: a plain [context.Context]
// rather than an [ai.ToolContext], since delegation needs only the context for
// agent resolution, sub-agent execution, and artifact merging.
func (a *Agents) delegate(ref aix.AgentRef, st *agentsState) func(context.Context, delegateInput) (delegationResult, error) {
	return func(ctx context.Context, in delegateInput) (delegationResult, error) {
		return a.runDelegation(ctx, ref, st, in.Task)
	}
}

// beginDelegation is the prologue every delegation shares, synchronous or
// background: it enforces MaxDelegations, reserves the delegation's invocation
// number, and resolves the sub-agent. A non-nil refusal is the tool result to
// return as-is: the model-facing text for a cap refusal or a resolution
// failure. conversation is the captured message list for optional history
// forwarding (see reserveDelegation); background launches ignore it.
//
// A resolution failure keeps its slot. The agent is misconfigured or missing,
// so every retry fails the same way, and refunding would mean the cap never
// bites on exactly the runaway loop it exists to stop.
func (a *Agents) beginDelegation(ctx context.Context, ref aix.AgentRef, st *agentsState) (invocationNum int, conversation []*ai.Message, agent *aix.AgentHandle, refusal *delegationResult) {
	invocationNum, conversation, ok := a.reserveDelegation(st)
	if !ok {
		logger.Warn(ctx, "delegation refused, limit reached", "agent", ref.Name, "limit", a.MaxDelegations)
		return 0, nil, nil, &delegationResult{Response: fmt.Sprintf(
			"Delegation limit reached (%d). Complete the task using information already gathered.", a.MaxDelegations)}
	}

	agent, err := resolveAgent(genkit.FromContext(ctx), ref)
	if err != nil {
		logger.Warn(ctx, "sub-agent resolution failed", "agent", ref.Name, "error", err)
		return 0, nil, nil, &delegationResult{Response: "Error: " + err.Error()}
	}
	return invocationNum, conversation, agent, nil
}

// runDelegation is the synchronous delegation body, shared by the plain
// delegation tool and the async-enabled variant when the model does not
// request background execution.
func (a *Agents) runDelegation(ctx context.Context, ref aix.AgentRef, st *agentsState, task string) (delegationResult, error) {
	invocationNum, conversation, agent, refusal := a.beginDelegation(ctx, ref, st)
	if refusal != nil {
		return *refusal, nil
	}

	// History rides in client-managed init state, which server-managed
	// agents reject; forward it only to client-managed sub-agents. The
	// filtering copy runs outside the state mutex.
	var history []*ai.Message
	if isClientManaged(agent) {
		history = recentTextHistory(conversation, a.HistoryLength)
	}

	logger.Debug(ctx, "delegating to sub-agent",
		"agent", ref.Name, "invocation", invocationNum, "historyMessages", len(history))
	start := time.Now()
	out, err := runSubAgent(ctx, agent, task, history, false)
	if err != nil {
		// The agent runtime resolves failures and interrupts gracefully (see
		// foldDelegationOutput), so this only fires for exceptions outside
		// that handling (e.g. a rejected init payload). Surface it as tool
		// output and keep the slot: the payload is built the same way every
		// time, so a retry is refused the same way.
		logger.Warn(ctx, "sub-agent call failed", "agent", ref.Name, "error", err)
		return delegationResult{Response: fmt.Sprintf("Error calling agent %q: %v", ref.Name, err)}, nil
	}

	result := a.foldDelegationOutput(ctx, ref, out, fmt.Sprintf("%s_%d", ref.Name, invocationNum))
	logger.Debug(ctx, "sub-agent delegation finished",
		"agent", ref.Name, "finishReason", string(out.FinishReason),
		"durationMs", time.Since(start).Milliseconds(), "artifacts", len(result.Artifacts))
	return result, nil
}

// reserveDelegation enforces MaxDelegations and reserves the next delegation's
// invocation number, atomically, before any work happens. ok is false when the
// cap is reached. The returned conversation is the raw captured message list
// for optional history forwarding: treat it as read-only (the generate hook
// replaces the slice header wholesale under the mutex; messages are never
// mutated in place), and filter it outside the lock.
func (a *Agents) reserveDelegation(st *agentsState) (invocationNum int, conversation []*ai.Message, ok bool) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if a.MaxDelegations > 0 && st.delegations >= a.MaxDelegations {
		return 0, nil, false
	}
	st.delegations++
	st.seq++
	return st.seq, st.conversation, true
}

// releaseDelegation returns a reserved cap slot to a delegation whose refusal
// names a retry that can succeed, which today means only one thing: a
// background launch refused because the sub-agent cannot detach, whose refusal
// tells the model to delegate synchronously instead. That retry must not be
// turned away by a cap the refusal itself consumed.
//
// Every other refusal keeps its slot. A refusal that will repeat identically
// (an unresolvable agent, a rejected init payload, a sub-agent that fails on
// its own) is precisely the runaway MaxDelegations exists to bound, and
// refunding those would leave the cap unable to bite at all.
//
// The released slot's invocation number is never reissued (the allocator is
// the separate seq counter), so a concurrent live delegation cannot end up
// sharing an artifact namespace with a later one.
func (a *Agents) releaseDelegation(st *agentsState) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.delegations--
}

// foldDelegationOutput turns a settled sub-agent output into a delegation tool
// result: interrupts and failures become explanatory text, and artifacts are
// merged into the parent session under invocationID and surfaced per the
// configured strategy.
// A server-managed output names the run's last committed snapshot, so every
// settled result but an interrupt is stamped with the same
// "<agent>:<snapshotId>" handle background delegations mint, plus the outcome
// it settled in. The handle is what makes a delegation addressable after the
// fact: the background-task tools accept it, and it is the currency a resume
// spends.
func (a *Agents) foldDelegationOutput(ctx context.Context, ref aix.AgentRef, out *aix.AgentOutput[json.RawMessage], invocationID string) delegationResult {
	// Interrupted first: it is one of the reasons that carry no result, and it
	// is the one with an explanation of its own worth giving. It is also the
	// one settled outcome that carries no handle: continuing past it means
	// answering the interrupt, which the orchestrator cannot do.
	if out.FinishReason == aix.AgentFinishReasonInterrupted {
		// Reported as text, not propagated: there is no stateful sub-agent
		// runtime to resume into, so the orchestrator could never satisfy it.
		return delegationResult{Response: interruptedResponse(ref.Name)}
	}
	var result delegationResult
	if out.SnapshotID != "" && out.FinishReason != aix.AgentFinishReasonDetached {
		result.TaskID = formatTaskID(ref.Name, out.SnapshotID)
		result.Status = settledStatus(out.FinishReason)
	}
	if !out.FinishReason.CarriesResult() {
		// Blocked, truncated, aborted, or failed. The turn's last message is
		// whatever the agent got out before it stopped, so it explains the
		// outcome rather than answering the task, and reporting it as the
		// answer would hand the orchestrator partial work as if it were final.
		result.Response = fmt.Sprintf("Error calling agent %q: %s",
			ref.Name, subAgentFailureMessage(out.FinishReason, out.Error, out.Message))
		return result
	}

	subArtifacts := namedArtifacts(out.Artifacts)
	result.Response = messageText(out.Message)
	if result.Response == "" {
		result.Response = noFinalMessageResponse(len(subArtifacts))
	}
	if len(subArtifacts) > 0 {
		// Merge into the parent session under both strategies (no-op if there
		// is no active session, e.g. a plain genkit.Generate call).
		mergeArtifacts(ctx, ref.Name, invocationID, subArtifacts)
		result.Artifacts = delegatedArtifacts(invocationID, subArtifacts, a.strategy())
	}
	return result
}

// noFinalMessageResponse is the tool text reported for a run that settled on a
// result-carrying reason without a final model text: a custom agent that
// returned no message, or a model whose last message holds only tool requests.
// It says outright that the run succeeded and where its result is, so the
// orchestrator neither mistakes the silence for a failure nor repeats finished
// work.
func noFinalMessageResponse(artifacts int) string {
	switch artifacts {
	case 0:
		return "The task completed, but the agent gave no final message and produced no artifacts."
	case 1:
		return "The task completed, but the agent gave no final message; its result is in the one artifact it produced."
	default:
		return fmt.Sprintf("The task completed, but the agent gave no final message; its result is in the %d artifacts it produced.", artifacts)
	}
}

// settledStatus maps a settled finish reason onto the status vocabulary the
// background-task reports use: "completed" for every reason that carries a
// result, and the reason itself for the rest, which matches the snapshot
// status for failed and aborted runs and stays honest (e.g. "blocked",
// "length") for reasons that have no snapshot-status counterpart.
func settledStatus(reason aix.AgentFinishReason) string {
	if reason.CarriesResult() {
		return string(aix.SnapshotStatusCompleted)
	}
	return string(reason)
}

// interruptedResponse is the tool text reported when a sub-agent interrupted
// for input the orchestrator can never provide.
func interruptedResponse(agentName string) string {
	return fmt.Sprintf(
		"Sub-agent %q interrupted for additional input and could not complete the "+
			"task. Interactive sub-agent interrupts are not currently supported; try "+
			"delegating a more self-contained task.", agentName)
}

// subAgentFailureMessage explains to the orchestrator why a sub-agent turn
// produced no answer. It prefers the structured failure, falls back to whatever
// the agent managed to say before it stopped, and names the finish reason when
// it has neither. last may be nil.
//
// The fallbacks are not decoration. A snapshot the runtime finalizes as
// completed carries no Error even when its finish reason says the turn failed,
// so for a background task the agent's last message is often the only account
// of what happened, and dropping it leaves the model holding a placeholder it
// cannot act on.
func subAgentFailureMessage(reason aix.AgentFinishReason, err *status.Error, last *ai.Message) string {
	if err != nil && err.Message != "" {
		return err.Message
	}
	if reason == "" {
		return "Unknown sub-agent failure."
	}
	msg := fmt.Sprintf("the turn ended as %q without completing the task", reason)
	if text := messageText(last); text != "" {
		msg += "; the agent's last message was: " + text
	}
	return msg + "."
}

// runSubAgent runs the agent one-shot with the task. With no history the
// invocation starts a fresh session; with history the messages ride as
// client-managed init state ([aix.WithState]), which callers forward only to
// client-managed agents. With detach set, the input asks the sub-agent
// runtime to move the work to the background immediately: the returned output
// then carries the pending snapshot's ID and [aix.AgentFinishReasonDetached]
// while the sub-agent keeps working. Custom state is json.RawMessage
// throughout ([aix.AgentHandle]) since the sub-agent's State is unknown here.
func runSubAgent(ctx context.Context, agent *aix.AgentHandle, task string, history []*ai.Message, detach bool) (*aix.AgentOutput[json.RawMessage], error) {
	var opts []aix.InvocationOption[json.RawMessage]
	if len(history) > 0 {
		opts = append(opts, aix.WithState(&aix.SessionState[json.RawMessage]{Messages: history}))
	}
	return agent.Run(ctx, &aix.AgentInput{Detach: detach, Message: ai.NewUserTextMessage(task)}, opts...)
}

// isClientManaged reports whether the agent owns its state on the client (no
// session store), which is the only case that accepts seeded init state.
//
// Unknown or absent agent metadata is treated as not client-managed. That is
// the safe default: it avoids seeding init state into an agent that might
// reject it. This is intentionally stricter than the JS middleware, which
// forwards history unless state management is explicitly "server"; for
// genkit-defined agents the metadata is always set, so the two agree in
// practice.
func isClientManaged(agent *aix.AgentHandle) bool {
	meta := agent.Metadata()
	return meta != nil && meta.StateManagement == aix.AgentStateManagementClient
}

// mergeArtifacts namespaces the sub-agent's artifacts by invocation ID, tags
// them with their source, and merges them into the active session. It is a no-op
// when there is no active session.
func mergeArtifacts(ctx context.Context, source, invocationID string, arts []*aix.Artifact) {
	store := aix.ArtifactStoreFromContext(ctx)
	if store == nil {
		return
	}
	namespaced := make([]*aix.Artifact, 0, len(arts))
	for _, a := range arts {
		md := make(map[string]any, len(a.Metadata)+2)
		maps.Copy(md, a.Metadata)
		md["source"] = source
		md["invocationId"] = invocationID
		namespaced = append(namespaced, &aix.Artifact{
			Name:     invocationID + "/" + a.Name,
			Parts:    a.Parts,
			Metadata: md,
		})
	}
	store.AddArtifacts(namespaced...)
}

// delegatedArtifacts builds the tool-result artifact list, including content
// only under the inline strategy.
func delegatedArtifacts(invocationID string, arts []*aix.Artifact, strategy ArtifactStrategy) []delegatedArtifact {
	out := make([]delegatedArtifact, 0, len(arts))
	for _, a := range arts {
		da := delegatedArtifact{Name: invocationID + "/" + a.Name}
		if strategy == ArtifactStrategyInline {
			da.Content = artifactText(a)
		}
		out = append(out, da)
	}
	return out
}

// prefix resolves the delegation tool-name prefix, defaulting to "delegate_to".
func (a *Agents) prefix() string {
	if a.ToolPrefix == nil {
		return defaultToolPrefix
	}
	return *a.ToolPrefix
}

// taskToolNames are the names of the shared background-task tools for one
// [Agents] configuration.
type taskToolNames struct{ check, wait, abort string }

// all returns the names in a fixed order, for collision checking.
func (n taskToolNames) all() []string { return []string{n.check, n.wait, n.abort} }

// backgroundToolNames returns the names of the shared background-task tools
// for this configuration. They are the bare well-known names by default and
// prefixed with an explicitly set [Agents.ToolPrefix], so two Async instances
// with distinct prefixes can coexist on one generate call without colliding
// on the shared names (the default delegate_to prefix is a delegation verb,
// not an instance namespace, so it is deliberately not applied here, and a
// nil prefix is therefore the same as an empty one).
func (a *Agents) backgroundToolNames() taskToolNames {
	prefix := ""
	if a.ToolPrefix != nil {
		prefix = *a.ToolPrefix
	}
	return taskToolNames{
		check: makeToolName(prefix, checkBackgroundTasksToolName),
		wait:  makeToolName(prefix, waitBackgroundTasksToolName),
		abort: makeToolName(prefix, abortBackgroundTasksToolName),
	}
}

// strategy resolves the artifact strategy, defaulting to inline.
func (a *Agents) strategy() ArtifactStrategy {
	if a.ArtifactStrategy == ArtifactStrategySession {
		return ArtifactStrategySession
	}
	return ArtifactStrategyInline
}

// makeToolName builds a delegation tool name from the prefix and agent name. An
// empty prefix yields the bare agent name.
func makeToolName(prefix, agentName string) string {
	if prefix == "" {
		return agentName
	}
	return prefix + "_" + agentName
}

// buildInstructions renders the <sub-agents> system prompt block. g may be
// nil (e.g. outside an agent/Generate context), in which case only configured
// descriptions are used. With [Agents.Async] set, the block also explains
// background delegation and names this configuration's background-task tools.
func (a *Agents) buildInstructions(g *genkit.Genkit) string {
	prefix := a.prefix()
	var b strings.Builder
	b.WriteString("<sub-agents>\n")
	b.WriteString("You can delegate tasks to specialized sub-agents using their delegation tools:\n")
	for _, ref := range a.Agents {
		desc := ref.Description
		if desc == "" && g != nil {
			desc = discoverDescription(g, ref.Name)
		}
		if desc == "" {
			desc = "No description available."
		}
		fmt.Fprintf(&b, "  - %s: %s\n", makeToolName(prefix, ref.Name), desc)
	}
	b.WriteString("\n")
	b.WriteString("When a task is better handled by a specialized agent, delegate it using the ")
	b.WriteString("appropriate tool. Provide a clear, self-contained task description.\n")
	if a.Async {
		names := a.backgroundToolNames()
		b.WriteString("\n")
		b.WriteString("Delegations can run in the background: set \"background\": true on a ")
		b.WriteString("delegation tool call to get a taskId back immediately while the ")
		b.WriteString("sub-agent keeps working. Continue with other work, then collect ")
		b.WriteString("results with " + names.check + " (returns current ")
		b.WriteString("status without waiting) or " + names.wait + " ")
		b.WriteString("(blocks until the tasks settle). Use " + names.abort + " to stop ")
		b.WriteString("tasks whose results are no longer needed. Background tasks keep ")
		b.WriteString("running across turns, and task IDs from earlier tool results stay ")
		b.WriteString("valid: check them before delegating the same work again.\n")
	}
	b.WriteString("</sub-agents>")
	return b.String()
}

// discoverDescription returns the agent's description from its action
// descriptor, falling back to the backing prompt's description, or "" if none.
// The keys are built with [api.KeyFromName] rather than by hand: a prompt
// registers under "executable-prompt", not "prompt", so a literal key silently
// finds nothing.
func discoverDescription(g *genkit.Genkit, name string) string {
	for _, key := range []string{
		api.KeyFromName(api.ActionTypeAgent, name),
		api.KeyFromName(api.ActionTypeExecutablePrompt, name),
	} {
		if action := genkit.LookupAction(g, key); action != nil {
			if d := action.Desc().Description; d != "" {
				return d
			}
		}
	}
	return ""
}

// recentTextHistory returns up to n of the most recent user/model messages,
// each reduced to its non-empty text parts. Tool and tool-request parts are
// dropped: a model message mid-tool-loop can carry a toolRequest part with no
// matching response, which would confuse the sub-agent model. Returns nil when
// n <= 0.
func recentTextHistory(msgs []*ai.Message, n int) []*ai.Message {
	if n <= 0 {
		return nil
	}
	var filtered []*ai.Message
	for _, m := range msgs {
		if m == nil || (m.Role != ai.RoleUser && m.Role != ai.RoleModel) {
			continue
		}
		var parts []*ai.Part
		for _, p := range m.Content {
			if p != nil && p.IsText() && p.Text != "" {
				parts = append(parts, ai.NewTextPart(p.Text))
			}
		}
		if len(parts) > 0 {
			filtered = append(filtered, &ai.Message{Role: m.Role, Content: parts})
		}
	}
	if len(filtered) > n {
		filtered = filtered[len(filtered)-n:]
	}
	return filtered
}

// namedArtifacts returns the artifacts that have a non-empty name.
func namedArtifacts(arts []*aix.Artifact) []*aix.Artifact {
	out := make([]*aix.Artifact, 0, len(arts))
	for _, a := range arts {
		if a != nil && a.Name != "" {
			out = append(out, a)
		}
	}
	return out
}

// messageText joins a message's non-empty text parts with newlines.
func messageText(m *ai.Message) string {
	if m == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range m.Content {
		if p != nil && p.IsText() && p.Text != "" {
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(p.Text)
		}
	}
	return b.String()
}
