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

// requireGenkit classifies the absence of a Genkit instance on the context as
// a failed precondition: a wiring gap in how the middleware runs, not anything
// the caller sent. Both delegation and background-task reads resolve through
// it.
func requireGenkit(g *genkit.Genkit) error {
	if g == nil {
		return status.Errorf(status.ErrFailedPrecondition,
			"no Genkit instance on the context (the agents middleware must run within genkit.Generate or a genkit-defined agent)")
	}
	return nil
}

// resolveAgent looks the agent up by name through g and returns its handle.
// Resolution goes through the Genkit instance (the sanctioned path for
// third-party middleware) rather than the registry directly; the handle
// carries the agent's companion actions and capability metadata along with
// the run surface.
func resolveAgent(g *genkit.Genkit, ref aix.AgentRef) (*aix.AgentHandle, error) {
	if err := requireGenkit(g); err != nil {
		return nil, err
	}
	return genkitx.LookupAgent(g, ref.Name)
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
// wait_for_background_tasks tools. The pending snapshot (heartbeated while the
// worker lives, finalized in place with the cumulative state when the work
// settles) is the durable record, and task IDs are self-contained
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
	// the empty string uses bare agent names.
	ToolPrefix *string `json:"toolPrefix,omitempty" jsonschema_description:"Prefix for generated delegation tool names. Defaults to \"delegate_to\", so tools become delegate_to_<agent>. Set it to the empty string to use bare agent names."`
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
	// returns a task ID immediately, and two shared tools are added:
	// check_background_tasks (non-blocking status and results) and
	// wait_for_background_tasks (blocks until the listed tasks settle).
	// Background delegation requires server-managed sub-agents whose stores
	// implement [aix.SnapshotSubscriber].
	Async bool `json:"async,omitempty" jsonschema_description:"Enables background delegation: delegation tools accept a \"background\" flag, and the check_background_tasks / wait_for_background_tasks tools are added. Background delegation requires server-managed sub-agents whose session stores support detach."`
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
	// conversation is the latest request message list, captured each turn for
	// optional history forwarding.
	conversation []*ai.Message
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
	st := &agentsState{}

	tools := make([]ai.Tool, 0, len(a.Agents)+2)
	for _, ref := range a.Agents {
		desc := ref.Description
		if desc == "" {
			desc = fmt.Sprintf("Delegates a task to the %q sub-agent.", ref.Name)
		}
		name := makeToolName(prefix, ref.Name)
		// The async variant carries the extra "background" input flag, so the
		// two modes need distinct input schemas (tool schemas are static).
		if a.Async {
			tools = append(tools, aix.NewTool(name, desc, a.delegateAsync(ref, st)))
		} else {
			tools = append(tools, aix.NewTool(name, desc, a.delegate(ref, st)))
		}
	}
	if a.Async {
		tools = append(tools, a.backgroundTaskTools()...)
	}

	wrapGenerate := func(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
		// Capture the latest messages for optional history forwarding. The
		// delegation count is intentionally not reset here: this hook runs on
		// every tool-loop turn, but the count must accumulate across the whole
		// generate call (it starts at 0 when New allocates st).
		st.mu.Lock()
		st.conversation = params.Request.Messages
		st.mu.Unlock()

		instructions := buildAgentsInstructions(genkit.FromContext(ctx), a.Agents, prefix, a.Async)
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
	// TaskID is the background task handle ("<agent>:<snapshotId>") when the
	// delegation was started with background=true; empty otherwise. It is the
	// input to check_background_tasks / wait_for_background_tasks.
	TaskID string `json:"taskId,omitempty"`
	// Status is "pending" when a background delegation was started; empty for
	// synchronous delegations.
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

// runDelegation is the synchronous delegation body, shared by the plain
// delegation tool and the async-enabled variant when the model does not
// request background execution.
func (a *Agents) runDelegation(ctx context.Context, ref aix.AgentRef, st *agentsState, task string) (delegationResult, error) {
	invocationNum, history, ok := a.reserveDelegation(st)
	if !ok {
		logger.Warn(ctx, "delegation refused, limit reached", "agent", ref.Name, "limit", a.MaxDelegations)
		return delegationLimitResult(a.MaxDelegations), nil
	}

	agent, err := resolveAgent(genkit.FromContext(ctx), ref)
	if err != nil {
		logger.Warn(ctx, "sub-agent resolution failed", "agent", ref.Name, "error", err)
		return delegationResult{Response: "Error: " + err.Error()}, nil
	}

	// History rides in client-managed init state, which server-managed
	// agents reject; forward it only to client-managed sub-agents.
	if len(history) > 0 && !isClientManaged(agent) {
		history = nil
	}

	logger.Debug(ctx, "delegating to sub-agent",
		"agent", ref.Name, "invocation", invocationNum, "historyMessages", len(history))
	start := time.Now()
	out, err := runSubAgent(ctx, agent, task, history, false)
	if err != nil {
		// The agent runtime resolves failures and interrupts gracefully (see
		// foldDelegationOutput), so this only fires for exceptions outside
		// that handling (e.g. a rejected init payload). Surface it as tool
		// output.
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
// invocation number, atomically, before any work happens. It also snapshots the
// conversation history for optional forwarding. ok is false when the cap is
// reached.
func (a *Agents) reserveDelegation(st *agentsState) (invocationNum int, history []*ai.Message, ok bool) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if a.MaxDelegations > 0 && st.delegations >= a.MaxDelegations {
		return 0, nil, false
	}
	st.delegations++
	return st.delegations, recentTextHistory(st.conversation, a.HistoryLength), true
}

// releaseDelegation returns a reserved slot for a delegation the sub-agent
// rejected before any work ran, so the cap only counts delegations that did
// something. Number reuse by a later delegation is harmless: a rejected
// delegation produced no artifacts under its invocation ID.
func (a *Agents) releaseDelegation(st *agentsState) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.delegations--
}

// delegationLimitResult is the refusal returned once MaxDelegations is
// exhausted.
func delegationLimitResult(limit int) delegationResult {
	return delegationResult{Response: fmt.Sprintf(
		"Delegation limit reached (%d). Complete the task using information already gathered.", limit)}
}

// foldDelegationOutput turns a settled sub-agent output into a delegation tool
// result: interrupts and failures become explanatory text, and artifacts are
// merged into the parent session under invocationID and surfaced per the
// configured strategy.
func (a *Agents) foldDelegationOutput(ctx context.Context, ref aix.AgentRef, out *aix.AgentOutput[json.RawMessage], invocationID string) delegationResult {
	switch out.FinishReason {
	case aix.AgentFinishReasonInterrupted:
		// Reported as text, not propagated: there is no stateful sub-agent
		// runtime to resume into, so the orchestrator could never satisfy it.
		return delegationResult{Response: interruptedResponse(ref.Name)}
	case aix.AgentFinishReasonFailed:
		return delegationResult{Response: fmt.Sprintf(
			"Error calling agent %q: %s", ref.Name, subAgentFailureMessage(out.Error))}
	}

	result := delegationResult{Response: messageText(out.Message)}
	if result.Response == "" {
		result.Response = "(no response)"
	}

	subArtifacts := namedArtifacts(out.Artifacts)
	if len(subArtifacts) > 0 {
		// Merge into the parent session under both strategies (no-op if there
		// is no active session, e.g. a plain genkit.Generate call).
		mergeArtifacts(ctx, ref.Name, invocationID, subArtifacts)
		result.Artifacts = delegatedArtifacts(invocationID, subArtifacts, a.strategy())
	}
	return result
}

// interruptedResponse is the tool text reported when a sub-agent interrupted
// for input the orchestrator can never provide.
func interruptedResponse(agentName string) string {
	return fmt.Sprintf(
		"Sub-agent %q interrupted for additional input and could not complete the "+
			"task. Interactive sub-agent interrupts are not currently supported; try "+
			"delegating a more self-contained task.", agentName)
}

// subAgentFailureMessage extracts a human-readable message from a sub-agent's
// structured failure.
func subAgentFailureMessage(err *status.Error) string {
	if err != nil && err.Message != "" {
		return err.Message
	}
	return "Unknown sub-agent failure."
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

// buildAgentsInstructions renders the <sub-agents> system prompt block. g may be
// nil (e.g. outside an agent/Generate context), in which case only configured
// descriptions are used. With async set, the block also explains background
// delegation and the background-task tools.
func buildAgentsInstructions(g *genkit.Genkit, refs []aix.AgentRef, prefix string, async bool) string {
	var b strings.Builder
	b.WriteString("<sub-agents>\n")
	b.WriteString("You can delegate tasks to specialized sub-agents using their delegation tools:\n")
	for _, ref := range refs {
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
	if async {
		b.WriteString("\n")
		b.WriteString("Delegations can run in the background: set \"background\": true on a ")
		b.WriteString("delegation tool call to get a taskId back immediately while the ")
		b.WriteString("sub-agent keeps working. Continue with other work, then collect ")
		b.WriteString("results with " + checkBackgroundTasksToolName + " (returns current ")
		b.WriteString("status without waiting) or " + waitBackgroundTasksToolName + " ")
		b.WriteString("(blocks until the tasks settle). Background tasks keep running ")
		b.WriteString("across turns, and task IDs from earlier tool results stay valid: ")
		b.WriteString("check them before delegating the same work again.\n")
	}
	b.WriteString("</sub-agents>")
	return b.String()
}

// discoverDescription returns the agent's description from its action
// descriptor, falling back to the backing prompt's description, or "" if none.
func discoverDescription(g *genkit.Genkit, name string) string {
	for _, key := range []string{"/agent/" + name, "/prompt/" + name} {
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
