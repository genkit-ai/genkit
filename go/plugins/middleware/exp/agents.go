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
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
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

// resolveAgent looks the agent up by name through g. Resolution goes through
// the Genkit instance (the sanctioned path for third-party middleware) rather
// than the registry directly.
func resolveAgent(g *genkit.Genkit, ref aix.AgentRef) (api.BidiAction, error) {
	if g == nil {
		return nil, fmt.Errorf("no Genkit instance on the context (the agents middleware must run within genkit.Generate or a genkit-defined agent)")
	}
	action := genkit.LookupAction(g, "/agent/"+ref.Name)
	if action == nil {
		return nil, fmt.Errorf("agent %q not found in registry", ref.Name)
	}
	agent, ok := action.(api.BidiAction)
	if !ok {
		return nil, fmt.Errorf("%q is registered but is not an agent", ref.Name)
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
	Agents []aix.AgentRef `json:"agents,omitempty"`
	// ToolPrefix is the prefix for generated delegation tool names. A nil value
	// defaults to "delegate_to" (tools become delegate_to_<agent>); a pointer to
	// the empty string uses bare agent names.
	ToolPrefix *string `json:"toolPrefix,omitempty"`
	// MaxDelegations caps the number of sub-agent delegations per generate call,
	// preventing runaway delegation loops. 0 means unlimited.
	MaxDelegations int `json:"maxDelegations,omitempty"`
	// HistoryLength is the number of recent user/model messages forwarded to a
	// sub-agent as context. 0 means only the task description is sent. History is
	// forwarded only to client-managed sub-agents (those without a session
	// store); server-managed sub-agents receive only the task.
	HistoryLength int `json:"historyLength,omitempty"`
	// ArtifactStrategy controls how sub-agent artifacts are surfaced. Defaults to
	// ArtifactStrategyInline.
	ArtifactStrategy ArtifactStrategy `json:"artifactStrategy,omitempty"`
}

func (a *Agents) Name() string { return provider + "/agents" }

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
func (a *Agents) New(ctx context.Context) (*ai.Hooks, error) {
	if len(a.Agents) == 0 {
		return nil, fmt.Errorf("agents middleware requires at least one agent in the \"agents\" option")
	}
	for _, ref := range a.Agents {
		if ref.Name == "" {
			return nil, fmt.Errorf("agents middleware: every agent reference must have a name")
		}
	}

	prefix := a.prefix()
	st := &agentsState{}

	tools := make([]ai.Tool, 0, len(a.Agents))
	for _, ref := range a.Agents {
		desc := ref.Description
		if desc == "" {
			desc = fmt.Sprintf("Delegates a task to the %q sub-agent.", ref.Name)
		}
		tools = append(tools, aix.NewTool(makeToolName(prefix, ref.Name), desc, a.delegate(ref, st)))
	}

	wrapGenerate := func(ctx context.Context, params *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
		// Capture the latest messages for optional history forwarding. The
		// delegation count is intentionally not reset here: this hook runs on
		// every tool-loop turn, but the count must accumulate across the whole
		// generate call (it starts at 0 when New allocates st).
		st.mu.Lock()
		st.conversation = params.Request.Messages
		st.mu.Unlock()

		instructions := buildAgentsInstructions(genkit.FromContext(ctx), a.Agents, prefix)
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
	Task string `json:"task" jsonschema:"description=A clear, self-contained description of the task to delegate."`
}

// delegationResult is the output of a delegation tool.
type delegationResult struct {
	// Response is the sub-agent's text response.
	Response string `json:"response"`
	// Artifacts are the sub-agent's artifacts. Content is populated only under
	// ArtifactStrategyInline.
	Artifacts []delegatedArtifact `json:"artifacts,omitempty"`
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
		// Guard rail: enforce the delegation cap and reserve this delegation's
		// number, atomically, before doing any work.
		st.mu.Lock()
		if a.MaxDelegations > 0 && st.delegations >= a.MaxDelegations {
			st.mu.Unlock()
			return delegationResult{Response: fmt.Sprintf(
				"Delegation limit reached (%d). Complete the task using information already gathered.",
				a.MaxDelegations)}, nil
		}
		st.delegations++
		invocationNum := st.delegations
		history := recentTextHistory(st.conversation, a.HistoryLength)
		st.mu.Unlock()

		agent, err := resolveAgent(genkit.FromContext(ctx), ref)
		if err != nil {
			return delegationResult{Response: "Error: " + err.Error()}, nil
		}

		// History rides in client-managed init state, which server-managed
		// agents reject; forward it only to client-managed sub-agents.
		if len(history) > 0 && !isClientManaged(agent) {
			history = nil
		}

		out, err := runSubAgent(ctx, agent, in.Task, history)
		if err != nil {
			// The agent runtime resolves failures and interrupts gracefully (see
			// below), so this only fires for exceptions outside that handling
			// (e.g. a rejected init payload). Surface it as tool output.
			return delegationResult{Response: fmt.Sprintf("Error calling agent %q: %v", ref.Name, err)}, nil
		}

		switch out.FinishReason {
		case aix.AgentFinishReasonInterrupted:
			// Reported as text, not propagated: there is no stateful sub-agent
			// runtime to resume into, so the orchestrator could never satisfy it.
			return delegationResult{Response: fmt.Sprintf(
				"Sub-agent %q interrupted for additional input and could not complete the "+
					"task. Interactive sub-agent interrupts are not currently supported; try "+
					"delegating a more self-contained task.", ref.Name)}, nil
		case aix.AgentFinishReasonFailed:
			msg := "Unknown sub-agent failure."
			if out.Error != nil && out.Error.Message != "" {
				msg = out.Error.Message
			}
			return delegationResult{Response: fmt.Sprintf("Error calling agent %q: %s", ref.Name, msg)}, nil
		}

		result := delegationResult{Response: messageText(out.Message)}
		if result.Response == "" {
			result.Response = "(no response)"
		}

		subArtifacts := namedArtifacts(out.Artifacts)
		if len(subArtifacts) > 0 {
			invocationID := fmt.Sprintf("%s_%d", ref.Name, invocationNum)
			// Merge into the parent session under both strategies (no-op if there
			// is no active session, e.g. a plain genkit.Generate call).
			mergeArtifacts(ctx, ref.Name, invocationID, subArtifacts)
			result.Artifacts = delegatedArtifacts(invocationID, subArtifacts, a.strategy())
		}
		return result, nil
	}
}

// runSubAgent runs the agent one-shot with the task. Agents are bidi actions,
// so this always goes through RunBidiJSON: with no history the init is empty (a
// fresh one-shot session); with history it carries the messages as client-
// managed init state, which callers forward only to client-managed agents. The
// output is decoded with json.RawMessage as the custom-state type since the
// sub-agent's State is unknown here.
func runSubAgent(ctx context.Context, agent api.BidiAction, task string, history []*ai.Message) (*aix.AgentOutput[json.RawMessage], error) {
	inputJSON, err := json.Marshal(&aix.AgentInput{Message: ai.NewUserTextMessage(task)})
	if err != nil {
		return nil, err
	}

	var initJSON json.RawMessage
	if len(history) > 0 {
		initJSON, err = json.Marshal(aix.AgentInit[json.RawMessage]{
			State: &aix.SessionState[json.RawMessage]{Messages: history},
		})
		if err != nil {
			return nil, err
		}
	}

	res, err := agent.RunBidiJSON(ctx, inputJSON, nil, &api.BidiJSONOptions{Init: initJSON})
	if err != nil {
		return nil, err
	}

	var out aix.AgentOutput[json.RawMessage]
	if err := json.Unmarshal(res.Result, &out); err != nil {
		return nil, err
	}
	return &out, nil
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
func isClientManaged(agent api.BidiAction) bool {
	meta := agent.Desc().Metadata
	if meta == nil {
		return false
	}
	switch m := meta["agent"].(type) {
	case aix.AgentMetadata:
		return m.StateManagement == aix.AgentStateManagementClient
	case *aix.AgentMetadata:
		return m != nil && m.StateManagement == aix.AgentStateManagementClient
	case map[string]any:
		s, _ := m["stateManagement"].(string)
		return aix.AgentStateManagement(s) == aix.AgentStateManagementClient
	default:
		return false
	}
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
		for k, v := range a.Metadata {
			md[k] = v
		}
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
// descriptions are used.
func buildAgentsInstructions(g *genkit.Genkit, refs []aix.AgentRef, prefix string) string {
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
