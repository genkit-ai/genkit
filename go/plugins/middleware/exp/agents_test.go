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
	"errors"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
)

// toolModel defines a model with full tool/multiturn support backed by fn.
func toolModel(t *testing.T, g *genkit.Genkit, name string, fn ai.ModelFunc) ai.Model {
	t.Helper()
	return genkit.DefineModel(g, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true},
	}, fn)
}

// textResp is a model response carrying a single model text message.
func textResp(req *ai.ModelRequest, text string) *ai.ModelResponse {
	return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage(text)}
}

// toolReqResp is a model response that issues the given tool calls.
func toolReqResp(req *ai.ModelRequest, calls ...*ai.ToolRequest) *ai.ModelResponse {
	parts := make([]*ai.Part, 0, len(calls))
	for _, c := range calls {
		parts = append(parts, ai.NewToolRequestPart(c))
	}
	return &ai.ModelResponse{Request: req, Message: &ai.Message{Role: ai.RoleModel, Content: parts}}
}

// systemText concatenates the text parts of a system message.
func systemText(m *ai.Message) string {
	var b strings.Builder
	for _, p := range m.Content {
		if p != nil && p.IsText() {
			b.WriteString(p.Text)
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// hasToolResponse reports whether any message carries a tool response.
func hasToolResponse(msgs []*ai.Message) bool {
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.IsToolResponse() {
				return true
			}
		}
	}
	return false
}

// delegateOnceModel calls toolName once with the given task, then returns
// "done" after it sees any tool response.
func delegateOnceModel(t *testing.T, g *genkit.Genkit, name, toolName, task string) ai.Model {
	return toolModel(t, g, name, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if hasToolResponse(req.Messages) {
			return textResp(req, "done"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: toolName, Input: map[string]any{"task": task}}), nil
	})
}

// decodeDelegation re-decodes a tool response output into a delegationResult,
// tolerating either the raw struct or a JSON-normalized map.
func decodeDelegation(t *testing.T, v any) delegationResult {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal tool output: %v", err)
	}
	var dr delegationResult
	if err := json.Unmarshal(b, &dr); err != nil {
		t.Fatalf("unmarshal delegationResult: %v", err)
	}
	return dr
}

// delegationResponses collects every delegation tool response for toolName.
func delegationResponses(t *testing.T, msgs []*ai.Message, toolName string) []delegationResult {
	t.Helper()
	var out []delegationResult
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.IsToolResponse() && p.ToolResponse != nil && p.ToolResponse.Name == toolName {
				out = append(out, decodeDelegation(t, p.ToolResponse.Output))
			}
		}
	}
	return out
}

func TestAgentsValidation(t *testing.T) {
	if _, err := (&Agents{}).New(ctx); err == nil {
		t.Error("expected error when no agents are configured")
	}
	if _, err := (&Agents{Agents: []aix.AgentRef{{Name: ""}}}).New(ctx); err == nil {
		t.Error("expected error when an agent reference has no name")
	}
	if _, err := (&Agents{Agents: []aix.AgentRef{{Name: "ok"}}}).New(ctx); err != nil {
		t.Errorf("unexpected error for a valid config: %v", err)
	}
}

func TestAgentsInjectsSystemPrompt(t *testing.T) {
	g := newTestGenkit(t)

	// researcher's description is auto-discovered from its action descriptor.
	genkitx.DefineAgent[any](g, "researcher",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/researcher", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "researched"), nil
		}))},
		aix.WithDescription[any]("Searches the web and summarizes findings."),
	)

	var captured []*ai.Message
	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		captured = req.Messages
		return textResp(req, "ok"), nil
	})

	mw := &Agents{Agents: []aix.AgentRef{
		{Name: "researcher"},                            // discovered description
		{Name: "coder", Description: "Writes Go code."}, // explicit override (agent need not exist for the listing)
	}}
	if _, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("hi"), ai.WithUse(mw)); err != nil {
		t.Fatal(err)
	}

	sys := findSystem(captured)
	if sys == nil {
		t.Fatalf("expected a system message; got %v", captured)
	}
	text := systemText(sys)
	for _, want := range []string{
		"delegate_to_researcher: Searches the web and summarizes findings.",
		"delegate_to_coder: Writes Go code.",
		"<sub-agents>",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("system prompt missing %q; got:\n%s", want, text)
		}
	}
}

func TestAgentsDelegationRunsSubAgent(t *testing.T) {
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "researcher",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/researcher", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "research complete"), nil
		}))},
	)

	orch := delegateOnceModel(t, g, "test/orch", "delegate_to_researcher", "look into X")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("research X"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}

	got := delegationResponses(t, resp.History(), "delegate_to_researcher")
	if len(got) != 1 {
		t.Fatalf("expected 1 delegation response, got %d", len(got))
	}
	if got[0].Response != "research complete" {
		t.Errorf("delegation response = %q, want %q", got[0].Response, "research complete")
	}
}

func TestAgentsUnknownAgentReportsError(t *testing.T) {
	g := newTestGenkit(t)

	orch := delegateOnceModel(t, g, "test/orch", "delegate_to_ghost", "do it")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "ghost"}}} // never defined

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	got := delegationResponses(t, resp.History(), "delegate_to_ghost")
	if len(got) != 1 || !strings.Contains(got[0].Response, "not found") {
		t.Fatalf("expected a 'not found' delegation response, got %+v", got)
	}
}

func TestAgentsToolPrefix(t *testing.T) {
	bare := ""
	custom := "ask"
	cases := []struct {
		name   string
		prefix *string
		want   string
	}{
		{"default", nil, "delegate_to_researcher"},
		{"custom", &custom, "ask_researcher"},
		{"bare", &bare, "researcher"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := newTestGenkit(t)
			genkitx.DefineAgent[any](g, "researcher",
				aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/sub-"+tc.name, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
					return textResp(req, "ok"), nil
				}))},
			)
			orch := delegateOnceModel(t, g, "test/orch-"+tc.name, tc.want, "task")
			mw := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, ToolPrefix: tc.prefix}

			resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
			if err != nil {
				t.Fatal(err)
			}
			if got := delegationResponses(t, resp.History(), tc.want); len(got) != 1 {
				t.Fatalf("expected delegation via tool %q, got %d responses", tc.want, len(got))
			}
		})
	}
}

func TestAgentsMaxDelegations(t *testing.T) {
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "researcher",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/researcher", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "did work"), nil
		}))},
	)

	// Issue two delegations in a single turn; with MaxDelegations=1 exactly one
	// must be refused.
	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if hasToolResponse(req.Messages) {
			return textResp(req, "done"), nil
		}
		return toolReqResp(req,
			&ai.ToolRequest{Name: "delegate_to_researcher", Input: map[string]any{"task": "a"}},
			&ai.ToolRequest{Name: "delegate_to_researcher", Input: map[string]any{"task": "b"}},
		), nil
	})

	mw := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, MaxDelegations: 1}
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}

	got := delegationResponses(t, resp.History(), "delegate_to_researcher")
	if len(got) != 2 {
		t.Fatalf("expected 2 delegation responses, got %d", len(got))
	}
	var real, limited int
	for _, r := range got {
		switch {
		case r.Response == "did work":
			real++
		case strings.Contains(r.Response, "Delegation limit reached"):
			limited++
		default:
			t.Errorf("unexpected delegation response: %q", r.Response)
		}
	}
	if real != 1 || limited != 1 {
		t.Errorf("got real=%d limited=%d, want 1 and 1", real, limited)
	}
}

func TestAgentsForwardsHistory(t *testing.T) {
	g := newTestGenkit(t)

	// The sub-agent records the messages its model receives.
	var subMessages []*ai.Message
	genkitx.DefineAgent[any](g, "researcher",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/researcher", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			subMessages = req.Messages
			return textResp(req, "noted"), nil
		}))},
	)

	orch := delegateOnceModel(t, g, "test/orch", "delegate_to_researcher", "summarize the discussion")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, HistoryLength: 4}

	_, err := genkit.Generate(ctx, g,
		ai.WithModel(orch),
		ai.WithMessages(
			ai.NewUserTextMessage("the secret code is platypus"),
			ai.NewModelTextMessage("understood"),
		),
		ai.WithPrompt("now delegate"),
		ai.WithUse(mw),
	)
	if err != nil {
		t.Fatal(err)
	}

	var joined strings.Builder
	for _, m := range subMessages {
		joined.WriteString(messageText(m))
		joined.WriteByte('\n')
	}
	if !strings.Contains(joined.String(), "platypus") {
		t.Errorf("sub-agent did not receive forwarded history; saw:\n%s", joined.String())
	}
}

func TestAgentsSubAgentFailureReported(t *testing.T) {
	g := newTestGenkit(t)

	// A custom sub-agent whose turn fails.
	genkitx.DefineCustomAgent[any](g, "researcher",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				return nil, errors.New("kaboom")
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{}, nil
		},
	)

	orch := delegateOnceModel(t, g, "test/orch", "delegate_to_researcher", "go")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	got := delegationResponses(t, resp.History(), "delegate_to_researcher")
	if len(got) != 1 || !strings.Contains(got[0].Response, "Error calling agent") {
		t.Fatalf("expected an error delegation response, got %+v", got)
	}
}

// TestAgentsArtifactStrategies verifies that, run inside an orchestrator agent
// (so a session exists), sub-agent artifacts are merged into the parent session
// under both strategies and that inline includes content while session does not.
func TestAgentsArtifactStrategies(t *testing.T) {
	for _, strategy := range []ArtifactStrategy{ArtifactStrategyInline, ArtifactStrategySession} {
		t.Run(string(strategy), func(t *testing.T) {
			g := newTestGenkit(t)

			// A custom sub-agent that produces an artifact.
			genkitx.DefineCustomAgent[any](g, "writer",
				func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
					err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
						resp.SendArtifact(&aix.Artifact{
							Name:  "report.md",
							Parts: []*ai.Part{ai.NewTextPart("the report body")},
						})
						sess.AddMessages(ai.NewModelTextMessage("wrote the report"))
						return &aix.TurnResult{FinishReason: aix.AgentFinishReasonStop}, nil
					})
					if err != nil {
						return nil, err
					}
					return &aix.AgentResult{
						Message:   ai.NewModelTextMessage("wrote the report"),
						Artifacts: sess.Artifacts(),
					}, nil
				},
			)

			delegating := delegateOnceModel(t, g, "test/orch-model-"+string(strategy), "delegate_to_writer", "write a report")

			// The orchestrator is itself an agent, so the delegation runs within
			// a session that artifacts can merge into. Capture the inner generate
			// history to inspect the delegation tool response.
			var innerHistory []*ai.Message
			orchestrator := genkitx.DefineCustomAgent[any](g, "orchestrator",
				func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
					var last *ai.Message
					err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
						r, err := genkit.Generate(ctx, g,
							ai.WithModel(delegating),
							ai.WithMessages(input.Message),
							ai.WithUse(&Agents{
								Agents:           []aix.AgentRef{{Name: "writer"}},
								ArtifactStrategy: strategy,
							}),
						)
						if err != nil {
							return nil, err
						}
						innerHistory = r.History()
						last = r.Message
						return &aix.TurnResult{FinishReason: aix.AgentFinishReasonStop}, nil
					})
					if err != nil {
						return nil, err
					}
					return &aix.AgentResult{Message: last, Artifacts: sess.Artifacts()}, nil
				},
			)

			out, err := orchestrator.RunText(ctx, "please produce a report")
			if err != nil {
				t.Fatal(err)
			}

			// The sub-agent artifact is merged into the parent session, namespaced.
			if !hasArtifactNamed(out.Artifacts, "writer_1/report.md") {
				t.Errorf("expected merged artifact %q in parent session; got %v", "writer_1/report.md", artifactNames(out.Artifacts))
			}

			// Inline carries content in the tool result; session does not.
			got := delegationResponses(t, innerHistory, "delegate_to_writer")
			if len(got) != 1 || len(got[0].Artifacts) != 1 {
				t.Fatalf("expected 1 delegation response with 1 artifact, got %+v", got)
			}
			content := got[0].Artifacts[0].Content
			if strategy == ArtifactStrategyInline && !strings.Contains(content, "the report body") {
				t.Errorf("inline strategy should include artifact content, got %q", content)
			}
			if strategy == ArtifactStrategySession && content != "" {
				t.Errorf("session strategy should omit artifact content, got %q", content)
			}
		})
	}
}

func TestAgentRefCapturesNameAndDescription(t *testing.T) {
	g := newTestGenkit(t)
	a := genkitx.DefineAgent[any](g, "writer",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/writer", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "x"), nil
		}))},
		aix.WithDescription[any]("Writes things."),
	)

	ref := a.Ref()
	if ref.Name != "writer" {
		t.Errorf("Name = %q, want %q", ref.Name, "writer")
	}
	if ref.Description != "Writes things." {
		t.Errorf("Description = %q, want %q", ref.Description, "Writes things.")
	}
}

func TestAgentsDelegatesViaRef(t *testing.T) {
	g := newTestGenkit(t)
	researcher := genkitx.DefineAgent[any](g, "researcher",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/researcher", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "ref result"), nil
		}))},
	)

	orch := delegateOnceModel(t, g, "test/orch", "delegate_to_researcher", "go")
	mw := &Agents{Agents: []aix.AgentRef{researcher.Ref()}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("research"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	got := delegationResponses(t, resp.History(), "delegate_to_researcher")
	if len(got) != 1 || got[0].Response != "ref result" {
		t.Fatalf("delegation via Ref failed: %+v", got)
	}
}

func TestAgentsRefDescriptionTakesPrecedence(t *testing.T) {
	g := newTestGenkit(t)
	a := genkitx.DefineAgent[any](g, "writer",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/writer", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "x"), nil
		}))},
		aix.WithDescription[any]("Original description."),
	)

	ref := a.Ref()
	ref.Description = "Overridden in config." // user override on top of the instance

	var captured []*ai.Message
	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		captured = req.Messages
		return textResp(req, "ok"), nil
	})

	if _, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("hi"), ai.WithUse(&Agents{Agents: []aix.AgentRef{ref}})); err != nil {
		t.Fatal(err)
	}
	text := systemText(findSystem(captured))
	if !strings.Contains(text, "Overridden in config.") {
		t.Errorf("system prompt missing the override; got:\n%s", text)
	}
	if strings.Contains(text, "Original description.") {
		t.Errorf("override should replace the instance description; got:\n%s", text)
	}
}

// TestAgentsConfigSerialization guards the JSON-dispatch path used by the Dev
// UI: schema inference must not panic, and the config must round-trip.
func TestAgentsConfigSerialization(t *testing.T) {
	_ = ai.NewMiddleware("agents", &Agents{}) // must not panic on schema inference

	prefix := "ask"
	cfg := &Agents{
		Agents:           []aix.AgentRef{{Name: "researcher"}, {Name: "coder", Description: "Writes Go."}},
		ToolPrefix:       &prefix,
		MaxDelegations:   3,
		HistoryLength:    2,
		ArtifactStrategy: ArtifactStrategySession,
	}
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var got Agents
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Agents) != 2 || got.Agents[0].Name != "researcher" || got.Agents[1].Description != "Writes Go." {
		t.Errorf("agents lost in round trip: %+v", got.Agents)
	}
	if got.ToolPrefix == nil || *got.ToolPrefix != "ask" {
		t.Errorf("toolPrefix lost in round trip: %v", got.ToolPrefix)
	}
	if got.ArtifactStrategy != ArtifactStrategySession {
		t.Errorf("artifactStrategy lost in round trip: %q", got.ArtifactStrategy)
	}
}

// statemessages returns the conversation messages from a client-managed agent
// output's state.
func statemessages(out *aix.AgentOutput[any]) []*ai.Message {
	if out == nil || out.State == nil {
		return nil
	}
	return out.State.Messages
}

func hasArtifactNamed(arts []*aix.Artifact, name string) bool {
	for _, a := range arts {
		if a != nil && a.Name == name {
			return true
		}
	}
	return false
}

func artifactNames(arts []*aix.Artifact) []string {
	names := make([]string, 0, len(arts))
	for _, a := range arts {
		if a != nil {
			names = append(names, a.Name)
		}
	}
	return names
}
