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

// Agent conformance test runner (Go harness).
//
// Reads the shared spec from tests/specs/agent.yaml and executes each test
// case against harness-provided agent implementations. The spec is shared
// across language implementations (JS, Go, ...) to ensure cross-language
// compatibility of the Agent abstraction at the wire-protocol level. See
// docs/agents-conformance-testing.md for the full spec format reference and
// harness requirements, and js/ai/tests/agents_spec_test.ts for the
// reference (JS) harness this one mirrors.
//
// The harness lives in the external [package exp_test] (not [package exp]) so
// it exercises only the public API and the production in-memory store
// (localstore.InMemorySessionStore), matching what the JS harness does.
package exp_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/goccy/go-yaml"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/internal/registry"
)

// specPath is relative to this package directory (go/ai/exp).
const specPath = "../../../tests/specs/agent.yaml"

// customState is the single session-state type used by every conformance
// agent. A free-form map lets the custom-state agents manipulate arbitrary
// JSON ({counter, status, ...}) while the prompt agents simply ignore it,
// so one concrete Agent[State] type serves the whole (dynamically-typed)
// spec — mirroring the JS harness's use of a single dynamic agent type.
type customState = map[string]any

// ---------------------------------------------------------------------------
// Spec types
// ---------------------------------------------------------------------------

type specSuite struct {
	Tests []specTest `yaml:"tests"`
}

type specTest struct {
	Name        string           `yaml:"name"`
	Description string           `yaml:"description"`
	Agent       string           `yaml:"agent"`
	Steps       []map[string]any `yaml:"steps"`
}

// ---------------------------------------------------------------------------
// Tool input/output schemas
// ---------------------------------------------------------------------------

type interruptIn struct {
	Query string `json:"query"`
}
type interruptOut struct {
	Answer string `json:"answer"`
}
type restartIn struct {
	Action string `json:"action"`
}
type restartOut struct {
	Result string `json:"result"`
}

// ---------------------------------------------------------------------------
// Programmable model
// ---------------------------------------------------------------------------

// programmableModel is a model whose per-call response behavior is set per
// `send` step via setHandler. It mirrors the JS defineProgrammableModel
// helper: modelResponses[i] is returned for the i-th generate call and
// streamChunks[i] (if present) is emitted before it.
type programmableModel struct {
	mu      sync.Mutex
	handler func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error)
}

func (pm *programmableModel) setHandler(h func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error)) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.handler = h
}

func (pm *programmableModel) generate(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	pm.mu.Lock()
	h := pm.handler
	pm.mu.Unlock()
	if h == nil {
		return nil, fmt.Errorf("programmableModel: no handler set for this step")
	}
	return h(ctx, req, cb)
}

// ---------------------------------------------------------------------------
// Harness setup
// ---------------------------------------------------------------------------

type harness struct {
	pm     *programmableModel
	agents map[string]*exp.Agent[customState]
	// stores holds the in-memory store for each server-managed agent, keyed
	// by agent name, so the getSnapshotData/abort/waitUntilCompleted steps can
	// resolve snapshots directly (the public, local-caller path).
	stores map[string]*localstore.InMemorySessionStore[customState]
}

func setupHarness(t *testing.T) *harness {
	t.Helper()
	reg := registry.New()
	ai.ConfigureFormats(reg)

	pm := &programmableModel{}
	ai.DefineModel(reg, "programmableModel",
		&ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true}},
		pm.generate)
	ai.DefineGenerateAction(context.Background(), reg)

	// --- Tools ---

	testTool := ai.DefineTool(reg, "testTool", "A simple test tool",
		func(tc *ai.ToolContext, _ struct{}) (string, error) {
			return "tool called", nil
		})

	// interruptTool always pauses the turn, returning the tool request to the
	// client for external resolution (resume.respond).
	interruptTool := ai.DefineTool(reg, "interruptTool", "An interrupt tool",
		func(tc *ai.ToolContext, _ interruptIn) (interruptOut, error) {
			return interruptOut{}, tc.Interrupt(&ai.InterruptOptions{})
		})

	// restartTool interrupts on first call and succeeds when restarted with
	// resumed metadata (resume.restart).
	restartTool := ai.DefineTool(reg, "restartTool", "A tool that requires confirmation before executing",
		func(tc *ai.ToolContext, in restartIn) (restartOut, error) {
			if tc.Resumed == nil {
				return restartOut{}, tc.Interrupt(&ai.InterruptOptions{
					Metadata: map[string]any{"requiresConfirmation": true},
				})
			}
			return restartOut{Result: "confirmed: " + in.Action}, nil
		})

	h := &harness{
		pm:     pm,
		agents: map[string]*exp.Agent[customState]{},
		stores: map[string]*localstore.InMemorySessionStore[customState]{},
	}

	// newStore makes a fresh store for a server-managed agent and records it.
	newStore := func(name string) exp.AgentOption[customState] {
		s := localstore.NewInMemorySessionStore[customState]()
		h.stores[name] = s
		return exp.WithSessionStore[customState](s)
	}

	cfg := map[string]any{"temperature": 1}
	model := ai.WithModelName("programmableModel")

	// --- Prompt-backed agents ---

	h.agents["promptAgent"] = exp.DefineAgent[customState](reg, "promptAgent",
		exp.InlinePrompt{model, ai.WithConfig(cfg)})

	h.agents["promptAgentWithStore"] = exp.DefineAgent[customState](reg, "promptAgentWithStore",
		exp.InlinePrompt{model, ai.WithConfig(cfg)}, newStore("promptAgentWithStore"))

	h.agents["promptAgentWithTools"] = exp.DefineAgent[customState](reg, "promptAgentWithTools",
		exp.InlinePrompt{model, ai.WithConfig(cfg), ai.WithTools(testTool)})

	h.agents["promptAgentWithInterrupt"] = exp.DefineAgent[customState](reg, "promptAgentWithInterrupt",
		exp.InlinePrompt{model, ai.WithConfig(cfg), ai.WithTools(interruptTool)},
		newStore("promptAgentWithInterrupt"))

	h.agents["promptAgentWithRestartTool"] = exp.DefineAgent[customState](reg, "promptAgentWithRestartTool",
		exp.InlinePrompt{model, ai.WithConfig(cfg), ai.WithTools(restartTool)},
		newStore("promptAgentWithRestartTool"))

	// --- Custom agents ---

	// customAgentBlocking: server-managed, blocks until its context is
	// cancelled (abort). Used for abort-while-pending tests.
	h.agents["customAgentBlocking"] = exp.DefineCustomAgent(reg, "customAgentBlocking",
		func(ctx context.Context, _ exp.Responder, sess *exp.SessionRunner[customState]) (*exp.AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, _ *exp.AgentInput) (*exp.TurnResult, error) {
				<-ctx.Done()
				return nil, ctx.Err()
			}); err != nil {
				return nil, err
			}
			return &exp.AgentResult{Message: ai.NewModelTextMessage("unblocked")}, nil
		}, newStore("customAgentBlocking"))

	// customAgentFailing: server-managed, fails during processing. Used for
	// detach + background failure tests.
	h.agents["customAgentFailing"] = exp.DefineCustomAgent(reg, "customAgentFailing",
		func(ctx context.Context, _ exp.Responder, sess *exp.SessionRunner[customState]) (*exp.AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, _ *exp.AgentInput) (*exp.TurnResult, error) {
				return nil, errors.New("intentional failure")
			}); err != nil {
				return nil, err
			}
			return &exp.AgentResult{Message: ai.NewModelTextMessage("unreachable")}, nil
		}, newStore("customAgentFailing"))

	// customAgentWithArtifacts: client-managed, streams and dedupes artifacts.
	h.agents["customAgentWithArtifacts"] = exp.DefineCustomAgent(reg, "customAgentWithArtifacts",
		func(ctx context.Context, resp exp.Responder, sess *exp.SessionRunner[customState]) (*exp.AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, _ *exp.AgentInput) (*exp.TurnResult, error) {
				resp.SendArtifact(&exp.Artifact{Name: "doc1", Parts: []*ai.Part{ai.NewTextPart("v1")}})
				resp.SendArtifact(&exp.Artifact{Name: "doc1", Parts: []*ai.Part{ai.NewTextPart("v2")}})
				resp.SendArtifact(&exp.Artifact{Name: "doc2", Parts: []*ai.Part{ai.NewTextPart("other")}})
				return nil, nil
			}); err != nil {
				return nil, err
			}
			return &exp.AgentResult{
				Artifacts: sess.Artifacts(),
				Message:   ai.NewModelTextMessage("done"),
			}, nil
		})

	// customAgentWithCustomState: client-managed, increments custom.counter.
	h.agents["customAgentWithCustomState"] = exp.DefineCustomAgent(reg, "customAgentWithCustomState",
		counterAgentFunc())

	// customAgentWithMultiCustomState: client-managed, three sequential
	// custom-state updates in one turn (to exercise the customPatch contract).
	h.agents["customAgentWithMultiCustomState"] = exp.DefineCustomAgent(reg, "customAgentWithMultiCustomState",
		func(ctx context.Context, _ exp.Responder, sess *exp.SessionRunner[customState]) (*exp.AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, _ *exp.AgentInput) (*exp.TurnResult, error) {
				sess.UpdateCustom(func(customState) customState {
					return customState{"counter": float64(1), "status": "working"}
				})
				sess.UpdateCustom(func(s customState) customState {
					out := cloneCustom(s)
					out["counter"] = float64(2)
					return out
				})
				sess.UpdateCustom(func(s customState) customState {
					out := cloneCustom(s)
					out["status"] = "done"
					return out
				})
				return nil, nil
			}); err != nil {
				return nil, err
			}
			return &exp.AgentResult{Message: ai.NewModelTextMessage("done")}, nil
		})

	// customAgentWithArtifactsStore: server-managed, adds a numbered artifact
	// per invocation based on existing count.
	h.agents["customAgentWithArtifactsStore"] = exp.DefineCustomAgent(reg, "customAgentWithArtifactsStore",
		func(ctx context.Context, resp exp.Responder, sess *exp.SessionRunner[customState]) (*exp.AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, _ *exp.AgentInput) (*exp.TurnResult, error) {
				count := len(sess.Artifacts()) + 1
				resp.SendArtifact(&exp.Artifact{
					Name:  fmt.Sprintf("doc%d", count),
					Parts: []*ai.Part{ai.NewTextPart(fmt.Sprintf("content%d", count))},
				})
				return nil, nil
			}); err != nil {
				return nil, err
			}
			return &exp.AgentResult{
				Artifacts: sess.Artifacts(),
				Message:   ai.NewModelTextMessage("done"),
			}, nil
		}, newStore("customAgentWithArtifactsStore"))

	// customAgentWithCustomStateStore: server-managed counter agent.
	h.agents["customAgentWithCustomStateStore"] = exp.DefineCustomAgent(reg, "customAgentWithCustomStateStore",
		counterAgentFunc(), newStore("customAgentWithCustomStateStore"))

	return h
}

// counterAgentFunc returns an agent func that increments custom.counter by 1
// each turn (default 0 -> 1), used by both the client- and server-managed
// custom-state agents.
func counterAgentFunc() exp.AgentFunc[customState] {
	return func(ctx context.Context, _ exp.Responder, sess *exp.SessionRunner[customState]) (*exp.AgentResult, error) {
		if err := sess.Run(ctx, func(ctx context.Context, _ *exp.AgentInput) (*exp.TurnResult, error) {
			sess.UpdateCustom(func(s customState) customState {
				counter := float64(0)
				if s != nil {
					if c, ok := s["counter"].(float64); ok {
						counter = c
					}
				}
				return customState{"counter": counter + 1}
			})
			return nil, nil
		}); err != nil {
			return nil, err
		}
		return &exp.AgentResult{Message: ai.NewModelTextMessage("done")}, nil
	}
}

func cloneCustom(s customState) customState {
	out := make(customState, len(s)+1)
	for k, v := range s {
		out[k] = v
	}
	return out
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------

func TestAgentConformance(t *testing.T) {
	data, err := os.ReadFile(specPath)
	if err != nil {
		t.Fatalf("read spec %s: %v", specPath, err)
	}
	var suite specSuite
	if err := yaml.Unmarshal(data, &suite); err != nil {
		t.Fatalf("parse spec: %v", err)
	}
	if len(suite.Tests) == 0 {
		t.Fatal("spec contains no tests")
	}

	for _, tc := range suite.Tests {
		t.Run(tc.Name, func(t *testing.T) {
			h := setupHarness(t)
			agent, ok := h.agents[tc.Agent]
			if !ok {
				t.Fatalf("unknown agent %q in test %q", tc.Agent, tc.Name)
			}
			store := h.stores[tc.Agent] // nil for client-managed agents

			captures := map[string]any{}
			for i, step := range tc.Steps {
				stepType, _ := step["type"].(string)
				label := fmt.Sprintf("step[%d] (%s)", i, stepType)
				switch stepType {
				case "send":
					h.executeSend(t, label, agent, step, captures)
				case "getSnapshotData":
					executeGetSnapshotData(t, label, store, step, captures)
				case "abort":
					executeAbort(t, label, agent, store, step, captures)
				case "waitUntilCompleted":
					executeWaitUntilCompleted(t, label, store, step, captures)
				default:
					t.Fatalf("%s: unknown step type %q", label, stepType)
				}
				if t.Failed() {
					// Stop at the first failing step: later steps usually
					// depend on captures the failed step would have produced.
					return
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// send
// ---------------------------------------------------------------------------

func (h *harness) executeSend(t *testing.T, label string, agent *exp.Agent[customState], step map[string]any, captures map[string]any) {
	t.Helper()
	resolved := resolveTemplates(t, label, step, captures)

	// Program the model for this step.
	if _, hasResp := resolved["modelResponses"]; hasResp {
		var modelResponses []*ai.ModelResponse
		jsonConvert(t, label, resolved["modelResponses"], &modelResponses)
		var streamChunks [][]*ai.ModelResponseChunk
		if sc, ok := resolved["streamChunks"]; ok {
			jsonConvert(t, label, sc, &streamChunks)
		}
		counter := 0
		h.pm.setHandler(func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			i := counter
			counter++
			if cb != nil && i < len(streamChunks) {
				for _, chunk := range streamChunks[i] {
					if err := cb(ctx, chunk); err != nil {
						return nil, err
					}
				}
			}
			if i < len(modelResponses) {
				// Echo the request back on the response, as every real model
				// does. The prompt agent's tool loop relies on response.Request
				// to thread intermediate tool-request/response messages into
				// session history (modelResp.History()); without it the loop
				// falls back to recording only the final message.
				resp := modelResponses[i]
				resp.Request = req
				return resp, nil
			}
			return nil, fmt.Errorf("programmableModel: no response for generate call %d", i)
		})
	} else {
		h.pm.setHandler(nil)
	}

	// Build invocation options from init.
	var opts []exp.InvocationOption[customState]
	if initRaw, ok := resolved["init"].(map[string]any); ok {
		if sid := asString(initRaw["snapshotId"]); sid != "" {
			opts = append(opts, exp.WithSnapshotID[customState](sid))
		}
		if sess := asString(initRaw["sessionId"]); sess != "" {
			opts = append(opts, exp.WithSessionID[customState](sess))
		}
		if st, ok := initRaw["state"]; ok {
			var state exp.SessionState[customState]
			jsonConvert(t, label, st, &state)
			opts = append(opts, exp.WithState(&state))
		}
	}

	// Parse inputs.
	var inputs []*exp.AgentInput
	if in, ok := resolved["inputs"]; ok {
		jsonConvert(t, label, in, &inputs)
	}

	// Guard against a hung invocation (a regression could leave a turn
	// blocked forever): a real conformance run completes in milliseconds.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var (
		output        *exp.AgentOutput[customState]
		invocationErr error
		chunks        []*exp.AgentStreamChunk
	)

	conn, err := agent.Connect(ctx, opts...)
	if err != nil {
		invocationErr = err
	} else {
		// Drain the stream concurrently with sending so the agent's responder
		// never blocks on a full stream buffer while inputs are still being
		// sent (matches the JS harness's concurrent send/stream).
		done := make(chan struct{})
		go func() {
			defer close(done)
			for chunk, rerr := range conn.Receive() {
				if rerr != nil {
					return
				}
				chunks = append(chunks, chunk)
			}
		}()
		for _, input := range inputs {
			// Send errors are not fatal: once an invocation resolves (e.g. a
			// failed turn) further sends race it; the outcome is on Output.
			_ = conn.Send(input)
		}
		_ = conn.Close()
		<-done
		output, invocationErr = conn.Output()
	}

	// expectError: the turn is expected to throw (API misuse) rather than
	// resolve with a graceful finishReason:'failed' output. Assert the error's
	// status (exactly) and message (substring), then stop.
	if ee, ok := resolved["expectError"].(map[string]any); ok {
		assertThrownError(t, label, invocationErr, ee)
		return
	}

	// --- Assertions ---
	assertChunks(t, label, chunks, resolved["expectChunks"])
	if eo, ok := resolved["expectOutput"].(map[string]any); ok {
		assertOutput(t, label, output, invocationErr, eo)
	} else if invocationErr != nil {
		t.Fatalf("%s: invocation failed: %v", label, invocationErr)
	}

	// --- Captures (read from the raw step; these are plain names) ---
	if output != nil {
		if name, ok := step["captureSnapshotId"].(string); ok {
			if output.SnapshotID == "" {
				t.Fatalf("%s: captureSnapshotId %q requested but output has no snapshotId", label, name)
			}
			captures[name] = output.SnapshotID
		}
		if name, ok := step["captureState"].(string); ok {
			if output.State == nil {
				t.Fatalf("%s: captureState %q requested but output has no state", label, name)
			}
			captures[name] = canon(t, output.State)
		}
		if name, ok := step["captureSessionId"].(string); ok {
			if output.State == nil || output.State.SessionID == "" {
				t.Fatalf("%s: captureSessionId %q requested but output has no state.sessionId", label, name)
			}
			captures[name] = output.State.SessionID
		}
	} else {
		for _, key := range []string{"captureSnapshotId", "captureState", "captureSessionId"} {
			if name, ok := step[key].(string); ok {
				t.Fatalf("%s: %s %q requested but invocation produced no output (err: %v)", label, key, name, invocationErr)
			}
		}
	}
}

// assertChunks performs the semi-strict ordered chunk comparison: same length
// and order, with type-aware matching per chunk (turnEnd asserts presence and,
// when specified, finishReason; modelChunk/artifact/customPatch use contains).
func assertChunks(t *testing.T, label string, actual []*exp.AgentStreamChunk, expectRaw any) {
	t.Helper()
	if expectRaw == nil {
		return
	}
	expected, ok := canon(t, expectRaw).([]any)
	if !ok {
		t.Fatalf("%s: expectChunks is not a list", label)
	}
	actualCanon := make([]any, len(actual))
	for i, c := range actual {
		actualCanon[i] = canon(t, c)
	}
	if len(actualCanon) != len(expected) {
		t.Errorf("%s: expected %d chunks, got %d.\n  expected: %s\n  actual:   %s",
			label, len(expected), len(actualCanon), mustJSON(expected), mustJSON(actualCanon))
		return
	}
	for i := range expected {
		if err := matchChunk(actualCanon[i], expected[i]); err != nil {
			t.Errorf("%s: chunk[%d]: %v\n  expected: %s\n  actual:   %s",
				label, i, err, mustJSON(expected[i]), mustJSON(actualCanon[i]))
		}
	}
}

func matchChunk(actual, expected any) error {
	em, ok := expected.(map[string]any)
	if !ok {
		return matchContains(actual, expected, "chunk")
	}
	am, _ := actual.(map[string]any)
	switch {
	case hasKey(em, "turnEnd"):
		te, ok := am["turnEnd"].(map[string]any)
		if !ok {
			return fmt.Errorf("expected turnEnd chunk")
		}
		// snapshotId is dynamic; only finishReason is asserted when specified.
		if exTE, _ := em["turnEnd"].(map[string]any); exTE != nil {
			if fr, ok := exTE["finishReason"]; ok {
				if !reflect.DeepEqual(te["finishReason"], fr) {
					return fmt.Errorf("turnEnd.finishReason: want %v, got %v", fr, te["finishReason"])
				}
			}
		}
		return nil
	case hasKey(em, "modelChunk"):
		return matchContains(am["modelChunk"], em["modelChunk"], "modelChunk")
	case hasKey(em, "artifact"):
		return matchContains(am["artifact"], em["artifact"], "artifact")
	case hasKey(em, "customPatch"):
		return matchContains(am["customPatch"], em["customPatch"], "customPatch")
	default:
		return matchContains(actual, expected, "chunk")
	}
}

func assertOutput(t *testing.T, label string, output *exp.AgentOutput[customState], invocationErr error, expect map[string]any) {
	t.Helper()

	wantsFailure := false
	if fr, ok := expect["finishReason"].(string); ok && fr == "failed" {
		wantsFailure = true
	}
	if _, ok := expect["errorContains"]; ok {
		wantsFailure = true
	}

	if invocationErr != nil {
		// The implementation returned a transport-level error instead of a
		// graceful AgentOutput. Surface this explicitly: a spec that expects a
		// graceful failure (finishReason=failed) is not satisfied by a thrown
		// error, so this fails — and the message makes the mismatch obvious.
		status := "<none>"
		var ge *core.GenkitError
		if errors.As(invocationErr, &ge) {
			status = string(ge.Status)
		}
		if wantsFailure {
			t.Errorf("%s: expected a graceful failed AgentOutput, but the invocation returned a transport-level error "+
				"(status=%s): %v", label, status, invocationErr)
		} else {
			t.Fatalf("%s: invocation failed: %v", label, invocationErr)
		}
		return
	}
	if output == nil {
		t.Fatalf("%s: no output and no error", label)
	}

	out := canon(t, output).(map[string]any)

	if msg, ok := expect["message"]; ok {
		if err := matchContains(out["message"], canon(t, msg), "output.message"); err != nil {
			t.Errorf("%s: %v", label, err)
		}
	}
	if b, _ := expect["hasSnapshotId"].(bool); b {
		if s, _ := out["snapshotId"].(string); s == "" {
			t.Errorf("%s: expected output.snapshotId to be a non-empty string", label)
		}
	}
	if b, _ := expect["hasSessionId"].(bool); b {
		if !hasNonEmptySessionID(out["state"]) {
			t.Errorf("%s: expected output.state.sessionId to be a non-empty string, got: %s", label, mustJSON(out["state"]))
		}
	}
	if sc, ok := expect["stateContains"]; ok {
		if err := matchContains(out["state"], canon(t, sc), "output.state"); err != nil {
			t.Errorf("%s: %v", label, err)
		}
	}
	if ac, ok := expect["artifactsContain"]; ok {
		assertArtifactsContain(t, label, out["artifacts"], canon(t, ac))
	}
	if fr, ok := expect["finishReason"]; ok {
		if !reflect.DeepEqual(out["finishReason"], fr) {
			t.Errorf("%s: output.finishReason: want %v, got %v", label, fr, out["finishReason"])
		}
	}
	if ec, ok := expect["errorContains"].(map[string]any); ok {
		assertErrorContains(t, label, "output.error", out["error"], ec)
	}
}

func assertArtifactsContain(t *testing.T, label string, actual any, expected any) {
	t.Helper()
	arts, ok := actual.([]any)
	if !ok {
		t.Errorf("%s: expected output.artifacts to be a list, got %s", label, mustJSON(actual))
		return
	}
	expArts, _ := expected.([]any)
	for _, ea := range expArts {
		em, _ := ea.(map[string]any)
		name, _ := em["name"].(string)
		var found map[string]any
		for _, a := range arts {
			if am, ok := a.(map[string]any); ok {
				if n, _ := am["name"].(string); n == name {
					found = am
					break
				}
			}
		}
		if found == nil {
			t.Errorf("%s: expected artifact %q not found in output.artifacts %s", label, name, mustJSON(actual))
			continue
		}
		if err := matchContains(found, ea, fmt.Sprintf("artifact(%s)", name)); err != nil {
			t.Errorf("%s: %v", label, err)
		}
	}
}

// assertErrorContains matches a structured error: its presence and its status
// (exactly). The error message is intentionally NOT asserted — wording is
// implementation-specific and need not match across language harnesses; the
// cross-language contract is the error status (category).
func assertErrorContains(t *testing.T, label, path string, actual any, expect map[string]any) {
	t.Helper()
	errObj, ok := actual.(map[string]any)
	if !ok || errObj == nil {
		t.Errorf("%s: expected %s to be present, got %s", label, path, mustJSON(actual))
		return
	}
	if st, ok := expect["status"]; ok {
		if !reflect.DeepEqual(errObj["status"], st) {
			t.Errorf("%s: %s.status: want %v, got %v", label, path, st, errObj["status"])
		}
	}
}

// assertThrownError checks that an API-misuse send threw a transport-level
// error (rather than resolving with a graceful output) and that its status
// matches exactly. The error message is intentionally NOT asserted — wording
// is implementation-specific and need not match across language harnesses.
func assertThrownError(t *testing.T, label string, err error, expect map[string]any) {
	t.Helper()
	if err == nil {
		t.Errorf("%s: expected the turn to throw an error, but it resolved successfully", label)
		return
	}
	if st, ok := expect["status"].(string); ok {
		got := ""
		var ge *core.GenkitError
		if errors.As(err, &ge) {
			got = string(ge.Status)
		}
		if got != st {
			t.Errorf("%s: expectError.status: want %q, got %q (error: %v)", label, st, got, err)
		}
	}
}

// ---------------------------------------------------------------------------
// getSnapshotData
// ---------------------------------------------------------------------------

func executeGetSnapshotData(t *testing.T, label string, store *localstore.InMemorySessionStore[customState], step map[string]any, captures map[string]any) {
	t.Helper()
	if store == nil {
		t.Fatalf("%s: agent is client-managed (no store) — getSnapshotData not applicable", label)
	}
	resolved := resolveTemplates(t, label, step, captures)
	snapID := asString(resolved["snapshotId"])
	sessID := asString(resolved["sessionId"])
	if (snapID == "") == (sessID == "") {
		t.Fatalf("%s: getSnapshotData requires exactly one of snapshotId / sessionId", label)
	}

	ctx := context.Background()
	var (
		snap *exp.SessionSnapshot[customState]
		err  error
	)
	if snapID != "" {
		snap, err = store.GetSnapshot(ctx, snapID)
	} else {
		snap, err = store.GetLatestSnapshot(ctx, sessID)
	}

	if ee, ok := resolved["expectError"].(string); ok {
		if err == nil {
			t.Errorf("%s: expected error containing %q, but getSnapshotData succeeded", label, ee)
		} else if !strings.Contains(err.Error(), ee) {
			t.Errorf("%s: expected error containing %q, got: %v", label, ee, err)
		}
		return
	}
	if err != nil {
		t.Fatalf("%s: getSnapshotData: %v", label, err)
	}
	if snap == nil {
		t.Fatalf("%s: snapshot not found", label)
	}
	if es, ok := resolved["expectSnapshot"].(map[string]any); ok {
		assertSnapshot(t, label, snap, es)
	}
}

// ---------------------------------------------------------------------------
// abort
// ---------------------------------------------------------------------------

func executeAbort(t *testing.T, label string, agent *exp.Agent[customState], store *localstore.InMemorySessionStore[customState], step map[string]any, captures map[string]any) {
	t.Helper()
	if store == nil {
		t.Fatalf("%s: agent is client-managed (no store) — abort not applicable", label)
	}
	resolved := resolveTemplates(t, label, step, captures)
	snapID := asString(resolved["snapshotId"])
	if snapID == "" {
		t.Fatalf("%s: abort requires snapshotId", label)
	}

	ctx := context.Background()

	// Capture the status before aborting so we can assert expectPreviousStatus.
	// (Agent.Abort returns the status *after* the attempt, while the spec
	// asserts the *previous* status — semantically what JS's agent.abort()
	// returns — so the harness reads it directly first.)
	var previous exp.SnapshotStatus
	if snap, _ := store.GetSnapshot(ctx, snapID); snap != nil {
		previous = normalizeStatus(snap.Status)
	}
	if _, err := agent.Abort(ctx, snapID); err != nil {
		t.Fatalf("%s: abort: %v", label, err)
	}

	if raw, present := step["expectPreviousStatus"]; present {
		want := ""
		if raw != nil { // YAML `~` means "expect absent/empty".
			want = asString(resolved["expectPreviousStatus"])
		}
		if string(previous) != want {
			t.Errorf("%s: expectPreviousStatus: want %q, got %q", label, want, previous)
		}
	}
}

// ---------------------------------------------------------------------------
// waitUntilCompleted
// ---------------------------------------------------------------------------

func executeWaitUntilCompleted(t *testing.T, label string, store *localstore.InMemorySessionStore[customState], step map[string]any, captures map[string]any) {
	t.Helper()
	if store == nil {
		t.Fatalf("%s: agent is client-managed (no store) — waitUntilCompleted not applicable", label)
	}
	resolved := resolveTemplates(t, label, step, captures)
	snapID := asString(resolved["snapshotId"])
	if snapID == "" {
		t.Fatalf("%s: waitUntilCompleted requires snapshotId", label)
	}
	timeout := 5 * time.Second
	if tm, ok := resolved["timeoutMs"]; ok {
		timeout = time.Duration(toFloat(tm)) * time.Millisecond
	}

	terminal := map[exp.SnapshotStatus]bool{
		exp.SnapshotStatusCompleted: true,
		exp.SnapshotStatusFailed:    true,
		exp.SnapshotStatusAborted:   true,
	}

	ctx := context.Background()
	deadline := time.Now().Add(timeout)
	var snap *exp.SessionSnapshot[customState]
	for time.Now().Before(deadline) {
		s, err := store.GetSnapshot(ctx, snapID)
		if err != nil {
			t.Fatalf("%s: getSnapshot while polling: %v", label, err)
		}
		if s != nil && terminal[normalizeStatus(s.Status)] {
			snap = s
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if snap == nil {
		t.Fatalf("%s: snapshot %s did not reach a terminal status within %v", label, snapID, timeout)
	}
	if es, ok := resolved["expectSnapshot"].(map[string]any); ok {
		assertSnapshot(t, label, snap, es)
	}
}

// ---------------------------------------------------------------------------
// Snapshot assertions
// ---------------------------------------------------------------------------

func assertSnapshot(t *testing.T, label string, snap *exp.SessionSnapshot[customState], expect map[string]any) {
	t.Helper()
	snap.Status = normalizeStatus(snap.Status)
	actual := canon(t, snap).(map[string]any)

	if pid, ok := expect["parentId"]; ok {
		if !reflect.DeepEqual(actual["parentId"], pid) {
			t.Errorf("%s: snapshot.parentId: want %v, got %v", label, pid, actual["parentId"])
		}
	}
	if st, ok := expect["status"]; ok {
		if !reflect.DeepEqual(actual["status"], st) {
			t.Errorf("%s: snapshot.status: want %v, got %v", label, st, actual["status"])
		}
	}
	if fr, ok := expect["finishReason"]; ok {
		if !reflect.DeepEqual(actual["finishReason"], fr) {
			t.Errorf("%s: snapshot.finishReason: want %v, got %v", label, fr, actual["finishReason"])
		}
	}
	if b, _ := expect["hasSessionId"].(bool); b {
		if !hasNonEmptySessionID(actual["state"]) {
			t.Errorf("%s: expected snapshot.state.sessionId to be a non-empty string, got: %s", label, mustJSON(actual["state"]))
		}
	}
	if sc, ok := expect["stateContains"]; ok {
		if err := matchContains(actual["state"], canon(t, sc), "snapshot.state"); err != nil {
			t.Errorf("%s: %v", label, err)
		}
	}
	if ec, ok := expect["errorContains"].(map[string]any); ok {
		assertErrorContains(t, label, "snapshot.error", actual["error"], ec)
	}
}

// ---------------------------------------------------------------------------
// "Contains" matchers
// ---------------------------------------------------------------------------

// matchContains asserts that actual contains all fields specified in expected.
// Objects are matched key-by-key (extra actual keys ignored); arrays are
// matched as an ordered subsequence (each expected item appears in order, not
// necessarily contiguously); scalars must be deep-equal. All values must be in
// canonical JSON form (string keys, float64 numbers).
func matchContains(actual, expected any, path string) error {
	if expected == nil {
		return nil
	}
	switch exp := expected.(type) {
	case []any:
		act, ok := actual.([]any)
		if !ok {
			return fmt.Errorf("%s: expected array, got %T", path, actual)
		}
		return matchSubsequence(act, exp, path)
	case map[string]any:
		act, ok := actual.(map[string]any)
		if !ok {
			return fmt.Errorf("%s: expected object, got %T", path, actual)
		}
		for k, v := range exp {
			if err := matchContains(act[k], v, path+"."+k); err != nil {
				return err
			}
		}
		return nil
	default:
		if !reflect.DeepEqual(actual, expected) {
			return fmt.Errorf("%s: want %v (%T), got %v (%T)", path, expected, expected, actual, actual)
		}
		return nil
	}
}

func matchSubsequence(actual, expected []any, path string) error {
	idx := 0
	for i, want := range expected {
		found := false
		for idx < len(actual) {
			if matchContains(actual[idx], want, fmt.Sprintf("%s[%d]", path, idx)) == nil {
				found = true
				idx++
				break
			}
			idx++
		}
		if !found {
			return fmt.Errorf("%s: expected item %d not found in order: %s", path, i, mustJSON(want))
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Template resolution
// ---------------------------------------------------------------------------

var (
	fullTemplateRe   = regexp.MustCompile(`^\{\{(\w+)\}\}$`)
	inlineTemplateRe = regexp.MustCompile(`\{\{(\w+)\}\}`)
)

// resolveTemplates recursively replaces `{{name}}` references with previously
// captured values. A value that is exactly `{{name}}` is replaced by the
// captured value (which may be a non-string, e.g. a captured state object);
// inline occurrences are string-substituted.
func resolveTemplates(t *testing.T, label string, v any, captures map[string]any) map[string]any {
	t.Helper()
	resolved, err := resolveValue(v, captures)
	if err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	m, _ := resolved.(map[string]any)
	return m
}

func resolveValue(v any, captures map[string]any) (any, error) {
	switch x := v.(type) {
	case string:
		if m := fullTemplateRe.FindStringSubmatch(x); m != nil {
			val, ok := captures[m[1]]
			if !ok {
				return nil, fmt.Errorf("template reference {{%s}} not found in captures", m[1])
			}
			return val, nil
		}
		var missing string
		out := inlineTemplateRe.ReplaceAllStringFunc(x, func(s string) string {
			name := inlineTemplateRe.FindStringSubmatch(s)[1]
			val, ok := captures[name]
			if !ok {
				missing = name
				return s
			}
			if str, ok := val.(string); ok {
				return str
			}
			return string(mustJSON(val))
		})
		if missing != "" {
			return nil, fmt.Errorf("template reference {{%s}} not found in captures", missing)
		}
		return out, nil
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, val := range x {
			r, err := resolveValue(val, captures)
			if err != nil {
				return nil, err
			}
			out[k] = r
		}
		return out, nil
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			r, err := resolveValue(e, captures)
			if err != nil {
				return nil, err
			}
			out[i] = r
		}
		return out, nil
	default:
		return v, nil
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// canon converts any Go value (struct or yaml-decoded) into canonical JSON
// form (string-keyed maps, float64 numbers) via a JSON round-trip, so the
// expected (yaml) and actual (Go struct) sides compare apples-to-apples.
func canon(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("canon: marshal: %v", err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("canon: unmarshal: %v", err)
	}
	return out
}

func jsonConvert(t *testing.T, label string, src, dst any) {
	t.Helper()
	b, err := json.Marshal(src)
	if err != nil {
		t.Fatalf("%s: marshal %T: %v", label, src, err)
	}
	if err := json.Unmarshal(b, dst); err != nil {
		t.Fatalf("%s: unmarshal into %T: %v", label, dst, err)
	}
}

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return []byte(fmt.Sprintf("%v", v))
	}
	return b
}

func hasKey(m map[string]any, k string) bool {
	_, ok := m[k]
	return ok
}

func hasNonEmptySessionID(state any) bool {
	m, ok := state.(map[string]any)
	if !ok {
		return false
	}
	s, _ := m["sessionId"].(string)
	return s != ""
}

func normalizeStatus(s exp.SnapshotStatus) exp.SnapshotStatus {
	if s == "" {
		return exp.SnapshotStatusCompleted
	}
	return s
}

func asString(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	default:
		return fmt.Sprint(x)
	}
}

func toFloat(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int:
		return float64(x)
	case int64:
		return float64(x)
	case uint64:
		return float64(x)
	default:
		return 0
	}
}
