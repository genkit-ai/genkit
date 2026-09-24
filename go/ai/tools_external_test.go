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

// Tests for the resumable tool surface of tools.go that drive tools the
// way callers do, through the ai/tool runtime verbs. They live in package
// ai_test rather than in tools_test.go because ai/tool imports ai: an internal
// test file importing it would close an import cycle, so the external test
// package is the escape hatch.
package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/tool"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/registry"
	"github.com/google/go-cmp/cmp"
)

// newToolTestRegistry returns a registry with the formats and generate action
// configured, ready for ai.Generate / ai.GenerateStream.
func newToolTestRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	reg := registry.New()
	ai.ConfigureFormats(reg)
	ai.DefineGenerateAction(context.Background(), reg)
	return reg
}

// defineTestModel builds and registers a model, the two steps
// genkit.DefineModel fuses for an application.
func defineTestModel(reg api.Registry, name string, opts *ai.ModelOptions, fn ai.ModelFunc) ai.Model {
	m := ai.NewModel(name, opts, fn)
	m.Register(reg)
	return m
}

// defineTestTool builds and registers a tool written against ai.ToolContext
// from a function that only wants a context.Context. The adapter is what
// makes these tests double as coverage that the ai/tool verbs work from a
// ToolContext tool: the context they receive is the ToolContext itself.
func defineTestTool[In, Out any](reg api.Registry, name, description string, fn func(context.Context, In) (Out, error)) *ai.ToolAction[In, Out] {
	tl := ai.NewTool(name, description, func(tc *ai.ToolContext, in In) (Out, error) {
		return fn(tc, in)
	})
	tl.Register(reg)
	return tl
}

// defineTestResumableTool builds and registers a resumable tool, the
// two steps genkit.DefineResumableTool fuses for an application.
func defineTestResumableTool[In, Out, Res any](reg api.Registry, name, description string, fn ai.ResumableToolFunc[In, Out, Res], opts ...ai.ToolOption) *ai.ResumableToolAction[In, Out, Res] {
	tl := ai.NewResumableTool(name, description, fn, opts...)
	tl.Register(reg)
	return tl
}

// defineToolThenFinishModel defines "test/model": on the first turn it returns
// reqs (typically tool requests), and once a tool response is in history it
// returns the final text "done". This drives a single tool round per Generate.
func defineToolThenFinishModel(reg *registry.Registry, reqs ...*ai.Part) {
	defineTestModel(reg, "test/model",
		&ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			for _, m := range req.Messages {
				if m.Role == ai.RoleTool {
					return &ai.ModelResponse{
						Request:      req,
						Message:      ai.NewModelTextMessage("done"),
						FinishReason: ai.FinishReasonStop,
					}, nil
				}
			}
			return &ai.ModelResponse{
				Request:      req,
				Message:      &ai.Message{Role: ai.RoleModel, Content: reqs},
				FinishReason: ai.FinishReasonStop,
			}, nil
		})
}

type weatherIn struct {
	City string `json:"city"`
}

// TestTool_AttachParts verifies AttachParts folds extra content into the tool's
// multipart response without changing the function signature, for a tool
// written against ToolContext.
func TestTool_AttachParts(t *testing.T) {
	reg := newToolTestRegistry(t)
	shot := defineTestTool(reg, "screenshot", "takes a screenshot",
		func(ctx context.Context, _ struct{}) (string, error) {
			// A nil part is ignored, so a failed constructor result can be
			// passed without a check.
			tool.AttachParts(ctx, nil, ai.NewMediaPart("image/png", "pngbytes"))
			return "captured", nil
		})

	resp, err := shot.RunRawMultipart(context.Background(), struct{}{})
	if err != nil {
		t.Fatalf("RunRawMultipart: %v", err)
	}
	if resp.Output != "captured" {
		t.Errorf("output = %v, want %q", resp.Output, "captured")
	}
	if len(resp.Content) != 1 || !resp.Content[0].IsMedia() {
		t.Fatalf("expected one attached media part, got %+v", resp.Content)
	}
}

// TestMultipartTool_AttachParts verifies attached parts are appended to the
// content a multipart tool returns itself, rather than replacing it.
func TestMultipartTool_AttachParts(t *testing.T) {
	tl := ai.NewMultipartTool("chart", "charts and annotates",
		func(tc *ai.ToolContext, _ struct{}) (*ai.MultipartToolResponse, error) {
			tool.AttachParts(tc, ai.NewMediaPart("image/png", "annotation"))
			return &ai.MultipartToolResponse{
				Output:  "charted",
				Content: []*ai.Part{ai.NewMediaPart("image/png", "chart")},
			}, nil
		})

	resp, err := tl.RunRawMultipart(context.Background(), struct{}{})
	if err != nil {
		t.Fatalf("RunRawMultipart: %v", err)
	}
	if len(resp.Content) != 2 || resp.Content[0].Text != "chart" || resp.Content[1].Text != "annotation" {
		t.Fatalf("content = %+v, want the returned part followed by the attached one", resp.Content)
	}

	// A multipart function may return no response at all; the attached parts
	// still need somewhere to land.
	silent := ai.NewMultipartTool("silent", "attaches, returns nothing",
		func(tc *ai.ToolContext, _ struct{}) (*ai.MultipartToolResponse, error) {
			tool.AttachParts(tc, ai.NewMediaPart("image/png", "only"))
			return nil, nil
		})
	resp, err = silent.RunRawMultipart(context.Background(), struct{}{})
	if err != nil {
		t.Fatalf("RunRawMultipart: %v", err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "only" {
		t.Fatalf("content = %+v, want the attached part on an empty response", resp.Content)
	}
}

type reportItem struct {
	Name string `json:"name"`
}

type reportOut struct {
	Title string       `json:"title"`
	Items []reportItem `json:"items"`
}

// TestResumableTool_OutputSchemaMatchesNewTool guards against the multipart
// envelope leaking into the tool definition: a resumable tool must
// advertise the same output schema NewTool would for the same Out, including a
// pointer Out (whose zero value is a nil pointer) and a nested struct that
// exercises schema inlining.
func TestResumableTool_OutputSchemaMatchesNewTool(t *testing.T) {
	classic := ai.NewTool("classic", "d",
		func(tc *ai.ToolContext, _ weatherIn) (reportOut, error) { return reportOut{}, nil })
	want := classic.Definition().OutputSchema
	if want == nil {
		t.Fatal("ai.NewTool unexpectedly produced a nil output schema")
	}

	resumable := ai.NewResumableTool("resumable", "d",
		func(ctx context.Context, _ weatherIn, _ *confirmation) (reportOut, error) { return reportOut{}, nil })
	pointer := ai.NewResumableTool("pointer", "d",
		func(ctx context.Context, _ weatherIn, _ *confirmation) (*reportOut, error) { return nil, nil })

	for _, tc := range []struct {
		name string
		got  any
	}{
		{"struct output", resumable.Definition().OutputSchema},
		{"pointer output", pointer.Definition().OutputSchema},
	} {
		if !reflect.DeepEqual(tc.got, want) {
			t.Errorf("%s output schema = %#v\nwant %#v (matching ai.NewTool)", tc.name, tc.got, want)
		}
		props, _ := tc.got.(map[string]any)["properties"].(map[string]any)
		if _, ok := props["title"]; !ok {
			t.Errorf("%s output schema missing the real %q field: %#v", tc.name, "title", tc.got)
		}
		if _, ok := props["content"]; ok {
			t.Errorf("%s output schema leaked the multipart envelope (has %q): %#v", tc.name, "content", tc.got)
		}
	}
}

// TestResumableTool_OutputSchemaSurvivesLookup is the test that matters for
// what the model actually receives: the generate loop resolves tools by name
// out of the registry, so the real output schema has to survive that type
// erasure.
func TestResumableTool_OutputSchemaSurvivesLookup(t *testing.T) {
	reg := newToolTestRegistry(t)

	var gotTools []*ai.ToolDefinition
	defineTestModel(reg, "test/model",
		&ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			gotTools = req.Tools
			return &ai.ModelResponse{
				Request:      req,
				Message:      ai.NewModelTextMessage("done"),
				FinishReason: ai.FinishReasonStop,
			}, nil
		})

	report := defineTestResumableTool(reg, "report", "builds a report",
		func(ctx context.Context, _ weatherIn, _ *confirmation) (reportOut, error) { return reportOut{}, nil })

	if _, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("report"),
		ai.WithTools(report)); err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if len(gotTools) != 1 {
		t.Fatalf("model saw %d tools, want 1", len(gotTools))
	}
	props, _ := gotTools[0].OutputSchema["properties"].(map[string]any)
	if _, ok := props["title"]; !ok {
		t.Errorf("model saw output schema %#v, want the real Out type", gotTools[0].OutputSchema)
	}
	if _, ok := props["content"]; ok {
		t.Errorf("model saw the multipart envelope as the output schema: %#v", gotTools[0].OutputSchema)
	}
}

// TestResumableTool_ResumeSchemaAdvertised pins that a tool advertises the
// schema of its resume type, the one it validates a payload against:
// inferred from Res with its fields widened to accept null, surfaced as the
// definition's "resumeSchema" metadata, and intact after a registry lookup. A
// tool without a resume type advertises the object schema, since the loop
// delivers its resume payload as a map.
func TestResumableTool_ResumeSchemaAdvertised(t *testing.T) {
	reg := newTransferTestRegistry(t)
	transfer, _ := interruptOnce(t, reg)

	want := core.InferSchemaMap(confirmation{})
	want["properties"].(map[string]any)["approved"] = map[string]any{"type": []any{"boolean", "null"}}
	for _, tc := range []struct {
		name string
		tool ai.Tool
	}{
		{"defined", transfer},
		{"looked up", ai.LookupTool(reg, "transfer")},
	} {
		if diff := cmp.Diff(want, tc.tool.Definition().Metadata["resumeSchema"]); diff != "" {
			t.Errorf("%s: resumeSchema mismatch (-want +got):\n%s", tc.name, diff)
		}
	}

	plain := defineTestTool(reg, "plain", "no resume type",
		func(ctx context.Context, _ struct{}) (string, error) { return "", nil })
	if diff := cmp.Diff(map[string]any{"type": "object"}, plain.Definition().Metadata["resumeSchema"]); diff != "" {
		t.Errorf("plain tool resumeSchema mismatch (-want +got):\n%s", diff)
	}
}

// TestResumableTool_OutputSchemaOptions pins that NewResumableTool
// runs the output schema check NewTool runs (tools_test.go covers the option
// itself; both constructors share newTool): with a concrete Out the
// constructor panics, naming itself, rather than advertising a schema that
// disagrees with the type.
func TestResumableTool_OutputSchemaOptions(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for an output schema option with concrete Out")
		}
		err, ok := r.(error)
		if !ok || !strings.Contains(err.Error(), "ai.NewResumableTool") {
			t.Errorf("panic = %v, want it to name ai.NewResumableTool", r)
		}
	}()
	ai.NewResumableTool("t", "d",
		func(ctx context.Context, input any, res *struct{}) (string, error) { return "", nil },
		ai.WithOutputSchemaName("Answer"))
}

// TestTool_SendPartialNoOpWithoutStreaming confirms SendPartial is a safe no-op
// when no streaming callback is wired (here, a direct RunRaw).
func TestTool_SendPartialNoOpWithoutStreaming(t *testing.T) {
	reg := newToolTestRegistry(t)
	tl := defineTestTool(reg, "noop", "streams when it can",
		func(ctx context.Context, _ struct{}) (string, error) {
			tool.SendPartial(ctx, map[string]any{"progress": 50})
			return "ok", nil
		})

	out, err := tl.RunRaw(context.Background(), struct{}{})
	if err != nil {
		t.Fatalf("RunRaw: %v", err)
	}
	if out != "ok" {
		t.Errorf("output = %v, want %q", out, "ok")
	}
}

type transferIn struct {
	Amount float64 `json:"amount"`
}
type transferOut struct {
	Status string `json:"status"`
}
type transferInterrupt struct {
	Reason string  `json:"reason"`
	Amount float64 `json:"amount"`
}
type confirmation struct {
	Approved bool `json:"approved"`
}

// interruptOnce returns a resumable tool that pauses on its first pass and
// records what it saw when it re-executes, plus accessors for those recordings.
func interruptOnce(t *testing.T, reg *registry.Registry) (
	*ai.ResumableToolAction[transferIn, transferOut, confirmation],
	func() (*confirmation, transferIn, any),
) {
	t.Helper()
	var (
		gotResume   *confirmation
		gotInput    transferIn
		gotOriginal any
	)
	tl := defineTestResumableTool(reg, "transfer", "transfers money",
		func(ctx context.Context, in transferIn, res *confirmation) (transferOut, error) {
			if res == nil {
				return transferOut{}, tool.Interrupt(ctx, transferInterrupt{Reason: "large_amount", Amount: in.Amount})
			}
			gotResume, gotInput = res, in
			if orig, ok := tool.OriginalInput[transferIn](ctx); ok {
				gotOriginal = orig
			}
			if !res.Approved {
				return transferOut{Status: "cancelled"}, nil
			}
			return transferOut{Status: "completed"}, nil
		})
	return tl, func() (*confirmation, transferIn, any) { return gotResume, gotInput, gotOriginal }
}

// generateUntilInterrupt runs the first turn and returns the response plus its
// single interrupt part.
func generateUntilInterrupt(t *testing.T, reg *registry.Registry, tl ai.ToolRef) (*ai.ModelResponse, *ai.Part) {
	t.Helper()
	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("transfer 200"),
		ai.WithTools(tl))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	interrupts := resp.Interrupts()
	if len(interrupts) != 1 {
		t.Fatalf("expected 1 interrupt, got %d (finish=%s)", len(interrupts), resp.FinishReason)
	}
	return resp, interrupts[0]
}

// resumeWith continues an interrupted generation with the given restart or
// response parts and returns the final text.
func resumeWith(t *testing.T, reg *registry.Registry, resp *ai.ModelResponse, tl ai.ToolRef, opt ai.GenerateOption) string {
	t.Helper()
	resp2, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithMessages(resp.History()...),
		ai.WithTools(tl),
		opt)
	if err != nil {
		t.Fatalf("resume Generate: %v", err)
	}
	return resp2.Text()
}

// bareConfirmation is confirmation with its field optional, so the resume
// schema inferred from it admits the empty payload a bare restart delivers.
type bareConfirmation struct {
	Approved bool `json:"approved,omitempty"`
}

// interruptOnceBare is interruptOnce for a tool that accepts a bare restart:
// its resume type has no required field, where interruptOnce's has one and
// the loop rejects a bare restart of it (see TestRestart_ResumeDataValidated).
func interruptOnceBare(t *testing.T, reg *registry.Registry) (
	*ai.ResumableToolAction[transferIn, transferOut, bareConfirmation],
	func() *bareConfirmation,
) {
	t.Helper()
	var gotResume *bareConfirmation
	tl := defineTestResumableTool(reg, "transfer", "transfers money",
		func(ctx context.Context, in transferIn, res *bareConfirmation) (transferOut, error) {
			if res == nil {
				return transferOut{}, tool.Interrupt(ctx, nil)
			}
			gotResume = res
			return transferOut{Status: "completed"}, nil
		})
	return tl, func() *bareConfirmation { return gotResume }
}

func newTransferTestRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "transfer", Input: map[string]any{"amount": 200},
	}))
	return reg
}

// claim is Interrupted with a failed claim fatal, for the tests that assert
// what follows the claim.
func claim[In, Out, Res any](t *testing.T, tl *ai.ResumableToolAction[In, Out, Res], part *ai.Part) *ai.InterruptedCall[In, Out, Res] {
	t.Helper()
	call, ok := tl.Interrupted(part)
	if !ok {
		t.Fatalf("%s.Interrupted did not claim the tool's own interrupt", tl.Name())
	}
	return call
}

// TestResumableTool_TypedRestart pins the core interrupt/resume contract:
// the tool interrupts with typed data on the first pass, the caller claims the
// part with Interrupted and reads the input typed, reads the interrupt data
// with ai.InterruptAs, restarts with a typed value, and the value reaches the
// function's *Res parameter on re-execution.
func TestResumableTool_TypedRestart(t *testing.T) {
	reg := newTransferTestRegistry(t)
	transfer, recorded := interruptOnce(t, reg)
	resp, interrupt := generateUntilInterrupt(t, reg, transfer)

	call := claim(t, transfer, interrupt)
	if call.Input.Amount != 200 {
		t.Errorf("call.Input = %+v, want the input decoded from the wire {200}", call.Input)
	}
	if call.Part == nil || !call.Part.IsInterrupt() {
		t.Errorf("call.Part = %+v, want the interrupted part", call.Part)
	}
	meta, ok := ai.InterruptAs[transferInterrupt](call.Part)
	if !ok {
		t.Fatal("InterruptAs failed to decode the typed interrupt data")
	}
	if meta.Reason != "large_amount" || meta.Amount != 200 {
		t.Errorf("interrupt data = %+v, want {large_amount 200}", meta)
	}

	restart := call.Restart(confirmation{Approved: true})
	if got := resumeWith(t, reg, resp, transfer, ai.WithResume(restart)); got != "done" {
		t.Errorf("final text after restart = %q, want %q", got, "done")
	}
	gotResume, _, _ := recorded()
	if gotResume == nil || !gotResume.Approved {
		t.Errorf("resumed tool saw %+v, want Approved=true", gotResume)
	}
}

// TestResumableTool_BareRestart documents what a restart with the zero
// value delivers: the tool re-executes with a non-nil, zero-valued resume
// parameter, which is what makes a bare restart read as approval for tools
// that key on the presence of a resume.
func TestResumableTool_BareRestart(t *testing.T) {
	reg := newTransferTestRegistry(t)
	transfer, recorded := interruptOnce(t, reg)
	resp, interrupt := generateUntilInterrupt(t, reg, transfer)

	restart := claim(t, transfer, interrupt).Restart(confirmation{})
	if got := resumeWith(t, reg, resp, transfer, ai.WithResume(restart)); got != "done" {
		t.Errorf("final text after bare restart = %q, want %q", got, "done")
	}

	gotResume, _, _ := recorded()
	if gotResume == nil {
		t.Fatal("a bare restart must still deliver a non-nil resume parameter")
	}
	if gotResume.Approved {
		t.Errorf("bare restart resume = %+v, want the zero value", *gotResume)
	}
}

// TestResumableTool_RestartWithInput covers the caller revising the
// arguments before approving: the tool re-executes with the new input and can
// still read what it was originally called with.
func TestResumableTool_RestartWithInput(t *testing.T) {
	reg := newTransferTestRegistry(t)
	transfer, recorded := interruptOnce(t, reg)
	resp, interrupt := generateUntilInterrupt(t, reg, transfer)

	restart := claim(t, transfer, interrupt).RestartWithInput(transferIn{Amount: 50}, confirmation{Approved: true})
	if got := resumeWith(t, reg, resp, transfer, ai.WithResume(restart)); got != "done" {
		t.Errorf("final text = %q, want %q", got, "done")
	}

	_, gotInput, gotOriginal := recorded()
	if gotInput.Amount != 50 {
		t.Errorf("re-executed tool saw amount %v, want the new input 50", gotInput.Amount)
	}
	orig, ok := gotOriginal.(transferIn)
	if !ok || orig.Amount != 200 {
		t.Errorf("tool.OriginalInput = %+v (%T), want transferIn{200}", gotOriginal, gotOriginal)
	}
}

// TestResumableTool_RestartWithInputAmendsHistory pins that history
// records the request as a restart with a new input ran it: a second
// interrupt claims the new input, so restarting it runs that input again,
// and the completed call reads the new input in history, while the caller's
// history is left as it was.
func TestResumableTool_RestartWithInputAmendsHistory(t *testing.T) {
	type note struct {
		Note string `json:"note,omitempty"`
	}
	reg := newTransferTestRegistry(t)
	var ran []float64
	transfer := defineTestResumableTool(reg, "transfer", "transfers money",
		func(ctx context.Context, in transferIn, res *note) (transferOut, error) {
			if res == nil || res.Note == "" {
				return transferOut{}, tool.Interrupt(ctx, nil)
			}
			ran = append(ran, in.Amount)
			return transferOut{Status: "completed"}, nil
		})
	generate := func(opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
		return ai.Generate(context.Background(), reg, append([]ai.GenerateOption{
			ai.WithModelName("test/model"), ai.WithTools(transfer),
		}, opts...)...)
	}
	resp, err := generate(ai.WithPrompt("transfer 200"))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	original := resp.History()

	restart := claim(t, transfer, singleInterrupt(t, resp)).RestartWithInput(transferIn{Amount: 50}, note{})
	resp, err = generate(ai.WithMessages(original...), ai.WithResume(restart))
	if !errors.Is(err, status.ErrFailedPrecondition) || resp == nil {
		t.Fatalf("resume = (%v, %v), want the tool to interrupt again", resp, err)
	}
	again := claim(t, transfer, singleInterrupt(t, resp))
	if again.Input.Amount != 50 {
		t.Fatalf("second interrupt claims amount %v, want the new input 50", again.Input.Amount)
	}
	for _, m := range original {
		for _, p := range m.Content {
			if p.IsToolRequest() && fmt.Sprint(p.ToolRequest.Input) != "map[amount:200]" {
				t.Errorf("caller's history request = %v, want it untouched", p.ToolRequest.Input)
			}
		}
	}

	resp, err = generate(ai.WithMessages(resp.History()...), ai.WithResume(again.Restart(note{Note: "ok"})))
	if err != nil {
		t.Fatalf("second resume: %v", err)
	}
	if diff := cmp.Diff([]float64{50}, ran); diff != "" {
		t.Errorf("tool ran with (-want +got):\n%s", diff)
	}
	var recorded any
	for _, m := range resp.History() {
		for _, p := range m.Content {
			if p.IsToolRequest() {
				recorded = p.ToolRequest.Input
			}
		}
	}
	if in, ok := recorded.(transferIn); !ok || in.Amount != 50 {
		t.Errorf("history records input %#v, want transferIn{50}", recorded)
	}
}

// TestResumableTool_Respond resolves an interrupt with a pre-computed
// result instead of re-executing the tool. The output is validated against the
// tool's advertised output schema on the way through, so this also covers the
// schema surviving the registry lookup.
func TestResumableTool_Respond(t *testing.T) {
	reg := newTransferTestRegistry(t)
	transfer, recorded := interruptOnce(t, reg)
	resp, interrupt := generateUntilInterrupt(t, reg, transfer)

	response := claim(t, transfer, interrupt).Respond(transferOut{Status: "manually approved"})
	if got := resumeWith(t, reg, resp, transfer, ai.WithResume(response)); got != "done" {
		t.Errorf("final text after respond = %q, want %q", got, "done")
	}
	if gotResume, _, _ := recorded(); gotResume != nil {
		t.Error("Respond must resolve the interrupt without re-executing the tool")
	}
}

// TestPartToRestart_Flow covers the ai.Part verbs used by callers that don't
// have the tool value in scope (e.g. a UI handler holding only the part): they
// build the same parts as the typed verbs of a claimed call, so the loop run
// in TestResumableTool_TypedRestart covers both.
func TestPartToRestart_Flow(t *testing.T) {
	reg := newTransferTestRegistry(t)
	transfer, _ := interruptOnce(t, reg)
	_, interrupt := generateUntilInterrupt(t, reg, transfer)
	call := claim(t, transfer, interrupt)

	restart, err := interrupt.ToToolRestart(confirmation{Approved: true})
	if err != nil {
		t.Fatalf("ToToolRestart: %v", err)
	}
	if diff := cmp.Diff(call.Restart(confirmation{Approved: true}), restart); diff != "" {
		t.Errorf("ToToolRestart differs from InterruptedCall.Restart (-typed +part):\n%s", diff)
	}

	response, err := interrupt.ToToolResponse(transferOut{Status: "manually approved"})
	if err != nil {
		t.Fatalf("ToToolResponse: %v", err)
	}
	if diff := cmp.Diff(call.Respond(transferOut{Status: "manually approved"}), response); diff != "" {
		t.Errorf("ToToolResponse differs from InterruptedCall.Respond (-typed +part):\n%s", diff)
	}
}

type question struct {
	Text string `json:"text"`
}

// newQuestionTool builds and registers the pure-question tool: it always
// pauses, without data, and the answer is its output.
func newQuestionTool(reg api.Registry) *ai.ResumableToolAction[question, string, struct{}] {
	return defineTestResumableTool(reg, "askUser", "asks the user a question",
		func(ctx context.Context, _ question, _ *struct{}) (string, error) {
			return "", tool.Interrupt(ctx, nil)
		})
}

// TestResumableTool_QuestionPattern covers a tool whose whole job is to
// ask: the model's input is the question, the interrupt carries no data of
// its own, and Respond on the claimed call is the answer the model then sees.
// A restart re-asks, which the loop reports rather than repeating silently.
func TestResumableTool_QuestionPattern(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "askUser", Input: map[string]any{"text": "proceed?"},
	}))
	askUser := newQuestionTool(reg)

	resp, interrupt := generateUntilInterrupt(t, reg, askUser)
	call := claim(t, askUser, interrupt)
	if call.Input.Text != "proceed?" {
		t.Errorf("call.Input = %+v, want the question the model asked", call.Input)
	}
	if _, ok := ai.InterruptAs[map[string]any](interrupt); ok {
		t.Error("a question tool's interrupt must carry no data of its own")
	}

	if got := resumeWith(t, reg, resp, askUser, ai.WithResume(call.Respond("yes"))); got != "done" {
		t.Errorf("final text after respond = %q, want %q", got, "done")
	}

	_, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithMessages(resp.History()...),
		ai.WithTools(askUser),
		ai.WithResume(call.Restart(struct{}{})))
	if !errors.Is(err, status.ErrFailedPrecondition) {
		t.Errorf("restarting a question tool: err = %v, want FAILED_PRECONDITION for the repeated interrupt", err)
	}
}

// TestWithResume_MixedKinds resumes two interrupts from one turn with a
// single WithResume: one restarted, one answered.
func TestWithResume_MixedKinds(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg,
		ai.NewToolRequestPart(&ai.ToolRequest{Name: "transfer", Ref: "a", Input: map[string]any{"amount": 200}}),
		ai.NewToolRequestPart(&ai.ToolRequest{Name: "askUser", Ref: "b", Input: map[string]any{"text": "sure?"}}))
	transfer, recorded := interruptOnce(t, reg)
	askUser := newQuestionTool(reg)

	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("go"),
		ai.WithTools(transfer, askUser))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	interrupts := resp.Interrupts()
	if len(interrupts) != 2 {
		t.Fatalf("got %d interrupts, want 2", len(interrupts))
	}

	var parts []*ai.Part
	for _, part := range interrupts {
		if call, ok := transfer.Interrupted(part); ok {
			parts = append(parts, call.Restart(confirmation{Approved: true}))
		}
		if call, ok := askUser.Interrupted(part); ok {
			parts = append(parts, call.Respond("yes"))
		}
	}
	if len(parts) != 2 {
		t.Fatalf("claimed %d parts, want each interrupt claimed by exactly one tool", len(parts))
	}

	resumed, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithMessages(resp.History()...),
		ai.WithTools(transfer, askUser),
		ai.WithResume(parts...))
	if err != nil {
		t.Fatalf("resume Generate: %v", err)
	}
	if resumed.Text() != "done" {
		t.Errorf("final text = %q, want %q", resumed.Text(), "done")
	}
	if gotResume, _, _ := recorded(); gotResume == nil || !gotResume.Approved {
		t.Errorf("restarted tool saw %+v, want Approved=true", gotResume)
	}
}

// TestToolContextTool_InterruptAndResumeData covers the verbs from a tool
// written against ToolContext: it interrupts with tool.Interrupt and, when
// restarted, reads the typed answer with tool.ResumeData off the context it
// was given (the ToolContext itself).
func TestToolContextTool_InterruptAndResumeData(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{
		Name: "gate", Input: map[string]any{"amount": 200},
	}))

	var (
		gotResume confirmation
		gotOK     bool
	)
	gate := defineTestTool(reg, "gate", "interrupts once, then reads its resume data",
		func(ctx context.Context, in transferIn) (string, error) {
			res, ok := tool.ResumeData[confirmation](ctx)
			if !ok {
				return "", tool.Interrupt(ctx, transferInterrupt{Reason: "confirm", Amount: in.Amount})
			}
			gotResume, gotOK = res, ok
			return "ok", nil
		})

	resp, interrupt := generateUntilInterrupt(t, reg, gate)
	meta, ok := ai.InterruptAs[transferInterrupt](interrupt)
	if !ok || meta.Reason != "confirm" {
		t.Fatalf("InterruptAs = %+v, %v; want the typed interrupt data", meta, ok)
	}
	// A ToolContext tool is a ToolAction, whose resume type is a map.
	call := claim(t, gate, interrupt)
	if call.Input.Amount != 200 {
		t.Errorf("call.Input = %+v, want {200}", call.Input)
	}
	restart := call.Restart(map[string]any{"approved": true})
	if got := resumeWith(t, reg, resp, gate, ai.WithResume(restart)); got != "done" {
		t.Errorf("final text = %q, want %q", got, "done")
	}
	if !gotOK || !gotResume.Approved {
		t.Errorf("ResumeData = %+v, %v; want {true}, true", gotResume, gotOK)
	}
}

// TestInterrupted_ClaimsOnlyOwnUnresolvedInterrupts checks that Interrupted
// reports false for everything that is not an unresolved interrupt of the
// tool, and on a nil tool.
func TestInterrupted_ClaimsOnlyOwnUnresolvedInterrupts(t *testing.T) {
	mine := ai.NewResumableTool("mine", "d",
		func(ctx context.Context, _ struct{}, _ *confirmation) (string, error) { return "", nil })

	foreign := ai.NewToolRequestPart(&ai.ToolRequest{Name: "other"})
	foreign.Interrupt = &ai.ToolInterrupt{}
	resolved := ai.NewToolRequestPart(&ai.ToolRequest{Name: "mine"})
	resolved.Interrupt = &ai.ToolInterrupt{Resolved: true}
	plain := ai.NewToolRequestPart(&ai.ToolRequest{Name: "mine"})

	for name, part := range map[string]*ai.Part{
		"another tool's interrupt": foreign,
		"a resolved interrupt":     resolved,
		"a plain tool request":     plain,
		"a text part":              ai.NewTextPart("hi"),
		"a nil part":               nil,
	} {
		if _, ok := mine.Interrupted(part); ok {
			t.Errorf("Interrupted claimed %s", name)
		}
	}

	own := ai.NewToolRequestPart(&ai.ToolRequest{Name: "mine"})
	own.Interrupt = &ai.ToolInterrupt{}
	if _, ok := mine.Interrupted(own); !ok {
		t.Error("Interrupted did not claim the tool's own interrupt")
	}
	var nilTool *ai.ResumableToolAction[struct{}, string, confirmation]
	if _, ok := nilTool.Interrupted(own); ok {
		t.Error("a nil tool claimed a part")
	}
}

// TestNewResumableTool_RejectsNonObjectResumeType covers the documented
// constraint: resume data must serialize to a JSON object. A resume type that
// cannot is rejected at definition, which is what lets Restart on a claimed
// call return the part without an error.
func TestNewResumableTool_RejectsNonObjectResumeType(t *testing.T) {
	for _, tc := range []struct {
		name   string
		define func()
	}{
		{"a string", func() {
			ai.NewResumableTool("scalar", "d",
				func(ctx context.Context, _ struct{}, _ *string) (string, error) { return "", nil })
		}},
		// A struct by kind, but it encodes itself as a JSON string.
		{"time.Time", func() {
			ai.NewResumableTool("when", "d",
				func(ctx context.Context, _ struct{}, _ *time.Time) (string, error) { return "", nil })
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("expected a panic defining a tool whose resume type is %s", tc.name)
				}
				err, ok := r.(error)
				if !ok || !strings.Contains(err.Error(), "ai.NewResumableTool") || !strings.Contains(err.Error(), "JSON object") {
					t.Errorf("panic = %v, want it to name ai.NewResumableTool and the JSON object constraint", r)
				}
			}()
			tc.define()
		})
	}
}

// TestResume_ReadByJSONShape pins that a resume payload reads the same
// whether it stayed in process or crossed the wire: a Res with nil slice and
// pointer fields reaches the tool as built, since the compiler checked it; a
// struct of another type with the same fields decodes by its shape; and a
// named map type decodes from the plain map a wire hop yields.
func TestResume_ReadByJSONShape(t *testing.T) {
	type detail struct {
		Reason string `json:"reason"`
	}
	type answer struct {
		Approved bool     `json:"approved"`
		Notes    []string `json:"notes"`
		Detail   *detail  `json:"detail"`
	}
	type approval map[string]any

	for _, hop := range []struct {
		name string
		fn   func(t *testing.T, msgs []*ai.Message) []*ai.Message
	}{
		{"in process", func(_ *testing.T, msgs []*ai.Message) []*ai.Message { return msgs }},
		{"after a wire hop", viaJSON},
	} {
		t.Run(hop.name, func(t *testing.T) {
			// resumeOnce defines a tool with resume type Res that interrupts
			// once, restarts it with the part restart builds, and returns
			// the resume the tool saw.
			resumeOnce := func(t *testing.T, define func(reg *registry.Registry, saw func(any)) ai.Tool, restart func(*ai.Part) *ai.Part) any {
				t.Helper()
				reg := newTransferTestRegistry(t)
				var seen any
				tl := define(reg, func(v any) { seen = v })
				resp, err := ai.Generate(context.Background(), reg,
					ai.WithModelName("test/model"), ai.WithPrompt("go"), ai.WithTools(tl))
				if err != nil {
					t.Fatalf("Generate: %v", err)
				}
				history := resp.History()
				part := restart(singleInterrupt(t, resp))
				if hop.name != "in process" {
					history = hop.fn(t, history)
					part = hop.fn(t, []*ai.Message{{Role: ai.RoleTool, Content: []*ai.Part{part}}})[0].Content[0]
				}
				if _, err := ai.Generate(context.Background(), reg,
					ai.WithModelName("test/model"), ai.WithMessages(history...),
					ai.WithTools(tl), ai.WithResume(part)); err != nil {
					t.Fatalf("resume: %v", err)
				}
				return seen
			}
			defineAnswer := func(reg *registry.Registry, saw func(any)) ai.Tool {
				return defineTestResumableTool(reg, "transfer", "transfers money",
					func(ctx context.Context, _ transferIn, res *answer) (string, error) {
						if res == nil {
							return "", tool.Interrupt(ctx, nil)
						}
						saw(*res)
						return "ok", nil
					})
			}

			t.Run("a Res with nil fields", func(t *testing.T) {
				var tl *ai.ResumableToolAction[transferIn, string, answer]
				got := resumeOnce(t, func(reg *registry.Registry, saw func(any)) ai.Tool {
					tl = defineAnswer(reg, saw).(*ai.ResumableToolAction[transferIn, string, answer])
					return tl
				}, func(p *ai.Part) *ai.Part {
					return claim(t, tl, p).Restart(answer{Approved: true})
				})
				if a, ok := got.(answer); !ok || !a.Approved {
					t.Errorf("tool saw %#v, want an approved answer", got)
				}
			})

			t.Run("a struct of another type", func(t *testing.T) {
				got := resumeOnce(t, defineAnswer, func(p *ai.Part) *ai.Part {
					restart, err := p.ToToolRestart(struct {
						Approved bool     `json:"approved"`
						Notes    []string `json:"notes"`
						Detail   *detail  `json:"detail"`
					}{Approved: true, Notes: []string{}, Detail: &detail{Reason: "ok"}})
					if err != nil {
						t.Fatalf("ToToolRestart: %v", err)
					}
					return restart
				})
				if a, ok := got.(answer); !ok || !a.Approved || a.Detail == nil || a.Detail.Reason != "ok" {
					t.Errorf("tool saw %#v, want the answer decoded by shape", got)
				}
			})

			t.Run("a named map", func(t *testing.T) {
				var tl *ai.ResumableToolAction[transferIn, string, approval]
				got := resumeOnce(t, func(reg *registry.Registry, saw func(any)) ai.Tool {
					tl = defineTestResumableTool(reg, "transfer", "transfers money",
						func(ctx context.Context, _ transferIn, res *approval) (string, error) {
							if res == nil {
								return "", tool.Interrupt(ctx, nil)
							}
							saw(*res)
							return "ok", nil
						})
					return tl
				}, func(p *ai.Part) *ai.Part {
					return claim(t, tl, p).Restart(approval{"approved": true})
				})
				if a, ok := got.(approval); !ok || a["approved"] != true {
					t.Errorf("tool saw %#v, want the approval", got)
				}
			})
		})
	}
}

// TestInterrupt_NonObjectData_ReturnsClearError covers the same constraint on
// the interrupt side: interrupting with a scalar fails the call with a clear
// error when the loop records the interrupt, for both kinds of tool.
func TestInterrupt_NonObjectData_ReturnsClearError(t *testing.T) {
	for _, tc := range []struct {
		name   string
		define func(reg *registry.Registry) ai.Tool
	}{
		{"resumable", func(reg *registry.Registry) ai.Tool {
			return defineTestResumableTool(reg, "bad", "interrupts with a scalar",
				func(ctx context.Context, _ struct{}, _ *struct{}) (string, error) {
					return "", tool.Interrupt(ctx, "not an object")
				})
		}},
		{"plain", func(reg *registry.Registry) ai.Tool {
			return defineTestTool(reg, "bad", "interrupts with a scalar",
				func(ctx context.Context, _ struct{}) (string, error) {
					return "", tool.Interrupt(ctx, "not an object")
				})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reg := newToolTestRegistry(t)
			tl := tc.define(reg)
			defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "bad", Input: map[string]any{}}))

			_, err := ai.Generate(context.Background(), reg,
				ai.WithModelName("test/model"),
				ai.WithPrompt("go"),
				ai.WithTools(tl))
			if err == nil {
				t.Fatal("expected an error interrupting with non-object data")
			}
			if !strings.Contains(err.Error(), "JSON object") {
				t.Errorf("error = %q, want it to mention the JSON object constraint", err)
			}
		})
	}
}

// TestRestart_ResumeDataValidated pins that a restart's payload is validated
// against the schema inferred from Res before the tool re-executes, as the
// model's input is against In: a mistyped or missing field fails the resume
// with an error naming the field, and the tool never runs. The untyped verb is
// the path a payload from the wire takes; the typed Restart cannot build these
// payloads.
func TestRestart_ResumeDataValidated(t *testing.T) {
	for _, tc := range []struct {
		name   string
		resume any
		want   string
	}{
		{"mistyped field", map[string]any{"approved": "yes"}, "approved"},
		{"missing field on a bare restart", nil, "approved is required"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reg := newTransferTestRegistry(t)
			transfer, recorded := interruptOnce(t, reg)
			resp, interrupt := generateUntilInterrupt(t, reg, transfer)

			restart, err := interrupt.ToToolRestart(tc.resume)
			if err != nil {
				t.Fatalf("ToToolRestart: %v", err)
			}
			_, err = ai.Generate(context.Background(), reg,
				ai.WithModelName("test/model"),
				ai.WithMessages(resp.History()...),
				ai.WithTools(transfer),
				ai.WithResume(restart))
			if !errors.Is(err, status.ErrInvalidArgument) {
				t.Fatalf("resume error = %v, want status.ErrInvalidArgument", err)
			}
			if !strings.Contains(err.Error(), "resume data") || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %q, want it to name the resume data and %q", err, tc.want)
			}
			if gotResume, _, _ := recorded(); gotResume != nil {
				t.Errorf("tool re-executed with %+v; a rejected payload must not reach it", *gotResume)
			}
		})
	}
}

// TestResume_RejectedBeforeAnyToolRuns pins that a resume is checked as a
// whole before any tool runs: when one pending request has no resolution, or
// a restart payload or response that does not match the tool's schema, the
// resume fails and a sibling with a valid restart has not run.
func TestResume_RejectedBeforeAnyToolRuns(t *testing.T) {
	for _, tc := range []struct {
		name string
		want error
		// resolve returns the directives for the second request, "confirm".
		resolve func(t *testing.T, part *ai.Part) []*ai.Part
	}{
		{"no resolution", ai.ErrUnresolvedToolRequest, func(*testing.T, *ai.Part) []*ai.Part { return nil }},
		{"mistyped restart payload", status.ErrInvalidArgument, func(t *testing.T, part *ai.Part) []*ai.Part {
			restart, err := part.ToToolRestart(map[string]any{"approved": "yes"})
			if err != nil {
				t.Fatalf("ToToolRestart: %v", err)
			}
			return []*ai.Part{restart}
		}},
		{"mistyped response", status.ErrInvalidArgument, func(t *testing.T, part *ai.Part) []*ai.Part {
			respond, err := part.ToToolResponse(map[string]any{"not": "a string"})
			if err != nil {
				t.Fatalf("ToToolResponse: %v", err)
			}
			return []*ai.Part{respond}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reg := newToolTestRegistry(t)
			defineToolThenFinishModel(reg,
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "charge", Ref: "1", Input: map[string]any{}}),
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "confirm", Ref: "2", Input: map[string]any{}}))
			charges := 0
			charge := defineTestResumableTool(reg, "charge", "charges a card",
				func(ctx context.Context, _ struct{}, res *confirmation) (string, error) {
					if res == nil {
						return "", tool.Interrupt(ctx, nil)
					}
					charges++
					return "charged", nil
				})
			confirm := defineTestResumableTool(reg, "confirm", "asks to confirm",
				func(ctx context.Context, _ struct{}, res *confirmation) (string, error) {
					if res == nil {
						return "", tool.Interrupt(ctx, nil)
					}
					return "confirmed", nil
				})

			resp, err := ai.Generate(context.Background(), reg,
				ai.WithModelName("test/model"), ai.WithPrompt("go"), ai.WithTools(charge, confirm))
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			var parts []*ai.Part
			for _, part := range resp.Interrupts() {
				if call, ok := charge.Interrupted(part); ok {
					parts = append(parts, call.Restart(confirmation{Approved: true}))
				} else {
					parts = append(parts, tc.resolve(t, part)...)
				}
			}
			_, err = ai.Generate(context.Background(), reg,
				ai.WithModelName("test/model"), ai.WithMessages(resp.History()...),
				ai.WithTools(charge, confirm), ai.WithResume(parts...))
			if !errors.Is(err, tc.want) {
				t.Fatalf("resume error = %v, want %v", err, tc.want)
			}
			if charges != 0 {
				t.Errorf("charge ran %d times before the resume was rejected, want 0", charges)
			}
		})
	}
}

// TestSendPartial_StreamsPartialToolResponse asserts a tool's SendPartial calls
// arrive on the stream as partial tool responses, distinguishable via
// IsPartial / ToolResponses.
func TestSendPartial_StreamsPartialToolResponse(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "progressTool", Input: map[string]any{}}))

	defineTestTool(reg, "progressTool", "streams progress",
		func(ctx context.Context, _ struct{}) (string, error) {
			tool.SendPartial(ctx, map[string]any{"progress": 50})
			return "complete", nil
		})

	var partials []*ai.Part
	for val, err := range ai.GenerateStream(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("go"),
		ai.WithTools(ai.ToolName("progressTool"))) {
		if err != nil {
			t.Fatalf("GenerateStream: %v", err)
		}
		if val.Done {
			continue
		}
		for _, p := range val.Chunk.ToolResponses() {
			if p.IsPartial() {
				partials = append(partials, p)
			}
		}
	}

	if len(partials) == 0 {
		t.Fatal("expected at least one partial tool response on the stream")
	}
	if partials[0].ToolResponse.Name != "progressTool" {
		t.Errorf("partial tool name = %q, want %q", partials[0].ToolResponse.Name, "progressTool")
	}
}

// TestSendPartial_StreamsFromARestartedTool pins that a restarted tool
// streams as a tool of a model turn does: its SendPartial calls arrive on the
// resumed generation's stream, tagged with the request they belong to.
func TestSendPartial_StreamsFromARestartedTool(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "progressTool", Ref: "r1", Input: map[string]any{}}))
	progress := defineTestResumableTool(reg, "progressTool", "streams progress after approval",
		func(ctx context.Context, _ struct{}, res *confirmation) (string, error) {
			if res == nil {
				return "", tool.Interrupt(ctx, nil)
			}
			tool.SendPartial(ctx, map[string]any{"progress": 50})
			return "complete", nil
		})

	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithPrompt("go"), ai.WithTools(progress))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	restart := claim(t, progress, singleInterrupt(t, resp)).Restart(confirmation{Approved: true})

	var partials []*ai.Part
	for val, err := range ai.GenerateStream(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithMessages(resp.History()...),
		ai.WithTools(progress),
		ai.WithResume(restart)) {
		if err != nil {
			t.Fatalf("GenerateStream: %v", err)
		}
		if val.Done {
			continue
		}
		for _, p := range val.Chunk.ToolResponses() {
			if p.IsPartial() {
				partials = append(partials, p)
			}
		}
	}
	if len(partials) != 1 {
		t.Fatalf("got %d partial tool responses on the stream, want 1", len(partials))
	}
	if got := partials[0].ToolResponse; got.Name != "progressTool" || got.Ref != "r1" {
		t.Errorf("partial = %s#%s, want progressTool#r1", got.Name, got.Ref)
	}
}

// TestConcurrentStreamingTools_NoDataRace is the regression for the streaming
// race: when a model emits multiple tool calls in one turn and more than one
// streams via SendPartial, the per-tool senders run on concurrent goroutines.
// They must be serialized so they don't race on the shared stream callback.
// Run under `go test -race` to detect a regression.
func TestConcurrentStreamingTools_NoDataRace(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg,
		ai.NewToolRequestPart(&ai.ToolRequest{Name: "toolA", Input: map[string]any{}}),
		ai.NewToolRequestPart(&ai.ToolRequest{Name: "toolB", Input: map[string]any{}}))

	// A rendezvous so both tools enter their SendPartial loops at the same
	// time, maximizing the chance of overlapping callback invocations.
	var ready sync.WaitGroup
	ready.Add(2)
	start := make(chan struct{})
	go func() { ready.Wait(); close(start) }()

	streamer := func(ctx context.Context, _ struct{}) (string, error) {
		ready.Done()
		<-start
		for i := 0; i < 200; i++ {
			tool.SendPartial(ctx, map[string]any{"n": i})
		}
		return "ok", nil
	}
	defineTestTool(reg, "toolA", "streams", streamer)
	defineTestTool(reg, "toolB", "streams", streamer)

	for _, err := range ai.GenerateStream(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("go"),
		ai.WithTools(ai.ToolName("toolA"), ai.ToolName("toolB"))) {
		if err != nil {
			t.Fatalf("GenerateStream: %v", err)
		}
	}
}

// TestResumableTool_UnregisteredViaWithTools passes a resumable tool
// created with ai.NewResumableTool, never registered, straight to Generate.
// It is an ai.Tool, so the loop registers it for the call the way it does an
// unregistered ai.NewTool.
func TestResumableTool_UnregisteredViaWithTools(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "unregistered", Input: map[string]any{"city": "Lima"}}))

	ran := false
	tl := ai.NewResumableTool("unregistered", "d",
		func(ctx context.Context, in weatherIn, _ *confirmation) (reportOut, error) {
			ran = true
			return reportOut{}, nil
		})

	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("weather"),
		ai.WithTools(tl))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !ran {
		t.Error("the unregistered resumable tool did not run")
	}
	if got := resp.Text(); got != "done" {
		t.Errorf("Text() = %q, want done", got)
	}
}

// jsRestartOf builds the restart part the JS runtime's restartTool builds for
// an interrupted request: the interrupted part's metadata spread onto the
// restart, "interrupt" key included, with "resumed" added. It goes through
// JSON so the part is exactly what a peer runtime would send.
func jsRestartOf(t *testing.T, interrupt *ai.Part, resumed any) *ai.Part {
	t.Helper()
	raw, err := json.Marshal(interrupt)
	if err != nil {
		t.Fatalf("marshal interrupt: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal interrupt: %v", err)
	}
	meta, _ := wire["metadata"].(map[string]any)
	if meta == nil {
		meta = map[string]any{}
		wire["metadata"] = meta
	}
	meta["resumed"] = resumed
	raw, err = json.Marshal(wire)
	if err != nil {
		t.Fatalf("marshal restart: %v", err)
	}
	var restart ai.Part
	if err := json.Unmarshal(raw, &restart); err != nil {
		t.Fatalf("unmarshal restart: %v", err)
	}
	return &restart
}

// TestResume_AcceptsJSRestartShape pins that a restart part still carrying
// the interrupt it resolves, the shape every JS client sends, resumes the
// tool: the restart supersedes the interrupt rather than conflicting with it.
func TestResume_AcceptsJSRestartShape(t *testing.T) {
	reg := newTransferTestRegistry(t)
	transfer, saw := interruptOnce(t, reg)
	resp, interrupt := generateUntilInterrupt(t, reg, transfer)

	restart := jsRestartOf(t, interrupt, map[string]any{"approved": true})
	if !restart.IsInterrupt() || !restart.IsRestart() {
		t.Fatalf("restart = %+v, want both the interrupt and the restart state lifted", restart)
	}

	if got := resumeWith(t, reg, resp, transfer, ai.WithResume(restart)); got != "done" {
		t.Errorf("Text() = %q, want done", got)
	}
	if res, _, _ := saw(); res == nil || !res.Approved {
		t.Errorf("tool saw resume = %+v, want approved", res)
	}
}

// TestResume_NonObjectResumeMarkers pins how a resumed marker Go cannot
// deliver as an object is read, matching the JS runtime's truthiness rule:
// false is not a resumption and the tool re-executes afresh; any other
// non-object value is a bare restart, delivered as an empty payload to a tool
// whose resume type admits one.
func TestResume_NonObjectResumeMarkers(t *testing.T) {
	t.Run("false re-executes without a resume payload", func(t *testing.T) {
		reg := newTransferTestRegistry(t)
		transfer, saw := interruptOnce(t, reg)
		resp, interrupt := generateUntilInterrupt(t, reg, transfer)

		resp2, err := ai.Generate(context.Background(), reg,
			ai.WithModelName("test/model"),
			ai.WithMessages(resp.History()...),
			ai.WithTools(transfer),
			ai.WithResume(jsRestartOf(t, interrupt, false)))
		// The tool saw no resume, so it interrupted again, which the loop
		// reports as a failed precondition next to the partial response.
		if !errors.Is(err, status.ErrFailedPrecondition) || resp2 == nil {
			t.Fatalf("resume Generate = (%v, %v), want the re-interrupted partial and FAILED_PRECONDITION", resp2, err)
		}
		if resp2.FinishReason != ai.FinishReasonInterrupted {
			t.Errorf("FinishReason = %q, want interrupted", resp2.FinishReason)
		}
		if res, _, _ := saw(); res != nil {
			t.Errorf("tool saw resume = %+v, want none", res)
		}
	})

	for _, marker := range []any{"approved", 1.0, []any{"a"}} {
		t.Run(fmt.Sprintf("%T is a bare restart", marker), func(t *testing.T) {
			reg := newTransferTestRegistry(t)
			transfer, saw := interruptOnceBare(t, reg)
			resp, interrupt := generateUntilInterrupt(t, reg, transfer)

			if got := resumeWith(t, reg, resp, transfer, ai.WithResume(jsRestartOf(t, interrupt, marker))); got != "done" {
				t.Errorf("Text() = %q, want done", got)
			}
			if res := saw(); res == nil || res.Approved {
				t.Errorf("tool saw resume = %+v, want the zero value of a bare restart", res)
			}
		})
	}
}

// TestWithResume_NilPartIsReported pins that a nil in the resume list, which
// a deprecated verb returns for a part it cannot restart, is reported as such
// rather than as a part of the wrong kind.
func TestWithResume_NilPartIsReported(t *testing.T) {
	reg := newTransferTestRegistry(t)
	transfer, _ := interruptOnce(t, reg)
	resp, _ := generateUntilInterrupt(t, reg, transfer)

	_, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithMessages(resp.History()...),
		ai.WithTools(transfer),
		ai.WithResume(nil))
	if err == nil || !strings.Contains(err.Error(), "part is nil") {
		t.Errorf("error = %v, want it to report the nil part", err)
	}
}

// TestInterruptAs_DecodesIntoAnyMatchingType pins that the interrupt data a
// tool sent as a struct reads back into any type with the same JSON shape,
// in process as well as after a wire hop: the loop records the data as the
// JSON object it serializes to, so a handler in another package with its own
// view of the payload decodes it either way.
func TestInterruptAs_DecodesIntoAnyMatchingType(t *testing.T) {
	type transferInterruptView struct {
		Reason string  `json:"reason"`
		Amount float64 `json:"amount"`
	}
	reg := newTransferTestRegistry(t)
	transfer, _ := interruptOnce(t, reg)
	_, interrupt := generateUntilInterrupt(t, reg, transfer)

	if _, ok := interrupt.Interrupt.Data.(map[string]any); !ok {
		t.Errorf("Interrupt.Data = %T, want the JSON object the tool's struct serializes to", interrupt.Interrupt.Data)
	}
	view, ok := ai.InterruptAs[transferInterruptView](interrupt)
	if !ok || view.Reason != "large_amount" || view.Amount != 200 {
		t.Errorf("InterruptAs[view] = (%+v, %v), want the payload decoded", view, ok)
	}
	same, ok := ai.InterruptAs[transferInterrupt](interrupt)
	if !ok || same.Reason != "large_amount" {
		t.Errorf("InterruptAs[same type] = (%+v, %v), want the payload decoded", same, ok)
	}
}

// TestResume_PayloadReachesEveryReaderAlike pins that one restart reads the
// same through every reader: a map restarted in process keeps its Go types
// in the tool's resume parameter, in tool.ResumeData and in ai.ResumedValue,
// rather than widening to float64 in one of them.
func TestResume_PayloadReachesEveryReaderAlike(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "count", Input: map[string]any{}}))

	var (
		paramType, dataType string
		viaValue            int
	)
	count := defineTestResumableTool(reg, "count", "d",
		func(ctx context.Context, _ struct{}, res *map[string]any) (string, error) {
			if res == nil {
				return "", tool.Interrupt(ctx, nil)
			}
			paramType = fmt.Sprintf("%T", (*res)["n"])
			rd, _ := tool.ResumeData[map[string]any](ctx)
			dataType = fmt.Sprintf("%T", rd["n"])
			viaValue, _ = ai.ResumedValue[int](ctx, "n")
			return "ok", nil
		})

	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("count"),
		ai.WithTools(count))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	interrupts := resp.Interrupts()
	if len(interrupts) != 1 {
		t.Fatalf("expected 1 interrupt, got %d", len(interrupts))
	}
	call := claim(t, count, interrupts[0])
	if got := resumeWith(t, reg, resp, count, ai.WithResume(call.Restart(map[string]any{"n": 5}))); got != "done" {
		t.Errorf("Text() = %q, want done", got)
	}
	if paramType != "int" || dataType != "int" || viaValue != 5 {
		t.Errorf("resume parameter saw %s, tool.ResumeData saw %s, ai.ResumedValue saw %d; want int, int, 5", paramType, dataType, viaValue)
	}
}

// gate is an inline WrapTool middleware that holds every call until a restart
// answers it with {"ok": true}, logging what each invocation saw: "held",
// "answered", or "released". Two gates in one chain share the inline name,
// which the chain tells apart.
func gate(log *[]string) ai.MiddlewareFunc {
	return func(ctx context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{
			WrapTool: func(ctx context.Context, p *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
				if tool.Released(ctx) {
					*log = append(*log, "released")
					return next(ctx, p)
				}
				if answer, ok := tool.ResumeData[map[string]any](ctx); ok {
					*log = append(*log, "answered")
					if answer["ok"] == true {
						return next(ctx, p)
					}
				}
				*log = append(*log, "held")
				return nil, tool.Interrupt(ctx, map[string]any{"gate": "held"})
			},
		}, nil
	}
}

// singleInterrupt returns the one interrupt part of resp.
func singleInterrupt(t *testing.T, resp *ai.ModelResponse) *ai.Part {
	t.Helper()
	interrupts := resp.Interrupts()
	if len(interrupts) != 1 {
		t.Fatalf("expected 1 interrupt, got %d (finish=%s)", len(interrupts), resp.FinishReason)
	}
	return interrupts[0]
}

// viaJSON round-trips messages through JSON, as a client that stores or
// forwards a conversation does, so a test can pin what survives the wire.
func viaJSON(t *testing.T, msgs []*ai.Message) []*ai.Message {
	t.Helper()
	raw, err := json.Marshal(msgs)
	if err != nil {
		t.Fatalf("marshal history: %v", err)
	}
	var out []*ai.Message
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal history: %v", err)
	}
	return out
}

// TestRestart_AnswersTheStageThatInterrupted pins that a restart answers
// whoever interrupted. A WrapTool hook that holds a call raises its own
// interrupt: the tool declines to claim it, the restart that answers it is
// read by the hook alone, and the tool then runs as a fresh call, asks its
// own question with a nil resume parameter, and gets that answer while the
// hook, which released the call before, lets the restart through. The stage
// rides on the interrupted request in history, so the flow survives a wire
// hop.
func TestRestart_AnswersTheStageThatInterrupted(t *testing.T) {
	for _, tc := range []struct {
		name string
		hop  func(t *testing.T, msgs []*ai.Message) []*ai.Message
	}{
		{"in process", func(_ *testing.T, msgs []*ai.Message) []*ai.Message { return msgs }},
		{"after a wire hop", viaJSON},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reg := newTransferTestRegistry(t)
			transfer, saw := interruptOnce(t, reg)
			var log []string
			hold := gate(&log)
			resume := func(history []*ai.Message, part *ai.Part) (*ai.ModelResponse, error) {
				return ai.Generate(context.Background(), reg,
					ai.WithModelName("test/model"),
					ai.WithMessages(tc.hop(t, history)...),
					ai.WithTools(transfer),
					ai.WithUse(hold),
					ai.WithResume(part))
			}

			resp, err := ai.Generate(context.Background(), reg,
				ai.WithModelName("test/model"),
				ai.WithPrompt("transfer 200"),
				ai.WithTools(transfer),
				ai.WithUse(hold))
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			held := singleInterrupt(t, resp)
			if held.Interrupt == nil || held.Interrupt.RaisedBy != "inline" {
				t.Fatalf("held part interrupt = %+v, want one raised by the inline hook", held.Interrupt)
			}
			if _, ok := transfer.Interrupted(held); ok {
				t.Error("the tool claimed a hold its middleware raised")
			}

			// Answering the hook releases the call: the tool runs afresh,
			// with no resume, and asks its own question, which the loop
			// reports as a re-interrupt next to the partial response.
			restart, err := held.ToToolRestart(map[string]any{"ok": true})
			if err != nil {
				t.Fatalf("ToToolRestart: %v", err)
			}
			resp2, err := resume(resp.History(), restart)
			if !errors.Is(err, status.ErrFailedPrecondition) || resp2 == nil {
				t.Fatalf("resume = (%v, %v), want the tool's own interrupt under FAILED_PRECONDITION", resp2, err)
			}
			if res, _, _ := saw(); res != nil {
				t.Fatalf("tool saw resume = %+v, want none: the answer to the hook must not reach it", *res)
			}
			asked := singleInterrupt(t, resp2)
			if asked.Interrupt == nil || asked.Interrupt.RaisedBy != "" {
				t.Fatalf("re-interrupt = %+v, want one the tool raised", asked.Interrupt)
			}

			// Answering the tool passes the hook, which released the call.
			call := claim(t, transfer, asked)
			resp3, err := resume(resp2.History(), call.Restart(confirmation{Approved: true}))
			if err != nil {
				t.Fatalf("second resume: %v", err)
			}
			if resp3.Text() != "done" {
				t.Errorf("Text() = %q, want done", resp3.Text())
			}
			if res, _, _ := saw(); res == nil || !res.Approved {
				t.Errorf("tool saw resume = %+v, want approved", res)
			}
			if diff := cmp.Diff([]string{"held", "answered", "released"}, log); diff != "" {
				t.Errorf("hook saw (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRestart_TwoGatesAnswerInTurn pins the chain positions a restart is
// delivered by: with two holding hooks, answering the first reaches it alone
// and the second then holds; answering the second reaches it while the first,
// which released the call, lets it through; and answering the tool's own
// question passes both. Both hooks are inline, so they share a name and the
// chain tells them apart.
func TestRestart_TwoGatesAnswerInTurn(t *testing.T) {
	reg := newTransferTestRegistry(t)
	transfer, saw := interruptOnce(t, reg)
	var logA, logB []string
	a, b := gate(&logA), gate(&logB)
	generate := func(opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
		return ai.Generate(context.Background(), reg, append([]ai.GenerateOption{
			ai.WithModelName("test/model"), ai.WithTools(transfer), ai.WithUse(a, b),
		}, opts...)...)
	}
	answer := func(t *testing.T, part *ai.Part) *ai.Part {
		t.Helper()
		restart, err := part.ToToolRestart(map[string]any{"ok": true})
		if err != nil {
			t.Fatalf("ToToolRestart: %v", err)
		}
		return restart
	}

	resp, err := generate(ai.WithPrompt("transfer 200"))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	heldA := singleInterrupt(t, resp)
	if heldA.Interrupt.RaisedBy != "inline" {
		t.Fatalf("first hold raised by %q, want the first inline hook", heldA.Interrupt.RaisedBy)
	}

	resp2, err := generate(ai.WithMessages(resp.History()...), ai.WithResume(answer(t, heldA)))
	if !errors.Is(err, status.ErrFailedPrecondition) || resp2 == nil {
		t.Fatalf("answering the first hook = (%v, %v), want the second hook's hold", resp2, err)
	}
	heldB := singleInterrupt(t, resp2)
	if heldB.Interrupt.RaisedBy != "inline#2" {
		t.Fatalf("second hold raised by %q, want the second inline hook", heldB.Interrupt.RaisedBy)
	}

	resp3, err := generate(ai.WithMessages(resp2.History()...), ai.WithResume(answer(t, heldB)))
	if !errors.Is(err, status.ErrFailedPrecondition) || resp3 == nil {
		t.Fatalf("answering the second hook = (%v, %v), want the tool's own interrupt", resp3, err)
	}
	asked := singleInterrupt(t, resp3)
	if asked.Interrupt.RaisedBy != "" {
		t.Fatalf("re-interrupt raised by %q, want the tool", asked.Interrupt.RaisedBy)
	}
	if res, _, _ := saw(); res != nil {
		t.Fatalf("tool saw resume = %+v before it was answered", *res)
	}

	call := claim(t, transfer, asked)
	resp4, err := generate(ai.WithMessages(resp3.History()...), ai.WithResume(call.Restart(confirmation{Approved: true})))
	if err != nil {
		t.Fatalf("answering the tool: %v", err)
	}
	if resp4.Text() != "done" {
		t.Errorf("Text() = %q, want done", resp4.Text())
	}
	if res, _, _ := saw(); res == nil || !res.Approved {
		t.Errorf("tool saw resume = %+v, want approved", res)
	}
	if diff := cmp.Diff([]string{"held", "answered", "released", "released"}, logA); diff != "" {
		t.Errorf("first hook saw (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"held", "answered", "released"}, logB); diff != "" {
		t.Errorf("second hook saw (-want +got):\n%s", diff)
	}
}

// TestAttachParts_FromWrapToolHook pins that the part sink spans the whole
// tool call: a WrapTool hook attaches parts before and after running the
// tool, and they land on the tool response next to the tool's own, in call
// order, instead of vanishing because the sink was installed inside the call.
func TestAttachParts_FromWrapToolHook(t *testing.T) {
	reg := newToolTestRegistry(t)
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "shot", Input: map[string]any{}}))
	shot := defineTestTool(reg, "shot", "takes a screenshot",
		func(ctx context.Context, _ struct{}) (string, error) {
			tool.AttachParts(ctx, ai.NewTextPart("tool"))
			return "captured", nil
		})
	attach := ai.MiddlewareFunc(func(ctx context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{
			WrapTool: func(ctx context.Context, p *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
				tool.AttachParts(ctx, ai.NewTextPart("before"))
				resp, err := next(ctx, p)
				tool.AttachParts(ctx, ai.NewTextPart("after"))
				return resp, err
			},
		}, nil
	})

	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"),
		ai.WithPrompt("go"),
		ai.WithTools(shot),
		ai.WithUse(attach))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	var got []string
	for _, m := range resp.History() {
		if m.Role != ai.RoleTool {
			continue
		}
		for _, p := range m.Content[0].ToolResponse.Content {
			got = append(got, p.Text)
		}
	}
	if diff := cmp.Diff([]string{"before", "tool", "after"}, got); diff != "" {
		t.Errorf("attached parts mismatch (-want +got):\n%s", diff)
	}
}

// restartThen defines "outer", a resumable tool that interrupts on its
// first call and, once restarted, runs then, and generates until outer is
// restarted with an approval. It returns outer's output.
func restartThen(t *testing.T, reg *registry.Registry, then func(ctx context.Context) (string, error)) string {
	t.Helper()
	outer := defineTestResumableTool(reg, "outer", "outer",
		func(ctx context.Context, _ struct{}, res *confirmation) (string, error) {
			if res == nil {
				return "", tool.Interrupt(ctx, nil)
			}
			return then(ctx)
		})
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "outer", Input: map[string]any{}}))
	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithPrompt("go"), ai.WithTools(outer))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	call := claim(t, outer, singleInterrupt(t, resp))
	resp, err = ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithMessages(resp.History()...),
		ai.WithTools(outer), ai.WithResume(call.Restart(confirmation{Approved: true})))
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	for _, m := range resp.History() {
		for _, p := range m.Content {
			if p.IsToolResponse() && p.ToolResponse.Name == "outer" {
				out, _ := p.ToolResponse.Output.(string)
				return out
			}
		}
	}
	t.Fatal("no response from outer in history")
	return ""
}

// TestToolCall_NestedGenerateAnswersNoRestart pins that a Generate a restarted
// tool runs is a fresh generation: its tools see no resume, so a tool that
// asks for approval asks rather than taking the enclosing call's answer as
// its own.
func TestToolCall_NestedGenerateAnswersNoRestart(t *testing.T) {
	reg := newToolTestRegistry(t)
	innerResumed := false
	inner := defineTestTool(reg, "inner", "inner",
		func(ctx context.Context, _ struct{}) (string, error) {
			if _, ok := tool.ResumeData[map[string]any](ctx); ok {
				innerResumed = true
				return "ran", nil
			}
			return "", tool.Interrupt(ctx, nil)
		})
	defineTestModel(reg, "test/inner",
		&ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Request: req,
				Message: &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
					ai.NewToolRequestPart(&ai.ToolRequest{Name: "inner", Input: map[string]any{}}),
				}},
			}, nil
		})

	out := restartThen(t, reg, func(ctx context.Context) (string, error) {
		resp, err := ai.Generate(ctx, reg,
			ai.WithModelName("test/inner"), ai.WithPrompt("go"), ai.WithTools(inner))
		if err != nil {
			return "", err
		}
		return string(resp.FinishReason), nil
	})
	if innerResumed {
		t.Error("the nested tool took the enclosing restart's answer as its own")
	}
	if out != string(ai.FinishReasonInterrupted) {
		t.Errorf("nested generation finished %q, want %q", out, ai.FinishReasonInterrupted)
	}
}

// TestToolCall_DirectRunIsItsOwnCall pins that a tool a restarted tool runs
// directly is a call of its own: the parts it attaches land on its own
// response rather than the enclosing call's, and it sees no resume.
func TestToolCall_DirectRunIsItsOwnCall(t *testing.T) {
	reg := newToolTestRegistry(t)
	chartResumed := false
	chart := defineTestTool(reg, "chart", "chart",
		func(ctx context.Context, _ struct{}) (string, error) {
			_, chartResumed = tool.ResumeData[map[string]any](ctx)
			tool.AttachParts(ctx, ai.NewMediaPart("image/png", "chart"))
			return "chart", nil
		})

	out := restartThen(t, reg, func(ctx context.Context) (string, error) {
		resp, err := chart.RunRawMultipart(ctx, struct{}{})
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("chart parts: %d", len(resp.Content)), nil
	})
	if chartResumed {
		t.Error("the directly run tool took the enclosing restart's answer as its own")
	}
	if out != "chart parts: 1" {
		t.Errorf("outer saw %q, want the attached part on the directly run tool's response", out)
	}
}

// TestToolCall_RetriedToolStageKeepsTheRestart pins that a hook running the
// tool stage more than once, as a retry does, delivers the restart to every
// run: each run of the tool stage is the call's own.
func TestToolCall_RetriedToolStageKeepsTheRestart(t *testing.T) {
	reg := newToolTestRegistry(t)
	retry := ai.MiddlewareFunc(func(ctx context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{
			WrapTool: func(ctx context.Context, p *ai.ToolParams, next ai.ToolNext) (*ai.MultipartToolResponse, error) {
				if _, err := next(ctx, p); err != nil {
					return nil, err
				}
				return next(ctx, p)
			},
		}, nil
	})
	var runs []bool
	tl := defineTestResumableTool(reg, "outer", "outer",
		func(ctx context.Context, _ struct{}, res *confirmation) (string, error) {
			if res == nil {
				return "", tool.Interrupt(ctx, nil)
			}
			runs = append(runs, res.Approved)
			return "ok", nil
		})
	defineToolThenFinishModel(reg, ai.NewToolRequestPart(&ai.ToolRequest{Name: "outer", Input: map[string]any{}}))
	resp, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithPrompt("go"), ai.WithTools(tl), ai.WithUse(retry))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	call := claim(t, tl, singleInterrupt(t, resp))
	if _, err := ai.Generate(context.Background(), reg,
		ai.WithModelName("test/model"), ai.WithMessages(resp.History()...),
		ai.WithTools(tl), ai.WithUse(retry), ai.WithResume(call.Restart(confirmation{Approved: true}))); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if diff := cmp.Diff([]bool{true, true}, runs); diff != "" {
		t.Errorf("resume per run (-want +got):\n%s", diff)
	}
}

// TestRestart_ReleasesOnlyTheHooksThatLetTheCallThrough pins that a restart
// answering the tool releases only the hooks the interrupted call records
// letting it through, and only while the input stands: a gate added to the
// chain since holds the call, and so does the recorded gate once the restart
// replaces the input.
func TestRestart_ReleasesOnlyTheHooksThatLetTheCallThrough(t *testing.T) {
	setup := func(t *testing.T) (*registry.Registry, *ai.ResumableToolAction[transferIn, transferOut, confirmation], ai.MiddlewareFunc, *[]string, *ai.ModelResponse, *ai.Part) {
		t.Helper()
		reg := newTransferTestRegistry(t)
		transfer, _ := interruptOnce(t, reg)
		var log []string
		a := gate(&log)
		generate := func(opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
			return ai.Generate(context.Background(), reg, append([]ai.GenerateOption{
				ai.WithModelName("test/model"), ai.WithTools(transfer), ai.WithUse(a),
			}, opts...)...)
		}
		resp, err := generate(ai.WithPrompt("transfer 200"))
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		restart, err := singleInterrupt(t, resp).ToToolRestart(map[string]any{"ok": true})
		if err != nil {
			t.Fatalf("ToToolRestart: %v", err)
		}
		resp, err = generate(ai.WithMessages(resp.History()...), ai.WithResume(restart))
		if !errors.Is(err, status.ErrFailedPrecondition) || resp == nil {
			t.Fatalf("answering the gate = (%v, %v), want the tool's own interrupt", resp, err)
		}
		asked := singleInterrupt(t, resp)
		if diff := cmp.Diff([]string{"inline"}, asked.Interrupt.ReleasedBy); diff != "" {
			t.Fatalf("ReleasedBy (-want +got):\n%s", diff)
		}
		return reg, transfer, a, &log, resp, asked
	}

	t.Run("a gate added since holds", func(t *testing.T) {
		reg, transfer, a, _, resp, asked := setup(t)
		var logB []string
		b := gate(&logB)
		call := claim(t, transfer, asked)
		_, err := ai.Generate(context.Background(), reg,
			ai.WithModelName("test/model"), ai.WithMessages(viaJSON(t, resp.History())...),
			ai.WithTools(transfer), ai.WithUse(a, b),
			ai.WithResume(call.Restart(confirmation{Approved: true})))
		if !errors.Is(err, status.ErrFailedPrecondition) {
			t.Fatalf("resume = %v, want the new gate's hold", err)
		}
		if diff := cmp.Diff([]string{"held"}, logB); diff != "" {
			t.Errorf("new gate saw (-want +got):\n%s", diff)
		}
	})

	t.Run("a replaced input faces the gate again", func(t *testing.T) {
		reg, transfer, a, log, resp, asked := setup(t)
		call := claim(t, transfer, asked)
		_, err := ai.Generate(context.Background(), reg,
			ai.WithModelName("test/model"), ai.WithMessages(resp.History()...),
			ai.WithTools(transfer), ai.WithUse(a),
			ai.WithResume(call.RestartWithInput(transferIn{Amount: 5000}, confirmation{Approved: true})))
		if !errors.Is(err, status.ErrFailedPrecondition) {
			t.Fatalf("resume = %v, want the gate's hold on the new input", err)
		}
		if got := (*log)[len(*log)-1]; got != "held" {
			t.Errorf("gate's last decision = %q, want held; log %v", got, *log)
		}
	})
}
