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

package ai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/registry"
)

// The concrete primitives redeclare the methods promoted from their embedded
// core action so that those methods, and the interfaces they satisfy, appear
// in the package documentation. A redeclared method shadows the promoted one,
// so a forwarder that calls itself instead of the embedded action recurses
// until the stack overflows. Exercise every forwarder: Generate, Embed,
// Retrieve, and Evaluate are covered elsewhere, but RunJSON and
// RunJSONWithTelemetry are not called anywhere else in the suite, which is
// exactly where such a mistake would survive.
func TestPrimitiveMethodsForwardToEmbeddedAction(t *testing.T) {
	modelFn := func(ctx context.Context, req *ModelRequest, _ any, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
	}

	tests := []struct {
		name   string
		action api.Action
		atype  api.ActionType
		input  string
	}{
		{
			name:   "model",
			action: NewModelAction("test/model", nil, modelFn),
			atype:  api.ActionTypeModel,
			input:  `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`,
		},
		{
			name: "embedder",
			action: NewEmbedderAction("test/embedder", nil, func(ctx context.Context, req *EmbedRequest, _ any) (*EmbedResponse, error) {
				return &EmbedResponse{Embeddings: []*Embedding{{Embedding: []float32{1, 2}}}}, nil
			}),
			atype: api.ActionTypeEmbedder,
			input: `{"input":[{"content":[{"text":"hi"}]}]}`,
		},
		{
			name: "retriever",
			action: NewRetrieverAction("test/retriever", nil, func(ctx context.Context, req *RetrieverRequest, _ any) (*RetrieverResponse, error) {
				return &RetrieverResponse{Documents: []*Document{DocumentFromText("doc", nil)}}, nil
			}),
			atype: api.ActionTypeRetriever,
			input: `{"query":{"content":[{"text":"hi"}]}}`,
		},
		{
			name: "evaluator",
			action: NewEvaluatorAction("test/evaluator", &EvaluatorOptions{DisplayName: "Test", Definition: "Test"},
				func(ctx context.Context, req *EvaluatorCallbackRequest, _ any) (*EvaluatorCallbackResponse, error) {
					return &EvaluatorCallbackResponse{
						TestCaseId: req.Input.TestCaseId,
						Evaluation: []Score{{Id: "score", Score: 1, Status: "PASS"}},
					}, nil
				}),
			atype: api.ActionTypeEvaluator,
			input: `{"dataset":[{"testCaseId":"case-1","input":"x"}],"evalRunId":"run-1"}`,
		},
		{
			name: "tool",
			action: NewTool("test/tool", "a test tool", func(ctx *ToolContext, in string) (string, error) {
				return in + "!", nil
			}),
			atype: api.ActionTypeToolV2,
			input: `"hi"`,
		},
		{
			name: "background model",
			action: NewBackgroundModelAction("test/background", nil,
				func(ctx context.Context, req *ModelRequest, _ any) (*ModelOperation, error) {
					return &ModelOperation{ID: "op-1"}, nil
				},
				func(ctx context.Context, op *ModelOperation) (*ModelOperation, error) {
					return &ModelOperation{ID: op.ID, Done: true}, nil
				}),
			atype: api.ActionTypeBackgroundModel,
			input: `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := registry.New()
			tt.action.Register(r)

			desc := tt.action.Desc()
			if desc.Name != tt.action.Name() {
				t.Errorf("Desc().Name = %q, Name() = %q; the two forwarders disagree", desc.Name, tt.action.Name())
			}
			if desc.Type != tt.atype {
				t.Errorf("Desc().Type = %q, want %q", desc.Type, tt.atype)
			}
			if r.LookupAction(desc.Key) == nil {
				t.Errorf("Register() did not reach the registry: no action at key %q", desc.Key)
			}

			out, err := tt.action.RunJSON(context.Background(), json.RawMessage(tt.input), nil)
			if err != nil {
				t.Fatalf("RunJSON() error: %v", err)
			}
			if len(out) == 0 {
				t.Error("RunJSON() returned no output")
			}

			res, err := tt.action.RunJSONWithTelemetry(context.Background(), json.RawMessage(tt.input), nil)
			if err != nil {
				t.Fatalf("RunJSONWithTelemetry() error: %v", err)
			}
			if res == nil || len(res.Result) == 0 {
				t.Error("RunJSONWithTelemetry() returned no result")
			}
		})
	}
}

// A nil receiver reaches a forwarder whenever a caller holds an uninitialized
// primitive, and dereferencing the embedded action would panic. Every
// forwarder that can report an error returns INVALID_ARGUMENT instead, the way
// Generate, Embed, Evaluate, Retrieve, and RunRawMultipart already do.
func TestForwardersRejectNilReceivers(t *testing.T) {
	ctx := context.Background()
	in := json.RawMessage(`{}`)

	tests := []struct {
		name string
		call func() error
	}{
		{"model RunJSON", func() error { var m *ModelAction; _, err := m.RunJSON(ctx, in, nil); return err }},
		{"model RunJSONWithTelemetry", func() error {
			var m *ModelAction
			_, err := m.RunJSONWithTelemetry(ctx, in, nil)
			return err
		}},
		{"embedder RunJSON", func() error { var e *EmbedderAction; _, err := e.RunJSON(ctx, in, nil); return err }},
		{"embedder RunJSONWithTelemetry", func() error {
			var e *EmbedderAction
			_, err := e.RunJSONWithTelemetry(ctx, in, nil)
			return err
		}},
		{"retriever RunJSON", func() error { var r *RetrieverAction; _, err := r.RunJSON(ctx, in, nil); return err }},
		{"retriever RunJSONWithTelemetry", func() error {
			var r *RetrieverAction
			_, err := r.RunJSONWithTelemetry(ctx, in, nil)
			return err
		}},
		{"evaluator RunJSON", func() error { var e *EvaluatorAction; _, err := e.RunJSON(ctx, in, nil); return err }},
		{"evaluator RunJSONWithTelemetry", func() error {
			var e *EvaluatorAction
			_, err := e.RunJSONWithTelemetry(ctx, in, nil)
			return err
		}},
		{"tool RunJSON", func() error {
			var tl *ToolAction[any, any]
			_, err := tl.RunJSON(ctx, in, nil)
			return err
		}},
		{"tool RunJSONWithTelemetry", func() error {
			var tl *ToolAction[any, any]
			_, err := tl.RunJSONWithTelemetry(ctx, in, nil)
			return err
		}},
		{"background RunJSON", func() error {
			var b *BackgroundModelAction
			_, err := b.RunJSON(ctx, in, nil)
			return err
		}},
		{"background RunJSONWithTelemetry", func() error {
			var b *BackgroundModelAction
			_, err := b.RunJSONWithTelemetry(ctx, in, nil)
			return err
		}},
		{"background Start", func() error {
			var b *BackgroundModelAction
			_, err := b.Start(ctx, &ModelRequest{})
			return err
		}},
		{"background Check", func() error {
			var b *BackgroundModelAction
			_, err := b.Check(ctx, &ModelOperation{})
			return err
		}},
		{"background Cancel", func() error {
			var b *BackgroundModelAction
			_, err := b.Cancel(ctx, &ModelOperation{})
			return err
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call()
			if err == nil {
				t.Fatal("got nil error, want an error on a nil receiver")
			}
			if !errors.Is(err, status.ErrInvalidArgument) {
				t.Errorf("err = %v, want it to match status.ErrInvalidArgument", err)
			}
		})
	}
}

// A background model's lifecycle methods are forwarders too, and Cancel and
// SupportsCancel have no other coverage on the concrete type.
func TestBackgroundModelActionLifecycleForwarders(t *testing.T) {
	startFn := func(ctx context.Context, req *ModelRequest, _ any) (*ModelOperation, error) {
		return &ModelOperation{ID: "op-1"}, nil
	}
	checkFn := func(ctx context.Context, op *ModelOperation) (*ModelOperation, error) {
		return &ModelOperation{ID: op.ID, Done: true}, nil
	}

	t.Run("without cancel", func(t *testing.T) {
		b := NewBackgroundModelAction("test/no-cancel", nil, startFn, checkFn)
		if b.SupportsCancel() {
			t.Error("SupportsCancel() = true, want false when no cancel function is given")
		}
		if _, err := b.Cancel(context.Background(), &ModelOperation{ID: "op-1"}); err == nil {
			t.Error("Cancel() error = nil, want an error when cancellation is unsupported")
		}
	})

	t.Run("with cancel", func(t *testing.T) {
		b := NewBackgroundModelAction("test/cancel", &BackgroundModelOptions{
			Cancel: func(ctx context.Context, op *ModelOperation) (*ModelOperation, error) {
				return &ModelOperation{ID: op.ID, Done: true}, nil
			},
		}, startFn, checkFn)

		if !b.SupportsCancel() {
			t.Error("SupportsCancel() = false, want true")
		}

		op, err := b.Start(context.Background(), &ModelRequest{Messages: []*Message{NewUserTextMessage("hi")}})
		if err != nil {
			t.Fatalf("Start() error: %v", err)
		}
		if op.ID != "op-1" {
			t.Errorf("Start() ID = %q, want %q", op.ID, "op-1")
		}

		checked, err := b.Check(context.Background(), op)
		if err != nil {
			t.Fatalf("Check() error: %v", err)
		}
		if !checked.Done {
			t.Error("Check() Done = false, want true")
		}

		cancelled, err := b.Cancel(context.Background(), op)
		if err != nil {
			t.Fatalf("Cancel() error: %v", err)
		}
		if !cancelled.Done {
			t.Error("Cancel() Done = false, want true")
		}
	})
}
