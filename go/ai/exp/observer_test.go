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
	"errors"
	"slices"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/tracing"
)

// TestObserverSeesEveryModelCall pins where calls are reported: in the model
// action, which every route reaches. A report from the generate loop would
// miss the model the tool calls directly, and an agent boundary must not hide
// calls from an observer around it. Neither model sets Response.Request, which
// the observer still gets.
func TestObserverSeesEveryModelCall(t *testing.T) {
	reg := newTestRegistry(t)
	ai.ConfigureFormats(reg)
	inner := defineTestModel(reg, "test/inner", nil, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Message: ai.NewModelTextMessage("inner")}, nil
	})
	ask := defineTestTool(reg, "ask", "Asks another model.", func(tc *ai.ToolContext, _ any) (string, error) {
		resp, err := inner.Generate(tc, &ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("q")}}, nil)
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})
	defineTestModel(reg, "test/outer", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			if last := req.Messages[len(req.Messages)-1]; last.Role == ai.RoleTool {
				return &ai.ModelResponse{Message: ai.NewModelTextMessage("done")}, nil
			}
			return &ai.ModelResponse{Message: &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "ask"}),
			}}}, nil
		})
	af := DefineCustomAgent(reg, "observed",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				_, err := ai.Generate(ctx, reg, ai.WithModelName("test/outer"), ai.WithTools(ask), ai.WithPrompt("go"))
				return nil, err
			})
		})

	var (
		mu     sync.Mutex
		models []string
	)
	ctx := tracing.WithInstrumentation(t.Context(), Observer{
		ModelDone: func(_ context.Context, call *ModelCall) {
			mu.Lock()
			defer mu.Unlock()
			models = append(models, call.Model)
			if call.Response == nil || call.Response.Request == nil {
				t.Errorf("%s: Response = %+v, want one carrying the request", call.Model, call.Response)
			}
		},
	})
	if _, err := af.RunText(ctx, "go"); err != nil {
		t.Fatalf("RunText: %v", err)
	}
	slices.Sort(models)
	if want := []string{"test/inner", "test/outer", "test/outer"}; !slices.Equal(models, want) {
		t.Errorf("observed %v, want %v", models, want)
	}
}

// TestObserverSeesAFailedCall pins that a call that fails without a response
// still reaches the observer with one: the failure record, with the request
// and the error.
func TestObserverSeesAFailedCall(t *testing.T) {
	reg := newTestRegistry(t)
	boom := errors.New("boom")
	m := defineTestModel(reg, "test/failing", nil, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, boom
	})
	var got *ModelCall
	ctx := tracing.WithInstrumentation(t.Context(), Observer{ModelDone: func(_ context.Context, call *ModelCall) { got = call }})
	req := &ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("hi")}}
	if _, err := m.Generate(ctx, req, nil); !errors.Is(err, boom) {
		t.Fatalf("Generate error = %v, want %v", err, boom)
	}
	if got == nil || !errors.Is(got.Err, boom) {
		t.Fatalf("observed %+v, want the call with its error", got)
	}
	if r := got.Response; r == nil || r.FinishReason != ai.FinishReasonFailed || r.Error == nil || r.Request == nil {
		t.Errorf("Response = %+v, want a failure record with the request", r)
	}
}
