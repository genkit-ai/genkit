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

package ai

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
	"github.com/google/go-cmp/cmp"
)

// newTestRegistry creates a fresh registry for testing with formats configured.
func newTestRegistry(t *testing.T) api.Registry {
	t.Helper()
	r := registry.New()
	ConfigureFormats(r)
	return r
}

// fakeModelConfig holds configuration for creating a fake model.
type fakeModelConfig struct {
	name     string
	supports *ModelSupports
	handler  func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error)
}

// defaultModelSupports returns a ModelSupports with common capabilities enabled.
func defaultModelSupports() *ModelSupports {
	return &ModelSupports{
		Tools:       true,
		Multiturn:   true,
		ToolChoice:  true,
		SystemRole:  true,
		Constrained: ConstrainedSupportAll,
	}
}

// defineFakeModel creates a configurable fake model for testing.
func defineFakeModel(t *testing.T, r api.Registry, cfg fakeModelConfig) Model {
	t.Helper()
	if cfg.name == "" {
		cfg.name = "test/fakeModel"
	}
	if cfg.supports == nil {
		cfg.supports = defaultModelSupports()
	}
	if cfg.handler == nil {
		cfg.handler = func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("fake response"),
			}, nil
		}
	}
	return DefineModel(r, cfg.name, &ModelOptions{Supports: cfg.supports}, cfg.handler)
}

// echoModelHandler creates a handler that echoes back information about the request.
// Useful for verifying that options are properly passed through.
func echoModelHandler() func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
	return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		var parts []string

		// Echo messages
		for _, msg := range req.Messages {
			parts = append(parts, fmt.Sprintf("%s: %s", msg.Role, msg.Text()))
		}

		// Echo config if present
		if req.Config != nil {
			if cfg, ok := req.Config.(*GenerationCommonConfig); ok {
				parts = append(parts, fmt.Sprintf("temp=%.1f", cfg.Temperature))
			}
		}

		// Echo tool count
		if len(req.Tools) > 0 {
			parts = append(parts, fmt.Sprintf("tools=%d", len(req.Tools)))
		}

		// Echo tool choice
		if req.ToolChoice != "" {
			parts = append(parts, fmt.Sprintf("toolChoice=%s", req.ToolChoice))
		}

		// Echo docs count
		if len(req.Docs) > 0 {
			parts = append(parts, fmt.Sprintf("docs=%d", len(req.Docs)))
		}

		return &ModelResponse{
			Request: req,
			Message: NewModelTextMessage(strings.Join(parts, "; ")),
		}, nil
	}
}

// capturingModelHandler returns a handler that captures the request for inspection.
func capturingModelHandler(captured **ModelRequest) func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
	return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		*captured = req
		return &ModelResponse{
			Request: req,
			Message: NewModelTextMessage("captured"),
		}, nil
	}
}

// streamingModelHandler creates a handler that sends chunks before returning.
func streamingModelHandler(chunks []string, finalText string) func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
	return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		if cb != nil {
			for _, chunk := range chunks {
				if err := cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewTextPart(chunk)},
				}); err != nil {
					return nil, err
				}
			}
		}
		return &ModelResponse{
			Request: req,
			Message: NewModelTextMessage(finalText),
		}, nil
	}
}

// jsonModelHandler creates a handler that returns JSON output.
func jsonModelHandler(jsonOutput string) func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
	return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{
			Request: req,
			Message: &Message{
				Role:    RoleModel,
				Content: []*Part{NewJSONPart(jsonOutput)},
			},
		}, nil
	}
}

// toolCallingModelHandler creates a handler that makes a tool call on first request,
// then returns the final response after receiving the tool response.
func toolCallingModelHandler(toolName string, toolInput map[string]any, finalResponse string) func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
	callCount := 0
	return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		callCount++

		// Check if we already have a tool response
		hasToolResponse := false
		for _, msg := range req.Messages {
			for _, part := range msg.Content {
				if part.IsToolResponse() {
					hasToolResponse = true
					break
				}
			}
		}

		if !hasToolResponse && len(req.Tools) > 0 {
			// First call - request tool execution
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role: RoleModel,
					Content: []*Part{NewToolRequestPart(&ToolRequest{
						Name:  toolName,
						Input: toolInput,
					})},
				},
			}, nil
		}

		// Tool response received or no tools - return final response
		return &ModelResponse{
			Request: req,
			Message: NewModelTextMessage(finalResponse),
		}, nil
	}
}

// cmpPartEqual is a Part comparator for cmp.Diff that compares essential fields.
func cmpPartEqual(a, b *Part) bool {
	if a == nil || b == nil {
		return a == b
	}
	if a.Kind != b.Kind {
		return false
	}
	if a.Text != b.Text {
		return false
	}
	if a.ContentType != b.ContentType {
		return false
	}
	return true
}

// cmpPartComparer returns a cmp.Option for comparing Parts.
func cmpPartComparer() cmp.Option {
	return cmp.Comparer(cmpPartEqual)
}

// assertEqual compares two values and reports differences.
func assertEqual[T any](t *testing.T, got, want T, opts ...cmp.Option) {
	t.Helper()
	if diff := cmp.Diff(want, got, opts...); diff != "" {
		t.Errorf("mismatch (-want +got):\n%s", diff)
	}
}

// assertError verifies error is non-nil and contains expected substring.
func assertError(t *testing.T, err error, wantContains string) {
	t.Helper()
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), wantContains) {
		t.Errorf("error %q does not contain %q", err.Error(), wantContains)
	}
}

// assertNoError fails the test if err is not nil.
func assertNoError(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// assertPanic verifies that fn panics and the panic value contains wantContains.
func assertPanic(t *testing.T, fn func(), wantContains string) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic, got none")
		}
		msg := fmt.Sprint(r)
		if !strings.Contains(msg, wantContains) {
			t.Errorf("panic %q does not contain %q", msg, wantContains)
		}
	}()
	fn()
}

// assertNoPanic verifies that fn does not panic.
func assertNoPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("unexpected panic: %v", r)
		}
	}()
	fn()
}

// defineFakeTool creates a simple tool for testing.
func defineFakeTool(t *testing.T, r api.Registry, name, description string) Tool {
	t.Helper()
	return DefineTool(r, name, description,
		func(ctx *ToolContext, input struct {
			Value string `json:"value"`
		}) (string, error) {
			return "tool result: " + input.Value, nil
		})
}

// defineFakeEmbedder creates a simple embedder for testing.
func defineFakeEmbedder(t *testing.T, r api.Registry, name string) Embedder {
	t.Helper()
	return DefineEmbedder(r, name, nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
		embeddings := make([]*Embedding, len(req.Input))
		for i := range req.Input {
			embeddings[i] = &Embedding{
				Embedding: []float32{0.1, 0.2, 0.3},
			}
		}
		return &EmbedResponse{Embeddings: embeddings}, nil
	})
}

// defineFakeRetriever creates a simple retriever for testing.
func defineFakeRetriever(t *testing.T, r api.Registry, name string, docs []*Document) Retriever {
	t.Helper()
	return DefineRetriever(r, name, nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
		return &RetrieverResponse{Documents: docs}, nil
	})
}
