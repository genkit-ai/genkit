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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

// testTypedConfig is a provider-style config struct used to exercise the
// typed-config deserialization that NewTyped* constructors wrap around user
// functions.
type testTypedConfig struct {
	Temperature float64 `json:"temperature,omitempty"`
	MaxTokens   int     `json:"maxTokens,omitempty"`
}

// otherProviderConfig stands in for a different provider's config type. Its
// JSON shape is compatible with testTypedConfig so that requests carrying it
// pass the action's input schema validation and rejection happens on the Go
// type itself.
type otherProviderConfig struct {
	Temperature float64 `json:"temperature,omitempty"`
}

func TestModelTypedConfig(t *testing.T) {
	r := registry.New()

	var got testTypedConfig
	var gotReqConfig any
	m := NewTypedModel("test/typed-config", nil, func(ctx context.Context, req *ModelRequest, cfg testTypedConfig, cb ModelStreamCallback) (*ModelResponse, error) {
		got = cfg
		gotReqConfig = req.Config
		return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
	})
	m.Register(r)

	tests := []struct {
		name    string
		config  any
		want    testTypedConfig
		wantErr string
	}{
		{name: "nil config yields zero value", config: nil, want: testTypedConfig{}},
		{name: "exact type", config: testTypedConfig{Temperature: 0.5, MaxTokens: 10}, want: testTypedConfig{Temperature: 0.5, MaxTokens: 10}},
		{name: "pointer to exact type", config: &testTypedConfig{Temperature: 0.7}, want: testTypedConfig{Temperature: 0.7}},
		{name: "map is deserialized", config: map[string]any{"temperature": 0.9, "maxTokens": 5}, want: testTypedConfig{Temperature: 0.9, MaxTokens: 5}},
		{name: "mismatched struct type is rejected", config: otherProviderConfig{Temperature: 0.2}, wantErr: "invalid config type"},
		{name: "mismatched pointer type is rejected", config: &otherProviderConfig{Temperature: 0.2}, wantErr: "invalid config type"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got = testTypedConfig{}
			req := &ModelRequest{
				Messages: []*Message{NewUserTextMessage("hi")},
				Config:   tt.config,
			}
			_, err := m.Generate(context.Background(), req, nil)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("Generate() error = %v, want error containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Generate() unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("config = %+v, want %+v", got, tt.want)
			}
			if gotReqConfig != any(tt.want) {
				t.Errorf("req.Config = %#v, want normalized to %#v", gotReqConfig, tt.want)
			}
		})
	}
}

// TestModelConfigNormalizedBeforeBuiltins pins the boundary ordering: config
// resolution runs outermost in the model's built-in chain, so a mismatched
// config type is reported before capability validation (which would otherwise
// reject this request for containing unsupported media).
func TestModelConfigNormalizedBeforeBuiltins(t *testing.T) {
	r := registry.New()

	m := NewTypedModel("test/config-order", &ModelOptions{
		Supports: &ModelSupports{Multiturn: true}, // no media support
	}, func(ctx context.Context, req *ModelRequest, cfg testTypedConfig, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
	})
	m.Register(r)

	req := &ModelRequest{
		Messages: []*Message{NewUserMessage(NewMediaPart("image/png", "data:image/png;base64,aGk="))},
		Config:   otherProviderConfig{Temperature: 0.2},
	}
	_, err := m.Generate(context.Background(), req, nil)
	if err == nil || !strings.Contains(err.Error(), "invalid config type") {
		t.Fatalf("Generate() error = %v, want config type error before media support error", err)
	}
}

func TestTypedBackgroundModelMetadata(t *testing.T) {
	bm := NewTypedBackgroundModel("test/bg-metadata",
		&TypedBackgroundModelOptions{
			Metadata: map[string]any{
				"custom": "value",
				"model":  "caller values must not clobber the reserved keys",
			},
		},
		func(ctx context.Context, req *ModelRequest, cfg testTypedConfig) (*ModelOperation, error) {
			return &ModelOperation{ID: "op1"}, nil
		},
		func(ctx context.Context, op *ModelOperation) (*ModelOperation, error) {
			return op, nil
		},
		nil)

	metadata := bm.Desc().Metadata
	if got := metadata["custom"]; got != "value" {
		t.Errorf(`Metadata["custom"] = %v, want "value"`, got)
	}
	if got := metadata["type"]; got != api.ActionTypeBackgroundModel {
		t.Errorf(`Metadata["type"] = %v, want %v`, got, api.ActionTypeBackgroundModel)
	}
	if _, ok := metadata["model"].(map[string]any); !ok {
		t.Errorf(`Metadata["model"] = %v, want the built model info map`, metadata["model"])
	}
}

func TestBackgroundModelConfigValidation(t *testing.T) {
	r := registry.New()

	bm := NewTypedBackgroundModel("test/bg-typed-config", nil,
		func(ctx context.Context, req *ModelRequest, cfg testTypedConfig) (*ModelOperation, error) {
			return &ModelOperation{ID: "op1", Done: false}, nil
		},
		func(ctx context.Context, op *ModelOperation) (*ModelOperation, error) {
			return op, nil
		},
		nil)
	bm.Register(r)

	// A config that violates the inferred schema is rejected by input
	// validation at the action boundary, before conversion.
	req := &ModelRequest{
		Messages: []*Message{NewUserTextMessage("hi")},
		Config:   map[string]any{"unknownOption": true},
	}
	if _, err := bm.Start(context.Background(), req); err == nil || !strings.Contains(err.Error(), "schema") {
		t.Fatalf("Start() error = %v, want schema validation error", err)
	}

	// A valid map config passes validation and is converted.
	req = &ModelRequest{
		Messages: []*Message{NewUserTextMessage("hi")},
		Config:   map[string]any{"temperature": 0.5},
	}
	if _, err := bm.Start(context.Background(), req); err != nil {
		t.Fatalf("Start() unexpected error: %v", err)
	}
}

func TestModelConfigSchemaInference(t *testing.T) {
	newFn := func() TypedModelFunc[testTypedConfig] {
		return func(ctx context.Context, req *ModelRequest, cfg testTypedConfig, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
		}
	}

	configSchemaOf := func(t *testing.T, m *ModelAction) any {
		t.Helper()
		desc := m.Desc()
		modelMeta, ok := desc.Metadata["model"].(map[string]any)
		if !ok {
			t.Fatalf("missing model metadata: %+v", desc.Metadata)
		}
		return modelMeta["customOptions"]
	}

	t.Run("inferred from Config type", func(t *testing.T) {
		m := NewTypedModel("test/inferred-schema", nil, newFn())
		schema, ok := configSchemaOf(t, m).(map[string]any)
		if !ok {
			t.Fatalf("customOptions = %v, want inferred schema map", configSchemaOf(t, m))
		}
		props, ok := schema["properties"].(map[string]any)
		if !ok {
			t.Fatalf("schema has no properties: %v", schema)
		}
		if _, ok := props["temperature"]; !ok {
			t.Errorf("inferred schema missing temperature property: %v", props)
		}
	})

	t.Run("explicit ConfigSchema wins", func(t *testing.T) {
		override := map[string]any{"type": "object", "properties": map[string]any{"custom": map[string]any{"type": "string"}}}
		m := NewTypedModel("test/override-schema", &ModelOptions{ConfigSchema: override}, newFn())
		schema, ok := configSchemaOf(t, m).(map[string]any)
		if !ok {
			t.Fatalf("customOptions missing")
		}
		props := schema["properties"].(map[string]any)
		if _, ok := props["custom"]; !ok {
			t.Errorf("override schema not used: %v", schema)
		}
	})

	t.Run("any config infers no schema", func(t *testing.T) {
		m := NewTypedModel("test/any-schema", nil, func(ctx context.Context, req *ModelRequest, cfg any, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
		})
		if s, _ := configSchemaOf(t, m).(map[string]any); len(s) != 0 {
			t.Errorf("customOptions = %v, want no inferred schema for any config", s)
		}
	})

	t.Run("deprecated NewModel behaves like any config", func(t *testing.T) {
		m := NewModel("test/legacy-schema", nil, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
		})
		if s, _ := configSchemaOf(t, m.(*ModelAction)).(map[string]any); len(s) != 0 {
			t.Errorf("customOptions = %v, want no inferred schema for legacy models", s)
		}
	})
}

// TestDeprecatedModelPassesConfigThrough pins the compatibility contract of
// the deprecated untyped constructors: any config value, including another
// provider's struct, still reaches the model function unconverted.
func TestDeprecatedModelPassesConfigThrough(t *testing.T) {
	r := registry.New()

	var gotConfig any
	m := defineModel(r, "test/legacy-config", nil, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		gotConfig = req.Config
		return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
	})

	cfg := otherProviderConfig{Temperature: 0.4}
	req := &ModelRequest{
		Messages: []*Message{NewUserTextMessage("hi")},
		Config:   cfg,
	}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatalf("Generate() unexpected error: %v", err)
	}
	if gotConfig != any(cfg) {
		t.Errorf("req.Config = %#v, want passed through as %#v", gotConfig, cfg)
	}
}

func TestEmbedderTypedConfig(t *testing.T) {
	r := registry.New()

	var got testTypedConfig
	var gotReqOptions any
	e := NewTypedEmbedder("test/typed-config-embedder", nil, func(ctx context.Context, req *EmbedRequest, cfg testTypedConfig) (*EmbedResponse, error) {
		got = cfg
		gotReqOptions = req.Options
		return &EmbedResponse{}, nil
	})
	e.Register(r)

	req := &EmbedRequest{Options: map[string]any{"maxTokens": 7}}
	if _, err := e.Embed(context.Background(), req); err != nil {
		t.Fatalf("Embed() unexpected error: %v", err)
	}
	if got.MaxTokens != 7 {
		t.Errorf("config = %+v, want MaxTokens 7", got)
	}
	if gotReqOptions != any(got) {
		t.Errorf("req.Options = %#v, want normalized to %#v", gotReqOptions, got)
	}

	req = &EmbedRequest{Options: otherProviderConfig{Temperature: 0.1}}
	if _, err := e.Embed(context.Background(), req); err == nil || !strings.Contains(err.Error(), "invalid config type") {
		t.Fatalf("Embed() error = %v, want invalid config type error", err)
	}
}

func TestEvaluatorTypedConfig(t *testing.T) {
	r := registry.New()

	var got testTypedConfig
	e := NewTypedEvaluator("test/typed-config-evaluator", nil, func(ctx context.Context, req *EvaluatorCallbackRequest, cfg testTypedConfig) (*EvaluatorCallbackResponse, error) {
		got = cfg
		return &EvaluatorCallbackResponse{
			TestCaseId: req.Input.TestCaseId,
			Evaluation: []Score{{Id: "s", Score: 1, Status: ScoreStatusPass.String()}},
		}, nil
	})
	e.Register(r)

	req := &EvaluatorRequest{
		Dataset:      []*Example{{TestCaseId: "tc1", Input: "in"}},
		EvaluationId: "run1",
		Options:      &testTypedConfig{Temperature: 0.3},
	}
	if _, err := e.Evaluate(context.Background(), req); err != nil {
		t.Fatalf("Evaluate() unexpected error: %v", err)
	}
	if got.Temperature != 0.3 {
		t.Errorf("config = %+v, want Temperature 0.3", got)
	}

	var gotBatch testTypedConfig
	be := NewTypedBatchEvaluator("test/typed-config-batch-evaluator", nil, func(ctx context.Context, req *EvaluatorRequest, cfg testTypedConfig) (*EvaluatorResponse, error) {
		gotBatch = cfg
		return &EvaluatorResponse{}, nil
	})
	be.Register(r)
	breq := &EvaluatorRequest{
		Dataset:      []*Example{{TestCaseId: "tc1", Input: "in"}},
		EvaluationId: "run2",
		Options:      map[string]any{"temperature": 0.8},
	}
	if _, err := be.Evaluate(context.Background(), breq); err != nil {
		t.Fatalf("Evaluate() unexpected error: %v", err)
	}
	if gotBatch.Temperature != 0.8 {
		t.Errorf("config = %+v, want Temperature 0.8", gotBatch)
	}
}
