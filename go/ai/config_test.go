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
// typed-config deserialization that New*Action constructors wrap around user
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
	m := NewModelAction("test/typed-config", nil, func(ctx context.Context, req *ModelRequest, cfg testTypedConfig, cb ModelStreamCallback) (*ModelResponse, error) {
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

	m := NewModelAction("test/config-order", &ModelOptions{
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

func TestModelActionMetadata(t *testing.T) {
	m := NewModelAction("test/model-metadata",
		&ModelOptions{
			Metadata: map[string]any{
				"custom": "value",
				"model":  "caller values must not clobber the reserved keys",
			},
		},
		func(ctx context.Context, req *ModelRequest, cfg testTypedConfig, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
		})

	metadata := m.Desc().Metadata
	if got := metadata["custom"]; got != "value" {
		t.Errorf(`Metadata["custom"] = %v, want "value"`, got)
	}
	if got := metadata["type"]; got != api.ActionTypeModel {
		t.Errorf(`Metadata["type"] = %v, want %v`, got, api.ActionTypeModel)
	}
	if _, ok := metadata["model"].(map[string]any); !ok {
		t.Errorf(`Metadata["model"] = %v, want the built model info map`, metadata["model"])
	}
}

func TestEmbedderActionMetadata(t *testing.T) {
	e := NewEmbedderAction("test/embedder-metadata",
		&EmbedderOptions{
			Metadata: map[string]any{
				"custom": "value",
				"info":   "caller values must not clobber the reserved keys",
			},
		},
		func(ctx context.Context, req *EmbedRequest, cfg testTypedConfig) (*EmbedResponse, error) {
			return &EmbedResponse{}, nil
		})

	metadata := e.Desc().Metadata
	if got := metadata["custom"]; got != "value" {
		t.Errorf(`Metadata["custom"] = %v, want "value"`, got)
	}
	if got := metadata["type"]; got != api.ActionTypeEmbedder {
		t.Errorf(`Metadata["type"] = %v, want %v`, got, api.ActionTypeEmbedder)
	}
	if _, ok := metadata["info"].(map[string]any); !ok {
		t.Errorf(`Metadata["info"] = %v, want the built embedder info map`, metadata["info"])
	}
}

func TestEvaluatorActionMetadata(t *testing.T) {
	e := NewEvaluatorAction("test/evaluator-metadata",
		&EvaluatorOptions{
			Metadata: map[string]any{
				"custom":    "value",
				"evaluator": "caller values must not clobber the reserved keys",
			},
		},
		func(ctx context.Context, req *EvaluatorCallbackRequest, cfg testTypedConfig) (*EvaluatorCallbackResponse, error) {
			return &EvaluatorCallbackResponse{}, nil
		})

	metadata := e.Desc().Metadata
	if got := metadata["custom"]; got != "value" {
		t.Errorf(`Metadata["custom"] = %v, want "value"`, got)
	}
	if got := metadata["type"]; got != api.ActionTypeEvaluator {
		t.Errorf(`Metadata["type"] = %v, want %v`, got, api.ActionTypeEvaluator)
	}
	evalInfo, ok := metadata["evaluator"].(map[string]any)
	if !ok {
		t.Errorf(`Metadata["evaluator"] = %v, want the built evaluator info map`, metadata["evaluator"])
	} else if evalInfo["customOptions"] == nil {
		t.Errorf(`Metadata["evaluator"]["customOptions"] = nil, want the inferred config schema advertised`)
	}
}

func TestBackgroundModelActionMetadata(t *testing.T) {
	bm := NewBackgroundModelAction("test/bg-metadata",
		&BackgroundModelOptions{
			ModelOptions: ModelOptions{
				Metadata: map[string]any{"inner": "kept", "shared": "inner"},
			},
			Metadata: map[string]any{
				"custom": "value",
				"shared": "outer",
				"model":  "caller values must not clobber the reserved keys",
			},
		},
		func(ctx context.Context, req *ModelRequest, cfg testTypedConfig) (*ModelOperation, error) {
			return &ModelOperation{ID: "op1"}, nil
		},
		func(ctx context.Context, op *ModelOperation) (*ModelOperation, error) {
			return op, nil
		})

	metadata := bm.Desc().Metadata
	if got := metadata["custom"]; got != "value" {
		t.Errorf(`Metadata["custom"] = %v, want "value"`, got)
	}
	if got := metadata["inner"]; got != "kept" {
		t.Errorf(`Metadata["inner"] = %v, want the embedded ModelOptions metadata carried through`, got)
	}
	if got := metadata["shared"]; got != "outer" {
		t.Errorf(`Metadata["shared"] = %v, want the top-level Metadata to win`, got)
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

	bm := NewBackgroundModelAction("test/bg-typed-config", nil,
		func(ctx context.Context, req *ModelRequest, cfg testTypedConfig) (*ModelOperation, error) {
			return &ModelOperation{ID: "op1", Done: false}, nil
		},
		func(ctx context.Context, op *ModelOperation) (*ModelOperation, error) {
			return op, nil
		})
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

// TestModelVersionWithTypedConfig pins that the framework-level "version" key
// stays admissible on the config slot even when the inferred Config type has
// no version field: version validation must see the key before conversion
// drops it, not have it rejected by the closed inferred schema.
func TestModelVersionWithTypedConfig(t *testing.T) {
	r := registry.New()
	m := NewModelAction("test/versioned", &ModelOptions{Versions: []string{"v1"}},
		func(ctx context.Context, req *ModelRequest, cfg testTypedConfig, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
		})
	m.Register(r)

	req := &ModelRequest{
		Messages: []*Message{NewUserTextMessage("hi")},
		Config:   map[string]any{"version": "v1", "temperature": 0.5},
	}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatalf("Generate() with a valid version: %v", err)
	}

	req = &ModelRequest{
		Messages: []*Message{NewUserTextMessage("hi")},
		Config:   map[string]any{"version": "v9"},
	}
	_, err := m.Generate(context.Background(), req, nil)
	if err == nil || !strings.Contains(err.Error(), "version") {
		t.Fatalf("Generate() error = %v, want a version validation error", err)
	}
	if strings.Contains(err.Error(), "schema") {
		t.Fatalf("Generate() error = %v, want the version error, not a schema rejection", err)
	}

	if _, ok := advertisedConfigProps(t, m)["version"]; !ok {
		t.Error("a model declaring Versions should advertise the version property")
	}
}

type nilableSub struct {
	A string `json:"a"`
}

// nilableConfig has one nilable field of each kind without omitempty, so its
// own partially filled value marshals nulls into the config slot.
type nilableConfig struct {
	Sub  *nilableSub       `json:"sub"`
	List []string          `json:"list"`
	M    map[string]string `json:"m"`
	Name *string           `json:"name"`
	T    float64           `json:"t,omitempty"`
}

// TestInferredConfigSchemaToleratesNulls pins that an inferred config schema
// accepts the config type's own partial value. A nilable field lacking
// omitempty marshals to null, so enforcing its declared type would reject
// every request that leaves it unset. Widening must not cost the checks that
// make enforcement worth having, so unknown and mistyped fields still fail.
func TestInferredConfigSchemaToleratesNulls(t *testing.T) {
	r := registry.New()
	m := NewModelAction("test/nilable", &ModelOptions{},
		func(ctx context.Context, req *ModelRequest, cfg nilableConfig, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
		})
	m.Register(r)

	generate := func(config any) error {
		_, err := m.Generate(context.Background(), &ModelRequest{
			Messages: []*Message{NewUserTextMessage("hi")},
			Config:   config,
		}, nil)
		return err
	}

	if err := generate(nilableConfig{T: 0.5}); err != nil {
		t.Errorf("a partial config of the action's own type was rejected: %v", err)
	}
	if err := generate(map[string]any{"sub": nil, "list": nil, "name": nil}); err != nil {
		t.Errorf("explicit nulls were rejected: %v", err)
	}
	if err := generate(map[string]any{"bogus": 1}); err == nil {
		t.Error("an unknown config field should still be rejected")
	}
	if err := generate(map[string]any{"t": "hot"}); err == nil {
		t.Error("a mistyped config value should still be rejected")
	}
	if err := generate(map[string]any{"sub": map[string]any{"nope": 1}}); err == nil {
		t.Error("an unknown nested field should still be rejected")
	}

	// The advertised schema stays clean: null tolerance is an enforcement
	// concern, and the dev UI should show the field's real type.
	props := advertisedConfigProps(t, m)
	sub, ok := props["sub"].(map[string]any)
	if !ok {
		t.Fatalf("advertised properties.sub = %v, want a schema", props["sub"])
	}
	if sub["type"] != "object" {
		t.Errorf("advertised properties.sub.type = %v, want a plain \"object\"", sub["type"])
	}
}

// TestModelWithoutVersionsHidesVersionProperty is the other half of
// [TestModelVersionWithTypedConfig]: version validation rejects every value
// when a model declares no versions, so advertising the key there would offer
// a config field that can only ever error.
func TestModelWithoutVersionsHidesVersionProperty(t *testing.T) {
	m := NewModelAction("test/unversioned", &ModelOptions{},
		func(ctx context.Context, req *ModelRequest, cfg testTypedConfig, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
		})

	if _, ok := advertisedConfigProps(t, m)["version"]; ok {
		t.Error("a model declaring no Versions should not advertise the version property")
	}
}

// advertisedConfigProps returns the properties of the config schema a model
// advertises to the dev UI.
func advertisedConfigProps(t *testing.T, m *ModelAction) map[string]any {
	t.Helper()
	model, ok := m.Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing, got %v", m.Desc().Metadata)
	}
	schema, ok := model["customOptions"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions is %T, want map[string]any", model["customOptions"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions has no properties map, got %v", schema)
	}
	return props
}

// strictFieldConfig has fields without omitempty; the inferred schema must
// not turn them into required config keys, since a config is partial by
// nature.
type strictFieldConfig struct {
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"maxTokens"`
}

func TestPartialConfigNotRequired(t *testing.T) {
	r := registry.New()
	m := NewModelAction("test/partial-config", nil,
		func(ctx context.Context, req *ModelRequest, cfg strictFieldConfig, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
		})
	m.Register(r)

	req := &ModelRequest{
		Messages: []*Message{NewUserTextMessage("hi")},
		Config:   map[string]any{"temperature": 0.5},
	}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatalf("Generate() with a partial config: %v", err)
	}

	// Stripping required must not open the schema: unknown keys stay rejected.
	req = &ModelRequest{
		Messages: []*Message{NewUserTextMessage("hi")},
		Config:   map[string]any{"unknownOption": true},
	}
	if _, err := m.Generate(context.Background(), req, nil); err == nil || !strings.Contains(err.Error(), "schema") {
		t.Fatalf("Generate() error = %v, want schema validation error for an unknown key", err)
	}
}

// TestPointerConfigNilStaysNil pins that a request carrying no config keeps a
// nil config slot when Config is a pointer type: boxing the typed nil would
// make req.Config compare non-nil while dereferencing it still panics.
func TestPointerConfigNilStaysNil(t *testing.T) {
	r := registry.New()
	var gotCfg *testTypedConfig
	var gotReqConfig any
	m := NewModelAction("test/pointer-config", nil,
		func(ctx context.Context, req *ModelRequest, cfg *testTypedConfig, cb ModelStreamCallback) (*ModelResponse, error) {
			gotCfg = cfg
			gotReqConfig = req.Config
			return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
		})
	m.Register(r)

	req := &ModelRequest{Messages: []*Message{NewUserTextMessage("hi")}}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatalf("Generate() unexpected error: %v", err)
	}
	if gotCfg != nil {
		t.Errorf("cfg = %v, want nil pointer for a request without config", gotCfg)
	}
	if gotReqConfig != nil {
		t.Errorf("req.Config = %#v, want nil for a request without config", gotReqConfig)
	}

	// A populated map still resolves and normalizes to the pointer type.
	req = &ModelRequest{
		Messages: []*Message{NewUserTextMessage("hi")},
		Config:   map[string]any{"temperature": 0.5},
	}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatalf("Generate() unexpected error: %v", err)
	}
	if gotCfg == nil || gotCfg.Temperature != 0.5 {
		t.Errorf("cfg = %+v, want Temperature 0.5", gotCfg)
	}
	if _, ok := gotReqConfig.(*testTypedConfig); !ok {
		t.Errorf("req.Config = %#v, want normalized to *testTypedConfig", gotReqConfig)
	}
}

// TestCallerRequestNotMutated pins that config normalization happens on the
// action's shallow copy of the request: the caller's struct keeps its raw
// config, so one request can be fanned out to actions with different Config
// types.
func TestCallerRequestNotMutated(t *testing.T) {
	r := registry.New()
	m := NewModelAction("test/no-mutation", nil,
		func(ctx context.Context, req *ModelRequest, cfg testTypedConfig, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{Message: NewModelTextMessage("ok"), Request: req}, nil
		})
	m.Register(r)

	raw := map[string]any{"temperature": 0.5}
	req := &ModelRequest{Messages: []*Message{NewUserTextMessage("hi")}, Config: raw}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatalf("Generate() unexpected error: %v", err)
	}
	if got, ok := req.Config.(map[string]any); !ok || got["temperature"] != 0.5 {
		t.Errorf("req.Config = %#v, want the caller's raw map left untouched", req.Config)
	}

	ret := NewRetrieverAction("test/no-mutation-retriever", nil,
		func(ctx context.Context, req *RetrieverRequest, cfg testTypedConfig) (*RetrieverResponse, error) {
			return &RetrieverResponse{}, nil
		})
	ret.Register(r)

	rreq := &RetrieverRequest{Query: DocumentFromText("q", nil), Options: raw}
	if _, err := ret.Retrieve(context.Background(), rreq); err != nil {
		t.Fatalf("Retrieve() unexpected error: %v", err)
	}
	if got, ok := rreq.Options.(map[string]any); !ok || got["temperature"] != 0.5 {
		t.Errorf("req.Options = %#v, want the caller's raw map left untouched", rreq.Options)
	}
}

func TestModelConfigSchemaInference(t *testing.T) {
	newFn := func() ModelActionFunc[testTypedConfig] {
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
		m := NewModelAction("test/inferred-schema", nil, newFn())
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
		m := NewModelAction("test/override-schema", &ModelOptions{ConfigSchema: override}, newFn())
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
		m := NewModelAction("test/any-schema", nil, func(ctx context.Context, req *ModelRequest, cfg any, cb ModelStreamCallback) (*ModelResponse, error) {
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
	e := NewEmbedderAction("test/typed-config-embedder", nil, func(ctx context.Context, req *EmbedRequest, cfg testTypedConfig) (*EmbedResponse, error) {
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
	e := NewEvaluatorAction("test/typed-config-evaluator", nil, func(ctx context.Context, req *EvaluatorCallbackRequest, cfg testTypedConfig) (*EvaluatorCallbackResponse, error) {
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
	be := NewBatchEvaluatorAction("test/typed-config-batch-evaluator", nil, func(ctx context.Context, req *EvaluatorRequest, cfg testTypedConfig) (*EvaluatorResponse, error) {
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

func TestRetrieverTypedConfig(t *testing.T) {
	r := registry.New()

	var got testTypedConfig
	var gotReqOptions any
	ret := NewRetrieverAction("test/typed-config-retriever", nil, func(ctx context.Context, req *RetrieverRequest, cfg testTypedConfig) (*RetrieverResponse, error) {
		got = cfg
		gotReqOptions = req.Options
		return &RetrieverResponse{}, nil
	})
	ret.Register(r)

	req := &RetrieverRequest{Query: DocumentFromText("q", nil), Options: map[string]any{"maxTokens": 7}}
	if _, err := ret.Retrieve(context.Background(), req); err != nil {
		t.Fatalf("Retrieve() unexpected error: %v", err)
	}
	if got.MaxTokens != 7 {
		t.Errorf("config = %+v, want MaxTokens 7", got)
	}
	if gotReqOptions != any(got) {
		t.Errorf("req.Options = %#v, want normalized to %#v", gotReqOptions, got)
	}

	req = &RetrieverRequest{Query: DocumentFromText("q", nil), Options: otherProviderConfig{Temperature: 0.1}}
	if _, err := ret.Retrieve(context.Background(), req); err == nil || !strings.Contains(err.Error(), "invalid config type") {
		t.Fatalf("Retrieve() error = %v, want invalid config type error", err)
	}
}

func TestRetrieverConfigSchemaInference(t *testing.T) {
	ret := NewRetrieverAction("test/inferred-retriever", nil, func(ctx context.Context, req *RetrieverRequest, cfg testTypedConfig) (*RetrieverResponse, error) {
		return &RetrieverResponse{}, nil
	})

	md, ok := ret.Desc().Metadata["retriever"].(map[string]any)
	if !ok {
		t.Fatalf("retriever metadata missing: %v", ret.Desc().Metadata)
	}
	schema, ok := md["customOptions"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions is not an inferred schema: %v", md["customOptions"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok || props["temperature"] == nil || props["maxTokens"] == nil {
		t.Errorf("inferred config schema = %v, want temperature and maxTokens", schema)
	}
}

func TestRetrieverActionMetadata(t *testing.T) {
	ret := NewRetrieverAction("test/retriever-metadata",
		&RetrieverOptions{
			Metadata: map[string]any{
				"custom":    "value",
				"retriever": "caller values must not clobber the reserved keys",
			},
		},
		func(ctx context.Context, req *RetrieverRequest, cfg testTypedConfig) (*RetrieverResponse, error) {
			return &RetrieverResponse{}, nil
		})

	metadata := ret.Desc().Metadata
	if got := metadata["custom"]; got != "value" {
		t.Errorf(`Metadata["custom"] = %v, want "value"`, got)
	}
	if got := metadata["type"]; got != api.ActionTypeRetriever {
		t.Errorf(`Metadata["type"] = %v, want %v`, got, api.ActionTypeRetriever)
	}
	if _, ok := metadata["retriever"].(map[string]any); !ok {
		t.Errorf(`Metadata["retriever"] = %v, want the built retriever info map`, metadata["retriever"])
	}
}
