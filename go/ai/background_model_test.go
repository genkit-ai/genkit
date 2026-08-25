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
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

func TestNewBackgroundModelInputSchema(t *testing.T) {
	startFn := func(ctx context.Context, req *ModelRequest) (*ModelOperation, error) {
		return &ModelOperation{ID: "op-1"}, nil
	}
	checkFn := func(ctx context.Context, op *ModelOperation) (*ModelOperation, error) {
		return op, nil
	}

	configSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"temperature": map[string]any{"type": "number"},
		},
	}

	t.Run("config schema is advertised on the start action", func(t *testing.T) {
		m := NewBackgroundModel("test/bgmodel", &BackgroundModelOptions{
			ModelOptions: ModelOptions{ConfigSchema: configSchema},
		}, startFn, checkFn)

		action, ok := m.(api.Action)
		if !ok {
			t.Fatal("background model does not implement api.Action")
		}

		props, ok := action.Desc().InputSchema["properties"].(map[string]any)
		if !ok {
			t.Fatalf("input schema has no properties: %v", action.Desc().InputSchema)
		}
		config, ok := props["config"].(map[string]any)
		if !ok {
			t.Fatalf("input schema has no config property: %v", props)
		}
		// The config slot is advertised as the schema or an explicit null so
		// that a typed-nil config passes input validation.
		anyOf, ok := config["anyOf"].([]any)
		if !ok || len(anyOf) != 2 {
			t.Fatalf("config property is not null-tolerant: %v", config)
		}
		schema, ok := anyOf[0].(map[string]any)
		if !ok {
			t.Fatalf("config property is not the supplied schema: %v", anyOf[0])
		}
		configProps, ok := schema["properties"].(map[string]any)
		if !ok {
			t.Fatalf("config property is not the supplied schema: %v", schema)
		}
		if configProps["temperature"] == nil {
			t.Errorf("config schema was not carried through, got %v", configProps)
		}
	})

	t.Run("caller metadata is merged and built-ins win", func(t *testing.T) {
		m := NewBackgroundModel("test/bgmodel", &BackgroundModelOptions{
			Metadata: map[string]any{"team": "media", "type": "bogus"},
		}, startFn, checkFn)

		md := m.(api.Action).Desc().Metadata
		if md["team"] != "media" {
			t.Errorf("caller metadata dropped: %v", md)
		}
		if md["type"] != api.ActionTypeBackgroundModel {
			t.Errorf("built-in type key must win over caller metadata, got %v", md["type"])
		}
	})

	t.Run("label doubles as the description", func(t *testing.T) {
		m := NewBackgroundModel("test/bgmodel", &BackgroundModelOptions{
			ModelOptions: ModelOptions{Label: "Test Background Model"},
		}, startFn, checkFn)

		r := registry.New()
		m.Register(r)
		for _, key := range []string{
			"/background-model/test/bgmodel",
			"/check-operation/test/bgmodel/check",
		} {
			action := r.LookupAction(key)
			if action == nil {
				t.Fatalf("action %q not registered", key)
			}
			if got := action.Desc().Description; got != "Test Background Model" {
				t.Errorf("%s description = %q, want the label", key, got)
			}
		}
	})

	t.Run("metadata description outranks a defaulted label", func(t *testing.T) {
		m := NewBackgroundModel("test/bgmodel", &BackgroundModelOptions{
			Metadata: map[string]any{"description": "generates videos"},
		}, startFn, checkFn)

		if got := m.(api.Action).Desc().Description; got != "generates videos" {
			t.Errorf("Description = %q, want the caller's metadata description", got)
		}
	})

	t.Run("explicit label outranks the metadata description", func(t *testing.T) {
		m := NewBackgroundModel("test/bgmodel", &BackgroundModelOptions{
			ModelOptions: ModelOptions{Label: "Test Background Model"},
			Metadata:     map[string]any{"description": "generates videos"},
		}, startFn, checkFn)

		if got := m.(api.Action).Desc().Description; got != "Test Background Model" {
			t.Errorf("Description = %q, want the explicit label", got)
		}
	})

	t.Run("input schema is inferred without a config schema", func(t *testing.T) {
		m := NewBackgroundModel("test/bgmodel", nil, startFn, checkFn)

		action, ok := m.(api.Action)
		if !ok {
			t.Fatal("background model does not implement api.Action")
		}

		props, ok := action.Desc().InputSchema["properties"].(map[string]any)
		if !ok {
			t.Fatalf("input schema has no properties: %v", action.Desc().InputSchema)
		}
		if props["messages"] == nil {
			t.Errorf("input schema is not a ModelRequest schema, got %v", props)
		}
	})
}
