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
		configProps, ok := config["properties"].(map[string]any)
		if !ok {
			t.Fatalf("config property is not the supplied schema: %v", config)
		}
		if configProps["temperature"] == nil {
			t.Errorf("config schema was not carried through, got %v", configProps)
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
