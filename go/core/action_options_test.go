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

package core

import (
	"context"
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

func TestNewActionWithOptions(t *testing.T) {
	t.Run("nil options infers schemas", func(t *testing.T) {
		a := NewActionWithOptions(api.ActionTypeCustom, "double", nil,
			func(ctx context.Context, n int) (int, error) { return n * 2, nil })

		got, err := a.Run(context.Background(), 5, nil)
		if err != nil {
			t.Fatalf("Run error: %v", err)
		}
		if got != 10 {
			t.Errorf("got %d, want 10", got)
		}

		desc := a.Desc()
		if desc.InputSchema == nil || desc.OutputSchema == nil {
			t.Errorf("schemas not inferred: input=%v output=%v", desc.InputSchema, desc.OutputSchema)
		}
		if desc.StreamSchema != nil {
			t.Errorf("non-streaming action advertises StreamSchema %v", desc.StreamSchema)
		}
	})

	t.Run("description field wins over metadata", func(t *testing.T) {
		a := NewActionWithOptions(api.ActionTypeCustom, "desc", &ActionOptions{
			Description: "explicit",
			Metadata:    map[string]any{"description": "from metadata"},
		}, func(ctx context.Context, in struct{}) (bool, error) { return true, nil })

		if got := a.Desc().Description; got != "explicit" {
			t.Errorf("Description = %q, want %q", got, "explicit")
		}
	})

	t.Run("description falls back to metadata", func(t *testing.T) {
		a := NewActionWithOptions(api.ActionTypeCustom, "desc-meta", &ActionOptions{
			Metadata: map[string]any{"description": "from metadata"},
		}, func(ctx context.Context, in struct{}) (bool, error) { return true, nil })

		if got := a.Desc().Description; got != "from metadata" {
			t.Errorf("Description = %q, want %q", got, "from metadata")
		}
	})

	t.Run("explicit schemas override inference", func(t *testing.T) {
		in := map[string]any{"type": "string", "title": "in"}
		out := map[string]any{"type": "string", "title": "out"}
		stream := map[string]any{"type": "string", "title": "stream"}
		a := NewStreamingActionWithOptions(api.ActionTypeCustom, "override", &ActionOptions{
			InputSchema:  in,
			OutputSchema: out,
			StreamSchema: stream,
		}, func(ctx context.Context, s string, cb StreamCallback[string]) (string, error) {
			return s, nil
		})

		desc := a.Desc()
		if !reflect.DeepEqual(desc.InputSchema, in) {
			t.Errorf("InputSchema = %v, want %v", desc.InputSchema, in)
		}
		if !reflect.DeepEqual(desc.OutputSchema, out) {
			t.Errorf("OutputSchema = %v, want %v", desc.OutputSchema, out)
		}
		if !reflect.DeepEqual(desc.StreamSchema, stream) {
			t.Errorf("StreamSchema = %v, want %v", desc.StreamSchema, stream)
		}
	})

	t.Run("streaming action infers stream schema", func(t *testing.T) {
		a := NewStreamingActionWithOptions(api.ActionTypeCustom, "streamer", nil,
			func(ctx context.Context, n int, cb StreamCallback[string]) (int, error) {
				return n, nil
			})

		if a.Desc().StreamSchema == nil {
			t.Error("streaming action did not infer StreamSchema")
		}
	})

	t.Run("register makes the action resolvable", func(t *testing.T) {
		r := registry.New()
		NewActionWithOptions(api.ActionTypeCustom, "registered", nil,
			func(ctx context.Context, s string) (string, error) { return s, nil }).Register(r)

		if a := ResolveActionFor[string, string, struct{}](r, api.ActionTypeCustom, "registered"); a == nil {
			t.Error("action not resolvable after Register")
		}
	})
}

// TestDeprecatedConstructorsDelegate pins that the deprecated flat-argument
// constructors produce the same descriptor as the options-struct equivalents
// they now wrap.
func TestDeprecatedConstructorsDelegate(t *testing.T) {
	metadata := map[string]any{"description": "legacy"}
	inputSchema := map[string]any{"type": "string"}
	fn := func(ctx context.Context, s string) (string, error) { return s, nil }

	oldA := NewAction("legacy", api.ActionTypeCustom, metadata, inputSchema, fn)
	newA := NewActionWithOptions(api.ActionTypeCustom, "legacy", &ActionOptions{
		Metadata:    metadata,
		InputSchema: inputSchema,
	}, fn)

	if !reflect.DeepEqual(oldA.Desc(), newA.Desc()) {
		t.Errorf("descriptors diverge:\nold: %+v\nnew: %+v", oldA.Desc(), newA.Desc())
	}
}
