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

func TestNewActionOf(t *testing.T) {
	t.Run("nil options infers schemas", func(t *testing.T) {
		a := NewActionOf(api.ActionTypeCustom, "double", nil,
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
		a := NewActionOf(api.ActionTypeCustom, "desc", &ActionOptions{
			Description: "explicit",
			Metadata:    map[string]any{"description": "from metadata"},
		}, func(ctx context.Context, in struct{}) (bool, error) { return true, nil })

		if got := a.Desc().Description; got != "explicit" {
			t.Errorf("Description = %q, want %q", got, "explicit")
		}
		// The fallback is one-way: metadata reaches the descriptor as given,
		// so the caller's key is never rewritten to match Description.
		if got := a.Desc().Metadata["description"]; got != "from metadata" {
			t.Errorf(`Metadata["description"] = %v, want %q`, got, "from metadata")
		}
	})

	t.Run("description falls back to metadata", func(t *testing.T) {
		a := NewActionOf(api.ActionTypeCustom, "desc-meta", &ActionOptions{
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
		a := NewStreamingActionOf(api.ActionTypeCustom, "override", &ActionOptions{
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
		a := NewStreamingActionOf(api.ActionTypeCustom, "streamer", nil,
			func(ctx context.Context, n int, cb StreamCallback[string]) (int, error) {
				return n, nil
			})

		if a.Desc().StreamSchema == nil {
			t.Error("streaming action did not infer StreamSchema")
		}
	})

	t.Run("register makes the action resolvable", func(t *testing.T) {
		r := registry.New()
		NewActionOf(api.ActionTypeCustom, "registered", nil,
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
	newA := NewActionOf(api.ActionTypeCustom, "legacy", &ActionOptions{
		Metadata:    metadata,
		InputSchema: inputSchema,
	}, fn)

	if !reflect.DeepEqual(oldA.Desc(), newA.Desc()) {
		t.Errorf("descriptors diverge:\nold: %+v\nnew: %+v", oldA.Desc(), newA.Desc())
	}
}

// TestRegisterClearsDynamicMarker pins that the "dynamic" metadata marker
// ("created at runtime, outside any registry") is registration-derived: it
// survives on detached actions and is removed when the action registers into
// a definition-time registry. Registration into a per-request child registry
// (how Generate handles detached tools) keeps the marker and never writes to
// the metadata map, which may be caller-owned and shared across concurrent
// requests.
func TestRegisterClearsDynamicMarker(t *testing.T) {
	newDynamic := func(name string, metadata map[string]any) *Action[string, string, struct{}] {
		return NewActionOf(api.ActionTypeCustom, name, &ActionOptions{Metadata: metadata},
			func(ctx context.Context, s string) (string, error) { return s, nil })
	}

	t.Run("definition-time registration strips the marker", func(t *testing.T) {
		r := registry.New()
		a := newDynamic("detached", map[string]any{"dynamic": true})

		if v, ok := a.Desc().Metadata["dynamic"]; !ok || v != true {
			t.Fatalf("unregistered action missing dynamic marker: %v", a.Desc().Metadata)
		}
		a.Register(r)
		if _, ok := a.Desc().Metadata["dynamic"]; ok {
			t.Errorf("registered action still carries dynamic marker: %v", a.Desc().Metadata)
		}
	})

	t.Run("child-registry registration keeps the marker", func(t *testing.T) {
		child := registry.New().NewChild()
		a := newDynamic("detached", map[string]any{"dynamic": true})

		a.Register(child)
		if v, ok := a.Desc().Metadata["dynamic"]; !ok || v != true {
			t.Errorf("child registration stripped the dynamic marker: %v", a.Desc().Metadata)
		}
	})

	t.Run("caller's metadata map is not mutated", func(t *testing.T) {
		r := registry.New()
		metadata := map[string]any{"dynamic": true, "keep": "me"}
		a := newDynamic("detached", metadata)

		a.Register(r)
		if v, ok := metadata["dynamic"]; !ok || v != true {
			t.Errorf("Register mutated the caller's metadata map: %v", metadata)
		}
		if _, ok := a.Desc().Metadata["keep"]; !ok {
			t.Errorf("stripped metadata lost unrelated keys: %v", a.Desc().Metadata)
		}
	})

	t.Run("actions sharing a metadata map stay independent", func(t *testing.T) {
		r := registry.New()
		shared := map[string]any{"dynamic": true}
		a := newDynamic("one", shared)
		b := newDynamic("two", shared)

		a.Register(r)
		if _, ok := b.Desc().Metadata["dynamic"]; !ok {
			t.Errorf("registering one action stripped the marker from its sibling: %v", b.Desc().Metadata)
		}
	})
}
