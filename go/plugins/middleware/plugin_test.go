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

package middleware

import (
	"context"
	"reflect"
	"slices"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/schematest"
)

// The Dev UI and cross-runtime callers dispatch middleware by name with a
// JSON config, which every plugin middleware serves from one registered
// prototype. Each dispatch must build its own config off that prototype:
// a field one call sets must not stay set for calls after it, and the
// registered prototype itself must stay untouched.
func TestJSONDispatchDoesNotLeakConfigBetweenCalls(t *testing.T) {
	r := newTestRegistry(t)
	calls := 0
	m := defineModel(t, r, "test/alwaysfail", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		calls++
		return nil, core.NewError(core.UNAVAILABLE, "always failing")
	})
	prototype := Retry{}
	registerTestMiddleware(r, "retry", prototype)

	dispatch := func(config map[string]any) {
		t.Helper()
		calls = 0
		_, err := ai.GenerateWithRequest(ctx, r, &ai.GenerateActionOptions{
			Model:    m.Name(),
			Messages: []*ai.Message{ai.NewUserTextMessage("hello")},
			Use:      []*ai.MiddlewareRef{{Name: provider + "/retry", Config: config}},
		}, nil, nil)
		if err == nil {
			t.Fatal("expected the always-failing model to return an error")
		}
	}

	dispatch(map[string]any{"maxRetries": 5, "noJitter": true})
	if calls != 6 { // 1 initial + 5 retries
		t.Fatalf("got %d model calls, want 6", calls)
	}

	dispatch(map[string]any{})
	if calls != 4 { // 1 initial + the default 3 retries
		t.Errorf("got %d model calls, want 4: maxRetries leaked from the previous dispatch", calls)
	}

	if !reflect.DeepEqual(prototype, Retry{}) {
		t.Errorf("prototype mutated by dispatch: %+v", prototype)
	}
}

// TestConfigSchemasDocumented asserts that every field the Dev UI renders for
// a middleware config carries a description. Schema inference does not read Go
// doc comments -- only a `jsonschema_description:"..."` struct tag reaches the
// schema -- so a field documented in Go alone shows up as a bare input box.
//
// Walking the descriptors rather than a fixed list of types means a new
// middleware, or a new field on an existing one, cannot ship undocumented.
func TestConfigSchemasDocumented(t *testing.T) {
	descs, err := (&Middleware{}).Middlewares(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(descs) == 0 {
		t.Fatal("plugin returned no middleware descriptors")
	}
	for _, d := range descs {
		if d.Description == "" {
			t.Errorf("middleware %q has no description", d.Name)
		}
		schematest.AssertDescribed(t, d.Name, d.ConfigSchema)
	}
}

// TestDescriptionsUseTheDedicatedTag guards the whole package against the
// silent truncation described on [schematest.AssertNoInlineDescriptions]. It
// covers the tool input schemas too, which no middleware descriptor exposes.
func TestDescriptionsUseTheDedicatedTag(t *testing.T) {
	schematest.AssertNoInlineDescriptions(t, ".")
}

// TestStatusEnumsAreCanonical asserts that every "statuses" field offers the
// full canonical status set, so the Dev UI renders a picker rather than a
// free-text box and a name added to [status.Names] cannot be left out.
//
// The enum lives on the two struct tags rather than on a JSONSchema method on
// [status.Name] itself, because that type is also the status field of a
// serialized error. Action output is validated against its inferred schema, so
// an enum there would reject an error that crossed the wire from another
// runtime carrying a status this build does not know. Middleware config is
// never validated -- the descriptor schema only drives the UI -- so the enum
// is safe here.
func TestStatusEnumsAreCanonical(t *testing.T) {
	descs, err := (&Middleware{}).Middlewares(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := status.Names()
	found := 0
	for _, d := range descs {
		props, _ := d.ConfigSchema["properties"].(map[string]any)
		field, ok := props["statuses"].(map[string]any)
		if !ok {
			continue
		}
		found++
		items, _ := field["items"].(map[string]any)
		raw, _ := items["enum"].([]any)
		got := make([]status.Name, 0, len(raw))
		for _, v := range raw {
			name, _ := v.(string)
			got = append(got, status.Name(name))
		}
		if !slices.Equal(slices.Sorted(slices.Values(got)), slices.Sorted(slices.Values(want))) {
			t.Errorf("%s.statuses enum = %v, want the canonical set %v", d.Name, got, want)
		}
	}
	if found == 0 {
		t.Error("no middleware exposes a statuses field; drop this test or restore the enum tag")
	}
}
