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
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
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
