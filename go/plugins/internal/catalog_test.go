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

package internal

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
)

// TestLookupOverride pins that a caller's entry is found under either spelling
// of a model ID, keyed and looked up. Callers write the prefixed form
// everywhere else, and an ignored key is a silent no-op.
func TestLookupOverride(t *testing.T) {
	overrides := map[string]ai.ModelOptions{
		"bare":          {Label: "bare entry"},
		"acme/prefixed": {Label: "prefixed entry"},
	}
	for _, tt := range []struct {
		name, id, want string
	}{
		{"bare key, bare id", "bare", "bare entry"},
		{"bare key, prefixed id", "acme/bare", "bare entry"},
		{"prefixed key, bare id", "prefixed", "prefixed entry"},
		{"prefixed key, prefixed id", "acme/prefixed", "prefixed entry"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := LookupOverride(overrides, "acme", tt.id)
			if !ok {
				t.Fatalf("LookupOverride(%q) found nothing", tt.id)
			}
			if got.Label != tt.want {
				t.Errorf("LookupOverride(%q).Label = %q, want %q", tt.id, got.Label, tt.want)
			}
		})
	}
	if _, ok := LookupOverride(overrides, "acme", "absent"); ok {
		t.Error("LookupOverride found an entry for an ID with none")
	}
}

// TestTrimProvider pins that only the prefix is removed, so an ID carrying
// slashes of its own (a tuned Vertex endpoint, say) survives intact.
func TestTrimProvider(t *testing.T) {
	for _, tt := range []struct{ provider, id, want string }{
		{"acme", "acme/model", "model"},
		{"acme", "model", "model"},
		{"acme", "acme/projects/p/endpoints/123", "projects/p/endpoints/123"},
		{"acme", "other/model", "other/model"},
	} {
		if got := TrimProvider(tt.provider, tt.id); got != tt.want {
			t.Errorf("TrimProvider(%q, %q) = %q, want %q", tt.provider, tt.id, got, tt.want)
		}
	}
}
