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

package anthropic_test

import (
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/plugins/anthropic"
)

func TestCacheControlHelper(t *testing.T) {
	got := anthropic.CacheControl(nil)
	want := map[string]any{
		"cache_control": map[string]any{"type": "ephemeral"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("CacheControl(nil) = %#v, want %#v", got, want)
	}

	got = anthropic.CacheControl(&anthropic.CacheControlOptions{TTL: "1h"})
	want = map[string]any{
		"cache_control": map[string]any{"type": "ephemeral", "ttl": "1h"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("CacheControl(1h) = %#v, want %#v", got, want)
	}
}
