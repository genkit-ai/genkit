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

package modelgarden_test

import (
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

// TestDefineModelBeforeInit verifies that calling DefineModel on a
// modelgarden plugin before Init returns an error rather than panicking on a
// nil/uninitialized client.
func TestDefineModelBeforeInit(t *testing.T) {
	opts := &ai.ModelOptions{Label: "test"}

	t.Run("Anthropic", func(t *testing.T) {
		a := &modelgarden.Anthropic{}
		_, err := a.DefineModel("claude-test", opts)
		if err == nil {
			t.Fatal("expected error when DefineModel called before Init, got nil")
		}
		if !strings.Contains(err.Error(), "not initialized") {
			t.Fatalf("expected 'not initialized' error, got: %v", err)
		}
	})

	t.Run("Llama", func(t *testing.T) {
		l := &modelgarden.Llama{}
		_, err := l.DefineModel("llama-test", opts)
		if err == nil {
			t.Fatal("expected error when DefineModel called before Init, got nil")
		}
		if !strings.Contains(err.Error(), "not initialized") {
			t.Fatalf("expected 'not initialized' error, got: %v", err)
		}
	})
}
