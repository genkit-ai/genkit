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

package exp

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/genkit"
)

// constructors holds one closure per exported Define* entry point in this
// package. Each calls its constructor with throwaway arguments: the
// experimental guard runs before anything else, so the nil prompt/func is never
// dereferenced.
var constructors = map[string]func(g *genkit.Genkit){
	"DefineAgent":             func(g *genkit.Genkit) { DefineAgent[any](g, "a", nil) },
	"DefinePromptAgent":       func(g *genkit.Genkit) { DefinePromptAgent[any](g, "a") },
	"DefineCustomAgent":       func(g *genkit.Genkit) { DefineCustomAgent[any](g, "a", nil) },
	"DefineTool":              func(g *genkit.Genkit) { DefineTool[any, any](g, "t", "desc", nil) },
	"DefineInterruptibleTool": func(g *genkit.Genkit) { DefineInterruptibleTool[any, any, any](g, "t", "desc", nil) },
	"DefineStreamingFlow":     func(g *genkit.Genkit) { DefineStreamingFlow[any, any, any](g, "f", nil) },
}

// TestExperimentalGate pins the contract that the experimental surface is
// unreachable unless the Genkit instance opted in via genkit.WithExperimental.
func TestExperimentalGate(t *testing.T) {
	t.Run("panics without WithExperimental", func(t *testing.T) {
		for name, call := range constructors {
			t.Run(name, func(t *testing.T) {
				g := genkit.Init(context.Background())
				defer func() {
					r := recover()
					if r == nil {
						t.Fatalf("%s did not panic without genkit.WithExperimental()", name)
					}
					if msg := fmt.Sprint(r); !strings.Contains(msg, "WithExperimental") {
						t.Errorf("%s panic does not point at the fix: %q", name, msg)
					}
				}()
				call(g)
			})
		}
	})

	t.Run("opens the gate with WithExperimental", func(t *testing.T) {
		// DefineStreamingFlow has no model or prompt dependency, so once the
		// guard passes it constructs cleanly: a good proxy for "the gate opened".
		g := genkit.Init(context.Background(), genkit.WithExperimental())
		flow := DefineStreamingFlow(g, "test/gated", func(ctx context.Context, n int, stream chan<- int) (string, error) {
			return "ok", nil
		})
		if flow == nil {
			t.Fatal("DefineStreamingFlow returned nil with genkit.WithExperimental() set")
		}
	})
}
