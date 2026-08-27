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
	"testing"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
)

func TestLookupAgent(t *testing.T) {
	t.Run("resolves and runs a defined agent", func(t *testing.T) {
		g := genkit.Init(context.Background(), genkit.WithExperimental())
		DefineCustomAgent(g, "echo",
			func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
				if err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
					sess.AddMessages(ai.NewModelTextMessage("echo: " + input.Message.Text()))
					return nil, nil
				}); err != nil {
					return nil, err
				}
				return sess.Result(), nil
			})

		h := LookupAgent(g, "echo")
		if got := h.Name(); got != "echo" {
			t.Errorf("Name() = %q, want %q", got, "echo")
		}
		out, err := h.Run(context.Background(), &aix.AgentInput{Message: ai.NewUserTextMessage("hi")})
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if got, want := out.Message.Text(), "echo: hi"; got != want {
			t.Errorf("Message.Text() = %q, want %q", got, want)
		}
	})

	t.Run("unknown agent is nil", func(t *testing.T) {
		g := genkit.Init(context.Background(), genkit.WithExperimental())
		if h := LookupAgent(g, "ghost"); h != nil {
			t.Fatalf("LookupAgent(unregistered) = %+v, want nil", h)
		}
	})

	t.Run("does not require the experimental gate", func(t *testing.T) {
		// LookupAgent only reads the registry, so unlike the constructors it
		// must not panic without genkit.WithExperimental; with no agents
		// registered it simply reports nil.
		g := genkit.Init(context.Background())
		if h := LookupAgent(g, "anything"); h != nil {
			t.Fatalf("LookupAgent(unregistered) = %+v, want nil", h)
		}
	})
}
