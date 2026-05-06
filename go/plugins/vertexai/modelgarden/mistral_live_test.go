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
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

func TestMistralLive(t *testing.T) {
	if _, ok := requireEnv("GOOGLE_CLOUD_PROJECT"); !ok {
		t.Skip("GOOGLE_CLOUD_PROJECT not found in the environment")
	}
	if _, ok := requireEnv("GOOGLE_CLOUD_LOCATION"); !ok {
		t.Skip("GOOGLE_CLOUD_LOCATION not found in the environment")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&modelgarden.Mistral{}))

	t.Run("invalid model", func(t *testing.T) {
		m := modelgarden.MistralModel(g, "mistral-does-not-exist")
		if m != nil {
			t.Fatalf("model should have been empty, got: %#v", m)
		}
	})

	t.Run("mistral small generation", func(t *testing.T) {
		m := modelgarden.MistralModel(g, "mistral-small-2503")
		if m == nil {
			t.Fatal("mistral-small-2503 model was not registered")
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithSystem("You are a helpful assistant. Reply in one short sentence."),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Say hello."))),
		)
		if err != nil {
			t.Fatal(err)
		}
		if strings.TrimSpace(resp.Text()) == "" {
			t.Fatal("expected a non-empty response")
		}
	})

	t.Run("codestral generation", func(t *testing.T) {
		m := modelgarden.MistralModel(g, "codestral-2")
		if m == nil {
			t.Fatal("codestral-2 model was not registered")
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Write a one-line Python function that returns 42."))),
		)
		if err != nil {
			t.Fatal(err)
		}
		if strings.TrimSpace(resp.Text()) == "" {
			t.Fatal("expected a non-empty response")
		}
	})
}
