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

package googlegenai

import (
	"context"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

func testModelOptions() *ai.ModelOptions {
	return &ai.ModelOptions{
		Label:    "Test model",
		Supports: &ai.ModelSupports{Multiturn: true, Tools: true},
	}
}

// TestRegisterModel pins what separates registering from building: the model
// RegisterModel returns is the one a lookup by name finds, while the
// deprecated DefineModel leaves the registry alone.
func TestRegisterModel(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	ctx := context.Background()
	ga := &GoogleAI{}
	g := genkit.Init(ctx, genkit.WithPlugins(ga))

	const registered = "gemini-register-test"
	if ga.IsDefinedModel(g, registered) {
		t.Fatalf("IsDefinedModel(%q) = true before registering", registered)
	}
	if _, err := ga.RegisterModel(g, registered, testModelOptions()); err != nil {
		t.Fatalf("RegisterModel() error = %v", err)
	}
	if !ga.IsDefinedModel(g, registered) {
		t.Errorf("IsDefinedModel(%q) = false after registering", registered)
	}
	// The registered model is the one a lookup by name serves, capabilities
	// and all, rather than a default-capability model resolved on demand.
	found := genkit.LookupModel(g, "googleai/"+registered)
	if found == nil {
		t.Fatalf("LookupModel(%q) = nil after registering", registered)
	}
	model, ok := found.(api.Action).Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing: %v", found.(api.Action).Desc().Metadata)
	}
	if got := model["label"]; got != "Test model" {
		t.Errorf("label = %v, want the registered model's %q", got, "Test model")
	}

	const built = "gemini-define-test"
	if _, err := ga.DefineModel(g, built, testModelOptions()); err != nil {
		t.Fatalf("DefineModel() error = %v", err)
	}
	if ga.IsDefinedModel(g, built) {
		t.Errorf("DefineModel(%q) registered the model, want the deprecated builder to leave the registry alone", built)
	}
}

// TestRegisterEmbedder pins the same split for embedders.
func TestRegisterEmbedder(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	ctx := context.Background()
	ga := &GoogleAI{}
	g := genkit.Init(ctx, genkit.WithPlugins(ga))

	const registered = "embedding-register-test"
	if ga.IsDefinedEmbedder(g, registered) {
		t.Fatalf("IsDefinedEmbedder(%q) = true before registering", registered)
	}
	if _, err := ga.RegisterEmbedder(g, registered, &ai.EmbedderOptions{Label: "Test embedder"}); err != nil {
		t.Fatalf("RegisterEmbedder() error = %v", err)
	}
	if !ga.IsDefinedEmbedder(g, registered) {
		t.Errorf("IsDefinedEmbedder(%q) = false after registering", registered)
	}

	const built = "embedding-define-test"
	if _, err := ga.DefineEmbedder(g, built, &ai.EmbedderOptions{Label: "Test embedder"}); err != nil {
		t.Fatalf("DefineEmbedder() error = %v", err)
	}
	if ga.IsDefinedEmbedder(g, built) {
		t.Errorf("DefineEmbedder(%q) registered the embedder, want the deprecated builder to leave the registry alone", built)
	}
}

// TestIsDefinedDoesNotResolve pins the guard's contract on a plugin that
// resolves actions on demand: it reports what is registered now, not what
// could be resolved. A resolving lookup would register the very action the
// caller is checking for, which is what makes the answer true afterwards.
func TestIsDefinedDoesNotResolve(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	ctx := context.Background()
	ga := &GoogleAI{}
	g := genkit.Init(ctx, genkit.WithPlugins(ga))

	const resolvable = "gemini-3-not-yet-released"
	if ga.IsDefinedModel(g, resolvable) {
		t.Fatalf("IsDefinedModel(%q) = true for a resolvable but unregistered model", resolvable)
	}
	if genkit.LookupModel(g, "googleai/"+resolvable) == nil {
		t.Fatalf("LookupModel(%q) = nil, want the plugin to resolve it", resolvable)
	}
	if !ga.IsDefinedModel(g, resolvable) {
		t.Errorf("IsDefinedModel(%q) = false after the resolving lookup registered it", resolvable)
	}

	const embedder = "text-embedding-not-yet-released"
	if ga.IsDefinedEmbedder(g, embedder) {
		t.Fatalf("IsDefinedEmbedder(%q) = true for a resolvable but unregistered embedder", embedder)
	}
}
