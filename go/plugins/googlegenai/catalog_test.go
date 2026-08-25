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
	"os"
	"path/filepath"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

// useFakeADC points Application Default Credentials at a throwaway file.
//
// VertexAI.Init detects ADC unconditionally and panics when it finds none, so
// a test that initializes that backend otherwise passes on a machine holding
// gcloud credentials and fails everywhere else, CI included. Detection stops at
// the first source it finds and GOOGLE_APPLICATION_CREDENTIALS is the first one
// consulted, so this decides the answer either way rather than only filling a
// gap.
//
// The file never authenticates anything: detection parses it, and none of the
// construction that follows issues a request. A test that reaches the API is a
// live test and takes real credentials.
func useFakeADC(t *testing.T) {
	t.Helper()
	const creds = `{
  "type": "authorized_user",
  "client_id": "fake-client-id.apps.googleusercontent.com",
  "client_secret": "fake-client-secret",
  "refresh_token": "fake-refresh-token",
  "quota_project_id": "test-project"
}`
	path := filepath.Join(t.TempDir(), "application_default_credentials.json")
	if err := os.WriteFile(path, []byte(creds), 0o600); err != nil {
		t.Fatalf("writing fake ADC: %v", err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", path)
}

// modelLabel reads the label an action advertises, which is where a caller's
// entry shows up once the plugin has described the model.
func modelLabel(t *testing.T, a api.Action) string {
	t.Helper()
	model, ok := a.Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing: %v", a.Desc().Metadata)
	}
	label, _ := model["label"].(string)
	return label
}

// TestModelsOverrideReachesResolution is the reason capabilities live in plugin
// config. Nothing registers the model up front: the first lookup drives the
// plugin's ResolveAction, and the caller's entry is what describes what comes
// back. No ordering makes this miss, which is what a registration call could
// not promise, since resolving a name registers it and a later registration of
// the same name would panic.
func TestModelsOverrideReachesResolution(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	const id = "gemini-not-yet-released"
	ga := &GoogleAI{Models: map[string]ai.ModelOptions{
		id: {Label: "Custom Gemini", Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
	}}
	g := genkit.Init(context.Background(), genkit.WithPlugins(ga))

	m := genkit.LookupModel(g, "googleai/"+id)
	if m == nil {
		t.Fatal("LookupModel() = nil, want the plugin to resolve the model")
	}
	if got := modelLabel(t, m.(api.Action)); got != "Custom Gemini" {
		t.Errorf("label = %q, want the entry's capabilities to describe the resolved model", got)
	}
}

// TestEmbeddersOverrideReachesResolution pins the same for embedders.
func TestEmbeddersOverrideReachesResolution(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	const id = "text-embedding-not-yet-released"
	ga := &GoogleAI{Embedders: map[string]ai.EmbedderOptions{
		id: {Label: "Custom Embedder", Dimensions: 256},
	}}
	g := genkit.Init(context.Background(), genkit.WithPlugins(ga))

	e := genkit.LookupEmbedder(g, "googleai/"+id)
	if e == nil {
		t.Fatal("LookupEmbedder() = nil, want the plugin to resolve the embedder")
	}
	// An embedder's label rides under "info", which is the shape JS settled on.
	info, ok := e.(api.Action).Desc().Metadata["info"].(map[string]any)
	if !ok {
		t.Fatalf("embedder info metadata missing: %v", e.(api.Action).Desc().Metadata)
	}
	if info["label"] != "Custom Embedder" {
		t.Errorf("label = %v, want the entry's label", info["label"])
	}
	if info["dimensions"] != 256 {
		t.Errorf("dimensions = %v, want the entry's 256", info["dimensions"])
	}
}

// TestVeoOverrideReachesResolution pins that a background model is described
// from the same entry, so Veo is not a hole in the config.
func TestVeoOverrideReachesResolution(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	const id = "veo-not-yet-released"
	ga := &GoogleAI{Models: map[string]ai.ModelOptions{
		id: {Label: "Custom Veo"},
	}}
	g := genkit.Init(context.Background(), genkit.WithPlugins(ga))

	m := genkit.LookupBackgroundModel(g, "googleai/"+id)
	if m == nil {
		t.Fatal("LookupBackgroundModel() = nil, want the plugin to resolve the background model")
	}
	if got := modelLabel(t, m.(api.Action)); got != "Custom Veo" {
		t.Errorf("label = %q, want the entry's label", got)
	}
}

// TestVertexAIOverrideReachesResolution pins that the second backend reads the
// same config, since the two plugins share every resolution path.
func TestVertexAIOverrideReachesResolution(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "test-project")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
	// The Vertex backend authenticates with ADC, not an API key.
	useFakeADC(t)

	const id = "gemini-not-yet-released"
	v := &VertexAI{Models: map[string]ai.ModelOptions{
		id: {Label: "Custom Vertex Gemini"},
	}}
	g := genkit.Init(context.Background(), genkit.WithPlugins(v))

	m := genkit.LookupModel(g, "vertexai/"+id)
	if m == nil {
		t.Fatal("LookupModel() = nil, want the plugin to resolve the model")
	}
	if got := modelLabel(t, m.(api.Action)); got != "Custom Vertex Gemini" {
		t.Errorf("label = %q, want the entry's label", got)
	}
}

// TestCatalogOverlaysRatherThanReplaces pins the merge rule: an entry replaces
// only the fields it sets, so pinning one capability keeps the config schema
// the model needs to accept a request at all.
func TestCatalogOverlaysRatherThanReplaces(t *testing.T) {
	const id = "gemini-flash-latest"
	base := GetModelOptions(id, googleAIProvider)

	c := catalog{provider: googleAIProvider, models: map[string]ai.ModelOptions{
		id: {Label: "Custom Gemini"},
	}}
	got := c.modelOptions(id)

	if got.Label != "Custom Gemini" {
		t.Errorf("Label = %q, want the entry's", got.Label)
	}
	if got.ConfigSchema == nil {
		t.Error("ConfigSchema = nil, want the resolved schema kept by an entry that does not set one")
	}
	if got.Supports != base.Supports {
		t.Error("Supports replaced, want the resolved value kept by an entry that does not set one")
	}
}

// TestCatalogWithoutOverridesMatchesPlugin pins that an empty config changes
// nothing, so the fields are additive for every application that ignores them.
func TestCatalogWithoutOverridesMatchesPlugin(t *testing.T) {
	const id = "gemini-flash-latest"
	c := catalog{provider: googleAIProvider}

	if got, want := c.modelOptions(id), GetModelOptions(id, googleAIProvider); got.Label != want.Label {
		t.Errorf("modelOptions(%q).Label = %q, want %q", id, got.Label, want.Label)
	}
	const embedder = "text-embedding-004"
	if got, want := c.embedderOptions(embedder), GetEmbedderOptions(embedder, googleAIProvider); got.Label != want.Label {
		t.Errorf("embedderOptions(%q).Label = %q, want %q", embedder, got.Label, want.Label)
	}
}

// TestDefineModelDoesNotRegister pins the deprecated builder: it hands back a
// model without touching the registry, which is why capabilities passed to it
// never reach the model that serves a request.
func TestDefineModelDoesNotRegister(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	ga := &GoogleAI{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(ga))

	const id = "gemini-define-test"
	opts := &ai.ModelOptions{Label: "Test model", Supports: &ai.ModelSupports{Multiturn: true}}
	if _, err := ga.DefineModel(g, id, opts); err != nil {
		t.Fatalf("DefineModel() error = %v", err)
	}
	if isDefined(g, api.ActionTypeModel, googleAIProvider, id) {
		t.Errorf("DefineModel(%q) registered the model, want the deprecated builder to leave the registry alone", id)
	}
}

// TestDefineModelAcceptsOverriddenID pins that an entry makes an ID the plugin
// does not ship a known one, so the deprecated builder stops rejecting it.
func TestDefineModelAcceptsOverriddenID(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	const id = "gemini-not-yet-released"
	ga := &GoogleAI{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(ga))
	if _, err := ga.DefineModel(g, id, nil); err == nil {
		t.Errorf("DefineModel(%q, nil) error = nil for an unknown model, want it refused", id)
	}

	withEntry := &GoogleAI{Models: map[string]ai.ModelOptions{id: {Label: "Custom Gemini"}}}
	g2 := genkit.Init(context.Background(), genkit.WithPlugins(withEntry))
	m, err := withEntry.DefineModel(g2, id, nil)
	if err != nil {
		t.Fatalf("DefineModel(%q, nil) error = %v with an entry describing it", id, err)
	}
	if got := modelLabel(t, m.(api.Action)); got != "Custom Gemini" {
		t.Errorf("label = %q, want the entry's label", got)
	}
}

// TestIsDefinedEmbedderDoesNotResolve pins that asking whether an embedder is
// defined must not itself resolve and register one, since these plugins
// resolve on demand and would otherwise answer true for any ID.
func TestIsDefinedEmbedderDoesNotResolve(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	ga := &GoogleAI{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(ga))

	const id = "text-embedding-not-yet-released"
	if ga.IsDefinedEmbedder(g, id) {
		t.Fatalf("IsDefinedEmbedder(%q) = true for a resolvable but unregistered embedder", id)
	}
}

// TestDefineModelTrimsProviderPrefix pins that the ID spelling
// ai.WithModelName takes is accepted. The prefix is applied downstream by
// concatenation, so an untrimmed ID would name "googleai/googleai/x", an
// action no lookup reaches, while isDefined agreed with the broken key.
func TestDefineModelTrimsProviderPrefix(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	ga := &GoogleAI{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(ga))

	opts := &ai.ModelOptions{Label: "Test model", Supports: &ai.ModelSupports{Multiturn: true}}
	m, err := ga.DefineModel(g, "googleai/gemini-prefixed-test", opts)
	if err != nil {
		t.Fatalf("DefineModel() error = %v", err)
	}
	if got := m.Name(); got != "googleai/gemini-prefixed-test" {
		t.Errorf("model name = %q, want the prefix applied once", got)
	}

	e, err := ga.DefineEmbedder(g, "googleai/embedding-prefixed-test", &ai.EmbedderOptions{Label: "Test embedder"})
	if err != nil {
		t.Fatalf("DefineEmbedder() error = %v", err)
	}
	if got := e.Name(); got != "googleai/embedding-prefixed-test" {
		t.Errorf("embedder name = %q, want the prefix applied once", got)
	}
}

// TestDefineModelRejectsBackgroundModel pins that a Veo ID cannot be built as
// a plain model. Its modality speaks a different API method and deserializes a
// different config type, so the result would advertise video fields on an
// action that can only fail at the API.
func TestDefineModelRejectsBackgroundModel(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	ga := &GoogleAI{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(ga))

	if _, err := ga.DefineModel(g, veo31GeneratePreview, nil); err == nil {
		t.Errorf("DefineModel(%q) = nil error, want it refused as a background model", veo31GeneratePreview)
	}
	opts := &ai.ModelOptions{Label: "Test model", Supports: &ai.ModelSupports{Multiturn: true}}
	if _, err := ga.DefineModel(g, veo31GeneratePreview, opts); err == nil {
		t.Errorf("DefineModel(%q, opts) = nil error, want the modality checked before opts", veo31GeneratePreview)
	}
}

// TestOverrideKeysAcceptProviderPrefix pins that both spellings of a model ID
// reach the same entry. Callers write the prefixed form everywhere else
// (ai.WithModelName("googleai/...")), and an ignored key is a silent no-op,
// the worst way for a config map to fail.
func TestOverrideKeysAcceptProviderPrefix(t *testing.T) {
	for _, key := range []string{"gemini-flash-latest", "googleai/gemini-flash-latest"} {
		ga := &GoogleAI{Models: map[string]ai.ModelOptions{key: {Label: "custom"}}}
		for _, id := range []string{"gemini-flash-latest", "googleai/gemini-flash-latest"} {
			if got := ga.catalog().modelOptions(id).Label; got != "custom" {
				t.Errorf("Models[%q] did not apply to %q: label = %q", key, id, got)
			}
			if !ga.catalog().modelOverridden(id) {
				t.Errorf("Models[%q] not reported as overriding %q", key, id)
			}
		}
	}
	v := &VertexAI{Embedders: map[string]ai.EmbedderOptions{"vertexai/text-embedding-005": {Label: "custom"}}}
	if got := v.catalog().embedderOptions("text-embedding-005").Label; got != "custom" {
		t.Errorf("prefixed embedder key did not apply: label = %q", got)
	}
}
