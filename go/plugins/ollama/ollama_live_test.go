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

package ollama_test

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	ollamaPlugin "github.com/firebase/genkit/go/plugins/ollama"
)

var (
	serverAddress    = flag.String("server-address", "http://localhost:11434", "Ollama server address")
	modelName        = flag.String("model-name", "tinyllama", "model name")
	dynamicModelName = flag.String("dynamic-model-name", "moondream", "model name for dynamic discovery test (must not be in hardcoded lists)")
	liveTimeout      = flag.Duration("live-timeout", 2*time.Minute, "timeout for live Ollama requests")
	testLive         = flag.Bool("test-live", false, "run live tests")
)

type liveShowResponse struct {
	Capabilities *[]string `json:"capabilities"`
}

func getLiveModelCapabilities(t *testing.T, ctx context.Context, modelName string) ([]string, bool) {
	t.Helper()

	body, err := json.Marshal(map[string]string{"model": modelName})
	if err != nil {
		t.Fatalf("failed to encode /api/show request: %v", err)
	}
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(*serverAddress, "/")+"/api/show",
		bytes.NewReader(body),
	)
	if err != nil {
		t.Fatalf("failed to create /api/show request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("failed to call /api/show for model %q: %v", modelName, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Logf("/api/show returned status %d for model %q; expecting fallback capabilities", resp.StatusCode, modelName)
		return nil, false
	}

	var showResp liveShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&showResp); err != nil {
		t.Fatalf("failed to decode /api/show response for model %q: %v", modelName, err)
	}
	if showResp.Capabilities == nil {
		return nil, false
	}
	return *showResp.Capabilities, true
}

func assertLiveCapabilities(t *testing.T, desc api.ActionDesc, capabilities []string, detected bool) {
	t.Helper()

	modelMetadata, ok := desc.Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata for %q has type %T, want map[string]any", desc.Name, desc.Metadata["model"])
	}
	supports, ok := modelMetadata["supports"].(map[string]any)
	if !ok {
		t.Fatalf("supports metadata for %q has type %T, want map[string]any", desc.Name, modelMetadata["supports"])
	}

	// ListActions and ResolveAction historically enabled tools and media when
	// capability detection was unavailable.
	wantTools := true
	wantMedia := true
	if detected {
		wantTools = slices.Contains(capabilities, "tools")
		wantMedia = slices.Contains(capabilities, "vision")
	}
	if got, ok := supports["tools"].(bool); !ok || got != wantTools {
		t.Errorf("%q tools support = %v, want %v from /api/show capabilities %v", desc.Name, supports["tools"], wantTools, capabilities)
	}
	if got, ok := supports["media"].(bool); !ok || got != wantMedia {
		t.Errorf("%q media support = %v, want %v from /api/show capabilities %v", desc.Name, supports["media"], wantMedia, capabilities)
	}
}

func sameLiveModelName(got, want string) bool {
	got = strings.TrimPrefix(got, "ollama/")
	return got == want || got == want+":latest" || got+":latest" == want
}

// Live tests require a running Ollama server. Use -server-address to override
// the default http://localhost:11434 endpoint.
func TestLive(t *testing.T) {
	if !*testLive {
		t.Skip("skipping go/plugins/ollama live test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), *liveTimeout)
	defer cancel()

	o := &ollamaPlugin.Ollama{
		ServerAddress: *serverAddress,
		Timeout:       int(liveTimeout.Seconds()),
	}
	g := genkit.Init(ctx, genkit.WithPlugins(o))

	// Define the model
	o.DefineModel(g, ollamaPlugin.ModelDefinition{Name: *modelName, Type: "chat"}, nil)

	// Use the Ollama model
	m := ollamaPlugin.Model(g, *modelName)
	if m == nil {
		t.Fatalf(`failed to find model: %s`, *modelName)
	}

	// Generate a response from the model
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithConfig(&ollamaPlugin.GenerateContentConfig{Temperature: ollamaPlugin.Ptr(1.0), Think: ollamaPlugin.ThinkEnabled(true)}),
		ai.WithPrompt("I'm hungry what should I eat?"),
	)
	if err != nil {
		t.Fatalf("failed to generate response: %s", err)
	}

	if resp == nil {
		t.Fatalf("response is nil")
	}

	// Get the text from the response
	text := resp.Text()
	t.Logf("Full response: %s", text)

	// Assert that the response text is as expected
	if text == "" {
		t.Fatalf("expected non-empty response, got: %s", text)
	}
}

// TestLiveDynamicDiscovery verifies that a model NOT registered via DefineModel
// can be discovered and used through the DynamicPlugin interface (ListActions + ResolveAction).
func TestLiveDynamicDiscovery(t *testing.T) {
	if !*testLive {
		t.Skip("skipping go/plugins/ollama live dynamic discovery test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), *liveTimeout)
	defer cancel()
	o := &ollamaPlugin.Ollama{
		ServerAddress: *serverAddress,
		Timeout:       int(liveTimeout.Seconds()),
	}
	g := genkit.Init(ctx, genkit.WithPlugins(o))
	capabilities, detected := getLiveModelCapabilities(t, ctx, *dynamicModelName)
	t.Logf("/api/show capabilities for %q: %v (detected: %v)", *dynamicModelName, capabilities, detected)

	// Verify ListActions discovers local models
	actions := o.ListActions(ctx)
	if len(actions) == 0 {
		t.Fatal("ListActions() returned no actions, ensure Ollama has local models")
	}
	t.Logf("ListActions() discovered %d models:", len(actions))
	for _, a := range actions {
		t.Logf("  - %s", a.Name)
	}
	var discovered *api.ActionDesc
	for i := range actions {
		if sameLiveModelName(actions[i].Name, *dynamicModelName) {
			discovered = &actions[i]
			break
		}
	}
	if discovered == nil {
		t.Fatalf("ListActions() did not include dynamic model %q", *dynamicModelName)
	}
	assertLiveCapabilities(t, *discovered, capabilities, detected)

	// Use a model that is NOT in the hardcoded lists via LookupModel,
	// which triggers ResolveAction under the hood.
	m := ollamaPlugin.Model(g, *dynamicModelName)
	if m == nil {
		t.Fatalf("Model(%q) returned nil — ResolveAction did not work", *dynamicModelName)
	}
	resolvedAction, ok := m.(api.Action)
	if !ok {
		t.Fatalf("resolved model %q does not implement api.Action", *dynamicModelName)
	}
	assertLiveCapabilities(t, resolvedAction.Desc(), capabilities, detected)

	// Generate a response from the dynamically resolved model
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithConfig(&ai.GenerationCommonConfig{Temperature: 1}),
		ai.WithPrompt("Say hello in one sentence."),
	)
	if err != nil {
		t.Fatalf("failed to generate with dynamic model %q: %s", *dynamicModelName, err)
	}

	text := resp.Text()
	t.Logf("Dynamic model %q response: %s", *dynamicModelName, text)
	if text == "" {
		t.Fatalf("expected non-empty response from dynamic model %q", *dynamicModelName)
	}
}
