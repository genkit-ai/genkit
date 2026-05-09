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
	"context"
	"flag"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	ollamaPlugin "github.com/firebase/genkit/go/plugins/ollama"
)

var (
	serverAddress    = flag.String("server-address", "http://localhost:11434", "Ollama server address")
	modelName        = flag.String("model-name", "tinyllama", "model name")
	dynamicModelName = flag.String("dynamic-model-name", "moondream", "model name for dynamic discovery test (must not be in hardcoded lists)")
	testLive         = flag.Bool("test-live", false, "run live tests")
)

/*
To run this test, you need to have the Ollama server running. You can set the server address using the OLLAMA_SERVER_ADDRESS environment variable.
If the environment variable is not set, the test will default to http://localhost:11434 (the default address for the Ollama server).
*/
func TestLive(t *testing.T) {
	if !*testLive {
		t.Skip("skipping go/plugins/ollama live test")
	}

	ctx := context.Background()

	o := &ollamaPlugin.Ollama{ServerAddress: *serverAddress, Timeout: 60}
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

	ctx := context.Background()
	o := &ollamaPlugin.Ollama{ServerAddress: *serverAddress}
	g := genkit.Init(ctx, genkit.WithPlugins(o))

	// Verify ListActions discovers local models
	actions := o.ListActions(ctx)
	if len(actions) == 0 {
		t.Fatal("ListActions() returned no actions, ensure Ollama has local models")
	}
	t.Logf("ListActions() discovered %d models:", len(actions))
	for _, a := range actions {
		t.Logf("  - %s", a.Name)
	}

	// Use a model that is NOT in the hardcoded lists via LookupModel,
	// which triggers ResolveAction under the hood.
	m := ollamaPlugin.Model(g, *dynamicModelName)
	if m == nil {
		t.Fatalf("Model(%q) returned nil — ResolveAction did not work", *dynamicModelName)
	}

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
