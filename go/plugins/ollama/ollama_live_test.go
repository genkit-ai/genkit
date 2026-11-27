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
	"net"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	ollamaPlugin "github.com/firebase/genkit/go/plugins/ollama"
)

// The live test runs against the Ollama server OLLAMA_HOST names, with these
// models pulled. Each can be overridden through the environment variable
// beside it.
const (
	// defaultModel serves tools and thinking.
	defaultModel = "qwen3:4b" // GENKIT_OLLAMA_MODEL
	// defaultVisionModel accepts images.
	defaultVisionModel = "qwen2.5vl:3b" // GENKIT_OLLAMA_VISION_MODEL
	// defaultEmbedder serves 768-dimension embeddings.
	defaultEmbedder = "nomic-embed-text" // GENKIT_OLLAMA_EMBEDDER
	// defaultDynamicModel is in none of the plugin's static lists, so its
	// capabilities can only come from the server.
	defaultDynamicModel = "moondream" // GENKIT_OLLAMA_DYNAMIC_MODEL
)

// envOr returns the value of the environment variable name, or def when it is
// not set.
func envOr(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}

// serverAddress turns host, as OLLAMA_HOST spells it, into a server URL the
// way the Ollama CLI does: a host without a scheme is http and, without a
// port, on the default 11434. A host with a scheme is used as given.
func serverAddress(host string) string {
	if strings.Contains(host, "://") {
		return host
	}
	hostport, path, _ := strings.Cut(host, "/")
	if _, _, err := net.SplitHostPort(hostport); err != nil {
		hostport = net.JoinHostPort(strings.Trim(hostport, "[]"), "11434")
	}
	return "http://" + hostport + strings.TrimSuffix("/"+path, "/")
}

func TestPluginLive(t *testing.T) {
	// OLLAMA_HOST, as the Ollama CLI reads it, says where the server is,
	// and so stands in for the API key other providers gate on.
	server := serverAddress(livetest.Env(t, "OLLAMA_HOST"))
	o := &ollamaPlugin.Ollama{ServerAddress: server, Timeout: 300}
	g := livetest.Init(t, o)

	model := "ollama/" + envOr("GENKIT_OLLAMA_MODEL", defaultModel)
	livetest.Run(t, g, livetest.Suite{
		Model: ai.NewModelRef(model, &ollamaPlugin.GenerateContentConfig{Think: ollamaPlugin.ThinkEnabled(false)}),
		ReasoningModel: ai.NewModelRef(model, &ollamaPlugin.GenerateContentConfig{
			Think: ollamaPlugin.ThinkEnabled(true),
		}),
		ReasoningContent: true,
		VisionModel:      ai.NewModelRef("ollama/"+envOr("GENKIT_OLLAMA_VISION_MODEL", defaultVisionModel), nil),
		LimitConfig: &ollamaPlugin.GenerateContentConfig{
			NumPredict: ollamaPlugin.Ptr(16),
			Think:      ollamaPlugin.ThinkEnabled(false),
		},
	})

	// The plugin's embedders are defined explicitly, never discovered.
	livetest.RunEmbedder(t, g, livetest.EmbedderSuite{
		Embedder:   o.DefineEmbedder(g, envOr("GENKIT_OLLAMA_EMBEDDER", defaultEmbedder), 768, nil),
		Dimensions: 768,
		Normalized: true,
	})

	// A model in none of the static lists is found through ListActions and
	// ResolveAction, with the capabilities the server reports for it.
	t.Run("dynamic discovery", func(t *testing.T) {
		dynamic := envOr("GENKIT_OLLAMA_DYNAMIC_MODEL", defaultDynamicModel)
		capabilities, detected := getLiveModelCapabilities(t, t.Context(), server, dynamic)
		actions := o.ListActions(t.Context())
		i := slices.IndexFunc(actions, func(a api.ActionDesc) bool { return sameLiveModelName(a.Name, dynamic) })
		if i < 0 {
			t.Fatalf("ListActions() did not include %q; pull it first", dynamic)
		}
		assertLiveCapabilities(t, actions[i], capabilities, detected)

		m := ollamaPlugin.Model(g, dynamic)
		if m == nil {
			t.Fatalf("Model(%q) = nil, want the model resolved", dynamic)
		}
		resolved, ok := m.(api.Action)
		if !ok {
			t.Fatalf("Model(%q) is a %T, not an action", dynamic, m)
		}
		assertLiveCapabilities(t, resolved.Desc(), capabilities, detected)

		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModel(m),
			ai.WithPrompt("Say hello in one sentence."))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if strings.TrimSpace(resp.Text()) == "" {
			t.Error("Text() is empty")
		}
	})
}

type liveShowResponse struct {
	Capabilities *[]string `json:"capabilities"`
}

func getLiveModelCapabilities(t *testing.T, ctx context.Context, serverAddress, modelName string) ([]string, bool) {
	t.Helper()

	body, err := json.Marshal(map[string]string{"model": modelName})
	if err != nil {
		t.Fatalf("failed to encode /api/show request: %v", err)
	}
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(serverAddress, "/")+"/api/show",
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
