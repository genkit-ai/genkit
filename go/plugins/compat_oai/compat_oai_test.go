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

package compat_oai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// headerRecordingServer serves one chat completion and reports the headers of
// the last request it answered.
func headerRecordingServer(t *testing.T) (url string, lastHeader func() http.Header) {
	t.Helper()
	var mu sync.Mutex
	var header http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		header = r.Header.Clone()
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	t.Cleanup(server.Close)
	return server.URL, func() http.Header {
		mu.Lock()
		defer mu.Unlock()
		return header
	}
}

// assertHeaderAbsent fails unless the header carries no values at all.
// [http.Header.Get] cannot tell an absent header from one sent with an empty
// value, and clearing the identity without a WithHeaderDel leaves exactly
// that on the wire, so asserting on Get would pass either way.
func assertHeaderAbsent(t *testing.T, header http.Header, name string) {
	t.Helper()
	if values := header.Values(name); len(values) != 0 {
		t.Errorf("%s = %q, want the header dropped rather than blanked", name, values)
	}
}

// generateOnce runs one request through an initialized plugin.
func generateOnce(t *testing.T, o *OpenAICompatible) {
	t.Helper()
	ctx := context.Background()
	g := genkit.Init(ctx)
	genkit.RegisterAction(g, o.NewModel("m", ai.ModelOptions{}))
	if _, err := genkit.Generate(ctx, g, ai.WithModelName(o.Provider+"/m"), ai.WithPrompt("hi")); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
}

// TestInitDropsInheritedOpenAIIdentity pins the boundary the OpenAI SDK does
// not draw for us: openai.NewClient prepends OPENAI_API_KEY, OPENAI_ORG_ID,
// and OPENAI_PROJECT_ID to every client, so a plugin serving another provider
// would forward OpenAI's identity to that provider's endpoint. None of the
// three may reach the wire, whether the plugin configures a key of its own or
// not, and the plugin's own key must survive the scrub.
func TestInitDropsInheritedOpenAIIdentity(t *testing.T) {
	tests := []struct {
		name     string
		apiKey   string
		wantAuth string
	}{
		{
			name:     "provider key wins",
			apiKey:   "provider-key",
			wantAuth: "Bearer provider-key",
		},
		{
			// The plugins that panic on a missing key never get here; a
			// plugin that instead authenticates out of Opts, or not at all,
			// must send nothing rather than OpenAI's key.
			name:     "no key sends none",
			apiKey:   "",
			wantAuth: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OPENAI_API_KEY", "sk-must-not-leak")
			t.Setenv("OPENAI_ORG_ID", "org-must-not-leak")
			t.Setenv("OPENAI_PROJECT_ID", "proj-must-not-leak")

			url, lastHeader := headerRecordingServer(t)
			o := &OpenAICompatible{Provider: "testprovider", APIKey: tt.apiKey, BaseURL: url}
			o.Init(context.Background())
			generateOnce(t, o)

			got := lastHeader()
			if tt.wantAuth == "" {
				assertHeaderAbsent(t, got, "Authorization")
			} else if auth := got.Get("Authorization"); auth != tt.wantAuth {
				t.Errorf("Authorization = %q, want %q", auth, tt.wantAuth)
			}
			assertHeaderAbsent(t, got, "OpenAI-Organization")
			assertHeaderAbsent(t, got, "OpenAI-Project")
		})
	}
}

// TestInitOptsOverrideTheIdentityScrub pins the scrub as a floor rather than a
// ban: it applies before what the plugin composes, so a plugin that wants an
// OpenAI-shaped header sets it and wins. This is how the openai plugin keeps
// serving its own organization and project.
func TestInitOptsOverrideTheIdentityScrub(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-must-not-leak")
	t.Setenv("OPENAI_ORG_ID", "org-must-not-leak")
	t.Setenv("OPENAI_PROJECT_ID", "proj-must-not-leak")

	url, lastHeader := headerRecordingServer(t)
	o := &OpenAICompatible{
		Provider: "testprovider",
		APIKey:   "provider-key",
		BaseURL:  url,
		Opts: []option.RequestOption{
			option.WithOrganization("org-chosen"),
			option.WithProject("proj-chosen"),
		},
	}
	o.Init(context.Background())
	generateOnce(t, o)

	got := lastHeader()
	if org := got.Get("OpenAI-Organization"); org != "org-chosen" {
		t.Errorf("OpenAI-Organization = %q, want the configured organization", org)
	}
	if project := got.Get("OpenAI-Project"); project != "proj-chosen" {
		t.Errorf("OpenAI-Project = %q, want the configured project", project)
	}
	if auth := got.Get("Authorization"); auth != "Bearer provider-key" {
		t.Errorf("Authorization = %q, want the plugin's own key", auth)
	}
}

// TestClientForKeyKeepsTheIdentityScrub pins the scrub on the second place a
// client is built. A per-request key override clones the plugin's options, so
// the scrub has to live in them rather than only in the client Init makes.
func TestClientForKeyKeepsTheIdentityScrub(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-must-not-leak")
	t.Setenv("OPENAI_ORG_ID", "org-must-not-leak")
	t.Setenv("OPENAI_PROJECT_ID", "proj-must-not-leak")

	url, lastHeader := headerRecordingServer(t)
	o := &OpenAICompatible{Provider: "testprovider", APIKey: "provider-key", BaseURL: url}
	o.Init(context.Background())

	client := o.clientForKey("request-key")
	if _, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Model:    "m",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	}); err != nil {
		t.Fatalf("Completions.New() error = %v", err)
	}

	got := lastHeader()
	if auth := got.Get("Authorization"); auth != "Bearer request-key" {
		t.Errorf("Authorization = %q, want the per-request key", auth)
	}
	assertHeaderAbsent(t, got, "OpenAI-Organization")
	assertHeaderAbsent(t, got, "OpenAI-Project")
}
