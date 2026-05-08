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

package openai_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	compat_oai "github.com/firebase/genkit/go/plugins/compat_oai/openai"
	"github.com/openai/openai-go/option"
)

// TestToolCallReasoning verifies that the compat_oai plugin correctly handles
// reasoning_content for OpenAI-compatible backends that require it
func TestToolCallReasoning(t *testing.T) {
	vars := []string{"MOONSHOT_API_KEY", "MOONSHOT_BASE_URL", "MOONSHOT_MODEL"}
	var missing []string
	for _, v := range vars {
		if os.Getenv(v) == "" {
			missing = append(missing, v)
		}
	}
	if len(missing) > 0 {
		t.Skipf("missing env vars: %s", strings.Join(missing, ", "))
	}

	ctx := t.Context()
	apiKey := os.Getenv(vars[0])
	baseURL := os.Getenv(vars[1])
	model := os.Getenv(vars[2])

	oai := &compat_oai.OpenAI{
		APIKey: apiKey,
		Opts:   []option.RequestOption{option.WithBaseURL(baseURL)},
	}
	g := genkit.Init(ctx, genkit.WithPlugins(oai))

	gablorkenTool := defineGablorkenTool(g)

	baseOpts := []ai.GenerateOption{ai.WithModelName("openai/" + model)}

	tests := []struct {
		name         string
		tool         ai.ToolRef
		firstPrompt  string
		secondPrompt string
		wantText     string
		streaming    bool
	}{
		{
			name:         "prompt eval",
			tool:         gablorkenTool,
			firstPrompt:  "what is a gablorken of 2 over 4?",
			secondPrompt: "What was the result of the previous calculation?",
			wantText:     "16",
		},
		{
			name:         "prompt eval with streaming",
			tool:         gablorkenTool,
			firstPrompt:  "what is a gablorken of 3 over 3?",
			secondPrompt: "What was the result of the previous calculation?",
			wantText:     "27",
			streaming:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := append([]ai.GenerateOption{}, baseOpts...)
			opts = append(opts, ai.WithPrompt(tt.firstPrompt), ai.WithTools(tt.tool))
			if tt.streaming {
				opts = append(opts, ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					t.Logf("chunk text=%s", chunk.Text())
					return nil
				}))
			}
			resp1, err := genkit.Generate(ctx, g, opts...)
			if err != nil {
				t.Fatal("Generate failed:", err)
			}
			if resp1.Text() == "" {
				t.Fatalf("empty response from the model: %#v", resp1)
			}
			t.Logf("first model response: %v", resp1.Text())

			resp2, err := genkit.Generate(ctx, g, append(baseOpts,
				ai.WithPrompt(tt.secondPrompt),
				ai.WithMessages(resp1.History()...),
			)...)
			if err != nil {
				t.Fatalf("could not send second request with %v", err)
			}
			if !strings.Contains(resp2.Text(), tt.wantText) {
				t.Errorf("second response does not contain %q: %s", tt.wantText, resp2.Text())
			}
			t.Logf("second model response: %v", resp2.Text())
		})
	}
}
