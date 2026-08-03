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

package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
)

func boolPtr(v bool) *bool { return &v }

func int64Ptr(v int64) *int64 { return &v }

func assertJSONObjectEqual(t *testing.T, got []byte, want string) {
	t.Helper()
	var gotObj, wantObj any
	if err := json.Unmarshal(got, &gotObj); err != nil {
		t.Fatalf("unmarshal got: %v", err)
	}
	if err := json.Unmarshal([]byte(want), &wantObj); err != nil {
		t.Fatalf("unmarshal want: %v", err)
	}
	gotNorm, _ := json.Marshal(gotObj)
	wantNorm, _ := json.Marshal(wantObj)
	if string(gotNorm) != string(wantNorm) {
		t.Fatalf("json = %s, want %s", gotNorm, wantNorm)
	}
}

func TestToAnthropicThinkingConfig(t *testing.T) {
	tests := []struct {
		name     string
		in       any
		wantOK   bool
		wantErr  string
		wantJSON string
	}{
		{
			name:     "enabled with budget",
			in:       map[string]any{"enabled": true, "budgetTokens": 2048},
			wantOK:   true,
			wantJSON: `{"type":"enabled","budget_tokens":2048}`,
		},
		{
			name:     "budgetTokens alone implies enabled",
			in:       map[string]any{"budgetTokens": 1536.0},
			wantOK:   true,
			wantJSON: `{"type":"enabled","budget_tokens":1536}`,
		},
		{
			name:     "explicitly disabled",
			in:       map[string]any{"enabled": false},
			wantOK:   true,
			wantJSON: `{"type":"disabled"}`,
		},
		{
			name:     "adaptive with display",
			in:       map[string]any{"adaptive": true, "display": "summarized"},
			wantOK:   true,
			wantJSON: `{"type":"adaptive","display":"summarized"}`,
		},
		{
			name:     "adaptive omitted display",
			in:       map[string]any{"adaptive": true, "display": "omitted"},
			wantOK:   true,
			wantJSON: `{"type":"adaptive","display":"omitted"}`,
		},
		{
			name:   "empty config is no-op",
			in:     map[string]any{},
			wantOK: false,
		},
		{
			name:    "enabled without budget",
			in:      map[string]any{"enabled": true},
			wantErr: "budgetTokens is required",
		},
		{
			name:    "budget below minimum",
			in:      map[string]any{"enabled": true, "budgetTokens": 512},
			wantErr: "budgetTokens must be >= 1024",
		},
		{
			name:    "non-integer budget",
			in:      map[string]any{"enabled": true, "budgetTokens": 1024.5},
			wantErr: "budgetTokens must be an integer",
		},
		{
			name:    "enabled and adaptive conflict",
			in:      map[string]any{"enabled": true, "budgetTokens": 1024, "adaptive": true},
			wantErr: "cannot use both enabled and adaptive",
		},
		{
			name:    "invalid display",
			in:      map[string]any{"adaptive": true, "display": "full"},
			wantErr: `display must be "summarized" or "omitted"`,
		},
		{
			name:    "display without adaptive",
			in:      map[string]any{"enabled": true, "budgetTokens": 2048, "display": "summarized"},
			wantErr: "display can only be set when adaptive",
		},
		{
			name:     "disabled ignores low budgetTokens",
			in:       map[string]any{"enabled": false, "budgetTokens": 100},
			wantOK:   true,
			wantJSON: `{"type":"disabled"}`,
		},
		{
			name:     "adaptive ignores budgetTokens",
			in:       map[string]any{"adaptive": true, "budgetTokens": 100},
			wantOK:   true,
			wantJSON: `{"type":"adaptive"}`,
		},
		{
			name: "typed ThinkingConfig",
			in: ThinkingConfig{
				Enabled:      boolPtr(true),
				BudgetTokens: int64Ptr(2048),
			},
			wantOK:   true,
			wantJSON: `{"type":"enabled","budget_tokens":2048}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok, err := toAnthropicThinkingConfig(tt.in)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if !tt.wantOK {
				return
			}
			if tt.wantJSON != "" {
				b, err := json.Marshal(got)
				if err != nil {
					t.Fatalf("marshal: %v", err)
				}
				assertJSONObjectEqual(t, b, tt.wantJSON)
			}
		})
	}
}

func TestConfigFromRequestThinkingMap(t *testing.T) {
	got, err := configFromRequest(&ai.ModelRequest{
		Config: map[string]any{
			"max_tokens": 256,
			"thinking": map[string]any{
				"enabled":      true,
				"budgetTokens": 2048,
			},
		},
	})
	if err != nil {
		t.Fatalf("configFromRequest: %v", err)
	}
	if got.MaxTokens != 256 {
		t.Errorf("MaxTokens = %d, want 256", got.MaxTokens)
	}
	b, err := json.Marshal(got.Thinking)
	if err != nil {
		t.Fatalf("marshal thinking: %v", err)
	}
	assertJSONObjectEqual(t, b, `{"type":"enabled","budget_tokens":2048}`)
}

func TestConfigFromRequestPreservesSDKThinkingShape(t *testing.T) {
	got, err := configFromRequest(&ai.ModelRequest{
		Config: map[string]any{
			"thinking": map[string]any{
				"type":          "enabled",
				"budget_tokens": 2048,
			},
		},
	})
	if err != nil {
		t.Fatalf("configFromRequest: %v", err)
	}
	b, err := json.Marshal(got.Thinking)
	if err != nil {
		t.Fatalf("marshal thinking: %v", err)
	}
	assertJSONObjectEqual(t, b, `{"type":"enabled","budget_tokens":2048}`)
}

func TestConfigSchemaOverlaysThinking(t *testing.T) {
	schema := ConfigSchema(anthropic.MessageNewParams{})
	props, _ := schema["properties"].(map[string]any)
	thinking, _ := props["thinking"].(map[string]any)
	if thinking == nil {
		t.Fatal("missing thinking schema")
	}
	tProps, _ := thinking["properties"].(map[string]any)
	budget, _ := tProps["budgetTokens"].(map[string]any)
	switch min := budget["minimum"].(type) {
	case int:
		if min != minThinkingBudgetTokens {
			t.Fatalf("budgetTokens.minimum = %d, want %d", min, minThinkingBudgetTokens)
		}
	case float64:
		if int(min) != minThinkingBudgetTokens {
			t.Fatalf("budgetTokens.minimum = %v, want %d", min, minThinkingBudgetTokens)
		}
	case int64:
		if int(min) != minThinkingBudgetTokens {
			t.Fatalf("budgetTokens.minimum = %d, want %d", min, minThinkingBudgetTokens)
		}
	default:
		t.Fatalf("budgetTokens.minimum = %#v", budget["minimum"])
	}
	display, _ := tProps["display"].(map[string]any)
	enum, _ := display["enum"].([]any)
	if len(enum) != 2 {
		t.Fatalf("display.enum = %#v", display["enum"])
	}
}
