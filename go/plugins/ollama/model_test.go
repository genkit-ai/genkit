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

package ollama

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestOllamaChatRequest_MarshalJSON(t *testing.T) {
	req := &ollamaChatRequest{
		Model: "qwen3",
		Think: ThinkEnabled(true),
		Options: map[string]any{
			"temperature": 0.7,
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	jsonStr := string(data)
	if !strings.Contains(jsonStr, `"think":true`) {
		t.Errorf("expected json to contain \"think\":true, got: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"options":{"temperature":0.7}`) {
		t.Errorf("expected json to contain \"options\":{\"temperature\":0.7}, got: %s", jsonStr)
	}
}

func TestOllamaChatRequest_FormatField(t *testing.T) {
	t.Run("string json mode", func(t *testing.T) {
		req := &ollamaChatRequest{Model: "llama3", Format: "json"}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		got := string(data)
		if !strings.Contains(got, `"format":"json"`) {
			t.Errorf("expected \"format\":\"json\", got: %s", got)
		}
	})

	t.Run("schema object mode", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"answer": map[string]any{"type": "string"},
			},
		}
		req := &ollamaChatRequest{Model: "llama3", Format: schema}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		got := string(data)
		// format must be a JSON object, not the string "json"
		if strings.Contains(got, `"format":"json"`) {
			t.Errorf("format should be a JSON object, not the string \"json\": %s", got)
		}
		if !strings.Contains(got, `"format":{"`) {
			t.Errorf("expected format to be a JSON object, got: %s", got)
		}
		if !strings.Contains(got, `"type":"object"`) {
			t.Errorf("expected schema type in format, got: %s", got)
		}
	})

	t.Run("nil omits field", func(t *testing.T) {
		req := &ollamaChatRequest{Model: "llama3"}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		got := string(data)
		if strings.Contains(got, `"format"`) {
			t.Errorf("expected \"format\" key to be absent, got: %s", got)
		}
	})
}

func TestOllamaChatRequest_ApplyOptions(t *testing.T) {
	tests := []struct {
		name string
		cfg  GenerateContentConfig
		want *ollamaChatRequest
	}{
		{
			name: "configured values",
			cfg: GenerateContentConfig{
				Seed:        Ptr(42),
				Temperature: Ptr(0.7),
				Think:       ThinkEnabled(true),
				KeepAlive:   "10m",
			},
			want: &ollamaChatRequest{
				Think:     ThinkEnabled(true),
				KeepAlive: "10m",
				Options: map[string]any{
					"seed":        42,
					"temperature": 0.7,
				},
			},
		},
		{
			name: "explicit zero values",
			cfg: GenerateContentConfig{
				Seed:        Ptr(0),
				Temperature: Ptr(0.0),
			},
			want: &ollamaChatRequest{
				Options: map[string]any{
					"seed":        0,
					"temperature": 0.0,
				},
			},
		},
		{
			name: "thinking effort",
			cfg: GenerateContentConfig{
				Think: ThinkEffort("high"),
			},
			want: &ollamaChatRequest{
				Think: ThinkEffort("high"),
			},
		},
		{
			name: "zero config",
			cfg:  GenerateContentConfig{},
			want: &ollamaChatRequest{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &ollamaChatRequest{}
			req.ApplyOptions(tt.cfg)

			if !reflect.DeepEqual(req, tt.want) {
				t.Errorf(
					"unexpected result:\nwant: %#v\n got: %#v",
					tt.want,
					req,
				)
			}
		})
	}
}
