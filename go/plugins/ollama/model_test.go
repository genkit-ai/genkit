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

func TestOllamaChatRequest_ApplyGenerateContentConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  *GenerateContentConfig
		want *ollamaChatRequest
	}{
		{
			name: "seed, temperature, and think",
			cfg: &GenerateContentConfig{
				Seed:        Ptr(42),
				Temperature: Ptr(0.7),
				Think:       ThinkEnabled(true),
			},
			want: &ollamaChatRequest{
				Think: ThinkEnabled(true),
				Options: map[string]any{
					"seed":        42,
					"temperature": 0.7,
				},
			},
		},
		{
			name: "zero values are preserved",
			cfg: &GenerateContentConfig{
				Seed:        Ptr(0),
				Temperature: Ptr(0.0),
				Think:       ThinkEnabled(true),
			},
			want: &ollamaChatRequest{
				Think: ThinkEnabled(true),
				Options: map[string]any{
					"seed":        0,
					"temperature": 0.0,
				},
			},
		},
		{
			name: "sampling options",
			cfg: &GenerateContentConfig{
				TopK: Ptr(40),
				TopP: Ptr(0.9),
				MinP: Ptr(0.05),
			},
			want: &ollamaChatRequest{
				Options: map[string]any{
					"top_k": 40,
					"top_p": 0.9,
					"min_p": 0.05,
				},
			},
		},
		{
			name: "stop, context, and prediction limits",
			cfg: &GenerateContentConfig{
				Stop:       []string{"END"},
				NumCtx:     Ptr(2048),
				NumPredict: Ptr(128),
			},
			want: &ollamaChatRequest{
				Options: map[string]any{
					"stop":        []string{"END"},
					"num_ctx":     2048,
					"num_predict": 128,
				},
			},
		},
		{
			name: "think effort (GPT-OSS)",
			cfg: &GenerateContentConfig{
				Think: ThinkEffort("high"),
			},
			want: &ollamaChatRequest{
				Think: ThinkEffort("high"),
			},
		},
		{
			name: "keep alive",
			cfg: &GenerateContentConfig{
				KeepAlive: "10m",
			},
			want: &ollamaChatRequest{
				KeepAlive: "10m",
			},
		},
		{
			name: "nil config",
			cfg:  nil,
			want: &ollamaChatRequest{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &ollamaChatRequest{}

			req.applyGenerateContentConfig(tt.cfg)

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
