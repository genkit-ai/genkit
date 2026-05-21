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

package googlegenai

import (
	"slices"
	"testing"
)

func TestCategorizeModel(t *testing.T) {
	tests := []struct {
		name             string
		modelName        string
		supportedActions []string
		wantPlaced       bool
		// pickBucket returns the slice that should have received modelName.
		// nil means "no bucket should have received it."
		pickBucket func(*genaiModels) []string
	}{
		{
			name:       "deprecated model is filtered out",
			modelName:  "gemini-3-pro-preview",
			wantPlaced: false,
			pickBucket: nil,
		},
		{
			name:             "embedder via supportedActions",
			modelName:        "text-embedding-004",
			supportedActions: []string{"embedContent"},
			wantPlaced:       true,
			pickBucket:       func(m *genaiModels) []string { return m.embedders },
		},
		{
			name:       "embedder via name fallback",
			modelName:  "text-embedding-005",
			wantPlaced: true,
			pickBucket: func(m *genaiModels) []string { return m.embedders },
		},
		{
			name:       "imagen routed to imagen bucket",
			modelName:  "imagen-3.0-generate-001",
			wantPlaced: true,
			pickBucket: func(m *genaiModels) []string { return m.imagen },
		},
		{
			name:       "veo routed to veo bucket",
			modelName:  "veo-3.0-generate-001",
			wantPlaced: true,
			pickBucket: func(m *genaiModels) []string { return m.veo },
		},
		{
			name:       "gemini routed to gemini bucket",
			modelName:  "gemini-2.5-pro",
			wantPlaced: true,
			pickBucket: func(m *genaiModels) []string { return m.gemini },
		},
		{
			name:       "gemma routed to gemini bucket",
			modelName:  "gemma-2",
			wantPlaced: true,
			pickBucket: func(m *genaiModels) []string { return m.gemini },
		},
		{
			name:       "unknown prefix is skipped",
			modelName:  "totally-unrelated-model",
			wantPlaced: false,
			pickBucket: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var m genaiModels
			got := categorizeModel(&m, tc.modelName, tc.supportedActions)
			if got != tc.wantPlaced {
				t.Fatalf("categorizeModel placed=%v, want %v", got, tc.wantPlaced)
			}
			if tc.pickBucket == nil {
				return
			}
			bucket := tc.pickBucket(&m)
			if !slices.Contains(bucket, tc.modelName) {
				t.Errorf("expected %q in bucket, got %v", tc.modelName, bucket)
			}
		})
	}
}

func TestListGenaiModelsFiltersDeprecated(t *testing.T) {
	// Drives the same categorization the live loop would, without touching
	// the SDK. Proves a deprecated name dropped at categorization time never
	// reaches any bucket — which is the contract listGenaiModels relies on.
	var m genaiModels
	categorizeModel(&m, "gemini-3-pro-preview", nil)
	categorizeModel(&m, "gemini-2.5-pro", nil)

	if slices.Contains(m.gemini, "gemini-3-pro-preview") {
		t.Errorf("deprecated model leaked into gemini bucket: %v", m.gemini)
	}
	if !slices.Contains(m.gemini, "gemini-2.5-pro") {
		t.Errorf("expected gemini-2.5-pro in bucket, got %v", m.gemini)
	}
}
