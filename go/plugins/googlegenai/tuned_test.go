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

import "testing"

func TestClassifyModelTunedEndpoint(t *testing.T) {
	cases := []struct {
		name string
		want ModelType
	}{
		{"endpoints/1234567890", ModelTypeGemini},
		{"projects/my-proj/locations/us-central1/endpoints/1234567890", ModelTypeGemini},
		{"gemini-2.5-flash", ModelTypeGemini},
		{"imagen-3.0-generate-001", ModelTypeImagen},
		{"veo-3.0-generate-001", ModelTypeVeo},
		{"text-embedding-004", ModelTypeEmbedder},
		{"random-name-with-no-prefix", ModelTypeUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyModel(tc.name); got != tc.want {
				t.Fatalf("ClassifyModel(%q) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

func TestIsTunedGeminiName(t *testing.T) {
	cases := []struct {
		name string
		want bool
	}{
		{"endpoints/1234567890", true},
		{"projects/p/locations/us-central1/endpoints/999", true},
		{"gemini-2.5-flash", false},
		{"imagen-3.0-generate-001", false},
		{"projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := isTunedGeminiName(tc.name); got != tc.want {
			t.Errorf("isTunedGeminiName(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}
