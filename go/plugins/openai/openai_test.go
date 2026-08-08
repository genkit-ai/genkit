// Copyright 2026 Google LLC
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

package openai

import "testing"

func TestIsGenerativeModel(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{name: "gpt-4.1", want: true},
		{name: "gpt-5.4", want: true},
		{name: "o4-mini", want: true},
		{name: "text-embedding-3-small", want: false},
		{name: "omni-moderation-latest", want: false},
		{name: "tts-1", want: false},
		{name: "whisper-1", want: false},
		{name: "dall-e-3", want: false},
		{name: "gpt-image-1", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isGenerativeModel(tt.name); got != tt.want {
				t.Errorf("isGenerativeModel(%q) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
