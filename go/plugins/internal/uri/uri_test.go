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

package uri

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestData(t *testing.T) {
	tests := []struct {
		input    *ai.Part
		wantType string
		wantData string
		wantErr  bool
	}{
		{
			input:    ai.NewMediaPart("text/plain", "gs://storage"),
			wantType: "text/plain",
			wantData: "gs://storage",
		},
		{
			input:   ai.NewMediaPart("", "gs://storage"),
			wantErr: true,
		},
		{
			input:    ai.NewMediaPart("", "data:text/plain,a"),
			wantType: "text/plain",
			wantData: "a",
		},
		{
			input:    ai.NewMediaPart("text/plain", "data:,b"),
			wantType: "text/plain",
			wantData: "b",
		},
		{
			input:    ai.NewMediaPart("text/plain", "data:image/jpeg,c"),
			wantType: "text/plain",
			wantData: "c",
		},
		{
			input:    ai.NewMediaPart("", "data:text/plain;base64,ZA=="),
			wantType: "text/plain",
			wantData: "d",
		},
		{
			input:   ai.NewMediaPart("", "data:text/plain;base64,bad"),
			wantErr: true,
		},
		{
			input:   ai.NewTextPart("e"),
			wantErr: true,
		},
		{
			// A data part carrying a data URI on Data (not Text) still converts.
			input:    ai.NewDataPart("data:text/plain;base64,ZA=="),
			wantType: "text/plain",
			wantData: "d",
		},
		{
			input:    ai.NewDataPart("data:image/png;base64,aGVsbG8="),
			wantType: "image/png",
			wantData: "hello",
		},
	}

	for i, test := range tests {
		gotType, gotData, gotErr := Data(test.input)
		if gotErr != nil {
			if !test.wantErr {
				t.Errorf("case %d: unexpected error %v", i, gotErr)
			}
		} else if test.wantErr {
			t.Errorf("case %d: unexpected success", i)
		} else {
			if gotType != test.wantType {
				t.Errorf("case %d: got type %q, want %q", i, gotType, test.wantType)
			}
			if string(gotData) != test.wantData {
				t.Errorf("case %d: got data %q, want %q", i, gotData, test.wantData)
			}
		}
	}
}
