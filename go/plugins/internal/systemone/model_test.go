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

package systemone

import (
	"encoding/json"
	"maps"
	"slices"
	"strings"
	"testing"
)

func TestAnswersTextRejectsMissingAnswer(t *testing.T) {
	resp := &Response{Info: Info{Answers: map[string]map[string]any{"department": {"type": "choice", "choice": "billing"}}}}
	if _, err := AnswersText(resp, testQuestions, false); err == nil || !strings.Contains(err.Error(), `"frustration"`) {
		t.Errorf("error = %v, want one naming the first missing question in ID order", err)
	}
}

func TestAnswersProjectedOntoDeclaredFields(t *testing.T) {
	// A field the API or a gateway adds to an answer must not reach the
	// message: the answer schemas are closed, so it would fail validation on
	// every call. The untouched answers stay on the response's Info.
	resp := &Response{Info: Info{Answers: map[string]map[string]any{
		"department":  {"type": KindChoice, "choice": "billing", "probabilities": map[string]any{"billing": 1.0}, "confidence": 1.0, "explanation": "new"},
		"is_urgent":   {"type": KindNoul, "noul": 0.9, "reasoning": "new"},
		"frustration": {"type": KindScore, "score": 1.0, "legend": map[string]any{"0": "Calm"}, "probabilities": map[string]any{"0": 1.0}, "confidence": 1.0, "rank": 3},
	}}}
	text, err := AnswersText(resp, testQuestions, false)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]map[string]any
	if err := json.Unmarshal([]byte(text), &got); err != nil {
		t.Fatal(err)
	}
	want := map[string][]string{
		"department":  {"choice", "confidence", "probabilities"},
		"is_urgent":   {"noul"},
		"frustration": {"confidence", "legend", "probabilities", "score"},
	}
	for id, fields := range want {
		if keys := slices.Sorted(maps.Keys(got[id])); !slices.Equal(keys, fields) {
			t.Errorf("%s fields = %v, want %v", id, keys, fields)
		}
	}
}
