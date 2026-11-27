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
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/internal/base"
)

func TestCompileQuestionsRejects(t *testing.T) {
	prop := func(fields map[string]any) map[string]any {
		return map[string]any{"type": "object", "properties": map[string]any{"q": fields}}
	}
	tests := []struct {
		name   string
		schema map[string]any
		want   string
	}{
		{"no properties", map[string]any{"type": "string"}, "no properties"},
		{"plain field", prop(map[string]any{"type": "string", "description": "d"}), "not a question"},
		{"no instructions", prop(map[string]any{KindKeyword: KindNoul}), "no instructions"},
		{"unknown kind", prop(map[string]any{KindKeyword: "vibe", "description": "d"}), "unknown type"},
		{"choice without options", prop(map[string]any{KindKeyword: KindChoice, "description": "d"}), "no options"},
		{"one level", prop(map[string]any{KindKeyword: KindScore, "description": "d", LevelsKeyword: []any{"only"}}), "at least two levels"},
		{"half a noul", prop(map[string]any{KindKeyword: KindNoul, "description": "d", TrueKeyword: "yes"}), "only one side"},
		{"guidance for no option", prop(map[string]any{
			KindKeyword: KindChoice, "description": "d",
			"properties":    map[string]any{"choice": map[string]any{"oneOf": []any{map[string]any{"const": "a"}}}},
			GuidanceKeyword: map[string]any{"b": "x"},
		}), "not one of its options"},
		{"guidance for no level", prop(map[string]any{
			KindKeyword: KindScore, "description": "d", LevelsKeyword: []any{"low", "high"},
			GuidanceKeyword: map[string]any{"2": "x"},
		}), "does not have"},
		{"guidance for one side", prop(map[string]any{
			KindKeyword: KindNoul, "description": "d",
			GuidanceKeyword: map[string]any{"true": map[string]any{"examples": []any{"today"}}},
		}), "only one side"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := CompileQuestions(tt.schema, "")
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %v, want one containing %q", err, tt.want)
			}
		})
	}
}

func TestEnumQuestion(t *testing.T) {
	// The options keep the order the values were given in.
	got, err := EnumQuestion(map[string]any{"enum": []string{"technical", "billing"}, "description": "Which team?"}, "")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]Question{EnumQuestionID: {
		Type:         KindChoice,
		Instructions: "Which team?",
		Criteria:     Options{{"technical", "technical"}, {"billing", "billing"}},
	}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("enum question = %s, want %s", base.JSONString(got), base.JSONString(want))
	}

	got, err = EnumQuestion(map[string]any{"enum": []any{"a", "b"}}, "")
	if err != nil {
		t.Fatal(err)
	}
	if got[EnumQuestionID].Instructions == "" {
		t.Error("an enum without a description got no default instructions")
	}

	// The system text is the question, and a schema description follows it.
	got, err = EnumQuestion(map[string]any{"enum": []any{"a", "b"}}, "Which team?")
	if err != nil {
		t.Fatal(err)
	}
	if got[EnumQuestionID].Instructions != "Which team?" {
		t.Errorf("instructions = %q, want the system text alone", got[EnumQuestionID].Instructions)
	}
	got, err = EnumQuestion(map[string]any{"enum": []any{"a", "b"}, "description": "Pick one."}, "Context.")
	if err != nil {
		t.Fatal(err)
	}
	if got[EnumQuestionID].Instructions != "Context.\n\nPick one." {
		t.Errorf("instructions = %q, want the system text and then the description", got[EnumQuestionID].Instructions)
	}

	if _, err := EnumQuestion(map[string]any{"type": "string"}, ""); err == nil {
		t.Error("a schema without enum values was accepted")
	}

	resp := &Response{Info: Info{Answers: map[string]map[string]any{EnumQuestionID: {"type": "choice", "choice": "billing"}}}}
	text, err := AnswersText(resp, got, true)
	if err != nil || text != "billing" {
		t.Errorf("enum answer text = %q, %v; want the option itself", text, err)
	}
}

func TestNilGuidanceKeepsTheDescription(t *testing.T) {
	if got := withWhat(map[string]any(nil), "Payments"); got != "Payments" {
		t.Errorf("withWhat(nil map) = %v, want the description", got)
	}
}
