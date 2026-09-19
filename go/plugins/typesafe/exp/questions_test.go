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

package exp

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/internal/base"
)

// The decision type the tests share: one question of each kind.

type dept string

func (dept) Criteria() map[dept]string {
	return map[dept]string{
		"billing":   "Payments, invoicing, refunds",
		"technical": "Bugs, outages, integrations",
		"other":     "None of the above",
	}
}

type anger int

func (anger) Levels() []string { return []string{"Calm", "Concerned but civil", "Very angry"} }

type urgent struct{}

func (urgent) Criteria() (yes, no string) { return "Explicitly time-sensitive", "No urgency expressed" }

type triage struct {
	Department  Choice[dept]   `json:"department" jsonschema_description:"Which team should handle this?"`
	IsUrgent    NoulOf[urgent] `json:"is_urgent" jsonschema_description:"Does the ticket explicitly communicate time pressure?"`
	Frustration Score[anger]   `json:"frustration" jsonschema_description:"How frustrated is the customer?"`
}

// triageQuestions is what the triage schema compiles to.
var triageQuestions = map[string]question{
	"department": {
		Type:         kindChoice,
		Instructions: "Which team should handle this?",
		Criteria: map[string]string{
			"billing":   "Payments, invoicing, refunds",
			"technical": "Bugs, outages, integrations",
			"other":     "None of the above",
		},
	},
	"is_urgent": {
		Type:         kindNoul,
		Instructions: "Does the ticket explicitly communicate time pressure?",
		Criteria:     map[string]string{"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
	},
	"frustration": {
		Type:         kindScore,
		Instructions: "How frustrated is the customer?",
		Criteria:     []string{"Calm", "Concerned but civil", "Very angry"},
	},
}

// triageAnswers is a calibrated answer set, as the API returns it.
const triageAnswers = `{
	"department":  {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.84, "technical": 0.15, "other": 0.01}, "confidence": 0.6},
	"is_urgent":   {"type": "noul", "noul": 0.93},
	"frustration": {"type": "score", "score": 1.3, "legend": {"0": "Calm", "1": "Concerned but civil", "2": "Very angry"}, "probabilities": {"0": 0, "1": 0.7, "2": 0.3}, "confidence": 0.54}
}`

func triageSchema(t *testing.T) map[string]any {
	t.Helper()
	return base.SchemaAsMap(base.InferJSONSchema(triage{}))
}

func property(t *testing.T, schema map[string]any, name string) map[string]any {
	t.Helper()
	props, _ := schema["properties"].(map[string]any)
	prop, ok := props[name].(map[string]any)
	if !ok {
		t.Fatalf("schema has no property %q: %s", name, base.JSONString(schema))
	}
	return prop
}

func TestSchemaEncodesQuestions(t *testing.T) {
	schema := triageSchema(t)

	department := property(t, schema, "department")
	if got := department[kindKeyword]; got != kindChoice {
		t.Errorf("department %s = %v, want %q", kindKeyword, got, kindChoice)
	}
	if got := department["description"]; got != "Which team should handle this?" {
		t.Errorf("department description = %v: the field tag was not merged onto the type's schema", got)
	}
	if got := department["additionalProperties"]; got != false {
		t.Errorf("department additionalProperties = %v, want false", got)
	}
	choice := property(t, department, "choice")
	oneOf, _ := choice["oneOf"].([]any)
	var consts []string
	for _, option := range oneOf {
		o := option.(map[string]any)
		consts = append(consts, o["const"].(string))
		if o["description"] == "" {
			t.Errorf("option %v has no description", o["const"])
		}
	}
	if want := []string{"billing", "other", "technical"}; !reflect.DeepEqual(consts, want) {
		t.Errorf("choice options = %v, want %v (sorted)", consts, want)
	}

	urgent := property(t, schema, "is_urgent")
	if got := urgent[kindKeyword]; got != kindNoul {
		t.Errorf("is_urgent %s = %v, want %q", kindKeyword, got, kindNoul)
	}
	if got := urgent[trueKeyword]; got != "Explicitly time-sensitive" {
		t.Errorf("is_urgent %s = %v: the criteria type was not applied", trueKeyword, got)
	}
	if got := urgent[falseKeyword]; got != "No urgency expressed" {
		t.Errorf("is_urgent %s = %v: the criteria type was not applied", falseKeyword, got)
	}
	if got := property(t, urgent, "noul")["type"]; got != "number" {
		t.Errorf("is_urgent noul type = %v: NoulOf did not keep the plain answer schema", got)
	}

	frustration := property(t, schema, "frustration")
	if got := frustration[kindKeyword]; got != kindScore {
		t.Errorf("frustration %s = %v, want %q", kindKeyword, got, kindScore)
	}
	if got := base.JSONString(frustration[levelsKeyword]); got != `["Calm","Concerned but civil","Very angry"]` {
		t.Errorf("frustration %s = %s", levelsKeyword, got)
	}
	if got := property(t, frustration, "score")["maximum"]; got != float64(2) {
		t.Errorf("score maximum = %v, want 2", got)
	}
}

func TestCompileQuestions(t *testing.T) {
	got, err := compileQuestions(triageSchema(t), "")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, triageQuestions) {
		t.Errorf("compiled questions:\n got %s\nwant %s", base.JSONString(got), base.JSONString(triageQuestions))
	}
}

func TestPreambleLeadsEveryQuestion(t *testing.T) {
	got, err := compileQuestions(triageSchema(t), "The state is a support ticket.")
	if err != nil {
		t.Fatal(err)
	}
	for id, q := range got {
		if want := "The state is a support ticket.\n\n" + triageQuestions[id].Instructions; q.Instructions != want {
			t.Errorf("%s instructions = %q, want %q", id, q.Instructions, want)
		}
	}
}

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
		{"no instructions", prop(map[string]any{kindKeyword: kindNoul}), "no instructions"},
		{"unknown kind", prop(map[string]any{kindKeyword: "vibe", "description": "d"}), "unknown type"},
		{"choice without options", prop(map[string]any{kindKeyword: kindChoice, "description": "d"}), "no options"},
		{"one level", prop(map[string]any{kindKeyword: kindScore, "description": "d", levelsKeyword: []any{"only"}}), "at least two levels"},
		{"half a noul", prop(map[string]any{kindKeyword: kindNoul, "description": "d", trueKeyword: "yes"}), "only one side"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := compileQuestions(tt.schema, "")
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %v, want one containing %q", err, tt.want)
			}
		})
	}
}

func TestAnswersFillTheType(t *testing.T) {
	var resp response
	if err := json.Unmarshal([]byte(`{"model":"jev-1.13.0","answers":`+triageAnswers+`,"usage":{"input_tokens":312,"output_tokens":48}}`), &resp); err != nil {
		t.Fatal(err)
	}
	text, err := answersText(&resp, triageQuestions, false)
	if err != nil {
		t.Fatal(err)
	}

	// The text must validate against the type's own schema, extension
	// keywords included, since the JSON format validates it before it is
	// parsed into the type.
	var parsed any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatal(err)
	}
	if err := base.ValidateValue(parsed, triageSchema(t)); err != nil {
		t.Fatalf("answers do not validate against the decision type's schema: %v\n%s", err, text)
	}

	var out triage
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		t.Fatal(err)
	}
	if out.Department.Choice != "billing" || out.Department.Confidence != 0.6 || out.Department.Probabilities["billing"] != 0.84 {
		t.Errorf("department = %+v", out.Department)
	}
	if out.IsUrgent.Probability != 0.93 {
		t.Errorf("is_urgent = %+v", out.IsUrgent)
	}
	if out.Frustration.Score != 1.3 || out.Frustration.Legend["1"] != "Concerned but civil" || out.Frustration.Probabilities["2"] != 0.3 {
		t.Errorf("frustration = %+v", out.Frustration)
	}
	if strings.Contains(text, `"type"`) {
		t.Errorf("the type discriminator leaked into the message text: %s", text)
	}
}

type noCriteria struct{}

func (noCriteria) Criteria() (yes, no string) { return "", "" }

type halfCriteria struct{}

func (halfCriteria) Criteria() (yes, no string) { return "yes means this", "" }

func TestNoulWithoutCriteriaOmitsTheField(t *testing.T) {
	type handoff struct {
		WantsHuman Noul               `json:"wants_human" jsonschema_description:"Does the user ask for a person?"`
		Empty      NoulOf[noCriteria] `json:"empty" jsonschema_description:"Is the sky blue? The type gives an empty pair, so this is a plain noul."`
	}
	questions, err := compileQuestions(base.SchemaAsMap(base.InferJSONSchema(handoff{})), "")
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"wants_human", "empty"} {
		if wire := base.JSONString(questions[id]); strings.Contains(wire, "criteria") {
			t.Errorf("%s: a noul without criteria put the field on the wire: %s", id, wire)
		}
	}
}

func TestNoulOfRejectsHalfCriteria(t *testing.T) {
	type decision struct {
		Half NoulOf[halfCriteria] `json:"half" jsonschema_description:"d"`
	}
	_, err := compileQuestions(base.SchemaAsMap(base.InferJSONSchema(decision{})), "")
	if err == nil || !strings.Contains(err.Error(), "only one side") {
		t.Errorf("error = %v, want the half pair rejected before the model is called", err)
	}
}

func TestNoulOfFillsFromThePlainAnswer(t *testing.T) {
	var out NoulOf[urgent]
	if err := json.Unmarshal([]byte(`{"noul": 0.93}`), &out); err != nil {
		t.Fatal(err)
	}
	if out.Probability != 0.93 {
		t.Errorf("out = %+v, want the probability", out)
	}
	if text := base.JSONString(out); text != `{"noul":0.93}` {
		t.Errorf("JSON = %s, want the plain answer", text)
	}
	// Every instantiation shares one underlying struct, so a helper that
	// takes the plain Noul takes any of them by conversion.
	if plain := Noul(out); plain.Probability != 0.93 {
		t.Errorf("Noul(out) = %+v", plain)
	}
}

func TestAnswersTextRejectsMissingAnswer(t *testing.T) {
	resp := &response{Answers: map[string]map[string]any{"department": {"type": "choice", "choice": "billing"}}}
	if _, err := answersText(resp, triageQuestions, false); err == nil || !strings.Contains(err.Error(), `"frustration"`) {
		t.Errorf("error = %v, want one naming the first missing question in ID order", err)
	}
}

func TestEnumQuestion(t *testing.T) {
	got, err := enumQuestion(map[string]any{"enum": []string{"billing", "technical"}, "description": "Which team?"}, "")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]question{enumQuestionID: {
		Type:         kindChoice,
		Instructions: "Which team?",
		Criteria:     map[string]string{"billing": "billing", "technical": "technical"},
	}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("enum question = %s, want %s", base.JSONString(got), base.JSONString(want))
	}

	got, err = enumQuestion(map[string]any{"enum": []any{"a", "b"}}, "")
	if err != nil {
		t.Fatal(err)
	}
	if got[enumQuestionID].Instructions == "" {
		t.Error("an enum without a description got no default instructions")
	}

	// The system text is the question, and a schema description follows it.
	got, err = enumQuestion(map[string]any{"enum": []any{"a", "b"}}, "Which team?")
	if err != nil {
		t.Fatal(err)
	}
	if got[enumQuestionID].Instructions != "Which team?" {
		t.Errorf("instructions = %q, want the system text alone", got[enumQuestionID].Instructions)
	}
	got, err = enumQuestion(map[string]any{"enum": []any{"a", "b"}, "description": "Pick one."}, "Context.")
	if err != nil {
		t.Fatal(err)
	}
	if got[enumQuestionID].Instructions != "Context.\n\nPick one." {
		t.Errorf("instructions = %q, want the system text and then the description", got[enumQuestionID].Instructions)
	}

	if _, err := enumQuestion(map[string]any{"type": "string"}, ""); err == nil {
		t.Error("a schema without enum values was accepted")
	}

	resp := &response{Answers: map[string]map[string]any{enumQuestionID: {"type": "choice", "choice": "billing"}}}
	text, err := answersText(resp, got, true)
	if err != nil || text != "billing" {
		t.Errorf("enum answer text = %q, %v; want the option itself", text, err)
	}
}

func TestScoreLevelAndLabel(t *testing.T) {
	for _, tt := range []struct {
		score float64
		level int
		label string
	}{
		{1.3, 1, "Concerned but civil"},
		{1.5, 2, "Very angry"},
		{2.7, 2, "Very angry"},
		{-0.4, 0, "Calm"},
	} {
		s := Score[anger]{Score: tt.score}
		if s.Level() != tt.level || s.Label() != tt.label {
			t.Errorf("Score %v: level %d %q, want %d %q", tt.score, s.Level(), s.Label(), tt.level, tt.label)
		}
	}
}

func TestChoiceRankedAndMargin(t *testing.T) {
	c := Choice[dept]{Choice: "billing", Probabilities: map[dept]float64{"billing": 0.75, "technical": 0.25, "other": 0}}
	if got, want := c.Ranked(), []dept{"billing", "technical", "other"}; !slices.Equal(got, want) {
		t.Errorf("Ranked = %v, want %v", got, want)
	}
	if got := c.Margin(); got != 0.5 {
		t.Errorf("Margin = %v, want 0.5", got)
	}

	tied := Choice[dept]{Probabilities: map[dept]float64{"technical": 0.5, "billing": 0.5}}
	if got, want := tied.Ranked(), []dept{"billing", "technical"}; !slices.Equal(got, want) {
		t.Errorf("tied Ranked = %v, want ties by name %v", got, want)
	}
	if got := tied.Margin(); got != 0 {
		t.Errorf("tied Margin = %v, want 0", got)
	}

	var empty Choice[dept]
	if len(empty.Ranked()) != 0 || empty.Margin() != 0 {
		t.Errorf("no distribution: Ranked = %v, Margin = %v, want none and 0", empty.Ranked(), empty.Margin())
	}
}
