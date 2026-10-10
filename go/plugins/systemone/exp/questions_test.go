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
	"strconv"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal/systemone"
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
var triageQuestions = map[string]systemone.Question{
	"department": {
		Type:         systemone.KindChoice,
		Instructions: "Which team should handle this?",
		Criteria: systemone.Options{
			{Name: "billing", Description: "Payments, invoicing, refunds"},
			{Name: "other", Description: "None of the above"},
			{Name: "technical", Description: "Bugs, outages, integrations"},
		},
	},
	"is_urgent": {
		Type:         systemone.KindNoul,
		Instructions: "Does the ticket explicitly communicate time pressure?",
		Criteria:     map[string]any{"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
	},
	"frustration": {
		Type:         systemone.KindScore,
		Instructions: "How frustrated is the customer?",
		Criteria:     []any{"Calm", "Concerned but civil", "Very angry"},
		Labels:       []string{"Calm", "Concerned but civil", "Very angry"},
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
	if got := department[systemone.KindKeyword]; got != systemone.KindChoice {
		t.Errorf("department %s = %v, want %q", systemone.KindKeyword, got, systemone.KindChoice)
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
	if got := urgent[systemone.KindKeyword]; got != systemone.KindNoul {
		t.Errorf("is_urgent %s = %v, want %q", systemone.KindKeyword, got, systemone.KindNoul)
	}
	if got := urgent[systemone.TrueKeyword]; got != "Explicitly time-sensitive" {
		t.Errorf("is_urgent %s = %v: the criteria type was not applied", systemone.TrueKeyword, got)
	}
	if got := urgent[systemone.FalseKeyword]; got != "No urgency expressed" {
		t.Errorf("is_urgent %s = %v: the criteria type was not applied", systemone.FalseKeyword, got)
	}
	if got := property(t, urgent, "noul")["type"]; got != "number" {
		t.Errorf("is_urgent noul type = %v: NoulOf did not keep the plain answer schema", got)
	}

	frustration := property(t, schema, "frustration")
	if got := frustration[systemone.KindKeyword]; got != systemone.KindScore {
		t.Errorf("frustration %s = %v, want %q", systemone.KindKeyword, got, systemone.KindScore)
	}
	if got := base.JSONString(frustration[systemone.LevelsKeyword]); got != `["Calm","Concerned but civil","Very angry"]` {
		t.Errorf("frustration %s = %s", systemone.LevelsKeyword, got)
	}
	if got := property(t, frustration, "score")["maximum"]; got != float64(2) {
		t.Errorf("score maximum = %v, want 2", got)
	}
}

func TestCompileQuestions(t *testing.T) {
	got, err := systemone.CompileQuestions(triageSchema(t), "")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, triageQuestions) {
		t.Errorf("compiled questions:\n got %s\nwant %s", base.JSONString(got), base.JSONString(triageQuestions))
	}
}

func TestPreambleLeadsEveryQuestion(t *testing.T) {
	got, err := systemone.CompileQuestions(triageSchema(t), "The state is a support ticket.")
	if err != nil {
		t.Fatal(err)
	}
	for id, q := range got {
		if want := "The state is a support ticket.\n\n" + triageQuestions[id].Instructions.(string); q.Instructions != want {
			t.Errorf("%s instructions = %q, want %q", id, q.Instructions, want)
		}
	}

	// The schema's own description follows the system text.
	schema := triageSchema(t)
	schema["description"] = "Triage a ticket for the support queue."
	got, err = systemone.CompileQuestions(schema, "The state is a support ticket.")
	if err != nil {
		t.Fatal(err)
	}
	for id, q := range got {
		if want := "The state is a support ticket.\n\nTriage a ticket for the support queue.\n\n" + triageQuestions[id].Instructions.(string); q.Instructions != want {
			t.Errorf("%s instructions = %q, want %q", id, q.Instructions, want)
		}
	}
}

func TestAnswersFillTheType(t *testing.T) {
	var resp systemone.Response
	if err := json.Unmarshal([]byte(`{"model":"jev-1.13.0","answers":`+triageAnswers+`,"usage":{"input_tokens":312,"output_tokens":48}}`), &resp); err != nil {
		t.Fatal(err)
	}
	text, err := systemone.AnswersText(&resp, triageQuestions, false)
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
	questions, err := systemone.CompileQuestions(base.SchemaAsMap(base.InferJSONSchema(handoff{})), "")
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
	_, err := systemone.CompileQuestions(base.SchemaAsMap(base.InferJSONSchema(decision{})), "")
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

// The guided decision type: one question of each kind with structured
// guidance on some keys, and plain strings on the rest.

type guidedDept string

func (guidedDept) Criteria() map[guidedDept]string {
	return map[guidedDept]string{
		"billing":   "Payments, invoicing, refunds",
		"technical": "Bugs, outages, integrations",
		"other":     "None of the above",
	}
}

func (guidedDept) Guidance() map[guidedDept]any {
	return map[guidedDept]any{
		"billing":   map[string]any{"not_for": "Where an order is", "examples": []string{"I was charged twice."}},
		"technical": map[string]any{"what": "Anything that is broken", "examples": []string{"The app crashes."}},
		"other":     nil,
	}
}

type guidedAnger int

func (guidedAnger) Levels() []string { return []string{"Calm", "Concerned but civil", "Very angry"} }

func (guidedAnger) Guidance() map[int]any {
	return map[int]any{2: map[string]any{"signals": []string{"threats", "all caps"}}}
}

type guidedUrgent struct{}

func (guidedUrgent) Criteria() (yes, no string) {
	return "Explicitly time-sensitive", "No urgency expressed"
}

func (guidedUrgent) Guidance() (yes, no any) {
	return map[string]any{"examples": []string{"today", "by Friday"}}, nil
}

type guidedTriage struct {
	Department  Choice[guidedDept]   `json:"department" jsonschema_description:"Which team should handle this?"`
	IsUrgent    NoulOf[guidedUrgent] `json:"is_urgent" jsonschema_description:"Does the ticket explicitly communicate time pressure?"`
	Frustration Score[guidedAnger]   `json:"frustration" jsonschema_description:"How frustrated is the customer?"`
}

func TestGuidanceOnTheWire(t *testing.T) {
	questions, err := systemone.CompileQuestions(base.SchemaAsMap(base.InferJSONSchema(guidedTriage{})), "")
	if err != nil {
		t.Fatal(err)
	}
	// An object without a what gets the string as its what; one with a
	// what keeps it; a nil value and an absent key keep the string.
	want := map[string]string{
		"department": `{"billing":{"examples":["I was charged twice."],"not_for":"Where an order is","what":"Payments, invoicing, refunds"},` +
			`"other":"None of the above",` +
			`"technical":{"examples":["The app crashes."],"what":"Anything that is broken"}}`,
		"frustration": `["Calm","Concerned but civil",{"signals":["threats","all caps"],"what":"Very angry"}]`,
		"is_urgent":   `{"false":"No urgency expressed","true":{"examples":["today","by Friday"],"what":"Explicitly time-sensitive"}}`,
	}
	for id, criteria := range want {
		if got := base.JSONString(questions[id].Criteria); got != criteria {
			t.Errorf("%s criteria on the wire:\n got %s\nwant %s", id, got, criteria)
		}
	}

	// The schema a model reads keeps the strings as the descriptions.
	schema := base.SchemaAsMap(base.InferJSONSchema(guidedTriage{}))
	oneOf := property(t, property(t, schema, "department"), "choice")["oneOf"].([]any)
	for _, raw := range oneOf {
		option := raw.(map[string]any)
		if option["const"] == "billing" && option["description"] != "Payments, invoicing, refunds" {
			t.Errorf("billing description = %v, want the criteria string", option["description"])
		}
	}
}

func TestLegendKeepsTheRubricStrings(t *testing.T) {
	// With guidance on a level the API echoes the guidance in the legend,
	// which Score.Legend cannot hold, so the legend is rebuilt from the
	// rubric's strings.
	questions, err := systemone.CompileQuestions(base.SchemaAsMap(base.InferJSONSchema(guidedTriage{})), "")
	if err != nil {
		t.Fatal(err)
	}
	resp := &systemone.Response{Info: systemone.Info{Answers: map[string]map[string]any{
		"department": {"type": systemone.KindChoice, "choice": "billing", "probabilities": map[string]any{"billing": 1.0}, "confidence": 1.0},
		"is_urgent":  {"type": systemone.KindNoul, "noul": 0.9},
		"frustration": {"type": systemone.KindScore, "score": 1.8, "probabilities": map[string]any{"1": 0.2, "2": 0.8}, "confidence": 0.7,
			"legend": map[string]any{"0": "Calm", "1": "Concerned but civil", "2": map[string]any{"what": "Very angry", "signals": []any{"threats"}}}},
	}}}
	text, err := systemone.AnswersText(resp, questions, false)
	if err != nil {
		t.Fatal(err)
	}
	var parsed any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatal(err)
	}
	if err := base.ValidateValue(parsed, base.SchemaAsMap(base.InferJSONSchema(guidedTriage{}))); err != nil {
		t.Fatalf("answers do not validate: %v\n%s", err, text)
	}
	var out guidedTriage
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		t.Fatal(err)
	}
	if out.Frustration.Legend["2"] != "Very angry" || out.Frustration.Label() != "Very angry" {
		t.Errorf("legend = %v, label = %q, want the rubric strings", out.Frustration.Legend, out.Frustration.Label())
	}
}

func TestScoreClampedToTheRubric(t *testing.T) {
	schema := triageSchema(t)
	questions, err := systemone.CompileQuestions(schema, "")
	if err != nil {
		t.Fatal(err)
	}
	for _, score := range []float64{2.0000000000000004, -1e-17} {
		var resp systemone.Response
		answers := strings.Replace(triageAnswers, `"score": 1.3`, `"score": `+strconv.FormatFloat(score, 'g', -1, 64), 1)
		if err := json.Unmarshal([]byte(`{"answers":`+answers+`}`), &resp); err != nil {
			t.Fatal(err)
		}
		text, err := systemone.AnswersText(&resp, questions, false)
		if err != nil {
			t.Fatal(err)
		}
		var parsed any
		if err := json.Unmarshal([]byte(text), &parsed); err != nil {
			t.Fatal(err)
		}
		if err := base.ValidateValue(parsed, schema); err != nil {
			t.Errorf("score %v: answers do not validate: %v", score, err)
		}
	}
}

func TestRuntimeQuestions(t *testing.T) {
	schema := Schema(map[string]Question{
		"tool": ChoiceQuestion{
			Instructions: "Which tool serves the request?",
			Options: []ChoiceOption{
				{Name: "search", Criteria: "Look something up on the web"},
				{Name: "calendar", Criteria: "Read or change the user's calendar", Guidance: map[string]any{"not_for": "Reminders"}},
				{Name: "none"},
			},
		},
		"effort": ScoreQuestion{
			Instructions: map[string]any{"field": map[string]any{"name": "request", "description": "What the user asked for"}},
			Levels:       []string{"Trivial", "Some work", "A project"},
			Guidance:     map[int]any{2: map[string]any{"signals": []string{"several steps"}}},
		},
		"personal": NoulQuestion{
			Instructions: "Does the request involve the user's own data?",
			Yes:          "Names the user's files, mail, or calendar",
			No:           "Asks about the world at large",
		},
	})

	questions, err := systemone.CompileQuestions(schema, "")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		// The options keep the order given, not the sorted one.
		"tool": `{"type":"choice","instructions":"Which tool serves the request?",` +
			`"criteria":{"search":"Look something up on the web","calendar":{"not_for":"Reminders","what":"Read or change the user's calendar"},"none":"none"}}`,
		"effort": `{"type":"score","instructions":{"field":{"description":"What the user asked for","name":"request"}},` +
			`"criteria":["Trivial","Some work",{"signals":["several steps"],"what":"A project"}]}`,
		"personal": `{"type":"noul","instructions":"Does the request involve the user's own data?",` +
			`"criteria":{"false":"Asks about the world at large","true":"Names the user's files, mail, or calendar"}}`,
	}
	for id, wire := range want {
		if got := base.JSONString(questions[id]); got != wire {
			t.Errorf("%s on the wire:\n got %s\nwant %s", id, got, wire)
		}
	}

	// A preamble goes beside structured instructions, and in front of text.
	questions, err = systemone.CompileQuestions(schema, "The state is a user request.")
	if err != nil {
		t.Fatal(err)
	}
	if got := base.JSONString(questions["effort"].Instructions); got != `["The state is a user request.",{"field":{"description":"What the user asked for","name":"request"}}]` {
		t.Errorf("structured instructions with a preamble = %s", got)
	}
	if got := questions["tool"].Instructions; got != "The state is a user request.\n\nWhich tool serves the request?" {
		t.Errorf("text instructions with a preamble = %q", got)
	}
	// An array takes the preamble as its first element rather than nested.
	// A raw schema map can carry the array as a []string or a
	// json.RawMessage, and those spread the same way; bytes encode as a
	// string and stay nested.
	raw := Schema(map[string]Question{
		"q":     NoulQuestion{Instructions: []any{"Is the request urgent?", map[string]any{"field": "deadline"}}},
		"steps": NoulQuestion{Instructions: "placeholder"},
		"json":  NoulQuestion{Instructions: "placeholder"},
		"bytes": NoulQuestion{Instructions: "placeholder"},
	})
	props := raw["properties"].(map[string]any)
	props["steps"].(map[string]any)[systemone.InstructionsKeyword] = []string{"Read the request.", "Is it urgent?"}
	props["json"].(map[string]any)[systemone.InstructionsKeyword] = json.RawMessage(`["Read the request.",{"field":"deadline"}]`)
	props["bytes"].(map[string]any)[systemone.InstructionsKeyword] = []byte("Is it urgent?")
	listed, err := systemone.CompileQuestions(raw, "The state is a user request.")
	if err != nil {
		t.Fatal(err)
	}
	for id, want := range map[string]string{
		"q":     `["The state is a user request.","Is the request urgent?",{"field":"deadline"}]`,
		"steps": `["The state is a user request.","Read the request.","Is it urgent?"]`,
		"json":  `["The state is a user request.","Read the request.",{"field":"deadline"}]`,
		"bytes": `["The state is a user request.","SXMgaXQgdXJnZW50Pw=="]`,
	} {
		if got := base.JSONString(listed[id].Instructions); got != want {
			t.Errorf("%s instructions with a preamble = %s, want %s", id, got, want)
		}
	}

	// The answers fill a map of Answer.
	var resp systemone.Response
	if err := json.Unmarshal([]byte(`{"answers":{`+
		`"tool":{"type":"choice","choice":"calendar","probabilities":{"search":0.1,"calendar":0.85,"none":0.05},"confidence":0.7},`+
		`"effort":{"type":"score","score":0.4,"probabilities":{"0":0.6,"1":0.4,"2":0},"confidence":0.3,"legend":{"0":"Trivial","1":"Some work","2":{"what":"A project"}}},`+
		`"personal":{"type":"noul","noul":0.91}}}`), &resp); err != nil {
		t.Fatal(err)
	}
	text, err := systemone.AnswersText(&resp, questions, false)
	if err != nil {
		t.Fatal(err)
	}
	var parsed any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatal(err)
	}
	if err := base.ValidateValue(parsed, schema); err != nil {
		t.Fatalf("answers do not validate against the runtime schema: %v\n%s", err, text)
	}
	var answers map[string]Answer
	if err := json.Unmarshal([]byte(text), &answers); err != nil {
		t.Fatal(err)
	}
	if a := answers["tool"]; a.Choice != "calendar" || a.Probabilities["calendar"] != 0.85 || a.Confidence != 0.7 {
		t.Errorf("tool = %+v", a)
	}
	if a := answers["effort"]; a.Score != 0.4 || a.Legend["2"] != "A project" {
		t.Errorf("effort = %+v, want the level strings in the legend", a)
	}
	if a := answers["personal"]; a.Probability != 0.91 {
		t.Errorf("personal = %+v", a)
	}
}

func TestRuntimeQuestionsRejects(t *testing.T) {
	tests := []struct {
		name      string
		questions map[string]Question
		want      string
	}{
		{"nil question", map[string]Question{"q": nil}, "not a question"},
		{"no instructions", map[string]Question{"q": NoulQuestion{}}, "no instructions"},
		{"no options", map[string]Question{"q": ChoiceQuestion{Instructions: "d"}}, "no options"},
		{"option twice", map[string]Question{"q": ChoiceQuestion{Instructions: "d", Options: []ChoiceOption{{Name: "a"}, {Name: "a"}}}}, "twice"},
		{"one level", map[string]Question{"q": ScoreQuestion{Instructions: "d", Levels: []string{"only"}}}, "two levels"},
		{"half criteria", map[string]Question{"q": NoulQuestion{Instructions: "d", Yes: "y"}}, "only one side"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := systemone.CompileQuestions(Schema(tt.questions), "")
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %v, want one containing %q", err, tt.want)
			}
		})
	}
}

func TestAnswerRoundTripKeepsZeros(t *testing.T) {
	// A zero is a real reading: a noul of 0, a score clamped to 0, the
	// confidence of a flat distribution. Re-encoding an answer, as a flow
	// that returns one does, must keep it, and must not add another kind's
	// fields.
	var answers map[string]Answer
	if err := json.Unmarshal([]byte(`{
		"flat":  {"choice": "a", "probabilities": {"a": 0.5, "b": 0.5}, "confidence": 0},
		"low":   {"score": 0, "probabilities": {"0": 1}, "confidence": 1, "legend": {"0": "None"}},
		"never": {"noul": 0}
	}`), &answers); err != nil {
		t.Fatal(err)
	}
	for id, kind := range map[string]string{"flat": systemone.KindChoice, "low": systemone.KindScore, "never": systemone.KindNoul} {
		if answers[id].Type != kind {
			t.Errorf("%s type = %q, want %q read from its fields", id, answers[id].Type, kind)
		}
	}
	want := `{"flat":{"choice":"a","confidence":0,"probabilities":{"a":0.5,"b":0.5},"type":"choice"},` +
		`"low":{"confidence":1,"legend":{"0":"None"},"probabilities":{"0":1},"score":0,"type":"score"},` +
		`"never":{"noul":0,"type":"noul"}}`
	if got := base.JSONString(answers); got != want {
		t.Errorf("re-encoded answers:\n got %s\nwant %s", got, want)
	}
}

func TestResponseInfoOfNilInfo(t *testing.T) {
	if info := ResponseInfo(&ai.ModelResponse{Raw: (*Info)(nil)}); info.Model != "" {
		t.Errorf("info of a nil Info = %+v, want zero", info)
	}
}
