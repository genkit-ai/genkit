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
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
)

// fakeJev answers whatever questions it is sent, so the tests can check the
// whole path from a Go type to the wire and back. A choice is answered with
// its first option in sorted order, a score with 1.3, a noul with 0.93.
type fakeJev struct {
	recorder
	models string // the GET /v1/models reply
	status int    // when set, every call fails with it
}

func (f *fakeJev) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method == http.MethodGet {
		if f.models == "" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = io.WriteString(w, f.models)
		return
	}
	_, body := f.record(req)
	if f.status != 0 {
		w.Header().Set("Retry-After", "0")
		w.WriteHeader(f.status)
		return
	}

	questions, _ := body["questions"].(map[string]any)
	answers := map[string]any{}
	for id, raw := range questions {
		q := raw.(map[string]any)
		switch q["type"] {
		case kindChoice:
			criteria := q["criteria"].(map[string]any)
			keys := slices.Sorted(maps.Keys(criteria))
			probabilities := map[string]float64{}
			for i, k := range keys {
				probabilities[k] = 0.1
				if i == 0 {
					probabilities[k] = 1 - 0.1*float64(len(keys)-1)
				}
			}
			answers[id] = map[string]any{"type": kindChoice, "choice": keys[0], "probabilities": probabilities, "confidence": 0.6}
		case kindScore:
			levels := q["criteria"].([]any)
			legend := map[string]any{}
			for i, level := range levels {
				legend[strconv.Itoa(i)] = level
			}
			answers[id] = map[string]any{"type": kindScore, "score": 1.3, "legend": legend, "probabilities": map[string]float64{"0": 0, "1": 0.7, "2": 0.3}, "confidence": 0.54}
		case kindNoul:
			answers[id] = map[string]any{"type": kindNoul, "noul": 0.93}
		}
	}
	reply := map[string]any{
		"model":   "jev-1.13.0",
		"answers": answers,
		"usage":   map[string]int{"input_tokens": 312, "output_tokens": 48},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(reply)
}

// newGenkit starts a fake endpoint and a Genkit with the plugin pointed at
// it. The endpoint defaults to the direct one.
func newGenkit(t *testing.T, fake *fakeJev, ep *Endpoint) *genkit.Genkit {
	t.Helper()
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	return genkit.Init(t.Context(), genkit.WithPlugins(&TypeSafe{
		APIKey:   "test-key",
		BaseURL:  srv.URL,
		Endpoint: ep,
	}))
}

func TestGenerateData(t *testing.T) {
	// The documented call: the model, the state, and the type. The
	// questions ride on the output schema, so no format is named.
	fake := &fakeJev{}
	g := newGenkit(t, fake, nil)
	out, resp, err := genkit.GenerateData[triage](t.Context(), g,
		ai.WithModelName("typesafe/jev-1.13.0"),
		ai.WithPrompt("I was charged twice and need the duplicate refunded today."))
	if err != nil {
		t.Fatal(err)
	}

	// What went over the wire.
	_, body := fake.last(t)
	if body["model"] != "jev-1.13.0" {
		t.Errorf("model = %v", body["model"])
	}
	if body["state"] != "I was charged twice and need the duplicate refunded today." {
		t.Errorf("state = %v: a single text part is the string state, nothing added", body["state"])
	}
	var questions map[string]question
	if err := json.Unmarshal([]byte(base.JSONString(body["questions"])), &questions); err != nil {
		t.Fatal(err)
	}
	// Compared as JSON: the round trip turns a []string into []any.
	if got, want := base.JSONString(questions), base.JSONString(triageQuestions); got != want {
		t.Errorf("questions on the wire:\n got %s\nwant %s", got, want)
	}

	// What came back, typed.
	if out.Department.Choice != "billing" || out.Department.Confidence != 0.6 || out.Department.Probabilities["billing"] != 0.8 {
		t.Errorf("department = %+v", out.Department)
	}
	if out.IsUrgent.Probability != 0.93 {
		t.Errorf("is_urgent = %+v", out.IsUrgent)
	}
	if out.Frustration.Score != 1.3 || out.Frustration.Legend["2"] != "Very angry" {
		t.Errorf("frustration = %+v", out.Frustration)
	}

	// What the trace and the caller see on the response.
	if resp.Usage == nil || resp.Usage.InputTokens != 312 || resp.Usage.OutputTokens != 48 || resp.Usage.TotalTokens != 360 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	custom, _ := resp.Custom.(map[string]any)
	if custom["model"] != "jev-1.13.0" || custom["answers"] == nil {
		t.Errorf("custom = %v", resp.Custom)
	}
	if resp.FinishReason != ai.FinishReasonStop {
		t.Errorf("finish reason = %v", resp.FinishReason)
	}
}

func TestStateShapes(t *testing.T) {
	fake := &fakeJev{}
	g := newGenkit(t, fake, nil)
	const model = "typesafe/jev-latest"
	decide := func(t *testing.T, opts ...ai.GenerateOption) any {
		t.Helper()
		opts = append(opts, ai.WithModelName(model), ai.WithOutputType(triage{}))
		if _, err := genkit.Generate(t.Context(), g, opts...); err != nil {
			t.Fatal(err)
		}
		_, body := fake.last(t)
		return body["state"]
	}

	t.Run("object", func(t *testing.T) {
		state := decide(t, ai.WithPromptParts(ai.NewDataPart(map[string]any{"ticket": "hi", "tier": "business"})))
		if want := map[string]any{"ticket": "hi", "tier": "business"}; !reflect.DeepEqual(state, want) {
			t.Errorf("state = %v, want the object itself", state)
		}
	})
	t.Run("history", func(t *testing.T) {
		state := decide(t, ai.WithMessages(
			ai.NewSystemMessage(ai.NewTextPart("Be terse.")),
			ai.NewUserMessage(ai.NewTextPart("Hi")),
			ai.NewModelMessage(ai.NewTextPart("Hello. How can I help?")),
			ai.NewUserMessage(ai.NewTextPart("My card was charged twice.")),
		))
		// The system message is instructions, not a record of the state.
		want := []any{
			map[string]any{"role": "user", "content": "Hi"},
			map[string]any{"role": "model", "content": "Hello. How can I help?"},
			map[string]any{"role": "user", "content": "My card was charged twice."},
		}
		if !reflect.DeepEqual(state, want) {
			t.Errorf("state = %v, want records with roles", state)
		}
	})
	t.Run("history with loop plumbing", func(t *testing.T) {
		// A history recorded from another model's turn keeps the output
		// instructions the loop injected for that model. They are plumbing,
		// not something anyone said, so they reach neither the state nor
		// the questions.
		plumbing := ai.NewTextPart("Output should be in JSON format.")
		plumbing.Metadata = map[string]any{"purpose": "output"}
		state := decide(t, ai.WithMessages(
			ai.NewSystemMessage(plumbing),
			ai.NewUserMessage(ai.NewTextPart("My card was charged twice."), plumbing),
			ai.NewModelMessage(ai.NewTextPart(`{"refund": true}`)),
		))
		want := []any{
			map[string]any{"role": "user", "content": "My card was charged twice."},
			map[string]any{"role": "model", "content": `{"refund": true}`},
		}
		if !reflect.DeepEqual(state, want) {
			t.Errorf("state = %v, want the plumbing left out", state)
		}
		_, body := fake.last(t)
		if got := base.JSONString(body["questions"]); strings.Contains(got, "JSON format") {
			t.Errorf("the plumbing reached the questions: %s", got)
		}
	})
	t.Run("documents", func(t *testing.T) {
		state := decide(t,
			ai.WithPrompt("Is the refund policy 30 days?"),
			ai.WithDocs(
				ai.DocumentFromText("Refunds within 30 days.", map[string]any{"id": "policy-1"}),
				ai.DocumentFromText("Shipping takes a week.", nil),
			))
		want := map[string]any{
			"messages": []any{map[string]any{"role": "user", "content": "Is the refund policy 30 days?"}},
			"context": []any{
				map[string]any{"content": "Refunds within 30 days.", "metadata": map[string]any{"id": "policy-1"}},
				"Shipping takes a week.",
			},
		}
		if !reflect.DeepEqual(state, want) {
			t.Errorf("state = %s, want %s", base.JSONString(state), base.JSONString(want))
		}
	})
	t.Run("stateJSON", func(t *testing.T) {
		state := decide(t, ai.WithPrompt(`{"ticket": "hi"}`), ai.WithConfig(&Config{StateJSON: true}))
		if want := map[string]any{"ticket": "hi"}; !reflect.DeepEqual(state, want) {
			t.Errorf("state = %v, want the parsed object", state)
		}
	})
	t.Run("extra", func(t *testing.T) {
		decide(t, ai.WithPrompt("hi"), ai.WithConfig(&Config{Extra: map[string]any{"session_id": "s-1"}}))
		_, body := fake.last(t)
		if body["session_id"] != "s-1" {
			t.Errorf("session_id = %v", body["session_id"])
		}
	})
}

func TestRefusals(t *testing.T) {
	fake := &fakeJev{}
	g := newGenkit(t, fake, nil)
	const model = "typesafe/jev-latest"

	t.Run("no output type", func(t *testing.T) {
		_, err := genkit.Generate(t.Context(), g, ai.WithModelName(model), ai.WithPrompt("hi"))
		if err == nil || !strings.Contains(err.Error(), "output type") {
			t.Errorf("error = %v", err)
		}
	})
	t.Run("plain output type", func(t *testing.T) {
		type mood struct {
			Mood string `json:"mood" jsonschema_description:"How does the customer feel?"`
		}
		_, _, err := genkit.GenerateData[mood](t.Context(), g, ai.WithModelName(model), ai.WithPrompt("hi"))
		if err == nil || !strings.Contains(err.Error(), "not a question") {
			t.Errorf("error = %v", err)
		}
		if fake.calls() != 0 {
			t.Error("the endpoint was called for a type that is not a question set")
		}
	})
	t.Run("media", func(t *testing.T) {
		_, err := genkit.Generate(t.Context(), g, ai.WithModelName(model), ai.WithOutputType(triage{}),
			ai.WithMessages(ai.NewUserMessage(ai.NewMediaPart("image/png", "data:image/png;base64,AAAA"))))
		if err == nil || !strings.Contains(err.Error(), "media") {
			t.Errorf("error = %v", err)
		}
	})
	t.Run("stateJSON on prose", func(t *testing.T) {
		_, err := genkit.Generate(t.Context(), g, ai.WithModelName(model), ai.WithOutputType(triage{}),
			ai.WithPrompt("not json"), ai.WithConfig(&Config{StateJSON: true}))
		if err == nil || !strings.Contains(err.Error(), "not JSON") {
			t.Errorf("error = %v", err)
		}
	})
}

func TestEnumFormat(t *testing.T) {
	fake := &fakeJev{}
	g := newGenkit(t, fake, nil)

	// The enum option carries no description, so the question gets the
	// default instructions.
	resp, err := genkit.Generate(t.Context(), g,
		ai.WithModelName("typesafe/jev-latest"),
		ai.WithOutputEnums("technical", "billing"),
		ai.WithPrompt("My card was charged twice."))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "billing" {
		t.Errorf("text = %q, want the chosen option", resp.Text())
	}
	_, body := fake.last(t)
	q := body["questions"].(map[string]any)[enumQuestionID].(map[string]any)
	if q["type"] != kindChoice || q["instructions"] == "" {
		t.Errorf("enum question on the wire = %v", q)
	}

	// A schema with a description names the instructions.
	if _, err := genkit.Generate(t.Context(), g,
		ai.WithModelName("typesafe/jev-latest"),
		ai.WithOutputSchema(map[string]any{"description": "Which team should handle this?", "enum": []string{"technical", "billing"}}),
		ai.WithOutputFormat(ai.OutputFormatEnum),
		ai.WithPrompt("My card was charged twice.")); err != nil {
		t.Fatal(err)
	}
	_, body = fake.last(t)
	q = body["questions"].(map[string]any)[enumQuestionID].(map[string]any)
	if q["instructions"] != "Which team should handle this?" {
		t.Errorf("enum question on the wire = %v", q)
	}

	// The system message is the question, and it leaves the state alone.
	if _, err := genkit.Generate(t.Context(), g,
		ai.WithModelName("typesafe/jev-latest"),
		ai.WithSystem("Which team should handle this?"),
		ai.WithOutputEnums("technical", "billing"),
		ai.WithPrompt("My card was charged twice.")); err != nil {
		t.Fatal(err)
	}
	_, body = fake.last(t)
	q = body["questions"].(map[string]any)[enumQuestionID].(map[string]any)
	if q["instructions"] != "Which team should handle this?" {
		t.Errorf("enum question on the wire = %v", q)
	}
	if body["state"] != "My card was charged twice." {
		t.Errorf("state = %v, want the prompt alone", body["state"])
	}
}

func TestSystemMessageIsInstructions(t *testing.T) {
	fake := &fakeJev{}
	g := newGenkit(t, fake, nil)
	const model = "typesafe/jev-latest"

	t.Run("preamble on every question", func(t *testing.T) {
		if _, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithSystem("The state is a support ticket."),
			ai.WithPrompt("My card was charged twice.")); err != nil {
			t.Fatal(err)
		}
		_, body := fake.last(t)
		if body["state"] != "My card was charged twice." {
			t.Errorf("state = %v, want the prompt alone", body["state"])
		}
		for id, raw := range body["questions"].(map[string]any) {
			q := raw.(map[string]any)
			if want := "The state is a support ticket.\n\n" + triageQuestions[id].Instructions; q["instructions"] != want {
				t.Errorf("%s instructions = %q, want %q", id, q["instructions"], want)
			}
		}
	})
	t.Run("system alone is not state", func(t *testing.T) {
		_, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithMessages(ai.NewSystemMessage(ai.NewTextPart("The state is a support ticket."))))
		if err == nil || !strings.Contains(err.Error(), "no state") {
			t.Errorf("error = %v", err)
		}
	})
	t.Run("system is text only", func(t *testing.T) {
		_, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithMessages(
				ai.NewSystemMessage(ai.NewDataPart(map[string]any{"tier": "business"})),
				ai.NewUserMessage(ai.NewTextPart("hi"))))
		if err == nil || !strings.Contains(err.Error(), "text only") {
			t.Errorf("error = %v", err)
		}
	})
}

func TestOpenRouterThroughGenerate(t *testing.T) {
	fake := &fakeJev{}
	g := newGenkit(t, fake, OpenRouter())
	out, _, err := genkit.GenerateData[triage](t.Context(), g,
		ai.WithModelName("typesafe/jev-1.13.0"),
		ai.WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}
	req, body := fake.last(t)
	if req.URL.Path != "/api/alpha/decisions" || body["model"] != "typesafe/jev-1.13" {
		t.Errorf("request = %s model=%v", req.URL.Path, body["model"])
	}
	if out.Department.Choice != "billing" {
		t.Errorf("out = %+v", out)
	}
}

func TestListActions(t *testing.T) {
	t.Run("listed by the API", func(t *testing.T) {
		fake := &fakeJev{models: `{"models":[{"name":"jev-1.13.0","description":"Current"},{"name":"jev-1.14.0-preview"}]}`}
		g := newGenkit(t, fake, nil)
		plugin := genkit.LookupPlugin(g, provider).(*TypeSafe)
		var names []string
		for _, desc := range plugin.ListActions(t.Context()) {
			if desc.Type != api.ActionTypeModel {
				t.Errorf("listed a %s action", desc.Type)
			}
			names = append(names, desc.Name)
		}
		if want := []string{"typesafe/jev-1.13.0", "typesafe/jev-1.14.0-preview"}; !reflect.DeepEqual(names, want) {
			t.Errorf("listed %v, want %v", names, want)
		}
	})
	t.Run("listing unavailable", func(t *testing.T) {
		g := newGenkit(t, &fakeJev{}, nil)
		plugin := genkit.LookupPlugin(g, provider).(*TypeSafe)
		var names []string
		for _, desc := range plugin.ListActions(t.Context()) {
			names = append(names, desc.Name)
		}
		if want := []string{"typesafe/jev-latest", "typesafe/jev-1.13.0"}; !reflect.DeepEqual(names, want) {
			t.Errorf("listed %v, want the known models %v", names, want)
		}
	})
	t.Run("gateway", func(t *testing.T) {
		g := newGenkit(t, &fakeJev{}, Cloudflare("acct"))
		plugin := genkit.LookupPlugin(g, provider).(*TypeSafe)
		if descs := plugin.ListActions(t.Context()); len(descs) != 1 || descs[0].Name != "typesafe/jev-latest" {
			t.Errorf("listed %v", descs)
		}
	})
	t.Run("resolves any id", func(t *testing.T) {
		g := newGenkit(t, &fakeJev{}, nil)
		if m := genkit.LookupModel(g, "typesafe/jev-9.9.9"); m == nil {
			t.Error("an unlisted version did not resolve")
		}
		if genkit.LookupModel(g, "typesafe/jev-latest") == nil {
			t.Error("the alias did not resolve")
		}
		for _, id := range []string{"jev-latest", "typesafe/jev-latest"} {
			if ref := ModelRef(id, nil); ref.Name() != "typesafe/jev-latest" || ref.Config() != nil {
				t.Errorf("ModelRef(%q) = %q with config %v, want typesafe/jev-latest with none", id, ref.Name(), ref.Config())
			}
		}
	})
}

func TestInitRequiresAKey(t *testing.T) {
	t.Setenv("TYPESAFE_API_KEY", "")
	defer func() {
		if r := recover(); r == nil || !strings.Contains(r.(string), "TYPESAFE_API_KEY") {
			t.Errorf("Init without a key: recovered %v, want a panic naming the variable", r)
		}
	}()
	(&TypeSafe{}).Init(t.Context())
}

func TestAnswersProjectedOntoDeclaredFields(t *testing.T) {
	// A field the API or a gateway adds to an answer must not reach the
	// message: the answer schemas are closed, so it would fail validation on
	// every call. The untouched answers stay on Custom.
	resp := &response{Answers: map[string]map[string]any{
		"department":  {"type": kindChoice, "choice": "billing", "probabilities": map[string]any{"billing": 1.0}, "confidence": 1.0, "explanation": "new"},
		"is_urgent":   {"type": kindNoul, "noul": 0.9, "reasoning": "new"},
		"frustration": {"type": kindScore, "score": 1.0, "legend": map[string]any{"0": "Calm"}, "probabilities": map[string]any{"0": 1.0}, "confidence": 1.0, "rank": 3},
	}}
	text, err := answersText(resp, triageQuestions, false)
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

func TestPromptFile(t *testing.T) {
	// A prompt file names the decision type as a registered schema and
	// renders the state as JSON. The registered schema keeps the question
	// keywords through the registry, and no output format needs naming.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "triage.prompt"), []byte(`---
model: typesafe/jev-1.13.0
config:
  stateJSON: true
input:
  schema: ticketInput
output:
  schema: triage
---
{{role "system"}}
The state is a support ticket.

{{role "user"}}
{
  "ticket": {{json ticket}}
}
`), 0o644); err != nil {
		t.Fatal(err)
	}
	type ticketInput struct {
		Ticket string `json:"ticket"`
	}
	fake := &fakeJev{}
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	g := genkit.Init(t.Context(),
		genkit.WithPlugins(&TypeSafe{APIKey: "test-key", BaseURL: srv.URL}),
		genkit.WithPromptDir(dir))
	genkit.DefineSchemasFor(g, ticketInput{}, triage{})

	prompt := genkit.LookupPrompt(g, "triage")
	if prompt == nil {
		t.Fatal("triage.prompt was not loaded")
	}
	resp, err := prompt.Execute(t.Context(), ai.WithInput(ticketInput{Ticket: "charged twice"}))
	if err != nil {
		t.Fatal(err)
	}
	var out triage
	if err := resp.Output(&out); err != nil {
		t.Fatal(err)
	}
	if out.Department.Choice != "billing" || out.Frustration.Score != 1.3 {
		t.Errorf("out = %+v", out)
	}
	_, body := fake.last(t)
	if !reflect.DeepEqual(body["state"], map[string]any{"ticket": "charged twice"}) {
		t.Errorf("state = %v, want the rendered JSON as an object", body["state"])
	}
	questions, _ := body["questions"].(map[string]any)
	department, _ := questions["department"].(map[string]any)
	if got, _ := department["instructions"].(string); !strings.HasPrefix(got, "The state is a support ticket.") {
		t.Errorf("instructions = %q, want the system message in front", got)
	}
}
