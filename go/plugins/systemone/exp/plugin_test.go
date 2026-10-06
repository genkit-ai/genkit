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
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal/systemone"
	"github.com/firebase/genkit/go/plugins/internal/systemone/systemonetest"
)

// newGenkit starts a fake endpoint and a Genkit with the TypeSafe plugin
// pointed at it.
func newGenkit(t *testing.T, fake *systemonetest.Server) *genkit.Genkit {
	t.Helper()
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	return genkit.Init(t.Context(), genkit.WithPlugins(testTypeSafe(srv.URL)))
}

// testTypeSafe is the TypeSafe plugin pointed at a fake endpoint.
func testTypeSafe(baseURL string) *SystemOne {
	p := TypeSafe()
	p.APIKey = "test-key"
	p.BaseURL = baseURL
	return p
}

func TestGenerateData(t *testing.T) {
	// The documented call: the model, the state, and the type. The
	// questions ride on the output schema, so no format is named.
	fake := &systemonetest.Server{}
	g := newGenkit(t, fake)
	out, resp, err := genkit.GenerateData[triage](t.Context(), g,
		ai.WithModelName("typesafe/jev-1.13.0"),
		ai.WithPrompt("I was charged twice and need the duplicate refunded today."))
	if err != nil {
		t.Fatal(err)
	}

	// What went over the wire.
	_, body := fake.Last(t)
	if body["model"] != "jev-1.13.0" {
		t.Errorf("model = %v", body["model"])
	}
	if body["state"] != "I was charged twice and need the duplicate refunded today." {
		t.Errorf("state = %v: a single text part is the string state, nothing added", body["state"])
	}
	var questions map[string]systemone.Question
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
	if info := ResponseInfo(resp); info.Model != "jev-1.13.0" || info.Answers["department"]["choice"] != "billing" {
		t.Errorf("info = %+v", info)
	}
	if resp.FinishReason != ai.FinishReasonStop {
		t.Errorf("finish reason = %v", resp.FinishReason)
	}

	// A response that travelled as JSON, as a flow's output or a trace
	// does, keeps the same info.
	var decoded ai.ModelResponse
	if err := json.Unmarshal([]byte(base.JSONString(resp)), &decoded); err != nil {
		t.Fatal(err)
	}
	if info := ResponseInfo(&decoded); info.Model != "jev-1.13.0" || info.Answers["department"]["choice"] != "billing" {
		t.Errorf("info after a JSON round trip = %+v", info)
	}
	if info := ResponseInfo(&ai.ModelResponse{Raw: "another model's"}); info.Model != "" {
		t.Errorf("info of another model's response = %+v, want zero", info)
	}
}

func TestStateShapes(t *testing.T) {
	fake := &systemonetest.Server{}
	g := newGenkit(t, fake)
	const model = "typesafe/jev-latest"
	decide := func(t *testing.T, opts ...ai.GenerateOption) any {
		t.Helper()
		opts = append(opts, ai.WithModelName(model), ai.WithOutputType(triage{}))
		if _, err := genkit.Generate(t.Context(), g, opts...); err != nil {
			t.Fatal(err)
		}
		_, body := fake.Last(t)
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
		_, body := fake.Last(t)
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
				&ai.Document{Content: []*ai.Part{ai.NewDataPart(map[string]any{"sku": "A-1", "stock": 3})}},
			))
		want := map[string]any{
			"messages": []any{map[string]any{"role": "user", "content": "Is the refund policy 30 days?"}},
			"context": []any{
				map[string]any{"content": "Refunds within 30 days.", "metadata": map[string]any{"id": "policy-1"}},
				"Shipping takes a week.",
				map[string]any{"sku": "A-1", "stock": float64(3)},
			},
		}
		if !reflect.DeepEqual(state, want) {
			t.Errorf("state = %s, want %s", base.JSONString(state), base.JSONString(want))
		}
	})
	t.Run("document with media", func(t *testing.T) {
		_, err := genkit.Generate(t.Context(), g, ai.WithModelName(model), ai.WithOutputType(triage{}),
			ai.WithPrompt("Is this the product?"),
			ai.WithDocs(&ai.Document{Content: []*ai.Part{ai.NewMediaPart("image/png", "data:image/png;base64,AAAA")}}))
		if err == nil || !strings.Contains(err.Error(), "media") {
			t.Errorf("error = %v, want the media part refused", err)
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
		_, body := fake.Last(t)
		if body["session_id"] != "s-1" {
			t.Errorf("session_id = %v", body["session_id"])
		}
	})
}

func TestRefusals(t *testing.T) {
	fake := &systemonetest.Server{}
	g := newGenkit(t, fake)
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
		if fake.Calls() != 0 {
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
	fake := &systemonetest.Server{}
	g := newGenkit(t, fake)

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
	_, body := fake.Last(t)
	q := body["questions"].(map[string]any)[systemone.EnumQuestionID].(map[string]any)
	if q["type"] != systemone.KindChoice || q["instructions"] == "" {
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
	_, body = fake.Last(t)
	q = body["questions"].(map[string]any)[systemone.EnumQuestionID].(map[string]any)
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
	_, body = fake.Last(t)
	q = body["questions"].(map[string]any)[systemone.EnumQuestionID].(map[string]any)
	if q["instructions"] != "Which team should handle this?" {
		t.Errorf("enum question on the wire = %v", q)
	}
	if body["state"] != "My card was charged twice." {
		t.Errorf("state = %v, want the prompt alone", body["state"])
	}
}

func TestSystemMessageIsInstructions(t *testing.T) {
	fake := &systemonetest.Server{}
	g := newGenkit(t, fake)
	const model = "typesafe/jev-latest"

	t.Run("preamble on every question", func(t *testing.T) {
		if _, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithSystem("The state is a support ticket."),
			ai.WithPrompt("My card was charged twice.")); err != nil {
			t.Fatal(err)
		}
		_, body := fake.Last(t)
		if body["state"] != "My card was charged twice." {
			t.Errorf("state = %v, want the prompt alone", body["state"])
		}
		for id, raw := range body["questions"].(map[string]any) {
			q := raw.(map[string]any)
			if want := "The state is a support ticket.\n\n" + triageQuestions[id].Instructions.(string); q["instructions"] != want {
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

func TestRuntimeQuestionsThroughGenerate(t *testing.T) {
	// Options known only at run time: the schema is built from data, and
	// the answers come back as a map of Answer.
	fake := &systemonetest.Server{}
	g := newGenkit(t, fake)
	tools := []struct{ name, description string }{
		{"search", "Look something up on the web"},
		{"calendar", "Read or change the user's calendar"},
	}
	options := make([]ChoiceOption, 0, len(tools))
	for _, tool := range tools {
		options = append(options, ChoiceOption{Name: tool.name, Criteria: tool.description})
	}
	answers, _, err := genkit.GenerateData[map[string]Answer](t.Context(), g,
		ai.WithModelName("typesafe/jev-latest"),
		ai.WithOutputSchema(Schema(map[string]Question{
			"tool":     ChoiceQuestion{Instructions: "Which tool serves the request?", Options: options},
			"personal": NoulQuestion{Instructions: "Does the request involve the user's own data?"},
		})),
		ai.WithPrompt("What is on my calendar tomorrow?"))
	if err != nil {
		t.Fatal(err)
	}
	_, body := fake.Last(t)
	tool, _ := body["questions"].(map[string]any)["tool"].(map[string]any)
	if got := base.JSONString(tool["criteria"]); got != `{"calendar":"Read or change the user's calendar","search":"Look something up on the web"}` {
		t.Errorf("tool criteria on the wire = %s", got)
	}
	// The fake picks the first option by name.
	if a := (*answers)["tool"]; a.Choice != "calendar" || a.Confidence != 0.6 {
		t.Errorf("tool = %+v", a)
	}
	if a := (*answers)["personal"]; a.Probability != 0.93 {
		t.Errorf("personal = %+v", a)
	}
}

func TestListActions(t *testing.T) {
	names := func(g *genkit.Genkit) []string {
		var names []string
		for _, desc := range genkit.LookupPlugin(g, "typesafe").(*SystemOne).ListActions(t.Context()) {
			if desc.Type != api.ActionTypeModel {
				t.Errorf("listed a %s action", desc.Type)
			}
			names = append(names, desc.Name)
		}
		return names
	}
	t.Run("listed by the API", func(t *testing.T) {
		// The listing adds to the models known ahead, without repeats.
		g := newGenkit(t, &systemonetest.Server{Models: `{"models":[{"name":"jev-1.13.0","description":"Current"},{"name":"jev-1.14.0-preview"}]}`})
		if got, want := names(g), []string{"typesafe/jev-1.13.0", "typesafe/jev-1.14.0-preview", "typesafe/jev-latest"}; !reflect.DeepEqual(got, want) {
			t.Errorf("listed %v, want %v", got, want)
		}
	})
	t.Run("listing unavailable", func(t *testing.T) {
		g := newGenkit(t, &systemonetest.Server{})
		if got, want := names(g), []string{"typesafe/jev-1.13.0", "typesafe/jev-latest"}; !reflect.DeepEqual(got, want) {
			t.Errorf("listed %v, want the known models %v", got, want)
		}
	})
	t.Run("listing kept", func(t *testing.T) {
		// The Dev UI lists actions often; the server is asked once.
		fake := &systemonetest.Server{Models: `{"models":[{"name":"jev-1.13.0"},{"name":""}]}`}
		g := newGenkit(t, fake)
		names(g)
		if got, want := names(g), []string{"typesafe/jev-1.13.0", "typesafe/jev-latest"}; !reflect.DeepEqual(got, want) {
			t.Errorf("listed %v, want %v without the entry that has no ID", got, want)
		}
		if n := len(fake.Listings()); n != 1 {
			t.Errorf("the server was asked for %d listings, want 1", n)
		}
	})
	t.Run("models read at Init", func(t *testing.T) {
		// Keys may carry the prefix, and a change after Init has no
		// effect, so it cannot race with a listing.
		srv := httptest.NewServer(&systemonetest.Server{})
		t.Cleanup(srv.Close)
		p := testTypeSafe(srv.URL)
		p.Models = map[string]ModelSpec{"typesafe/jev-1.13.0": {Label: "Pinned"}}
		g := genkit.Init(t.Context(), genkit.WithPlugins(p))
		p.Models["jev-1.14.0"] = ModelSpec{}
		p.Provider = "renamed"
		if got, want := names(g), []string{"typesafe/jev-1.13.0"}; !reflect.DeepEqual(got, want) {
			t.Fatalf("listed %v, want %v", got, want)
		}
		if ref := p.ModelRef("jev-1.13.0", nil); ref.Name() != "typesafe/jev-1.13.0" {
			t.Errorf("ModelRef after a rename = %q, want the name Init read", ref.Name())
		}
		if desc := p.ListActions(t.Context())[0]; !strings.Contains(base.JSONString(desc.Metadata), `"label":"Pinned"`) {
			t.Errorf("the label from the prefixed key was not used")
		}
	})
	t.Run("resolves any id", func(t *testing.T) {
		g := newGenkit(t, &systemonetest.Server{})
		if m := genkit.LookupModel(g, "typesafe/jev-9.9.9"); m == nil {
			t.Error("an unlisted version did not resolve")
		}
		if ref := TypeSafe().ModelRef("jev-latest", nil); ref.Name() != "typesafe/jev-latest" || ref.Config() != nil {
			t.Errorf("ModelRef = %q with config %v, want typesafe/jev-latest with none", ref.Name(), ref.Config())
		}
		if ref := TypeSafe().ModelRef("typesafe/jev-latest", nil); ref.Name() != "typesafe/jev-latest" {
			t.Errorf("ModelRef of the full name = %q, want it taken as Models takes it", ref.Name())
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
	TypeSafe().Init(t.Context())
}

func TestInitRequires(t *testing.T) {
	for _, tt := range []struct {
		name   string
		plugin *SystemOne
		want   string
	}{
		{"a provider", &SystemOne{BaseURL: "http://localhost:11434"}, "Provider"},
		{"a base URL", &SystemOne{Provider: "local"}, "BaseURL"},
		{"one key per model", &SystemOne{Provider: "local", BaseURL: "http://localhost:11434",
			Models: map[string]ModelSpec{"d1": {}, "local/d1": {}}}, "twice"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil || !strings.Contains(r.(string), tt.want) {
					t.Errorf("recovered %v, want a panic naming %s", r, tt.want)
				}
			}()
			tt.plugin.Init(t.Context())
		})
	}
}

func TestGenericServer(t *testing.T) {
	// A server with no constructor: the fields are the whole setup, and a
	// local one takes no key.
	fake := &systemonetest.Server{}
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	g := genkit.Init(t.Context(), genkit.WithPlugins(
		&SystemOne{Provider: "local", BaseURL: srv.URL + "/decisions/"},
		&SystemOne{Provider: "custom", BaseURL: srv.URL, Path: "decide"},
		&SystemOne{Provider: "whole", BaseURL: srv.URL + "/endpoint", Path: "/"},
	))

	if _, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("local/vendor/d1"), ai.WithPrompt("hi")); err != nil {
		t.Fatal(err)
	}
	req, body := fake.Last(t)
	if req.URL.Path != "/decisions/v1/systemone" {
		t.Errorf("path = %s, want the default path under the base URL", req.URL.Path)
	}
	if body["model"] != "vendor/d1" {
		t.Errorf("model = %v, want the server's ID as given", body["model"])
	}
	if got := req.Header.Values("Authorization"); got != nil {
		t.Errorf("Authorization = %q, want none without a key", got)
	}

	// A server's ID is kept whole, even one that starts with the provider.
	if _, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("local/local/clef"), ai.WithPrompt("hi")); err != nil {
		t.Fatal(err)
	}
	if _, body := fake.Last(t); body["model"] != "local/clef" {
		t.Errorf("model = %v, want local/clef", body["model"])
	}

	// A path without a leading slash still joins the base URL.
	if _, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("custom/d1"), ai.WithPrompt("hi")); err != nil {
		t.Fatal(err)
	}
	if req, _ := fake.Last(t); req.URL.Path != "/decide" {
		t.Errorf("path = %s, want /decide", req.URL.Path)
	}

	// A path of "/" posts to the base URL itself.
	if _, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("whole/d1"), ai.WithPrompt("hi")); err != nil {
		t.Fatal(err)
	}
	if req, _ := fake.Last(t); req.URL.Path != "/endpoint" {
		t.Errorf("path = %s, want the base URL itself", req.URL.Path)
	}
}

func TestWireHooks(t *testing.T) {
	// A server whose wire is not the native one is served from the
	// fields: Route wraps the request, and Unwrap takes the response out
	// of the server's envelope.
	fake := &systemonetest.Server{Wrap: func(reply map[string]any) any {
		return map[string]any{"result": reply}
	}}
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	g := genkit.Init(t.Context(), genkit.WithPlugins(&SystemOne{
		Provider: "wrapped",
		BaseURL:  srv.URL,
		Path:     "/run",
		Route: func(model string, body map[string]any) (string, any, error) {
			delete(body, "model")
			return "/" + model, map[string]any{"input": body}, nil
		},
		Unwrap: func(body []byte) ([]byte, error) {
			var envelope struct{ Result json.RawMessage }
			err := json.Unmarshal(body, &envelope)
			return envelope.Result, err
		},
	}))

	out, _, err := genkit.GenerateData[triage](t.Context(), g, ai.WithModelName("wrapped/d1"), ai.WithPrompt("hi"))
	if err != nil {
		t.Fatal(err)
	}
	req, body := fake.Last(t)
	if req.URL.Path != "/run/d1" {
		t.Errorf("path = %s, want the route's path after Path", req.URL.Path)
	}
	if input, _ := body["input"].(map[string]any); input["state"] != "hi" || input["model"] != nil {
		t.Errorf("body = %v, want the native body under input, without a model", body)
	}
	if out.Department.Choice != "billing" {
		t.Errorf("the envelope was not unwrapped: %+v", out)
	}
}

func TestExtraCannotReplaceTheRequest(t *testing.T) {
	fake := &systemonetest.Server{}
	g := newGenkit(t, fake)
	for _, field := range []string{"model", "state", "questions", "images"} {
		_, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("typesafe/jev-latest"),
			ai.WithPrompt("hi"),
			ai.WithConfig(&Config{Extra: map[string]any{field: "x"}}))
		if err == nil || !strings.Contains(err.Error(), field) {
			t.Errorf("extra %s: error = %v, want it refused", field, err)
		}
	}
	if fake.Calls() != 0 {
		t.Error("a request with a reserved extra field was sent")
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
	fake := &systemonetest.Server{}
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	g := genkit.Init(t.Context(),
		genkit.WithPlugins(testTypeSafe(srv.URL)),
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
	_, body := fake.Last(t)
	if !reflect.DeepEqual(body["state"], map[string]any{"ticket": "charged twice"}) {
		t.Errorf("state = %v, want the rendered JSON as an object", body["state"])
	}
	questions, _ := body["questions"].(map[string]any)
	department, _ := questions["department"].(map[string]any)
	if got, _ := department["instructions"].(string); !strings.HasPrefix(got, "The state is a support ticket.") {
		t.Errorf("instructions = %q, want the system message in front", got)
	}
}
