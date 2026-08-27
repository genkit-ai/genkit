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

package ai

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
)

type fnTestInput struct {
	Name  string `json:"name"`
	Theme string `json:"theme"`
}

// capturePrompt defines a prompt against a request-capturing model.
func capturePrompt(t *testing.T, name string, opts ...PromptOption) (api.Registry, Prompt, *func() *ModelRequest) {
	t.Helper()
	r := newTestRegistry(t)
	var captured *ModelRequest
	m := defineFakeModel(t, r, fakeModelConfig{
		name: "test/capture_" + name,
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			captured = req
			return &ModelResponse{Message: NewModelTextMessage("ok")}, nil
		},
	})
	get := func() *ModelRequest { return captured }
	p := DefinePrompt(r, name, append([]PromptOption{WithModel(m)}, opts...)...)
	return r, p, &get
}

// TestContentFnInputTypeAcrossCallPaths covers the defect that made these
// options unusable: the concrete type of the value handed to a content function
// used to depend on how the prompt was invoked, so a function written against
// the prompt's input type panicked from the Dev UI and from the default-input
// path.
func TestContentFnInputTypeAcrossCallPaths(t *testing.T) {
	var seen []fnTestInput
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{name: "test/echoTyped"})

	p := DefinePrompt(r, "typedInput",
		WithModel(m),
		WithInputType(fnTestInput{Theme: "pirate"}),
		WithPromptFn(func(ctx context.Context, in fnTestInput) (string, error) {
			seen = append(seen, in)
			return "hi " + in.Name, nil
		}),
	)

	t.Run("typed Go value", func(t *testing.T) {
		seen = nil
		if _, err := p.Execute(context.Background(), WithInput(fnTestInput{Name: "bob", Theme: "ninja"})); err != nil {
			t.Fatal(err)
		}
		if want := (fnTestInput{Name: "bob", Theme: "ninja"}); seen[0] != want {
			t.Errorf("input = %+v, want %+v", seen[0], want)
		}
	})

	t.Run("default input from WithInputType", func(t *testing.T) {
		seen = nil
		if _, err := p.Execute(context.Background()); err != nil {
			t.Fatal(err)
		}
		if want := (fnTestInput{Theme: "pirate"}); seen[0] != want {
			t.Errorf("input = %+v, want %+v", seen[0], want)
		}
	})

	t.Run("map input", func(t *testing.T) {
		seen = nil
		if _, err := p.Execute(context.Background(), WithInput(map[string]any{"name": "carol", "theme": "noir"})); err != nil {
			t.Fatal(err)
		}
		if want := (fnTestInput{Name: "carol", Theme: "noir"}); seen[0] != want {
			t.Errorf("input = %+v, want %+v", seen[0], want)
		}
	})

	t.Run("reflection API", func(t *testing.T) {
		seen = nil
		act := r.ResolveAction("/executable-prompt/typedInput")
		if act == nil {
			t.Fatal("prompt action not registered")
		}
		if _, err := act.RunJSON(context.Background(), json.RawMessage(`{"name":"dave","theme":"noir"}`), nil); err != nil {
			t.Fatal(err)
		}
		if want := (fnTestInput{Name: "dave", Theme: "noir"}); seen[0] != want {
			t.Errorf("input = %+v, want %+v", seen[0], want)
		}
	})

	t.Run("Generate has no input and gets the zero value", func(t *testing.T) {
		seen = nil
		_, err := Generate(context.Background(), r,
			WithModel(m),
			WithPromptFn(func(ctx context.Context, in fnTestInput) (string, error) {
				seen = append(seen, in)
				return "hi", nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}
		if seen[0] != (fnTestInput{}) {
			t.Errorf("input = %+v, want zero value", seen[0])
		}
	})
}

// TestContentFnLooseInputKeepsSchemaTypes extends the guarantee above from the
// declared type to the Go types inside it. A loosely typed input cannot restore
// integer-ness from JSON on its own, so without normalization the same field
// arrived as an int64 over the wire and a float64 in process.
func TestContentFnLooseInputKeepsSchemaTypes(t *testing.T) {
	type counted struct {
		Count int64  `json:"count"`
		Name  string `json:"name"`
	}

	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{name: "test/looseInput"})

	var seen []any
	p := DefinePrompt(r, "looseInput",
		WithModel(m),
		WithInputType(counted{Count: 7}),
		WithPromptFn(func(ctx context.Context, in map[string]any) (string, error) {
			seen = append(seen, in["count"])
			return "hi", nil
		}),
	)

	ctx := context.Background()
	const want = int64(1230000000)

	t.Run("typed Go value", func(t *testing.T) {
		seen = nil
		if _, err := p.Execute(ctx, WithInput(counted{Count: want})); err != nil {
			t.Fatal(err)
		}
		if seen[0] != any(want) {
			t.Errorf("count = %T(%v), want int64(%d)", seen[0], seen[0], want)
		}
	})

	t.Run("reflection API", func(t *testing.T) {
		seen = nil
		act := r.ResolveAction("/executable-prompt/looseInput")
		if act == nil {
			t.Fatal("prompt action not registered")
		}
		if _, err := act.RunJSON(ctx, json.RawMessage(`{"count":1230000000,"name":"a"}`), nil); err != nil {
			t.Fatal(err)
		}
		if seen[0] != any(want) {
			t.Errorf("count = %T(%v), want int64(%d)", seen[0], seen[0], want)
		}
	})

	t.Run("default input from WithInputType", func(t *testing.T) {
		seen = nil
		if _, err := p.Execute(ctx); err != nil {
			t.Fatal(err)
		}
		if seen[0] != any(int64(7)) {
			t.Errorf("count = %T(%v), want int64(7)", seen[0], seen[0])
		}
	})
}

// TestUntypedContentFnStillCompiles pins the source compatibility of the
// generic retrofit: functions written against the old untyped signature still
// satisfy the options, with In inferred as any.
func TestUntypedContentFnStillCompiles(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{name: "test/echoUntyped"})

	var systemFn PromptFn = func(ctx context.Context, input any) (string, error) {
		return "system", nil
	}
	var messagesFn MessagesFn = func(ctx context.Context, input any) ([]*Message, error) {
		return []*Message{NewModelTextMessage("history")}, nil
	}

	p := DefinePrompt(r, "untypedFns",
		WithModel(m),
		WithInputType(fnTestInput{}),
		WithSystemFn(systemFn),
		WithMessagesFn(messagesFn),
		WithPromptFn(func(ctx context.Context, input any) (string, error) {
			// The untyped path still receives the raw value as before.
			if _, ok := input.(fnTestInput); !ok {
				t.Errorf("input type = %T, want ai.fnTestInput", input)
			}
			return "prompt", nil
		}),
	)

	if _, err := p.Execute(context.Background(), WithInput(fnTestInput{Name: "bob"})); err != nil {
		t.Fatal(err)
	}
}

// TestContentFnResultIsNotATemplate covers the behavior change: a function has
// already produced its final content, so the result must not be reinterpreted
// as a dotprompt template. Previously text with literal braces failed to parse
// and text resembling a placeholder was silently substituted.
func TestContentFnResultIsNotATemplate(t *testing.T) {
	literal := "how do I write {{#if x}} in handlebars? Also {{name}} and a bare {{"

	t.Run("prompt fn", func(t *testing.T) {
		_, p, get := capturePrompt(t, "verbatimPrompt",
			WithInputType(fnTestInput{}),
			WithPromptFn(func(ctx context.Context, in fnTestInput) (string, error) {
				return literal, nil
			}),
		)
		if _, err := p.Execute(context.Background(), WithInput(fnTestInput{Name: "bob"})); err != nil {
			t.Fatalf("verbatim text should not be parsed as a template: %v", err)
		}
		req := (*get)()
		if got := req.Messages[len(req.Messages)-1].Text(); got != literal {
			t.Errorf("user text = %q, want %q", got, literal)
		}
	})

	t.Run("system fn", func(t *testing.T) {
		_, p, get := capturePrompt(t, "verbatimSystem",
			WithInputType(fnTestInput{}),
			WithSystemFn(func(ctx context.Context, in fnTestInput) (string, error) {
				return literal, nil
			}),
		)
		if _, err := p.Execute(context.Background(), WithInput(fnTestInput{Name: "bob"})); err != nil {
			t.Fatalf("verbatim text should not be parsed as a template: %v", err)
		}
		req := (*get)()
		if req.Messages[0].Role != RoleSystem {
			t.Errorf("Messages[0].Role = %q, want %q", req.Messages[0].Role, RoleSystem)
		}
		if got := req.Messages[0].Text(); got != literal {
			t.Errorf("system text = %q, want %q", got, literal)
		}
	})

	t.Run("static messages", func(t *testing.T) {
		_, p, get := capturePrompt(t, "verbatimStatic",
			WithInputType(fnTestInput{}),
			WithMessages(NewUserTextMessage(literal)),
			WithPrompt("ok"),
		)
		if _, err := p.Execute(context.Background(), WithInput(fnTestInput{Name: "bob"})); err != nil {
			t.Fatalf("verbatim history should not be parsed as a template: %v", err)
		}
		req := (*get)()
		if got := req.Messages[0].Text(); got != literal {
			t.Errorf("message text = %q, want %q", got, literal)
		}
	})

	t.Run("messages fn", func(t *testing.T) {
		_, p, get := capturePrompt(t, "verbatimMessages",
			WithInputType(fnTestInput{}),
			WithMessagesFn(func(ctx context.Context, in fnTestInput) ([]*Message, error) {
				return []*Message{NewUserTextMessage(literal)}, nil
			}),
		)
		if _, err := p.Execute(context.Background(), WithInput(fnTestInput{Name: "bob"})); err != nil {
			t.Fatalf("verbatim history should not be parsed as a template: %v", err)
		}
		req := (*get)()
		if got := req.Messages[0].Text(); got != literal {
			t.Errorf("message text = %q, want %q", got, literal)
		}
	})

	t.Run("WithPrompt template text is still rendered", func(t *testing.T) {
		_, p, get := capturePrompt(t, "stillTemplated",
			WithInputType(fnTestInput{}),
			WithPrompt("talk like a {{theme}}"),
		)
		if _, err := p.Execute(context.Background(), WithInput(fnTestInput{Theme: "ninja"})); err != nil {
			t.Fatal(err)
		}
		req := (*get)()
		if got, want := req.Messages[0].Text(), "talk like a ninja"; got != want {
			t.Errorf("user text = %q, want %q", got, want)
		}
	})
}

// TestHistoryStaysOutOfGenerationContext guards the scoping of the history
// context: Execute attaches history to a context used only for Render, so the
// model, tools, and any prompt nested under the generate loop must never see
// it. Before the fix, a prompt executed from inside a tool inherited the outer
// prompt's entire conversation.
func TestHistoryStaysOutOfGenerationContext(t *testing.T) {
	r := newTestRegistry(t)
	var leaked []*Message
	m := defineFakeModel(t, r, fakeModelConfig{
		name: "test/leakModel",
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			leaked = HistoryFromContext(ctx)
			return &ModelResponse{Message: NewModelTextMessage("ok")}, nil
		},
	})
	p := DefinePrompt(r, "leakProbe", WithModel(m), WithPrompt("q"))
	if _, err := p.Execute(context.Background(), WithMessages(NewUserTextMessage("outer history"))); err != nil {
		t.Fatal(err)
	}
	if leaked != nil {
		t.Errorf("history leaked into the generation context: %v", leaked)
	}
}

// TestCallerMessagesAreNotMutated collects every angle on one invariant: a
// message the caller owns is never written to by the framework. Messages reach
// generation from four places and each has its own way of going wrong, so they
// are one test rather than four unrelated ones with near-identical names.
//
// Callers reuse message slices across turns, so an in-place append would leak
// output instructions or a metadata stamp into stored history.
func TestCallerMessagesAreNotMutated(t *testing.T) {
	t.Run("from a content function", messagesFromFnAreNotMutated)
	t.Run("declared with WithMessages, stamped by the model", declaredMessagesSurviveMetadataStamping)
	t.Run("declared with WithMessages, appended downstream", declaredMessagesAreCopiedPerExecution)
	t.Run("passed straight to GenerateWithRequest", callerMessagesSurviveInstructionInjection)
}

// TestFnMessagesNotMutatedByInstructionInjection guards the uniform cloning in
// renderMessages: messages returned by WithMessagesFn typically alias a
// session or the caller's own slice, and a later stage that appends to its
// target message in place would corrupt that caller-owned history.
func messagesFromFnAreNotMutated(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{
		name: "test/fnMutModel",
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{Message: NewModelTextMessage(`{"name":"x","theme":"y"}`)}, nil
		},
	})

	t.Run("prompt-level MessagesFn", func(t *testing.T) {
		shared := NewUserTextMessage("session-owned")
		p := DefinePrompt(r, "fnMutPrompt",
			WithModel(m),
			WithMessagesFn(func(ctx context.Context, _ any) ([]*Message, error) {
				return []*Message{shared}, nil
			}),
			WithOutputType(fnTestInput{}),
			WithCustomConstrainedOutput(),
		)
		if _, err := p.Execute(context.Background()); err != nil {
			t.Fatal(err)
		}
		if len(shared.Content) != 1 {
			t.Errorf("caller-owned message mutated: len(Content) = %d, want 1", len(shared.Content))
		}
	})

	t.Run("execute-time MessagesFn", func(t *testing.T) {
		shared := NewUserTextMessage("session-owned")
		p := DefinePrompt(r, "fnMutExec",
			WithModel(m),
			WithOutputType(fnTestInput{}),
			WithCustomConstrainedOutput(),
		)
		if _, err := p.Execute(context.Background(), WithMessagesFn(func(ctx context.Context, _ any) ([]*Message, error) {
			return []*Message{shared}, nil
		})); err != nil {
			t.Fatal(err)
		}
		if len(shared.Content) != 1 {
			t.Errorf("caller-owned message mutated: len(Content) = %d, want 1", len(shared.Content))
		}
	})
}

// TestHistoryPlacementByMessagesForm covers who owns the messages passed to
// Execute. A prompt that declares its own conversation places them, since only
// it knows where its few-shot examples end; a prompt that declares none gets
// them in the middle.
func TestHistoryPlacementByMessagesForm(t *testing.T) {
	history := []*Message{NewUserTextMessage("history in"), NewModelTextMessage("history out")}

	t.Run("no declared messages puts history in the middle", func(t *testing.T) {
		_, p, get := capturePrompt(t, "historyMiddle",
			WithSystem("persona"),
			WithPrompt("now answer"),
		)
		if _, err := p.Execute(context.Background(), WithMessages(history...)); err != nil {
			t.Fatal(err)
		}
		assertMessages(t, (*get)(), "system:persona", "user:history in", "model:history out", "user:now answer")
	})

	t.Run("static messages own the conversation", func(t *testing.T) {
		_, p, get := capturePrompt(t, "staticOwns",
			WithMessages(NewUserTextMessage("few-shot in"), NewModelTextMessage("few-shot out")),
			WithPrompt("now answer"),
		)
		if _, err := p.Execute(context.Background(), WithMessages(history...)); err != nil {
			t.Fatal(err)
		}
		assertMessages(t, (*get)(), "user:few-shot in", "model:few-shot out", "user:now answer")
	})

	// Without an explicit {{history}} dotprompt places the conversation before
	// the template's final user message, which is where a caller expects it.
	t.Run("template places history implicitly", func(t *testing.T) {
		_, p, get := capturePrompt(t, "templateImplicit",
			WithMessagesTemplate(`{{role "system"}}persona
{{role "user"}}few-shot in
{{role "model"}}few-shot out
{{role "user"}}now answer`),
		)
		if _, err := p.Execute(context.Background(), WithMessages(history...)); err != nil {
			t.Fatal(err)
		}
		assertMessages(t, (*get)(), "system:persona", "user:few-shot in", "model:few-shot out",
			"user:history in", "model:history out", "user:now answer")
	})

	t.Run("template places history explicitly", func(t *testing.T) {
		_, p, get := capturePrompt(t, "templateExplicit",
			WithMessagesTemplate(`{{role "system"}}persona
{{history}}
{{role "user"}}few-shot in`),
		)
		if _, err := p.Execute(context.Background(), WithMessages(history...)); err != nil {
			t.Fatal(err)
		}
		assertMessages(t, (*get)(), "system:persona", "user:history in", "model:history out", "user:few-shot in")
	})
}

// TestMessagesTemplateRendersInput covers the case the .prompt loader and the
// menu sample rely on: a multi-turn template whose turns are built from the
// prompt's input, which is the whole reason the conversation slot is compiled.
func TestMessagesTemplateRendersInput(t *testing.T) {
	type item struct {
		Title string `json:"title"`
	}
	type menuInput struct {
		MenuData []item `json:"menuData"`
	}

	_, p, get := capturePrompt(t, "menuPreamble",
		WithInputType(menuInput{}),
		WithMessagesTemplate(`{{role "user"}}What's on the menu?
{{role "model"}}Today:{{#each menuData}} {{this.title}}{{/each}}`),
	)
	if _, err := p.Execute(context.Background(), WithInput(menuInput{MenuData: []item{{Title: "Soup"}, {Title: "Pie"}}})); err != nil {
		t.Fatal(err)
	}
	assertMessages(t, (*get)(), "user:What's on the menu?", "model:Today: Soup Pie")
}

// TestMessagesTemplateKeepsNonTextHistory covers the fidelity of the history
// round trip: dotprompt decides where the messages go, but the originals are
// restored, so kinds it cannot represent survive.
func TestMessagesTemplateKeepsNonTextHistory(t *testing.T) {
	toolReq := &Message{Role: RoleModel, Content: []*Part{
		NewToolRequestPart(&ToolRequest{Name: "lookup", Input: map[string]any{"q": "x"}}),
	}}

	_, p, get := capturePrompt(t, "toolHistory",
		WithMessagesTemplate(`{{role "system"}}persona
{{role "user"}}now answer`),
	)
	if _, err := p.Execute(context.Background(), WithMessages(toolReq)); err != nil {
		t.Fatal(err)
	}
	req := (*get)()
	for _, msg := range req.Messages {
		for _, part := range msg.Content {
			if part == nil {
				t.Fatalf("history round trip produced a nil part: %v", req.Messages)
			}
			if part.ToolRequest != nil && part.ToolRequest.Name == "lookup" {
				return
			}
		}
	}
	t.Errorf("tool request did not survive the history round trip: %v", req.Messages)
}

// TestSingleMessageSlotsRejectRoleBlocks pins the slot contract: WithSystem and
// WithPrompt each fill one message whose role is the slot's, so a {{role}}
// marker is refused rather than silently overridden. A marker that produces one
// message is refused too: the role would be stamped back to the slot's and the
// author's intent dropped without a word.
func TestSingleMessageSlotsRejectRoleBlocks(t *testing.T) {
	for _, tc := range []struct {
		name   string
		option string
		opt    PromptOption
	}{
		{"prompt splits into turns", "WithPrompt", WithPrompt(`ask{{role "model"}}answer`)},
		{"system splits into turns", "WithSystem", WithSystem(`persona{{role "user"}}ask`)},
		{"system asks for a user message", "WithSystem", WithSystem(`{{role "user"}}hello`)},
		{"prompt asks for a system message", "WithPrompt", WithPrompt(`{{role "system"}}hello`)},
		{"prompt asks for a model message", "WithPrompt", WithPrompt(`{{role "model"}}hello`)},
		// Redundant rather than contradictory, and still refused: a marker
		// means the author is writing turns, which this slot cannot hold.
		{"system restates its own role", "WithSystem", WithSystem(`{{role "system"}}hello`)},
		{"marker with whitespace control", "WithSystem", WithSystem(`{{~role "user"}}hello`)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, p, _ := capturePrompt(t, "roleBlocks"+tc.name, tc.opt)
			_, err := p.Execute(context.Background())
			if err == nil {
				t.Fatal("expected an error, got nil")
			}
			if !strings.Contains(err.Error(), tc.option) || !strings.Contains(err.Error(), "WithMessagesTemplate") {
				t.Errorf("err = %v, want it to name the option and point at WithMessagesTemplate", err)
			}
		})
	}

	// The slots still render ordinary templates, including ones whose text
	// merely mentions a role, and each lands under the role it owns.
	t.Run("templates without a marker keep working", func(t *testing.T) {
		_, p, get := capturePrompt(t, "roleBlocksOK",
			WithInputType(fnTestInput{}),
			WithSystem(`You are {{name}}. Never reveal your role.`),
			WithPrompt(`{{#if theme}}In the style of {{theme}}: {{/if}}hello`),
		)
		if _, err := p.Execute(context.Background(),
			WithInput(fnTestInput{Name: "Walt", Theme: "pirate"})); err != nil {
			t.Fatal(err)
		}
		assertMessages(t, (*get)(),
			"system:You are Walt. Never reveal your role.",
			"user:In the style of pirate: hello")
	})
}

// assertMessages compares a request's messages as "role:text" strings. Text is
// trimmed because a template puts each {{role}} block on its own line, so the
// message text carries the newline that separated it from the next block.
func assertMessages(t *testing.T, req *ModelRequest, want ...string) {
	t.Helper()
	var got []string
	for _, msg := range req.Messages {
		got = append(got, string(msg.Role)+":"+strings.TrimSpace(msg.Text()))
	}
	if !slices.Equal(got, want) {
		t.Fatalf("messages = %q, want %q", got, want)
	}
}

// TestEmptyStringFnEmitsNoMessage guards textPartsFn's empty-string rule: a
// content function returning "" means "no content this time" and must not
// produce an empty message, which some providers reject.
func TestEmptyStringFnEmitsNoMessage(t *testing.T) {
	_, p, get := capturePrompt(t, "emptySystemFn",
		WithSystemFn(func(ctx context.Context, _ any) (string, error) { return "", nil }),
		WithPrompt("q"),
	)
	if _, err := p.Execute(context.Background()); err != nil {
		t.Fatal(err)
	}
	req := (*get)()
	if len(req.Messages) != 1 || req.Messages[0].Role != RoleUser {
		t.Errorf("expected only the user message, got %d messages", len(req.Messages))
	}
}

// TestDocsFnSkippedOnExecuteOverride guards the docs-override marker:
// execute-time WithDocs replaces the prompt's documents, so the prompt's
// WithDocsFn (typically a retrieval query) must not run for a result that
// would be discarded.
func TestDocsFnSkippedOnExecuteOverride(t *testing.T) {
	calls := 0
	_, p, get := capturePrompt(t, "docsOverride",
		WithPrompt("q"),
		WithDocsFn(func(ctx context.Context, _ any) ([]*Document, error) {
			calls++
			return []*Document{DocumentFromText("expensive retrieval", nil)}, nil
		}),
	)
	if _, err := p.Execute(context.Background(), WithDocs(DocumentFromText("override", nil))); err != nil {
		t.Fatal(err)
	}
	req := (*get)()
	if calls != 0 {
		t.Errorf("DocsFn ran %d times despite execute-time docs override", calls)
	}
	if len(req.Docs) != 1 || req.Docs[0].Content[0].Text != "override" {
		t.Errorf("request docs = %+v, want the override doc", req.Docs)
	}
}

// TestConversationMergeSemantics pins how the conversation options combine.
// Verbatim messages accumulate and the template is a last-win slot, the two
// standard rules. Combining the two kinds is refused: a template lays out the
// whole conversation, so nothing else has a position relative to it.
func TestConversationMergeSemantics(t *testing.T) {
	msg := func(text string) *Message { return NewUserTextMessage(text) }
	collect := func(fn MessagesFn) []string {
		t.Helper()
		if fn == nil {
			return nil
		}
		msgs, err := fn(context.Background(), nil)
		if err != nil {
			t.Fatalf("messages fn: %v", err)
		}
		var got []string
		for _, m := range msgs {
			got = append(got, m.Content[0].Text)
		}
		return got
	}
	apply := func(opts ...PromptOption) *promptOptions {
		t.Helper()
		p := &promptOptions{}
		for _, o := range opts {
			o.applyPrompt(p)
		}
		return p
	}

	t.Run("verbatim messages accumulate across both variants", func(t *testing.T) {
		p := apply(
			WithMessages(msg("one")),
			WithMessagesFn(func(context.Context, any) ([]*Message, error) {
				return []*Message{msg("two")}, nil
			}),
			WithMessages(msg("three")),
		)
		if got, want := collect(p.MessagesFn), []string{"one", "two", "three"}; !slices.Equal(got, want) {
			t.Errorf("messages = %v, want %v", got, want)
		}
	})

	t.Run("repeating the template takes the last one", func(t *testing.T) {
		p := apply(WithMessagesTemplate("first"), WithMessagesTemplate("second"))
		if p.MessagesText == nil || *p.MessagesText != "second" {
			t.Errorf("template = %v, want %q", p.MessagesText, "second")
		}
	})

	for _, tt := range []struct {
		name string
		opts []PromptOption
	}{
		{"template then messages", []PromptOption{
			WithMessagesTemplate("t"), WithMessages(msg("m"))}},
		{"messages then template", []PromptOption{
			WithMessages(msg("m")), WithMessagesTemplate("t")}},
		{"template then messages fn", []PromptOption{
			WithMessagesTemplate("t"),
			WithMessagesFn(func(context.Context, any) ([]*Message, error) { return nil, nil })}},
	} {
		t.Run("refused: "+tt.name, func(t *testing.T) {
			r := newTestRegistry(t)
			m := defineFakeModel(t, r, fakeModelConfig{name: "test/exclusive"})
			defer func() {
				rec := recover()
				if rec == nil {
					t.Fatal("expected a panic combining a template with verbatim messages")
				}
				text, _ := rec.(string)
				if !strings.Contains(text, "WithMessagesTemplate") || !strings.Contains(text, "WithMessages") {
					t.Errorf("panic = %v, want it to name both options", rec)
				}
			}()
			DefinePrompt(r, "exclusive_"+tt.name, append([]PromptOption{WithModel(m)}, tt.opts...)...)
		})
	}
}

// TestLiteralPercentSurvivesWithPrompt guards the Sprintf guard in
// WithSystem/WithPrompt: with no args the text is used as-is, so a literal %
// is not corrupted into %!x(MISSING) noise.
func TestLiteralPercentSurvivesWithPrompt(t *testing.T) {
	_, p, get := capturePrompt(t, "literalPercent",
		WithPrompt("Give me 50% off %s copy"),
	)
	if _, err := p.Execute(context.Background()); err != nil {
		t.Fatal(err)
	}
	req := (*get)()
	if got, want := req.Messages[0].Text(), "Give me 50% off %s copy"; got != want {
		t.Errorf("prompt text = %q, want %q", got, want)
	}
}

// TestStaticMessagesNotSharedAcrossExecutions guards the defensive cloning of
// prompt-declared messages. Messages given to WithMessages live on the prompt
// and are reused by every execution, so handing out the originals to a stage
// that appends to a message in place would let one execution alter the prompt
// for the next.
func declaredMessagesSurviveMetadataStamping(t *testing.T) {
	shared := NewUserTextMessage("stored on the prompt")
	shared.Metadata = map[string]any{"origin": "config"}

	r := newTestRegistry(t)
	var captured *ModelRequest
	m := defineFakeModel(t, r, fakeModelConfig{
		name: "test/sharedMessages",
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			captured = req
			// Stamp metadata the way middleware or the agent loop does; the
			// write must land on a clone, never on the prompt's stored map.
			for _, msg := range req.Messages {
				if msg.Metadata == nil {
					msg.Metadata = map[string]any{}
				}
				msg.Metadata["stamped"] = true
			}
			return &ModelResponse{Message: NewModelTextMessage(`{"name":"x","theme":"y"}`)}, nil
		},
	})

	// Custom constrained output makes the formatter inject output instructions
	// rather than relying on the model's native support, and with no system
	// message the target is the last user message: the shared one.
	p := DefinePrompt(r, "sharedMessages",
		WithModel(m),
		WithMessages(shared),
		WithOutputType(fnTestInput{}),
		WithCustomConstrainedOutput(),
	)

	for i := range 2 {
		if _, err := p.Execute(context.Background()); err != nil {
			t.Fatalf("execution %d: %v", i, err)
		}
		if got := len(shared.Content); got != 1 {
			t.Fatalf("execution %d mutated the prompt's stored message: len(Content) = %d, want 1", i, got)
		}
		if _, ok := shared.Metadata["stamped"]; ok {
			t.Fatalf("execution %d: metadata stamp leaked into the prompt's stored message: %v", i, shared.Metadata)
		}
		if got := len(captured.Messages[0].Content); got != 2 {
			t.Errorf("execution %d: len(request Content) = %d, want 2 (text + injected instructions)", i, got)
		}
	}
}

// TestPartsFnMultimodal covers content functions that return non-text parts,
// which previously required smuggling a {{media}} helper through the template.
func TestPartsFnMultimodal(t *testing.T) {
	r := newTestRegistry(t)
	var captured *ModelRequest
	m := defineFakeModel(t, r, fakeModelConfig{
		name: "test/mediaModel",
		supports: &ModelSupports{
			Media:      true,
			Multiturn:  true,
			SystemRole: true,
		},
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			captured = req
			return &ModelResponse{Message: NewModelTextMessage("ok")}, nil
		},
	})

	p := DefinePrompt(r, "partsFn",
		WithModel(m),
		WithInputType(fnTestInput{}),
		WithPromptPartsFn(func(ctx context.Context, in fnTestInput) ([]*Part, error) {
			return []*Part{
				NewTextPart("describe this for " + in.Name),
				NewMediaPart("image/png", "https://example.com/a.png"),
			}, nil
		}),
	)

	if _, err := p.Execute(context.Background(), WithInput(fnTestInput{Name: "bob"})); err != nil {
		t.Fatal(err)
	}

	content := captured.Messages[len(captured.Messages)-1].Content
	if len(content) != 2 {
		t.Fatalf("len(content) = %d, want 2", len(content))
	}
	if got, want := content[0].Text, "describe this for bob"; got != want {
		t.Errorf("content[0].Text = %q, want %q", got, want)
	}
	if !content[1].IsMedia() {
		t.Errorf("content[1] is not media: %+v", content[1])
	}
}

// TestWithDocsFn covers input-dependent document selection at prompt definition
// time, the hook for retrieval that depends on the input.
func TestWithDocsFn(t *testing.T) {
	r := newTestRegistry(t)
	var captured *ModelRequest
	m := defineFakeModel(t, r, fakeModelConfig{
		name: "test/docsFnModel",
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			captured = req
			return &ModelResponse{Message: NewModelTextMessage("ok")}, nil
		},
	})

	t.Run("resolved from input", func(t *testing.T) {
		p := DefinePrompt(r, "docsFn",
			WithModel(m),
			WithInputType(fnTestInput{}),
			WithPrompt("question"),
			WithDocsFn(func(ctx context.Context, in fnTestInput) ([]*Document, error) {
				return []*Document{DocumentFromText("doc for "+in.Name, nil)}, nil
			}),
		)
		if _, err := p.Execute(context.Background(), WithInput(fnTestInput{Name: "bob"})); err != nil {
			t.Fatal(err)
		}
		if len(captured.Docs) != 1 {
			t.Fatalf("len(Docs) = %d, want 1", len(captured.Docs))
		}
		if got, want := captured.Docs[0].Content[0].Text, "doc for bob"; got != want {
			t.Errorf("doc text = %q, want %q", got, want)
		}
	})

	t.Run("static docs at definition time", func(t *testing.T) {
		p := DefinePrompt(r, "staticDocs",
			WithModel(m),
			WithPrompt("question"),
			WithDocs(DocumentFromText("static doc", nil)),
		)
		if _, err := p.Execute(context.Background()); err != nil {
			t.Fatal(err)
		}
		if len(captured.Docs) != 1 {
			t.Fatalf("len(Docs) = %d, want 1", len(captured.Docs))
		}
		if got, want := captured.Docs[0].Content[0].Text, "static doc"; got != want {
			t.Errorf("doc text = %q, want %q", got, want)
		}
	})

	t.Run("error propagates", func(t *testing.T) {
		p := DefinePrompt(r, "docsFnErr",
			WithModel(m),
			WithPrompt("question"),
			WithDocsFn(func(ctx context.Context, in fnTestInput) ([]*Document, error) {
				return nil, errTestDocs
			}),
		)
		_, err := p.Execute(context.Background())
		if err == nil || !strings.Contains(err.Error(), "resolving docs") {
			t.Errorf("err = %v, want it to mention resolving docs", err)
		}
	})
}

var errTestDocs = errTest("docs unavailable")

type errTest string

func (e errTest) Error() string { return string(e) }

// TestHistoryFromContext covers the history handoff: a prompt that declares its
// own messages owns the caller's history rather than having it appended behind
// the function's back.
func TestHistoryFromContext(t *testing.T) {
	t.Run("MessagesFn reads and reorders history", func(t *testing.T) {
		_, p, get := capturePrompt(t, "historyFn",
			WithSystem("sys"),
			WithPrompt("now answer"),
			WithMessagesFn(func(ctx context.Context, _ any) ([]*Message, error) {
				history := HistoryFromContext(ctx)
				// Prove the function can do what plain appending cannot:
				// collapse the history into a single summary message.
				summary := make([]string, 0, len(history))
				for _, m := range history {
					summary = append(summary, m.Text())
				}
				return []*Message{NewModelTextMessage("summary: " + strings.Join(summary, "|"))}, nil
			}),
		)

		if _, err := p.Execute(context.Background(), WithMessages(
			NewUserTextMessage("first"),
			NewModelTextMessage("second"),
		)); err != nil {
			t.Fatal(err)
		}

		req := (*get)()
		if len(req.Messages) != 3 {
			t.Fatalf("len(Messages) = %d, want 3 (system, summary, user)", len(req.Messages))
		}
		if got, want := req.Messages[1].Text(), "summary: first|second"; got != want {
			t.Errorf("Messages[1] = %q, want %q", got, want)
		}
	})

	t.Run("no history attached outside Execute", func(t *testing.T) {
		if got := HistoryFromContext(context.Background()); got != nil {
			t.Errorf("HistoryFromContext = %v, want nil", got)
		}
	})

	t.Run("execute-time messages used when prompt declares none", func(t *testing.T) {
		_, p, get := capturePrompt(t, "historyFallback",
			WithSystem("sys"),
			WithPrompt("now answer"),
		)
		if _, err := p.Execute(context.Background(), WithMessages(NewModelTextMessage("prior"))); err != nil {
			t.Fatal(err)
		}
		req := (*get)()
		if len(req.Messages) != 3 {
			t.Fatalf("len(Messages) = %d, want 3", len(req.Messages))
		}
		if got, want := req.Messages[1].Text(), "prior"; got != want {
			t.Errorf("Messages[1] = %q, want %q", got, want)
		}
	})

	// A prompt that declares static messages owns the conversation: its
	// examples are not a place to splice a caller's history into, and the
	// forms that can place it are the template and the function.
	t.Run("static prompt messages own the conversation", func(t *testing.T) {
		_, p, get := capturePrompt(t, "historyPlusStatic",
			WithSystem("sys"),
			WithMessages(NewUserTextMessage("example in"), NewModelTextMessage("example out")),
			WithPrompt("now answer"),
		)
		if _, err := p.Execute(context.Background(), WithMessages(NewModelTextMessage("prior"))); err != nil {
			t.Fatal(err)
		}
		assertMessages(t, (*get)(), "system:sys", "user:example in", "model:example out", "user:now answer")
	})
}

// TestDocsAccumulateAcrossVariants pins that the three document options add to
// one set rather than replacing each other, and that the fixed documents come
// before the computed ones. Resolution used to overwrite the fixed documents
// with the function's result, so declaring both silently lost the fixed ones.
func TestDocsAccumulateAcrossVariants(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{name: "test/docsAccum"})

	p := DefinePrompt(r, "docsAccum",
		WithModel(m),
		WithPrompt("hi"),
		WithTextDocs("fixed one"),
		WithDocsFn(func(ctx context.Context, _ any) ([]*Document, error) {
			return []*Document{DocumentFromText("computed", nil)}, nil
		}),
		WithTextDocs("fixed two"),
	)

	opts, err := p.Render(context.Background(), nil)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	var got []string
	for _, d := range opts.Docs {
		got = append(got, d.Content[0].Text)
	}
	want := []string{"fixed one", "fixed two", "computed"}
	if !slices.Equal(got, want) {
		t.Errorf("docs = %v, want %v", got, want)
	}
}

// TestContentFnErrorPropagates makes sure a coercion failure surfaces as an
// error rather than a panic, which is what the old type assertions produced.
func TestContentFnErrorPropagates(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{name: "test/coerceErr"})

	type numeric struct {
		Count int `json:"name"` // deliberately mistyped against the input
	}

	p := DefinePrompt(r, "coerceErr",
		WithModel(m),
		WithInputSchema(map[string]any{
			"type":       "object",
			"properties": map[string]any{"name": map[string]any{"type": "string"}},
		}),
		WithPromptFn(func(ctx context.Context, in numeric) (string, error) {
			return "unreachable", nil
		}),
	)

	_, err := p.Execute(context.Background(), WithInput(map[string]any{"name": "bob"}))
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	// The classification is the contract: a caller branches on the sentinel,
	// not on the wording. Matching the base sentinel too pins that the failure
	// is the caller's input rather than an internal fault.
	if !errors.Is(err, ErrInputTypeMismatch) {
		t.Errorf("err = %v, want it to match ErrInputTypeMismatch", err)
	}
	if !errors.Is(err, status.ErrInvalidArgument) {
		t.Errorf("err = %v, want it to match status.ErrInvalidArgument", err)
	}
	if got := status.Of(err); got != status.InvalidArgument {
		t.Errorf("status = %v, want %v", got, status.InvalidArgument)
	}
	if !strings.Contains(err.Error(), "WithPromptFn input") {
		t.Errorf("err = %v, want it to name the option that rejected the input", err)
	}
}

// TestContentFnErrorNamesItsOption covers a prompt filling several slots from
// functions: the input can satisfy one and not another, so the error has to say
// which one it was rather than reporting an anonymous content function.
func TestContentFnErrorNamesItsOption(t *testing.T) {
	type ticket struct {
		Question string `json:"question"`
	}
	type query struct {
		Terms []string `json:"terms"`
	}

	tests := []struct {
		name string
		opt  PromptOption
		want string
	}{
		{"docs", WithDocsFn(func(ctx context.Context, q query) ([]*Document, error) {
			return nil, nil
		}), "WithDocsFn input"},
		{"system", WithSystemFn(func(ctx context.Context, q query) (string, error) {
			return "", nil
		}), "WithSystemFn input"},
		{"prompt parts", WithPromptPartsFn(func(ctx context.Context, q query) ([]*Part, error) {
			return nil, nil
		}), "WithPromptPartsFn input"},
		{"messages", WithMessagesFn(func(ctx context.Context, q query) ([]*Message, error) {
			return nil, nil
		}), "WithMessagesFn input"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newTestRegistry(t)
			m := defineFakeModel(t, r, fakeModelConfig{name: "test/named_" + tt.name})
			// The user prompt takes the input the caller actually supplies, so
			// only the slot under test can be the one that fails.
			p := DefinePrompt(r, "named_"+tt.name, WithModel(m), WithInputType(ticket{}),
				WithPromptFn(func(ctx context.Context, tk ticket) (string, error) {
					return tk.Question, nil
				}),
				tt.opt)

			_, err := p.Execute(context.Background(), WithInput(ticket{Question: "why"}))
			if err == nil {
				t.Fatal("expected an error, got nil")
			}
			if !errors.Is(err, ErrInputTypeMismatch) {
				t.Errorf("err = %v, want it to match ErrInputTypeMismatch", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

// TestContentFnUnrelatedStructIsClassified covers the other way the conversion
// fails: a struct that cannot be reinterpreted as the function's struct. A JSON
// round-trip between the two would succeed while leaving every field zero, so
// this is refused rather than silently handing the function a blank value. It
// classifies as the same sentinel as a decode failure, so a caller branching on
// bad input catches both without knowing which one it hit.
func TestContentFnUnrelatedStructIsClassified(t *testing.T) {
	r := newTestRegistry(t)
	m := defineFakeModel(t, r, fakeModelConfig{name: "test/unrelatedStruct"})

	type declared struct {
		Theme string `json:"theme"`
	}
	type supplied struct {
		Topic string `json:"topic"`
	}

	p := DefinePrompt(r, "unrelatedStruct",
		WithModel(m),
		WithPromptFn(func(ctx context.Context, in declared) (string, error) {
			return "unreachable", nil
		}),
	)

	_, err := p.Execute(context.Background(), WithInput(supplied{Topic: "pirates"}))
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	if !errors.Is(err, ErrInputTypeMismatch) {
		t.Errorf("err = %v, want it to match ErrInputTypeMismatch", err)
	}
	if got := status.Of(err); got != status.InvalidArgument {
		t.Errorf("status = %v, want %v", got, status.InvalidArgument)
	}
}

// findMessageText returns the message whose first text part equals want.
func findMessageText(t *testing.T, msgs []*Message, want string) *Message {
	t.Helper()
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.IsText() && p.Text == want {
				return m
			}
		}
	}
	t.Fatalf("no message with text %q in %v", want, msgs)
	return nil
}

// TestStaticMessagesAreVerbatim covers the divergence that made conversation
// history unusable: text given to [WithMessages] used to be compiled as a
// dotprompt template, so a user asking about handlebars turned into a parse
// error rather than a request. Only the string form of the conversation is
// compiled, and the Generate path never templated messages either, so the
// prompt path was alone in doing it.
func TestStaticMessagesAreVerbatim(t *testing.T) {
	const handlebars = "how do I write {{#if x}} in handlebars?"

	t.Run("declared on the prompt", func(t *testing.T) {
		_, p, get := capturePrompt(t, "staticVerbatimDef",
			WithMessages(NewUserTextMessage(handlebars)),
			WithPrompt("hi"),
		)
		if _, err := p.Execute(context.Background()); err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		findMessageText(t, (*get)().Messages, handlebars)
	})

	t.Run("supplied at execution time", func(t *testing.T) {
		_, p, get := capturePrompt(t, "staticVerbatimExec", WithPrompt("hi"))
		if _, err := p.Execute(context.Background(),
			WithMessages(NewUserTextMessage(handlebars)),
		); err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		findMessageText(t, (*get)().Messages, handlebars)
	})

	// A prompt's own text is still a template, so one request can carry an
	// expanded prompt alongside an unexpanded message.
	t.Run("prompt text still renders", func(t *testing.T) {
		_, p, get := capturePrompt(t, "staticVerbatimMixed",
			WithInputType(fnTestInput{}),
			WithMessages(NewUserTextMessage("literal {{name}}")),
			WithPrompt("rendered {{name}}"),
		)
		if _, err := p.Execute(context.Background(),
			WithInput(fnTestInput{Name: "bob"}),
		); err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		msgs := (*get)().Messages
		findMessageText(t, msgs, "literal {{name}}")
		findMessageText(t, msgs, "rendered bob")
	})
}

// TestStaticMessagesCopiedPerExecution covers why declared messages are copied
// rather than forwarded by reference: a prompt's declared messages are shared
// by every execution of it, so one execution's downstream mutation would
// otherwise be visible to the next.
func declaredMessagesAreCopiedPerExecution(t *testing.T) {
	shared := NewUserTextMessage("history")
	_, p, get := capturePrompt(t, "staticIsolation",
		WithMessages(shared),
		WithPrompt("hi"),
	)

	for i := 0; i < 2; i++ {
		if _, err := p.Execute(context.Background()); err != nil {
			t.Fatalf("Execute() %d error = %v", i, err)
		}

		got := findMessageText(t, (*get)().Messages, "history")
		if got == shared {
			t.Fatal("request reused the declared *Message; later stages mutate messages in place")
		}

		// Stand in for a downstream stage that appends to its target.
		got.Content = append(got.Content, NewTextPart("injected"))

		if len(shared.Content) != 1 {
			t.Fatalf("declared message grew to %d parts after execution %d", len(shared.Content), i)
		}
	}
}

// TestStaticPartsOptions covers the value forms of the multi-part content
// options, for prompts whose media is fixed rather than input-dependent.
func TestStaticPartsOptions(t *testing.T) {
	r := newTestRegistry(t)
	var captured *ModelRequest
	m := defineFakeModel(t, r, fakeModelConfig{
		name: "test/staticParts",
		supports: &ModelSupports{
			Media:      true,
			Multiturn:  true,
			SystemRole: true,
		},
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			captured = req
			return &ModelResponse{Message: NewModelTextMessage("ok")}, nil
		},
	})

	p := DefinePrompt(r, "staticParts",
		WithModel(m),
		WithSystemParts(NewTextPart("you are a vision assistant")),
		WithPromptParts(
			NewTextPart("what is in this image? {{not_a_template}}"),
			NewMediaPart("image/png", "https://example.com/a.png"),
		),
	)
	if _, err := p.Execute(context.Background()); err != nil {
		t.Fatal(err)
	}

	if got, want := len(captured.Messages), 2; got != want {
		t.Fatalf("len(Messages) = %d, want %d", got, want)
	}
	if captured.Messages[0].Role != RoleSystem || captured.Messages[0].Text() != "you are a vision assistant" {
		t.Errorf("system message = %+v", captured.Messages[0])
	}
	user := captured.Messages[1]
	if user.Role != RoleUser || len(user.Content) != 2 {
		t.Fatalf("user message = %+v", user)
	}
	// Parts are verbatim: the braces are not compiled as a template.
	if got, want := user.Content[0].Text, "what is in this image? {{not_a_template}}"; got != want {
		t.Errorf("user text = %q, want %q", got, want)
	}
	if !user.Content[1].IsMedia() {
		t.Errorf("Content[1] is not media: %+v", user.Content[1])
	}
}

// TestStaticPartsConflictWithText makes sure the value and template forms of
// the same slot are still mutually exclusive.
func TestStaticPartsConflictWithText(t *testing.T) {
	for _, tt := range []struct {
		name string
		opts []PromptingOption
	}{
		{"system", []PromptingOption{WithSystem("text"), WithSystemParts(NewTextPart("parts"))}},
		{"prompt", []PromptingOption{WithPrompt("text"), WithPromptParts(NewTextPart("parts"))}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			opts := &promptOptions{}
			for _, o := range tt.opts {
				o.applyPrompt(opts)
			}
			// The parts form arrived last, so it holds the slot and the text
			// form is gone: one message, one winner.
			gotText, gotFn := opts.SystemText, opts.SystemFn
			if tt.name == "prompt" {
				gotText, gotFn = opts.PromptText, opts.PromptFn
			}
			if gotText != nil {
				t.Errorf("text = %q, want it cleared by the later parts option", *gotText)
			}
			if gotFn == nil {
				t.Fatal("parts option did not fill the slot")
			}
			parts, err := gotFn(context.Background(), nil)
			if err != nil {
				t.Fatalf("parts fn: %v", err)
			}
			if len(parts) != 1 || parts[0].Text != "parts" {
				t.Errorf("parts = %+v, want one part %q", parts, "parts")
			}
		})
	}
}

// TestInstructionInjectionDoesNotMutateCallerMessages exercises the
// copy-on-write in injectInstructions end to end. The copy itself is unit
// tested in format_test.go; this runs it through the public entry point,
// GenerateWithRequest, which takes the caller's own message slice and is what
// made an in-place append reach a conversation the caller keeps.
func callerMessagesSurviveInstructionInjection(t *testing.T) {
	r := newTestRegistry(t)
	var captured *ModelRequest
	defineFakeModel(t, r, fakeModelConfig{
		name: "test/injectCOW",
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			captured = req
			return &ModelResponse{Message: NewModelTextMessage(`{"name":"x","theme":"y"}`)}, nil
		},
	})

	mine := NewUserTextMessage("my own message")
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"name": map[string]any{"type": "string"}},
	}

	for i := range 2 {
		_, err := GenerateWithRequest(context.Background(), r, &GenerateActionOptions{
			Model:    "test/injectCOW",
			Messages: []*Message{mine},
			Output: &GenerateActionOutputConfig{
				Format:      OutputFormatJSON,
				JsonSchema:  schema,
				Constrained: false,
			},
		}, nil, nil)
		if err != nil {
			t.Fatalf("run %d: %v", i, err)
		}
		if got := len(mine.Content); got != 1 {
			t.Fatalf("run %d mutated the caller's message: len(Content) = %d, want 1", i, got)
		}
		// Every run must still get its own instructions, which the bail-out
		// would have skipped had the previous run mutated the original.
		if got := len(captured.Messages[0].Content); got != 2 {
			t.Errorf("run %d: request Content = %d parts, want 2 (text + instructions)", i, got)
		}
	}
}

// TestNilMessagesAreDropped covers a nil slipping into a conversation, which
// reaches the framework from a caller-owned slice or a user-written function
// and used to have no safe path through it: the template renderer dereferenced
// it, and every other path carried it as far as the action's output schema,
// which rejects a null message.
func TestNilMessagesAreDropped(t *testing.T) {
	// The conversation is placed by a template, so it round-trips through
	// dotprompt as a placeholder and back. Dropping nils anywhere but on the
	// way in would shift that positional restore.
	t.Run("template", func(t *testing.T) {
		_, p, get := capturePrompt(t, "nil_tpl",
			WithMessagesTemplate(`{{role "system"}}sys{{history}}{{role "user"}}end`))
		ctx := NewHistoryContext(context.Background(), []*Message{
			NewUserTextMessage("A"), nil, NewUserTextMessage("B"),
		})
		if _, err := p.Execute(ctx); err != nil {
			t.Fatalf("Execute: %v", err)
		}
		assertNoNilMessages(t, (*get)().Messages)
		assertUserTexts(t, (*get)().Messages, "A", "B", "end")
	})

	// No conversation of its own, so the caller's is used directly.
	t.Run("placed for the prompt", func(t *testing.T) {
		_, p, get := capturePrompt(t, "nil_plain", WithPrompt("q"))
		if _, err := p.Execute(context.Background(), WithMessages(
			NewUserTextMessage("A"), nil, NewUserTextMessage("B"),
		)); err != nil {
			t.Fatalf("Execute: %v", err)
		}
		assertNoNilMessages(t, (*get)().Messages)
		assertUserTexts(t, (*get)().Messages, "A", "B", "q")
	})

	// The prompt's own conversation, returned by a user's function.
	t.Run("from a content function", func(t *testing.T) {
		_, p, get := capturePrompt(t, "nil_fn",
			WithMessagesFn(func(context.Context, any) ([]*Message, error) {
				return []*Message{NewUserTextMessage("A"), nil, NewUserTextMessage("B")}, nil
			}))
		if _, err := p.Execute(context.Background()); err != nil {
			t.Fatalf("Execute: %v", err)
		}
		assertNoNilMessages(t, (*get)().Messages)
		assertUserTexts(t, (*get)().Messages, "A", "B")
	})

	// A prompt's own function reads the conversation back, so it must see the
	// same filtered slice the renderer does.
	t.Run("read back by HistoryFromContext", func(t *testing.T) {
		var seen []*Message
		_, p, _ := capturePrompt(t, "nil_readback",
			WithMessagesFn(func(ctx context.Context, _ any) ([]*Message, error) {
				seen = HistoryFromContext(ctx)
				return seen, nil
			}))
		if _, err := p.Execute(context.Background(), WithMessages(
			NewUserTextMessage("A"), nil, NewUserTextMessage("B"),
		)); err != nil {
			t.Fatalf("Execute: %v", err)
		}
		assertNoNilMessages(t, seen)
	})
}

// TestCompactMessagesKeepsSliceWhenClean pins the allocation-free path: a
// conversation without nils is passed through, not copied.
func TestCompactMessagesKeepsSliceWhenClean(t *testing.T) {
	src := []*Message{NewUserTextMessage("A"), NewUserTextMessage("B")}
	if got := compactMessages(src); &got[0] != &src[0] {
		t.Error("compactMessages copied a slice that had no nils")
	}
	if got := compactMessages(nil); got != nil {
		t.Errorf("compactMessages(nil) = %v, want nil", got)
	}
}

func assertNoNilMessages(t *testing.T, msgs []*Message) {
	t.Helper()
	for i, m := range msgs {
		if m == nil {
			t.Fatalf("messages[%d] is nil", i)
		}
	}
}

// assertUserTexts checks the user turns in order, which is what carries the
// conversation in these cases.
func assertUserTexts(t *testing.T, msgs []*Message, want ...string) {
	t.Helper()
	var got []string
	for _, m := range msgs {
		if m.Role == RoleUser {
			got = append(got, m.Text())
		}
	}
	if !slices.Equal(got, want) {
		t.Errorf("user turns = %v, want %v", got, want)
	}
}
