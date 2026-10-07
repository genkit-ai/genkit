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

package livetest

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
)

// capitalFacts is the structured output target; asking about France keeps
// every assertion a stable substring check.
type capitalFacts struct {
	City    string `json:"city"`
	Country string `json:"country"`
}

// gablorkenAnswer is the typed output of the tools-then-output case.
type gablorkenAnswer struct {
	Result float64 `json:"result"`
}

// planet is the item type of the array and JSONL cases.
type planet struct {
	Name     string `json:"name"`
	Position int    `json:"position"`
	Fact     string `json:"fact"`
}

// transferPrompt asks for a transfer the tool pauses on for approval.
const transferPrompt = "Send 50 dollars to Ada with the transferFunds tool and tell me the confirmation code."

// planetsPrompt asks for eight items with a sentence each, long enough that a
// streamed list arrives over several chunks.
const planetsPrompt = "List the eight planets of the solar system in order from the sun, with each planet's name, its position (1 for Mercury), and one full sentence about it."

// gen runs a generation against model on the suite's Genkit instance.
func (r *runner) gen(t *testing.T, model ai.ModelArg, opts ...ai.GenerateOption) *ai.ModelResponse {
	t.Helper()
	resp, err := genkit.Generate(r.ctx, r.g, append([]ai.GenerateOption{ai.WithModel(model)}, opts...)...)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	return resp
}

// wantText fails t unless text contains every one of want, ignoring case.
func wantText(t *testing.T, text string, want ...string) {
	t.Helper()
	for _, w := range want {
		if !containsFold(text, w) {
			t.Errorf("text = %q, want it to contain %q", text, w)
		}
	}
}

// wantReply fails t unless resp's text contains every one of want, ignoring
// case, and says how the response ended when it does not.
func wantReply(t *testing.T, resp *ai.ModelResponse, want ...string) {
	t.Helper()
	for _, w := range want {
		if !containsFold(resp.Text(), w) {
			t.Errorf("Text() = %q (finish reason %q, %d tool calls in history), want it to contain %q",
				resp.Text(), resp.FinishReason, len(toolRequests(resp.History())), w)
		}
	}
}

// toolRequests returns every tool request in msgs.
func toolRequests(msgs []*ai.Message) []*ai.ToolRequest {
	var reqs []*ai.ToolRequest
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.IsToolRequest() {
				reqs = append(reqs, p.ToolRequest)
			}
		}
	}
	return reqs
}

func generateCases() []liveCase {
	return []liveCase{
		// Basics.
		{"generate", always, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithPrompt("What is the capital of France? Reply with just the city name."))
			wantReply(t, resp, "paris")
			if resp.FinishReason != ai.FinishReasonStop {
				t.Errorf("FinishReason = %q, want %q", resp.FinishReason, ai.FinishReasonStop)
			}
		}},
		{"system prompt", needSystemRole, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithSystem("You are named Quixotebot. When asked your name, reply with exactly that name."),
				ai.WithPrompt("What is your name?"))
			wantReply(t, resp, "quixotebot")
		}},
		{"history", needMultiturn, func(t *testing.T, r *runner) {
			first := r.gen(t, r.s.Model,
				ai.WithPrompt("My name is Zebulon Quixote. Greet me in one short sentence."))
			resp := r.gen(t, r.s.Model,
				ai.WithMessages(first.History()...),
				ai.WithPrompt("What is my name? Reply with just the name."))
			wantReply(t, resp, "zebulon")
		}},
		{"streaming", always, func(t *testing.T, r *runner) {
			var streamed strings.Builder
			chunks := 0
			resp := r.gen(t, r.s.Model,
				ai.WithPrompt("Write one short paragraph about the ocean."),
				ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
					chunks++
					streamed.WriteString(chunk.Text())
					return nil
				}))
			if chunks <= 1 {
				t.Errorf("chunks = %d, want the response in multiple chunks", chunks)
			}
			if streamed.String() != resp.Text() {
				t.Errorf("streamed text = %q, want the final text %q", streamed.String(), resp.Text())
			}
		}},
		// Streams only report usage when the request opts in, so the
		// streamed counts are the half that goes missing.
		{"usage reported", always, func(t *testing.T, r *runner) {
			for _, streaming := range []bool{false, true} {
				opts := []ai.GenerateOption{ai.WithPrompt("Name one primary color. Answer with the word alone.")}
				if streaming {
					opts = append(opts, ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error { return nil }))
				}
				u := r.gen(t, r.s.Model, opts...).Usage
				if u == nil || u.InputTokens == 0 || u.OutputTokens == 0 || u.TotalTokens == 0 {
					t.Errorf("Usage (streaming %v) = %+v, want input, output and total token counts", streaming, u)
				}
			}
		}},
		{"output limit", func(r *runner) string {
			if r.s.LimitConfig == nil {
				return "Suite.LimitConfig is not set"
			}
			return ""
		}, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithConfig(r.s.LimitConfig),
				ai.WithPrompt("Write a long essay about the history of sailing."))
			if resp.FinishReason != ai.FinishReasonLength {
				t.Errorf("FinishReason = %q, want %q for a capped output", resp.FinishReason, ai.FinishReasonLength)
			}
		}},
		{"cancel while streaming", always, func(t *testing.T, r *runner) {
			ctx, cancel := context.WithCancel(r.ctx)
			defer cancel()
			var cancelledAt time.Time
			resp, err := genkit.Generate(ctx, r.g,
				ai.WithModel(r.s.Model),
				ai.WithPrompt("Count from 1 to 500, one number per line, with no other text."),
				ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
					if cancelledAt.IsZero() && chunk.Text() != "" {
						cancelledAt = time.Now()
						cancel()
					}
					return nil
				}))
			if cancelledAt.IsZero() {
				t.Fatalf("Generate() returned before any text streamed (err %v)", err)
			}
			if err == nil {
				t.Fatal("Generate() error = nil, want the cancellation")
			}
			if got := status.Of(err); got != status.Cancelled {
				t.Errorf("status = %q, want %q: %v", got, status.Cancelled, err)
			}
			if elapsed := time.Since(cancelledAt); elapsed > 10*time.Second {
				t.Errorf("Generate() returned %v after the cancel, want it to stop promptly", elapsed)
			}
			if resp != nil && resp.FinishReason != ai.FinishReasonAborted {
				t.Errorf("FinishReason = %q, want %q on the partial response", resp.FinishReason, ai.FinishReasonAborted)
			}
		}},
		// A plugin that resolves only the models it registered fails an
		// unknown one locally, so the provider's answer is never checked.
		{"unknown model", func(r *runner) string {
			if genkit.LookupModel(r.g, unknownModel(r)) == nil {
				return "the plugin resolves no unregistered model names, so none reaches the provider"
			}
			return ""
		}, func(t *testing.T, r *runner) {
			wantRefused(t, r, r.g, ai.NewModelRef(unknownModel(r), nil), status.NotFound)
		}},
		{"bad api key", func(r *runner) string {
			if r.s.BadKeyPlugin == nil {
				return "Suite.BadKeyPlugin is not set"
			}
			return ""
		}, func(t *testing.T, r *runner) {
			g := genkit.Init(r.ctx, genkit.WithPlugins(r.s.BadKeyPlugin))
			wantRefused(t, r, g, r.s.Model, status.Unauthenticated)
		}},

		// Tools.
		{"tool calling", needTools, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.gablorken),
				ai.WithPrompt("Use the gablorken tool with value 4 and over 2, then reply with just the number it returns."))
			wantReply(t, resp, "17")
		}},
		{"tool calling streaming", needTools, func(t *testing.T, r *runner) {
			chunks := 0
			resp := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.gablorken),
				ai.WithPrompt("Use the gablorken tool with value 4 and over 2, then reply with just the number it returns."),
				ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error {
					chunks++
					return nil
				}))
			if chunks == 0 {
				t.Error("chunks = 0, want streamed chunks across the tool round trip")
			}
			wantReply(t, resp, "17")
		}},
		{"parallel tool calls", needTools, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.gablorken),
				ai.WithPrompt("I need two gablorkens: value 2 over 3, and value 3 over 2. They are independent, so request both tool calls at once. Then reply with both numbers."))
			wantReply(t, resp, "9", "10")
			parallel := false
			for _, m := range resp.History() {
				if m.Role == ai.RoleModel && len(toolRequests([]*ai.Message{m})) >= 2 {
					parallel = true
				}
			}
			if !parallel {
				t.Error("no model message requested both tools at once, want a parallel tool call")
			}
		}},
		{"sequential tool calls", needTools, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.gablorken),
				ai.WithPrompt("Compute the gablorken of value 2 over 3. Then compute the gablorken of that result over 2. Use the tool for each step and reply with just the final number."))
			wantReply(t, resp, "82")
			if n := len(toolRequests(resp.History())); n < 2 {
				t.Errorf("tool requests = %d, want one per step", n)
			}
		}},
		{"tool choice none", needAll(needTools, needToolChoice), func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.gablorken),
				ai.WithToolChoice(ai.ToolChoiceNone),
				ai.WithPrompt("What is the gablorken of 4 over 2?"))
			// The only contract is that no tool is called: a model denied
			// its tool may answer with anything, including nothing.
			if reqs := resp.ToolRequests(); len(reqs) != 0 {
				t.Errorf("ToolRequests() = %d requests, want none when the choice forbids tools", len(reqs))
			}
		}},
		{"tool choice required", needAll(needTools, needToolChoice), func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.gablorken),
				ai.WithToolChoice(ai.ToolChoiceRequired),
				ai.WithReturnToolRequests(true),
				ai.WithPrompt("Say hello."))
			if len(resp.ToolRequests()) == 0 {
				t.Error("ToolRequests() is empty, want the forced tool call back")
			}
		}},
		{"tool failure then retry", needTools, func(t *testing.T, r *runner) {
			r.tools.failLookups.Store(1)
			defer r.tools.failLookups.Store(0)
			resp, err := genkit.Generate(r.ctx, r.g,
				ai.WithModel(r.s.Model),
				ai.WithTools(r.tools.lookupOrder),
				ai.WithPrompt("Call the lookupOrder tool for order 1234 and tell me its carrier."))
			if !errors.Is(err, ai.ErrToolFailed) {
				t.Fatalf("Generate() error = %v, want %v", err, ai.ErrToolFailed)
			}
			if resp == nil {
				t.Fatal("Generate() response = nil, want the partial response")
			}
			// The partial's history ends where the failed turn began, so
			// sending it again is the retry.
			retry := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.lookupOrder),
				ai.WithMessages(resp.History()...))
			wantReply(t, retry, "quokka")
		}},
		{"interrupt then restart", needTools, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.transfer),
				ai.WithPrompt(transferPrompt))
			interrupt := wantInterrupt(t, resp)
			restart, err := r.tools.transfer.RestartWith(interrupt,
				ai.WithResumedMetadata[transferInput](map[string]any{"approved": true}))
			if err != nil {
				t.Fatalf("RestartWith() error = %v", err)
			}
			resumed := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.transfer),
				ai.WithMessages(resp.History()...),
				ai.WithToolRestarts(restart))
			wantReply(t, resumed, transferCode)
		}},
		{"interrupt then respond", needTools, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.transfer),
				ai.WithPrompt("Send 50 dollars to Ada with the transferFunds tool and tell me whether it went through."))
			interrupt := wantInterrupt(t, resp)
			respond, err := r.tools.transfer.RespondWith(interrupt,
				transferResult{Status: "rejected by the account owner"})
			if err != nil {
				t.Fatalf("RespondWith() error = %v", err)
			}
			resumed := r.gen(t, r.s.Model,
				ai.WithTools(r.tools.transfer),
				ai.WithMessages(resp.History()...),
				ai.WithToolResponses(respond))
			if !containsAnyFold(resumed.Text(), "reject", "declin", "denied", "not go through", "did not", "didn't") {
				t.Errorf("text = %q, want it to report the rejected transfer", resumed.Text())
			}
		}},
		{"media in tool response", func(r *runner) string {
			if !r.s.ToolResponseMedia {
				return "Suite.ToolResponseMedia is not set"
			}
			return needVisionTools(r)
		}, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.vision,
				ai.WithTools(r.tools.swatch),
				ai.WithPrompt("Fetch the swatch named primary with the fetchSwatch tool and tell me its color in one word."))
			wantReply(t, resp, "red")
		}},
		{"tools then structured output", needTools, func(t *testing.T, r *runner) {
			answer, resp, err := genkit.GenerateData[gablorkenAnswer](r.ctx, r.g,
				ai.WithModel(r.s.Model),
				ai.WithTools(r.tools.gablorken),
				ai.WithPrompt("Use the gablorken tool with value 4 and over 2 and report its result."))
			if err != nil {
				t.Fatalf("GenerateData() error = %v", err)
			}
			if answer.Result != 17 {
				t.Errorf("Result = %v (%d tool calls in history), want 17", answer.Result, len(toolRequests(resp.History())))
			}
		}},

		// Structured output.
		{"structured output", always, func(t *testing.T, r *runner) {
			facts, _, err := genkit.GenerateData[capitalFacts](r.ctx, r.g,
				ai.WithModel(r.s.Model),
				ai.WithPrompt("What is the capital city of France? Fill in the city and its country."))
			if err != nil {
				t.Fatalf("GenerateData() error = %v", err)
			}
			wantText(t, facts.City, "paris")
		}},
		{"structured output streaming", always, func(t *testing.T, r *runner) {
			chunks := 0
			var facts *capitalFacts
			for val, err := range genkit.GenerateDataStream[capitalFacts](r.ctx, r.g,
				ai.WithModel(r.s.Model),
				ai.WithPrompt("What is the capital city of France? Fill in the city and its country."),
			) {
				if err != nil {
					t.Fatalf("GenerateDataStream() error = %v", err)
				}
				if val.Done {
					out := val.Output
					facts = &out
				} else {
					chunks++
				}
			}
			if chunks == 0 {
				t.Error("chunks = 0, want the structured output streamed")
			}
			if facts == nil {
				t.Fatal("the stream never delivered the final value")
			}
			wantText(t, facts.City, "paris")
		}},
		{"json mode", always, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithPrompt("Name the capital city of France."),
				ai.WithOutputFormat(ai.OutputFormatJSON),
				ai.WithOutputInstructions("Reply with a JSON object holding a single string field named city."))
			var out map[string]any
			if err := resp.Output(&out); err != nil {
				t.Fatalf("Output() error = %v on %q", err, resp.Text())
			}
			city, _ := out["city"].(string)
			wantText(t, city, "paris")
		}},
		{"array output", always, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithPrompt(planetsPrompt),
				ai.WithOutputType([]planet{}),
				ai.WithOutputFormat(ai.OutputFormatArray))
			var planets []planet
			if err := resp.Output(&planets); err != nil {
				t.Fatalf("Output() error = %v on %q", err, resp.Text())
			}
			wantPlanets(t, planets)
		}},
		{"array output streaming", always, func(t *testing.T, r *runner) {
			wantPlanetStream(t, r, ai.OutputFormatArray)
		}},
		{"jsonl output streaming", always, func(t *testing.T, r *runner) {
			wantPlanetStream(t, r, ai.OutputFormatJSONL)
		}},
		{"enum output", always, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.Model,
				ai.WithPrompt("What color is a ripe banana?"),
				ai.WithOutputEnums("red", "green", "blue", "yellow"))
			if got := strings.TrimSpace(resp.Text()); got != "yellow" {
				t.Errorf("Text() = %q, want %q", got, "yellow")
			}
		}},
		// A model that only follows the format instructions answers this one
		// with "yellow", the true answer outside the set; only a constraint
		// applied on the wire keeps it to the set.
		{"enum output is constrained", needConstrained, func(t *testing.T, r *runner) {
			resp, err := genkit.Generate(r.ctx, r.g,
				ai.WithModel(r.s.Model),
				ai.WithPrompt("What color is a ripe banana? Name its true color."),
				ai.WithOutputEnums("red", "green", "blue"))
			if err != nil {
				t.Fatalf("Generate() error = %v, want an answer held to the enum by the provider", err)
			}
			switch got := strings.TrimSpace(resp.Text()); got {
			case "red", "green", "blue":
			default:
				t.Errorf("Text() = %q, want one of the enum values", got)
			}
		}},

		// Media.
		{"image input", needVision, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.vision,
				ai.WithMessages(ai.NewUserMessage(
					ai.NewMediaPart("image/png", RedImage),
					ai.NewTextPart("What is the dominant color of this image? Reply with one word."),
				)))
			wantReply(t, resp, "red")
		}},

		// Reasoning.
		{"reasoning", needNonStreamReasoning, func(t *testing.T, r *runner) {
			resp := r.gen(t, r.s.ReasoningModel,
				ai.WithPrompt("Is 91 a prime number? Answer yes or no."))
			if resp.Text() == "" {
				t.Error("Text() is empty")
			}
			if r.s.ReasoningContent && resp.Reasoning() == "" {
				t.Error("Reasoning() is empty, want the thinking content")
			}
		}},
		{"reasoning streaming", needReasoning, func(t *testing.T, r *runner) {
			var reasoning, text strings.Builder
			resp := r.gen(t, r.s.ReasoningModel,
				ai.WithPrompt("Is 91 a prime number? Answer yes or no."),
				ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
					reasoning.WriteString(chunk.Reasoning())
					text.WriteString(chunk.Text())
					return nil
				}))
			if text.String() == "" {
				t.Fatal("streamed text is empty")
			}
			if resp.Text() != text.String() {
				t.Errorf("final text = %q, want the streamed %q", resp.Text(), text.String())
			}
			if r.s.ReasoningContent {
				if reasoning.String() == "" {
					t.Fatal("streamed reasoning is empty, want the thinking content")
				}
				if resp.Reasoning() != reasoning.String() {
					t.Errorf("final reasoning = %q, want the streamed %q", resp.Reasoning(), reasoning.String())
				}
			}
		}},
		// Reasoning providers tie their thinking to the tool calls it
		// produced (Gemini thought signatures, Anthropic thinking
		// signatures, DeepSeek reasoning content), and reject a later turn
		// whose history drops or garbles them.
		{"reasoning with tools across turns", needReasoningTools, func(t *testing.T, r *runner) {
			var streamOpts []ai.GenerateOption
			if r.s.StreamOnlyReasoning {
				streamOpts = append(streamOpts, ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error { return nil }))
			}
			turn := func(history []*ai.Message, prompt string, want ...string) *ai.ModelResponse {
				t.Helper()
				opts := append([]ai.GenerateOption{
					ai.WithTools(r.tools.gablorken),
					ai.WithMessages(history...),
					ai.WithPrompt(prompt),
				}, streamOpts...)
				resp := r.gen(t, r.s.ReasoningModel, opts...)
				wantReply(t, resp, want...)
				return resp
			}
			first := turn(nil, "Use the gablorken tool with value 3 and over 2, then reply with just the number.", "10")
			if r.s.ReasoningContent && !hasReasoning(first.History()) {
				t.Error("first turn kept no reasoning in its history, want the thinking that led to the tool call")
			}
			second := turn(first.History(), "Call the gablorken tool exactly once, with value 10 and over 2. Reply with just the number it returns.", "101")
			turn(second.History(), "What were the two gablorken results? Reply with both numbers.", "10", "101")
		}},
	}
}

// unknownModel is a model name under the provider of Model that no provider
// serves.
func unknownModel(r *runner) string {
	provider, _, _ := strings.Cut(r.s.Model.Name(), "/")
	return provider + "/livetest-no-such-model"
}

// wantRefused checks that the provider refuses a request to model on g with
// the status want, on a plain call and on a stream. A streamed request
// returns before the response arrives, so its refusal surfaces at the stream
// rather than at the call, and has to classify the same way.
func wantRefused(t *testing.T, r *runner, g *genkit.Genkit, model ai.ModelArg, want status.Name) {
	t.Helper()
	for _, streaming := range []bool{false, true} {
		opts := []ai.GenerateOption{ai.WithModel(model), ai.WithPrompt("Hello.")}
		if streaming {
			opts = append(opts, ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error { return nil }))
		}
		resp, err := genkit.Generate(r.ctx, g, opts...)
		if err == nil {
			t.Errorf("Generate(streaming %v) error = nil, want the request refused (text %q)", streaming, resp.Text())
			continue
		}
		if got, ok := status.Classified(err); !ok || got != want {
			t.Errorf("Generate(streaming %v) status = %q (classified %v), want %q: %v", streaming, got, ok, want, err)
		}
	}
}

// wantInterrupt returns the single interrupt resp stopped on.
func wantInterrupt(t *testing.T, resp *ai.ModelResponse) *ai.Part {
	t.Helper()
	if resp.FinishReason != ai.FinishReasonInterrupted {
		t.Fatalf("FinishReason = %q, want %q (text %q)", resp.FinishReason, ai.FinishReasonInterrupted, resp.Text())
	}
	interrupts := resp.Interrupts()
	if len(interrupts) != 1 {
		t.Fatalf("Interrupts() = %d parts, want the one transfer", len(interrupts))
	}
	return interrupts[0]
}

// hasReasoning reports whether any message in msgs carries a reasoning part.
func hasReasoning(msgs []*ai.Message) bool {
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.IsReasoning() {
				return true
			}
		}
	}
	return false
}

// wantPlanets checks the eight planets the planet prompts ask for.
func wantPlanets(t *testing.T, planets []planet) {
	t.Helper()
	want := []string{"mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"}
	if len(planets) != len(want) {
		t.Fatalf("planets = %+v, want %d of them", planets, len(want))
	}
	for i, p := range planets {
		if !containsFold(p.Name, want[i]) || p.Position != i+1 {
			t.Errorf("planets[%d] = %+v, want %s at position %d", i, p, want[i], i+1)
		}
	}
}

// wantPlanetStream checks that a list format streams its items as they
// complete: over more than one chunk, each item once, and adding up to the
// final output.
func wantPlanetStream(t *testing.T, r *runner, format string) {
	t.Helper()
	var streamed []planet
	itemChunks := 0
	var final *ai.ModelResponse
	for val, err := range genkit.GenerateStream(r.ctx, r.g,
		ai.WithModel(r.s.Model),
		ai.WithPrompt(planetsPrompt),
		ai.WithOutputType([]planet{}),
		ai.WithOutputFormat(format),
	) {
		if err != nil {
			t.Fatalf("GenerateStream() error = %v", err)
		}
		if val.Done {
			final = val.Response
			continue
		}
		var items []planet
		if err := val.Chunk.Output(&items); err != nil {
			continue // a chunk that completes no item has no output
		}
		if len(items) > 0 {
			itemChunks++
			streamed = append(streamed, items...)
		}
	}
	if final == nil {
		t.Fatal("the stream never delivered the final response")
	}
	var planets []planet
	if err := final.Output(&planets); err != nil {
		t.Fatalf("Output() error = %v on %q", err, final.Text())
	}
	wantPlanets(t, planets)
	if itemChunks < 2 {
		t.Errorf("items arrived in %d chunks, want them streamed as they complete", itemChunks)
	}
	if len(streamed) != len(planets) {
		t.Errorf("streamed %d items, want each of the %d final items once", len(streamed), len(planets))
	}
}
