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

package middleware

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// scriptedToolModel returns a tool-loop model handler: call i requests the
// tool with inputs[i], and the call after the last input returns finalText.
// usages[i] is the inputTokens reported for call i (the last value repeats
// for later calls). Every call's request messages are recorded in seen, and
// the handler stamps the request onto the response the way real model
// plugins do.
func scriptedToolModel(toolName string, inputs []map[string]any, usages []int, finalText string, seen *[][]*ai.Message) ai.ModelFunc {
	calls := 0
	return func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		call := calls
		calls++
		*seen = append(*seen, slices.Clone(req.Messages))

		usage := usages[min(call, len(usages)-1)]
		resp := &ai.ModelResponse{
			Request: req,
			Usage:   &ai.GenerationUsage{InputTokens: usage, OutputTokens: 10},
		}
		if call < len(inputs) {
			resp.Message = ai.NewMessage(ai.RoleModel, nil,
				ai.NewToolRequestPart(&ai.ToolRequest{Name: toolName, Input: inputs[call]}))
		} else {
			resp.Message = ai.NewModelTextMessage(finalText)
		}
		return resp, nil
	}
}

// defineToolLoopModel registers a tool-capable model backed by fn.
func defineToolLoopModel(t *testing.T, g *genkit.Genkit, name string, fn ai.ModelFunc) ai.Model {
	t.Helper()
	return genkit.DefineModel(g, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true},
	}, fn)
}

// defineEchoTool registers a tool that echoes payload characters: input
// {"v": string} returns "result:" + v.
func defineEchoTool(t *testing.T, g *genkit.Genkit, name string) ai.Tool {
	t.Helper()
	return genkit.DefineTool(g, name, "echoes its input",
		func(ctx *ai.ToolContext, input struct {
			V string `json:"v"`
		}) (string, error) {
			return "result:" + input.V, nil
		})
}

// defineSummarizer registers a summarizer model that records each prompt it
// receives and returns the next text from texts (the last repeats).
func defineSummarizer(t *testing.T, g *genkit.Genkit, prompts *[]string, texts ...string) ai.Model {
	t.Helper()
	calls := 0
	return defineTestModel(t, g, "test/summarizer", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		*prompts = append(*prompts, req.Messages[len(req.Messages)-1].Text())
		text := texts[min(calls, len(texts)-1)]
		calls++
		return &ai.ModelResponse{
			Request: req,
			Message: ai.NewModelTextMessage(text),
			Usage:   &ai.GenerationUsage{InputTokens: 200, OutputTokens: 50},
		}, nil
	})
}

// stampOf returns the compression metadata object on a message, or nil.
func stampOf(m *ai.Message) map[string]any {
	if m == nil || m.Metadata == nil {
		return nil
	}
	stamp, _ := m.Metadata[CompressionMetadataKey].(map[string]any)
	return stamp
}

// toolOutputs returns the output string of every tool response in msgs, in
// order.
func toolOutputs(msgs []*ai.Message) []string {
	var outs []string
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.IsToolResponse() {
				outs = append(outs, toolOutputString(p.ToolResponse.Output))
			}
		}
	}
	return outs
}

// assertNoOrphanToolResponses fails if msgs contains a tool response whose
// matching tool request (by ref) is absent.
func assertNoOrphanToolResponses(t *testing.T, msgs []*ai.Message) {
	t.Helper()
	refs := map[string]bool{}
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.IsToolRequest() {
				refs[p.ToolRequest.Ref] = true
			}
		}
	}
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.IsToolResponse() && !refs[p.ToolResponse.Ref] {
				t.Errorf("tool response %q (ref %q) has no matching tool request in the view",
					p.ToolResponse.Name, p.ToolResponse.Ref)
			}
		}
	}
}

// bigHistory builds n alternating user/model text messages of size chars
// each, with a unique marker in each.
func bigHistory(n, chars int) []*ai.Message {
	msgs := make([]*ai.Message, n)
	for i := range n {
		role := ai.RoleUser
		if i%2 == 1 {
			role = ai.RoleModel
		}
		msgs[i] = ai.NewTextMessage(role, fmt.Sprintf("msg-%d ", i)+strings.Repeat("x", chars))
	}
	return msgs
}

func TestCompressionPassthrough(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	tool := defineEchoTool(t, g, "echo")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("echo",
		[]map[string]any{{"v": "1"}}, []int{700}, "done", &seen))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("hello"),
		ai.WithTools(tool),
		ai.WithUse(&ContextCompression{MaxInputTokens: 100_000}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "done" {
		t.Errorf("got %q, want %q", resp.Text(), "done")
	}
	if got := len(seen[0]); got != 1 {
		t.Errorf("first call saw %d messages, want 1", got)
	}
	if got := len(seen[1]); got != 3 {
		t.Errorf("second call saw %d messages, want 3", got)
	}
	// Model messages carry the usage annotation even without compaction.
	history := resp.History()
	if v, ok := stampOf(history[1])["inputTokens"]; !ok || v != 700 {
		t.Errorf("model message inputTokens stamp = %v, want 700", v)
	}
	if v, ok := stampOf(resp.Message)["inputTokens"]; !ok || v != 700 {
		t.Errorf("final message inputTokens stamp = %v, want 700", v)
	}
	for _, msgs := range seen {
		for _, m := range msgs {
			if _, ok := stampOf(m)["summary"]; ok {
				t.Error("no compaction expected, found a summary stamp")
			}
		}
	}
}

func TestCompressionSafetyCapAlwaysOn(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	payload := strings.Repeat("H", 5000)
	tool := defineEchoTool(t, g, "huge")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("huge",
		[]map[string]any{{"v": payload}}, []int{100}, "capped", &seen))

	// No triggers configured at all: the cap still applies to every call.
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(tool),
		ai.WithUse(&ContextCompression{MaxToolResponseChars: 100}),
	)
	if err != nil {
		t.Fatal(err)
	}

	viewOuts := toolOutputs(seen[1])
	if len(viewOuts) != 1 || !strings.Contains(viewOuts[0], "[TRUNCATED: Response was") {
		t.Errorf("view tool output not capped: %.80q", viewOuts)
	}
	if len(viewOuts[0]) > 100+200 {
		t.Errorf("capped output is %d chars, want ~100 plus marker", len(viewOuts[0]))
	}
	// The caller-visible history keeps the full output.
	histOuts := toolOutputs(resp.History())
	if len(histOuts) != 1 || !strings.Contains(histOuts[0], payload) {
		t.Error("history tool output was rewritten; the cap must only affect the view")
	}
}

func TestCompressionDedupe(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	tool := defineEchoTool(t, g, "read")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("read",
		[]map[string]any{{"v": "a"}, {"v": "a"}, {"v": "b"}}, []int{100}, "done", &seen))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(tool),
		ai.WithUse(&ContextCompression{DedupeToolResponses: &CompressionDedupe{}}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Final call sees three tool responses: the older duplicate of input
	// {"v":"a"} is elided; the newest "a" and the only "b" survive.
	got := toolOutputs(seen[3])
	want := []string{defaultDedupeNotice, "result:a", "result:b"}
	if !slices.Equal(got, want) {
		t.Errorf("view tool outputs = %q, want %q", got, want)
	}
	// History keeps all three outputs untouched.
	hist := toolOutputs(resp.History())
	want = []string{"result:a", "result:a", "result:b"}
	if !slices.Equal(hist, want) {
		t.Errorf("history tool outputs = %q, want %q", hist, want)
	}
}

func TestCompressionDedupeNameOnly(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	tool := defineEchoTool(t, g, "state")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("state",
		[]map[string]any{{"v": "a"}, {"v": "b"}, {"v": "c"}}, []int{100}, "done", &seen))

	_, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(tool),
		ai.WithUse(&ContextCompression{
			DedupeToolResponses: &CompressionDedupe{MatchBy: CompressionDedupeNameOnly},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	got := toolOutputs(seen[3])
	want := []string{defaultDedupeNotice, defaultDedupeNotice, "result:c"}
	if !slices.Equal(got, want) {
		t.Errorf("view tool outputs = %q, want %q", got, want)
	}
}

func TestCompressionTruncateToolResponses(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	payload := strings.Repeat("Z", 200)
	tool := defineEchoTool(t, g, "verbose")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("verbose",
		[]map[string]any{{"v": payload + "1"}, {"v": payload + "2"}, {"v": payload + "3"}},
		[]int{100}, "done", &seen))

	_, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(tool),
		ai.WithUse(&ContextCompression{
			TruncateToolResponses: &CompressionToolTruncation{MaxChars: 50, PreserveRecent: 1},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	got := toolOutputs(seen[3])
	if len(got) != 3 {
		t.Fatalf("final call saw %d tool outputs, want 3", len(got))
	}
	for i, out := range got[:2] {
		if !strings.HasSuffix(out, "…[truncated]") || len(out) > 50+20 {
			t.Errorf("older tool output %d not truncated: %.80q", i, out)
		}
	}
	if !strings.Contains(got[2], payload+"3") {
		t.Errorf("newest tool output must be preserved, got %.80q", got[2])
	}
}

func TestCompressionTokenTriggerWithSummarizer(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	var prompts []string
	defineSummarizer(t, g, &prompts, "S1")
	tool := defineEchoTool(t, g, "research")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("research",
		[]map[string]any{{"v": "1"}, {"v": "2"}}, []int{500, 5000, 800}, "done", &seen))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("research the topic"),
		ai.WithTools(tool),
		ai.WithUse(&ContextCompression{
			MaxInputTokens: 1000,
			PreserveRecent: 2,
			Summarizer:     &CompressionSummarizer{Model: ai.NewModelRef("test/summarizer", nil)},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "done" {
		t.Fatalf("got %q, want %q", resp.Text(), "done")
	}

	// The first two calls are below the threshold and uncompacted.
	if len(seen[0]) != 1 || len(seen[1]) != 3 {
		t.Errorf("early calls saw %d and %d messages, want 1 and 3", len(seen[0]), len(seen[1]))
	}

	// The third call runs after a 5000-token report against a 1000 budget:
	// everything but the last two messages is folded into the summary.
	final := seen[2]
	if len(final) != 3 {
		t.Fatalf("final call saw %d messages, want 3 (summary + 2 recent)", len(final))
	}
	if final[0].Role != ai.RoleUser || !strings.HasPrefix(final[0].Text(), summaryPrefix) || !strings.Contains(final[0].Text(), "S1") {
		t.Errorf("first view message is not the summary: %.100q", final[0].Text())
	}
	assertNoOrphanToolResponses(t, final)

	if len(prompts) != 1 {
		t.Fatalf("summarizer called %d times, want 1", len(prompts))
	}
	if !strings.Contains(prompts[0], "research the topic") || !strings.Contains(prompts[0], "[Tool call: research") {
		t.Errorf("summarizer prompt missing conversation rendering: %.200q", prompts[0])
	}

	// The caller sees the full, unreplaced history with the stamps in place.
	if len(resp.Request.Messages) != 5 {
		t.Fatalf("resp.Request has %d messages, want the full 5", len(resp.Request.Messages))
	}
	history := resp.History()
	if len(history) != 6 {
		t.Fatalf("history has %d messages, want 6", len(history))
	}
	stamp := stampOf(history[2])
	if stamp == nil || stamp["summary"] != "S1" {
		t.Fatalf("boundary stamp = %v, want summary S1 on message 2", stamp)
	}
	stats, _ := stamp["stats"].(map[string]any)
	if stats == nil {
		t.Fatal("boundary stamp has no stats")
	}
	if stats["trigger"] != "inputTokens" || stats["inputTokens"] != 5000 || stats["messagesCompacted"] != 3 || stats["summarized"] != true {
		t.Errorf("stats = %v", stats)
	}
	if stats["summaryModel"] != "test/summarizer" || stats["summaryInputTokens"] != 200 {
		t.Errorf("summary stats = %v", stats)
	}
	if hist := toolOutputs(history); !slices.Equal(hist, []string{"result:1", "result:2"}) {
		t.Errorf("history tool outputs rewritten: %q", hist)
	}
}

func TestCompressionEstimateTriggersFirstCall(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	var prompts []string
	defineSummarizer(t, g, &prompts, "S1")
	m := defineTestModel(t, g, "test/model", scriptedToolModel("unused",
		nil, []int{100}, "done", &seen))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithMessages(bigHistory(6, 2000)...),
		ai.WithPrompt("final question"),
		ai.WithUse(&ContextCompression{
			MaxInputTokens: 1000,
			PreserveRecent: 1,
			Summarizer:     &CompressionSummarizer{Model: ai.NewModelRef("test/summarizer", nil)},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// No usage annotation exists yet, so the character estimate (about 3000
	// tokens against a 1000 budget) triggers on the very first call.
	first := seen[0]
	if len(first) != 2 {
		t.Fatalf("first call saw %d messages, want 2 (summary + prompt)", len(first))
	}
	if !strings.Contains(first[0].Text(), "S1") || first[1].Text() != "final question" {
		t.Errorf("unexpected view: %.80q / %.80q", first[0].Text(), first[1].Text())
	}

	history := resp.History()
	if len(history) != 8 {
		t.Fatalf("history has %d messages, want 8", len(history))
	}
	stats, _ := stampOf(history[5])["stats"].(map[string]any)
	if stats == nil {
		t.Fatal("boundary stamp missing on message 5")
	}
	if _, ok := stats["estimatedTokens"]; !ok {
		t.Errorf("stats should record the estimated reading: %v", stats)
	}
}

func TestCompressionNoticeOnlyWithoutSummarizer(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	m := defineTestModel(t, g, "test/model", scriptedToolModel("unused",
		nil, []int{100}, "done", &seen))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithMessages(bigHistory(6, 2000)...),
		ai.WithPrompt("final question"),
		ai.WithUse(&ContextCompression{MaxInputTokens: 1000, PreserveRecent: 1}),
	)
	if err != nil {
		t.Fatal(err)
	}

	first := seen[0]
	if len(first) != 2 {
		t.Fatalf("first call saw %d messages, want 2 (notice + prompt)", len(first))
	}
	if first[0].Text() != defaultTruncationNotice {
		t.Errorf("first view message = %.100q, want the truncation notice", first[0].Text())
	}
	stamp := stampOf(resp.History()[5])
	if stamp == nil {
		t.Fatal("boundary stamp missing")
	}
	if stamp["summary"] != "" {
		t.Errorf("summary = %q, want empty for a notice-only compaction", stamp["summary"])
	}
	stats, _ := stamp["stats"].(map[string]any)
	if stats["summarized"] != false {
		t.Errorf("stats = %v, want summarized false", stats)
	}
}

func TestCompressionIncrementalSummaries(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	var prompts []string
	defineSummarizer(t, g, &prompts, "S1", "S2")
	tool := defineEchoTool(t, g, "step")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("step",
		[]map[string]any{{"v": "1"}, {"v": "2"}}, []int{5000}, "done", &seen))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("marker-original-request"),
		ai.WithTools(tool),
		ai.WithUse(&ContextCompression{
			MaxInputTokens: 1000,
			PreserveRecent: 2,
			Summarizer:     &CompressionSummarizer{Model: ai.NewModelRef("test/summarizer", nil)},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(prompts) != 2 {
		t.Fatalf("summarizer called %d times, want 2", len(prompts))
	}
	// The first compaction covers the original prompt.
	if !strings.Contains(prompts[0], "marker-original-request") || strings.Contains(prompts[0], "[Previous summary]") {
		t.Errorf("first summarizer prompt: %.200q", prompts[0])
	}
	// The second folds the previous summary plus only the new messages; the
	// already-covered prompt is not re-rendered.
	if !strings.Contains(prompts[1], "[Previous summary]\nS1") {
		t.Errorf("second summarizer prompt missing previous summary: %.200q", prompts[1])
	}
	if strings.Contains(prompts[1], "marker-original-request") {
		t.Errorf("second summarizer prompt re-renders covered messages: %.200q", prompts[1])
	}

	// Both boundary stamps remain in the history; the model only ever sees
	// the newest summary.
	history := resp.History()
	if got := stampOf(history[0])["summary"]; got != "S1" {
		t.Errorf("first boundary summary = %v, want S1", got)
	}
	if got := stampOf(history[2])["summary"]; got != "S2" {
		t.Errorf("second boundary summary = %v, want S2", got)
	}
	final := seen[len(seen)-1]
	if len(final) != 3 || !strings.Contains(final[0].Text(), "S2") || strings.Contains(final[0].Text(), "S1") {
		t.Errorf("final view should carry only the newest summary: %.120q", final[0].Text())
	}
}

func TestCompressionSummarizerFailureFailsOpen(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	defineTestModel(t, g, "test/summarizer", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return nil, fmt.Errorf("summarizer unavailable")
	})
	tool := defineEchoTool(t, g, "work")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("work",
		[]map[string]any{{"v": "1"}, {"v": "2"}}, []int{5000}, "completed", &seen))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("do work"),
		ai.WithTools(tool),
		ai.WithUse(&ContextCompression{
			MaxInputTokens: 1000,
			PreserveRecent: 2,
			Summarizer:     &CompressionSummarizer{Model: ai.NewModelRef("test/summarizer", nil)},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "completed" {
		t.Fatalf("got %q, want %q", resp.Text(), "completed")
	}
	// Every call proceeded with the full, uncompacted view and nothing was
	// stamped as a boundary.
	final := seen[len(seen)-1]
	if len(final) != 5 {
		t.Errorf("final call saw %d messages, want the full 5", len(final))
	}
	for i, m := range resp.History() {
		if _, ok := stampOf(m)["summary"]; ok {
			t.Errorf("message %d has a boundary stamp after summarizer failure", i)
		}
	}
}

func TestCompressionSummarizerMissingFailsFast(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	m := defineTestModel(t, g, "test/model", scriptedToolModel("unused",
		nil, []int{100}, "done", &seen))

	_, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("hello"),
		ai.WithUse(&ContextCompression{
			MaxInputTokens: 1000,
			Summarizer:     &CompressionSummarizer{Model: ai.NewModelRef("test/nonexistent", nil)},
		}),
	)
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("err = %v, want a summarizer-not-found error", err)
	}
	if len(seen) != 0 {
		t.Errorf("model was called %d times before the config error surfaced, want 0", len(seen))
	}
}

func TestCompressionBoundarySnapsPastToolMessages(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	m := defineTestModel(t, g, "test/model", scriptedToolModel("unused",
		nil, []int{100}, "done", &seen))

	// A history whose natural boundary would orphan the pending tool
	// response: ... model(toolRequest) | tool(toolResponse) ...
	history := []*ai.Message{
		ai.NewUserTextMessage(strings.Repeat("x", 6000)),
		ai.NewMessage(ai.RoleModel, nil, ai.NewToolRequestPart(&ai.ToolRequest{Name: "calc", Ref: "r1", Input: map[string]any{"v": "1"}})),
		ai.NewMessage(ai.RoleTool, nil, ai.NewToolResponsePart(&ai.ToolResponse{Name: "calc", Ref: "r1", Output: "one"})),
		ai.NewMessage(ai.RoleModel, nil, ai.NewToolRequestPart(&ai.ToolRequest{Name: "calc", Ref: "r2", Input: map[string]any{"v": "2"}})),
		ai.NewMessage(ai.RoleTool, nil, ai.NewToolResponsePart(&ai.ToolResponse{Name: "calc", Ref: "r2", Output: "two"})),
	}

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithMessages(history...),
		ai.WithPrompt("continue"),
		ai.WithUse(&ContextCompression{MaxInputTokens: 1000, PreserveRecent: 2}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// PreserveRecent 2 would keep [tool(r2), user], starting with an
	// orphaned tool response; the boundary pulls back so the response stays
	// with the model message that requested it.
	first := seen[0]
	if len(first) != 4 {
		t.Fatalf("view has %d messages, want 4 (notice + request pair + prompt)", len(first))
	}
	if first[len(first)-1].Text() != "continue" {
		t.Fatalf("last view message = %.40q, want the prompt", first[len(first)-1].Text())
	}
	if first[1].Role == ai.RoleTool {
		t.Error("view starts with a tool message after the notice")
	}
	assertNoOrphanToolResponses(t, first)
	if _, ok := stampOf(resp.History()[2])["summary"]; !ok {
		t.Error("boundary should have pulled back to the tool message at index 2")
	}
}

// TestCompressionCompactsMidToolLoop asserts that a compaction can fire while
// the history ends in tool responses — the normal state mid-loop — by
// widening the kept window to the model message that requested them.
func TestCompressionCompactsMidToolLoop(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	tool := defineEchoTool(t, g, "step")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("step",
		[]map[string]any{{"v": "1"}, {"v": "2"}}, []int{5000}, "done", &seen))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(tool),
		// PreserveRecent 1 would start the kept window on a tool response
		// every mid-loop turn; the boundary widens instead of never firing.
		ai.WithUse(&ContextCompression{MaxInputTokens: 1000, PreserveRecent: 1}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "done" {
		t.Fatalf("got %q, want %q", resp.Text(), "done")
	}

	compacted := false
	for _, msgs := range seen {
		if len(msgs) > 0 && msgs[0].Text() == defaultTruncationNotice {
			compacted = true
			assertNoOrphanToolResponses(t, msgs)
		}
	}
	if !compacted {
		t.Error("no call saw a compacted view; mid-loop compaction never fired")
	}
}

func TestCompressionMaxMessagesTrigger(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	m := defineTestModel(t, g, "test/model", scriptedToolModel("unused",
		nil, []int{100}, "done", &seen))

	_, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithMessages(bigHistory(7, 10)...),
		ai.WithPrompt("final"),
		ai.WithUse(&ContextCompression{MaxMessages: 4}),
	)
	if err != nil {
		t.Fatal(err)
	}

	first := seen[0]
	if len(first) != 4 {
		t.Fatalf("first call saw %d messages, want 4 (notice + 3 recent)", len(first))
	}
	if first[0].Text() != defaultTruncationNotice {
		t.Errorf("first view message = %.80q, want the truncation notice", first[0].Text())
	}
}

func TestCompressionOvershootShrinksWindow(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	tool := defineEchoTool(t, g, "step")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("step",
		[]map[string]any{{"v": "1"}, {"v": "2"}}, []int{500, 5000, 800}, "done", &seen))

	_, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("go"),
		ai.WithTools(tool),
		// PreserveRecent 4 would keep everything (5 messages, nothing to
		// fold); at 5x over budget the window shrinks to 2, so the
		// compaction fires.
		ai.WithUse(&ContextCompression{MaxInputTokens: 1000, PreserveRecent: 4}),
	)
	if err != nil {
		t.Fatal(err)
	}

	final := seen[2]
	if len(final) != 3 {
		t.Fatalf("final call saw %d messages, want 3 (notice + 2 recent)", len(final))
	}
}

func TestCompressionInvalidMatchBy(t *testing.T) {
	g := newTestGenkit(t)
	m := defineTestModel(t, g, "test/model", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("nope")}, nil
	})
	_, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("hello"),
		ai.WithUse(&ContextCompression{DedupeToolResponses: &CompressionDedupe{MatchBy: "bogus"}}),
	)
	if err == nil || !strings.Contains(err.Error(), "matchBy") {
		t.Fatalf("err = %v, want an invalid matchBy error", err)
	}
}

func TestCompressionJSONDispatch(t *testing.T) {
	r := newTestRegistry(t)
	var seen [][]*ai.Message
	m := defineModel(t, r, "test/model", scriptedToolModel("unused",
		nil, []int{100}, "done", &seen))
	registerTestMiddleware(r, "compress the context", ContextCompression{})

	msgs := append(bigHistory(6, 2000), ai.NewUserTextMessage("final question"))
	_, err := ai.GenerateWithRequest(ctx, r, &ai.GenerateActionOptions{
		Model:    m.Name(),
		Messages: msgs,
		Use: []*ai.MiddlewareRef{{
			Name:   provider + "/contextCompression",
			Config: map[string]any{"maxInputTokens": 1000, "preserveRecent": 1},
		}},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	first := seen[0]
	if len(first) != 2 {
		t.Fatalf("first call saw %d messages, want 2 (notice + prompt)", len(first))
	}
	if first[0].Text() != defaultTruncationNotice {
		t.Errorf("first view message = %.80q, want the truncation notice", first[0].Text())
	}
}

// TestCompressionPreservesPromptScaffolding asserts that system messages and
// prompt-template messages (tagged by the exp agent runtime, which re-renders
// them every turn) are never compacted away and never chosen as a boundary.
func TestCompressionPreservesPromptScaffolding(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	m := defineTestModel(t, g, "test/model", scriptedToolModel("unused",
		nil, []int{100}, "done", &seen))

	scaffold := ai.NewUserTextMessage("scaffold-instructions")
	scaffold.Metadata = map[string]any{promptScaffoldKey: true}
	msgs := append([]*ai.Message{ai.NewSystemMessage(ai.NewTextPart("be helpful")), scaffold}, bigHistory(6, 2000)...)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithMessages(msgs...),
		ai.WithPrompt("final"),
		ai.WithUse(&ContextCompression{MaxInputTokens: 1000, PreserveRecent: 1}),
	)
	if err != nil {
		t.Fatal(err)
	}

	first := seen[0]
	if len(first) != 4 {
		t.Fatalf("view has %d messages, want 4 (system + scaffold + notice + final)", len(first))
	}
	if first[0].Role != ai.RoleSystem {
		t.Errorf("view[0] role = %q, want system", first[0].Role)
	}
	if first[1].Text() != "scaffold-instructions" {
		t.Errorf("view[1] = %.60q, want the scaffold message", first[1].Text())
	}
	if first[2].Text() != defaultTruncationNotice {
		t.Errorf("view[2] = %.60q, want the truncation notice", first[2].Text())
	}
	history := resp.History()
	for _, i := range []int{0, 1} {
		if _, ok := stampOf(history[i])["summary"]; ok {
			t.Errorf("boundary stamped on preserved message %d", i)
		}
	}
	if _, ok := stampOf(history[7])["summary"]; !ok {
		t.Error("boundary stamp missing from the last covered durable message")
	}
}

// TestCompressionPersistsAcrossCalls exercises the chat-style flow: the app
// persists resp.History() (JSON round-trip, as session stores and clients
// do) and passes it back on the next Generate. The boundary stamp must
// survive persistence, drive extraction on the next call without a fresh
// summarizer call, and keep working with JSON-degraded number types.
func TestCompressionPersistsAcrossCalls(t *testing.T) {
	g := newTestGenkit(t)
	var seen [][]*ai.Message
	var prompts []string
	defineSummarizer(t, g, &prompts, "S1")
	tool := defineEchoTool(t, g, "step")
	m := defineToolLoopModel(t, g, "test/model", scriptedToolModel("step",
		[]map[string]any{{"v": "1"}, {"v": "2"}}, []int{500, 5000, 800}, "done", &seen))

	cc := &ContextCompression{
		MaxInputTokens: 1000,
		PreserveRecent: 2,
		Summarizer:     &CompressionSummarizer{Model: ai.NewModelRef("test/summarizer", nil)},
	}

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithPrompt("first task"),
		ai.WithTools(tool),
		ai.WithUse(cc),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(prompts) != 1 {
		t.Fatalf("summarizer called %d times in the first call, want 1", len(prompts))
	}

	// Persist the history the way a session store or client app would.
	raw, err := json.Marshal(resp.History())
	if err != nil {
		t.Fatal(err)
	}
	var persisted []*ai.Message
	if err := json.Unmarshal(raw, &persisted); err != nil {
		t.Fatal(err)
	}

	m2 := defineToolLoopModel(t, g, "test/model2", scriptedToolModel("step",
		nil, []int{300}, "follow-up done", &seen))
	resp2, err := genkit.Generate(ctx, g,
		ai.WithModel(m2),
		ai.WithMessages(persisted...),
		ai.WithPrompt("follow-up"),
		ai.WithTools(tool),
		ai.WithUse(cc),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp2.Text() != "follow-up done" {
		t.Fatalf("got %q, want %q", resp2.Text(), "follow-up done")
	}

	// The persisted stamp drives extraction on the new call: the view starts
	// with the stored summary and no new summarizer call was needed.
	view := seen[len(seen)-1]
	if !strings.Contains(view[0].Text(), "S1") {
		t.Errorf("view does not start with the persisted summary: %.100q", view[0].Text())
	}
	for _, msg := range view {
		if msg.Text() == "first task" {
			t.Error("compacted message leaked back into the view after persistence")
		}
	}
	if len(prompts) != 1 {
		t.Errorf("summarizer called %d times total, want still 1 (stamp reused)", len(prompts))
	}
	// The full history keeps growing on top of the persisted one.
	if got, want := len(resp2.History()), len(persisted)+2; got != want {
		t.Errorf("history has %d messages, want %d", got, want)
	}
}

// TestCompressionConcurrentUse shares one middleware value across concurrent
// Generate calls. The middleware keeps no state of its own — everything lives
// in per-call message metadata — so this must be race-free (run with -race).
func TestCompressionConcurrentUse(t *testing.T) {
	g := newTestGenkit(t)
	cc := &ContextCompression{MaxInputTokens: 1000, PreserveRecent: 1}

	var wg sync.WaitGroup
	for i := range 8 {
		m := defineTestModel(t, g, fmt.Sprintf("test/model%d", i), func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Request: req,
				Message: ai.NewModelTextMessage("ok"),
				Usage:   &ai.GenerationUsage{InputTokens: 100, OutputTokens: 5},
			}, nil
		})
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(m),
				ai.WithMessages(bigHistory(6, 2000)...),
				ai.WithPrompt("go"),
				ai.WithUse(cc),
			)
			if err != nil {
				t.Error(err)
				return
			}
			if resp.Text() != "ok" {
				t.Errorf("got %q, want %q", resp.Text(), "ok")
			}
		}()
	}
	wg.Wait()
}

func TestCompressionStreamingPassesThrough(t *testing.T) {
	g := newTestGenkit(t)
	var chunks []string
	m := defineTestModel(t, g, "test/model", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if cb != nil {
			for _, c := range []string{"hel", "lo"} {
				if err := cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart(c)}}); err != nil {
					return nil, err
				}
			}
		}
		return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("hello")}, nil
	})

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithMessages(bigHistory(6, 2000)...),
		ai.WithPrompt("final"),
		ai.WithUse(&ContextCompression{MaxInputTokens: 1000, PreserveRecent: 1}),
		ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
			chunks = append(chunks, chunk.Text())
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "hello" {
		t.Errorf("got %q, want %q", resp.Text(), "hello")
	}
	if !slices.Contains(chunks, "hel") || !slices.Contains(chunks, "lo") {
		t.Errorf("streamed chunks = %q, want both model chunks", chunks)
	}
}
