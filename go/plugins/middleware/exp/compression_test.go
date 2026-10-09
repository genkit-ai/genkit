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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
)

// ccFixture is a Genkit instance with a model that records the requests it
// receives and echoes each one on its response, as model plugins do.
type ccFixture struct {
	g        *genkit.Genkit
	model    ai.Model
	requests []*ai.ModelRequest
}

// replyFunc answers the call-th request (from 1) to a fixture model.
type replyFunc func(call int, req *ai.ModelRequest) *ai.ModelResponse

// textReply answers every call with text, reporting inputTokens.
func textReply(text string, inputTokens int) replyFunc {
	return func(int, *ai.ModelRequest) *ai.ModelResponse {
		return &ai.ModelResponse{
			Message: ai.NewModelTextMessage(text),
			Usage:   &ai.GenerationUsage{InputTokens: inputTokens},
		}
	}
}

func newCCFixture(t *testing.T, reply replyFunc) *ccFixture {
	t.Helper()
	f := &ccFixture{g: genkit.Init(context.Background())}
	f.model = f.defineModel("test/main", reply, &f.requests)
	return f
}

// defineModel defines a model that appends each request to *requests.
func (f *ccFixture) defineModel(name string, reply replyFunc, requests *[]*ai.ModelRequest) ai.Model {
	var mu sync.Mutex
	return genkit.DefineModelAction(f.g, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true, Media: true},
	}, func(ctx context.Context, req *ai.ModelRequest, _ any, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		mu.Lock()
		*requests = append(*requests, req)
		call := len(*requests)
		mu.Unlock()
		resp := reply(call, req)
		resp.Request = req
		return resp, nil
	})
}

// defineSummarizer defines a summarizer model that answers with text and
// records the prompts it receives.
func (f *ccFixture) defineSummarizer(name, text string) (ai.ModelRef, *[]string) {
	var prompts []string
	var requests []*ai.ModelRequest
	f.defineModel(name, func(_ int, req *ai.ModelRequest) *ai.ModelResponse {
		prompts = append(prompts, req.Messages[0].Text())
		return &ai.ModelResponse{Message: ai.NewModelTextMessage(text), FinishReason: ai.FinishReasonStop}
	}, &requests)
	return ai.NewModelRef(name, nil), &prompts
}

// generate runs Generate with mw over msgs and fails t on error.
func (f *ccFixture) generate(t *testing.T, mw *ContextCompression, msgs []*ai.Message, opts ...ai.GenerateOption) *ai.ModelResponse {
	t.Helper()
	opts = append([]ai.GenerateOption{ai.WithModel(f.model), ai.WithMessages(msgs...), ai.WithUse(mw)}, opts...)
	resp, err := genkit.Generate(context.Background(), f.g, opts...)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return resp
}

// sent returns the messages of the newest model request.
func (f *ccFixture) sent(t *testing.T) []*ai.Message {
	t.Helper()
	if len(f.requests) == 0 {
		t.Fatal("the model received no request")
	}
	return f.requests[len(f.requests)-1].Messages
}

func userMsg(text string) *ai.Message   { return ai.NewUserTextMessage(text) }
func modelMsg(text string) *ai.Message  { return ai.NewModelTextMessage(text) }
func systemMsg(text string) *ai.Message { return ai.NewSystemTextMessage(text) }

// toolCallMsg returns a model message requesting the given tool calls.
func toolCallMsg(reqs ...*ai.ToolRequest) *ai.Message {
	msg := &ai.Message{Role: ai.RoleModel}
	for _, r := range reqs {
		msg.Content = append(msg.Content, ai.NewToolRequestPart(r))
	}
	return msg
}

// toolResultMsg returns a tool message carrying the given tool responses.
func toolResultMsg(resps ...*ai.ToolResponse) *ai.Message {
	msg := &ai.Message{Role: ai.RoleTool}
	for _, r := range resps {
		msg.Content = append(msg.Content, ai.NewToolResponsePart(r))
	}
	return msg
}

// stats returns the compression stats on resp, or nil.
func stats(resp *ai.ModelResponse) map[string]any {
	custom, _ := resp.Custom.(map[string]any)
	s, _ := custom[compressionKey].(map[string]any)
	return s
}

// toolMessages returns the tool messages of msgs.
func toolMessages(msgs []*ai.Message) []*ai.Message {
	var out []*ai.Message
	for _, m := range msgs {
		if m.Role == ai.RoleTool {
			out = append(out, m)
		}
	}
	return out
}

// output returns the output of the i-th part of m as a string.
func output(t *testing.T, m *ai.Message, i int) string {
	t.Helper()
	if i >= len(m.Content) || m.Content[i].ToolResponse == nil {
		t.Fatalf("part %d of %v is not a tool response", i, m)
	}
	s, ok := m.Content[i].ToolResponse.Output.(string)
	if !ok {
		t.Fatalf("output of part %d = %T, want string", i, m.Content[i].ToolResponse.Output)
	}
	return s
}

// texts returns the first text of each message, for comparing message lists.
func texts(msgs []*ai.Message) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		switch {
		case len(m.Content) == 0:
		case m.Content[0].IsToolRequest():
			out[i] = string(m.Role) + ":call " + m.Content[0].ToolRequest.Name
		case m.Content[0].IsToolResponse():
			out[i] = string(m.Role) + ":result " + m.Content[0].ToolResponse.Name
		default:
			out[i] = string(m.Role) + ":" + m.Content[0].Text
		}
	}
	return out
}

var (
	truncatedMarker = regexp.MustCompile(`\[Truncated \d+ characters\]`)
	noticePattern   = regexp.MustCompile(`\[NOTE\] Some earlier messages`)
)

func TestContextCompressionSkipsBelowBudget(t *testing.T) {
	f := newCCFixture(t, textReply("response", 50))
	resp := f.generate(t, &ContextCompression{MaxInputTokens: 1000}, []*ai.Message{userMsg("short prompt")})

	if resp.Text() != "response" {
		t.Errorf("text = %q, want %q", resp.Text(), "response")
	}
	if got := len(f.sent(t)); got != 1 {
		t.Errorf("model received %d messages, want 1", got)
	}
	if s := stats(resp); s != nil {
		t.Errorf("stats = %v, want none", s)
	}
}

func TestContextCompressionTriggersOnEstimateBeforeAnyUsage(t *testing.T) {
	f := newCCFixture(t, textReply("response", 50))
	resp := f.generate(t, &ContextCompression{
		MaxInputTokens:        50,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 100, PreserveRecent: -1},
	}, []*ai.Message{
		// Reasoning and data parts count toward the estimate.
		{Role: ai.RoleModel, Content: []*ai.Part{
			ai.NewReasoningPart(strings.Repeat("R", 400), nil),
			ai.NewDataPart(map[string]any{"payload": strings.Repeat("D", 400)}),
		}},
		toolResultMsg(&ai.ToolResponse{Name: "search", Ref: "1", Output: strings.Repeat("X", 500)}),
		userMsg("summarize"),
	})

	s := stats(resp)
	if s["triggered"] != true || s["toolResponsesTruncated"] != 1 {
		t.Errorf("stats = %v, want triggered with 1 truncated response", s)
	}
	if out := output(t, toolMessages(f.sent(t))[0], 0); !truncatedMarker.MatchString(out) {
		t.Errorf("tool output = %q, want a truncation marker", out)
	}
}

func TestContextCompressionTruncatesOlderToolResponsesInToolLoop(t *testing.T) {
	f := newCCFixture(t, func(call int, _ *ai.ModelRequest) *ai.ModelResponse {
		switch call {
		case 1, 2:
			return &ai.ModelResponse{
				Message: toolCallMsg(&ai.ToolRequest{Name: "heavyTool", Input: map[string]any{"query": fmt.Sprintf("call%d", call)}}),
				Usage:   &ai.GenerationUsage{InputTokens: []int{200, 500}[call-1]},
			}
		default:
			return &ai.ModelResponse{Message: modelMsg("finished"), Usage: &ai.GenerationUsage{InputTokens: 100}}
		}
	})
	heavy := genkit.DefineTool(f.g, "heavyTool", "returns large data",
		func(ctx *ai.ToolContext, in struct {
			Query string `json:"query"`
		}) (string, error) {
			return "Result for " + in.Query + ": " + strings.Repeat("X", 300), nil
		})

	resp := f.generate(t, &ContextCompression{
		MaxInputTokens:        150,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 50, PreserveRecent: 1},
	}, []*ai.Message{userMsg("Run tool calls")}, ai.WithTools(heavy))

	if resp.Text() != "finished" || len(f.requests) != 3 {
		t.Fatalf("text = %q after %d calls, want %q after 3", resp.Text(), len(f.requests), "finished")
	}
	tools := toolMessages(f.requests[2].Messages)
	if len(tools) != 2 {
		t.Fatalf("third call received %d tool messages, want 2", len(tools))
	}
	if out := output(t, tools[0], 0); !truncatedMarker.MatchString(out) {
		t.Errorf("older tool output = %q, want truncated", out)
	}
	if out := output(t, tools[1], 0); strings.Contains(out, "[Truncated") {
		t.Errorf("newest tool output = %q, want intact", out)
	}
}

func TestContextCompressionCapsOversizedToolResponses(t *testing.T) {
	f := newCCFixture(t, func(call int, _ *ai.ModelRequest) *ai.ModelResponse {
		if call == 1 {
			return &ai.ModelResponse{Message: toolCallMsg(&ai.ToolRequest{Name: "hugeTool", Input: map[string]any{}}), Usage: &ai.GenerationUsage{InputTokens: 200}}
		}
		return &ai.ModelResponse{Message: modelMsg("done"), Usage: &ai.GenerationUsage{InputTokens: 100}}
	})
	huge := genkit.DefineTool(f.g, "hugeTool", "returns large data",
		func(ctx *ai.ToolContext, in struct{}) (string, error) { return strings.Repeat("x", 1000), nil })

	f.generate(t, &ContextCompression{MaxInputTokens: 100, MaxToolResponseChars: 100},
		[]*ai.Message{userMsg("Call huge tool")}, ai.WithTools(huge))

	if out := output(t, toolMessages(f.requests[1].Messages)[0], 0); !strings.Contains(out, "[TRUNCATED: Response was 1000 chars") {
		t.Errorf("tool output = %q, want the safety cap marker", out)
	}
}

func TestContextCompressionCapsToolResponsesWithoutTokenBudget(t *testing.T) {
	f := newCCFixture(t, textReply("ok", 300))
	resp := f.generate(t, &ContextCompression{MaxToolResponseChars: 200}, []*ai.Message{
		userMsg("run"),
		toolCallMsg(&ai.ToolRequest{Name: "bigTool", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "bigTool", Output: strings.Repeat("Z", 1000)}),
	})

	s := stats(resp)
	if s["toolResponsesSafetyCapped"] != 1 {
		t.Errorf("stats = %v, want 1 capped response", s)
	}
	if n, _ := s["inputTokensBefore"].(int); n <= 0 {
		t.Errorf("inputTokensBefore = %v, want positive", s["inputTokensBefore"])
	}
	if out := output(t, toolMessages(f.sent(t))[0], 0); !strings.Contains(out, "[TRUNCATED: Response was 1000 chars") {
		t.Errorf("tool output = %q, want the safety cap marker", out)
	}
}

func TestContextCompressionMessageCap(t *testing.T) {
	tests := []struct {
		name string
		mw   *ContextCompression
		msgs []*ai.Message
		want []string
	}{
		{
			name: "inserts a standalone notice and starts with a user message",
			mw:   &ContextCompression{MaxMessages: 4},
			msgs: []*ai.Message{userMsg("msg 1"), modelMsg("msg 2"), userMsg("msg 3"), modelMsg("msg 4"), userMsg("msg 5")},
			want: []string{"system:" + defaultTruncationNotice, "user:msg 3", "model:msg 4", "user:msg 5"},
		},
		{
			name: "uses a custom notice",
			mw:   &ContextCompression{MaxMessages: 2, TruncationNotice: "Custom drop notice"},
			msgs: []*ai.Message{userMsg("1"), modelMsg("2"), userMsg("3")},
			want: []string{"system:Custom drop notice", "user:3"},
		},
		{
			name: "keeps system messages",
			mw:   &ContextCompression{MaxMessages: 2, NoTruncationNotice: true},
			msgs: []*ai.Message{systemMsg("System Instructions"), userMsg("msg 1"), modelMsg("msg 2"), userMsg("msg 3")},
			want: []string{"system:System Instructions", "user:msg 3"},
		},
		{
			name: "never keeps an orphaned tool response",
			mw:   &ContextCompression{MaxMessages: 3, NoTruncationNotice: true},
			msgs: []*ai.Message{
				userMsg("hello"),
				toolCallMsg(&ai.ToolRequest{Name: "tool", Input: map[string]any{}}),
				toolResultMsg(&ai.ToolResponse{Name: "tool", Output: "result"}),
				modelMsg("result is ok"),
				userMsg("next"),
			},
			want: []string{"user:next"},
		},
		{
			name: "drops a leading tool message",
			mw:   &ContextCompression{MaxMessages: 2, NoTruncationNotice: true},
			msgs: []*ai.Message{
				userMsg("msg 1"),
				toolCallMsg(&ai.ToolRequest{Name: "myTool", Input: map[string]any{}}),
				toolResultMsg(&ai.ToolResponse{Name: "myTool", Output: "result"}),
				userMsg("msg 2"),
			},
			want: []string{"user:msg 2"},
		},
		{
			name: "keeps nothing but the notice when it takes the only slot",
			mw:   &ContextCompression{MaxMessages: 1},
			msgs: []*ai.Message{userMsg("msg 1"), modelMsg("msg 2")},
			want: []string{"system:" + defaultTruncationNotice},
		},
		{
			name: "counts the system message the notice merges into",
			mw:   &ContextCompression{MaxMessages: 4},
			msgs: []*ai.Message{systemMsg("System prompt"), userMsg("user 1"), modelMsg("model 1"), userMsg("user 2"), modelMsg("model 2"), userMsg("user 3")},
			want: []string{"system:System prompt", "user:user 2", "model:model 2", "user:user 3"},
		},
		{
			name: "merges the notice into the first of several system messages",
			mw:   &ContextCompression{MaxMessages: 3},
			msgs: []*ai.Message{systemMsg("System 1"), systemMsg("System 2"), userMsg("user 1"), modelMsg("model 1"), userMsg("user 2")},
			want: []string{"system:System 1", "system:System 2", "user:user 2"},
		},
		{
			name: "anchors the user message that started a tool loop",
			mw:   &ContextCompression{MaxMessages: 6},
			msgs: []*ai.Message{
				systemMsg("Sys"),
				userMsg("Investigate Project Alpha"),
				toolCallMsg(&ai.ToolRequest{Name: "t1", Input: map[string]any{"step": 1}}),
				toolResultMsg(&ai.ToolResponse{Name: "t1", Output: "r1"}),
				toolCallMsg(&ai.ToolRequest{Name: "t2", Input: map[string]any{"step": 2}}),
				toolResultMsg(&ai.ToolResponse{Name: "t2", Output: "r2"}),
				toolCallMsg(&ai.ToolRequest{Name: "t3", Input: map[string]any{"step": 3}}),
				toolResultMsg(&ai.ToolResponse{Name: "t3", Output: "r3"}),
			},
			want: []string{"system:Sys", "user:Investigate Project Alpha", "model:call t2", "tool:result t2", "model:call t3", "tool:result t3"},
		},
		{
			name: "keeps the newest tool turn when one slot is left beside the anchor",
			mw:   &ContextCompression{MaxMessages: 3},
			msgs: []*ai.Message{
				systemMsg("Sys"),
				userMsg("Original task"),
				toolCallMsg(&ai.ToolRequest{Name: "t1", Input: map[string]any{"step": 1}}),
				toolResultMsg(&ai.ToolResponse{Name: "t1", Output: "r1"}),
				toolCallMsg(&ai.ToolRequest{Name: "t2", Input: map[string]any{"step": 2}}),
				toolResultMsg(&ai.ToolResponse{Name: "t2", Output: "r2"}),
			},
			want: []string{"system:Sys", "user:Original task", "model:call t2", "tool:result t2"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newCCFixture(t, textReply("done", 50))
			f.generate(t, tt.mw, tt.msgs)
			if diff := cmp.Diff(tt.want, texts(f.sent(t))); diff != "" {
				t.Errorf("model received (-want +got):\n%s", diff)
			}
		})
	}
}

func TestContextCompressionNoticeFlagsTheSystemMessage(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	f.generate(t, &ContextCompression{MaxMessages: 4}, []*ai.Message{
		systemMsg("System prompt"), userMsg("user 1"), modelMsg("model 1"), userMsg("user 2"), modelMsg("model 2"), userMsg("user 3"),
	})
	sys := f.sent(t)[0]
	if len(sys.Content) != 2 || !noticePattern.MatchString(sys.Content[1].Text) {
		t.Errorf("system content = %v, want the notice appended", texts([]*ai.Message{{Content: sys.Content[1:]}}))
	}
	if !hasMessageFlag(sys, ccNotice) {
		t.Errorf("system metadata = %v, want the notice flag", sys.Metadata)
	}
}

func TestContextCompressionDoesNotDuplicateNotice(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	notice := "[NOTE] Custom drop notice"
	sys := &ai.Message{
		Role:     ai.RoleSystem,
		Metadata: map[string]any{compressionKey: map[string]any{ccNotice: true}},
		Content:  []*ai.Part{ai.NewTextPart("System prompt"), ai.NewTextPart("\n\n" + notice)},
	}
	f.generate(t, &ContextCompression{MaxMessages: 2, TruncationNotice: notice},
		[]*ai.Message{sys, userMsg("user 1"), modelMsg("model 1"), userMsg("user 2")})
	if got := len(f.sent(t)[0].Content); got != 2 {
		t.Errorf("system message has %d parts, want 2", got)
	}
}

func TestContextCompressionLeavesTruncatedPartsAlone(t *testing.T) {
	already := "12345\n\n[Truncated 95 characters]"
	f := newCCFixture(t, textReply("ok", 500))
	resp := f.generate(t, &ContextCompression{
		MaxInputTokens:        100,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 5, PreserveRecent: -1},
	}, []*ai.Message{
		userMsg("run tool"),
		toolCallMsg(&ai.ToolRequest{Name: "myTool", Input: map[string]any{}}),
		{Role: ai.RoleTool, Content: []*ai.Part{{
			Kind:         ai.PartToolResponse,
			ToolResponse: &ai.ToolResponse{Name: "myTool", Output: already},
			Metadata:     map[string]any{compressionKey: map[string]any{ccTruncated: true}},
		}}},
		userMsg("next question"),
	})
	if out := output(t, toolMessages(f.sent(t))[0], 0); out != already {
		t.Errorf("tool output = %q, want %q", out, already)
	}
	if s := stats(resp); s != nil {
		t.Errorf("stats = %v, want none", s)
	}
}

func TestContextCompressionStampsToolParts(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	history := []*ai.Message{
		userMsg("query"),
		toolCallMsg(&ai.ToolRequest{Name: "toolA", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "toolA", Output: strings.Repeat("A", 500)}),
		userMsg("next"),
	}
	resp := f.generate(t, &ContextCompression{
		MaxInputTokens:        10,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 50, PreserveRecent: -1},
	}, history)

	sentTool := toolMessages(f.sent(t))[0]
	if !hasCompressionFlag(sentTool.Content[0].Metadata, ccTruncated) {
		t.Errorf("sent part metadata = %v, want the truncated flag", sentTool.Content[0].Metadata)
	}
	if sentTool.Metadata != nil {
		t.Errorf("sent message metadata = %v, want none", sentTool.Metadata)
	}

	// The history keeps the original output, flagged for the view.
	histTool := toolMessages(resp.History())[0]
	if out := output(t, histTool, 0); out != strings.Repeat("A", 500) {
		t.Errorf("history output = %.20q..., want the original", out)
	}
	cc := compressionMeta(histTool.Content[0].Metadata)
	if cc[ccTruncated] != true || cc[ccRawOutput] != true || cc[ccMaxChars] != 50 {
		t.Errorf("history part stamp = %v, want truncated, rawOutput, maxChars 50", cc)
	}
}

func TestContextCompressionCappedPartBecomesTruncatable(t *testing.T) {
	mw := &ContextCompression{
		MaxInputTokens:        10,
		MaxToolResponseChars:  200,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 50, PreserveRecent: 1},
	}
	f := newCCFixture(t, textReply("ok", 50))
	f.generate(t, mw, []*ai.Message{
		userMsg("q"),
		toolCallMsg(&ai.ToolRequest{Name: "toolA", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "toolA", Output: strings.Repeat("X", 500)}),
		userMsg("q2"),
	})
	capped := toolMessages(f.sent(t))[0]
	if cc := compressionMeta(capped.Content[0].Metadata); cc[ccCapped] != true || cc[ccTruncated] != nil {
		t.Fatalf("first part stamp = %v, want capped only", cc)
	}

	// Once newer tool output arrives, the capped part is truncated too.
	f.generate(t, mw, []*ai.Message{
		userMsg("q"),
		toolCallMsg(&ai.ToolRequest{Name: "toolA", Input: map[string]any{}}),
		capped,
		userMsg("q2"),
		toolCallMsg(&ai.ToolRequest{Name: "toolB", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "toolB", Output: "recent tool output"}),
		userMsg("q3"),
	})
	part := f.sent(t)[2].Content[0]
	if cc := compressionMeta(part.Metadata); cc[ccTruncated] != true || cc[ccCapped] != true {
		t.Errorf("part stamp = %v, want truncated and capped", cc)
	}
	if out := part.ToolResponse.Output.(string); !strings.HasPrefix(out, strings.Repeat("X", 50)) {
		t.Errorf("output = %q, want 50 X's first", out)
	}
}

func TestContextCompressionPreservesNewestToolMessageWithParallelParts(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	f.generate(t, &ContextCompression{
		MaxInputTokens:        50,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 20, PreserveRecent: 1},
	}, []*ai.Message{
		userMsg("fetch reports"),
		toolCallMsg(&ai.ToolRequest{Name: "fetch", Input: map[string]any{"id": "1"}}, &ai.ToolRequest{Name: "fetch", Input: map[string]any{"id": "2"}}),
		toolResultMsg(
			&ai.ToolResponse{Name: "fetch", Output: "OldA-" + strings.Repeat("X", 200)},
			&ai.ToolResponse{Name: "fetch", Output: "OldB-" + strings.Repeat("X", 200)},
		),
		toolCallMsg(
			&ai.ToolRequest{Name: "fetch", Input: map[string]any{"id": "3"}},
			&ai.ToolRequest{Name: "fetch", Input: map[string]any{"id": "4"}},
			&ai.ToolRequest{Name: "fetch", Input: map[string]any{"id": "5"}},
		),
		toolResultMsg(
			&ai.ToolResponse{Name: "fetch", Output: "New1-" + strings.Repeat("Y", 200)},
			&ai.ToolResponse{Name: "fetch", Output: "New2-" + strings.Repeat("Y", 200)},
			&ai.ToolResponse{Name: "fetch", Output: "New3-" + strings.Repeat("Y", 200)},
		),
	})
	tools := toolMessages(f.sent(t))
	for i := range 2 {
		if out := output(t, tools[0], i); !truncatedMarker.MatchString(out) {
			t.Errorf("older part %d = %q, want truncated", i, out)
		}
	}
	for i := range 3 {
		if out := output(t, tools[1], i); strings.Contains(out, "[Truncated") {
			t.Errorf("newest part %d = %q, want intact", i, out)
		}
	}
}

func TestContextCompressionInputTokensStampTriggersLaterCalls(t *testing.T) {
	mw := &ContextCompression{
		MaxInputTokens:        500,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 20, PreserveRecent: -1},
	}
	f := newCCFixture(t, textReply("reply", 800))
	first := f.generate(t, mw, []*ai.Message{userMsg("hi")})
	if n := compressionMeta(first.Message.Metadata)[ccInputTokens]; n != 800 {
		t.Fatalf("inputTokens stamp = %v, want 800", n)
	}

	// The stamp survives persistence as JSON.
	var stamped *ai.Message
	b, err := json.Marshal(first.Message)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &stamped); err != nil {
		t.Fatal(err)
	}
	second := f.generate(t, mw, []*ai.Message{
		userMsg("hi"),
		toolResultMsg(&ai.ToolResponse{Name: "search", Output: "Short text exceeding 20 chars for truncation test"}),
		stamped,
		userMsg("follow up"),
	})
	s := stats(second)
	if s["triggered"] != true || s["inputTokensBefore"] != 800 {
		t.Errorf("stats = %v, want triggered by 800 input tokens", s)
	}
	if out := output(t, toolMessages(f.sent(t))[0], 0); !truncatedMarker.MatchString(out) {
		t.Errorf("tool output = %q, want truncated", out)
	}
}

func TestContextCompressionCutsWholeCharacters(t *testing.T) {
	f := newCCFixture(t, textReply("ok", 50))
	f.generate(t, &ContextCompression{
		MaxInputTokens:        10,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 5, PreserveRecent: -1},
	}, []*ai.Message{
		userMsg("q"),
		toolResultMsg(&ai.ToolResponse{Name: "emojiTool", Output: "abcd😀" + strings.Repeat("Z", 200)}),
	})
	out := output(t, toolMessages(f.sent(t))[0], 0)
	// Go counts characters as runes, so the emoji is the fifth character.
	if !strings.HasPrefix(out, "abcd😀\n\n[Truncated 200 characters]") || !utf8.ValidString(out) {
		t.Errorf("output = %q, want the first five characters whole", out)
	}
}

func TestContextCompressionDeduplication(t *testing.T) {
	t.Run("replaces older responses to the same call in a tool loop", func(t *testing.T) {
		for _, ref := range []bool{false, true} {
			f := newCCFixture(t, func(call int, _ *ai.ModelRequest) *ai.ModelResponse {
				if call <= 2 {
					req := &ai.ToolRequest{Name: "search", Input: map[string]any{"query": "same-query"}}
					if ref {
						req.Ref = fmt.Sprintf("call_unique_%d", call)
					}
					return &ai.ModelResponse{Message: toolCallMsg(req), Usage: &ai.GenerationUsage{InputTokens: []int{200, 500}[call-1]}}
				}
				return &ai.ModelResponse{Message: modelMsg("finished"), Usage: &ai.GenerationUsage{InputTokens: 100}}
			})
			search := genkit.DefineTool(f.g, "search", "search tool",
				func(ctx *ai.ToolContext, in struct {
					Query string `json:"query"`
				}) (string, error) {
					return "Result for " + in.Query + ": " + strings.Repeat("A", 500), nil
				})
			f.generate(t, &ContextCompression{
				MaxInputTokens:        150,
				DedupeToolResponses:   &CompressionDedupe{MatchBy: CompressionDedupeNameAndInput},
				TruncateToolResponses: &CompressionToolTruncation{MaxChars: 50, PreserveRecent: -1},
			}, []*ai.Message{userMsg("Search multiple times")}, ai.WithTools(search))

			if out := output(t, toolMessages(f.requests[2].Messages)[0], 0); !strings.Contains(out, "Deduplicated") {
				t.Errorf("ref %v: older output = %q, want deduplicated", ref, out)
			}
		}
	})

	t.Run("keeps calls with different inputs", func(t *testing.T) {
		f := newCCFixture(t, func(call int, _ *ai.ModelRequest) *ai.ModelResponse {
			if call <= 2 {
				q := []string{"first-query", "second-query"}[call-1]
				return &ai.ModelResponse{Message: toolCallMsg(&ai.ToolRequest{Name: "search", Input: map[string]any{"query": q}}), Usage: &ai.GenerationUsage{InputTokens: []int{200, 500}[call-1]}}
			}
			return &ai.ModelResponse{Message: modelMsg("finished"), Usage: &ai.GenerationUsage{InputTokens: 100}}
		})
		search := genkit.DefineTool(f.g, "search", "search tool",
			func(ctx *ai.ToolContext, in struct {
				Query string `json:"query"`
			}) (string, error) {
				return "Result for " + in.Query + ": " + strings.Repeat("B", 50), nil
			})
		f.generate(t, &ContextCompression{
			MaxInputTokens:      150,
			DedupeToolResponses: &CompressionDedupe{},
		}, []*ai.Message{userMsg("Search with distinct queries")}, ai.WithTools(search))

		for i, m := range toolMessages(f.requests[2].Messages) {
			if out := output(t, m, 0); strings.Contains(out, "Deduplicated") {
				t.Errorf("tool message %d = %q, want intact", i, out)
			}
		}
	})

	t.Run("replaces only the duplicate among parallel parts", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		f.generate(t, &ContextCompression{MaxInputTokens: 50, DedupeToolResponses: &CompressionDedupe{}}, []*ai.Message{
			userMsg("run parallel tools"),
			toolCallMsg(
				&ai.ToolRequest{Name: "fetch", Ref: "call_1", Input: map[string]any{"id": "shared"}},
				&ai.ToolRequest{Name: "fetch", Ref: "call_2", Input: map[string]any{"id": "unique"}},
			),
			toolResultMsg(
				&ai.ToolResponse{Name: "fetch", Ref: "call_1", Output: "Shared output 1 " + strings.Repeat("X", 200)},
				&ai.ToolResponse{Name: "fetch", Ref: "call_2", Output: "Unique output 2 " + strings.Repeat("Y", 200)},
			),
			toolCallMsg(&ai.ToolRequest{Name: "fetch", Ref: "call_3", Input: map[string]any{"id": "shared"}}),
			toolResultMsg(&ai.ToolResponse{Name: "fetch", Ref: "call_3", Output: "Shared output 3 " + strings.Repeat("Z", 200)}),
		})
		tools := toolMessages(f.sent(t))
		if out := output(t, tools[0], 0); !strings.Contains(out, "Deduplicated") {
			t.Errorf("older shared output = %q, want deduplicated", out)
		}
		if out := output(t, tools[0], 1); !strings.HasPrefix(out, "Unique output 2 ") {
			t.Errorf("unique output = %q, want intact", out)
		}
		if out := output(t, tools[1], 0); !strings.HasPrefix(out, "Shared output 3 ") {
			t.Errorf("newest shared output = %q, want intact", out)
		}
	})

	t.Run("matches refless responses by position past reasoning parts", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		f.generate(t, &ContextCompression{MaxMessages: 2, DedupeToolResponses: &CompressionDedupe{KeepRecent: 1}}, []*ai.Message{
			userMsg("fetch both reports"),
			{Role: ai.RoleModel, Content: []*ai.Part{
				ai.NewReasoningPart("Thinking...", nil),
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "fetchReport", Input: map[string]any{"reportId": "report-101"}}),
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "fetchReport", Input: map[string]any{"reportId": "report-102"}}),
			}},
			toolResultMsg(
				&ai.ToolResponse{Name: "fetchReport", Output: "Report 101 content"},
				&ai.ToolResponse{Name: "fetchReport", Output: "Report 102 content"},
			),
		})
		tool := toolMessages(f.sent(t))[0]
		if a, b := output(t, tool, 0), output(t, tool, 1); a != "Report 101 content" || b != "Report 102 content" {
			t.Errorf("outputs = %q, %q, want both intact", a, b)
		}
	})

	t.Run("matches refless responses across turns", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		f.generate(t, &ContextCompression{MaxInputTokens: 50, DedupeToolResponses: &CompressionDedupe{KeepRecent: 1}}, []*ai.Message{
			userMsg("fetch reports"),
			{Role: ai.RoleModel, Content: []*ai.Part{
				ai.NewReasoningPart("First turn reasoning", nil),
				ai.NewTextPart("Fetching 101 and 102"),
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "fetchReport", Input: map[string]any{"reportId": "report-101"}}),
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "fetchReport", Input: map[string]any{"reportId": "report-102"}}),
			}},
			toolResultMsg(
				&ai.ToolResponse{Name: "fetchReport", Output: "Report 101 v1 " + strings.Repeat("X", 100)},
				&ai.ToolResponse{Name: "fetchReport", Output: "Report 102 v1 " + strings.Repeat("Y", 100)},
			),
			{Role: ai.RoleModel, Content: []*ai.Part{
				ai.NewReasoningPart("Second turn reasoning", nil),
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "fetchReport", Input: map[string]any{"reportId": "report-102"}}),
			}},
			toolResultMsg(&ai.ToolResponse{Name: "fetchReport", Output: "Report 102 v2 " + strings.Repeat("Z", 100)}),
		})
		tools := toolMessages(f.sent(t))
		if out := output(t, tools[0], 0); !strings.HasPrefix(out, "Report 101 v1 ") {
			t.Errorf("report 101 = %q, want intact", out)
		}
		if out := output(t, tools[0], 1); !strings.Contains(out, "Deduplicated") {
			t.Errorf("older report 102 = %q, want deduplicated", out)
		}
		if out := output(t, tools[1], 0); !strings.HasPrefix(out, "Report 102 v2 ") {
			t.Errorf("newest report 102 = %q, want intact", out)
		}
	})

	t.Run("does not match reused refs, mixed ordering, or orphans", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		f.generate(t, &ContextCompression{MaxInputTokens: 20, DedupeToolResponses: &CompressionDedupe{KeepRecent: 1}}, []*ai.Message{
			userMsg("run"),
			toolResultMsg(&ai.ToolResponse{Name: "orphan", Output: "Orphan 1"}, &ai.ToolResponse{Name: "orphan", Output: "Orphan 2"}),
			toolCallMsg(
				&ai.ToolRequest{Name: "fetchReport", Ref: "0", Input: map[string]any{"id": "a"}},
				&ai.ToolRequest{Name: "fetchReport", Input: map[string]any{"id": "b"}},
			),
			toolResultMsg(
				&ai.ToolResponse{Name: "fetchReport", Output: "Out B"},
				&ai.ToolResponse{Name: "fetchReport", Ref: "0", Output: "Out A"},
			),
			toolCallMsg(&ai.ToolRequest{Name: "fetchReport", Ref: "0", Input: map[string]any{"id": "c"}}),
			toolResultMsg(&ai.ToolResponse{Name: "fetchReport", Ref: "0", Output: "Out C " + strings.Repeat("X", 200)}),
		})
		tools := toolMessages(f.sent(t))
		want := [][]string{{"Orphan 1", "Orphan 2"}, {"Out B", "Out A"}}
		for i, outs := range want {
			for j, w := range outs {
				if out := output(t, tools[i], j); out != w {
					t.Errorf("tool message %d part %d = %q, want %q", i, j, out, w)
				}
			}
		}
	})

	t.Run("drops multipart content with the output", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		newMedia := ai.NewMediaPart("image/png", "data:image/png;base64,NEW_SCREENSHOT")
		f.generate(t, &ContextCompression{MaxInputTokens: 50, DedupeToolResponses: &CompressionDedupe{}}, []*ai.Message{
			userMsg("take screenshots"),
			toolCallMsg(&ai.ToolRequest{Name: "screenshot", Ref: "shot_1", Input: map[string]any{"page": "home"}}),
			toolResultMsg(&ai.ToolResponse{
				Name: "screenshot", Ref: "shot_1",
				Output:  "Captured screenshot 1 " + strings.Repeat("X", 200),
				Content: []*ai.Part{ai.NewMediaPart("image/png", "data:image/png;base64,OLD_SCREENSHOT")},
			}),
			toolCallMsg(&ai.ToolRequest{Name: "screenshot", Ref: "shot_2", Input: map[string]any{"page": "home"}}),
			toolResultMsg(&ai.ToolResponse{
				Name: "screenshot", Ref: "shot_2",
				Output:  "Captured screenshot 2 " + strings.Repeat("Y", 200),
				Content: []*ai.Part{newMedia},
			}),
		})
		tools := toolMessages(f.sent(t))
		if out := output(t, tools[0], 0); !strings.Contains(out, "Deduplicated") || tools[0].Content[0].ToolResponse.Content != nil {
			t.Errorf("older response = %q with content %v, want deduplicated without content", out, tools[0].Content[0].ToolResponse.Content)
		}
		if newest := tools[1].Content[0].ToolResponse; len(newest.Content) != 1 || newest.Content[0] != newMedia {
			t.Errorf("newest content = %v, want the media part intact", newest.Content)
		}
	})

	t.Run("never replaces a sole response", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		f.generate(t, &ContextCompression{MaxInputTokens: 10, DedupeToolResponses: &CompressionDedupe{}}, []*ai.Message{
			userMsg("check single tool call"),
			toolCallMsg(&ai.ToolRequest{Name: "fetch", Ref: "call_1", Input: map[string]any{"id": "1"}}),
			toolResultMsg(&ai.ToolResponse{Name: "fetch", Ref: "call_1", Output: "Only response " + strings.Repeat("X", 200)}),
		})
		if out := output(t, toolMessages(f.sent(t))[0], 0); !strings.HasPrefix(out, "Only response ") {
			t.Errorf("output = %q, want intact", out)
		}
	})
}

func TestContextCompressionSummarizes(t *testing.T) {
	f := newCCFixture(t, func(call int, _ *ai.ModelRequest) *ai.ModelResponse {
		if call <= 3 {
			return &ai.ModelResponse{Message: toolCallMsg(&ai.ToolRequest{Name: "step", Input: map[string]any{"step": call}}), Usage: &ai.GenerationUsage{InputTokens: 500}}
		}
		return &ai.ModelResponse{Message: modelMsg("all done"), Usage: &ai.GenerationUsage{InputTokens: 200}}
	})
	summarizer, _ := f.defineSummarizer("test/summarizer", "Summary of past events: steps were executed.")
	step := genkit.DefineTool(f.g, "step", "step",
		func(ctx *ai.ToolContext, in struct {
			Step int `json:"step"`
		}) (string, error) {
			return fmt.Sprintf("output %d", in.Step), nil
		})

	resp := f.generate(t, &ContextCompression{
		MaxInputTokens: 100,
		Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 2},
	}, []*ai.Message{systemMsg("System instructions"), userMsg("Run multi turn steps")}, ai.WithTools(step))

	if resp.Text() != "all done" {
		t.Errorf("text = %q, want %q", resp.Text(), "all done")
	}
	found := false
	for _, m := range f.sent(t) {
		found = found || strings.Contains(m.Text(), "Summary of past events")
	}
	if !found {
		t.Errorf("final call received %v, want the summary", texts(f.sent(t)))
	}
}

func TestContextCompressionCustomSummarizationPrompt(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	summarizer, prompts := f.defineSummarizer("test/summarizer", "Custom summary result")
	f.generate(t, &ContextCompression{
		MaxInputTokens: 10,
		Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 1, Prompt: "TLDR THIS: {conversation}\nEND TLDR"},
	}, []*ai.Message{
		userMsg("A long discussion part 1"), modelMsg("A long discussion response 1"),
		userMsg("A long discussion part 2"), modelMsg("A long discussion response 2"),
	})
	if len(*prompts) != 1 {
		t.Fatalf("summarizer called %d times, want 1", len(*prompts))
	}
	p := (*prompts)[0]
	if !strings.HasPrefix(p, "TLDR THIS:") || !strings.Contains(p, "A long discussion part 1") || !strings.HasSuffix(p, "END TLDR") {
		t.Errorf("prompt = %q, want the template filled in", p)
	}
}

func TestContextCompressionSkipSummarizationThreshold(t *testing.T) {
	history := []*ai.Message{
		toolResultMsg(&ai.ToolResponse{Name: "huge", Ref: "1", Output: strings.Repeat("X", 2000)}),
		userMsg("msg 2"), modelMsg("msg 3"), userMsg("msg 4"),
	}
	f := newCCFixture(t, textReply("done", 50))
	summarizer, prompts := f.defineSummarizer("test/summarizer", "Summary")
	resp := f.generate(t, &ContextCompression{
		MaxInputTokens:             100,
		TruncateToolResponses:      &CompressionToolTruncation{MaxChars: 100, PreserveRecent: -1},
		Summarize:                  &CompressionSummarizer{Model: summarizer, PreserveRecent: 1},
		SkipSummarizationThreshold: 0.25,
	}, history)
	if len(*prompts) != 0 {
		t.Errorf("summarizer called %d times, want 0", len(*prompts))
	}
	if s := stats(resp); s["summarizationSkipped"] != true {
		t.Errorf("stats = %v, want summarization skipped", s)
	}

	// Savings over the threshold do not skip summarization while the
	// context stays over budget.
	f = newCCFixture(t, textReply("done", 50))
	summarizer, prompts = f.defineSummarizer("test/summarizer", "Summary")
	resp = f.generate(t, &ContextCompression{
		MaxInputTokens:             150,
		TruncateToolResponses:      &CompressionToolTruncation{MaxChars: 100, PreserveRecent: -1},
		Summarize:                  &CompressionSummarizer{Model: summarizer, PreserveRecent: 1},
		SkipSummarizationThreshold: 0.25,
	}, []*ai.Message{
		toolResultMsg(&ai.ToolResponse{Name: "heavy", Ref: "1", Output: strings.Repeat("T", 700)}),
		userMsg(strings.Repeat("U", 350)), modelMsg(strings.Repeat("M", 350)), userMsg("latest question"),
	})
	if len(*prompts) != 1 {
		t.Errorf("summarizer called %d times, want 1", len(*prompts))
	}
	if s := stats(resp); s["summarized"] != true || s["summarizationSkipped"] != false {
		t.Errorf("stats = %v, want summarized and not skipped", s)
	}
}

func TestContextCompressionSkippedSummaryKeepsMessages(t *testing.T) {
	history := []*ai.Message{
		userMsg("u1"),
		toolCallMsg(&ai.ToolRequest{Name: "bigTool", Ref: "t1", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "bigTool", Ref: "t1", Output: strings.Repeat("X", 3000)}),
		modelMsg("m1"), userMsg("u2"), modelMsg("m2"), userMsg("u3"),
	}
	f := newCCFixture(t, textReply("done", 50))
	summarizer, prompts := f.defineSummarizer("test/summarizer", "Summary")
	resp := f.generate(t, &ContextCompression{
		MaxInputTokens:             500,
		PreserveRecent:             2,
		TruncateToolResponses:      &CompressionToolTruncation{MaxChars: 50, PreserveRecent: -1},
		Summarize:                  &CompressionSummarizer{Model: summarizer},
		SkipSummarizationThreshold: 0.3,
	}, history)
	s := stats(resp)
	if len(*prompts) != 0 || s["summarizationSkipped"] != true || s["toolResponsesTruncated"] != 1 ||
		s["messagesAfter"] != 7 || s["truncationNoticeInserted"] != false || len(f.sent(t)) != 7 {
		t.Errorf("summarizer calls = %d, stats = %v, sent %d messages; want no call, 7 messages kept", len(*prompts), s, len(f.sent(t)))
	}

	// Without a summarizer, cheap truncation that meets the budget keeps the
	// messages too.
	resp = f.generate(t, &ContextCompression{
		MaxInputTokens:        500,
		PreserveRecent:        2,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 50, PreserveRecent: -1},
	}, history)
	if s := stats(resp); s["toolResponsesTruncated"] != 1 || s["messagesAfter"] != 7 || len(f.sent(t)) != 7 {
		t.Errorf("stats = %v, sent %d messages; want 7 kept", s, len(f.sent(t)))
	}

	// A stamped token count that stays over budget after scaling does not
	// skip summarization.
	f.generate(t, &ContextCompression{
		MaxInputTokens:             500,
		PreserveRecent:             2,
		TruncateToolResponses:      &CompressionToolTruncation{MaxChars: 50, PreserveRecent: -1},
		Summarize:                  &CompressionSummarizer{Model: summarizer, PreserveRecent: 2},
		SkipSummarizationThreshold: 0.3,
	}, []*ai.Message{
		userMsg("u1"),
		toolCallMsg(&ai.ToolRequest{Name: "bigTool", Ref: "t1", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "bigTool", Ref: "t1", Output: strings.Repeat("X", 300)}),
		modelMsg(strings.Repeat("Y", 300)),
		userMsg("u2"),
		{Role: ai.RoleModel, Metadata: map[string]any{compressionKey: map[string]any{ccInputTokens: 1000}}, Content: []*ai.Part{ai.NewTextPart("m2")}},
		userMsg("u3"),
	})
	if len(*prompts) != 1 {
		t.Errorf("summarizer called %d times, want 1", len(*prompts))
	}
}

func TestContextCompressionShrinksWindowsOnOvershoot(t *testing.T) {
	var history []*ai.Message
	for i := range 10 {
		text := fmt.Sprintf("Message number %d: %s", i, strings.Repeat("X", 300))
		if i%2 == 0 {
			history = append(history, userMsg(text))
		} else {
			history = append(history, modelMsg(text))
		}
	}
	f := newCCFixture(t, textReply("done", 50))
	f.generate(t, &ContextCompression{MaxInputTokens: 50, MaxMessages: 6, PreserveRecent: 4, NoTruncationNotice: true}, history)
	if got := len(f.sent(t)); got != 2 {
		t.Errorf("model received %d messages, want 2", got)
	}
}

// withCallerMetadata gives m and its parts metadata of their own, so a stamp
// written in place would show up in the caller's messages.
func withCallerMetadata(m *ai.Message) *ai.Message {
	m.Metadata = map[string]any{"source": "caller", compressionKey: map[string]any{"clientKey": true}}
	for _, p := range m.Content {
		p.Metadata = map[string]any{"source": "caller"}
	}
	return m
}

func TestContextCompressionKeepsHistory(t *testing.T) {
	long := strings.Repeat("TOOL_OUTPUT_", 50)
	history := []*ai.Message{
		withCallerMetadata(userMsg("earlier question")),
		withCallerMetadata(modelMsg("earlier answer")),
		withCallerMetadata(userMsg("run heavy tool")),
		withCallerMetadata(toolCallMsg(&ai.ToolRequest{Name: "heavyTool", Input: map[string]any{}})),
		withCallerMetadata(toolResultMsg(&ai.ToolResponse{Name: "heavyTool", Output: long})),
		withCallerMetadata(modelMsg("looked at it")),
		withCallerMetadata(userMsg("and now?")),
	}
	before, err := json.Marshal(history)
	if err != nil {
		t.Fatal(err)
	}
	f := newCCFixture(t, textReply("done", 50))
	resp := f.generate(t, &ContextCompression{
		MaxMessages:           6,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 10, PreserveRecent: -1},
	}, history)

	sent := f.sent(t)
	if got := texts(sent); len(got) != 6 || got[1] != "user:run heavy tool" {
		t.Fatalf("model received %v, want the notice and the 5 newest messages", got)
	}
	if out := output(t, sent[3], 0); !strings.Contains(out, "[Truncated ") {
		t.Errorf("model received %q, want truncated", out)
	}
	// The model echoed the view, but the response carries the full messages,
	// stamped.
	full := resp.History()
	if len(full) != 8 || output(t, full[4], 0) != long {
		t.Fatalf("history = %v, want the 7 original messages and the reply", texts(full))
	}
	if !hasCompressionFlag(full[4].Content[0].Metadata, ccRawOutput) || compressionMeta(full[1].Metadata)[ccStats] == nil {
		t.Errorf("history stamps: part %v, boundary %v; want the part edit and a boundary", full[4].Content[0].Metadata, full[1].Metadata)
	}
	if diff := cmp.Diff(texts(sent), texts(ResolveCompressedHistory(full[:7]))); diff != "" {
		t.Errorf("resolved history differs from the model's view (-sent +resolved):\n%s", diff)
	}
	// The caller's messages are never modified.
	after, err := json.Marshal(history)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Errorf("history modified:\nbefore %s\nafter  %s", before, after)
	}
}

func TestContextCompressionStampsSummaryBoundary(t *testing.T) {
	longPrompt := strings.Repeat("original user research prompt ", 20)
	f := newCCFixture(t, textReply("done all", 50))
	summarizer, _ := f.defineSummarizer("test/summarizer", "MOCK_SUMMARY_TEXT")
	resp := f.generate(t, &ContextCompression{
		MaxInputTokens: 100,
		Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 1},
	}, []*ai.Message{
		userMsg(longPrompt),
		toolCallMsg(&ai.ToolRequest{Name: "tool1", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "tool1", Output: "TOOL_1_RESULT"}),
		userMsg("followup question"),
	})

	if s := stats(resp); s["summarized"] != true {
		t.Errorf("stats = %v, want summarized", s)
	}
	sent := f.sent(t)
	if len(sent) != 2 || !strings.Contains(sent[0].Text(), "MOCK_SUMMARY_TEXT") || sent[1].Text() != "followup question" {
		t.Errorf("model received %v, want the summary and the followup", texts(sent))
	}
	req := resp.Request.Messages
	if len(req) != 4 || req[0].Text() != longPrompt {
		t.Fatalf("request messages = %v, want the 4 originals", texts(req))
	}
	cc := compressionMeta(req[2].Metadata)
	if cc[ccSummary] != "MOCK_SUMMARY_TEXT" || cc[ccStats] == nil {
		t.Errorf("boundary stamp = %v, want the summary and stats", cc)
	}
	if got := len(ResolveCompressedHistory(req)); got != 2 {
		t.Errorf("resolved %d messages, want 2", got)
	}
}

func TestContextCompressionReplaceHistory(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	resp := f.generate(t, &ContextCompression{
		MaxInputTokens:     50,
		ReplaceHistory:     true,
		MaxMessages:        1,
		NoTruncationNotice: true,
	}, []*ai.Message{userMsg(strings.Repeat("A", 500)), modelMsg("response 1"), userMsg("user 2")})

	if s := stats(resp); s["triggered"] != true {
		t.Errorf("stats = %v, want triggered", s)
	}
	if got := texts(f.sent(t)); len(got) != 1 {
		t.Errorf("model received %v, want 1 message", got)
	}
	if got := texts(resp.Request.Messages); len(got) != 1 || got[0] != "user:user 2" {
		t.Errorf("request messages = %v, want only the kept message", got)
	}
}

func TestContextCompressionNewestSummaryWins(t *testing.T) {
	f := newCCFixture(t, textReply("done", 500))
	count := 0
	var requests []*ai.ModelRequest
	f.defineModel("test/summarizer", func(int, *ai.ModelRequest) *ai.ModelResponse {
		count++
		return &ai.ModelResponse{Message: modelMsg(fmt.Sprintf("SUMMARY_%d", count)), FinishReason: ai.FinishReasonStop}
	}, &requests)
	mw := &ContextCompression{
		MaxInputTokens: 50,
		Summarize:      &CompressionSummarizer{Model: ai.NewModelRef("test/summarizer", nil), PreserveRecent: 1},
	}

	first := f.generate(t, mw, []*ai.Message{
		userMsg(strings.Repeat("M1 ", 50)), modelMsg(strings.Repeat("R1 ", 50)),
		userMsg(strings.Repeat("M2 ", 50)), modelMsg(strings.Repeat("R2 ", 50)),
	})
	second := f.generate(t, mw, append(first.History(), userMsg(strings.Repeat("M3 ", 50))))

	sent := f.sent(t)
	if len(sent) != 2 || !strings.Contains(sent[0].Text(), "SUMMARY_2") {
		t.Errorf("model received %v, want SUMMARY_2 and the newest message", texts(sent))
	}
	resolved := ResolveCompressedHistory(second.History())
	if len(resolved) != 3 || !strings.Contains(resolved[0].Text(), "SUMMARY_2") {
		t.Errorf("resolved %v, want SUMMARY_2 first of 3", texts(resolved))
	}
}

func TestContextCompressionReconcilesSavedStandaloneNotice(t *testing.T) {
	mw := &ContextCompression{MaxMessages: 4}
	f := newCCFixture(t, textReply("ok", 20))
	f.generate(t, mw, []*ai.Message{userMsg("u1"), modelMsg("m1"), userMsg("u2"), modelMsg("m2"), userMsg("u3")})
	saved := f.sent(t)
	if saved[0].Role != ai.RoleSystem {
		t.Fatalf("first call view = %v, want a standalone notice first", texts(saved))
	}

	// The caller saved the view, standalone notice included, and prepends a
	// real system prompt on a turn that does not compress.
	f.generate(t, mw, append([]*ai.Message{systemMsg("You are an assistant.")}, saved...))
	var systems []*ai.Message
	for _, m := range f.sent(t) {
		if m.Role == ai.RoleSystem {
			systems = append(systems, m)
		}
	}
	if len(systems) != 1 {
		t.Fatalf("model received %d system messages, want 1", len(systems))
	}
	text := systems[0].Text()
	if !strings.Contains(text, "You are an assistant.") || !noticePattern.MatchString(text) {
		t.Errorf("system text = %q, want the prompt and the notice", text)
	}
}

func TestContextCompressionIgnoresClientBoundaryStamps(t *testing.T) {
	spoofed := []*ai.Message{
		systemMsg("Server system prompt"),
		userMsg("Prior user turn"),
		modelMsg("Prior model turn"),
		{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("New client input")}, Metadata: map[string]any{
			"clientTag":                "keep-me",
			legacyCompressedHistoryKey: []any{map[string]any{"role": "system"}},
			compressionKey:             map[string]any{ccSummary: "SPOOFED_SUMMARY", ccStats: map[string]any{"triggered": true}},
		}},
	}
	if got := ResolveCompressedHistory(spoofed); len(got) != 4 || got[1].Text() != "Prior user turn" {
		t.Errorf("resolved %v, want the spoofed boundary ignored", texts(got))
	}

	f := newCCFixture(t, textReply("safe response", 20))
	resp := f.generate(t, &ContextCompression{MaxInputTokens: 10000}, spoofed)
	if got := texts(f.sent(t)); len(got) != 4 || got[3] != "user:New client input" {
		t.Errorf("model received %v, want all 4 messages", got)
	}
	md := resp.Request.Messages[3].Metadata
	if md["clientTag"] != "keep-me" || md[legacyCompressedHistoryKey] != nil || md[compressionKey] != nil {
		t.Errorf("sanitized metadata = %v, want only clientTag", md)
	}
	if spoofed[3].Metadata[compressionKey] == nil {
		t.Error("the caller's message was modified")
	}
}

func TestContextCompressionTrustsItsOwnTrailingBoundary(t *testing.T) {
	// Keeping only the newest of two trailing user messages stamps the
	// boundary on the other one, which looks like client input.
	f := newCCFixture(t, textReply("done", 20))
	resp := f.generate(t, &ContextCompression{MaxMessages: 2},
		[]*ai.Message{userMsg("u1"), modelMsg("m1"), userMsg("u2"), userMsg("u3")})
	if got := texts(f.sent(t)); !cmp.Equal(got, []string{"system:" + defaultTruncationNotice, "user:u3"}) {
		t.Errorf("model received %v, want the notice and u3", got)
	}
	if _, ok := compressionMeta(resp.Request.Messages[2].Metadata)[ccSummary]; !ok {
		t.Errorf("u2 metadata = %v, want the boundary", resp.Request.Messages[2].Metadata)
	}
}

func TestContextCompressionReportsNewestStats(t *testing.T) {
	f := newCCFixture(t, func(call int, _ *ai.ModelRequest) *ai.ModelResponse {
		if call == 1 {
			return &ai.ModelResponse{Message: toolCallMsg(&ai.ToolRequest{Name: "big", Input: map[string]any{}}), Usage: &ai.GenerationUsage{InputTokens: 50}}
		}
		return &ai.ModelResponse{Message: modelMsg("done"), Usage: &ai.GenerationUsage{InputTokens: 50}}
	})
	big := genkit.DefineTool(f.g, "big", "returns large data", func(ctx *ai.ToolContext, in struct{}) (string, error) {
		return strings.Repeat("N", 300), nil
	})
	// Both iterations exceed the budget and truncate a tool response.
	resp := f.generate(t, &ContextCompression{
		MaxInputTokens:        20,
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 10, PreserveRecent: -1},
	}, []*ai.Message{
		userMsg("go"),
		toolCallMsg(&ai.ToolRequest{Name: "big", Ref: "old", Input: map[string]any{"page": 1}}),
		toolResultMsg(&ai.ToolResponse{Name: "big", Ref: "old", Output: strings.Repeat("O", 300)}),
	}, ai.WithTools(big))
	if s := stats(resp); s["toolResponsesTruncated"] != 2 {
		t.Errorf("stats = %v, want 2 truncated responses across both iterations", s)
	}
}

func TestContextCompressionKeepsTailLimitsAfterSummary(t *testing.T) {
	oversized := strings.Repeat("X", 500)
	f := newCCFixture(t, textReply("done", 40))
	summarizer, _ := f.defineSummarizer("test/summarizer", "SUMMARIZED_PREFIX")
	first := f.generate(t, &ContextCompression{
		MaxInputTokens:        350,
		MaxToolResponseChars:  100,
		DedupeToolResponses:   &CompressionDedupe{KeepRecent: 1},
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 30, PreserveRecent: 1},
		Summarize:             &CompressionSummarizer{Model: summarizer, PreserveRecent: 6},
	}, []*ai.Message{
		userMsg(strings.Repeat("Old user prompt ", 20)),
		modelMsg(strings.Repeat("Old model reply ", 20)),
		toolCallMsg(&ai.ToolRequest{Name: "dupTool", Input: map[string]any{"q": 1}}),
		toolResultMsg(&ai.ToolResponse{Name: "dupTool", Output: "DUP_OUTPUT_1"}),
		toolCallMsg(&ai.ToolRequest{Name: "dupTool", Input: map[string]any{"q": 1}}),
		toolResultMsg(&ai.ToolResponse{Name: "dupTool", Output: strings.Repeat("Y", 200)}),
		toolCallMsg(&ai.ToolRequest{Name: "bigTool", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "bigTool", Output: oversized}),
	})

	sent := f.sent(t)
	if len(sent) != 7 || !strings.Contains(sent[0].Text(), "SUMMARIZED_PREFIX") {
		t.Fatalf("model received %v, want the summary and 6 messages", texts(sent))
	}
	for i, want := range map[int]string{2: "[Deduplicated:", 4: "[Truncated ", 6: "[TRUNCATED: Response was 500 chars"} {
		if out := output(t, sent[i], 0); !strings.Contains(out, want) {
			t.Errorf("message %d output = %q, want %q", i, out, want)
		}
	}
	if out := output(t, first.History()[7], 0); out != oversized {
		t.Errorf("history output = %.20q..., want the original", out)
	}

	// The capped stamp keeps an oversized part from triggering compression
	// again, while the model still receives it capped.
	second := f.generate(t, &ContextCompression{MaxInputTokens: 100000, MaxToolResponseChars: 100},
		append(first.History(), userMsg("Next turn question")))
	if s := stats(second); s != nil {
		t.Errorf("stats = %v, want none", s)
	}
	for _, m := range toolMessages(f.sent(t)) {
		if m.Content[0].ToolResponse.Name == "bigTool" && !strings.Contains(output(t, m, 0), "[TRUNCATED: Response was 500 chars") {
			t.Errorf("bigTool output = %q, want capped", output(t, m, 0))
		}
	}
}

func TestContextCompressionKeepsFreshSystemPrompt(t *testing.T) {
	f := newCCFixture(t, textReply("ok", 30))
	summarizer, _ := f.defineSummarizer("test/summarizer", "SUMMARY_OF_TURN_1")
	first := f.generate(t, &ContextCompression{
		MaxInputTokens: 50,
		Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 1},
	}, []*ai.Message{
		systemMsg("System state: Date is Monday"),
		userMsg(strings.Repeat("User turn 1 ", 30)),
		modelMsg(strings.Repeat("Model turn 1 ", 30)),
		userMsg("User turn 2"),
	})

	// The caller renders a fresh system prompt in front of the history.
	next := append([]*ai.Message{systemMsg("System state: Date is Tuesday")}, first.History()[1:]...)
	next = append(next, userMsg("What day is it?"))
	f.generate(t, &ContextCompression{
		MaxInputTokens: 10000,
		Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 1},
	}, next)
	sent := f.sent(t)
	if sent[0].Text() != "System state: Date is Tuesday" || !strings.Contains(sent[1].Text(), "SUMMARY_OF_TURN_1") {
		t.Errorf("model received %v, want the fresh system prompt then the summary", texts(sent))
	}
}

func TestContextCompressionBoundaryNeverMovesBack(t *testing.T) {
	mw := &ContextCompression{MaxMessages: 4}
	f := newCCFixture(t, textReply("final answer", 30))
	first := f.generate(t, mw, []*ai.Message{
		userMsg("Initial task prompt"),
		toolCallMsg(&ai.ToolRequest{Name: "step1", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "step1", Output: "DROPPED_STEP_1"}),
		toolCallMsg(&ai.ToolRequest{Name: "step2", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "step2", Output: "DROPPED_STEP_2"}),
		toolCallMsg(&ai.ToolRequest{Name: "step3", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "step3", Output: "STEP_3_ON_TURN_1"}),
	})
	second := f.generate(t, mw, append(slices.Clone(first.Request.Messages),
		toolCallMsg(&ai.ToolRequest{Name: "step4", Input: map[string]any{}}),
		toolResultMsg(&ai.ToolResponse{Name: "step4", Output: "LATEST_STEP_4"}),
	))

	check := func(label string, msgs []*ai.Message) {
		for _, m := range msgs {
			b, _ := json.Marshal(m.Content)
			for _, dropped := range []string{"DROPPED_STEP_1", "DROPPED_STEP_2", "STEP_3_ON_TURN_1"} {
				if strings.Contains(string(b), dropped) {
					t.Errorf("%s holds %s", label, dropped)
				}
			}
		}
	}
	check("the model's view", f.sent(t))
	resolved := ResolveCompressedHistory(second.History())
	check("the resolved history", resolved)
	if resolved[1].Text() != "Initial task prompt" {
		t.Errorf("resolved %v, want the initial prompt anchored", texts(resolved))
	}
	found := false
	for _, m := range toolMessages(resolved) {
		found = found || output(t, m, 0) == "LATEST_STEP_4"
	}
	if !found {
		t.Errorf("resolved %v, want the latest step", texts(resolved))
	}
}

func TestContextCompressionSummarizerRendering(t *testing.T) {
	payload := "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo="
	f := newCCFixture(t, textReply("ok", 50))
	summarizer, prompts := f.defineSummarizer("test/summarizer", "Rich summary")
	f.generate(t, &ContextCompression{
		MaxInputTokens: 20,
		Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 1},
	}, []*ai.Message{
		ai.NewUserMessage(
			ai.NewTextPart("Check these attachments "+strings.Repeat("X", 200)),
			&ai.Part{Kind: ai.PartMedia, Text: "data:image/png;base64," + payload},
			ai.NewMediaPart("application/pdf", "https://example.com/spec.pdf"),
		),
		{Role: ai.RoleModel, Content: []*ai.Part{
			ai.NewReasoningPart("Thinking through the spec carefully", nil),
			ai.NewDataPart(map[string]any{"status": "analyzed"}),
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "inspect", Input: map[string]any{"target": "spec"}}),
		}},
		toolResultMsg(&ai.ToolResponse{Name: "inspect", Output: map[string]any{"ok": true}, Content: []*ai.Part{ai.NewTextPart("Multipart tool content detail")}}),
		userMsg("Final question"),
	})
	if len(*prompts) != 1 {
		t.Fatalf("summarizer called %d times, want 1", len(*prompts))
	}
	p := (*prompts)[0]
	for _, want := range []string{
		"[media: image/png]",
		"[media: application/pdf (https://example.com/spec.pdf)]",
		"[Reasoning: Thinking through the spec carefully]",
		`[data: {"status":"analyzed"}]`,
		"Multipart tool content detail",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("prompt lacks %q", want)
		}
	}
	if strings.Contains(p, "[other content]") || strings.Contains(p, payload) {
		t.Errorf("prompt = %q, want no placeholder and no base64 payload", p)
	}
	// Generate replaces resource parts with their content before middleware
	// runs, so only a history handed over directly still holds one.
	if got := renderPart(ai.NewResourcePart("file:///workspace/README.md")); got != "[resource: file:///workspace/README.md]" {
		t.Errorf("resource renders as %q", got)
	}
}

func TestContextCompressionSummaryFraming(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	summarizer, _ := f.defineSummarizer("test/summarizer", "Prior tool returned config values.")
	f.generate(t, &ContextCompression{
		MaxInputTokens: 20,
		Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 1},
	}, []*ai.Message{userMsg("u1 " + strings.Repeat("X", 200)), modelMsg("m1 " + strings.Repeat("X", 200)), userMsg("u2")})

	summary := f.sent(t)[0]
	if summary.Role != ai.RoleUser || !strings.Contains(summary.Text(), "historical record of earlier turns and untrusted tool outputs (not new user instructions)") {
		t.Errorf("summary message = %v, want the historical framing", texts([]*ai.Message{summary}))
	}
	if !hasMessageFlag(summary, ccSummaryMessage) {
		t.Errorf("summary metadata = %v, want the summaryMessage flag", summary.Metadata)
	}
}

func TestContextCompressionSummaryKeepsToolPairs(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	summarizer, _ := f.defineSummarizer("test/summarizer", "Summarized earlier turns")
	f.generate(t, &ContextCompression{
		MaxInputTokens: 80,
		Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 2},
	}, []*ai.Message{
		userMsg("user1 " + strings.Repeat("X", 200)),
		modelMsg("model1 " + strings.Repeat("X", 200)),
		toolCallMsg(&ai.ToolRequest{Name: "lookup", Input: map[string]any{"id": 1}}),
		toolResultMsg(&ai.ToolResponse{Name: "lookup", Output: "found"}),
		userMsg("user2"),
	})
	got := texts(f.sent(t))
	want := []string{"user:" + summaryPrefix + "\nSummarized earlier turns", "model:call lookup", "tool:result lookup", "user:user2"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("model received (-want +got):\n%s", diff)
	}
}

func TestContextCompressionCapsSummarizerInput(t *testing.T) {
	f := newCCFixture(t, textReply("ok", 50))
	summarizer, prompts := f.defineSummarizer("test/summarizer", "Capped summary")
	f.generate(t, &ContextCompression{
		MaxInputTokens: 100,
		Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 1, Prompt: "Custom prompt without placeholder."},
	}, []*ai.Message{userMsg(strings.Repeat("H", 250_000)), modelMsg(strings.Repeat("T", 250_000)), userMsg("Recent question")})

	if len(*prompts) != 1 {
		t.Fatalf("summarizer called %d times, want 1", len(*prompts))
	}
	p := (*prompts)[0]
	if !strings.HasPrefix(p, "Custom prompt without placeholder.\n\nConversation to summarize:\n") ||
		!regexp.MustCompile(`\.\.\.\[\d+ chars of conversation omitted\]\.\.\.`).MatchString(p) || len(p) >= 410_000 {
		t.Errorf("prompt (%d chars) = %.80q..., want the conversation appended and capped", len(p), p)
	}
}

func TestContextCompressionSummarizerCall(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	var requests []*ai.ModelRequest
	var gotAuth any
	f.defineModel("test/summarizer", func(_ int, req *ai.ModelRequest) *ai.ModelResponse {
		return &ai.ModelResponse{Message: modelMsg("Valid summary"), FinishReason: ai.FinishReasonStop}
	}, &requests)
	genkit.DefineModelAction(f.g, "test/authSummarizer", nil, func(ctx context.Context, req *ai.ModelRequest, _ any, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		gotAuth = core.FromContext(ctx)["auth"]
		return &ai.ModelResponse{Message: modelMsg("Valid summary"), FinishReason: ai.FinishReasonStop}, nil
	})

	// The ref's config reaches the summarizer as is.
	config := map[string]any{"temperature": 0.1}
	f.generate(t, &ContextCompression{
		MaxInputTokens: 20,
		Summarize:      &CompressionSummarizer{Model: ai.NewModelRef("test/summarizer", config), PreserveRecent: 1},
	}, []*ai.Message{userMsg("u1 " + strings.Repeat("X", 200)), modelMsg("m1 " + strings.Repeat("X", 200)), userMsg("u2")})
	if len(requests) != 1 || !cmp.Equal(requests[0].Config, config) {
		t.Errorf("summarizer requests = %d, config %v; want 1 with %v", len(requests), requests[0].Config, config)
	}

	// The summarizer runs with the caller's action context.
	ctx := core.WithActionContext(context.Background(), core.ActionContext{"auth": map[string]any{"uid": "user-123"}})
	if _, err := genkit.Generate(ctx, f.g,
		ai.WithModel(f.model),
		ai.WithMessages(userMsg("u1 "+strings.Repeat("X", 200)), modelMsg("m1 "+strings.Repeat("X", 200)), userMsg("u2")),
		ai.WithUse(&ContextCompression{
			MaxInputTokens: 20,
			Summarize:      &CompressionSummarizer{Model: ai.NewModelRef("test/authSummarizer", nil), PreserveRecent: 1},
		}),
	); err != nil {
		t.Fatal(err)
	}
	if !cmp.Equal(gotAuth, map[string]any{"uid": "user-123"}) {
		t.Errorf("summarizer auth = %v, want the caller's", gotAuth)
	}
}

func TestContextCompressionFallsBackWhenSummaryFails(t *testing.T) {
	for _, tt := range []struct {
		name  string
		reply *ai.ModelResponse
	}{
		{"blocked", &ai.ModelResponse{Message: modelMsg("Partial"), FinishReason: ai.FinishReasonBlocked}},
		{"cut off", &ai.ModelResponse{Message: modelMsg("Partial"), FinishReason: ai.FinishReasonLength}},
		{"empty", &ai.ModelResponse{Message: modelMsg("  "), FinishReason: ai.FinishReasonStop}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			f := newCCFixture(t, textReply("ok", 50))
			var requests []*ai.ModelRequest
			f.defineModel("test/summarizer", func(int, *ai.ModelRequest) *ai.ModelResponse { return tt.reply }, &requests)
			resp := f.generate(t, &ContextCompression{
				MaxInputTokens:     100,
				PreserveRecent:     3,
				NoTruncationNotice: true,
				Summarize:          &CompressionSummarizer{Model: ai.NewModelRef("test/summarizer", nil), PreserveRecent: 3},
			}, []*ai.Message{
				userMsg("u1 " + strings.Repeat("X", 200)), modelMsg("m1 " + strings.Repeat("X", 200)),
				userMsg("u2"), modelMsg("m2"), userMsg("u3"),
			})
			if got := texts(f.sent(t)); !cmp.Equal(got, []string{"user:u2", "model:m2", "user:u3"}) {
				t.Errorf("model received %v, want the 3 newest messages", got)
			}
			if s := stats(resp); s["summarized"] != false {
				t.Errorf("stats = %v, want not summarized", s)
			}
		})
	}
}

func TestContextCompressionPropagatesCancellation(t *testing.T) {
	f := newCCFixture(t, textReply("done", 10))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	genkit.DefineModelAction(f.g, "test/cancellingSummarizer", nil, func(ctx context.Context, _ *ai.ModelRequest, _ any, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		cancel()
		return nil, ctx.Err()
	})
	_, err := genkit.Generate(ctx, f.g,
		ai.WithModel(f.model),
		ai.WithMessages(fiveMessageHistory()...),
		ai.WithUse(&ContextCompression{
			MaxInputTokens: 200,
			Summarize:      &CompressionSummarizer{Model: ai.NewModelRef("test/cancellingSummarizer", nil)},
		}),
	)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if len(f.requests) != 0 {
		t.Errorf("the main model was called %d times after the cancellation", len(f.requests))
	}
}

// fiveMessageHistory is about 250 tokens of conversation.
func fiveMessageHistory() []*ai.Message {
	return []*ai.Message{
		userMsg("u1 " + strings.Repeat("A", 170)),
		modelMsg("m1 " + strings.Repeat("B", 170)),
		userMsg("u2 " + strings.Repeat("C", 170)),
		modelMsg("m2 " + strings.Repeat("D", 170)),
		userMsg("u3 " + strings.Repeat("E", 170)),
	}
}

func TestContextCompressionSummarizesShortHistories(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	summarizer, prompts := f.defineSummarizer("test/summarizer", "Summary of turn 1")
	resp := f.generate(t, &ContextCompression{
		MaxInputTokens: 200,
		Summarize:      &CompressionSummarizer{Model: summarizer},
	}, fiveMessageHistory())

	s := stats(resp)
	if len(*prompts) != 1 || s["summarized"] != true || s["truncationNoticeInserted"] != false {
		t.Errorf("summarizer calls = %d, stats = %v; want 1 summary and no notice", len(*prompts), s)
	}
	if !strings.Contains(f.sent(t)[0].Text(), "Summary of turn 1") {
		t.Errorf("model received %v, want the summary first", texts(f.sent(t)))
	}
	if r := ResolveCompressedHistory(resp.History()); !strings.Contains(r[0].Text(), "Summary of turn 1") {
		t.Errorf("resolved %v, want the summary first", texts(r))
	}
}

func TestContextCompressionSummaryRespectsMaxMessages(t *testing.T) {
	var history []*ai.Message
	for i := range 10 {
		text := fmt.Sprintf("Turn %d: %s", i, strings.Repeat("X", 50))
		if i%2 == 0 {
			history = append(history, userMsg(text))
		} else {
			history = append(history, modelMsg(text))
		}
	}
	f := newCCFixture(t, textReply("done", 50))
	summarizer, _ := f.defineSummarizer("test/summarizer", "Preserved summary text")
	f.generate(t, &ContextCompression{
		MaxInputTokens: 150,
		MaxMessages:    6,
		Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 6},
	}, history)
	sent := f.sent(t)
	if len(sent) != 6 || !strings.Contains(sent[0].Text(), "Preserved summary text") {
		t.Errorf("model received %v, want 6 messages starting with the summary", texts(sent))
	}
}

func TestContextCompressionTruncatesToPreserveRecentWithoutOtherStrategies(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	f.generate(t, &ContextCompression{MaxInputTokens: 100, PreserveRecent: 3, NoTruncationNotice: true}, []*ai.Message{
		userMsg("u1 " + strings.Repeat("X", 100)), modelMsg("m1 " + strings.Repeat("X", 100)),
		userMsg("u2 " + strings.Repeat("X", 100)), modelMsg("m2 " + strings.Repeat("X", 100)),
		userMsg("u3"), modelMsg("m3"), userMsg("u4"),
	})
	if got := texts(f.sent(t)); !cmp.Equal(got, []string{"user:u3", "model:m3", "user:u4"}) {
		t.Errorf("model received %v, want the 3 newest messages", got)
	}
}

func TestContextCompressionMultipartToolResponses(t *testing.T) {
	f := newCCFixture(t, textReply("done", 50))
	run := func(mw *ContextCompression, out any, content ...*ai.Part) *ai.ModelResponse {
		t.Helper()
		return f.generate(t, mw, []*ai.Message{
			userMsg("run"),
			toolCallMsg(&ai.ToolRequest{Name: "t", Ref: "1", Input: map[string]any{}}),
			toolResultMsg(&ai.ToolResponse{Name: "t", Ref: "1", Output: out, Content: content}),
		})
	}
	sentResponse := func() *ai.ToolResponse { return toolMessages(f.sent(t))[0].Content[0].ToolResponse }

	t.Run("folds text content into the truncated output", func(t *testing.T) {
		large := strings.Repeat("X", 200_000)
		resp := run(&ContextCompression{
			MaxInputTokens:        1000,
			MaxToolResponseChars:  500,
			TruncateToolResponses: &CompressionToolTruncation{MaxChars: 100, PreserveRecent: -1},
		}, "ok", ai.NewTextPart(large))
		want := "ok\n\n" + strings.Repeat("X", 96) + "\n\n[Truncated 199904 characters]"
		if s := stats(resp); s["toolResponsesTruncated"] != 1 || s["toolResponsesSafetyCapped"] != 0 {
			t.Errorf("stats = %v, want 1 truncated and 0 capped", s)
		}
		if got := sentResponse(); got.Output != want || got.Content != nil {
			t.Errorf("sent %.40q... with content %v, want %.40q... without", got.Output, got.Content, want)
		}
		if got := toolMessages(resp.History())[0].Content[0].ToolResponse.Content[0].Text; got != large {
			t.Errorf("history content = %.20q..., want the original", got)
		}
		if got := toolMessages(ResolveCompressedHistory(resp.History()))[0].Content[0].ToolResponse; got.Output != want || got.Content != nil {
			t.Errorf("resolved %.40q... with content %v, want the sent response", got.Output, got.Content)
		}
	})

	t.Run("caps under budget", func(t *testing.T) {
		resp := run(&ContextCompression{MaxInputTokens: 100_000, MaxToolResponseChars: 500}, "ok",
			ai.NewTextPart(strings.Repeat("A", 100)), ai.NewTextPart(strings.Repeat("B", 1000)), ai.NewTextPart(strings.Repeat("C", 500)))
		want := "ok\n\n" + strings.Repeat("A", 100) + "\n\n" + strings.Repeat("B", 394) +
			"\n\n---\n\n[TRUNCATED: Response was 1602 chars but only first 500 are shown.]"
		if s := stats(resp); s["toolResponsesSafetyCapped"] != 1 {
			t.Errorf("stats = %v, want 1 capped", s)
		}
		if got := sentResponse(); got.Output != want || got.Content != nil {
			t.Errorf("sent %q, want %q", got.Output, want)
		}
	})

	t.Run("drops content when the output alone exceeds the limit", func(t *testing.T) {
		run(&ContextCompression{
			MaxInputTokens:        50,
			MaxToolResponseChars:  500,
			TruncateToolResponses: &CompressionToolTruncation{MaxChars: 100, PreserveRecent: -1},
		}, strings.Repeat("O", 200), ai.NewTextPart(strings.Repeat("C", 300)))
		want := strings.Repeat("O", 100) + "\n\n[Truncated 400 characters]"
		if got := sentResponse(); got.Output != want || got.Content != nil {
			t.Errorf("sent %q, want %q", got.Output, want)
		}
	})

	t.Run("counts inline media at a flat size", func(t *testing.T) {
		dataURL := "data:image/png;base64," + strings.Repeat("A", 500_000)
		media := ai.NewMediaPart("image/png", dataURL)
		resp := run(&ContextCompression{MaxInputTokens: 2000}, "ok", media)
		if s := stats(resp); s != nil {
			t.Errorf("stats = %v, want none", s)
		}
		if got := sentResponse(); len(got.Content) != 1 || got.Content[0].Text != dataURL {
			t.Error("the media part was not sent intact")
		}

		resp = run(&ContextCompression{
			MaxInputTokens:        100,
			TruncateToolResponses: &CompressionToolTruncation{MaxChars: 100, PreserveRecent: -1},
		}, "ok", media)
		want := fmt.Sprintf("ok\n\n[media: image/png]\n\n[Truncated %d characters]", len(dataURL)-len("[media: image/png]"))
		if s := stats(resp); s["toolResponsesTruncated"] != 1 {
			t.Errorf("stats = %v, want 1 truncated", s)
		}
		if got := sentResponse(); got.Output != want || got.Content != nil {
			t.Errorf("sent %q, want %q", got.Output, want)
		}
	})

	t.Run("describes remote media whole or not at all", func(t *testing.T) {
		url := "https://example.com/assets/" + strings.Repeat("img", 100) + ".jpg"
		media := ai.NewMediaPart("image/jpeg", url)
		run(&ContextCompression{
			MaxInputTokens:        20,
			TruncateToolResponses: &CompressionToolTruncation{MaxChars: 50, PreserveRecent: -1},
		}, "ok", media)
		want := fmt.Sprintf("ok\n\n[media: image/jpeg]\n\n[Truncated %d characters]", len(url)-len("[media: image/jpeg]"))
		if got := sentResponse(); got.Output != want || got.Content != nil {
			t.Errorf("sent %q, want %q", got.Output, want)
		}

		run(&ContextCompression{
			MaxInputTokens:        20,
			TruncateToolResponses: &CompressionToolTruncation{MaxChars: 10, PreserveRecent: -1},
		}, "ok", media)
		if want := fmt.Sprintf("ok\n\n[Truncated %d characters]", len(url)); sentResponse().Output != want {
			t.Errorf("sent %q, want %q", sentResponse().Output, want)
		}
	})

	t.Run("keeps a structured part that fits exactly", func(t *testing.T) {
		data := ai.NewDataPart(map[string]any{"status": "ready"})
		run(&ContextCompression{
			MaxInputTokens:        20,
			TruncateToolResponses: &CompressionToolTruncation{MaxChars: 20, PreserveRecent: -1},
		}, "ok", data, ai.NewTextPart(strings.Repeat("X", 200)))
		if got := sentResponse(); got.Output != "ok\n\n[Truncated 200 characters]" || len(got.Content) != 1 || got.Content[0] != data {
			t.Errorf("sent %q with content %v, want the data part kept", got.Output, got.Content)
		}
	})
}

func TestContextCompressionToleratesNilParts(t *testing.T) {
	// A part decoded from a JSON null is a nil pointer. Every strategy passes
	// over it, and the model's input validation reports it.
	big := strings.Repeat("x", 600)
	call := func(ref string) *ai.Message {
		m := toolCallMsg(&ai.ToolRequest{Name: "search", Ref: ref, Input: map[string]any{"q": "same"}})
		m.Content = append([]*ai.Part{nil}, m.Content...)
		return m
	}
	result := func(ref string) *ai.Message {
		m := toolResultMsg(&ai.ToolResponse{Name: "search", Ref: ref, Output: big, Content: []*ai.Part{nil, ai.NewTextPart(big)}})
		m.Content = append(m.Content, nil)
		return m
	}
	history := []*ai.Message{
		systemMsg("sys"), userMsg("go"),
		call("1"), result("1"),
		call("2"), result("2"),
		call("3"), result("3"),
	}

	f := newCCFixture(t, textReply("done", 50))
	rec := &viewRecorder{}
	_, err := genkit.Generate(context.Background(), f.g, ai.WithModel(f.model), ai.WithMessages(history...), ai.WithUse(&ContextCompression{
		MaxMessages:           6,
		MaxToolResponseChars:  1000,
		DedupeToolResponses:   &CompressionDedupe{},
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 50, PreserveRecent: 1},
	}, rec.middleware()))
	if !errors.Is(err, status.ErrInvalidArgument) {
		t.Errorf("Generate error = %v, want the model's input validation error", err)
	}
	if len(rec.views) != 1 {
		t.Fatalf("the model was called %d times, want 1", len(rec.views))
	}
	view := rec.views[0]
	tools := toolMessages(view)
	if len(view) != 6 || len(tools) != 2 {
		t.Fatalf("view = %s, want 6 messages and the 2 newest tool turns", renderMessages(view))
	}
	if got := tools[0].Content[0].ToolResponse.Output; got != defaultDedupeNotice {
		t.Errorf("older response = %q, want the dedupe notice", got)
	}
	if got := tools[1].Content[0].ToolResponse.Output.(string); !strings.Contains(got, "[TRUNCATED: Response was 1200 chars") {
		t.Errorf("newest response = %.40q..., want it capped", got)
	}

	// A stamped history resolves past the nil parts and messages.
	stamped := slices.Clone(history)
	stamped[3] = &ai.Message{Role: ai.RoleTool, Content: []*ai.Part{nil, {
		Kind:         ai.PartToolResponse,
		ToolResponse: &ai.ToolResponse{Name: "search", Ref: "1", Output: big},
		Metadata:     map[string]any{compressionKey: map[string]any{ccTruncated: true, ccMaxChars: 10, ccRawOutput: true}},
	}}}
	stamped = append(stamped, nil)
	got := ResolveCompressedHistory(stamped)
	if want := "xxxxxxxxxx\n\n[Truncated 590 characters]"; got[3].Content[1].ToolResponse.Output != want {
		t.Errorf("resolved output = %q, want %q", got[3].Content[1].ToolResponse.Output, want)
	}
}

func TestContextCompressionStats(t *testing.T) {
	t.Run("reports the compression on the response", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		resp := f.generate(t, &ContextCompression{MaxMessages: 2, NoTruncationNotice: true},
			[]*ai.Message{userMsg("hello"), modelMsg("response 1"), userMsg("world")})
		if s := stats(resp); s["triggered"] != true || s["messagesOriginal"] != 3 || s["messagesAfter"] != 1 {
			t.Errorf("stats = %v, want 3 messages compressed to 1", s)
		}
	})

	t.Run("reports a compression in a later iteration", func(t *testing.T) {
		f := newCCFixture(t, func(call int, _ *ai.ModelRequest) *ai.ModelResponse {
			if call == 1 {
				return &ai.ModelResponse{Message: toolCallMsg(&ai.ToolRequest{Name: "step", Input: map[string]any{}}), Usage: &ai.GenerationUsage{InputTokens: 500}}
			}
			return &ai.ModelResponse{Message: modelMsg("done"), Usage: &ai.GenerationUsage{InputTokens: 100}}
		})
		step := genkit.DefineTool(f.g, "step", "step", func(ctx *ai.ToolContext, in struct{}) (string, error) { return "tool result", nil })
		resp := f.generate(t, &ContextCompression{
			MaxInputTokens:        200,
			TruncateToolResponses: &CompressionToolTruncation{MaxChars: 5, PreserveRecent: -1},
		}, []*ai.Message{userMsg("test metadata")}, ai.WithTools(step))
		if s := stats(resp); s["triggered"] != true || s["toolResponsesTruncated"] != 1 {
			t.Errorf("stats = %v, want 1 truncated response", s)
		}
	})

	t.Run("keeps per-call state apart across concurrent calls", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		mw := &ContextCompression{MaxInputTokens: 100, MaxMessages: 2, NoTruncationNotice: true}
		histories := [][]*ai.Message{
			{userMsg("Req 1 message 1"), modelMsg("Req 1 message 2"), userMsg("Req 1 message 3")},
			{userMsg("Req 2 single message")},
		}
		resps := make([]*ai.ModelResponse, len(histories))
		var wg sync.WaitGroup
		for i, h := range histories {
			wg.Go(func() {
				resp, err := genkit.Generate(context.Background(), f.g, ai.WithModel(f.model), ai.WithMessages(h...), ai.WithUse(mw))
				if err != nil {
					t.Error(err)
				}
				resps[i] = resp
			})
		}
		wg.Wait()
		if s := stats(resps[0]); s["messagesOriginal"] != 3 || s["messagesAfter"] != 1 {
			t.Errorf("first stats = %v, want 3 messages compressed to 1", s)
		}
		if s := stats(resps[1]); s != nil {
			t.Errorf("second stats = %v, want none", s)
		}
	})

	t.Run("shares one history between concurrent calls", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		mw := &ContextCompression{
			MaxInputTokens:        50,
			MaxMessages:           4,
			TruncateToolResponses: &CompressionToolTruncation{MaxChars: 10, PreserveRecent: -1},
		}
		history := []*ai.Message{
			withCallerMetadata(userMsg("q")),
			withCallerMetadata(toolCallMsg(&ai.ToolRequest{Name: "t", Input: map[string]any{}})),
			withCallerMetadata(toolResultMsg(&ai.ToolResponse{Name: "t", Output: strings.Repeat("X", 400)})),
			withCallerMetadata(modelMsg("a")),
			withCallerMetadata(userMsg("q2")),
		}
		var wg sync.WaitGroup
		for range 4 {
			wg.Go(func() {
				if _, err := genkit.Generate(context.Background(), f.g, ai.WithModel(f.model), ai.WithMessages(history...), ai.WithUse(mw)); err != nil {
					t.Error(err)
				}
			})
		}
		wg.Wait()
	})
}

func TestContextCompressionProtectedMessages(t *testing.T) {
	scaffold := func(m *ai.Message) *ai.Message {
		m.Metadata = map[string]any{"_genkit_prompt": true}
		return m
	}
	withInstructions := func(m *ai.Message) *ai.Message {
		p := ai.NewTextPart("Output JSON matching the schema.")
		p.Metadata = map[string]any{"purpose": "output"}
		m.Content = append(m.Content, p)
		return m
	}

	t.Run("summarization keeps the output instructions", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		summarizer, prompts := f.defineSummarizer("test/summarizer", "SUMMARY")
		resp := f.generate(t, &ContextCompression{
			MaxInputTokens: 50,
			Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 2},
		}, []*ai.Message{
			userMsg("old question " + strings.Repeat("X", 200)),
			modelMsg("old answer " + strings.Repeat("X", 200)),
			withInstructions(userMsg("Report on the cluster")),
			toolCallMsg(&ai.ToolRequest{Name: "probe", Input: map[string]any{"n": 1}}),
			toolResultMsg(&ai.ToolResponse{Name: "probe", Output: strings.Repeat("P", 200)}),
			toolCallMsg(&ai.ToolRequest{Name: "probe", Input: map[string]any{"n": 2}}),
			toolResultMsg(&ai.ToolResponse{Name: "probe", Output: "latest"}),
		})
		want := []string{"user:Report on the cluster", "user:" + summaryPrefix + "\nSUMMARY", "model:call probe", "tool:result probe"}
		if diff := cmp.Diff(want, texts(f.sent(t))); diff != "" {
			t.Errorf("model received (-want +got):\n%s", diff)
		}
		if len(*prompts) != 1 || strings.Contains((*prompts)[0], "Report on the cluster") {
			t.Errorf("summarizer prompts = %q, want one without the protected message", *prompts)
		}
		if diff := cmp.Diff(want, texts(ResolveCompressedHistory(resp.History()[:7]))); diff != "" {
			t.Errorf("resolved history (-want +got):\n%s", diff)
		}
	})

	t.Run("summarization makes room under the cap", func(t *testing.T) {
		// The lifted message takes a slot, so the summary keeps fewer recent
		// messages instead of giving way to truncation.
		f := newCCFixture(t, textReply("done", 50))
		summarizer, _ := f.defineSummarizer("test/summarizer", "SUMMARY")
		f.generate(t, &ContextCompression{
			MaxMessages: 6,
			Summarize:   &CompressionSummarizer{Model: summarizer},
		}, []*ai.Message{
			systemMsg("Sys"),
			scaffold(userMsg("Task")),
			toolCallMsg(&ai.ToolRequest{Name: "t1", Input: map[string]any{}}),
			toolResultMsg(&ai.ToolResponse{Name: "t1", Output: "r1"}),
			toolCallMsg(&ai.ToolRequest{Name: "t2", Input: map[string]any{}}),
			toolResultMsg(&ai.ToolResponse{Name: "t2", Output: "r2"}),
			toolCallMsg(&ai.ToolRequest{Name: "t3", Input: map[string]any{}}),
			toolResultMsg(&ai.ToolResponse{Name: "t3", Output: "r3"}),
		})
		want := []string{"system:Sys", "user:Task", "user:" + summaryPrefix + "\nSUMMARY", "model:call t3", "tool:result t3"}
		if diff := cmp.Diff(want, texts(f.sent(t))); diff != "" {
			t.Errorf("model received (-want +got):\n%s", diff)
		}
	})

	t.Run("truncation keeps prompt scaffolding and counts it", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		f.generate(t, &ContextCompression{MaxMessages: 5}, []*ai.Message{
			scaffold(systemMsg("Agent instructions")),
			scaffold(userMsg("Example question")),
			scaffold(modelMsg("Example answer")),
			userMsg("q1"), modelMsg("a1"), userMsg("q2"), modelMsg("a2"), userMsg("q3"),
		})
		want := []string{"system:Agent instructions", "user:Example question", "model:Example answer", "user:q3"}
		if diff := cmp.Diff(want, texts(f.sent(t))); diff != "" {
			t.Errorf("model received (-want +got):\n%s", diff)
		}
	})

	t.Run("a protected user message stands in for the anchor", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		resp := f.generate(t, &ContextCompression{MaxMessages: 4}, []*ai.Message{
			systemMsg("Sys"),
			userMsg("Earlier question"),
			modelMsg("Earlier answer"),
			withInstructions(userMsg("Original task")),
			toolCallMsg(&ai.ToolRequest{Name: "t1", Input: map[string]any{}}),
			toolResultMsg(&ai.ToolResponse{Name: "t1", Output: "r1"}),
			toolCallMsg(&ai.ToolRequest{Name: "t2", Input: map[string]any{}}),
			toolResultMsg(&ai.ToolResponse{Name: "t2", Output: "r2"}),
		})
		want := []string{"system:Sys", "user:Original task", "model:call t2", "tool:result t2"}
		if diff := cmp.Diff(want, texts(f.sent(t))); diff != "" {
			t.Errorf("model received (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(want, texts(ResolveCompressedHistory(resp.History()[:8]))); diff != "" {
			t.Errorf("resolved history (-want +got):\n%s", diff)
		}
	})

	t.Run("the boundary never lands on scaffolding", func(t *testing.T) {
		f := newCCFixture(t, textReply("done", 50))
		summarizer, _ := f.defineSummarizer("test/summarizer", "SUMMARY")
		resp := f.generate(t, &ContextCompression{
			MaxInputTokens: 50,
			Summarize:      &CompressionSummarizer{Model: summarizer, PreserveRecent: 2},
		}, []*ai.Message{
			userMsg("history " + strings.Repeat("X", 300)),
			modelMsg("reply " + strings.Repeat("X", 300)),
			scaffold(userMsg("Template reminder")),
			toolCallMsg(&ai.ToolRequest{Name: "probe", Input: map[string]any{}}),
			toolResultMsg(&ai.ToolResponse{Name: "probe", Output: "latest"}),
		})
		for i, m := range resp.Request.Messages {
			if _, ok := compressionMeta(m.Metadata)[ccSummary]; ok && isPinned(m) {
				t.Errorf("boundary stamped on scaffolding message %d", i)
			}
		}
		want := []string{"user:" + summaryPrefix + "\nSUMMARY", "user:Template reminder", "model:call probe", "tool:result probe"}
		if diff := cmp.Diff(want, texts(f.sent(t))); diff != "" {
			t.Errorf("model received (-want +got):\n%s", diff)
		}
	})
}

// TestContextCompressionRestoresHistoryUnderRetry runs a retry outside the
// middleware. The middleware replaces the request on the params it receives,
// which is safe only while every model call builds fresh params.
func TestContextCompressionRestoresHistoryUnderRetry(t *testing.T) {
	long := strings.Repeat("Z", 500)
	failures := 1
	f := &ccFixture{g: genkit.Init(context.Background())}
	f.model = genkit.DefineModelAction(f.g, "test/flaky", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, _ any, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			f.requests = append(f.requests, req)
			if failures > 0 {
				failures--
				return nil, status.Errorf(status.ErrUnavailable, "try again")
			}
			return &ai.ModelResponse{Request: req, Message: modelMsg("ok"), Usage: &ai.GenerationUsage{InputTokens: 10}}, nil
		})

	retry := ai.MiddlewareFunc(func(ctx context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{WrapModel: func(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
			resp, err := next(ctx, params)
			if err != nil {
				return next(ctx, params)
			}
			return resp, err
		}}, nil
	})
	resp, err := genkit.Generate(context.Background(), f.g,
		ai.WithModel(f.model),
		ai.WithMessages(userMsg("q"), toolCallMsg(&ai.ToolRequest{Name: "big", Input: map[string]any{}}), toolResultMsg(&ai.ToolResponse{Name: "big", Output: long})),
		ai.WithUse(retry, &ContextCompression{MaxToolResponseChars: 100}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(f.requests) != 2 {
		t.Fatalf("model called %d times, want 2", len(f.requests))
	}
	for i, req := range f.requests {
		if out := output(t, req.Messages[2], 0); !strings.Contains(out, "[TRUNCATED:") {
			t.Errorf("attempt %d sent %.20q..., want capped", i+1, out)
		}
	}
	if out := output(t, resp.History()[2], 0); out != long {
		t.Errorf("history output = %.20q..., want the original", out)
	}
}

func TestContextCompressionJSONDispatch(t *testing.T) {
	g := genkit.Init(context.Background(), genkit.WithPlugins(&Middleware{}))
	var requests []*ai.ModelRequest
	f := &ccFixture{g: g}
	f.model = f.defineModel("test/main", textReply("done", 50), &requests)
	f.defineSummarizer("test/summarizer", "JSON SUMMARY")

	resp, err := genkit.GenerateWithRequest(context.Background(), g, &ai.GenerateActionOptions{
		Model: "test/main",
		Messages: []*ai.Message{
			userMsg("u1 " + strings.Repeat("X", 200)), modelMsg("m1 " + strings.Repeat("X", 200)), userMsg("u2"),
		},
		Use: []*ai.MiddlewareRef{{
			Name: ContextCompression{}.Name(),
			Config: map[string]any{
				"maxInputTokens": 20,
				"summarize":      map[string]any{"model": "test/summarizer", "preserveRecent": 1},
			},
		}},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(requests[0].Messages[0].Text(), "JSON SUMMARY") {
		t.Errorf("model received %v, want the summary", texts(requests[0].Messages))
	}
	if stats(resp)["summarized"] != true {
		t.Errorf("stats = %v, want summarized", stats(resp))
	}
}

func TestContextCompressionHistoryRoundTripsThroughJSON(t *testing.T) {
	mw := &ContextCompression{
		MaxInputTokens:        100,
		DedupeToolResponses:   &CompressionDedupe{},
		TruncateToolResponses: &CompressionToolTruncation{MaxChars: 30, PreserveRecent: -1},
		MaxMessages:           5,
	}
	f := newCCFixture(t, textReply("done", 50))
	resp := f.generate(t, mw, []*ai.Message{
		systemMsg("Sys"),
		userMsg("u1"),
		toolCallMsg(&ai.ToolRequest{Name: "fetch", Ref: "a", Input: map[string]any{"id": 1}}),
		toolResultMsg(&ai.ToolResponse{Name: "fetch", Ref: "a", Output: strings.Repeat("A", 300)}),
		toolCallMsg(&ai.ToolRequest{Name: "fetch", Ref: "b", Input: map[string]any{"id": 1}}),
		toolResultMsg(&ai.ToolResponse{Name: "fetch", Ref: "b", Output: strings.Repeat("B", 300)}),
		userMsg("u2"),
	})
	history := resp.History()
	b, err := json.Marshal(history)
	if err != nil {
		t.Fatal(err)
	}
	var decoded []*ai.Message
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatal(err)
	}
	want, got := ResolveCompressedHistory(history), ResolveCompressedHistory(decoded)
	wantJSON, _ := json.Marshal(want)
	gotJSON, _ := json.Marshal(got)
	if string(wantJSON) != string(gotJSON) {
		t.Errorf("resolved view changed through JSON:\nbefore %s\nafter  %s", wantJSON, gotJSON)
	}
}

func TestContextCompressionHonorsJSBoundaryFields(t *testing.T) {
	// A boundary the JS runtime recorded with preserveSystem: false drops the
	// leading system messages it covers.
	history := []*ai.Message{
		systemMsg("Old system"),
		userMsg("u1"),
		{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("m1")}, Metadata: map[string]any{compressionKey: map[string]any{
			ccSummary: "JS SUMMARY", ccStats: map[string]any{"triggered": true}, ccPreserveSystem: false,
		}}},
		userMsg("u2"),
	}
	want := []string{"user:" + summaryPrefix + "\nJS SUMMARY", "user:u2"}
	if diff := cmp.Diff(want, texts(ResolveCompressedHistory(history))); diff != "" {
		t.Errorf("resolved (-want +got):\n%s", diff)
	}
}

func TestContextCompressionValidation(t *testing.T) {
	g := genkit.Init(context.Background())
	ctx := genkitContext(t, g)
	for _, tt := range []struct {
		name string
		mw   ContextCompression
		want error
	}{
		{"negative token budget", ContextCompression{MaxInputTokens: -1}, status.ErrInvalidArgument},
		{"negative message cap", ContextCompression{MaxMessages: -1}, status.ErrInvalidArgument},
		{"negative window", ContextCompression{PreserveRecent: -1}, status.ErrInvalidArgument},
		{"threshold above 1", ContextCompression{SkipSummarizationThreshold: 1.5}, status.ErrInvalidArgument},
		{"negative threshold", ContextCompression{SkipSummarizationThreshold: -0.1}, status.ErrInvalidArgument},
		{"unknown matchBy", ContextCompression{DedupeToolResponses: &CompressionDedupe{MatchBy: "name"}}, status.ErrInvalidArgument},
		{"negative keepRecent", ContextCompression{DedupeToolResponses: &CompressionDedupe{KeepRecent: -1}}, status.ErrInvalidArgument},
		{"missing maxChars", ContextCompression{TruncateToolResponses: &CompressionToolTruncation{}}, status.ErrInvalidArgument},
		{"missing summarizer model", ContextCompression{Summarize: &CompressionSummarizer{}}, status.ErrInvalidArgument},
		{"negative summary window", ContextCompression{Summarize: &CompressionSummarizer{Model: ai.NewModelRef("test/s", nil), PreserveRecent: -1}}, status.ErrInvalidArgument},
		{"unknown summarizer model", ContextCompression{Summarize: &CompressionSummarizer{Model: ai.NewModelRef("test/missing", nil)}}, ai.ErrModelNotFound},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.mw.New(ctx); !errors.Is(err, tt.want) {
				t.Errorf("New() error = %v, want %v", err, tt.want)
			}
		})
	}
}

// genkitContext returns a context carrying g, as Generate seeds it.
func genkitContext(t *testing.T, g *genkit.Genkit) context.Context {
	t.Helper()
	var seeded context.Context
	m := genkit.DefineModelAction(g, "test/seed", nil, func(ctx context.Context, req *ai.ModelRequest, _ any, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		seeded = ctx
		return &ai.ModelResponse{Message: modelMsg("ok")}, nil
	})
	if _, err := genkit.Generate(context.Background(), g, ai.WithModel(m), ai.WithPrompt("seed")); err != nil {
		t.Fatal(err)
	}
	return seeded
}
