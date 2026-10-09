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
	"cmp"
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/ai/tool"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	gocmp "github.com/google/go-cmp/cmp"
)

// To run: GEMINI_API_KEY=... go test -run TestContextCompressionLive -v

const (
	liveModel      = "googleai/gemini-3.5-flash"
	liveSummarizer = "googleai/gemini-3.5-flash-lite"
)

// viewRecorder records the messages each model call receives. Listed after
// ContextCompression, it sees the compressed view the model is sent.
type viewRecorder struct {
	mu    sync.Mutex
	views [][]*ai.Message
}

func (v *viewRecorder) middleware() ai.Middleware {
	return ai.MiddlewareFunc(func(ctx context.Context) (*ai.Hooks, error) {
		return &ai.Hooks{WrapModel: func(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
			v.mu.Lock()
			v.views = append(v.views, params.Request.Messages)
			v.mu.Unlock()
			return next(ctx, params)
		}}, nil
	})
}

// reportDetails returns a long, distinct body for a report.
func reportDetails(id string) string {
	var sb strings.Builder
	for i := range 12 {
		fmt.Fprintf(&sb, "Section %d of %s: latency, throughput, error budgets and rollout notes for cluster us-central-%d. ", i, id, i)
	}
	return sb.String()
}

func TestContextCompressionLive(t *testing.T) {
	apiKey := cmp.Or(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY"))
	if apiKey == "" {
		t.Skip("set GEMINI_API_KEY or GOOGLE_API_KEY to run the live test")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{APIKey: apiKey}))

	searchReports := genkit.DefineTool(g, "searchReports", "Searches for report IDs about a topic.",
		func(ctx *ai.ToolContext, in struct {
			Topic string `json:"topic"`
		}) ([]string, error) {
			return []string{"report-101", "report-102", "report-103"}, nil
		})
	fetchReport := genkit.DefineTool(g, "fetchReport", "Fetches the full text of one report by ID.",
		func(ctx *ai.ToolContext, in struct {
			ReportID string `json:"reportId"`
		}) (map[string]any, error) {
			return map[string]any{"id": in.ReportID, "title": "Project Alpha " + in.ReportID, "details": reportDetails(in.ReportID)}, nil
		})

	t.Run("research loop compresses mid-run and keeps the history", func(t *testing.T) {
		rec := &viewRecorder{}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName(liveModel),
			ai.WithSystem("You are a research assistant. Search for reports first, then fetch every report, one fetchReport call per turn, then write a short summary of all of them. Older tool results are shortened to save space, which is expected: never fetch a report twice."),
			ai.WithPrompt("Investigate Project Alpha."),
			ai.WithTools(searchReports, fetchReport),
			ai.WithMaxTurns(10),
			ai.WithUse(
				&ContextCompression{
					MaxInputTokens:             900,
					DedupeToolResponses:        &CompressionDedupe{},
					TruncateToolResponses:      &CompressionToolTruncation{MaxChars: 200, PreserveRecent: 1},
					Summarize:                  &CompressionSummarizer{Model: ai.NewModelRef(liveSummarizer, nil), PreserveRecent: 2},
					SkipSummarizationThreshold: 0.3,
				},
				rec.middleware(),
			),
		)
		if err != nil {
			t.Fatal(err)
		}
		if strings.TrimSpace(resp.Text()) == "" {
			t.Fatal("empty final answer")
		}
		s := stats(resp)
		t.Logf("stats: %v", s)
		if s["triggered"] != true {
			t.Fatalf("stats = %v, want a compression", s)
		}

		history := resp.History()
		fetched := 0
		for _, m := range toolMessages(history) {
			for _, p := range m.Content {
				if p.ToolResponse.Name != "fetchReport" {
					continue
				}
				fetched++
				out, _ := p.ToolResponse.Output.(map[string]any)
				if id, _ := out["id"].(string); out["details"] != reportDetails(id) {
					t.Errorf("history lost the full output of %v", out["id"])
				}
			}
		}
		if fetched == 0 {
			t.Fatal("the model fetched no report")
		}
		for i, m := range history[:len(history)-1] {
			if m.Role == ai.RoleModel && compressionMeta(m.Metadata)[ccInputTokens] == nil {
				t.Errorf("model message %d has no inputTokens stamp", i)
			}
		}

		// The last call's view is what the history resolves to.
		last := rec.views[len(rec.views)-1]
		t.Logf("last view: %q", texts(last))
		if diff := gocmp.Diff(texts(last), texts(ResolveCompressedHistory(history[:len(history)-1]))); diff != "" {
			t.Errorf("resolved history differs from the last view (-view +resolved):\n%s", diff)
		}
		if len(last) >= len(history)-1 && s["summarized"] == true {
			t.Errorf("last view holds %d messages of %d, want fewer after a summary", len(last), len(history)-1)
		}
	})

	t.Run("streaming tool loop keeps the history", func(t *testing.T) {
		rec := &viewRecorder{}
		chunks := 0
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName(liveModel),
			ai.WithSystem("You are a research assistant. Search for reports first, then fetch every report, one fetchReport call per turn, then write a short summary of all of them. Older tool results are shortened to save space, which is expected: never fetch a report twice."),
			ai.WithPrompt("Investigate Project Alpha."),
			ai.WithTools(searchReports, fetchReport),
			ai.WithMaxTurns(10),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				chunks++
				return nil
			}),
			ai.WithUse(
				&ContextCompression{
					MaxInputTokens:        900,
					TruncateToolResponses: &CompressionToolTruncation{MaxChars: 200, PreserveRecent: 1},
				},
				rec.middleware(),
			),
		)
		if err != nil {
			t.Fatal(err)
		}
		if chunks == 0 {
			t.Error("no chunks streamed")
		}
		if s := stats(resp); s["toolResponsesTruncated"] == nil || s["toolResponsesTruncated"] == 0 {
			t.Fatalf("stats = %v, want truncated responses", s)
		}
		history := resp.History()
		for _, m := range toolMessages(history) {
			for _, p := range m.Content {
				if out, _ := p.ToolResponse.Output.(map[string]any); p.ToolResponse.Name == "fetchReport" && out["details"] != reportDetails(out["id"].(string)) {
					t.Errorf("history lost the full output of %v", out["id"])
				}
			}
		}
		last := rec.views[len(rec.views)-1]
		if diff := gocmp.Diff(texts(last), texts(ResolveCompressedHistory(history[:len(history)-1]))); diff != "" {
			t.Errorf("resolved history differs from the last view (-view +resolved):\n%s", diff)
		}
	})

	t.Run("resume after an interrupt keeps tool pairs", func(t *testing.T) {
		type approval struct {
			Approved bool `json:"approved"`
		}
		publish := genkit.DefineResumableTool(g, "publish", "Publishes a one-line summary of the reports. Needs approval.",
			func(ctx context.Context, in struct {
				Summary string `json:"summary"`
			}, ok *approval) (string, error) {
				if ok == nil {
					return "", tool.Interrupt(ctx, map[string]any{"reason": "needs approval"})
				}
				return "published", nil
			})
		mw := &ContextCompression{
			MaxInputTokens:        700,
			TruncateToolResponses: &CompressionToolTruncation{MaxChars: 150, PreserveRecent: 1},
		}
		rec := &viewRecorder{}
		opts := []ai.GenerateOption{ai.WithModelName(liveModel), ai.WithTools(fetchReport, publish), ai.WithMaxTurns(10), ai.WithUse(mw, rec.middleware())}
		resp, err := genkit.Generate(ctx, g, append(opts,
			ai.WithSystem("Fetch report-101, report-102 and report-103 with fetchReport, one call per turn. Then call publish once with a one-line summary, and reply with one word when it is published."),
			ai.WithPrompt("Go."))...)
		if err != nil {
			t.Fatal(err)
		}
		interrupts := resp.Interrupts()
		if len(interrupts) != 1 {
			t.Fatalf("got %d interrupts (finish %q, text %q), want 1", len(interrupts), resp.FinishReason, resp.Text())
		}
		call, ok := publish.Interrupted(interrupts[0])
		if !ok {
			t.Fatalf("interrupt %v is not a publish call", interrupts[0].ToolRequest.Name)
		}
		paused := len(rec.views)
		resp, err = genkit.Generate(ctx, g, append(opts, ai.WithMessages(resp.History()...), ai.WithResume(call.Restart(approval{Approved: true})))...)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answer: %q, stats: %v", resp.Text(), stats(resp))
		if len(rec.views) == paused {
			t.Fatal("the resumed call never reached the model")
		}
		for i, view := range rec.views {
			for j, m := range view {
				if m.Role == ai.RoleTool && (j == 0 || view[j-1].Role != ai.RoleModel && view[j-1].Role != ai.RoleTool) {
					t.Errorf("call %d view = %v, want every tool message after its request", i, texts(view))
				}
			}
		}
		published := false
		for _, m := range toolMessages(resp.History()) {
			for _, p := range m.Content {
				published = published || p.ToolResponse.Name == "publish" && p.ToolResponse.Output == "published"
			}
		}
		if !published {
			t.Error("history holds no published response")
		}
	})

	t.Run("agent turns compress across the session", func(t *testing.T) {
		// An inline middleware cannot ride in a prompt, whose rendered options
		// must encode as JSON, so the session's messages show the compression.
		ag := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{APIKey: apiKey}), genkit.WithExperimental())
		agent := genkitx.DefineAgent(ag, "notes",
			aix.InlinePrompt{
				ai.WithModelName(liveModel),
				ai.WithSystem("You are a concise assistant. Acknowledge notes in one short sentence."),
				ai.WithUse(&ContextCompression{
					MaxInputTokens: 500,
					Summarize:      &CompressionSummarizer{Model: ai.NewModelRef(liveSummarizer, nil), PreserveRecent: 2},
				}),
			},
			aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
		)
		turns := []string{"Remember this for later: my project codename is BLUE-HERON-7."}
		for i := range 3 {
			turns = append(turns, fmt.Sprintf("Background note %d: %s", i, reportDetails(fmt.Sprintf("note-%d", i))))
		}
		turns = append(turns, "What is my project codename? Answer with the codename only.")
		var sessionID string
		var out *aix.AgentOutput[any]
		for i, text := range turns {
			var opts []aix.InvocationOption[any]
			if sessionID != "" {
				opts = append(opts, aix.WithSessionID[any](sessionID))
			}
			var err error
			out, err = agent.RunText(ctx, text, opts...)
			if err != nil {
				t.Fatalf("turn %d: %v", i, err)
			}
			if out.Error != nil {
				t.Fatalf("turn %d failed: %v", i, out.Error)
			}
			sessionID = out.SessionID
		}
		answer := out.Message.Text()
		t.Logf("answer: %q", answer)
		if !strings.Contains(answer, "BLUE-HERON-7") {
			t.Errorf("answer = %q, want the codename", answer)
		}
		snap, err := agent.GetLatestSnapshot(ctx, sessionID)
		if err != nil {
			t.Fatal(err)
		}
		boundaries := 0
		for _, m := range snap.State.Messages {
			if _, ok := compressionMeta(m.Metadata)[ccSummary]; ok {
				boundaries++
			}
			if isPinned(m) {
				t.Errorf("session holds prompt scaffolding %v", texts([]*ai.Message{m}))
			}
		}
		if boundaries == 0 {
			t.Errorf("session messages carry no boundary, want the compression recorded")
		}
		msgs := snap.State.Messages
		if view := ResolveCompressedHistory(msgs[:len(msgs)-1]); !hasMessageFlag(view[0], ccSummaryMessage) {
			t.Errorf("resolved session = %v, want it to start with the summary", texts(view))
		}
	})

	t.Run("summary carries facts into later calls", func(t *testing.T) {
		mw := &ContextCompression{
			MaxInputTokens: 400,
			Summarize:      &CompressionSummarizer{Model: ai.NewModelRef(liveSummarizer, nil), PreserveRecent: 2},
		}
		history := []*ai.Message{
			userMsg("Remember this for later: my project codename is BLUE-HERON-7."),
			modelMsg("Noted. Your project codename is BLUE-HERON-7."),
		}
		for i := range 4 {
			history = append(history,
				userMsg(fmt.Sprintf("Background note %d: %s", i, reportDetails(fmt.Sprintf("note-%d", i)))),
				modelMsg(fmt.Sprintf("Thanks, I have read background note %d.", i)),
			)
		}
		history = append(history, userMsg("What is my project codename? Answer with the codename only."))

		rec := &viewRecorder{}
		resp, err := genkit.Generate(ctx, g, ai.WithModelName(liveModel), ai.WithMessages(history...), ai.WithUse(mw, rec.middleware()))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answer: %q, stats: %v", resp.Text(), stats(resp))
		if stats(resp)["summarized"] != true {
			t.Fatalf("stats = %v, want a summary", stats(resp))
		}
		view := rec.views[0]
		if strings.Contains(view[0].Text(), "Remember this for later") || !hasMessageFlag(view[0], ccSummaryMessage) {
			t.Errorf("view = %v, want the summary in place of the first messages", texts(view))
		}
		if !strings.Contains(resp.Text(), "BLUE-HERON-7") {
			t.Errorf("answer = %q, want the codename from the summary", resp.Text())
		}

		// The next call continues from the stamped history.
		next := append(resp.History(), userMsg("Spell the codename backwards."))
		rec.views = nil
		if _, err := genkit.Generate(ctx, g, ai.WithModelName(liveModel), ai.WithMessages(next...), ai.WithUse(mw, rec.middleware())); err != nil {
			t.Fatal(err)
		}
		if !hasMessageFlag(rec.views[0][0], ccSummaryMessage) {
			t.Errorf("second call view = %v, want it to start from the summary", texts(rec.views[0]))
		}
	})

	// A hard message cap drops the turns that recorded progress, so the tool
	// tracks progress itself and serves the next page whatever cursor the
	// model passes.
	var pagesRead int
	readPage := genkit.DefineTool(g, "readPage", "Reads the next page of the Project Alpha audit.",
		func(ctx *ai.ToolContext, in struct {
			Cursor int `json:"cursor"`
		}) (map[string]any, error) {
			if pagesRead == 4 {
				return map[string]any{"done": true, "text": "No more pages. Reply now."}, nil
			}
			pagesRead++
			return map[string]any{"page": pagesRead, "of": 4, "text": reportDetails(fmt.Sprintf("page-%d", pagesRead))}, nil
		})

	t.Run("message cap keeps a valid tool conversation", func(t *testing.T) {
		rec := &viewRecorder{}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName(liveModel),
			ai.WithSystem("Read the audit with readPage, one call per turn, until it reports no more pages. Then reply with one sentence about the audit."),
			ai.WithPrompt("Read the Project Alpha audit."),
			ai.WithTools(readPage),
			ai.WithMaxTurns(10),
			ai.WithUse(&ContextCompression{MaxMessages: 5}, rec.middleware()),
		)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answer: %q, stats: %v", resp.Text(), stats(resp))
		if stats(resp)["triggered"] != true {
			t.Fatalf("stats = %v, want the cap to apply", stats(resp))
		}
		for i, view := range rec.views {
			if len(view) > 5 {
				t.Errorf("call %d received %d messages, want at most 5", i, len(view))
			}
			if len(view) > 1 && view[1].Role != ai.RoleUser {
				t.Errorf("call %d view = %v, want a user message after the system message", i, texts(view))
			}
		}
	})
}
