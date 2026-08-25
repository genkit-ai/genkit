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
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
)

// CompressionMetadataKey is the message-metadata key under which
// [ContextCompression] records its state. Client applications can read it to
// render compression affordances without any Genkit dependency:
//
//   - Every model message gains {"inputTokens": N}, the provider-reported
//     input token count of the call that produced it. This is both the
//     compression trigger signal and a per-turn context-size indicator.
//   - The last message covered by a compaction gains {"summary": "...",
//     "stats": {...}}. A client can render a "history was compacted here"
//     divider after that message; every message stays in the history.
//
// The newest message carrying a "summary" entry (the compaction boundary)
// determines what the model sees: system messages, then the summary as a
// synthetic user message, then every message after the boundary. Older
// boundary stamps remain in place as historical markers.
const CompressionMetadataKey = "contextCompression"

// promptScaffoldKey marks messages rendered fresh from a prompt template each
// turn (see the agent runtime in genkit/exp). Such messages are not durable
// history: they are always preserved in the view, excluded from compaction
// coverage, and never chosen as a boundary, since a stamp on one would be
// lost on the next render.
const promptScaffoldKey = "_genkit_prompt"

// Default tuning values, matching the JS contextCompression middleware where
// a counterpart exists.
const (
	defaultPreserveRecent         = 6
	defaultMaxToolResponseChars   = 400_000
	defaultDedupeKeepRecent       = 1
	defaultTruncatePreserveRecent = 2

	// estCharsPerToken is the character-to-token ratio used when no
	// provider-reported usage is available yet. Deliberately rough; real
	// usage from the previous call takes over as soon as it exists.
	estCharsPerToken = 4
	// estTokensPerMediaPart is the flat estimate for a media part. Counting
	// a base64 payload as characters would wildly overestimate, and ignoring
	// media entirely would underestimate; a flat per-part cost is closest to
	// how providers bill vision input.
	estTokensPerMediaPart = 1000
)

const defaultDedupeNotice = "[Deduplicated: This tool response has been removed to save context. " +
	"A more recent response from the same tool call exists later in the conversation.]"

const defaultTruncationNotice = "[NOTE] Some earlier messages in this conversation have been removed to stay within " +
	"context limits. The most recent messages are preserved. Pay close attention to the " +
	"latest messages and any conversation summary above."

const summaryPrefix = "[Previous conversation summary — This session continues from a prior conversation " +
	"that was compressed to save context. The summary below captures all important details:]"

// conversationPlaceholder is the token in a summarization prompt that is
// replaced with the rendered conversation. A custom prompt without it gets
// the conversation appended instead, so a forgotten placeholder degrades to a
// working prompt rather than summarizing nothing.
const conversationPlaceholder = "{conversation}"

const defaultSummarizePrompt = `You are summarizing a conversation between a user, an AI assistant, and tool calls/responses. Create a comprehensive summary that preserves all information needed to continue the task seamlessly.

Before providing your summary, analyze the conversation chronologically in a <thinking> block to ensure completeness.

Your summary MUST include the following sections:

1. **Primary Request and Intent**: The user's original request and any modifications to it.
2. **Key Decisions and Facts**: Important decisions made, facts established, and data retrieved from tools.
3. **Tool Interactions**: Summary of tool calls made, their results, and any notable outputs. Include specific data values that were retrieved.
4. **Task Evolution**: If the task changed during the conversation, document the progression:
   - Original task
   - Modifications (with context for why)
   - Current active task
5. **Current State**: What was being worked on immediately before this summary. Include specifics (names, values, identifiers).
6. **Pending Work**: Any remaining tasks or next steps that were discussed but not completed.

Important guidelines:
- Preserve ALL specific data values, names, identifiers, and configuration details
- Include relevant direct quotes from tool responses that contain critical data
- Be thorough — information not in this summary will be permanently lost
- Do NOT include pleasantries or meta-commentary about the summarization process

Conversation to summarize:
{conversation}

Summary:`

// CompressionDedupeMatch selects how duplicate tool responses are identified.
type CompressionDedupeMatch string

const (
	// CompressionDedupeNameAndInput matches tool responses whose tool name
	// and request input are both identical. The input is resolved from the
	// corresponding tool request via the call ref.
	CompressionDedupeNameAndInput CompressionDedupeMatch = "name-and-input"
	// CompressionDedupeNameOnly matches tool responses by tool name alone.
	// Useful for tools that always return the latest state regardless of
	// input.
	CompressionDedupeNameOnly CompressionDedupeMatch = "name-only"
)

// CompressionSummarizer configures the LLM summarization step of
// [ContextCompression].
type CompressionSummarizer struct {
	// Model is the model used to produce summaries, typically a cheaper and
	// faster one than the primary model. Use [ai.NewModelRef] to attach
	// config; the ref's config is used verbatim for summary calls.
	Model ai.ModelRef `json:"model" jsonschema_description:"Model used to produce summaries, typically a cheaper and faster one than the primary model. Its config is used verbatim for summary calls."`
	// Prompt overrides the default summarization prompt. The literal
	// {conversation} is replaced with a text rendering of the messages to
	// summarize; if absent, the rendering is appended to the prompt.
	Prompt string `json:"prompt,omitempty" jsonschema_description:"Custom summarization prompt. The literal {conversation} is replaced with a text rendering of the messages to summarize; if absent, the rendering is appended."`
}

// CompressionDedupe configures duplicate tool response elision in
// [ContextCompression].
type CompressionDedupe struct {
	// MatchBy selects how duplicates are identified. Defaults to
	// [CompressionDedupeNameAndInput].
	MatchBy CompressionDedupeMatch `json:"matchBy,omitempty" jsonschema_description:"How duplicate tool responses are identified. Defaults to name-and-input." jsonschema:"enum=name-and-input,enum=name-only"`
	// KeepRecent is how many of the most recent responses to keep for each
	// duplicate group. Defaults to 1 and is never lower: the newest response
	// always survives.
	KeepRecent int `json:"keepRecent,omitempty" jsonschema_description:"How many of the most recent responses to keep for each duplicate group. Defaults to 1; the newest response always survives."`
	// Notice replaces the elided response outputs. A default notice is used
	// when empty.
	Notice string `json:"notice,omitempty" jsonschema_description:"Replacement text for elided duplicate tool responses. A default notice is used when empty."`
}

// CompressionToolTruncation configures tool response truncation in
// [ContextCompression].
type CompressionToolTruncation struct {
	// MaxChars is the maximum serialized length of a tool response output;
	// longer outputs are cut at this length with a truncation marker.
	// Required: a zero value disables this strategy.
	MaxChars int `json:"maxChars" jsonschema_description:"Maximum serialized length of a tool response output; longer outputs are cut at this length with a truncation marker. Zero disables this strategy."`
	// PreserveRecent is how many of the most recent tool messages are left
	// untouched. Defaults to 2; set negative to truncate all.
	PreserveRecent int `json:"preserveRecent,omitempty" jsonschema_description:"How many of the most recent tool messages are left untouched. Defaults to 2; set negative to truncate all."`
}

// ContextCompression is a middleware that keeps the model's context window in
// check during long conversations and agentic tool loops, without ever
// discarding the caller-visible history.
//
// It hooks the Model stage. Before each model call it derives a compressed
// view of the accumulated messages and sends that view to the provider; the
// request and response the caller observes (including
// [ai.ModelResponse.History]) still carry the complete history. All state is
// recorded as message metadata under [CompressionMetadataKey]:
//
//   - Each model message is annotated with the call's reported input token
//     count. That annotation is the trigger signal for the next call, and it
//     survives wherever the messages are persisted, so compression keeps
//     working across separate Generate calls and process restarts. Until the
//     first annotation exists, a character-based token estimate stands in.
//   - When a trigger fires, older messages are folded into a summary
//     produced by the Summarizer model (or a plain truncation notice when no
//     summarizer is configured). The summary is stamped on the last message
//     it covers — the compaction boundary — and that message stays in the
//     history, so a client can render a "history was compacted here" marker
//     in place. Later compactions fold the previous summary plus the
//     messages since, so each boundary supersedes the ones before it.
//
// Independent of the triggers, three cheap strategies apply to every call
// when configured: a hard safety cap on any single tool response
// (MaxToolResponseChars), duplicate tool response elision
// (DedupeToolResponses), and tool response truncation
// (TruncateToolResponses). These rewrite only the view sent to the provider,
// never the history.
//
// System messages, and prompt-template messages an agent re-renders each
// turn, are always preserved and never summarized.
//
// Usage:
//
//	resp, err := genkit.Generate(ctx, g,
//	    ai.WithModel(m),
//	    ai.WithPrompt("Research this topic thoroughly."),
//	    ai.WithTools(searchTool),
//	    ai.WithUse(&middleware.ContextCompression{
//	        MaxInputTokens:        80_000,
//	        DedupeToolResponses:   &middleware.CompressionDedupe{},
//	        TruncateToolResponses: &middleware.CompressionToolTruncation{MaxChars: 2000},
//	        Summarizer: &middleware.CompressionSummarizer{
//	            Model: ai.NewModelRef("googleai/gemini-flash-lite-latest", nil),
//	        },
//	    }),
//	)
type ContextCompression struct {
	// MaxInputTokens triggers a compaction when the context size reaches
	// this many tokens, measured by the previous call's provider-reported
	// input tokens when available and a character-based estimate otherwise.
	// Zero disables the token trigger.
	MaxInputTokens int `json:"maxInputTokens,omitempty" jsonschema_description:"Compact when the context size reaches this many tokens, measured by the previous call's reported input tokens when available and a character-based estimate otherwise. Zero disables the token trigger."`
	// MaxMessages triggers a compaction when the view sent to the model
	// would exceed this many messages, and bounds the number of messages
	// kept at a compaction. Zero disables the message trigger.
	MaxMessages int `json:"maxMessages,omitempty" jsonschema_description:"Compact when the view sent to the model would exceed this many messages. Zero disables the message trigger."`
	// PreserveRecent is how many recent messages are kept out of a
	// compaction. Defaults to 6. When the context is far over budget the
	// window shrinks automatically (halved beyond 1.5x, floor of 2 beyond
	// 2x).
	PreserveRecent int `json:"preserveRecent,omitempty" jsonschema_description:"How many recent messages are kept out of a compaction. Defaults to 6. Shrinks automatically when the context is far over budget."`
	// Summarizer folds compacted messages into an LLM-produced summary.
	// When nil, compacted messages are replaced by a truncation notice
	// instead, and their content is no longer visible to the model.
	Summarizer *CompressionSummarizer `json:"summarizer,omitempty" jsonschema_description:"Folds compacted messages into an LLM-produced summary. When absent, compacted messages are replaced by a truncation notice instead."`
	// MaxToolResponseChars is a hard cap on the serialized length of any
	// single tool response sent to the model, applied on every call as a
	// safety net against one response consuming the context window.
	// Defaults to 400000 (roughly 100k tokens); set negative to disable.
	MaxToolResponseChars int `json:"maxToolResponseChars,omitempty" jsonschema_description:"Hard cap on the serialized length of any single tool response sent to the model, applied on every call. Defaults to 400000; set negative to disable."`
	// DedupeToolResponses elides older duplicate tool responses from the
	// view on every call, keeping the most recent. Nil disables it.
	DedupeToolResponses *CompressionDedupe `json:"dedupeToolResponses,omitempty" jsonschema_description:"Elide older duplicate tool responses from the view on every call, keeping the most recent."`
	// TruncateToolResponses truncates verbose tool response outputs in the
	// view on every call, leaving the most recent tool messages untouched.
	// Nil disables it.
	TruncateToolResponses *CompressionToolTruncation `json:"truncateToolResponses,omitempty" jsonschema_description:"Truncate verbose tool response outputs in the view on every call, leaving the most recent tool messages untouched."`
	// TruncationNotice overrides the notice inserted in place of compacted
	// messages when no Summarizer is configured.
	TruncationNotice string `json:"truncationNotice,omitempty" jsonschema_description:"Custom notice inserted in place of compacted messages when no summarizer is configured."`
}

// Name implements [ai.Middleware].
func (c ContextCompression) Name() string { return provider + "/contextCompression" }

// New implements [ai.Middleware], hooking the model stage.
func (c ContextCompression) New(ctx context.Context) (*ai.Hooks, error) {
	if c.DedupeToolResponses != nil {
		switch c.DedupeToolResponses.MatchBy {
		case "", CompressionDedupeNameAndInput, CompressionDedupeNameOnly:
		default:
			return nil, status.Errorf(status.ErrInvalidArgument,
				"contextCompression: unknown dedupe matchBy %q", c.DedupeToolResponses.MatchBy)
		}
	}
	if c.Summarizer != nil {
		if c.Summarizer.Model.Name() == "" {
			return nil, status.Errorf(status.ErrInvalidArgument,
				"contextCompression: summarizer requires a model")
		}
		// Surface a summarizer typo at the start of the call rather than many
		// turns in, when the first compaction fires. The instance is absent
		// when the raw ai package is used without genkit.Init; the same
		// lookup then runs (and can still fail) at compaction time.
		if g := genkit.FromContext(ctx); g != nil {
			if genkit.LookupModel(g, c.Summarizer.Model.Name()) == nil {
				return nil, status.Errorf(ai.ErrModelNotFound,
					"contextCompression: summarizer model %q not found", c.Summarizer.Model.Name())
			}
		}
	}
	return &ai.Hooks{WrapModel: c.wrapModel}, nil
}

func (c *ContextCompression) preserveRecent() int {
	if c.PreserveRecent > 0 {
		return c.PreserveRecent
	}
	return defaultPreserveRecent
}

func (c *ContextCompression) maxToolResponseChars() int {
	switch {
	case c.MaxToolResponseChars > 0:
		return c.MaxToolResponseChars
	case c.MaxToolResponseChars < 0:
		return 0 // Disabled.
	default:
		return defaultMaxToolResponseChars
	}
}

func (c *ContextCompression) dedupeKeepRecent() int {
	if c.DedupeToolResponses != nil && c.DedupeToolResponses.KeepRecent > defaultDedupeKeepRecent {
		return c.DedupeToolResponses.KeepRecent
	}
	return defaultDedupeKeepRecent
}

func (c *ContextCompression) dedupeNotice() string {
	if c.DedupeToolResponses != nil && c.DedupeToolResponses.Notice != "" {
		return c.DedupeToolResponses.Notice
	}
	return defaultDedupeNotice
}

func (c *ContextCompression) truncatePreserveRecent() int {
	t := c.TruncateToolResponses
	switch {
	case t == nil:
		return defaultTruncatePreserveRecent
	case t.PreserveRecent > 0:
		return t.PreserveRecent
	case t.PreserveRecent < 0:
		return 0
	default:
		return defaultTruncatePreserveRecent
	}
}

func (c *ContextCompression) truncationNotice() string {
	if c.TruncationNotice != "" {
		return c.TruncationNotice
	}
	return defaultTruncationNotice
}

func (c *ContextCompression) summarizePrompt() string {
	if c.Summarizer != nil && c.Summarizer.Prompt != "" {
		return c.Summarizer.Prompt
	}
	return defaultSummarizePrompt
}

// wrapModel derives the compressed view, runs a compaction when a trigger
// fires, sends the view to the model, and restores the caller-visible
// request.
func (c *ContextCompression) wrapModel(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
	orig := params.Request
	view, stats := c.buildView(orig.Messages)

	if plan, ok := c.planCompaction(orig.Messages, view); ok {
		compacted, err := c.compact(ctx, orig.Messages, view, plan)
		if err != nil {
			return nil, err
		}
		if compacted {
			view, stats = c.buildView(orig.Messages)
		}
	}

	req := orig
	if stats.changed() || viewDiffers(view, orig.Messages) {
		vr := *orig
		vr.Messages = flattenView(view)
		req = &vr
		logger.Debug(ctx, "compressed model view built",
			"messages", len(orig.Messages),
			"viewMessages", len(vr.Messages),
			"toolResponsesCapped", stats.capped,
			"toolResponsesDeduplicated", stats.deduplicated,
			"toolResponsesTruncated", stats.truncated)
	}

	resp, err := next(ctx, &ai.ModelParams{Request: req, Callback: params.Callback})
	if err != nil || resp == nil {
		return resp, err
	}
	if req != orig && resp.Request != nil {
		// The model plugin stamps the request it received onto the response,
		// which History() then treats as the conversation so far. The caller
		// must see the full history, not the compressed view.
		resp.Request = orig
	}
	if resp.Message != nil && resp.Usage != nil && resp.Usage.InputTokens > 0 {
		mergeStamp(resp.Message, map[string]any{"inputTokens": resp.Usage.InputTokens})
	}
	return resp, nil
}

// viewEntry is one message of the compressed view. origIndex is the message's
// index in the original request, or -1 for the synthetic summary message. msg
// is the original pointer until a strategy needs to rewrite it, at which
// point it becomes a clone; original messages are never mutated except for
// metadata stamps under [CompressionMetadataKey].
type viewEntry struct {
	origIndex int
	msg       *ai.Message
	cloned    bool
}

// mutable returns the entry's message, cloning it first so the original
// stays untouched.
func (e *viewEntry) mutable() *ai.Message {
	if !e.cloned {
		e.msg = e.msg.Clone()
		e.cloned = true
	}
	return e.msg
}

// viewStats counts the per-call cheap-strategy rewrites, for logging.
type viewStats struct {
	capped       int
	deduplicated int
	truncated    int
}

func (s viewStats) changed() bool { return s.capped+s.deduplicated+s.truncated > 0 }

// flattenView flattens a view back into a message slice.
func flattenView(view []viewEntry) []*ai.Message {
	msgs := make([]*ai.Message, len(view))
	for i, e := range view {
		msgs[i] = e.msg
	}
	return msgs
}

// viewDiffers reports whether the view is anything other than the original
// messages in order.
func viewDiffers(view []viewEntry, msgs []*ai.Message) bool {
	if len(view) != len(msgs) {
		return true
	}
	for i, e := range view {
		if e.msg != msgs[i] {
			return true
		}
	}
	return false
}

// buildView derives the messages to send to the model this call: the newest
// compaction boundary is applied (system and prompt-template messages, then
// the stored summary, then everything after the boundary), and the cheap
// strategies rewrite tool responses within the result.
func (c *ContextCompression) buildView(msgs []*ai.Message) ([]viewEntry, viewStats) {
	boundary := newestBoundary(msgs)
	view := make([]viewEntry, 0, len(msgs)+1)
	for i, m := range msgs {
		if m == nil {
			continue
		}
		if i <= boundary && !alwaysPreserved(m) {
			continue
		}
		view = append(view, viewEntry{origIndex: i, msg: m})
	}
	if boundary >= 0 {
		// Insert the summary after the preserved head, right where the
		// compacted messages were.
		at := 0
		for at < len(view) && view[at].origIndex <= boundary {
			at++
		}
		view = slices.Insert(view, at, viewEntry{origIndex: -1, msg: summaryMessage(c.boundaryText(msgs[boundary]))})
	}

	var stats viewStats
	stats.capped = applyToolResponseCap(view, c.maxToolResponseChars())
	if c.DedupeToolResponses != nil {
		stats.deduplicated = c.applyDedupe(view)
	}
	if c.TruncateToolResponses != nil && c.TruncateToolResponses.MaxChars > 0 {
		stats.truncated = c.applyTruncate(view)
	}
	return view, stats
}

// boundaryText returns the text the model sees in place of the messages a
// boundary covers: the stored summary behind a standard prefix, or the
// truncation notice for a summarizer-less compaction.
func (c *ContextCompression) boundaryText(boundary *ai.Message) string {
	if summary, _ := readStamp(boundary)["summary"].(string); summary != "" {
		return summaryPrefix + "\n" + summary
	}
	return c.truncationNotice()
}

// summaryMessage builds the synthetic user message carrying a compaction
// summary. It exists only in the view sent to the model and is marked as
// synthetic so traces are self-explanatory.
func summaryMessage(text string) *ai.Message {
	m := ai.NewUserTextMessage(text)
	m.Metadata = map[string]any{CompressionMetadataKey: map[string]any{"summaryMessage": true}}
	return m
}

// alwaysPreserved reports whether a message is never compacted away: system
// messages, and prompt-template scaffolding an agent re-renders each turn.
func alwaysPreserved(m *ai.Message) bool {
	if m == nil {
		return false
	}
	if m.Role == ai.RoleSystem {
		return true
	}
	tagged, _ := m.Metadata[promptScaffoldKey].(bool)
	return tagged
}

// readStamp returns the [CompressionMetadataKey] object on a message, or nil.
func readStamp(m *ai.Message) map[string]any {
	if m == nil || m.Metadata == nil {
		return nil
	}
	stamp, _ := m.Metadata[CompressionMetadataKey].(map[string]any)
	return stamp
}

// mergeStamp merges fields into the message's [CompressionMetadataKey]
// object, cloning the existing object so previously shared references (e.g.
// a session snapshot) are not mutated.
func mergeStamp(m *ai.Message, fields map[string]any) {
	if m.Metadata == nil {
		m.Metadata = make(map[string]any, 1)
	}
	stamp := maps.Clone(readStamp(m))
	if stamp == nil {
		stamp = make(map[string]any, len(fields))
	}
	maps.Copy(stamp, fields)
	m.Metadata[CompressionMetadataKey] = stamp
}

// newestBoundary returns the index of the newest message stamped with a
// compaction (a "summary" entry, possibly empty for summarizer-less
// compactions), or -1.
func newestBoundary(msgs []*ai.Message) int {
	for i, msg := range slices.Backward(msgs) {
		if _, ok := readStamp(msg)["summary"]; ok {
			return i
		}
	}
	return -1
}

// lastReportedInputTokens returns the input token count stamped on the
// newest model message, if any. Stamps round-trip through JSON persistence,
// so the value may arrive as any numeric type.
func lastReportedInputTokens(msgs []*ai.Message) (int, bool) {
	for _, msg := range slices.Backward(msgs) {
		if msg == nil || msg.Role != ai.RoleModel {
			continue
		}
		if v, ok := readStamp(msg)["inputTokens"]; ok {
			return asInt(v)
		}
	}
	return 0, false
}

func asInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	case json.Number:
		f, err := n.Float64()
		if err != nil {
			return 0, false
		}
		return int(f), true
	default:
		return 0, false
	}
}

// estimateTokens roughly estimates the token count of the view. Used only
// until a provider-reported count is stamped on the history.
func estimateTokens(view []viewEntry) int {
	chars, media := 0, 0
	for _, e := range view {
		for _, p := range e.msg.Content {
			switch {
			case p == nil:
			case p.IsMedia():
				media++
			case p.IsText():
				chars += len(p.Text)
			case p.IsToolRequest(), p.IsToolResponse():
				chars += len(marshalJSON(p))
			}
		}
	}
	return chars/estCharsPerToken + media*estTokensPerMediaPart
}

// marshalJSON serializes v for length checks and rendering, returning "" on
// failure rather than propagating an error from an estimation path.
func marshalJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(b)
}

// toolOutputString serializes a tool response output (or request input) the
// way a provider would see it on the wire: strings verbatim, everything else
// as JSON.
func toolOutputString(out any) string {
	if s, ok := out.(string); ok {
		return s
	}
	return marshalJSON(out)
}

// replaceToolOutput swaps a tool response part's output on a cloned part, so
// the original part is untouched.
func replaceToolOutput(e *viewEntry, partIndex int, output string) {
	m := e.mutable()
	p := m.Content[partIndex].Clone()
	tr := *p.ToolResponse
	tr.Output = output
	p.ToolResponse = &tr
	m.Content[partIndex] = p
}

// applyToolResponseCap hard-truncates any single tool response larger than
// maxChars, regardless of position. Returns the number of responses capped.
func applyToolResponseCap(view []viewEntry, maxChars int) int {
	if maxChars <= 0 {
		return 0
	}
	capped := 0
	for i := range view {
		e := &view[i]
		for j, p := range e.msg.Content {
			if !p.IsToolResponse() {
				continue
			}
			out := toolOutputString(p.ToolResponse.Output)
			if len(out) <= maxChars {
				continue
			}
			replaceToolOutput(e, j, out[:maxChars]+
				"\n\n---\n\n[TRUNCATED: Response was "+strconv.Itoa(len(out))+
				" chars but only the first "+strconv.Itoa(maxChars)+" are shown.]")
			capped++
		}
	}
	return capped
}

// dedupeKeys returns the duplicate-group key of each tool response, keyed by
// view position and part index. With name-and-input matching, the request
// input is resolved from the matching tool request via the call ref; a
// response whose request cannot be found gets a ref-unique key and is never
// elided.
func (c *ContextCompression) dedupeKeys(view []viewEntry) map[[2]int]string {
	matchBy := c.DedupeToolResponses.MatchBy
	if matchBy == "" {
		matchBy = CompressionDedupeNameAndInput
	}

	inputs := map[string]string{}
	if matchBy == CompressionDedupeNameAndInput {
		for _, e := range view {
			if e.msg.Role != ai.RoleModel {
				continue
			}
			for _, p := range e.msg.Content {
				if p.IsToolRequest() {
					req := p.ToolRequest
					inputs[req.Name+"\x00"+req.Ref] = marshalJSON(req.Input)
				}
			}
		}
	}

	keys := map[[2]int]string{}
	for i, e := range view {
		for j, p := range e.msg.Content {
			if !p.IsToolResponse() {
				continue
			}
			resp := p.ToolResponse
			if matchBy == CompressionDedupeNameOnly {
				keys[[2]int{i, j}] = resp.Name
				continue
			}
			if input, ok := inputs[resp.Name+"\x00"+resp.Ref]; ok {
				keys[[2]int{i, j}] = resp.Name + "\x00" + input
			} else {
				keys[[2]int{i, j}] = resp.Name + "\x00#ref:" + resp.Ref
			}
		}
	}
	return keys
}

// applyDedupe replaces all but the most recent KeepRecent responses of each
// duplicate group with the dedupe notice. Returns the number elided.
func (c *ContextCompression) applyDedupe(view []viewEntry) int {
	keys := c.dedupeKeys(view)
	if len(keys) == 0 {
		return 0
	}
	groups := map[string][][2]int{}
	for i, e := range view {
		for j := range e.msg.Content {
			if key, ok := keys[[2]int{i, j}]; ok {
				groups[key] = append(groups[key], [2]int{i, j})
			}
		}
	}

	keep := c.dedupeKeepRecent()
	notice := c.dedupeNotice()
	deduplicated := 0
	for _, positions := range groups {
		if len(positions) <= keep {
			continue
		}
		for _, pos := range positions[:len(positions)-keep] {
			replaceToolOutput(&view[pos[0]], pos[1], notice)
			deduplicated++
		}
	}
	return deduplicated
}

// applyTruncate truncates tool response outputs beyond MaxChars, leaving the
// most recent PreserveRecent tool messages untouched. Returns the number
// truncated.
func (c *ContextCompression) applyTruncate(view []viewEntry) int {
	maxChars := c.TruncateToolResponses.MaxChars
	preserve := c.truncatePreserveRecent()

	truncated := 0
	seenToolMsgs := 0
	for i := len(view) - 1; i >= 0; i-- {
		e := &view[i]
		if e.msg.Role != ai.RoleTool {
			continue
		}
		seenToolMsgs++
		if seenToolMsgs <= preserve {
			continue
		}
		for j, p := range e.msg.Content {
			if !p.IsToolResponse() {
				continue
			}
			out := toolOutputString(p.ToolResponse.Output)
			if len(out) <= maxChars {
				continue
			}
			replaceToolOutput(e, j, out[:maxChars]+"…[truncated]")
			truncated++
		}
	}
	return truncated
}

// compactionPlan captures one compaction decision: where the boundary lies
// and what triggered it.
type compactionPlan struct {
	// boundary is the original index of the last message the compaction
	// covers.
	boundary int
	// trigger names the trigger that fired: "inputTokens" or "maxMessages".
	trigger string
	// reading is the context-size reading that fired a token trigger.
	reading int
	// estimated reports whether reading came from the character estimate
	// rather than provider-reported usage.
	estimated bool
}

// planCompaction decides whether a compaction should run this call, and where
// its boundary lies.
func (c *ContextCompression) planCompaction(msgs []*ai.Message, view []viewEntry) (compactionPlan, bool) {
	plan := compactionPlan{}
	if c.MaxInputTokens > 0 {
		reading, known := lastReportedInputTokens(msgs)
		if !known {
			reading = estimateTokens(view)
		}
		if reading > c.MaxInputTokens {
			plan.trigger = "inputTokens"
			plan.reading = reading
			plan.estimated = !known
		}
	}
	if plan.trigger == "" && c.MaxMessages > 0 && len(view) > c.MaxMessages {
		plan.trigger = "maxMessages"
	}
	if plan.trigger == "" {
		return plan, false
	}

	keep := c.preserveRecent()
	if plan.trigger == "inputTokens" {
		keep = adjustForOvershoot(float64(plan.reading)/float64(c.MaxInputTokens), keep)
	}

	// Candidates are durable, compactable view messages: everything after
	// the previous boundary except system and prompt-template messages, and
	// except the synthetic summary.
	var candidates []int // Original indices.
	for _, e := range view {
		if e.origIndex >= 0 && !alwaysPreserved(e.msg) {
			candidates = append(candidates, e.origIndex)
		}
	}
	if c.MaxMessages > 0 {
		// Everything that is not a candidate survives a compaction, plus the
		// summary message that replaces the covered candidates.
		surviving := len(view) - len(candidates) + 1
		keep = min(keep, c.MaxMessages-surviving)
	}
	keep = max(keep, 1)
	if len(candidates)-keep < 1 {
		return plan, false // Nothing new to fold.
	}

	// The kept window must not begin with tool responses whose requests
	// would be folded away; pull the boundary back until the window starts
	// clean, keeping each tool response with the model message that
	// requested it.
	pos := len(candidates) - keep - 1
	for pos >= 0 && isToolMessage(msgs, candidates[pos+1]) {
		pos--
	}
	if pos < 0 {
		return plan, false // Nothing left to fold once the window starts clean.
	}
	plan.boundary = candidates[pos]
	return plan, true
}

func isToolMessage(msgs []*ai.Message, i int) bool {
	return i >= 0 && i < len(msgs) && msgs[i] != nil && msgs[i].Role == ai.RoleTool
}

// adjustForOvershoot shrinks the preserved window when the context is far
// over budget: halved (floor 2) beyond 1.5x, hard floor of 2 beyond 2x.
func adjustForOvershoot(overshoot float64, keep int) int {
	switch {
	case overshoot >= 2.0:
		return min(keep, 2)
	case overshoot >= 1.5:
		return max(2, keep/2)
	default:
		return keep
	}
}

// compact runs one compaction: it produces the summary text (via the
// summarizer model when configured) and stamps it with the compaction stats
// on the boundary message in the original history. A transient summarizer
// failure logs a warning and reports false, leaving this call uncompacted;
// deterministic configuration failures return an error.
func (c *ContextCompression) compact(ctx context.Context, msgs []*ai.Message, view []viewEntry, plan compactionPlan) (bool, error) {
	prevBoundary := newestBoundary(msgs)
	covered := 0
	for _, e := range view {
		if e.origIndex > prevBoundary && e.origIndex <= plan.boundary && !alwaysPreserved(e.msg) {
			covered++
		}
	}

	stats := map[string]any{
		"trigger":           plan.trigger,
		"messagesCompacted": covered,
		"summarized":        false,
	}
	if plan.trigger == "inputTokens" {
		if plan.estimated {
			stats["estimatedTokens"] = plan.reading
		} else {
			stats["inputTokens"] = plan.reading
		}
		stats["overshoot"] = math.Round(float64(plan.reading)/float64(c.MaxInputTokens)*100) / 100
	}

	summary := ""
	if c.Summarizer != nil {
		g := genkit.FromContext(ctx)
		if g == nil {
			return false, status.Errorf(status.ErrFailedPrecondition,
				"contextCompression: summarizer requires a genkit instance in context; initialize with genkit.Init")
		}
		m := genkit.LookupModel(g, c.Summarizer.Model.Name())
		if m == nil {
			return false, status.Errorf(ai.ErrModelNotFound,
				"contextCompression: summarizer model %q not found", c.Summarizer.Model.Name())
		}

		start := time.Now()
		text, err := c.summarize(ctx, m, msgs, view, prevBoundary, plan.boundary, stats)
		if err != nil {
			logger.Warn(ctx, "context compaction summarization failed, continuing uncompacted",
				"model", c.Summarizer.Model.Name(), "error", err)
			return false, nil
		}
		summary = text
		stats["summarized"] = true
		stats["summaryModel"] = c.Summarizer.Model.Name()
		stats["summaryDurationMs"] = time.Since(start).Milliseconds()
	}

	mergeStamp(msgs[plan.boundary], map[string]any{"summary": summary, "stats": stats})

	// A compaction folds prior context and, when summarizing, spends a
	// billed model call; it deserves more than debug visibility.
	logger.Info(ctx, "context compacted",
		"trigger", plan.trigger,
		"messagesCompacted", covered,
		"boundary", plan.boundary,
		"summarized", c.Summarizer != nil,
		"summaryChars", len(summary))
	return true, nil
}

// summarize renders the covered conversation and asks the summarizer model
// to fold it (together with any previous summary) into a fresh summary. All
// returned errors are transient: the caller absorbs them and proceeds
// uncompacted.
func (c *ContextCompression) summarize(ctx context.Context, m ai.Model, msgs []*ai.Message, view []viewEntry, prevBoundary, boundary int, stats map[string]any) (string, error) {
	var sb strings.Builder
	if prevBoundary >= 0 {
		if prev, _ := readStamp(msgs[prevBoundary])["summary"].(string); prev != "" {
			sb.WriteString("[Previous summary]\n")
			sb.WriteString(prev)
			sb.WriteString("\n\n[New messages]\n")
		} else {
			sb.WriteString("[Note: some earlier messages were previously dropped without summarization.]\n\n")
		}
	}
	renderConversation(&sb, view, prevBoundary, boundary)

	prompt := c.summarizePrompt()
	if strings.Contains(prompt, conversationPlaceholder) {
		prompt = strings.ReplaceAll(prompt, conversationPlaceholder, sb.String())
	} else {
		prompt = prompt + "\n\nConversation to summarize:\n" + sb.String()
	}

	resp, err := m.Generate(ctx, &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage(prompt)},
		Config:   c.Summarizer.Model.Config(),
	}, nil)
	if err != nil {
		return "", err
	}
	text := resp.Text()
	if strings.TrimSpace(text) == "" {
		return "", status.Errorf(status.ErrInternal, "summarizer returned an empty summary")
	}
	if resp.Usage != nil {
		stats["summaryInputTokens"] = resp.Usage.InputTokens
		stats["summaryOutputTokens"] = resp.Usage.OutputTokens
	}
	return text, nil
}

// renderConversation writes a plain-text rendering of the view messages whose
// original index lies in (prevBoundary, boundary], for the summarizer prompt.
// It renders from the view, not the raw history, so capped and deduplicated
// tool outputs do not blow up the summarizer's own context.
func renderConversation(sb *strings.Builder, view []viewEntry, prevBoundary, boundary int) {
	for _, e := range view {
		if e.origIndex <= prevBoundary || e.origIndex > boundary || alwaysPreserved(e.msg) {
			continue
		}
		sb.WriteString(string(e.msg.Role))
		sb.WriteString(": ")
		for i, p := range e.msg.Content {
			if i > 0 {
				sb.WriteString(" ")
			}
			switch {
			case p == nil:
			case p.IsText():
				sb.WriteString(p.Text)
			case p.IsToolRequest():
				sb.WriteString("[Tool call: ")
				sb.WriteString(p.ToolRequest.Name)
				sb.WriteString("(")
				sb.WriteString(marshalJSON(p.ToolRequest.Input))
				sb.WriteString(")]")
			case p.IsToolResponse():
				sb.WriteString("[Tool response: ")
				sb.WriteString(p.ToolResponse.Name)
				sb.WriteString(" -> ")
				sb.WriteString(toolOutputString(p.ToolResponse.Output))
				sb.WriteString("]")
			case p.IsMedia():
				sb.WriteString("[media: ")
				sb.WriteString(p.ContentType)
				sb.WriteString("]")
			}
		}
		sb.WriteString("\n")
	}
}
