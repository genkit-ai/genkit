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

package a2ui

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/uuid"
)

// provider is the plugin/middleware namespace.
const provider = "a2ui"

// Instruction placement options for [Config.Instructions].
const (
	// InstructionsSystem appends A2UI capabilities to the system prompt
	// (default).
	InstructionsSystem = "system"
	// InstructionsNone injects nothing (useful if you supply your own
	// instructions).
	InstructionsNone = "none"
)

// Config is the configuration for the A2UI [ai.Middleware]. Add it to a generate
// call with [github.com/firebase/genkit/go/ai.WithUse].
//
// Example:
//
//	resp, err := genkit.Generate(ctx, g,
//	    ai.WithModel(m),
//	    ai.WithPrompt("show me the weather in Tokyo"),
//	    ai.WithUse(&a2ui.Config{}), // defaults to the bundled basic catalog
//	)
//
// Middleware ordering: A2UI keeps per-turn streaming state (a stream parser and
// its minted surface ids) for the model call it wraps. Place any retrying or
// fallback middleware (which re-invokes the model) OUTSIDE A2UI so each attempt
// gets a fresh A2UI turn, i.e. WithUse(retry, &a2ui.Config{}) rather than
// WithUse(&a2ui.Config{}, retry). WithUse(A, B) means A wraps B.
type Config struct {
	// Catalog describes what the agent may render, provided inline. When set it

	// takes precedence over CatalogID. Not serialized, so it is only honored for
	// code-defined use (not JSON/Dev-UI dispatch); prefer CatalogID with
	// [LoadCatalog] for a registry-backed catalog that also survives dispatch
	// and appears in the Dev UI.
	Catalog *Catalog `json:"-"`

	// CatalogID references a catalog registered with [LoadCatalog] by its id.
	// The middleware resolves it from the registry at call time. Defaults to
	// [DefaultCatalogID] (the bundled basic catalog) when neither Catalog nor
	// CatalogID is set.
	CatalogID string `json:"catalogId,omitempty"`

	// Instructions controls where the catalog's capabilities are injected.
	// InstructionsSystem (default) appends A2UI instructions to the system
	// prompt; InstructionsNone injects nothing.
	Instructions string `json:"instructions,omitempty"`

	// Validate controls validation of emitted envelopes against the catalog.
	// ValidateWarn (default) logs and drops bad blocks; ValidateStrict returns
	// an error; ValidateOff skips checking. An unrecognized value is rejected by
	// New rather than silently downgraded.
	//
	// This validates envelope structure and component type names against the
	// catalog only. It is a well-formedness check, not sanitization: even under
	// ValidateStrict, model-controlled values (an Image's url, a Text's inline
	// Markdown, any other prop) pass through untouched. Prop sanitization is the
	// renderer/catalog's responsibility, and hosts should CSP-restrict remote
	// sources. See the "Security and the trust boundary" section of the README.
	Validate ValidateMode `json:"validate,omitempty"`

	// SurfaceID sets the surface-id policy. Provide a fixed id to reuse for
	// every surface; leave empty for a fresh UUID per surface.
	SurfaceID string `json:"surfaceId,omitempty"`

	// Version is the protocol version stamped on emitted envelopes. Defaults to
	// [DefaultVersion].
	Version string `json:"version,omitempty"`
}

// Name returns the middleware's stable identifier.
func (c *Config) Name() string { return provider }

// New produces the per-call [ai.Hooks] bundle that implements the A2UI
// integration.
func (c *Config) New(ctx context.Context) (*ai.Hooks, error) {
	explicitCatalog := c.Catalog
	catalogID := c.CatalogID
	validate := c.Validate
	if validate == "" {
		validate = ValidateWarn
	}
	if !validValidateModes[validate] {
		return nil, fmt.Errorf(
			"a2ui: invalid Validate %q; want one of %q, %q, %q",
			validate, ValidateStrict, ValidateWarn, ValidateOff)
	}
	version := c.Version
	if version == "" {
		version = DefaultVersion
	}
	if !supportedVersions[version] {
		return nil, fmt.Errorf(
			"a2ui: unsupported Version %q; want one of %s",
			version, strings.Join(supportedVersionList(), ", "))
	}
	instructions := c.Instructions
	if instructions == "" {
		instructions = InstructionsSystem
	}
	if instructions != InstructionsSystem && instructions != InstructionsNone {
		return nil, fmt.Errorf(
			"a2ui: invalid Instructions %q; want %q or %q",
			instructions, InstructionsSystem, InstructionsNone)
	}
	nextSurfaceID := surfaceIDFactory(c.SurfaceID)

	wrapModel := func(ctx context.Context, params *ai.ModelParams, next ai.ModelNext) (*ai.ModelResponse, error) {
		// Resolve the catalog for this turn: an explicit inline catalog wins,
		// otherwise look CatalogID up from the registry (via the Genkit instance
		// seeded into the context), falling back to the bundled basic catalog.
		catalog, err := resolveCatalog(genkit.FromContext(ctx), explicitCatalog, catalogID)
		if err != nil {
			return nil, err
		}

		// Share surface ids between the streamed parse and the final-message
		// parse of this single turn, so the same surface gets the same id in
		// both (see replayableSurfaceIDs).
		surfaceIDs := replayableSurfaceIDs(nextSurfaceID)

		// 0) Sanitize any inbound a2ui data parts (e.g. a surface action sent
		//    back as the next turn, or replayed history) into model-readable
		//    text, so the underlying model's converter never sees the a2ui mime
		//    type.
		params.Request = sanitizeInboundA2UI(params.Request)

		// 1) Inject catalog instructions into the system prompt.
		if instructions != InstructionsNone {
			params.Request = injectInstructions(params.Request, catalog)
		}

		// 2) Wrap the streaming callback so streamed text is split into prose
		//    deltas + whole a2ui parts as blocks complete.
		streamParser := newStreamParser(parserOptions{
			catalog:   catalog,
			validate:  validate,
			version:   version,
			surfaceID: surfaceIDs.next,
		})
		origCallback := params.Callback
		if origCallback != nil {
			params.Callback = func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
				transformed, err := transformChunk(chunk, streamParser)
				if err != nil {
					return err
				}
				if transformed == nil {
					return nil
				}
				return origCallback(ctx, transformed)
			}
		}

		// 3) Run the downstream model.
		resp, err := next(ctx, params)
		if err != nil {
			return resp, err
		}

		// A turn that finished abnormally (blocked, aborted, interrupted,
		// other) may carry partial, half-emitted text whose a2ui block never
		// completed. Core deliberately skips output parsing for those finishes;
		// mirror that here so a malformed partial block does not turn a
		// recoverable abnormal finish into an opaque strict-mode parse error
		// that discards FinishReason, FinishMessage, and Usage. Return the
		// response untouched.
		if resp != nil && isAbnormalFinish(resp.FinishReason) {
			return resp, nil
		}

		// Flush the stream parser so the last withheld prose tail (the parser
		// holds back up to a partial opening fence) and any unterminated
		// trailing block still reach the streaming consumer. Without this,
		// clients that render purely from stream deltas would show truncated
		// prose / miss a final block (the aggregated message recovers it, but
		// the stream would not). On a flush error, still return the committed
		// response alongside the error rather than discarding it.
		if origCallback != nil {
			tail, err := streamParser.flush()
			if err != nil {
				return resp, err
			}
			if parts := partsFromSegments(tail); len(parts) > 0 {
				if err := origCallback(ctx, &ai.ModelResponseChunk{Content: parts}); err != nil {
					return resp, err
				}
			}
		}

		// 4) Transform the final message. The final parse replays the same
		//    surface ids the stream minted. On a parse error (strict mode),
		//    return the original response alongside the error so callers keep
		//    the model's committed state.
		surfaceIDs.reset()
		out, err := transformResponse(resp, catalog, validate, version, surfaceIDs.replayNext)
		if err != nil {
			return resp, err
		}
		return out, nil
	}

	return &ai.Hooks{WrapModel: wrapModel}, nil
}

// Plugin provides A2UI as a Genkit plugin so the middleware appears in the Dev
// UI and can be referenced by name. Registering the plugin is optional: the
// middleware works when passed directly to [ai.WithUse]. Register it with
// [github.com/firebase/genkit/go/genkit.WithPlugins] during Init.
type Plugin struct{}

// Name returns the plugin's unique identifier.
func (p *Plugin) Name() string { return provider }

// Init implements the plugin interface. A2UI registers no actions.
func (p *Plugin) Init(ctx context.Context) []api.Action { return nil }

// Middlewares exposes the A2UI middleware descriptor to Genkit.
func (p *Plugin) Middlewares(ctx context.Context) ([]*ai.MiddlewareDesc, error) {
	return []*ai.MiddlewareDesc{
		ai.NewMiddleware(
			"Adds A2UI (Agent-to-UI) streaming UI support: injects catalog "+
				"capabilities into the prompt and rewrites emitted UI blocks into "+
				"a2ui data parts.",
			&Config{},
		),
	}, nil
}

// surfaceIDFactory resolves the configured surface-id policy into a factory. A
// fixed policy always returns that id; an empty policy mints a fresh UUID per
// surface.
func surfaceIDFactory(policy string) func() string {
	if policy != "" {
		return func() string { return policy }
	}
	return func() string { return uuid.NewString() }
}

// replayableSurfaceIDs wraps a surface-id factory so a single model turn's
// streamed parse and its final-message parse mint the same surface ids.
//
// A turn is parsed twice: once incrementally over the streamed chunks, and once
// over the aggregated final message. Each parse pulls surface ids from the
// factory. With the default UUID policy those two parses would otherwise produce
// different ids for the same surface. So while streaming we generate and record
// ids in order (next); before re-parsing the final message we reset, then
// replayNext hands back the recorded ids in the same order (only generating a
// fresh id if the final parse yields more blocks than the stream did).
type surfaceIDReplay struct {
	// mu guards generated/cursor. In practice the streamed parse and the
	// final-message parse of a turn never overlap, but the lock keeps this safe
	// if a transport ever delivers chunks from multiple goroutines.
	mu        sync.Mutex
	base      func() string
	generated []string
	cursor    int
}

func replayableSurfaceIDs(base func() string) *surfaceIDReplay {
	return &surfaceIDReplay{base: base}
}

func (r *surfaceIDReplay) next() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.nextLocked()
}

// nextLocked mints and records a fresh id. The caller must hold r.mu.
func (r *surfaceIDReplay) nextLocked() string {
	id := r.base()
	r.generated = append(r.generated, id)
	return id
}

func (r *surfaceIDReplay) replayNext() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.cursor < len(r.generated) {
		id := r.generated[r.cursor]
		r.cursor++
		return id
	}
	return r.nextLocked()
}

func (r *surfaceIDReplay) reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cursor = 0
}

// isAbnormalFinish reports whether a finish reason means the model stopped
// without a clean, complete answer (blocked, aborted, interrupted, other).
// Mirrors ai's unexported FinishReason.isAbnormal so the middleware skips
// parsing exactly the turns core skips output parsing for.
func isAbnormalFinish(fr ai.FinishReason) bool {
	switch fr {
	case ai.FinishReasonBlocked, ai.FinishReasonAborted, ai.FinishReasonInterrupted, ai.FinishReasonOther:
		return true
	default:
		return false
	}
}

// partsFromSegments turns parsed segments into ordered prose + a2ui parts,
// preserving the exact source order.
func partsFromSegments(segments []segment) []*ai.Part {
	var out []*ai.Part
	for _, seg := range segments {
		if seg.isEnvelope {
			out = append(out, newPart(seg.envelopes))
		} else if seg.prose != "" {
			out = append(out, ai.NewTextPart(seg.prose))
		}
	}
	return out
}

// partsForTextPart turns the segments a single source text part produced into
// output parts. When the part carried no a2ui block (the parser yielded exactly
// its own text back as one prose run), the original part is returned untouched
// so its Metadata and ContentType survive — critical for Gemini thought
// signatures (attached as Metadata["signature"] on a plain text part and read
// back on the next request; losing them degrades or breaks a thinking model in
// a tool loop) and to avoid re-typing an application/json text part to
// plain/text. Only a part that actually contained a fence is rebuilt.
func partsForTextPart(src *ai.Part, segments []segment) []*ai.Part {
	if len(segments) == 1 && !segments[0].isEnvelope && segments[0].prose == src.Text {
		return []*ai.Part{src}
	}
	return partsFromSegments(segments)
}

// injectInstructions appends A2UI instructions to (or creates) the system
// message.
func injectInstructions(req *ai.ModelRequest, catalog *Catalog) *ai.ModelRequest {
	text := RenderCatalogInstructions(catalog)
	newReq := *req
	newReq.Messages = append([]*ai.Message(nil), req.Messages...)

	for i, msg := range newReq.Messages {
		if msg == nil || msg.Role != ai.RoleSystem {
			continue
		}
		msgCopy := msg.Clone()
		msgCopy.Content = append(msgCopy.Content, ai.NewTextPart("\n\n"+text))
		newReq.Messages[i] = msgCopy
		return &newReq
	}

	newReq.Messages = append(
		[]*ai.Message{ai.NewSystemMessage(ai.NewTextPart(text))},
		newReq.Messages...,
	)
	return &newReq
}

// transformChunk transforms a single streamed chunk; returns nil if there is
// nothing to emit. The parser is shared across chunks (streaming state must
// persist), so this does not flush; the middleware flushes once after the model
// call to drain the final withheld tail.
func transformChunk(chunk *ai.ModelResponseChunk, parser *streamParser) (*ai.ModelResponseChunk, error) {
	if chunk == nil || len(chunk.Content) == 0 {
		return chunk, nil
	}
	var newContent []*ai.Part
	for _, part := range chunk.Content {
		if part.IsText() && part.Text != "" {
			segments, err := parser.push(part.Text)
			if err != nil {
				return nil, err
			}
			newContent = append(newContent, partsForTextPart(part, segments)...)
		} else {
			newContent = append(newContent, part)
		}
	}
	if len(newContent) == 0 {
		return nil, nil
	}
	newChunk := *chunk
	newChunk.Content = newContent
	return &newChunk, nil
}

// transformResponse transforms the final response message: prose text + a2ui
// parts.
func transformResponse(resp *ai.ModelResponse, catalog *Catalog, validate ValidateMode, version string, surfaceID func() string) (*ai.ModelResponse, error) {
	if resp == nil || resp.Message == nil || resp.Message.Content == nil {
		return resp, nil
	}
	parser := newStreamParser(parserOptions{
		catalog:   catalog,
		validate:  validate,
		version:   version,
		surfaceID: surfaceID,
	})
	var newContent []*ai.Part
	for _, part := range resp.Message.Content {
		if part.IsText() {
			// Push then flush per text part (matching the JS middleware) so a
			// part's output lands in place, before any following non-text part.
			// Flushing once after the whole loop would move text the parser held
			// back from part N behind every later part: a tool-calling turn
			// [Text("Checking the weather."), toolRequest] would otherwise come
			// out as [Text("Checking the "), toolRequest, Text("weather.")],
			// reordering recorded history and the messages replayed to the model
			// next iteration. Per-part flush keeps ordering; the trade-off is
			// that an a2ui block split across two adjacent text parts of the
			// final message is not stitched back together, which does not happen
			// in practice (the aggregator coalesces adjacent text).
			pushed, err := parser.push(part.Text)
			if err != nil {
				return nil, err
			}
			flushed, err := parser.flush()
			if err != nil {
				return nil, err
			}
			segments := append(pushed, flushed...)
			newContent = append(newContent, partsForTextPart(part, segments)...)
		} else {
			newContent = append(newContent, part)
		}
	}
	newResp := *resp
	msgCopy := resp.Message.Clone()
	msgCopy.Content = newContent
	newResp.Message = msgCopy
	return &newResp, nil
}

// sanitizeInboundA2UI converts inbound a2ui data parts in the request into
// model-readable text.
//
// The a2ui data part (mime application/a2ui+json) is meaningful to the client
// renderer, but the underlying model's message converter does not understand
// it. When a rendered surface's action is sent back as the next turn's input —
// or when prior assistant turns containing surfaces are replayed as history —
// we replace those parts with a compact text summary so the model can reason
// about them.
func sanitizeInboundA2UI(req *ai.ModelRequest) *ai.ModelRequest {
	changed := false
	messages := make([]*ai.Message, len(req.Messages))
	for i, message := range req.Messages {
		messages[i] = message
		if message == nil || message.Content == nil {
			continue
		}
		msgChanged := false
		var content []*ai.Part
		for _, part := range message.Content {
			if IsPart(part) {
				msgChanged = true
				envelopes := EnvelopesFromParts([]*ai.Part{part})
				if text := summarizeEnvelopes(envelopes); text != "" {
					content = append(content, ai.NewTextPart(text))
				}
			} else {
				content = append(content, part)
			}
		}
		if !msgChanged {
			continue
		}
		msgCopy := message.Clone()
		msgCopy.Content = content
		messages[i] = msgCopy
		changed = true
	}
	if !changed {
		return req
	}
	newReq := *req
	newReq.Messages = messages
	return &newReq
}

// summarizeEnvelopes summarizes a batch of a2ui envelopes / actions into a short
// text string.
func summarizeEnvelopes(envelopes []Envelope) string {
	var lines []string
	seen := map[string]bool{}
	add := func(line string) {
		if !seen[line] {
			seen[line] = true
			lines = append(lines, line)
		}
	}
	for _, e := range envelopes {
		if action, ok := e["action"].(map[string]any); ok {
			name, _ := action["name"].(string)
			surfaceID, _ := action["surfaceId"].(string)
			ctx := ""
			if c, ok := action["context"].(map[string]any); ok && len(c) > 0 {
				if b, err := json.Marshal(c); err == nil {
					ctx = " context=" + string(b)
				}
			}
			add(fmt.Sprintf("[UI action %q on surface %s%s]", name, surfaceID, ctx))
		} else if cs, ok := e["createSurface"].(map[string]any); ok {
			surfaceID, _ := cs["surfaceId"].(string)
			add(fmt.Sprintf("[UI surface %s created]", surfaceID))
		} else if _, ok := e["updateComponents"]; ok {
			add("[rendered UI surface]")
		} else if _, ok := e["updateDataModel"]; ok {
			add("[rendered UI surface]")
		} else if _, ok := e["deleteSurface"]; ok {
			add("[rendered UI surface]")
		}
	}
	return strings.Join(lines, " ")
}
