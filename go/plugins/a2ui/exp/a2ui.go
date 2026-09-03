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

// Instruction placement options for [Surfaces.Instructions].
const (
	// InstructionsSystem appends A2UI capabilities to the system prompt
	// (default).
	InstructionsSystem = "system"
	// InstructionsNone injects nothing (useful if you supply your own
	// instructions).
	InstructionsNone = "none"
)

// Surfaces is the A2UI [ai.Middleware]: it lets the model stream UI surfaces
// drawn from a catalog. Add it to a generate call with
// [github.com/firebase/genkit/go/ai.WithUse].
//
// Example:
//
//	resp, err := genkit.Generate(ctx, g,
//	    ai.WithModel(m),
//	    ai.WithPrompt("show me the weather in Tokyo"),
//	    ai.WithUse(&a2uix.Surfaces{}), // defaults to the bundled basic catalog
//	)
//
// Middleware ordering: A2UI keeps per-turn streaming state (a stream parser and
// its minted surface ids) for the model call it wraps. Place any retrying or
// fallback middleware (which re-invokes the model) OUTSIDE A2UI so each attempt
// gets a fresh A2UI turn, i.e. WithUse(retry, &a2uix.Surfaces{}) rather than
// WithUse(&a2uix.Surfaces{}, retry). WithUse(A, B) means A wraps B.
//
// Every field is per-call configuration; the [A2UI] plugin only registers the
// middleware by name and carries no settings of its own.
type Surfaces struct {
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
	CatalogID string `json:"catalogId,omitempty" jsonschema_description:"Id of a catalog registered with LoadCatalog, resolved from the registry at call time. Defaults to \"basic\", the bundled basic catalog."`

	// Instructions controls where the catalog's capabilities are injected.
	// InstructionsSystem (default) appends A2UI instructions to the system
	// prompt; InstructionsNone injects nothing.
	Instructions string `json:"instructions,omitempty" jsonschema:"enum=system,enum=none" jsonschema_description:"Where the catalog's capabilities are injected. \"system\" appends A2UI instructions to the system prompt; \"none\" injects nothing. Defaults to \"system\"."`

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
	Validate ValidateMode `json:"validate,omitempty" jsonschema:"enum=strict,enum=warn,enum=off" jsonschema_description:"Validation of emitted envelopes against the catalog. \"warn\" logs and drops bad blocks; \"strict\" returns an error; \"off\" skips checking. This checks structure and component names only, never prop values. Defaults to \"warn\"."`

	// SurfaceID sets the surface-id policy. Provide a fixed id to reuse for
	// every surface; leave empty for a fresh UUID per surface.
	SurfaceID string `json:"surfaceId,omitempty" jsonschema_description:"Fixed id to reuse for every surface. Defaults to a fresh UUID per surface."`

	// Version is the protocol version stamped on emitted envelopes. Defaults to
	// [DefaultVersion].
	Version string `json:"version,omitempty" jsonschema_description:"Protocol version stamped on emitted envelopes. Defaults to \"v0.9\"."`
}

// Name returns the middleware's stable identifier.
func (c *Surfaces) Name() string { return provider }

// New produces the per-call [ai.Hooks] bundle that implements the A2UI
// integration.
func (c *Surfaces) New(ctx context.Context) (*ai.Hooks, error) {
	explicitCatalog := c.Catalog
	catalogID := c.CatalogID
	validate := cmp.Or(c.Validate, ValidateWarn)
	if !validValidateModes[validate] {
		return nil, fmt.Errorf(
			"a2ui: invalid Validate %q; want one of %q, %q, %q",
			validate, ValidateStrict, ValidateWarn, ValidateOff)
	}
	version := cmp.Or(c.Version, DefaultVersion)
	if !supportedVersions[version] {
		return nil, fmt.Errorf(
			"a2ui: unsupported Version %q; want one of %s",
			version, strings.Join(supportedVersionList(), ", "))
	}
	instructions := cmp.Or(c.Instructions, InstructionsSystem)
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

// A2UI provides the [Surfaces] middleware as a Genkit plugin, so it appears in
// the Dev UI and can be referenced by name (for example from a prompt file's
// `use:` list). Registering the plugin is optional: the middleware works when
// passed directly to [ai.WithUse]. Register it with
// [github.com/firebase/genkit/go/genkit.WithPlugins] during Init:
//
//	g := genkit.Init(ctx, genkit.WithPlugins(&a2uix.A2UI{}))
//
// The plugin carries no settings; every option lives on the per-call [Surfaces].
type A2UI struct{}

// Name returns the plugin's unique identifier, which is also the registered
// name of the [Surfaces] middleware.
func (p *A2UI) Name() string { return provider }

// Init implements [api.Plugin]. A2UI registers no actions.
func (p *A2UI) Init(ctx context.Context) []api.Action { return nil }

// Middlewares implements [ai.MiddlewarePlugin], exposing the [Surfaces]
// middleware descriptor to Genkit.
func (p *A2UI) Middlewares(ctx context.Context) ([]*ai.MiddlewareDesc, error) {
	return []*ai.MiddlewareDesc{
		ai.NewMiddleware(
			"Adds A2UI (Agent-to-UI) streaming UI support: injects catalog "+
				"capabilities into the prompt and rewrites emitted UI blocks into "+
				"a2ui data parts.",
			&Surfaces{},
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

	// flushHeld drains whatever the parser is still holding (a withheld prose
	// tail, or an unterminated trailing block) and appends it. Called at every
	// non-text boundary and once at the end.
	flushHeld := func() error {
		tail, err := parser.flush()
		if err != nil {
			return err
		}
		newContent = append(newContent, partsFromSegments(tail)...)
		return nil
	}

	for _, part := range resp.Message.Content {
		if part.IsText() && part.Text != "" {
			// Push WITHOUT flushing between consecutive text parts so an a2ui
			// block that spans several adjacent text parts is stitched back
			// together. The model's final message is not guaranteed to coalesce
			// adjacent text: the Gemini plugin, for instance, aggregates a turn
			// into ~30 separate text parts (fence, JSON body split many ways,
			// close fence, then a trailing empty-text part carrying the thought
			// signature), so a per-part flush would reset the parser mid-block
			// and leak the whole surface back out as raw prose. This mirrors the
			// streaming path, which shares one parser across all chunks and
			// flushes only once at the end.
			segments, err := parser.push(part.Text)
			if err != nil {
				return nil, err
			}
			newContent = append(newContent, partsForTextPart(part, segments)...)
		} else {
			// A non-text part (e.g. a toolRequest) or an empty-text part (e.g.
			// the trailing thought-signature carrier) is a boundary: flush any
			// held tail so it lands before this part, preserving order. This is
			// what a tool-calling turn [Text("Checking the weather."),
			// toolRequest] relies on so the prose is not reordered behind the
			// toolRequest. Then carry the part through untouched so its metadata
			// (thought signatures, content type) survives.
			if err := flushHeld(); err != nil {
				return nil, err
			}
			newContent = append(newContent, part)
		}
	}
	if err := flushHeld(); err != nil {
		return nil, err
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
	var messages []*ai.Message
	for _, message := range req.Messages {
		if message == nil || message.Content == nil {
			messages = append(messages, message)
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
			messages = append(messages, message)
			continue
		}
		changed = true
		// Drop a message that sanitizing emptied out. This happens when its only
		// content was an a2ui part whose envelopes all summarized to nothing
		// (e.g. an empty or all-unrecognized envelope array). Sending empty
		// content downstream would make providers like Gemini and Vertex reject
		// the request, so skip the message entirely instead.
		if len(content) == 0 {
			continue
		}
		msgCopy := message.Clone()
		msgCopy.Content = content
		messages = append(messages, msgCopy)
	}
	if !changed {
		return req
	}
	newReq := *req
	newReq.Messages = messages
	return &newReq
}

// summarizeEnvelopes converts an array of a2ui envelopes from an inbound message
// part back into model-readable text — the inverse of the outbound
// block-to-part transform.
//
// The two envelope kinds are handled differently on purpose:
//
//   - Assistant-authored surface envelopes (createSurface, updateComponents,
//     updateDataModel, deleteSurface) are reconstructed as the canonical a2ui
//     fenced block the model originally emitted. Replaying a prior turn's
//     surface as history therefore shows the model its own UI output in the
//     exact format it is asked to produce, reinforcing correct behavior.
//     (Summarizing it to a sentinel like [rendered UI surface] instead taught
//     the model to emit that literal string in place of a real block.)
//   - Client-synthesized action envelopes never had a block form, so they become
//     a short text summary the model can reason about.
//
// Consecutive surface envelopes are grouped into a single block (one surface is
// usually several envelopes: create + update(s)). Unknown envelope shapes are
// dropped, so an all-unrecognized (or empty) envelope array summarizes to an
// empty string; sanitizeInboundA2UI then drops the emptied message rather than
// sending empty content downstream.
func summarizeEnvelopes(envelopes []Envelope) string {
	var out []string
	var pendingSurface []Envelope

	flushSurface := func() {
		if len(pendingSurface) == 0 {
			return
		}
		// Keep the real surface ids verbatim. The model may not reuse them for a
		// NEW surface: the parser forces a fresh id onto every createSurface
		// block (see forceSurfaceID), so a copied id can't overwrite a prior
		// surface. Keeping the real ids lets the model correlate a replayed
		// action ([UI action ... on surface <id>]) with the surface it targeted,
		// which matters when several surfaces are on screen at once.
		//
		// Encode compactly (not pretty-printed): fewer tokens, and it collapses
		// the payload to a single line so the block is exactly three lines (open
		// fence, JSON, close fence). Because JSON escapes any newline inside a
		// string as \n, an A2UI Text value containing a fenced code sample can't
		// put a literal ``` at the start of a line, so it can never prematurely
		// close this block (the parser's close fence is line-anchored).
		if b, err := json.Marshal(pendingSurface); err == nil {
			out = append(out, "```a2ui\n"+string(b)+"\n```")
		}
		pendingSurface = nil
	}

	for _, e := range envelopes {
		if e == nil {
			continue
		}
		if action, ok := e["action"].(map[string]any); ok {
			// Emit any buffered surface block before the action, preserving order.
			flushSurface()
			name, _ := action["name"].(string)
			surfaceID, _ := action["surfaceId"].(string)
			ctx := ""
			if c, ok := action["context"].(map[string]any); ok && len(c) > 0 {
				if b, err := json.Marshal(c); err == nil {
					ctx = " context=" + string(b)
				}
			}
			out = append(out, fmt.Sprintf("[UI action %q on surface %s%s]", name, surfaceID, ctx))
		} else if hasSurfaceEnvelope(e) {
			pendingSurface = append(pendingSurface, e)
		}
	}
	flushSurface()
	return strings.Join(out, "\n")
}

// hasSurfaceEnvelope reports whether e is an assistant-authored surface envelope
// (createSurface, updateComponents, updateDataModel, or deleteSurface).
func hasSurfaceEnvelope(e Envelope) bool {
	for _, key := range []string{"createSurface", "updateComponents", "updateDataModel", "deleteSurface"} {
		if v, ok := e[key]; ok && v != nil {
			return true
		}
	}
	return false
}
