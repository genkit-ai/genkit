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

// Config is the configuration for the A2UI [Middleware]. Add it to a generate
// call with [github.com/firebase/genkit/go/ai.WithUse].
//
// Example:
//
//	resp, err := genkit.Generate(ctx, g,
//	    ai.WithModel(m),
//	    ai.WithPrompt("show me the weather in Tokyo"),
//	    ai.WithUse(&a2ui.Config{}), // defaults to the bundled basic catalog
//	)
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
	// an error; ValidateOff skips checking.
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
	version := c.Version
	if version == "" {
		version = DefaultVersion
	}
	instructions := c.Instructions
	if instructions == "" {
		instructions = InstructionsSystem
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
		if params.Callback != nil {
			orig := params.Callback
			params.Callback = func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
				transformed, err := transformChunk(chunk, streamParser)
				if err != nil {
					return err
				}
				if transformed == nil {
					return nil
				}
				return orig(ctx, transformed)
			}
		}

		// 3) Run the downstream model, then transform the final message. The
		//    final parse replays the same surface ids the stream minted.
		resp, err := next(ctx, params)
		if err != nil {
			return resp, err
		}
		surfaceIDs.reset()
		return transformResponse(resp, catalog, validate, version, surfaceIDs.replayNext)
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
	base      func() string
	generated []string
	cursor    int
}

func replayableSurfaceIDs(base func() string) *surfaceIDReplay {
	return &surfaceIDReplay{base: base}
}

func (r *surfaceIDReplay) next() string {
	id := r.base()
	r.generated = append(r.generated, id)
	return id
}

func (r *surfaceIDReplay) replayNext() string {
	if r.cursor < len(r.generated) {
		id := r.generated[r.cursor]
		r.cursor++
		return id
	}
	return r.next()
}

func (r *surfaceIDReplay) reset() { r.cursor = 0 }

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
// nothing to emit.
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
			newContent = append(newContent, partsFromSegments(segments)...)
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
			// Combine the streamed-push and final-flush segments so ordering
			// (prose before/after a block) is preserved in the aggregated
			// message too.
			pushed, err := parser.push(part.Text)
			if err != nil {
				return nil, err
			}
			flushed, err := parser.flush()
			if err != nil {
				return nil, err
			}
			newContent = append(newContent, partsFromSegments(append(pushed, flushed...))...)
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
