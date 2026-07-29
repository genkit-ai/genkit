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
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
)

// ValidateMode controls how the parser handles malformed or invalid envelopes.
type ValidateMode string

const (
	// ValidateStrict throws (returns an error) on malformed JSON or unknown
	// components.
	ValidateStrict ValidateMode = "strict"
	// ValidateWarn logs a warning and drops the offending block/envelope,
	// keeping the rest of the turn alive. This is the default.
	ValidateWarn ValidateMode = "warn"
	// ValidateOff passes envelopes through unchecked.
	ValidateOff ValidateMode = "off"
)

// openFenceRE matches the opening fence (```a2ui) case-insensitively.
var openFenceRE = regexp.MustCompile("(?i)```[ \t]*a2ui[ \t]*\r?\n")

// maxPartialFence is the longest prefix of an opening fence, used to hold back a
// partial fence between chunks.
const maxPartialFence = len("```a2ui\n")

// segment is a single ordered piece of parsed output: either a run of prose or
// one completed A2UI envelope batch. Segments preserve the exact source order,
// so prose that appears after a block is not reordered ahead of it.
type segment struct {
	// prose is non-empty for a prose segment.
	prose string
	// envelopes is non-nil for an envelope segment.
	envelopes []Envelope
	// isEnvelope distinguishes an (empty) envelope batch from a prose segment.
	isEnvelope bool
}

// parserOptions controls how the parser finalizes envelopes.
type parserOptions struct {
	// catalog is used to validate component references. May be nil.
	catalog *Catalog
	// validate selects the validation mode.
	validate ValidateMode
	// version is the protocol version stamped onto envelopes lacking one.
	version string
	// surfaceID produces the surface id substituted for the model's placeholder.
	surfaceID func() string
}

// streamParser is an incremental A2UI extractor. Create one per model turn,
// push text deltas as they arrive, and flush at the end to drain any trailing
// block.
type streamParser struct {
	opts    parserOptions
	buffer  string
	inBlock bool
	// currentSurfaceID is the stable surface id for the current block
	// (placeholders map to this).
	currentSurfaceID string
	hasSurfaceID     bool
}

func newStreamParser(opts parserOptions) *streamParser {
	if opts.validate == "" {
		opts.validate = ValidateWarn
	}
	if opts.version == "" {
		opts.version = DefaultVersion
	}
	return &streamParser{opts: opts}
}

// push feeds a chunk of model text, returning prose + any completed blocks.
func (p *streamParser) push(text string) ([]segment, error) {
	p.buffer += text
	return p.drain(false)
}

// flush drains any remaining buffered content at end of stream.
func (p *streamParser) flush() ([]segment, error) {
	return p.drain(true)
}

func (p *streamParser) drain(final bool) ([]segment, error) {
	var segments []segment
	// proseBuf accumulates prose across loop iterations so consecutive prose
	// runs (e.g. when a partial fence is held back) coalesce into one segment.
	var proseBuf string
	flushProse := func() {
		if proseBuf != "" {
			segments = append(segments, segment{prose: proseBuf})
			proseBuf = ""
		}
	}

	// Loop because a single push may contain multiple prose/block transitions.
	for {
		if !p.inBlock {
			loc := openFenceRE.FindStringIndex(p.buffer)
			if loc == nil {
				// No opening fence (yet). Emit prose, but hold back a tail that
				// could be the start of an incomplete opening fence, unless
				// finalizing.
				if final {
					proseBuf += p.buffer
					p.buffer = ""
				} else {
					keep := maxPartialFence
					if keep > len(p.buffer) {
						keep = len(p.buffer)
					}
					safeLen := len(p.buffer) - keep
					if safeLen > 0 {
						proseBuf += p.buffer[:safeLen]
						p.buffer = p.buffer[safeLen:]
					}
				}
				break
			}
			// Emit prose before the fence, then enter the block.
			proseBuf += p.buffer[:loc[0]]
			p.buffer = p.buffer[loc[1]:]
			p.inBlock = true
			p.currentSurfaceID = p.opts.surfaceID()
			p.hasSurfaceID = true
			continue
		}

		// In a block: look for the closing fence.
		closeIdx := strings.Index(p.buffer, "```")
		if closeIdx < 0 {
			if final {
				// Unterminated block at end of stream — try to parse what we have.
				batch, err := p.finalizeBlock(p.buffer)
				if err != nil {
					return segments, err
				}
				if batch != nil {
					flushProse()
					segments = append(segments, segment{envelopes: batch, isEnvelope: true})
				}
				p.buffer = ""
				p.inBlock = false
			}
			break
		}
		blockText := p.buffer[:closeIdx]
		p.buffer = p.buffer[closeIdx+3:]
		// Consume an optional trailing newline after the closing fence.
		p.buffer = trimLeadingNewline(p.buffer)
		p.inBlock = false
		batch, err := p.finalizeBlock(blockText)
		if err != nil {
			return segments, err
		}
		if batch != nil {
			flushProse()
			segments = append(segments, segment{envelopes: batch, isEnvelope: true})
		}
	}
	flushProse()
	return segments, nil
}

var leadingNewlineRE = regexp.MustCompile(`^[ \t]*\r?\n`)

func trimLeadingNewline(s string) string {
	return leadingNewlineRE.ReplaceAllString(s, "")
}

// reject handles a validation failure according to the configured mode: returns
// an error in strict, logs a warning in warn (the default), and is silent in
// off. Always returns (nil, err) where err is non-nil only in strict mode, so
// callers can `return p.reject(...)`.
func (p *streamParser) reject(message string) ([]Envelope, error) {
	full := "A2UI: " + message
	switch p.opts.validate {
	case ValidateOff:
		return nil, nil
	case ValidateStrict:
		return nil, fmt.Errorf("%s", full)
	default:
		slog.Warn(full + " (dropping block/envelope)")
		return nil, nil
	}
}

// finalizeBlock parses, validates, and normalizes one block's JSON into
// envelopes.
func (p *streamParser) finalizeBlock(raw string) ([]Envelope, error) {
	surfaceID := p.currentSurfaceID
	if !p.hasSurfaceID {
		surfaceID = p.opts.surfaceID()
	}
	p.currentSurfaceID = ""
	p.hasSurfaceID = false

	text := strings.TrimSpace(raw)
	if text == "" {
		return nil, nil
	}

	var parsed any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return p.reject(fmt.Sprintf("failed to parse envelope block as JSON: %v", err))
	}

	var rawEnvelopes []any
	if arr, ok := parsed.([]any); ok {
		rawEnvelopes = arr
	} else {
		rawEnvelopes = []any{parsed}
	}

	var out []Envelope
	for _, env := range rawEnvelopes {
		normalized, err := p.normalizeEnvelope(env, surfaceID)
		if err != nil {
			return nil, err
		}
		if normalized != nil {
			out = append(out, normalized)
		}
	}
	if len(out) == 0 {
		return nil, nil
	}

	// Guarantee the block opens with a createSurface, so the client always has
	// a surface before any update targets it. Models often emit only
	// updateComponents/updateDataModel on a follow-up (e.g. a "refresh") turn;
	// without this the renderer would drop those updates as "surface not
	// found". Idempotent re-creation is fine — it resets the surface.
	hasCreate := false
	for _, e := range out {
		if _, ok := e["createSurface"]; ok {
			hasCreate = true
			break
		}
	}
	if !hasCreate {
		catalogID := ""
		if p.opts.catalog != nil {
			catalogID = p.opts.catalog.ID
		}
		create := Envelope{
			"version": p.opts.version,
			"createSurface": map[string]any{
				"surfaceId": surfaceID,
				"catalogId": catalogID,
			},
		}
		out = append([]Envelope{create}, out...)
	}
	return out, nil
}

// normalizeEnvelope validates a single envelope, substitutes the real surface
// id for the placeholder, and stamps the protocol version.
func (p *streamParser) normalizeEnvelope(env any, surfaceID string) (Envelope, error) {
	m, ok := env.(map[string]any)
	if !ok {
		return p.rejectSingle("envelope must be an object.")
	}
	version, _ := m["version"].(string)
	if version == "" {
		version = p.opts.version
	}

	swapSurfaceID := func(payload map[string]any) {
		if payload == nil {
			return
		}
		sid, ok := payload["surfaceId"].(string)
		if !ok || sid == "" || sid == surfaceIDPlaceholder {
			payload["surfaceId"] = surfaceID
		}
	}

	if cs, ok := m["createSurface"].(map[string]any); ok {
		swapSurfaceID(cs)
		return Envelope{"version": version, "createSurface": cs}, nil
	}
	if uc, ok := m["updateComponents"].(map[string]any); ok {
		swapSurfaceID(uc)
		if p.opts.validate != ValidateOff {
			if msg := p.validateComponents(uc["components"]); msg != "" {
				return p.rejectSingle(msg)
			}
		}
		return Envelope{"version": version, "updateComponents": uc}, nil
	}
	if ud, ok := m["updateDataModel"].(map[string]any); ok {
		swapSurfaceID(ud)
		return Envelope{"version": version, "updateDataModel": ud}, nil
	}
	if ds, ok := m["deleteSurface"].(map[string]any); ok {
		swapSurfaceID(ds)
		return Envelope{"version": version, "deleteSurface": ds}, nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return p.rejectSingle(fmt.Sprintf("unknown envelope type (keys: %s).", strings.Join(keys, ", ")))
}

// rejectSingle is reject for a single envelope; it returns (nil, err) in strict
// mode and (nil, nil) otherwise so the envelope is dropped.
func (p *streamParser) rejectSingle(message string) (Envelope, error) {
	_, err := p.reject(message)
	return nil, err
}

// validateComponents ensures every component references a known catalog
// component. Returns an error message describing the first problem found, or ""
// if valid.
func (p *streamParser) validateComponents(components any) string {
	catalog := p.opts.catalog
	if catalog == nil {
		return ""
	}
	arr, ok := components.([]any)
	if !ok {
		return "updateComponents.components must be an array."
	}
	known := catalog.componentNames()
	hasRoot := false
	for _, c := range arr {
		if cm, ok := c.(map[string]any); ok {
			if id, _ := cm["id"].(string); id == "root" {
				hasRoot = true
				break
			}
		}
	}
	if !hasRoot {
		return `component list must contain a component id "root".`
	}
	for _, c := range arr {
		cm, ok := c.(map[string]any)
		if !ok {
			return `every component needs a "component" type name.`
		}
		name, ok := cm["component"].(string)
		if !ok {
			return `every component needs a "component" type name.`
		}
		if !known[name] {
			return fmt.Sprintf("component %q is not in catalog %q.", name, catalog.ID)
		}
	}
	return ""
}
