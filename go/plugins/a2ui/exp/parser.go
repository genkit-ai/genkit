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
//
// The fence is deliberately NOT anchored to line start (unlike closeFenceRE): a
// model may legitimately begin the block right after inline prose on the same
// line, and the streaming parser must still catch it. The required trailing
// newline guards against a false positive from prose that merely mentions the
// fence.
var openFenceRE = regexp.MustCompile("(?i)```a2ui[ \t]*\r?\n")

// partialOpenFenceRE matches, at the very end of the buffer, the longest suffix
// that could still be completing an opening fence on the next chunk: 1-3
// backticks, then a partial "a2ui" tag, then optional trailing spaces/tabs and
// an optional "\r" awaiting its "\n". It is held back from prose so a fence
// split across chunks is never leaked.
//
// A fixed-length holdback would be wrong on two counts: openFenceRE allows
// unbounded [ \t]* padding before the newline, so the incomplete-fence suffix
// has no fixed maximum length (a padded fence like "```a2ui   \r\n" split across
// chunks would leak backticks as prose); and a byte-count holdback can slice
// mid-rune, corrupting non-ASCII prose into U+FFFD. Anchoring to the end ($)
// holds back exactly the fence suffix, which is always ASCII, so the emitted
// prose always ends on a rune boundary. Mirrors the JS parser's
// PARTIAL_OPEN_FENCE_RE.
var partialOpenFenceRE = regexp.MustCompile("(?i)(?:`|``|```(?:a(?:2(?:u(?:i[ \t]*\r?)?)?)?)?)$")

// closeFenceRE matches the closing fence: ``` at the start of a line (optionally
// indented). Anchoring to line start (mirroring openFenceRE) is important: A2UI
// Text values "may use inline Markdown", so the JSON payload can legitimately
// contain a ``` inside a string. A bare strings.Index for "```" would match that
// and truncate the block mid-JSON, dropping the whole surface. Mirrors the JS
// parser's CLOSE_FENCE_RE.
var closeFenceRE = regexp.MustCompile("(?m)^[ \t]*```")

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
	// blockScan is the offset (a line start) from which the in-block
	// close-fence search resumes. Tracking it keeps a large block arriving in
	// many small deltas close to linear rather than O(n²): each drain scans only
	// the last, still-incomplete line plus newly appended bytes instead of
	// rescanning the whole block.
	blockScan int

	// knownComponents is the catalog's component-name set, computed once so
	// validation doesn't rebuild it per updateComponents envelope. Nil when no
	// catalog is configured.
	knownComponents map[string]bool
}

func newStreamParser(opts parserOptions) *streamParser {
	if opts.validate == "" {
		opts.validate = ValidateWarn
	}
	if opts.version == "" {
		opts.version = DefaultVersion
	}
	var known map[string]bool
	if opts.catalog != nil {
		known = opts.catalog.componentNames()
	}
	return &streamParser{opts: opts, knownComponents: known}
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
					// Hold back a trailing suffix that could still be completing
					// an opening fence on the next chunk (e.g. "```a2u" or
					// "```a2ui  \r"). The suffix is always ASCII, so the emitted
					// prose ends on a rune boundary — no mid-rune split.
					keep := 0
					if m := partialOpenFenceRE.FindStringIndex(p.buffer); m != nil {
						keep = len(p.buffer) - m[0]
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
			p.blockScan = 0
			p.currentSurfaceID = p.opts.surfaceID()
			p.hasSurfaceID = true
			continue
		}

		// In a block: look for the closing fence, anchored to line start so a
		// ``` inside a JSON string (A2UI Text may contain inline Markdown) does
		// not truncate the block. Resume the search from blockScan, which always
		// sits at a line start, so a large block arriving in many deltas stays
		// close to linear instead of rescanning the whole block each delta.
		loc := closeFenceRE.FindStringIndex(p.buffer[p.blockScan:])
		if loc == nil {
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
				p.blockScan = 0
			} else {
				// Advance the scan cursor to the start of the last (still
				// incomplete) line. The close fence is line-anchored, so it can
				// only appear at a line start; resuming there avoids rescanning
				// completed lines while keeping blockScan on a real line
				// boundary (so closeFenceRE's ^ never matches a false mid-line
				// position on the next drain). A fence split across this chunk
				// boundary is still caught because its own line has not
				// completed yet.
				if nl := strings.LastIndexByte(p.buffer, '\n'); nl+1 > p.blockScan {
					p.blockScan = nl + 1
				}
			}
			break
		}
		// loc is relative to buffer[blockScan:]; the ``` is preceded by the
		// matched line-start (^ or \n) plus optional indentation. Keep the block
		// text up to that match, dropping the newline/indent that begins the
		// close-fence match; TrimSpace in finalizeBlock removes the rest.
		matchStart := p.blockScan + loc[0]
		matchEnd := p.blockScan + loc[1]
		blockText := p.buffer[:matchStart]
		p.buffer = p.buffer[matchEnd:]

		// Consume an optional trailing newline after the closing fence.
		p.buffer = trimLeadingNewline(p.buffer)
		p.inBlock = false
		p.blockScan = 0
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

	hasCreate := false
	for _, e := range out {
		if _, ok := e["createSurface"]; ok {
			hasCreate = true
			break
		}
	}

	if hasCreate {
		// A full-surface render. createSurface means "new surface" by
		// definition, so the id the model wrote is never authoritative — force
		// every envelope in this block onto the freshly-minted surfaceID, even
		// if the model copied a real id from replayed history (which would
		// otherwise reuse and overwrite that prior surface in place). This is
		// the single chokepoint that guarantees a distinct surface per render,
		// so history can keep real ids verbatim (needed to correlate actions
		// with their surface) without risking id reuse.
		for _, e := range out {
			forceSurfaceID(e, surfaceID)
		}
		// Enforce the "must contain a root" protocol rule.
		if msg := validateRoot(out); msg != "" {
			return p.reject(msg)
		}
		return out, nil
	}

	// Update-only batch. Two cases:
	//
	//  1. The model targeted the placeholder (or omitted the id), so
	//     normalizeEnvelope swapped in our freshly-minted surfaceID. That's a
	//     fresh render the model forgot to createSurface for — synthesize one so
	//     the client has a surface before the updates land, and enforce the root
	//     rule as for any full render.
	//
	//  2. The updates target an explicit, pre-existing surface id (one the model
	//     learned from a prior turn, e.g. via a summarized action). That's a
	//     genuine incremental update: do NOT synthesize a createSurface (that
	//     would reset the surface and drop the update as "surface not found" if
	//     ids disagreed), and do NOT require a root.
	targetID := surfaceID
	for _, e := range out {
		if id := envelopeSurfaceID(e); id != "" {
			targetID = id
			break
		}
	}
	isFreshRender := targetID == surfaceID
	if !isFreshRender {
		return out, nil
	}

	if msg := validateRoot(out); msg != "" {
		return p.reject(msg)
	}
	// Guarantee the block opens with a createSurface, so the client always has
	// a surface before any update targets it. Idempotent re-creation is fine —
	// it resets the surface.
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
	return out, nil
}

// forceSurfaceID overwrites the surfaceId of whichever payload e carries with
// surfaceID, unconditionally (unlike the placeholder-only swap in
// normalizeEnvelope). Used to force every envelope in a createSurface-bearing
// block onto a single freshly-minted id, so a model that copied a real id from
// replayed history can't reuse and overwrite a prior surface. Mirrors the JS
// parser's forceSurfaceId.
func forceSurfaceID(e Envelope, surfaceID string) {
	for _, key := range []string{"createSurface", "updateComponents", "updateDataModel", "deleteSurface"} {
		if payload, ok := e[key].(map[string]any); ok {
			payload["surfaceId"] = surfaceID
			return
		}
	}
}

// envelopeSurfaceID reads the surface id an envelope targets, regardless of its
// variant. Used to decide whether an update-only batch targets the
// freshly-minted surface (a fresh render the model forgot to createSurface for)
// or an explicit existing one (a genuine incremental update).
func envelopeSurfaceID(e Envelope) string {
	for _, key := range []string{"createSurface", "updateComponents", "updateDataModel", "deleteSurface"} {
		if payload, ok := e[key].(map[string]any); ok {
			if id, _ := payload["surfaceId"].(string); id != "" {
				return id
			}
		}
	}
	return ""
}

// validateRoot enforces the "RE-RENDER THE WHOLE SURFACE" protocol rule for
// full-surface renders. The v0.9 spec requires that "one of the components in
// one of the component lists MUST have an id of root" — a batch-level rule, not
// a per-message one. A split render is therefore legal: the root (and layout)
// may live in one updateComponents while its leaf components arrive in a later
// one within the same batch. So this checks that at least one updateComponents
// in the batch declares a root, not that every one does. Returns an error
// message, or "" if valid. Unlike validateComponents this is a protocol check,
// not a catalog check, so it runs regardless of whether a catalog is configured.
// Mirrors the JS parser's validateRoot.
func validateRoot(envelopes []Envelope) string {
	sawComponentList := false
	for _, e := range envelopes {
		uc, ok := e["updateComponents"].(map[string]any)
		if !ok {
			continue
		}
		arr, ok := uc["components"].([]any)
		if !ok {
			continue
		}
		sawComponentList = true
		for _, c := range arr {
			if cm, ok := c.(map[string]any); ok {
				if id, _ := cm["id"].(string); id == "root" {
					return ""
				}
			}
		}
	}
	// Only a batch that actually carries component lists must declare a root.
	if !sawComponentList {
		return ""
	}
	return `component list must contain a component id "root".`
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

	// normalized copies the incoming envelope and stamps the version, preserving
	// any forward-compatible top-level keys the middleware does not inspect. The
	// A2UI protocol is "open-ended and versioned", so rebuilding a fresh
	// {version, <kind>} map (as the JS parser historically did) would silently
	// strip anything the spec adds later; copying keeps it intact.
	normalized := func() Envelope {
		out := make(Envelope, len(m)+1)
		for k, v := range m {
			out[k] = v
		}
		out["version"] = version
		return out
	}

	if cs, ok := m["createSurface"].(map[string]any); ok {
		swapSurfaceID(cs)
		return normalized(), nil
	}
	if uc, ok := m["updateComponents"].(map[string]any); ok {
		swapSurfaceID(uc)
		if p.opts.validate != ValidateOff {
			if msg := p.validateComponents(uc["components"]); msg != "" {
				return p.rejectSingle(msg)
			}
		}
		return normalized(), nil
	}
	if ud, ok := m["updateDataModel"].(map[string]any); ok {
		swapSurfaceID(ud)
		return normalized(), nil
	}
	if ds, ok := m["deleteSurface"].(map[string]any); ok {
		swapSurfaceID(ds)
		return normalized(), nil
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
//
// This checks component type names against the catalog only. It intentionally
// does NOT enforce the "must contain a root" protocol rule — that applies only
// to full-surface renders and is enforced at the batch level in finalizeBlock
// (an incremental update to an existing surface may legitimately patch a subtree
// without re-declaring root).
func (p *streamParser) validateComponents(components any) string {
	catalog := p.opts.catalog
	if catalog == nil {
		return ""
	}
	arr, ok := components.([]any)
	if !ok {
		return "updateComponents.components must be an array."
	}
	known := p.knownComponents
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
