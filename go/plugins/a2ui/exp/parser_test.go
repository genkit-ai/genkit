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
	"strings"
	"testing"
)

// fixedSurfaceID returns a surface-id factory yielding a constant id.
func fixedSurfaceID(id string) func() string {
	return func() string { return id }
}

// collect concatenates prose and returns all envelope batches from segments.
func collect(segs []segment) (prose string, batches [][]Envelope) {
	for _, s := range segs {
		if s.isEnvelope {
			batches = append(batches, s.envelopes)
		} else {
			prose += s.prose
		}
	}
	return prose, batches
}

func TestParserProseOnly(t *testing.T) {
	p := newStreamParser(parserOptions{surfaceID: fixedSurfaceID("s1")})
	segs, err := p.push("Hello, world.")
	if err != nil {
		t.Fatal(err)
	}
	flushed, err := p.flush()
	if err != nil {
		t.Fatal(err)
	}
	prose, batches := collect(append(segs, flushed...))
	if prose != "Hello, world." {
		t.Errorf("prose = %q, want %q", prose, "Hello, world.")
	}
	if len(batches) != 0 {
		t.Errorf("got %d batches, want 0", len(batches))
	}
}

func TestParserExtractsBlock(t *testing.T) {
	catalog := BasicCatalog()
	p := newStreamParser(parserOptions{
		catalog:   catalog,
		validate:  ValidateWarn,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "Here is a card:\n```a2ui\n" +
		`[{"createSurface":{"surfaceId":"SURFACE_ID","catalogId":"` + catalog.ID + `"}},` +
		`{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"root","component":"Text","text":"hi"}]}}]` +
		"\n```\nDone."
	segs, err := p.push(input)
	if err != nil {
		t.Fatal(err)
	}
	flushed, err := p.flush()
	if err != nil {
		t.Fatal(err)
	}
	prose, batches := collect(append(segs, flushed...))
	if !strings.Contains(prose, "Here is a card:") || !strings.Contains(prose, "Done.") {
		t.Errorf("prose = %q, missing expected text", prose)
	}
	if len(batches) != 1 {
		t.Fatalf("got %d batches, want 1", len(batches))
	}
	batch := batches[0]
	if len(batch) != 2 {
		t.Fatalf("got %d envelopes, want 2", len(batch))
	}
	cs, _ := batch[0]["createSurface"].(map[string]any)
	if cs["surfaceId"] != "s1" {
		t.Errorf("surfaceId = %v, want s1 (placeholder should be replaced)", cs["surfaceId"])
	}
}

func TestParserInjectsCreateSurface(t *testing.T) {
	catalog := BasicCatalog()
	p := newStreamParser(parserOptions{
		catalog:   catalog,
		validate:  ValidateWarn,
		surfaceID: fixedSurfaceID("s9"),
	})
	// Only an updateComponents; the parser must prepend a createSurface.
	input := "```a2ui\n" +
		`[{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"root","component":"Text","text":"hi"}]}}]` +
		"\n```"
	segs, err := p.push(input)
	if err != nil {
		t.Fatal(err)
	}
	flushed, _ := p.flush()
	_, batches := collect(append(segs, flushed...))
	if len(batches) != 1 {
		t.Fatalf("got %d batches, want 1", len(batches))
	}
	batch := batches[0]
	if len(batch) != 2 {
		t.Fatalf("got %d envelopes, want 2 (createSurface injected)", len(batch))
	}
	if _, ok := batch[0]["createSurface"]; !ok {
		t.Errorf("first envelope should be createSurface, got %v", batch[0])
	}
}

func TestParserSplitAcrossChunks(t *testing.T) {
	catalog := BasicCatalog()
	p := newStreamParser(parserOptions{
		catalog:   catalog,
		validate:  ValidateWarn,
		surfaceID: fixedSurfaceID("s1"),
	})
	full := "prefix ```a2ui\n" +
		`[{"createSurface":{"surfaceId":"SURFACE_ID","catalogId":"c"}},` +
		`{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"root","component":"Text","text":"hi"}]}}]` +
		"\n``` suffix"

	var allSegs []segment
	// Feed one rune at a time to stress the incremental fence handling.
	for _, r := range full {
		segs, err := p.push(string(r))
		if err != nil {
			t.Fatal(err)
		}
		allSegs = append(allSegs, segs...)
	}
	flushed, _ := p.flush()
	allSegs = append(allSegs, flushed...)

	prose, batches := collect(allSegs)
	if !strings.Contains(prose, "prefix") || !strings.Contains(prose, "suffix") {
		t.Errorf("prose = %q, missing prefix/suffix", prose)
	}
	if len(batches) != 1 {
		t.Fatalf("got %d batches, want 1", len(batches))
	}
	if len(batches[0]) != 2 {
		t.Errorf("got %d envelopes, want 2", len(batches[0]))
	}
}

func TestParserStrictRejectsUnknownComponent(t *testing.T) {
	catalog := BasicCatalog()
	p := newStreamParser(parserOptions{
		catalog:   catalog,
		validate:  ValidateStrict,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "```a2ui\n" +
		`[{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"root","component":"NoSuchThing"}]}}]` +
		"\n```"
	_, err := p.push(input)
	if err == nil {
		t.Fatal("expected error for unknown component in strict mode")
	}
	if !strings.Contains(err.Error(), "NoSuchThing") {
		t.Errorf("error = %v, want mention of NoSuchThing", err)
	}
}

func TestParserWarnDropsBadJSON(t *testing.T) {
	catalog := BasicCatalog()
	p := newStreamParser(parserOptions{
		catalog:   catalog,
		validate:  ValidateWarn,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "```a2ui\nnot valid json\n```"
	segs, err := p.push(input)
	if err != nil {
		t.Fatalf("warn mode should not error, got %v", err)
	}
	flushed, _ := p.flush()
	_, batches := collect(append(segs, flushed...))
	if len(batches) != 0 {
		t.Errorf("got %d batches, want 0 (bad JSON dropped)", len(batches))
	}
}

func TestParserStrictErrorsBadJSON(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateStrict,
		surfaceID: fixedSurfaceID("s1"),
	})
	_, err := p.push("```a2ui\n{ not json }\n```")
	if err == nil {
		t.Fatal("expected error for bad JSON in strict mode")
	}
}

func TestParserMissingRootRejected(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateStrict,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "```a2ui\n" +
		`[{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"body","component":"Text","text":"hi"}]}}]` +
		"\n```"
	_, err := p.push(input)
	if err == nil || !strings.Contains(err.Error(), "root") {
		t.Fatalf("expected root-required error, got %v", err)
	}
}

// A block whose only envelope targets a real, pre-existing surface id (one the
// model learned from a prior turn) is a genuine incremental update: the parser
// must NOT prepend a createSurface (which would reset the surface and make the
// client drop the update as "surface not found").
func TestParserIncrementalUpdateToExplicitSurface(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateWarn,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "```a2ui\n" +
		`[{"updateComponents":{"surfaceId":"existing-surface","components":[{"id":"root","component":"Text","text":"patched"}]}}]` +
		"\n```"
	segs, err := p.push(input)
	if err != nil {
		t.Fatal(err)
	}
	flushed, _ := p.flush()
	_, batches := collect(append(segs, flushed...))
	if len(batches) != 1 {
		t.Fatalf("got %d batches, want 1", len(batches))
	}
	if len(batches[0]) != 1 {
		t.Fatalf("got %d envelopes, want 1 (no synthesized createSurface)", len(batches[0]))
	}
	uc, ok := batches[0][0]["updateComponents"].(map[string]any)
	if !ok {
		t.Fatalf("expected updateComponents to pass through, got %v", batches[0][0])
	}
	if uc["surfaceId"] != "existing-surface" {
		t.Errorf("surfaceId = %v, want existing-surface", uc["surfaceId"])
	}
}

// The "must contain root" rule is a full-render protocol rule, not a catalog
// check. An incremental patch of an explicit existing surface may omit root,
// even under strict validation.
func TestParserRootlessIncrementalUpdateAllowed(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateStrict,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "```a2ui\n" +
		`[{"updateComponents":{"surfaceId":"existing-surface","components":[{"id":"subtitle","component":"Text","text":"patched"}]}}]` +
		"\n```"
	segs, err := p.push(input)
	if err != nil {
		t.Fatalf("rootless incremental update should not error, got %v", err)
	}
	flushed, _ := p.flush()
	_, batches := collect(append(segs, flushed...))
	if len(batches) != 1 || len(batches[0]) != 1 {
		t.Fatalf("got batches %v, want a single 1-envelope batch", batches)
	}
	if _, ok := batches[0][0]["updateComponents"].(map[string]any); !ok {
		t.Errorf("expected the update to pass through, got %v", batches[0][0])
	}
}

// The parser preserves forward-compatible top-level envelope keys it does not
// inspect, rather than rebuilding a fresh {version, <kind>} map.
func TestParserPreservesUnknownEnvelopeKeys(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateWarn,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "```a2ui\n" +
		`[{"createSurface":{"surfaceId":"SURFACE_ID","catalogId":"c"},"futureKey":"keepme"}]` +
		"\n```"
	segs, err := p.push(input)
	if err != nil {
		t.Fatal(err)
	}
	flushed, _ := p.flush()
	_, batches := collect(append(segs, flushed...))
	if len(batches) != 1 || len(batches[0]) != 1 {
		t.Fatalf("got batches %v, want a single 1-envelope batch", batches)
	}
	if batches[0][0]["futureKey"] != "keepme" {
		t.Errorf("futureKey = %v, want keepme (unknown keys must be preserved)", batches[0][0]["futureKey"])
	}
}

// Streaming non-ASCII prose one byte at a time must never emit a mid-rune split
// (the holdback used to slice on a byte boundary, corrupting "18°C" into U+FFFD).
func TestParserNoMidRuneSplitInStreamedProse(t *testing.T) {
	p := newStreamParser(parserOptions{surfaceID: fixedSurfaceID("s1")})
	full := "The temperature is 18°C and rising — 20°C soon. 日本語のテキストもね。"
	var got strings.Builder
	for i := 0; i < len(full); i++ {
		segs, err := p.push(full[i : i+1])
		if err != nil {
			t.Fatal(err)
		}
		prose, batches := collect(segs)
		if len(batches) != 0 {
			t.Fatalf("unexpected envelope batch in pure prose")
		}
		got.WriteString(prose)
	}
	flushed, _ := p.flush()
	prose, _ := collect(flushed)
	got.WriteString(prose)
	if got.String() != full {
		t.Errorf("reassembled prose = %q, want %q", got.String(), full)
	}
	if strings.ContainsRune(got.String(), '\uFFFD') {
		t.Errorf("prose contains U+FFFD (mid-rune split): %q", got.String())
	}
}

// A padded opening fence ("```a2ui   \n") split across chunk boundaries must not
// leak backticks as prose: the holdback covers the whole incomplete fence
// prefix, not a fixed 8 bytes.
func TestParserPaddedOpenFenceSplit(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateWarn,
		surfaceID: fixedSurfaceID("s1"),
	})
	full := "hi ```a2ui   \n" +
		`[{"createSurface":{"surfaceId":"SURFACE_ID","catalogId":"c"}},` +
		`{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"root","component":"Text","text":"x"}]}}]` +
		"\n```"
	var allSegs []segment
	for i := 0; i < len(full); i++ {
		segs, err := p.push(full[i : i+1])
		if err != nil {
			t.Fatal(err)
		}
		allSegs = append(allSegs, segs...)
	}
	flushed, _ := p.flush()
	allSegs = append(allSegs, flushed...)
	prose, batches := collect(allSegs)
	if strings.Contains(prose, "`") {
		t.Errorf("prose leaked backticks from a padded fence: %q", prose)
	}
	if len(batches) != 1 {
		t.Fatalf("got %d batches, want 1 (padded fence must still open a block)", len(batches))
	}
}

// A ``` inside a JSON string value (A2UI Text may contain inline Markdown) must
// not close the block: the close fence is anchored to line start.
func TestParserBacktickInsideJSONDoesNotCloseBlock(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateStrict,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "```a2ui\n" +
		`[{"createSurface":{"surfaceId":"SURFACE_ID","catalogId":"c"}},` +
		`{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"root","component":"Text","text":"use ` + "```" + ` to fence code"}]}}]` +
		"\n```\nDone."
	segs, err := p.push(input)
	if err != nil {
		t.Fatalf("inline ``` in JSON must not break parsing, got %v", err)
	}
	flushed, _ := p.flush()
	prose, batches := collect(append(segs, flushed...))
	if len(batches) != 1 {
		t.Fatalf("got %d batches, want 1", len(batches))
	}
	if !strings.Contains(prose, "Done.") {
		t.Errorf("prose after the block missing: %q", prose)
	}
	uc, _ := batches[0][1]["updateComponents"].(map[string]any)
	comps, _ := uc["components"].([]any)
	c0, _ := comps[0].(map[string]any)
	if !strings.Contains(c0["text"].(string), "```") {
		t.Errorf("component text lost its inline backticks: %v", c0["text"])
	}
}

// A full render may legally split its components across several updateComponents
// (root in one, leaves in another). Root is a batch-level requirement.
func TestParserSplitRenderRootInLaterList(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateStrict,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "```a2ui\n" +
		`[{"createSurface":{"surfaceId":"SURFACE_ID","catalogId":"c"}},` +
		`{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"body","component":"Text","text":"leaf"}]}},` +
		`{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"root","component":"Column","children":["body"]}]}}]` +
		"\n```"
	segs, err := p.push(input)
	if err != nil {
		t.Fatalf("split render with root in a later list should be accepted, got %v", err)
	}
	flushed, _ := p.flush()
	_, batches := collect(append(segs, flushed...))
	if len(batches) != 1 {
		t.Fatalf("got %d batches, want 1", len(batches))
	}
}

// Prose that merely mentions the fence mid-sentence must not open a block. A2UI
// Text "may use inline Markdown", so a model can write "I emit an ```a2ui block
// like this"; there the a2ui tag is followed by more text on the same line, not
// a newline, so the required trailing newline keeps it prose. This also guards
// against a fence with stray spaces between the backticks and the tag
// ("``` a2ui\n") being mistaken for an opening fence.
func TestParserInlineFenceMentionStaysProse(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateWarn,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "To render UI I emit an ```a2ui fenced block like this."
	segs, err := p.push(input)
	if err != nil {
		t.Fatal(err)
	}
	flushed, _ := p.flush()
	prose, batches := collect(append(segs, flushed...))
	if len(batches) != 0 {
		t.Fatalf("got %d batches, want 0 (inline mention must stay prose)", len(batches))
	}
	if !strings.Contains(prose, "```a2ui fenced block") {
		t.Errorf("prose = %q, want the inline fence mention preserved", prose)
	}
}

// A fence with spaces between the backticks and the tag ("``` a2ui\n") is NOT a
// valid opening fence: the tag must immediately follow the backticks (mirroring
// the JS OPEN_FENCE_RE). Such text stays prose.
func TestParserSpacedFenceTagNotOpened(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateWarn,
		surfaceID: fixedSurfaceID("s1"),
	})
	input := "``` a2ui\n" +
		`[{"createSurface":{"surfaceId":"SURFACE_ID","catalogId":"c"}}]` +
		"\n```"
	segs, err := p.push(input)
	if err != nil {
		t.Fatal(err)
	}
	flushed, _ := p.flush()
	_, batches := collect(append(segs, flushed...))
	if len(batches) != 0 {
		t.Errorf("got %d batches, want 0 (spaced fence tag must not open a block)", len(batches))
	}
}

// A large block delivered in many small deltas parses correctly (also exercises
// the incremental in-block scan cursor).
func TestParserLargeBlockManyDeltas(t *testing.T) {
	p := newStreamParser(parserOptions{
		catalog:   BasicCatalog(),
		validate:  ValidateWarn,
		surfaceID: fixedSurfaceID("s1"),
	})
	var sb strings.Builder
	sb.WriteString("```a2ui\n")
	sb.WriteString(`[{"createSurface":{"surfaceId":"SURFACE_ID","catalogId":"c"}},`)
	sb.WriteString(`{"updateComponents":{"surfaceId":"SURFACE_ID","components":[{"id":"root","component":"Column","children":["p"]},`)
	sb.WriteString(`{"id":"p","component":"Text","text":"`)
	sb.WriteString(strings.Repeat("lorem ipsum dolor sit amet ", 400)) // ~10KB
	sb.WriteString(`"}]}}]`)
	sb.WriteString("\n```")
	full := sb.String()

	var allSegs []segment
	for i := 0; i < len(full); i += 7 {
		end := i + 7
		if end > len(full) {
			end = len(full)
		}
		segs, err := p.push(full[i:end])
		if err != nil {
			t.Fatal(err)
		}
		allSegs = append(allSegs, segs...)
	}
	flushed, _ := p.flush()
	allSegs = append(allSegs, flushed...)
	_, batches := collect(allSegs)
	if len(batches) != 1 {
		t.Fatalf("got %d batches, want 1", len(batches))
	}
	if len(batches[0]) != 2 {
		t.Errorf("got %d envelopes, want 2", len(batches[0]))
	}
}
