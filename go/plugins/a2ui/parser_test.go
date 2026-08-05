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
