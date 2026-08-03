// Copyright 2025 Google LLC
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

package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
)

func TestDocumentPartToAnthropicRequest(t *testing.T) {
	part := Document(DocumentOptions{
		Source: DocumentSource{
			Type: "text",
			Data: "The grass is green. The sky is blue.",
		},
		Title:     "Nature Facts",
		Citations: &DocumentCitations{Enabled: true},
	})
	got, err := toAnthropicParts([]*ai.Part{part})
	if err != nil {
		t.Fatalf("toAnthropicParts: %v", err)
	}
	if len(got) != 1 || got[0].OfDocument == nil {
		t.Fatalf("expected document block, got %#v", got)
	}
	doc := got[0].OfDocument
	if doc.Source.OfText == nil || doc.Source.OfText.Data != "The grass is green. The sky is blue." {
		t.Errorf("text source = %#v", doc.Source.OfText)
	}
	wire := wireJSON(t, doc)
	if !strings.Contains(wire, `"title":"Nature Facts"`) {
		t.Errorf("title missing: %s", wire)
	}
	if !strings.Contains(wire, `"citations"`) || !strings.Contains(wire, `"enabled":true`) {
		t.Errorf("citations missing: %s", wire)
	}
}

func TestDocumentFileSourceRequiresBeta(t *testing.T) {
	_, err := toAnthropicParts([]*ai.Part{
		Document(DocumentOptions{
			Source: DocumentSource{Type: "file", FileID: "file_123"},
		}),
	})
	if err == nil || !strings.Contains(err.Error(), "beta") {
		t.Fatalf("expected beta error, got %v", err)
	}
}

func TestToGenkitResponseTextCitations(t *testing.T) {
	raw := `{
		"id": "msg_cite",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [{
			"type": "text",
			"text": "The grass is green.",
			"citations": [{
				"type": "char_location",
				"cited_text": "The grass is green.",
				"document_index": 0,
				"document_title": "Nature Facts",
				"start_char_index": 0,
				"end_char_index": 19,
				"file_id": null
			}]
		}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 1, "output_tokens": 2}
	}`
	var msg anthropic.Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	got, err := toGenkitResponse(&msg)
	if err != nil {
		t.Fatalf("toGenkitResponse: %v", err)
	}
	if len(got.Message.Content) != 1 {
		t.Fatalf("got %d parts", len(got.Message.Content))
	}
	part := got.Message.Content[0]
	if part.Text != "The grass is green." {
		t.Errorf("text = %q", part.Text)
	}
	citations, ok := part.Metadata["citations"].([]map[string]any)
	if !ok || len(citations) != 1 {
		t.Fatalf("citations metadata = %#v", part.Metadata)
	}
	c := citations[0]
	if c["type"] != "char_location" {
		t.Errorf("expected type char_location, got %#v", c["type"])
	}
	if c["documentIndex"] != int64(0) && c["documentIndex"] != float64(0) {
		t.Errorf("expected documentIndex 0, got %#v", c["documentIndex"])
	}
	if c["citedText"] != "The grass is green." || c["documentTitle"] != "Nature Facts" {
		t.Errorf("citation fields = %#v", c)
	}
	if c["startCharIndex"] != int64(0) && c["startCharIndex"] != float64(0) {
		t.Errorf("startCharIndex = %#v", c["startCharIndex"])
	}
}

func TestToGenkitResponseSkipsWebSearchCitations(t *testing.T) {
	raw := `{
		"id": "msg_web",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [{
			"type": "text",
			"text": "Found online.",
			"citations": [{
				"type": "web_search_result_location",
				"cited_text": "snippet",
				"url": "https://example.com",
				"title": "Example",
				"encrypted_index": "abc"
			}]
		}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 1, "output_tokens": 2}
	}`
	var msg anthropic.Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	got, err := toGenkitResponse(&msg)
	if err != nil {
		t.Fatalf("toGenkitResponse: %v", err)
	}
	part := got.Message.Content[0]
	if part.Metadata != nil {
		if _, ok := part.Metadata["citations"]; ok {
			t.Fatalf("web search citations should be skipped, got %#v", part.Metadata)
		}
	}
}

func TestCitationsDeltaToPart(t *testing.T) {
	delta := anthropic.RawContentBlockDeltaUnion{
		Type: "citations_delta",
		Citation: anthropic.CitationsDeltaCitationUnion{
			Type:            "page_location",
			CitedText:       "page text",
			DocumentIndex:   1,
			DocumentTitle:   "PDF Doc",
			StartPageNumber: 2,
			EndPageNumber:   3,
		},
	}
	p := citationsDeltaToPart(delta)
	if p == nil {
		t.Fatal("expected citation part")
	}
	if p.Text != "" {
		t.Errorf("text should be empty, got %q", p.Text)
	}
	citations := p.Metadata["citations"].([]map[string]any)
	if citations[0]["type"] != "page_location" || citations[0]["documentIndex"] != int64(1) {
		t.Errorf("citation = %#v", citations[0])
	}
}

func TestCitationsDeltaSkipsWebSearch(t *testing.T) {
	p := citationsDeltaToPart(anthropic.RawContentBlockDeltaUnion{
		Type: "citations_delta",
		Citation: anthropic.CitationsDeltaCitationUnion{
			Type:      "web_search_result_location",
			CitedText: "x",
			URL:       "https://example.com",
		},
	})
	if p != nil {
		t.Fatalf("expected nil for web search citation delta, got %#v", p)
	}
}
