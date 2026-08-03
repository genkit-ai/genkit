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
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
)

// DocumentOptions configures an Anthropic document content block with optional
// citations, matching the JS AnthropicDocumentOptions shape.
type DocumentOptions struct {
	Source    DocumentSource     `json:"source"`
	Title     string             `json:"title,omitempty"`
	Context   string             `json:"context,omitempty"`
	Citations *DocumentCitations `json:"citations,omitempty"`
}

// DocumentCitations enables Anthropic citations for a document.
type DocumentCitations struct {
	Enabled bool `json:"enabled"`
}

// DocumentSource is the document payload Anthropic should cite from.
// Type is one of: "text", "base64", "content", "url", "file".
type DocumentSource struct {
	Type      string                `json:"type"`
	Data      string                `json:"data,omitempty"`
	MediaType string                `json:"mediaType,omitempty"`
	FileID    string                `json:"fileId,omitempty"`
	URL       string                `json:"url,omitempty"`
	Content   []DocumentContentItem `json:"content,omitempty"`
}

// DocumentContentItem is a text or image item inside a content document source.
type DocumentContentItem struct {
	Type   string               `json:"type"`
	Text   string               `json:"text,omitempty"`
	Source *DocumentImageSource `json:"source,omitempty"`
}

// DocumentImageSource is a base64 image embedded in a content document.
type DocumentImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"mediaType"`
	Data      string `json:"data"`
}

// Document creates a custom Genkit part representing an Anthropic document
// with optional citations support (JS anthropicDocument parity).
func Document(opts DocumentOptions) *ai.Part {
	raw, err := json.Marshal(opts)
	if err != nil {
		// Marshal of plain structs should not fail; fall back to empty options.
		return ai.NewCustomPart(map[string]any{"anthropicDocument": map[string]any{}})
	}
	var asMap map[string]any
	_ = json.Unmarshal(raw, &asMap)
	return ai.NewCustomPart(map[string]any{"anthropicDocument": asMap})
}

func documentOptionsFromCustom(custom map[string]any) (DocumentOptions, bool, error) {
	raw, ok := custom["anthropicDocument"]
	if !ok || raw == nil {
		return DocumentOptions{}, false, nil
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return DocumentOptions{}, true, fmt.Errorf("invalid anthropicDocument: %w", err)
	}
	var opts DocumentOptions
	if err := json.Unmarshal(b, &opts); err != nil {
		return DocumentOptions{}, true, fmt.Errorf("invalid anthropicDocument: %w", err)
	}
	return opts, true, nil
}

func toDocumentBlock(opts DocumentOptions) (anthropic.ContentBlockParamUnion, error) {
	doc := anthropic.DocumentBlockParam{}
	if opts.Title != "" {
		doc.Title = anthropic.String(opts.Title)
	}
	if opts.Context != "" {
		doc.Context = anthropic.String(opts.Context)
	}
	if opts.Citations != nil {
		doc.Citations = anthropic.CitationsConfigParam{
			Enabled: anthropic.Bool(opts.Citations.Enabled),
		}
	}

	switch opts.Source.Type {
	case "text":
		doc.Source.OfText = &anthropic.PlainTextSourceParam{
			Data: opts.Source.Data,
		}
	case "base64":
		if opts.Source.MediaType != "" && opts.Source.MediaType != "application/pdf" {
			return anthropic.ContentBlockParamUnion{}, status.Errorf(
				ai.ErrUnsupportedByModel,
				"unsupported base64 document media type %q; only application/pdf is supported",
				opts.Source.MediaType,
			)
		}
		doc.Source.OfBase64 = &anthropic.Base64PDFSourceParam{
			Data: opts.Source.Data,
		}
	case "url":
		doc.Source.OfURL = &anthropic.URLPDFSourceParam{
			URL: opts.Source.URL,
		}
	case "content":
		items := make([]anthropic.ContentBlockSourceContentItemUnionParam, 0, len(opts.Source.Content))
		for _, item := range opts.Source.Content {
			switch item.Type {
			case "text":
				items = append(items, anthropic.ContentBlockSourceContentItemUnionParam{
					OfText: &anthropic.TextBlockParam{Text: item.Text},
				})
			case "image":
				if item.Source == nil {
					return anthropic.ContentBlockParamUnion{}, status.Errorf(ai.ErrInvalidPart, "document content image item missing source")
				}
				mediaType, err := imageMediaType(item.Source.MediaType)
				if err != nil {
					return anthropic.ContentBlockParamUnion{}, err
				}
				items = append(items, anthropic.ContentBlockSourceContentItemUnionParam{
					OfImage: &anthropic.ImageBlockParam{
						Source: anthropic.ImageBlockParamSourceUnion{
							OfBase64: &anthropic.Base64ImageSourceParam{
								Data:      item.Source.Data,
								MediaType: mediaType,
							},
						},
					},
				})
			default:
				return anthropic.ContentBlockParamUnion{}, status.Errorf(
					ai.ErrInvalidPart,
					"unsupported document content item type %q",
					item.Type,
				)
			}
		}
		doc.Source.OfContent = &anthropic.ContentBlockSourceParam{
			Content: anthropic.ContentBlockSourceContentUnionParam{
				OfContentBlockSourceContent: items,
			},
		}
	case "file":
		return anthropic.ContentBlockParamUnion{}, status.Errorf(
			ai.ErrUnsupportedByModel,
			"file-based document sources require the beta API; set apiVersion: \"beta\" in your plugin or request config",
		)
	default:
		return anthropic.ContentBlockParamUnion{}, status.Errorf(
			ai.ErrInvalidPart,
			"unsupported document source type %q",
			opts.Source.Type,
		)
	}

	return anthropic.ContentBlockParamUnion{OfDocument: &doc}, nil
}

func imageMediaType(mediaType string) (anthropic.Base64ImageSourceMediaType, error) {
	switch mediaType {
	case "image/jpeg":
		return anthropic.Base64ImageSourceMediaTypeImageJPEG, nil
	case "image/png":
		return anthropic.Base64ImageSourceMediaTypeImagePNG, nil
	case "image/gif":
		return anthropic.Base64ImageSourceMediaTypeImageGIF, nil
	case "image/webp":
		return anthropic.Base64ImageSourceMediaTypeImageWebP, nil
	default:
		return "", status.Errorf(
			ai.ErrUnsupportedByModel,
			"unsupported image media type for Anthropic document content: %q",
			mediaType,
		)
	}
}

// textBlockToPart converts an Anthropic text block (with optional citations)
// into a Genkit text part, matching JS textBlockToPart.
func textBlockToPart(text string, citations []anthropic.TextCitationUnion) *ai.Part {
	p := ai.NewTextPart(text)
	if len(citations) == 0 {
		return p
	}
	converted := make([]map[string]any, 0, len(citations))
	for _, c := range citations {
		if cite := fromAnthropicCitation(c); cite != nil {
			converted = append(converted, cite)
		}
	}
	if len(converted) > 0 {
		p.Metadata = map[string]any{"citations": converted}
	}
	return p
}

// fromAnthropicCitation converts a document citation to camelCase Genkit
// metadata. Web-search / search-result citations (no document_index) are skipped.
func fromAnthropicCitation(c anthropic.TextCitationUnion) map[string]any {
	if !c.JSON.DocumentIndex.Valid() {
		return nil
	}
	switch c.Type {
	case "char_location":
		out := map[string]any{
			"type":           "char_location",
			"citedText":      c.CitedText,
			"documentIndex":  c.DocumentIndex,
			"startCharIndex": c.StartCharIndex,
			"endCharIndex":   c.EndCharIndex,
		}
		optionalCitationFields(out, c)
		return out
	case "page_location":
		out := map[string]any{
			"type":            "page_location",
			"citedText":       c.CitedText,
			"documentIndex":   c.DocumentIndex,
			"startPageNumber": c.StartPageNumber,
			"endPageNumber":   c.EndPageNumber,
		}
		optionalCitationFields(out, c)
		return out
	case "content_block_location":
		out := map[string]any{
			"type":            "content_block_location",
			"citedText":       c.CitedText,
			"documentIndex":   c.DocumentIndex,
			"startBlockIndex": c.StartBlockIndex,
			"endBlockIndex":   c.EndBlockIndex,
		}
		optionalCitationFields(out, c)
		return out
	default:
		return nil
	}
}

func optionalCitationFields(out map[string]any, c anthropic.TextCitationUnion) {
	if c.JSON.DocumentTitle.Valid() && c.DocumentTitle != "" {
		out["documentTitle"] = c.DocumentTitle
	}
	if c.JSON.FileID.Valid() && c.FileID != "" {
		out["fileId"] = c.FileID
	}
}

// citationsDeltaToPart converts a citations_delta stream event into a Genkit
// part with empty text and citation metadata (JS citationsDeltaToPart parity).
func citationsDeltaToPart(delta anthropic.RawContentBlockDeltaUnion) *ai.Part {
	if delta.Type != "citations_delta" {
		return nil
	}
	cite := fromCitationDelta(delta.Citation)
	if cite == nil {
		return nil
	}
	p := ai.NewTextPart("")
	p.Metadata = map[string]any{"citations": []map[string]any{cite}}
	return p
}

func fromCitationDelta(c anthropic.CitationsDeltaCitationUnion) map[string]any {
	switch c.Type {
	case "char_location":
		out := map[string]any{
			"type":           "char_location",
			"citedText":      c.CitedText,
			"documentIndex":  c.DocumentIndex,
			"startCharIndex": c.StartCharIndex,
			"endCharIndex":   c.EndCharIndex,
		}
		if c.DocumentTitle != "" {
			out["documentTitle"] = c.DocumentTitle
		}
		if c.FileID != "" {
			out["fileId"] = c.FileID
		}
		return out
	case "page_location":
		out := map[string]any{
			"type":            "page_location",
			"citedText":       c.CitedText,
			"documentIndex":   c.DocumentIndex,
			"startPageNumber": c.StartPageNumber,
			"endPageNumber":   c.EndPageNumber,
		}
		if c.DocumentTitle != "" {
			out["documentTitle"] = c.DocumentTitle
		}
		if c.FileID != "" {
			out["fileId"] = c.FileID
		}
		return out
	case "content_block_location":
		out := map[string]any{
			"type":            "content_block_location",
			"citedText":       c.CitedText,
			"documentIndex":   c.DocumentIndex,
			"startBlockIndex": c.StartBlockIndex,
			"endBlockIndex":   c.EndBlockIndex,
		}
		if c.DocumentTitle != "" {
			out["documentTitle"] = c.DocumentTitle
		}
		if c.FileID != "" {
			out["fileId"] = c.FileID
		}
		return out
	default:
		// web_search_result_location / search_result_location / unknown
		return nil
	}
}
