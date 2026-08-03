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
	"github.com/firebase/genkit/go/ai"
	ant "github.com/firebase/genkit/go/plugins/internal/anthropic"
)

// DocumentOptions configures an Anthropic document content block with optional
// citations. Mirrors JS AnthropicDocumentOptions.
type DocumentOptions = ant.DocumentOptions

// DocumentCitations enables Anthropic citations for a document.
type DocumentCitations = ant.DocumentCitations

// DocumentSource is the document payload Anthropic should cite from.
type DocumentSource = ant.DocumentSource

// DocumentContentItem is a text or image item inside a content document source.
type DocumentContentItem = ant.DocumentContentItem

// DocumentImageSource is a base64 image embedded in a content document.
type DocumentImageSource = ant.DocumentImageSource

// Document creates a custom Genkit part representing an Anthropic document with
// optional citations support (JS anthropicDocument parity).
//
// Example:
//
//	ai.NewUserMessage(
//	  anthropic.Document(anthropic.DocumentOptions{
//	    Source: anthropic.DocumentSource{Type: "text", Data: "The grass is green."},
//	    Title: "Nature Facts",
//	    Citations: &anthropic.DocumentCitations{Enabled: true},
//	  }),
//	  ai.NewTextPart("What color is the grass?"),
//	)
func Document(opts DocumentOptions) *ai.Part {
	return ant.Document(opts)
}
