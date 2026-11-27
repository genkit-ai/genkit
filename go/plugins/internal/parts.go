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

package internal

import (
	"strings"

	"github.com/firebase/genkit/go/ai"
)

// MergeAdjacentText merges each run of adjacent text parts, and each run of
// adjacent reasoning parts, into one part. Plugins call it when they fold a
// stream back into the final message: a stream arrives one delta at a time,
// and a message with one part per delta splits the answer mid-sentence for
// any reader of a single part.
//
// Only bare parts merge. A part that carries metadata or custom data is a
// boundary and stays as it is, so a per-part provider payload never joins or
// moves to another part. Gemini, for one, ends a streamed text reply with an
// empty text part that carries the thought signature; it stays a separate
// part, which a middleware that rewrites text passes through untouched. Part
// order is kept. The input parts are not modified, because the stream
// callback has already handed them out.
func MergeAdjacentText(parts []*ai.Part) []*ai.Part {
	out := make([]*ai.Part, 0, len(parts))
	var b strings.Builder
	for i := 0; i < len(parts); {
		p := parts[i]
		j := i + 1
		if mergeable(p) {
			for j < len(parts) && mergeable(parts[j]) && parts[j].Kind == p.Kind {
				j++
			}
		}
		if j == i+1 {
			out = append(out, p)
			i = j
			continue
		}
		b.Reset()
		for _, q := range parts[i:j] {
			b.WriteString(q.Text)
		}
		out = append(out, &ai.Part{Kind: p.Kind, ContentType: p.ContentType, Text: b.String()})
		i = j
	}
	return out
}

// mergeable reports whether p is a text or reasoning part with nothing on it
// but its text.
func mergeable(p *ai.Part) bool {
	return (p.IsText() || p.IsReasoning()) && len(p.Metadata) == 0 && p.Custom == nil
}
