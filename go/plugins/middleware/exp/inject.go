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

import "github.com/firebase/genkit/go/ai"

// injectSystemText returns a copy of req with text placed in a system-message
// part tagged by marker. The marker lets a middleware find its own injected
// text on later tool-loop iterations:
//
//   - If a tagged part already exists, it is refreshed in place when text
//     changed, or left untouched when identical. Middleware whose text is
//     constant (e.g. a fixed tool listing) is injected once; middleware whose
//     text varies per turn (e.g. a live artifact listing) is refreshed.
//   - Otherwise the text is appended to the first system message.
//   - Otherwise a new system message carrying the text is prepended.
//
// The request and its messages are copied before mutation, so req is unchanged.
func injectSystemText(req *ai.ModelRequest, marker, text string) *ai.ModelRequest {
	newReq := *req
	newReq.Messages = append([]*ai.Message(nil), req.Messages...)

	// Refresh an existing tagged part in place.
	for i, msg := range newReq.Messages {
		if msg == nil {
			continue
		}
		for j, part := range msg.Content {
			if !hasMarker(part, marker) {
				continue
			}
			if part.Text == text {
				return &newReq
			}
			msgCopy := msg.Clone()
			msgCopy.Content[j] = systemTextPart(marker, text)
			newReq.Messages[i] = msgCopy
			return &newReq
		}
	}

	// Append to an existing system message.
	for i, msg := range newReq.Messages {
		if msg == nil || msg.Role != ai.RoleSystem {
			continue
		}
		msgCopy := msg.Clone()
		msgCopy.Content = append(msgCopy.Content, systemTextPart(marker, text))
		newReq.Messages[i] = msgCopy
		return &newReq
	}

	// Otherwise prepend a fresh system message.
	newReq.Messages = append(
		[]*ai.Message{ai.NewSystemMessage(systemTextPart(marker, text))},
		newReq.Messages...,
	)
	return &newReq
}

// systemTextPart builds the text part that carries middleware-injected system
// text, tagged with marker so later iterations can find and refresh it.
func systemTextPart(marker, text string) *ai.Part {
	p := ai.NewTextPart(text)
	p.Metadata = map[string]any{marker: true}
	return p
}

// hasMarker reports whether p is a text part tagged with marker.
func hasMarker(p *ai.Part, marker string) bool {
	if p == nil || !p.IsText() || p.Metadata == nil {
		return false
	}
	v, ok := p.Metadata[marker].(bool)
	return ok && v
}
