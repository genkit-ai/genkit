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

package systemone

import (
	"encoding/json"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
)

// SystemPreamble gathers the system messages into the text that goes in
// front of every question's instructions. The model has no system role: the
// state is what it evaluates and the questions are what it is told, so a
// system message belongs with the questions, never in the state. Only text
// can be instructions; loop plumbing left in a history is skipped.
func SystemPreamble(messages []*ai.Message) (string, error) {
	var texts []string
	for _, msg := range messages {
		if msg.Role != ai.RoleSystem {
			continue
		}
		var sb strings.Builder
		for _, part := range msg.Content {
			if isFormatInstructions(part) {
				continue
			}
			if !part.IsText() {
				return "", status.Errorf(status.ErrInvalidArgument, "systemone: a %s part cannot be instructions; a system message is text only", partKindName(part))
			}
			sb.WriteString(part.Text)
		}
		if text := strings.TrimSpace(sb.String()); text != "" {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n\n"), nil
}

// BuildState turns the request into the state: one message as its value,
// several as {role, content} records, and with documents attached an
// object of both, so a question can name the context it is about. System
// messages are instructions, not state, and are left out.
func BuildState(req *ai.ModelRequest, stateJSON bool) (any, error) {
	records := make([]map[string]any, 0, len(req.Messages))
	for _, msg := range req.Messages {
		if msg.Role == ai.RoleSystem {
			continue
		}
		value, err := messageValue(msg, stateJSON)
		if err != nil {
			return nil, err
		}
		if value == nil {
			continue
		}
		records = append(records, map[string]any{"role": string(msg.Role), "content": value})
	}
	if len(records) == 0 {
		return nil, status.Errorf(status.ErrInvalidArgument, "systemone: the request carries no state; a system message is instructions, so pass a prompt or messages too")
	}
	if len(req.Docs) > 0 {
		docs := make([]any, 0, len(req.Docs))
		for _, doc := range req.Docs {
			value, err := documentValue(doc)
			if err != nil {
				return nil, err
			}
			docs = append(docs, value)
		}
		return map[string]any{"messages": records, "context": docs}, nil
	}
	if len(records) == 1 {
		return records[0]["content"], nil
	}
	return records, nil
}

// messageValue is a message's contribution to the state: the value of its
// one part, or the values of all of them. A message with nothing to
// contribute yields nil.
func messageValue(msg *ai.Message, stateJSON bool) (any, error) {
	values := make([]any, 0, len(msg.Content))
	for _, part := range msg.Content {
		value, err := partValue(part, stateJSON)
		if err != nil {
			return nil, err
		}
		if value != nil {
			values = append(values, value)
		}
	}
	switch len(values) {
	case 0:
		return nil, nil
	case 1:
		return values[0], nil
	default:
		return values, nil
	}
}

// partValue is a part's contribution to the state. Text is sent as is, or
// as the JSON it holds when the config asks, kept verbatim so a number too
// large for a float64 reaches the model unchanged; a data part is sent as
// its value. Loop plumbing left in a history is skipped rather than sent as
// state, and any other kind of part is refused rather than stringified.
func partValue(part *ai.Part, stateJSON bool) (any, error) {
	if isFormatInstructions(part) {
		return nil, nil
	}
	switch {
	case part.IsText():
		if stateJSON {
			var value json.RawMessage
			if err := json.Unmarshal([]byte(part.Text), &value); err != nil {
				return nil, status.Errorf(status.ErrInvalidArgument, "systemone: stateJSON is set but the text is not JSON: %w", err)
			}
			return value, nil
		}
		return part.Text, nil
	case part.IsData():
		return part.Data, nil
	default:
		return nil, status.Errorf(status.ErrInvalidArgument, "systemone: a %s part cannot be state; the model takes text and JSON only", partKindName(part))
	}
}

// isFormatInstructions reports whether the part is the output instructions
// the generate loop injects for a model without constrained output. The
// loop never injects one for this model, but a history recorded from
// another model's turn keeps it in that turn's messages, and when such a
// history is the state the part is plumbing, not something anyone said.
func isFormatInstructions(part *ai.Part) bool {
	purpose, _ := part.Metadata["purpose"].(string)
	return purpose == "output"
}

// documentValue is a document's contribution to the context: its content,
// read as a message's is, or its content with its metadata when it has
// any, so a passage keeps its title and source for a question to refer
// to. Text stays text whatever the config says, since a retrieved passage
// is prose, not a rendered template.
func documentValue(doc *ai.Document) (any, error) {
	content, err := messageValue(&ai.Message{Content: doc.Content}, false)
	if err != nil {
		return nil, err
	}
	if len(doc.Metadata) == 0 {
		return content, nil
	}
	return map[string]any{"content": content, "metadata": doc.Metadata}, nil
}

// partKindName names a part's kind for an error message.
func partKindName(part *ai.Part) string {
	switch {
	case part.IsMedia():
		return "media"
	case part.IsToolRequest():
		return "tool request"
	case part.IsToolResponse():
		return "tool response"
	case part.IsReasoning():
		return "reasoning"
	case part.IsResource():
		return "resource"
	default:
		return "custom"
	}
}
