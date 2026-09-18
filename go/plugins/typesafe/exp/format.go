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
	"maps"
	"math"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// OutputFormatDecision is the name of the [DecisionFormat], for
// [ai.WithOutputFormat] and the output block of a prompt file.
const OutputFormatDecision = "decision"

// calibratedKey is the part metadata a System One model's answer carries,
// under the "typesafe" key, which is how the format tells a calibrated
// distribution from numbers another model made up.
const calibratedKey = "calibrated"

// DecisionFormat is an output format for decision types: output types whose
// fields are [Choice], [Score], and [Noul] values. It keeps the contract of
// those fields honest on every model.
//
// The format checks the output schema before any model is called, so a
// field that is not a question fails fast. On a System One model the
// questions ride on the schema and the answers come back calibrated. On any
// other model the questions are rendered as instructions instead, and the
// answer is hardened before it is parsed: probabilities and confidence are
// removed, a score is rounded to a whole level, and a noul becomes 0 or 1.
// Code that thresholds on those numbers then sees a zero confidence for
// what it is, unknown, rather than a number the model invented.
//
// The plugin registers the format during [genkit.Init]; on its own, define
// it with [genkit.DefineFormats].
type DecisionFormat struct{}

// Name implements [ai.Formatter].
func (DecisionFormat) Name() string { return OutputFormatDecision }

// Handler implements [ai.Formatter], compiling the schema's questions.
func (DecisionFormat) Handler(schema map[string]any) (ai.FormatHandler, error) {
	if schema == nil {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: the decision format needs an output type; call with ai.WithOutputType")
	}
	// A chat model receives the system message as itself, so the questions
	// are rendered without it.
	questions, err := compileQuestions(schema, "")
	if err != nil {
		return nil, err
	}
	return &decisionHandler{schema: schema, questions: questions}, nil
}

// decisionHandler is the per-call handler. It parses like the JSON format,
// with the hardening step in front of validation.
type decisionHandler struct {
	schema    map[string]any
	questions map[string]question
	// text is the text accumulated for the current streamed turn, index
	// that turn.
	text  string
	index int
}

// Config implements [ai.FormatHandler]. Constrained is requested; the loop
// grants it only on a model that declares native support.
func (h *decisionHandler) Config() ai.ModelOutputConfig {
	return ai.ModelOutputConfig{
		Format:      OutputFormatDecision,
		Schema:      h.schema,
		Constrained: true,
		ContentType: "application/json",
	}
}

// Instructions implements [ai.FormatHandler]. They reach a model only when
// the loop did not grant constrained output, so they are rendered on
// request.
func (h *decisionHandler) Instructions() string { return renderInstructions(h.questions) }

// ParseMessage implements [ai.FormatHandler]: the message's text becomes a
// single JSON part, hardened when uncalibrated and validated against the
// schema. Part metadata is carried over so a later [ParseOutput] still sees
// the calibration mark.
func (h *decisionHandler) ParseMessage(m *ai.Message) (*ai.Message, error) {
	if m == nil || len(m.Content) == 0 {
		return nil, status.Errorf(status.ErrInvalidOutput, "message has no content")
	}
	answers, others, metadata, err := h.parse(m)
	if err != nil {
		return nil, err
	}
	text, err := json.Marshal(answers)
	if err != nil {
		return nil, status.Errorf(status.ErrInvalidOutput, "typesafe: answers are not JSON: %w", err)
	}
	part := ai.NewJSONPart(string(text))
	part.Metadata = metadata
	return &ai.Message{Role: m.Role, Content: append([]*ai.Part{part}, others...), Metadata: m.Metadata}, nil
}

// ParseOutput implements [ai.StreamingFormatHandler].
func (h *decisionHandler) ParseOutput(m *ai.Message) (any, error) {
	if m == nil || len(m.Content) == 0 {
		return nil, status.Errorf(status.ErrInvalidOutput, "message has no content")
	}
	answers, _, _, err := h.parse(m)
	return answers, err
}

// ParseChunk implements [ai.StreamingFormatHandler]. A System One model
// does not stream, so chunks come from other models and are hardened.
// Nothing is returned until the accumulated text parses.
func (h *decisionHandler) ParseChunk(chunk *ai.ModelResponseChunk) (any, error) {
	if chunk.Index != h.index {
		h.index, h.text = chunk.Index, ""
	}
	h.text += chunk.Text()
	answers, err := decode(h.text)
	if err != nil {
		return nil, nil
	}
	h.harden(answers)
	return answers, nil
}

// decode reads the answer object out of text that may wrap it in markdown.
func decode(text string) (map[string]any, error) {
	extracted := base.ExtractJSONFromMarkdown(text)
	if extracted == "" {
		return nil, status.Errorf(status.ErrInvalidOutput, "typesafe: the model answered with no JSON")
	}
	var answers map[string]any
	if err := json.Unmarshal([]byte(extracted), &answers); err != nil {
		return nil, status.Errorf(status.ErrInvalidOutput, "typesafe: the answer is not a JSON object: %w", err)
	}
	return answers, nil
}

// parse reads the answers out of a message. The non-text parts and the
// text parts' metadata come back for [ParseMessage] to rebuild the message.
func (h *decisionHandler) parse(m *ai.Message) (answers map[string]any, others []*ai.Part, metadata map[string]any, err error) {
	var sb strings.Builder
	for _, part := range m.Content {
		if !part.IsText() {
			others = append(others, part)
			continue
		}
		sb.WriteString(part.Text)
		if metadata == nil && part.Metadata != nil {
			metadata = part.Metadata
		}
	}
	answers, err = decode(sb.String())
	if err != nil {
		return nil, nil, nil, err
	}
	if !calibrated(metadata) {
		h.harden(answers)
	}
	if err := base.ValidateValue(answers, h.schema); err != nil {
		return nil, nil, nil, err
	}
	return answers, others, metadata, nil
}

// calibrated reports whether a System One model produced the answer.
func calibrated(metadata map[string]any) bool {
	mark, _ := metadata["typesafe"].(map[string]any)
	ok, _ := mark[calibratedKey].(bool)
	return ok
}

// harden strips from an uncalibrated answer every number only a System One
// model can produce, and accepts the bare values a language model tends to
// answer with in place of the answer objects.
func (h *decisionHandler) harden(answers map[string]any) {
	for id, q := range h.questions {
		raw, ok := answers[id]
		if !ok {
			continue
		}
		answer, ok := raw.(map[string]any)
		if !ok {
			// An answer carries its value under its question type's name.
			answer = map[string]any{q.Type: raw}
		}
		delete(answer, "probabilities")
		delete(answer, "confidence")
		switch q.Type {
		case kindScore:
			delete(answer, "legend")
			if score, ok := answer["score"].(float64); ok {
				answer["score"] = math.Round(score)
			}
		case kindNoul:
			switch v := answer["noul"].(type) {
			case float64:
				answer["noul"] = math.Round(v)
			case bool:
				answer["noul"] = 0.0
				if v {
					answer["noul"] = 1.0
				}
			}
		}
		answers[id] = answer
	}
}

// renderInstructions writes the questions out for a model that has to be
// told, in the order of their IDs so the text is stable.
func renderInstructions(questions map[string]question) string {
	var sb strings.Builder
	sb.WriteString("Answer every question below with a single JSON object keyed by question id. Output only the JSON.\n")
	for _, id := range questionIDs(questions) {
		q := questions[id]
		fmt.Fprintf(&sb, "\n%s: %s\n", id, q.Instructions)
		switch q.Type {
		case kindChoice:
			criteria, _ := q.Criteria.(map[string]string)
			options := make([]string, 0, len(criteria))
			for _, key := range slices.Sorted(maps.Keys(criteria)) {
				options = append(options, fmt.Sprintf("%q (%s)", key, criteria[key]))
			}
			fmt.Fprintf(&sb, "Answer {\"choice\": <option>} with one of: %s.\n", strings.Join(options, ", "))
		case kindScore:
			levels, _ := q.Criteria.([]string)
			labels := make([]string, 0, len(levels))
			for i, level := range levels {
				labels = append(labels, fmt.Sprintf("%d = %s", i, level))
			}
			fmt.Fprintf(&sb, "Answer {\"score\": <level>} with the level number: %s.\n", strings.Join(labels, ", "))
		case kindNoul:
			sb.WriteString("Answer {\"noul\": 1} for yes or {\"noul\": 0} for no.")
			if criteria, ok := q.Criteria.(map[string]string); ok {
				fmt.Fprintf(&sb, " Yes means: %s. No means: %s.", criteria["true"], criteria["false"])
			}
			sb.WriteString("\n")
		}
	}
	return sb.String()
}
