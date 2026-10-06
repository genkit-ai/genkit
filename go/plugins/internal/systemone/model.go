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
	"context"
	"encoding/json"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
)

// reservedFields are the top-level fields a request builds itself, which
// an extra field cannot replace: a different model or questions would
// detach the request from the model that was called and the questions that
// were compiled from the output type.
var reservedFields = []string{"model", "state", "questions", "images", "audio", "videos"}

// Config is the per-request configuration of a decision model.
// go/plugins/systemone/exp exports it as Config.
type Config struct {
	// StateJSON parses each text part of the state as JSON, so a prompt
	// template that renders a JSON document produces an object state
	// rather than a string one. A text that is not JSON is an error. A
	// fallback to another model takes that model's config, so set it there
	// too, or send the state as a data part, which needs no parsing.
	StateJSON bool `json:"stateJSON,omitzero" jsonschema_description:"Parse each text part of the state as JSON, so a template that renders JSON produces an object state."`

	// Extra is merged into the top-level fields of the request body. It
	// reaches fields this package does not model, such as a gateway's
	// session_id or trace. It cannot name a field the request builds
	// itself: model, state, questions, images, audio, or videos.
	Extra map[string]any `json:"extra,omitempty" jsonschema_description:"Extra top-level request fields, such as a gateway's session_id. Cannot replace model, state, questions, images, audio, or videos."`
}

// NewModel builds the model action for one model ID under its registered
// name.
//
// The model claims constrained output, so the loop hands it the output
// schema untouched, and claims the system role and context so the loop
// does not rewrite either into instruction text that would land in the
// state.
func NewModel(c *Client, name, id string, spec Spec) *ai.ModelAction {
	m := &model{client: c, id: id, reads: spec.Reads}
	return ai.NewModelAction(name, &ai.ModelOptions{
		Label: spec.Label,
		Supports: &ai.ModelSupports{
			Media:       spec.Reads.Images || spec.Reads.Audio || spec.Reads.Video,
			Multiturn:   true,
			SystemRole:  true,
			Context:     true,
			Constrained: ai.ConstrainedSupportAll,
			Output:      []string{ai.OutputFormatJSON, ai.OutputFormatEnum},
			ContentType: []string{"application/json", "text/enum"},
		},
	}, m.generate)
}

// Spec describes one model to [NewModel].
type Spec struct {
	// Label names the model in the Dev UI.
	Label string
	// Reads is the media requests may carry. A kind the model does not
	// read is refused before a request is sent.
	Reads Reads
}

// model is one resolved model.
type model struct {
	client *Client
	id     string
	reads  Reads
}

// generate answers the questions the request's output schema encodes.
func (m *model) generate(ctx context.Context, req *ai.ModelRequest, cfg *Config, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	if req.Output == nil || (req.Output.Schema == nil && req.Output.Format != ai.OutputFormatEnum) {
		return nil, status.Errorf(status.ErrInvalidArgument, "systemone: the model answers questions encoded in an output type; call GenerateData with a decision type, or pass ai.WithOutputSchema(systemonex.Schema(questions)) for questions built at run time")
	}
	var opts Config
	if cfg != nil {
		opts = *cfg
	}
	for _, field := range reservedFields {
		if _, ok := opts.Extra[field]; ok {
			return nil, status.Errorf(status.ErrInvalidArgument, "systemone: extra field %q is one the request builds itself", field)
		}
	}
	enum := req.Output.Format == ai.OutputFormatEnum
	preamble, err := SystemPreamble(req.Messages)
	if err != nil {
		return nil, err
	}
	var questions map[string]Question
	if enum {
		questions, err = EnumQuestion(req.Output.Schema, preamble)
	} else {
		questions, err = CompileQuestions(req.Output.Schema, preamble)
	}
	if err != nil {
		return nil, err
	}
	state, media, err := BuildState(req, opts.StateJSON, m.reads)
	if err != nil {
		return nil, err
	}

	resp, err := m.client.Decide(ctx, m.id, &Request{State: state, Media: media, Questions: questions, Extra: opts.Extra})
	if err != nil {
		return nil, err
	}
	logger.Debug(ctx, "systemone: answered", "endpoint", m.client.Endpoint.Name, "model", resp.Model, "questions", len(questions), "inputTokens", resp.Usage.InputTokens, "outputTokens", resp.Usage.OutputTokens)

	text, err := AnswersText(resp, questions, enum)
	if err != nil {
		return nil, err
	}
	part := ai.NewJSONPart(text)
	if enum {
		part = ai.NewTextPart(text)
	}

	usage := &ai.GenerationUsage{
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		TotalTokens:  resp.Usage.InputTokens + resp.Usage.OutputTokens,
	}
	if resp.Usage.Cost != nil {
		usage.Custom = map[string]float64{"cost": *resp.Usage.Cost}
	}
	return &ai.ModelResponse{
		Message:      ai.NewModelMessage(part),
		FinishReason: ai.FinishReasonStop,
		Usage:        usage,
		Raw:          &resp.Info,
	}, nil
}

// Info is the raw response of a model built by [NewModel]: the model
// version that answered, a gateway's provider and generation ID, and the
// answers as the API sent them. go/plugins/systemone/exp exports it as
// Info.
type Info struct {
	Model    string                    `json:"model"`
	Provider string                    `json:"provider,omitempty"`
	ID       string                    `json:"id,omitempty"`
	Answers  map[string]map[string]any `json:"answers,omitempty"`
}

// answerFields lists the fields of a wire answer per question type, which
// are the fields the answer types declare. The answer schemas are closed,
// so a field the API or a gateway adds later is dropped here rather than
// failing every call at validation; the untouched answer is still on
// [Info.Answers].
var answerFields = map[string][]string{
	KindChoice: {"choice", "probabilities", "confidence"},
	KindScore:  {"score", "probabilities", "confidence", "legend"},
	KindNoul:   {"noul"},
}

// AnswersText renders the answers as the message text: for the enum
// format the chosen option itself, otherwise a JSON object keyed by
// question ID whose values are the answers projected onto the fields the
// output type declares. A score's legend is rebuilt from the rubric's
// strings: for a level sent with guidance the API echoes the guidance,
// which the legend's strings cannot hold. A score is clamped to the rubric,
// since float rounding in the expected level can land a hair past the top
// level, which the schema's bounds would reject along with every other
// answer in the call.
func AnswersText(resp *Response, questions map[string]Question, enum bool) (string, error) {
	answers := make(map[string]map[string]any, len(questions))
	for _, id := range questionIDs(questions) {
		answer, ok := resp.Answers[id]
		if !ok {
			return "", status.Errorf(status.ErrInternal, "systemone: no answer for question %q", id)
		}
		fields := answerFields[questions[id].Type]
		projected := make(map[string]any, len(fields))
		for _, field := range fields {
			if v, ok := answer[field]; ok {
				projected[field] = v
			}
		}
		if q := questions[id]; q.Labels != nil {
			projected["legend"] = q.legend()
			if score, ok := projected["score"].(float64); ok {
				projected["score"] = min(max(score, 0), float64(len(q.Labels)-1))
			}
		}
		answers[id] = projected
	}
	if enum {
		choice, _ := answers[EnumQuestionID]["choice"].(string)
		if choice == "" {
			return "", status.Errorf(status.ErrInternal, "systemone: the enum answer names no option")
		}
		return choice, nil
	}
	text, err := json.Marshal(answers)
	if err != nil {
		return "", status.Errorf(status.ErrInternal, "systemone: answers are not JSON: %w", err)
	}
	return string(text), nil
}
