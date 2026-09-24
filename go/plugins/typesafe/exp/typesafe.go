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

// Package exp provides a Genkit plugin for TypeSafe AI's System One models,
// of which jev is the first. A System One model does not generate text. It
// evaluates a state against typed questions and returns one typed answer
// per question, with calibrated probabilities, in one round trip of a few
// hundred milliseconds.
//
// The plugin serves jev as a model that speaks only constrained JSON, which
// is the subset of the generate API the model fits exactly. The questions
// are the fields of the output type, declared with [Choice], [Score],
// [Noul], and [NoulOf], so a decision is one typed generate call with
// nothing more than the model and the state:
//
//	type Dept string
//
//	func (Dept) Criteria() map[Dept]string {
//		return map[Dept]string{
//			"billing":   "Payments, invoicing, refunds",
//			"technical": "Bugs, outages, integrations",
//			"other":     "None of the above",
//		}
//	}
//
//	type Triage struct {
//		Department typesafe.Choice[Dept] `json:"department" jsonschema_description:"Which team should handle this?"`
//		IsUrgent   typesafe.Noul         `json:"is_urgent"  jsonschema_description:"Does the ticket explicitly communicate time pressure?"`
//	}
//
//	out, resp, err := genkit.GenerateData[Triage](ctx, g,
//		ai.WithModelName("typesafe/jev-1.13.0"),
//		ai.WithPrompt(ticket))
//	if out.Department.Confidence < 0.6 {
//		// route to a human
//	}
//
// Criteria and levels are strings on their types. [GuidedOption],
// [GuidedRubric], and [GuidedYesNo] add the structured form the wire
// format also takes, such as examples per option.
//
// The state is built from the user and model messages: one message is sent
// as its value, a string for a text part or the JSON of a data part;
// several messages are sent as an array of {role, content} records; with
// documents attached the state is {messages, context}. A system message is
// never state: it is instructions, put in front of every question, and for
// the enum format it is the question. Answers come back as one JSON text
// part shaped like the output type, with the response's resolved model
// version and raw answers on [ai.ModelResponse.Custom].
//
// The same questions reach jev through TypeSafe's own API or through a
// gateway; see [Endpoint].
//
// This package is a preview: its API may change in any minor release.
package exp

import (
	"cmp"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/plugins/internal"
)

const provider = "typesafe"

// TypeSafe is the plugin. Models are resolved on demand by ID, under the
// typesafe prefix: typesafe/jev-latest, typesafe/jev-1.13.0. Any ID is
// forwarded, since the API accepts a versioned ID it does not list; pin a
// version in production, because confidence thresholds tuned against one
// release do not carry over to the next.
type TypeSafe struct {
	// APIKey authenticates requests. When empty, the environment variable
	// the endpoint names is read: TYPESAFE_API_KEY for TypeSafe's own API,
	// OPENROUTER_API_KEY for OpenRouter, CLOUDFLARE_API_TOKEN for Cloudflare.
	APIKey string

	// BaseURL overrides the endpoint's origin, for a proxy or a private
	// deployment. A proxy that forwards the native protocol under a prefix,
	// such as LiteLLM, is reached by including the prefix here. For the
	// direct endpoint, TYPESAFE_BASE_URL is read when this is empty.
	BaseURL string

	// Endpoint selects the server, [Direct] when nil. See [OpenRouter] and
	// [Cloudflare] for the gateways.
	Endpoint *Endpoint

	// HTTPClient sends the requests. It is the escape hatch to transport
	// settings: timeouts, proxies, and client middleware. When nil, a
	// client with a 30-second timeout is used.
	HTTPClient *http.Client

	// Headers are sent on every request, after the authorization header.
	Headers http.Header

	mu      sync.Mutex
	client  *client
	initted bool
}

// Config is the per-request model configuration.
type Config struct {
	// StateJSON parses each text part of the state as JSON, so a prompt
	// template that renders a JSON document produces an object state
	// rather than a string one. A text that is not JSON is an error.
	StateJSON bool `json:"stateJSON,omitzero" jsonschema_description:"Parse each text part of the state as JSON, so a template that renders JSON produces an object state."`

	// Extra is merged over the top-level fields of the request body, last
	// write wins. It reaches fields this package does not model, such as
	// OpenRouter's provider, session_id, and trace.
	Extra map[string]any `json:"extra,omitempty" jsonschema_description:"Extra top-level request fields, merged over the ones the plugin builds."`
}

// Name implements [api.Plugin].
func (t *TypeSafe) Name() string { return provider }

// Init implements [api.Plugin]. It builds the client and panics without a
// key, since every request would fail. No actions are registered up front:
// models resolve by name.
func (t *TypeSafe) Init(ctx context.Context) []api.Action {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.initted {
		panic("typesafe: plugin already initialized")
	}

	ep := cmp.Or(t.Endpoint, Direct())
	apiKey := cmp.Or(t.APIKey, os.Getenv(ep.apiKeyEnv))
	if apiKey == "" {
		panic("typesafe: set APIKey or the " + ep.apiKeyEnv + " environment variable")
	}
	t.client = &client{
		http:    cmp.Or(t.HTTPClient, &http.Client{Timeout: requestTimeout}),
		baseURL: strings.TrimSuffix(cmp.Or(t.BaseURL, os.Getenv(ep.baseURLEnv), ep.baseURL), "/"),
		apiKey:  apiKey,
		headers: t.Headers,
		ep:      ep,
	}
	t.initted = true
	return nil
}

// ListActions implements [api.DynamicPlugin]. TypeSafe's own API lists its
// models; a gateway has no such listing, so the known IDs are advertised.
// A listing failure falls back to the known IDs too, so the Dev UI still
// shows the models while offline.
func (t *TypeSafe) ListActions(ctx context.Context) []api.ActionDesc {
	t.mu.Lock()
	c := t.client
	t.mu.Unlock()
	if c == nil {
		return nil
	}
	ids := c.ep.models
	if c.ep.modelsPath != "" {
		if models, err := c.listModels(ctx); err != nil {
			logger.Debug(ctx, "typesafe: model listing failed, advertising the known models", "error", err)
		} else if len(models) > 0 {
			ids = make([]string, 0, len(models))
			for _, m := range models {
				ids = append(ids, m.Name)
			}
		}
	}
	actions := make([]api.ActionDesc, 0, len(ids))
	for _, id := range ids {
		actions = append(actions, newModel(c, id).Desc())
	}
	return actions
}

// ResolveAction implements [api.DynamicPlugin]. Models are the only action
// type served. The ID is forwarded as given; whether it exists is for the
// endpoint to say when the first request is made.
func (t *TypeSafe) ResolveAction(atype api.ActionType, id string) api.Action {
	if atype != api.ActionTypeModel {
		return nil
	}
	t.mu.Lock()
	c := t.client
	t.mu.Unlock()
	if c == nil {
		return nil
	}
	return newModel(c, id)
}

// ModelRef returns a reference to a jev model with a config, for the places
// that take a reference rather than a name, such as a fallback list. The ID
// may carry the typesafe/ prefix or not. With no config to attach,
// [ai.WithModelName] with the full name is the usual way to pick the model.
func ModelRef(id string, config *Config) ai.ModelRef {
	if config == nil {
		return ai.NewModelRef(modelName(id), nil)
	}
	return ai.NewModelRef(modelName(id), config)
}

func modelName(id string) string {
	return api.NewName(provider, internal.TrimProvider(provider, id))
}

// newModel builds the model action for one ID. The model claims constrained
// output, so the loop hands it the output schema untouched, and claims the
// system role and context so the loop does not rewrite either into
// instruction text that would land in the state.
func newModel(c *client, id string) *ai.ModelAction {
	m := &model{client: c, id: id}
	return ai.NewModelAction(modelName(id), &ai.ModelOptions{
		Label: internal.ProviderLabel("TypeSafe", id),
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			SystemRole:  true,
			Context:     true,
			Constrained: ai.ConstrainedSupportAll,
			Output:      []string{ai.OutputFormatJSON, ai.OutputFormatEnum},
			ContentType: []string{"application/json", "text/enum"},
		},
	}, m.generate)
}

// model is one resolved model.
type model struct {
	client *client
	id     string
}

// generate answers the questions the request's output schema encodes.
func (m *model) generate(ctx context.Context, req *ai.ModelRequest, cfg *Config, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	if req.Output == nil || (req.Output.Schema == nil && req.Output.Format != ai.OutputFormatEnum) {
		return nil, status.Errorf(status.ErrInvalidArgument, "typesafe: the model answers questions encoded in an output type; call GenerateData with a decision type or pass ai.WithOutputType")
	}
	enum := req.Output.Format == ai.OutputFormatEnum
	preamble, err := systemPreamble(req.Messages)
	if err != nil {
		return nil, err
	}
	var questions map[string]question
	if enum {
		questions, err = enumQuestion(req.Output.Schema, preamble)
	} else {
		questions, err = compileQuestions(req.Output.Schema, preamble)
	}
	if err != nil {
		return nil, err
	}
	state, err := buildState(req, cfg)
	if err != nil {
		return nil, err
	}
	wire := &request{State: state, Questions: questions}
	if cfg != nil {
		wire.Extra = cfg.Extra
	}

	resp, err := m.client.decide(ctx, m.id, wire)
	if err != nil {
		return nil, err
	}
	logger.Debug(ctx, "typesafe: answered", "model", resp.Model, "questions", len(questions), "inputTokens", resp.Usage.InputTokens, "outputTokens", resp.Usage.OutputTokens)

	text, err := answersText(resp, questions, enum)
	if err != nil {
		return nil, err
	}
	part := ai.NewJSONPart(text)
	if enum {
		part = ai.NewTextPart(text)
	}

	custom := map[string]any{"model": resp.Model, "answers": resp.Answers}
	if resp.Provider != "" {
		custom["provider"] = resp.Provider
	}
	if resp.ID != "" {
		custom["id"] = resp.ID
	}
	if resp.Usage.Cost != nil {
		custom["cost"] = *resp.Usage.Cost
	}
	return &ai.ModelResponse{
		Message:      ai.NewModelMessage(part),
		FinishReason: ai.FinishReasonStop,
		Usage: &ai.GenerationUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
			TotalTokens:  resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
		Custom: custom,
	}, nil
}

// answerFields lists the fields of a wire answer per question type, which
// are the fields the answer types declare. The answer schemas are closed,
// so a field the API or a gateway adds later is dropped here rather than
// failing every call at validation; the untouched answer is still on
// [ai.ModelResponse.Custom].
var answerFields = map[string][]string{
	kindChoice: {"choice", "probabilities", "confidence"},
	kindScore:  {"score", "probabilities", "confidence", "legend"},
	kindNoul:   {"noul"},
}

// answersText renders the answers as the message text: for the enum
// format the chosen option itself, otherwise a JSON object keyed by
// question ID whose values are the answers projected onto the fields the
// output type declares. A score's legend is rebuilt from the rubric's
// strings: for a level sent with guidance the API echoes the guidance,
// which the legend's strings cannot hold. A score is clamped to the rubric,
// since float rounding in the expected level can land a hair past the top
// level, which the schema's bounds would reject along with every other
// answer in the call.
func answersText(resp *response, questions map[string]question, enum bool) (string, error) {
	answers := make(map[string]map[string]any, len(questions))
	for _, id := range questionIDs(questions) {
		answer, ok := resp.Answers[id]
		if !ok {
			return "", status.Errorf(status.ErrInternal, "typesafe: no answer for question %q", id)
		}
		fields := answerFields[questions[id].Type]
		projected := make(map[string]any, len(fields))
		for _, field := range fields {
			if v, ok := answer[field]; ok {
				projected[field] = v
			}
		}
		if q := questions[id]; q.labels != nil {
			projected["legend"] = q.legend()
			if score, ok := projected["score"].(float64); ok {
				projected["score"] = min(max(score, 0), float64(len(q.labels)-1))
			}
		}
		answers[id] = projected
	}
	if enum {
		choice, _ := answers[enumQuestionID]["choice"].(string)
		if choice == "" {
			return "", status.Errorf(status.ErrInternal, "typesafe: the enum answer names no option")
		}
		return choice, nil
	}
	text, err := json.Marshal(answers)
	if err != nil {
		return "", status.Errorf(status.ErrInternal, "typesafe: answers are not JSON: %w", err)
	}
	return string(text), nil
}

// systemPreamble gathers the system messages into the text that goes in
// front of every question's instructions. The model has no system role: the
// state is what it evaluates and the questions are what it is told, so a
// system message belongs with the questions, never in the state. Only text
// can be instructions; loop plumbing left in a history is skipped.
func systemPreamble(messages []*ai.Message) (string, error) {
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
				return "", status.Errorf(status.ErrInvalidArgument, "typesafe: a %s part cannot be instructions; a system message is text only", partKindName(part))
			}
			sb.WriteString(part.Text)
		}
		if text := strings.TrimSpace(sb.String()); text != "" {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n\n"), nil
}

// buildState turns the request into the state: one message as its value,
// several as {role, content} records, and with documents attached an
// object of both, so a question can name the context it is about. System
// messages are instructions, not state, and are left out.
func buildState(req *ai.ModelRequest, cfg *Config) (any, error) {
	records := make([]map[string]any, 0, len(req.Messages))
	for _, msg := range req.Messages {
		if msg.Role == ai.RoleSystem {
			continue
		}
		value, err := messageValue(msg, cfg)
		if err != nil {
			return nil, err
		}
		if value == nil {
			continue
		}
		records = append(records, map[string]any{"role": string(msg.Role), "content": value})
	}
	if len(records) == 0 {
		return nil, status.Errorf(status.ErrInvalidArgument, "typesafe: the request carries no state; a system message is instructions, so pass a prompt or messages too")
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
func messageValue(msg *ai.Message, cfg *Config) (any, error) {
	values := make([]any, 0, len(msg.Content))
	for _, part := range msg.Content {
		value, err := partValue(part, cfg)
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
func partValue(part *ai.Part, cfg *Config) (any, error) {
	if isFormatInstructions(part) {
		return nil, nil
	}
	switch {
	case part.IsText():
		if cfg != nil && cfg.StateJSON {
			var value json.RawMessage
			if err := json.Unmarshal([]byte(part.Text), &value); err != nil {
				return nil, status.Errorf(status.ErrInvalidArgument, "typesafe: stateJSON is set but the text is not JSON: %w", err)
			}
			return value, nil
		}
		return part.Text, nil
	case part.IsData():
		return part.Data, nil
	default:
		return nil, status.Errorf(status.ErrInvalidArgument, "typesafe: a %s part cannot be state; the model takes text and JSON only", partKindName(part))
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
	content, err := messageValue(&ai.Message{Content: doc.Content}, nil)
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
