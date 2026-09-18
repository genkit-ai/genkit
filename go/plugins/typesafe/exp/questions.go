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
	"maps"
	"slices"
	"strconv"

	"github.com/firebase/genkit/go/core/status"
	"github.com/invopop/jsonschema"
)

// The schema keywords a question is encoded with. The output schema is the
// one channel the generate loop hands a model besides the messages, so the
// question set rides on it: each property is one question, its description
// is the instructions, and these keywords carry what plain JSON Schema has
// no keyword for. [compileQuestions] reads them back.
const (
	// kindKeyword marks a property as a question and names its type.
	kindKeyword = "x-typesafe"
	// levelsKeyword carries a score question's ordered level descriptions.
	levelsKeyword = "x-levels"
	// trueKeyword and falseKeyword carry a noul question's optional
	// criteria. A field sets them through its jsonschema_extras tag:
	//
	//	jsonschema_extras:"x-true=Explicitly time-sensitive,x-false=No urgency expressed"
	trueKeyword  = "x-true"
	falseKeyword = "x-false"
)

// The question types, as the API names them.
const (
	kindChoice = "choice"
	kindScore  = "score"
	kindNoul   = "noul"
)

// Option is the key type of a [Choice]: a string type that lists its own
// options and what each one means. The criteria are what the model
// chooses among, so they belong to the type rather than to any one call.
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
type Option[T ~string] interface {
	~string
	Criteria() map[T]string
}

// Rubric supplies the ordered levels of a [Score], lowest first. The level's
// index is its number on the wire, so a three-level rubric scores 0 to 2.
type Rubric interface {
	Levels() []string
}

// Choice is the answer to a choice question: one option from T's criteria,
// the probability of every option, and how concentrated that distribution
// is. Its JSON is the wire answer, so an output type built from these
// fields is filled straight from the response.
//
// Probabilities and Confidence are set only by a System One model. The
// [DecisionFormat] clears them on any other model's answer, so a zero
// Confidence means unknown, never certain.
type Choice[T Option[T]] struct {
	Choice        T             `json:"choice"`
	Probabilities map[T]float64 `json:"probabilities,omitempty"`
	Confidence    float64       `json:"confidence,omitzero"`
}

// JSONSchema encodes the question: the options and their criteria as a
// oneOf of constants, marked as a choice question for [compileQuestions].
func (Choice[T]) JSONSchema() *jsonschema.Schema {
	var zero T
	criteria := zero.Criteria()
	options := make([]*jsonschema.Schema, 0, len(criteria))
	for _, key := range slices.Sorted(maps.Keys(criteria)) {
		options = append(options, &jsonschema.Schema{Const: string(key), Description: criteria[key]})
	}
	props := jsonschema.NewProperties()
	props.Set("choice", &jsonschema.Schema{Type: "string", OneOf: options})
	props.Set("probabilities", probabilitiesSchema())
	props.Set("confidence", &jsonschema.Schema{Type: "number"})
	return answerSchema(kindChoice, &jsonschema.Schema{Properties: props}, "choice")
}

// Noul is the answer to a yes/no question: the probability that the
// statement is true. A value near 0.5 means the model could not tell, not
// that the answer is "somewhat". There is no separate confidence; the
// probability is the whole answer.
//
// A field of this type may carry the true and false criteria in its
// jsonschema_extras tag under x-true and x-false.
type Noul struct {
	Probability float64 `json:"noul"`
}

// JSONSchema encodes the question, marked as a noul question for
// [compileQuestions].
func (Noul) JSONSchema() *jsonschema.Schema {
	props := jsonschema.NewProperties()
	props.Set("noul", &jsonschema.Schema{Type: "number", Minimum: "0", Maximum: "1"})
	return answerSchema(kindNoul, &jsonschema.Schema{Properties: props}, "noul")
}

// Score is the answer to a rubric question: the expected level, computed
// from the probability of each level, so it falls between levels when the
// model is split. Legend maps each level number back to its description.
//
// Probabilities, Confidence, and Legend are set only by a System One
// model. The [DecisionFormat] clears them on any other model's answer and
// rounds Score to a whole level.
type Score[L Rubric] struct {
	Score         float64            `json:"score"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence,omitzero"`
	Legend        map[string]string  `json:"legend,omitempty"`
}

// JSONSchema encodes the question: the level descriptions ride on the
// x-levels keyword, since a fractional score cannot be a oneOf of levels.
func (Score[L]) JSONSchema() *jsonschema.Schema {
	var zero L
	levels := zero.Levels()
	props := jsonschema.NewProperties()
	props.Set("score", &jsonschema.Schema{
		Type:    "number",
		Minimum: "0",
		Maximum: json.Number(strconv.Itoa(max(len(levels)-1, 0))),
	})
	props.Set("probabilities", probabilitiesSchema())
	props.Set("confidence", &jsonschema.Schema{Type: "number"})
	props.Set("legend", &jsonschema.Schema{Type: "object", AdditionalProperties: &jsonschema.Schema{Type: "string"}})
	s := answerSchema(kindScore, &jsonschema.Schema{Properties: props}, "score")
	s.Extras[levelsKeyword] = levels
	return s
}

func probabilitiesSchema() *jsonschema.Schema {
	return &jsonschema.Schema{Type: "object", AdditionalProperties: &jsonschema.Schema{Type: "number"}}
}

// answerSchema completes the object schema shared by the three answer
// types. It is closed, so the answer the model returns must be exactly the
// wire answer with the type discriminator removed.
func answerSchema(kind string, s *jsonschema.Schema, required ...string) *jsonschema.Schema {
	s.Type = "object"
	s.Required = required
	s.AdditionalProperties = jsonschema.FalseSchema
	s.Extras = map[string]any{kindKeyword: kind}
	return s
}

// question is one question on the wire.
type question struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	// Criteria is a map of option to description for choice and noul
	// questions and an ordered list of level descriptions for score
	// questions.
	Criteria any `json:"criteria,omitempty"`
}

// compileQuestions reads the questions an output schema encodes: one per
// property, each marked with the x-typesafe keyword the answer types emit.
// A property without the marker is rejected, since the model has no way to
// answer a shape it does not know. The preamble, the request's system text,
// goes in front of every question's own instructions.
func compileQuestions(schema map[string]any, preamble string) (map[string]question, error) {
	props, _ := schema["properties"].(map[string]any)
	if len(props) == 0 {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: the output schema has no properties; each question is a field of the output type")
	}
	questions := make(map[string]question, len(props))
	for id, raw := range props {
		prop, _ := raw.(map[string]any)
		kind, _ := prop[kindKeyword].(string)
		instructions, _ := prop["description"].(string)
		if kind == "" {
			return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: field %q is not a question; use a Choice, Score, or Noul field", id)
		}
		if instructions == "" {
			return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: question %q has no instructions; set jsonschema_description on the field", id)
		}
		q := question{Type: kind, Instructions: joinInstructions(preamble, instructions)}
		var err error
		switch kind {
		case kindChoice:
			q.Criteria, err = choiceCriteria(id, prop)
		case kindScore:
			q.Criteria, err = scoreCriteria(id, prop)
		case kindNoul:
			// Assigned only when present: a typed nil map inside the any
			// would marshal as null, which the gateways reject, where an
			// absent field is the documented way to give no criteria.
			var criteria map[string]string
			if criteria, err = noulCriteria(id, prop); criteria != nil {
				q.Criteria = criteria
			}
		default:
			err = status.Errorf(status.ErrInvalidSchema, "typesafe: question %q has unknown type %q", id, kind)
		}
		if err != nil {
			return nil, err
		}
		questions[id] = q
	}
	return questions, nil
}

// choiceCriteria reads the options back out of the oneOf a [Choice] emits.
// An option without a description is described by its own name, since the
// wire format takes a description per option and some gateways reject a
// null one.
func choiceCriteria(id string, prop map[string]any) (map[string]string, error) {
	props, _ := prop["properties"].(map[string]any)
	choice, _ := props["choice"].(map[string]any)
	oneOf, _ := choice["oneOf"].([]any)
	if len(oneOf) == 0 {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: choice question %q lists no options", id)
	}
	criteria := make(map[string]string, len(oneOf))
	for _, raw := range oneOf {
		option, _ := raw.(map[string]any)
		key, _ := option["const"].(string)
		if key == "" {
			return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: choice question %q has an option without a const", id)
		}
		description, _ := option["description"].(string)
		if description == "" {
			description = key
		}
		criteria[key] = description
	}
	return criteria, nil
}

// scoreCriteria reads the ordered levels off the x-levels keyword. The API
// takes two to ten levels; the lower bound is checked here because one
// level is not a scale, the upper bound is the API's to enforce.
func scoreCriteria(id string, prop map[string]any) ([]string, error) {
	levels, ok := stringList(prop[levelsKeyword])
	if !ok {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: score question %q has a level that is not a string", id)
	}
	if len(levels) < 2 {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: score question %q needs at least two levels", id)
	}
	return levels, nil
}

// noulCriteria reads the optional true and false criteria. They are sent
// as a pair or not at all: the API describes the pair, and one side alone
// would leave the other implied.
func noulCriteria(id string, prop map[string]any) (map[string]string, error) {
	yes, _ := prop[trueKeyword].(string)
	no, _ := prop[falseKeyword].(string)
	switch {
	case yes == "" && no == "":
		return nil, nil
	case yes == "" || no == "":
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: noul question %q sets only one of x-true and x-false", id)
	}
	return map[string]string{"true": yes, "false": no}, nil
}

// enumQuestionID names the one question an enum-format request asks.
const enumQuestionID = "choice"

// enumQuestion turns an enum output schema into a single choice question,
// so the built-in enum format works on the model with no decision type.
// The instructions are the preamble, the request's system text, followed
// by the schema's description when it has one; an option's description is
// its own name, since an enum carries none.
func enumQuestion(schema map[string]any, preamble string) (map[string]question, error) {
	values, ok := stringList(schema["enum"])
	if !ok || slices.Contains(values, "") {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: enum output values must be non-empty strings")
	}
	if len(values) == 0 {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: the enum output schema lists no values")
	}
	criteria := make(map[string]string, len(values))
	for _, v := range values {
		criteria[v] = v
	}
	description, _ := schema["description"].(string)
	instructions := joinInstructions(preamble, description)
	if instructions == "" {
		instructions = "Choose the option that best describes the state."
	}
	return map[string]question{
		enumQuestionID: {Type: kindChoice, Instructions: instructions, Criteria: criteria},
	}, nil
}

// joinInstructions puts the preamble in front of a question's own text as
// two paragraphs, or returns whichever of the two is present.
func joinInstructions(preamble, text string) string {
	switch {
	case preamble == "":
		return text
	case text == "":
		return preamble
	default:
		return preamble + "\n\n" + text
	}
}

// stringList reads a list of strings that arrived either as the []string a
// Go caller built or as the []any a JSON round trip produces. It reports
// false when an element is not a string, and an empty list for any other
// value.
func stringList(v any) ([]string, bool) {
	switch raw := v.(type) {
	case []string:
		return raw, true
	case []any:
		list := make([]string, 0, len(raw))
		for _, item := range raw {
			s, ok := item.(string)
			if !ok {
				return nil, false
			}
			list = append(list, s)
		}
		return list, true
	}
	return nil, true
}

// questionIDs returns the question IDs in a stable order.
func questionIDs(questions map[string]question) []string {
	return slices.Sorted(maps.Keys(questions))
}
