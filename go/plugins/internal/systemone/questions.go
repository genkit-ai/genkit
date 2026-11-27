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

// Package systemone implements the System One protocol that decision
// models speak: a state and a set of typed questions go out, and one typed
// answer per question comes back. It is shared by the plugins that serve
// such models; the question types callers declare decisions with are in
// the public package go/plugins/systemone/exp.
package systemone

import (
	"bytes"
	"encoding/json"
	"maps"
	"slices"
	"strconv"

	"github.com/firebase/genkit/go/core/status"
)

// The schema keywords a question is encoded with. The output schema is the
// one channel the generate loop hands a model besides the messages, so the
// question set rides on it: each property is one question, its description
// is the instructions, and these keywords carry what plain JSON Schema has
// no keyword for. [CompileQuestions] reads them back.
const (
	// KindKeyword marks a property as a question and names its type.
	KindKeyword = "x-systemone"
	// LevelsKeyword carries a score question's ordered level descriptions.
	LevelsKeyword = "x-levels"
	// TrueKeyword and FalseKeyword carry a noul question's optional
	// criteria, which a NoulOf emits from its type parameter.
	TrueKeyword  = "x-true"
	FalseKeyword = "x-false"
	// GuidanceKeyword carries a question's structured guidance from the
	// Guided companions, keyed as on the wire: by option, by level index,
	// or by "true" and "false".
	GuidanceKeyword = "x-guidance"
	// InstructionsKeyword carries a runtime question's instructions when
	// they are structured rather than text, which a description cannot
	// hold.
	InstructionsKeyword = "x-instructions"
)

// The question types, as the API names them.
const (
	KindChoice = "choice"
	KindScore  = "score"
	KindNoul   = "noul"
)

// Question is one question on the wire.
type Question struct {
	Type string `json:"type"`
	// Instructions is the question's text, or the structured value a
	// runtime question gives, after the preamble.
	Instructions any `json:"instructions"`
	// Criteria is the options and their descriptions for a choice question,
	// in order; a map of side to description for a noul question; and an
	// ordered list of level descriptions for a score question. A
	// description is a string, or the guidance that replaces it on the wire.
	Criteria any `json:"criteria,omitempty"`

	// Labels are a score question's level strings, which the legend is
	// built from whatever form the levels take on the wire.
	Labels []string `json:"-"`
}

// Options is a choice question's options with their descriptions, in the
// order they were declared. It encodes as a JSON object whose keys keep
// that order, where a Go map would sort them.
type Options []Option

// Option is one option of a choice question and its description: a
// string, or the guidance that replaces it on the wire.
type Option struct {
	Name        string
	Description any
}

// MarshalJSON implements [json.Marshaler].
func (c Options) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, entry := range c {
		if i > 0 {
			buf.WriteByte(',')
		}
		key, err := json.Marshal(entry.Name)
		if err != nil {
			return nil, err
		}
		value, err := json.Marshal(entry.Description)
		if err != nil {
			return nil, err
		}
		buf.Write(key)
		buf.WriteByte(':')
		buf.Write(value)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// legend maps each level number to its label, the shape of a Score's
// legend.
func (q Question) legend() map[string]any {
	legend := make(map[string]any, len(q.Labels))
	for i, label := range q.Labels {
		legend[strconv.Itoa(i)] = label
	}
	return legend
}

// CompileQuestions reads the questions an output schema encodes: one per
// property, each marked with the x-systemone keyword the answer types emit.
// A property without the marker is rejected, since the model has no way to
// answer a shape it does not know. The preamble, the request's system text,
// goes in front of every question's own instructions, followed by the
// schema's own description when it has one, as it is for the enum format.
func CompileQuestions(schema map[string]any, preamble string) (map[string]Question, error) {
	props, _ := schema["properties"].(map[string]any)
	if len(props) == 0 {
		return nil, status.Errorf(status.ErrInvalidSchema, "systemone: the output schema has no properties; each question is a field of the output type")
	}
	description, _ := schema["description"].(string)
	preamble = joinInstructions(preamble, description)
	questions := make(map[string]Question, len(props))
	for id, raw := range props {
		prop, _ := raw.(map[string]any)
		kind, _ := prop[KindKeyword].(string)
		if kind == "" {
			return nil, status.Errorf(status.ErrInvalidSchema, "systemone: field %q is not a question; use a Choice, Score, or Noul field", id)
		}
		instructions, err := questionInstructions(id, prop, preamble)
		if err != nil {
			return nil, err
		}
		q := Question{Type: kind, Instructions: instructions}
		switch kind {
		case KindChoice:
			q.Criteria, err = choiceCriteria(id, prop)
		case KindScore:
			q.Criteria, q.Labels, err = scoreCriteria(id, prop)
		case KindNoul:
			q.Criteria, err = noulCriteria(id, prop)
		default:
			err = status.Errorf(status.ErrInvalidSchema, "systemone: question %q has unknown type %q", id, kind)
		}
		if err != nil {
			return nil, err
		}
		questions[id] = q
	}
	return questions, nil
}

// questionInstructions reads a question's instructions: the structured
// value a runtime question carries on the x-instructions keyword, or the
// property's description. The preamble goes in front: as a paragraph of
// the text, as the first element of a value that encodes as a JSON array,
// or as the first element beside any other structured value.
func questionInstructions(id string, prop map[string]any, preamble string) (any, error) {
	if structured := prop[InstructionsKeyword]; structured != nil {
		if preamble == "" {
			return structured, nil
		}
		if items, ok := jsonArray(structured); ok {
			return append([]any{preamble}, items...), nil
		}
		return []any{preamble, structured}, nil
	}
	text, _ := prop["description"].(string)
	if text == "" {
		return nil, status.Errorf(status.ErrInvalidSchema, "systemone: question %q has no instructions; set jsonschema_description on the field, or Instructions on a runtime question", id)
	}
	return joinInstructions(preamble, text), nil
}

// jsonArray returns the elements of a value that encodes as a JSON array.
// A schema built by Schema arrives decoded, as []any; a raw schema map can
// carry a []string or a json.RawMessage, which go on the wire the same way.
func jsonArray(v any) ([]any, bool) {
	if items, ok := v.([]any); ok {
		return items, true
	}
	data, err := json.Marshal(v)
	if err != nil {
		return nil, false
	}
	var items []any
	if json.Unmarshal(data, &items) != nil || items == nil {
		return nil, false
	}
	return items, true
}

// choiceCriteria reads the options back out of the oneOf a Choice emits,
// in order, with the guidance of a GuidedOption in place of an option's
// string. An option with neither is described by its own name, since the
// wire format takes a description per option and some gateways reject a
// null one.
func choiceCriteria(id string, prop map[string]any) (Options, error) {
	props, _ := prop["properties"].(map[string]any)
	choice, _ := props["choice"].(map[string]any)
	oneOf, _ := choice["oneOf"].([]any)
	if len(oneOf) == 0 {
		return nil, status.Errorf(status.ErrInvalidSchema, "systemone: choice question %q lists no options", id)
	}
	guidance, _ := prop[GuidanceKeyword].(map[string]any)
	options := make(Options, 0, len(oneOf))
	seen := make(map[string]bool, len(oneOf))
	for _, raw := range oneOf {
		option, _ := raw.(map[string]any)
		key, _ := option["const"].(string)
		if key == "" {
			return nil, status.Errorf(status.ErrInvalidSchema, "systemone: choice question %q has an option without a const", id)
		}
		if seen[key] {
			return nil, status.Errorf(status.ErrInvalidSchema, "systemone: choice question %q lists option %q twice", id, key)
		}
		seen[key] = true
		description, _ := option["description"].(string)
		value := withWhat(guidance[key], description)
		if value == "" {
			value = key
		}
		options = append(options, Option{Name: key, Description: value})
	}
	for _, key := range slices.Sorted(maps.Keys(guidance)) {
		if !seen[key] {
			return nil, status.Errorf(status.ErrInvalidSchema, "systemone: choice question %q has guidance for %q, which is not one of its options", id, key)
		}
	}
	return options, nil
}

// scoreCriteria reads the ordered levels off the x-levels keyword, with
// the guidance of a GuidedRubric in place of a level's string, and
// returns the strings too for the legend. The API takes two to ten levels;
// the lower bound is checked here because one level is not a scale, the
// upper bound is the API's to enforce.
func scoreCriteria(id string, prop map[string]any) ([]any, []string, error) {
	levels, ok := stringList(prop[LevelsKeyword])
	if !ok {
		return nil, nil, status.Errorf(status.ErrInvalidSchema, "systemone: score question %q has a level that is not a string", id)
	}
	if len(levels) < 2 {
		return nil, nil, status.Errorf(status.ErrInvalidSchema, "systemone: score question %q needs at least two levels", id)
	}
	criteria := make([]any, len(levels))
	for i, level := range levels {
		criteria[i] = level
	}
	guidance, _ := prop[GuidanceKeyword].(map[string]any)
	for _, key := range slices.Sorted(maps.Keys(guidance)) {
		i, err := strconv.Atoi(key)
		if err != nil || i < 0 || i >= len(levels) {
			return nil, nil, status.Errorf(status.ErrInvalidSchema, "systemone: score question %q has guidance for level %s, which its rubric does not have", id, key)
		}
		criteria[i] = withWhat(guidance[key], levels[i])
	}
	return criteria, levels, nil
}

// noulCriteria reads the optional true and false criteria a NoulOf
// emits, with the guidance of a GuidedYesNo in place of a side's
// string. They are sent as a pair or not at all: the API describes the
// pair, and one side alone would leave the other implied. No criteria is
// an untyped nil, so the field stays absent rather than null, which the
// gateways reject.
func noulCriteria(id string, prop map[string]any) (any, error) {
	guidance, _ := prop[GuidanceKeyword].(map[string]any)
	criteria := make(map[string]any, 2)
	for side, keyword := range map[string]string{"true": TrueKeyword, "false": FalseKeyword} {
		text, _ := prop[keyword].(string)
		if value := withWhat(guidance[side], text); value != "" {
			criteria[side] = value
		}
	}
	switch len(criteria) {
	case 0:
		return nil, nil
	case 1:
		return nil, status.Errorf(status.ErrInvalidSchema, "systemone: noul question %q says what only one side means; give both yes and no, in Criteria or Guidance", id)
	}
	return criteria, nil
}

// withWhat is a description as it goes on the wire: the guidance when
// there is some, the string otherwise. An object with no "what" gets the
// string as its what, so guidance adds to the description rather than
// replacing it; any other guidance is sent as it is.
func withWhat(guidance any, description string) any {
	switch g := guidance.(type) {
	case nil:
		return description
	case map[string]any:
		if g == nil {
			return description
		}
		if _, ok := g["what"]; ok || description == "" {
			return g
		}
		g = maps.Clone(g)
		g["what"] = description
		return g
	default:
		return g
	}
}

// EnumQuestionID names the one question an enum-format request asks.
const EnumQuestionID = "choice"

// EnumQuestion turns an enum output schema into a single choice question,
// so the built-in enum format works on the model with no decision type.
// The instructions are the preamble, the request's system text, followed
// by the schema's description when it has one; the options keep the order
// the values were given in, and an option's description is its own name,
// since an enum carries none.
func EnumQuestion(schema map[string]any, preamble string) (map[string]Question, error) {
	values, ok := stringList(schema["enum"])
	if !ok || slices.Contains(values, "") {
		return nil, status.Errorf(status.ErrInvalidSchema, "systemone: enum output values must be non-empty strings")
	}
	if len(values) == 0 {
		return nil, status.Errorf(status.ErrInvalidSchema, "systemone: the enum output schema lists no values")
	}
	options := make(Options, 0, len(values))
	for _, v := range values {
		if slices.ContainsFunc(options, func(o Option) bool { return o.Name == v }) {
			return nil, status.Errorf(status.ErrInvalidSchema, "systemone: the enum output schema lists %q twice", v)
		}
		options = append(options, Option{Name: v, Description: v})
	}
	description, _ := schema["description"].(string)
	instructions := joinInstructions(preamble, description)
	if instructions == "" {
		instructions = "Choose the option that best describes the state."
	}
	return map[string]Question{
		EnumQuestionID: {Type: KindChoice, Instructions: instructions, Criteria: options},
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
func questionIDs(questions map[string]Question) []string {
	return slices.Sorted(maps.Keys(questions))
}
