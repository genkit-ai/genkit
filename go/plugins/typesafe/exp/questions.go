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
	"bytes"
	"cmp"
	"encoding/json"
	"maps"
	"math"
	"slices"
	"strconv"

	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
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
	// criteria, which [NoulOf] emits from its type parameter.
	trueKeyword  = "x-true"
	falseKeyword = "x-false"
	// guidanceKeyword carries a question's structured guidance from the
	// Guided companions, keyed as on the wire: by option, by level index,
	// or by "true" and "false".
	guidanceKeyword = "x-guidance"
	// instructionsKeyword carries a runtime question's instructions when
	// they are structured rather than text, which a description cannot
	// hold.
	instructionsKeyword = "x-instructions"
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
// A description is a string; [GuidedOption] adds the structured form the
// wire format also takes. Options known only at run time are asked with a
// [ChoiceQuestion] instead.
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
// Probabilities is the model's distribution over the options and sums to
// 1; Choice is the option with the most of it; Confidence, from 0 to 1, is
// how concentrated the distribution is.
type Choice[T Option[T]] struct {
	Choice        T             `json:"choice"`
	Probabilities map[T]float64 `json:"probabilities,omitempty"`
	Confidence    float64       `json:"confidence"`
}

// JSONSchema encodes the question: the options and their criteria as a
// oneOf of constants, marked as a choice question for [compileQuestions].
// The options go out sorted by name, since a map has no order of its own.
func (Choice[T]) JSONSchema() *jsonschema.Schema {
	var zero T
	criteria := zero.Criteria()
	options := make([]ChoiceOption, 0, len(criteria))
	for _, key := range slices.Sorted(maps.Keys(criteria)) {
		options = append(options, ChoiceOption{Name: string(key), Criteria: criteria[key]})
	}
	s := choiceSchema(options)
	if g, ok := any(zero).(GuidedOption[T]); ok {
		setGuidance(s, g.Guidance(), func(key T) string { return string(key) })
	}
	return s
}

// choiceSchema encodes a choice question's options, in order, with the
// guidance any of them carries.
func choiceSchema(options []ChoiceOption) *jsonschema.Schema {
	oneOf := make([]*jsonschema.Schema, 0, len(options))
	guidance := make(map[string]any)
	for _, option := range options {
		oneOf = append(oneOf, &jsonschema.Schema{Const: option.Name, Description: option.Criteria})
		if option.Guidance != nil {
			guidance[option.Name] = option.Guidance
		}
	}
	props := jsonschema.NewProperties()
	props.Set("choice", &jsonschema.Schema{Type: "string", OneOf: oneOf})
	props.Set("probabilities", probabilitiesSchema())
	props.Set("confidence", &jsonschema.Schema{Type: "number"})
	s := answerSchema(kindChoice, &jsonschema.Schema{Properties: props}, "choice")
	setGuidance(s, guidance, func(name string) string { return name })
	return s
}

// Ranked is the options from most to least likely, ties broken by name. It
// is empty when the answer carries no distribution.
func (c Choice[T]) Ranked() []T {
	return slices.SortedFunc(maps.Keys(c.Probabilities), func(a, b T) int {
		return cmp.Or(cmp.Compare(c.Probabilities[b], c.Probabilities[a]), cmp.Compare(a, b))
	})
}

// Margin is the probability gap between the two most likely options: how
// decisive the choice is, where Confidence reads the whole distribution.
// It is zero with fewer than two options in the distribution.
func (c Choice[T]) Margin() float64 {
	ranked := c.Ranked()
	if len(ranked) < 2 {
		return 0
	}
	return c.Probabilities[ranked[0]] - c.Probabilities[ranked[1]]
}

// YesNo supplies what yes and no mean for a [NoulOf] question, as the
// criteria of a [Choice] and the levels of a [Score] come from their
// types. Both sides are given: the API takes the pair or nothing.
//
//	type Urgent struct{}
//
//	func (Urgent) Criteria() (yes, no string) {
//		return "Names a deadline, or says now or today", "No time pressure is expressed"
//	}
type YesNo interface {
	Criteria() (yes, no string)
}

// NoCriteria is the [YesNo] of a plain [Noul]: the question is its
// description alone.
type NoCriteria struct{}

// Criteria implements [YesNo] with no criteria.
func (NoCriteria) Criteria() (yes, no string) { return "", "" }

// GuidedOption is an optional companion to [Option] for the structured
// form of a description the wire format takes: any JSON value, most often
// an object with labeled parts, such as what, not_for, and examples for an
// option. The strings of Criteria stay required, and the schema and the
// answers show them; guidance replaces a string on the wire only.
//
// A value that encodes as a JSON object with no "what" key gets the
// string as its "what", so guidance adds to the description rather than
// replacing it. Any other value is sent as it is. An option that is
// absent, or whose value is nil, keeps its string.
//
//	func (Dept) Guidance() map[Dept]any {
//		return map[Dept]any{
//			"billing": map[string]any{
//				"not_for":  "Progress of a refund already issued",
//				"examples": []string{"I was charged twice for one order."},
//			},
//		}
//	}
type GuidedOption[T ~string] interface {
	Guidance() map[T]any
}

// GuidedRubric is the [GuidedOption] of a [Rubric]: guidance keyed by
// level index, lowest level 0, such as summary and signals for a level.
// [Score.Legend] and [Score.Label] keep the strings of Levels.
type GuidedRubric interface {
	Guidance() map[int]any
}

// GuidedYesNo is the [GuidedOption] of a [YesNo]: guidance for the yes
// side and the no side, such as what and examples. The pair rule holds
// across both: each side needs a string or guidance, or neither does.
type GuidedYesNo interface {
	Guidance() (yes, no any)
}

// NoulOf is the answer to a yes/no question whose criteria come from C:
// the probability that the statement is true. A value near 0.5 means the
// model could not tell, not that the answer is "somewhat". There is no
// separate confidence; the probability is the whole answer.
type NoulOf[C YesNo] struct {
	Probability float64 `json:"noul"`
}

// Noul is a [NoulOf] with no criteria: the question is its description
// alone, which is the usual yes/no question.
type Noul = NoulOf[NoCriteria]

// JSONSchema encodes the question, marked as a noul question for
// [compileQuestions], with C's criteria on the x-true and x-false keywords
// when it has any.
func (NoulOf[C]) JSONSchema() *jsonschema.Schema {
	var zero C
	yes, no := zero.Criteria()
	var yesGuidance, noGuidance any
	if g, ok := any(zero).(GuidedYesNo); ok {
		yesGuidance, noGuidance = g.Guidance()
	}
	return noulSchema(yes, no, yesGuidance, noGuidance)
}

// noulSchema encodes a noul question with its optional criteria and
// guidance for each side.
func noulSchema(yes, no string, yesGuidance, noGuidance any) *jsonschema.Schema {
	props := jsonschema.NewProperties()
	props.Set("noul", &jsonschema.Schema{Type: "number", Minimum: "0", Maximum: "1"})
	s := answerSchema(kindNoul, &jsonschema.Schema{Properties: props}, "noul")
	if yes != "" || no != "" {
		s.Extras[trueKeyword] = yes
		s.Extras[falseKeyword] = no
	}
	setGuidance(s, map[bool]any{true: yesGuidance, false: noGuidance}, strconv.FormatBool)
	return s
}

// Score is the answer to a rubric question: the expected level, computed
// from the probability of each level, so it falls between levels when the
// model is split. Probabilities is that distribution, keyed by level
// number; Confidence, from 0 to 1, is how concentrated it is; Legend maps
// each level number back to its description, the string from L even when
// the level was sent with guidance.
type Score[L Rubric] struct {
	Score         float64            `json:"score"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence"`
	Legend        map[string]string  `json:"legend,omitempty"`
}

// JSONSchema encodes the question: the level descriptions ride on the
// x-levels keyword, since a fractional score cannot be a oneOf of levels.
func (Score[L]) JSONSchema() *jsonschema.Schema {
	var zero L
	var guidance map[int]any
	if g, ok := any(zero).(GuidedRubric); ok {
		guidance = g.Guidance()
	}
	return scoreSchema(zero.Levels(), guidance)
}

// scoreSchema encodes a score question's ordered levels and the guidance
// keyed by level index.
func scoreSchema(levels []string, guidance map[int]any) *jsonschema.Schema {
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
	setGuidance(s, guidance, strconv.Itoa)
	return s
}

// Level is the nearest whole level, clamped to the rubric.
func (s Score[L]) Level() int {
	var zero L
	n := len(zero.Levels())
	if n == 0 {
		return 0
	}
	return min(max(int(math.Round(s.Score)), 0), n-1)
}

// Label is the rubric's description of [Score.Level]. It comes from L, so
// it does not depend on the answer carrying a legend.
func (s Score[L]) Label() string {
	var zero L
	levels := zero.Levels()
	if len(levels) == 0 {
		return ""
	}
	return levels[s.Level()]
}

// Question is one question of a set built at run time, for questions whose
// options, levels, or instructions come from data: the tools or products
// on hand, the nodes of a taxonomy, a tenant's own categories.
// [ChoiceQuestion], [ScoreQuestion], and [NoulQuestion] implement it, and
// [Schema] turns a set of them into an output schema. A question known
// when the code is written reads better as a field of an output type.
type Question interface {
	schema() *jsonschema.Schema
}

// ChoiceQuestion asks for one of its options, which go out in the order
// given.
type ChoiceQuestion struct {
	// Instructions is the question: a string, or any JSON value, such as
	// an object that names the field of the state the question is about.
	Instructions any
	Options      []ChoiceOption
}

// ChoiceOption is one option of a [ChoiceQuestion]: the name that is the
// answer when it is chosen, what it means, and optionally the structured
// form of that meaning, as [GuidedOption] gives it for a [Choice]. An
// option with neither criteria nor guidance is described by its name.
type ChoiceOption struct {
	Name     string
	Criteria string
	Guidance any
}

// ScoreQuestion rates the state on its levels, lowest first, as a [Rubric]
// gives them, with guidance keyed by level index as [GuidedRubric] gives
// it.
type ScoreQuestion struct {
	// Instructions is the question: a string, or any JSON value.
	Instructions any
	Levels       []string
	Guidance     map[int]any
}

// NoulQuestion asks whether a statement is true, with what yes and no mean
// as [YesNo] and [GuidedYesNo] give them for a [NoulOf]: each side needs a
// string or guidance, or neither does.
type NoulQuestion struct {
	// Instructions is the question: a string, or any JSON value.
	Instructions            any
	Yes, No                 string
	YesGuidance, NoGuidance any
}

func (q ChoiceQuestion) schema() *jsonschema.Schema {
	return withInstructions(choiceSchema(q.Options), q.Instructions)
}

func (q ScoreQuestion) schema() *jsonschema.Schema {
	return withInstructions(scoreSchema(q.Levels, q.Guidance), q.Instructions)
}

func (q NoulQuestion) schema() *jsonschema.Schema {
	return withInstructions(noulSchema(q.Yes, q.No, q.YesGuidance, q.NoGuidance), q.Instructions)
}

// withInstructions puts a runtime question's instructions on its schema:
// text as the description, as a field's tag gives it, and any other value
// on the x-instructions keyword.
func withInstructions(s *jsonschema.Schema, instructions any) *jsonschema.Schema {
	switch v := instructions.(type) {
	case nil:
	case string:
		s.Description = v
	default:
		s.Extras[instructionsKeyword] = v
	}
	return s
}

// Schema is the output schema that asks the given questions, each answered
// under its key. Pass it with [ai.WithOutputSchema] and read the answers as
// a map of [Answer]:
//
//	options := make([]typesafex.ChoiceOption, 0, len(tools))
//	for _, tool := range tools {
//		options = append(options, typesafex.ChoiceOption{Name: tool.Name, Criteria: tool.Description})
//	}
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithModelName("typesafe/jev-1.13.0"),
//		ai.WithOutputSchema(typesafex.Schema(map[string]typesafex.Question{
//			"tool": typesafex.ChoiceQuestion{Instructions: "Which tool serves the request?", Options: options},
//		})),
//		ai.WithPrompt(request))
//	if err != nil {
//		return err
//	}
//	var answers map[string]typesafex.Answer
//	if err := resp.Output(&answers); err != nil {
//		return err
//	}
//	tool := answers["tool"].Choice
func Schema(questions map[string]Question) map[string]any {
	ids := slices.Sorted(maps.Keys(questions))
	props := jsonschema.NewProperties()
	for _, id := range ids {
		if q := questions[id]; q != nil {
			props.Set(id, q.schema())
		} else {
			// Left unmarked, so the request is refused as asking nothing.
			props.Set(id, &jsonschema.Schema{})
		}
	}
	return base.SchemaAsMap(&jsonschema.Schema{
		Type:                 "object",
		Properties:           props,
		Required:             ids,
		AdditionalProperties: jsonschema.FalseSchema,
	})
}

// Answer is the answer to a question of a [Schema], with the fields of its
// kind set. Type names the kind: "choice" sets Choice, Probabilities over
// the options, and Confidence; "score" sets Score, Probabilities keyed by
// level number, Confidence, and Legend; "noul" sets Probability alone.
// They mean what the same fields of [Choice], [Score], and [NoulOf] mean.
// Decoding an answer sets Type from the field that names the kind, so a
// response's answers need no discriminator.
type Answer struct {
	Type          string             `json:"type,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         float64            `json:"score,omitzero"`
	Probability   float64            `json:"noul,omitzero"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence,omitzero"`
	Legend        map[string]string  `json:"legend,omitempty"`
}

// UnmarshalJSON implements [json.Unmarshaler]. A missing type is read
// from the field every answer of a kind carries: choice, score, or noul.
func (a *Answer) UnmarshalJSON(data []byte) error {
	type plain Answer
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	if err := json.Unmarshal(data, (*plain)(a)); err != nil {
		return err
	}
	if a.Type == "" {
		for _, kind := range []string{kindChoice, kindScore, kindNoul} {
			if _, ok := fields[kind]; ok {
				a.Type = kind
				break
			}
		}
	}
	return nil
}

// MarshalJSON implements [json.Marshaler]. It writes the fields of the
// answer's kind, zeros included, so a score of 0 or a flat distribution's
// confidence of 0 survives a round trip, and none of another kind's. An
// answer with no Type writes its non-zero fields.
func (a Answer) MarshalJSON() ([]byte, error) {
	type plain Answer
	out := map[string]any{"type": a.Type}
	switch a.Type {
	case kindChoice:
		out["choice"], out["confidence"] = a.Choice, a.Confidence
		if a.Probabilities != nil {
			out["probabilities"] = a.Probabilities
		}
	case kindScore:
		out["score"], out["confidence"] = a.Score, a.Confidence
		if a.Probabilities != nil {
			out["probabilities"] = a.Probabilities
		}
		if a.Legend != nil {
			out["legend"] = a.Legend
		}
	case kindNoul:
		out["noul"] = a.Probability
	default:
		return json.Marshal(plain(a))
	}
	return json.Marshal(out)
}

func probabilitiesSchema() *jsonschema.Schema {
	return &jsonschema.Schema{Type: "object", AdditionalProperties: &jsonschema.Schema{Type: "number"}}
}

// setGuidance puts the non-nil guidance values on the schema's
// x-guidance keyword, keyed as on the wire.
func setGuidance[K comparable](s *jsonschema.Schema, guidance map[K]any, key func(K) string) {
	wire := make(map[string]any, len(guidance))
	for k, v := range guidance {
		if v != nil {
			wire[key(k)] = v
		}
	}
	if len(wire) > 0 {
		s.Extras[guidanceKeyword] = wire
	}
}

// answerSchema completes the object schema shared by the answer types. It
// is closed: answersText projects a wire answer onto exactly these fields
// before the generate loop validates it.
func answerSchema(kind string, s *jsonschema.Schema, required ...string) *jsonschema.Schema {
	s.Type = "object"
	s.Required = required
	s.AdditionalProperties = jsonschema.FalseSchema
	s.Extras = map[string]any{kindKeyword: kind}
	return s
}

// question is one question on the wire.
type question struct {
	Type string `json:"type"`
	// Instructions is the question's text, or the structured value a
	// runtime question gives, after the preamble.
	Instructions any `json:"instructions"`
	// Criteria is the options and their descriptions for a choice question,
	// in order; a map of side to description for a noul question; and an
	// ordered list of level descriptions for a score question. A
	// description is a string, or the guidance that replaces it on the wire.
	Criteria any `json:"criteria,omitempty"`

	// labels are a score question's level strings, which the legend is
	// built from whatever form the levels take on the wire.
	labels []string
}

// criteria is a choice question's options with their descriptions, in the
// order they were declared. It encodes as a JSON object whose keys keep
// that order, where a Go map would sort them.
type criteria []criterion

type criterion struct {
	option      string
	description any
}

// MarshalJSON implements [json.Marshaler].
func (c criteria) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, entry := range c {
		if i > 0 {
			buf.WriteByte(',')
		}
		key, err := json.Marshal(entry.option)
		if err != nil {
			return nil, err
		}
		value, err := json.Marshal(entry.description)
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

// legend maps each level number to its label, the shape of [Score.Legend].
func (q question) legend() map[string]any {
	legend := make(map[string]any, len(q.labels))
	for i, label := range q.labels {
		legend[strconv.Itoa(i)] = label
	}
	return legend
}

// compileQuestions reads the questions an output schema encodes: one per
// property, each marked with the x-typesafe keyword the answer types emit.
// A property without the marker is rejected, since the model has no way to
// answer a shape it does not know. The preamble, the request's system text,
// goes in front of every question's own instructions, followed by the
// schema's own description when it has one, as it is for the enum format.
func compileQuestions(schema map[string]any, preamble string) (map[string]question, error) {
	props, _ := schema["properties"].(map[string]any)
	if len(props) == 0 {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: the output schema has no properties; each question is a field of the output type")
	}
	description, _ := schema["description"].(string)
	preamble = joinInstructions(preamble, description)
	questions := make(map[string]question, len(props))
	for id, raw := range props {
		prop, _ := raw.(map[string]any)
		kind, _ := prop[kindKeyword].(string)
		if kind == "" {
			return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: field %q is not a question; use a Choice, Score, or Noul field", id)
		}
		instructions, err := questionInstructions(id, prop, preamble)
		if err != nil {
			return nil, err
		}
		q := question{Type: kind, Instructions: instructions}
		switch kind {
		case kindChoice:
			q.Criteria, err = choiceCriteria(id, prop)
		case kindScore:
			q.Criteria, q.labels, err = scoreCriteria(id, prop)
		case kindNoul:
			// Assigned only when present: a typed nil map inside the any
			// would marshal as null, which the gateways reject, where an
			// absent field is the documented way to give no criteria.
			var criteria map[string]any
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

// questionInstructions reads a question's instructions: the structured
// value a runtime question carries on the x-instructions keyword, or the
// property's description. The preamble goes in front: as a paragraph of
// the text, as the first element of an array, or as the first element
// beside any other structured value.
func questionInstructions(id string, prop map[string]any, preamble string) (any, error) {
	if structured := prop[instructionsKeyword]; structured != nil {
		if preamble == "" {
			return structured, nil
		}
		if items, ok := structured.([]any); ok {
			return append([]any{preamble}, items...), nil
		}
		return []any{preamble, structured}, nil
	}
	text, _ := prop["description"].(string)
	if text == "" {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: question %q has no instructions; set jsonschema_description on the field, or Instructions on a runtime question", id)
	}
	return joinInstructions(preamble, text), nil
}

// choiceCriteria reads the options back out of the oneOf a [Choice] emits,
// in order, with the guidance of a [GuidedOption] in place of an option's
// string. An option with neither is described by its own name, since the
// wire format takes a description per option and some gateways reject a
// null one.
func choiceCriteria(id string, prop map[string]any) (criteria, error) {
	props, _ := prop["properties"].(map[string]any)
	choice, _ := props["choice"].(map[string]any)
	oneOf, _ := choice["oneOf"].([]any)
	if len(oneOf) == 0 {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: choice question %q lists no options", id)
	}
	guidance, _ := prop[guidanceKeyword].(map[string]any)
	options := make(criteria, 0, len(oneOf))
	seen := make(map[string]bool, len(oneOf))
	for _, raw := range oneOf {
		option, _ := raw.(map[string]any)
		key, _ := option["const"].(string)
		if key == "" {
			return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: choice question %q has an option without a const", id)
		}
		if seen[key] {
			return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: choice question %q lists option %q twice", id, key)
		}
		seen[key] = true
		description, _ := option["description"].(string)
		value := withWhat(guidance[key], description)
		if value == "" {
			value = key
		}
		options = append(options, criterion{option: key, description: value})
	}
	for _, key := range slices.Sorted(maps.Keys(guidance)) {
		if !seen[key] {
			return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: choice question %q has guidance for %q, which is not one of its options", id, key)
		}
	}
	return options, nil
}

// scoreCriteria reads the ordered levels off the x-levels keyword, with
// the guidance of a [GuidedRubric] in place of a level's string, and
// returns the strings too for the legend. The API takes two to ten levels;
// the lower bound is checked here because one level is not a scale, the
// upper bound is the API's to enforce.
func scoreCriteria(id string, prop map[string]any) ([]any, []string, error) {
	levels, ok := stringList(prop[levelsKeyword])
	if !ok {
		return nil, nil, status.Errorf(status.ErrInvalidSchema, "typesafe: score question %q has a level that is not a string", id)
	}
	if len(levels) < 2 {
		return nil, nil, status.Errorf(status.ErrInvalidSchema, "typesafe: score question %q needs at least two levels", id)
	}
	criteria := make([]any, len(levels))
	for i, level := range levels {
		criteria[i] = level
	}
	guidance, _ := prop[guidanceKeyword].(map[string]any)
	for _, key := range slices.Sorted(maps.Keys(guidance)) {
		i, err := strconv.Atoi(key)
		if err != nil || i < 0 || i >= len(levels) {
			return nil, nil, status.Errorf(status.ErrInvalidSchema, "typesafe: score question %q has guidance for level %s, which its rubric does not have", id, key)
		}
		criteria[i] = withWhat(guidance[key], levels[i])
	}
	return criteria, levels, nil
}

// noulCriteria reads the optional true and false criteria a [NoulOf]
// emits, with the guidance of a [GuidedYesNo] in place of a side's
// string. They are sent as a pair or not at all: the API describes the
// pair, and one side alone would leave the other implied.
func noulCriteria(id string, prop map[string]any) (map[string]any, error) {
	guidance, _ := prop[guidanceKeyword].(map[string]any)
	criteria := make(map[string]any, 2)
	for side, keyword := range map[string]string{"true": trueKeyword, "false": falseKeyword} {
		text, _ := prop[keyword].(string)
		if value := withWhat(guidance[side], text); value != "" {
			criteria[side] = value
		}
	}
	switch len(criteria) {
	case 0:
		return nil, nil
	case 1:
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: noul question %q says what only one side means; give both yes and no, in Criteria or Guidance", id)
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

// enumQuestionID names the one question an enum-format request asks.
const enumQuestionID = "choice"

// enumQuestion turns an enum output schema into a single choice question,
// so the built-in enum format works on the model with no decision type.
// The instructions are the preamble, the request's system text, followed
// by the schema's description when it has one; the options keep the order
// the values were given in, and an option's description is its own name,
// since an enum carries none.
func enumQuestion(schema map[string]any, preamble string) (map[string]question, error) {
	values, ok := stringList(schema["enum"])
	if !ok || slices.Contains(values, "") {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: enum output values must be non-empty strings")
	}
	if len(values) == 0 {
		return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: the enum output schema lists no values")
	}
	options := make(criteria, 0, len(values))
	for _, v := range values {
		if slices.ContainsFunc(options, func(c criterion) bool { return c.option == v }) {
			return nil, status.Errorf(status.ErrInvalidSchema, "typesafe: the enum output schema lists %q twice", v)
		}
		options = append(options, criterion{option: v, description: v})
	}
	description, _ := schema["description"].(string)
	instructions := joinInstructions(preamble, description)
	if instructions == "" {
		instructions = "Choose the option that best describes the state."
	}
	return map[string]question{
		enumQuestionID: {Type: kindChoice, Instructions: instructions, Criteria: options},
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
