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

// Package exp serves decision models, the models that speak the System One
// protocol, and holds the question types a decision is declared with. A
// decision model does not generate text. It evaluates a state against typed
// questions and returns one typed answer per question, with calibrated
// probabilities.
//
// The plugin serves such a model as one that speaks only constrained JSON,
// which is the subset of the generate API it fits exactly. The questions
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
//		Department systemonex.Choice[Dept] `json:"department" jsonschema_description:"Which team should handle this?"`
//		IsUrgent   systemonex.Noul         `json:"is_urgent"  jsonschema_description:"Does the ticket explicitly communicate time pressure?"`
//	}
//
//	g := genkit.Init(ctx, genkit.WithPlugins(systemonex.TypeSafe()))
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
// format also takes, such as examples per option. Questions whose options
// or instructions come from data are built at run time with [Schema] and
// answered as a map of [Answer]. [ResponseInfo] reads the model version
// that answered and the raw answers off a response.
//
// The types are the same for every model that speaks the protocol, so one
// decision type works with each of them. [SystemOne] is the plugin that
// serves the models, from any server that speaks the protocol; [TypeSafe]
// and the other constructors set it up for the known ones.
//
// This package is a preview: its API may change in any minor release.
package exp

import (
	"cmp"
	"encoding/json"
	"maps"
	"math"
	"slices"
	"strconv"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal/systemone"
	"github.com/invopop/jsonschema"
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
// oneOf of constants, marked as a choice question.
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
	s := answerSchema(systemone.KindChoice, &jsonschema.Schema{Properties: props}, "choice")
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

// JSONSchema encodes the question, marked as a noul question, with C's
// criteria on the x-true and x-false keywords when it has any.
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
	s := answerSchema(systemone.KindNoul, &jsonschema.Schema{Properties: props}, "noul")
	if yes != "" || no != "" {
		s.Extras[systemone.TrueKeyword] = yes
		s.Extras[systemone.FalseKeyword] = no
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
	s := answerSchema(systemone.KindScore, &jsonschema.Schema{Properties: props}, "score")
	s.Extras[systemone.LevelsKeyword] = levels
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
		s.Extras[systemone.InstructionsKeyword] = v
	}
	return s
}

// Schema is the output schema that asks the given questions, each answered
// under its key. Pass it with [ai.WithOutputSchema] and read the answers as
// a map of [Answer]:
//
//	options := make([]systemonex.ChoiceOption, 0, len(tools))
//	for _, tool := range tools {
//		options = append(options, systemonex.ChoiceOption{Name: tool.Name, Criteria: tool.Description})
//	}
//	answers, _, err := genkit.GenerateData[map[string]systemonex.Answer](ctx, g,
//		ai.WithModelName("typesafe/jev-1.13.0"),
//		ai.WithOutputSchema(systemonex.Schema(map[string]systemonex.Question{
//			"tool": systemonex.ChoiceQuestion{Instructions: "Which tool serves the request?", Options: options},
//		})),
//		ai.WithPrompt(request))
//	if err != nil {
//		return err
//	}
//	tool := (*answers)["tool"].Choice
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
		for _, kind := range []string{systemone.KindChoice, systemone.KindScore, systemone.KindNoul} {
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
	case systemone.KindChoice:
		out["choice"], out["confidence"] = a.Choice, a.Confidence
		if a.Probabilities != nil {
			out["probabilities"] = a.Probabilities
		}
	case systemone.KindScore:
		out["score"], out["confidence"] = a.Score, a.Confidence
		if a.Probabilities != nil {
			out["probabilities"] = a.Probabilities
		}
		if a.Legend != nil {
			out["legend"] = a.Legend
		}
	case systemone.KindNoul:
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
		s.Extras[systemone.GuidanceKeyword] = wire
	}
}

// answerSchema completes the object schema shared by the answer types. It
// is closed: the model projects a wire answer onto exactly these fields
// before the generate loop validates it.
func answerSchema(kind string, s *jsonschema.Schema, required ...string) *jsonschema.Schema {
	s.Type = "object"
	s.Required = required
	s.AdditionalProperties = jsonschema.FalseSchema
	s.Extras = map[string]any{systemone.KindKeyword: kind}
	return s
}

// Info is what a decision model's response carries beside the answers in
// the message: the model version that answered, which is what confidence
// thresholds are tuned against, a gateway's provider and generation ID,
// and the answers as the API sent them, guidance echoes and fields the
// answer types do not declare included. [ResponseInfo] reads it.
type Info = systemone.Info

// ResponseInfo reads the [Info] of a decision model's response, from the
// model or from JSON that carried one, such as a flow's output or a stored
// trace. It is the zero Info for a response from any other model.
func ResponseInfo(resp *ai.ModelResponse) Info {
	if resp == nil {
		return Info{}
	}
	info, _ := base.ConvertTo[Info](resp.Raw)
	return info
}
