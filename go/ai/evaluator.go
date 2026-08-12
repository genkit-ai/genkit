// Copyright 2025 Google LLC
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

package ai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/core/tracing"
)

// EvaluatorFunc is the function type for evaluator implementations.
type EvaluatorFunc = func(context.Context, *EvaluatorCallbackRequest) (*EvaluatorCallbackResponse, error)

// EvaluatorActionFunc is an [EvaluatorFunc] that additionally receives
// the request's typed Config: the framework deserializes the request's raw
// options into it before calling the function (see [NewEvaluatorAction]).
type EvaluatorActionFunc[Config any] = func(context.Context, *EvaluatorCallbackRequest, Config) (*EvaluatorCallbackResponse, error)

// BatchEvaluatorFunc is the function type for batch evaluator implementations.
type BatchEvaluatorFunc = func(context.Context, *EvaluatorRequest) (*EvaluatorResponse, error)

// BatchEvaluatorActionFunc is a [BatchEvaluatorFunc] that additionally
// receives the request's typed Config: the framework deserializes the
// request's raw options into it before calling the function (see
// [NewBatchEvaluatorAction]).
type BatchEvaluatorActionFunc[Config any] = func(context.Context, *EvaluatorRequest, Config) (*EvaluatorResponse, error)

// Evaluator represents an evaluator. It is the type to accept as an argument
// and to look up by name; implementations are created with
// [NewEvaluatorAction] or [NewBatchEvaluatorAction], or their
// [genkit.DefineEvaluatorAction] and [genkit.DefineBatchEvaluatorAction]
// counterparts in an application.
type Evaluator interface {
	// Name returns the name of the evaluator.
	Name() string
	// Evaluates a dataset.
	Evaluate(ctx context.Context, req *EvaluatorRequest) (*EvaluatorResponse, error)
	// Register registers the evaluator with the given registry.
	Register(r api.Registry)
}

// EvaluatorArg is the interface for evaluator arguments. It can either be the evaluator action itself or a reference to be looked up.
type EvaluatorArg interface {
	Name() string
}

// EvaluatorRef is a struct to hold evaluator name and configuration.
type EvaluatorRef struct {
	name   string
	config any
}

// NewEvaluatorRef creates a new EvaluatorRef with the given name and configuration.
func NewEvaluatorRef(name string, config any) EvaluatorRef {
	return EvaluatorRef{name: name, config: config}
}

// Name returns the name of the evaluator.
func (e EvaluatorRef) Name() string {
	return e.name
}

// Config returns the configuration to use by default for this evaluator.
func (e EvaluatorRef) Config() any {
	return e.config
}

// EvaluatorAction is an evaluator backed by a registry action. It is the
// concrete type returned by [NewEvaluatorAction] and [NewBatchEvaluatorAction];
// pass it to [WithEvaluator], or return it from a plugin's Init for the
// framework to register.
//
// It implements [Evaluator] and [api.Action], so it can be passed anywhere
// either is accepted. It also promotes [core.Action.Run], the typed
// equivalent of [EvaluatorAction.Evaluate].
type EvaluatorAction struct {
	action[*EvaluatorRequest, *EvaluatorResponse, struct{}]
}

// Pinned here so that breaking either interface fails the build at the type
// rather than at a call site.
var (
	_ api.Action = (*EvaluatorAction)(nil)
	_ Evaluator  = (*EvaluatorAction)(nil)
)

// Name returns the registry name of the evaluator.
func (e *EvaluatorAction) Name() string { return e.action.Name() }

// Register registers the evaluator with r, making it available to lookups and
// to the Dev UI. A plugin that returns the evaluator from its Init does not
// need to call this.
func (e *EvaluatorAction) Register(r api.Registry) { e.action.Register(r) }

// Desc returns the evaluator's action descriptor: its name, schemas, and
// metadata.
func (e *EvaluatorAction) Desc() api.ActionDesc { return e.action.Desc() }

// RunJSON runs the evaluator on a JSON-encoded [EvaluatorRequest] and returns
// a JSON-encoded [EvaluatorResponse]. The framework uses it to serve
// reflection and registry-driven calls; prefer [EvaluatorAction.Evaluate].
func (e *EvaluatorAction) RunJSON(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (json.RawMessage, error) {
	if e == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Evaluator.RunJSON: evaluator called on a nil evaluator; check that all evaluators are defined")
	}
	return e.action.RunJSON(ctx, input, cb)
}

// RunJSONWithTelemetry is [EvaluatorAction.RunJSON] with the run's telemetry
// returned alongside the output.
func (e *EvaluatorAction) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (*api.ActionRunResult[json.RawMessage], error) {
	if e == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Evaluator.RunJSONWithTelemetry: evaluator called on a nil evaluator; check that all evaluators are defined")
	}
	return e.action.RunJSONWithTelemetry(ctx, input, cb)
}

// Example is a single example that requires evaluation
type Example struct {
	TestCaseId string   `json:"testCaseId,omitempty"`
	Input      any      `json:"input"`
	Output     any      `json:"output,omitempty"`
	Context    []any    `json:"context,omitempty"`
	Reference  any      `json:"reference,omitempty"`
	TraceIds   []string `json:"traceIds,omitempty"`
}

// EvaluatorRequest is the data we pass to evaluate a dataset.
// The Options field is specific to the actual evaluator implementation.
type EvaluatorRequest struct {
	Dataset      []*Example `json:"dataset"`
	EvaluationId string     `json:"evalRunId"`
	Options      any        `json:"options,omitempty"`
}

// ScoreStatus is an enum used to indicate if a Score has passed or failed. This
// drives additional features in tooling / the Dev UI.
type ScoreStatus int

const (
	ScoreStatusUnknown ScoreStatus = iota
	ScoreStatusFail
	ScoreStatusPass
)

var statusName = map[ScoreStatus]string{
	ScoreStatusUnknown: "UNKNOWN",
	ScoreStatusFail:    "FAIL",
	ScoreStatusPass:    "PASS",
}

// String returns the wire name of the score status ("PASS", "FAIL", or
// "UNKNOWN").
func (ss ScoreStatus) String() string {
	return statusName[ss]
}

// Score is the evaluation score that represents the result of an evaluator.
// This struct includes information such as the score (numeric, string or other
// types), the reasoning provided for this score (if any), the score status (if
// any) and other details.
type Score struct {
	Id      string         `json:"id,omitempty"`
	Score   any            `json:"score,omitempty"`
	Status  string         `json:"status,omitempty" jsonschema:"enum=UNKNOWN,enum=FAIL,enum=PASS"`
	Error   string         `json:"error,omitempty"`
	Details map[string]any `json:"details,omitempty"`
}

// EvaluationResult is the result of running the evaluator on a single Example.
// An evaluator may provide multiple scores simultaneously (e.g. if they are using
// an API to score on multiple criteria)
type EvaluationResult struct {
	TestCaseId string  `json:"testCaseId"`
	TraceID    string  `json:"traceId,omitempty"`
	SpanID     string  `json:"spanId,omitempty"`
	Evaluation []Score `json:"evaluation"`
}

// EvaluatorResponse is a collection of [EvaluationResult] structs, it
// represents the result on the entire input dataset.
type EvaluatorResponse = []EvaluationResult

// EvaluatorOptions configures an evaluator created with
// [NewEvaluatorAction] or [NewBatchEvaluatorAction].
type EvaluatorOptions struct {
	// ConfigSchema is the JSON schema for the evaluator's config.
	ConfigSchema map[string]any `json:"configSchema,omitempty"`
	// Metadata is arbitrary key-value data attached to the action descriptor.
	Metadata map[string]any `json:"-"`
	// DisplayName is the name of the evaluator as it appears in the UI.
	DisplayName string `json:"displayName"`
	// Definition is the definition of the evaluator.
	Definition string `json:"definition"`
	// IsBilled is a flag indicating if the evaluator is billed.
	IsBilled bool `json:"isBilled,omitempty"`
}

// EvaluatorCallbackRequest is the data we pass to the callback function
// provided in defineEvaluator. The Options field is specific to the actual
// evaluator implementation.
type EvaluatorCallbackRequest struct {
	Input   Example `json:"input"`
	Options any     `json:"options,omitempty"`
}

// EvaluatorCallbackResponse is the result on evaluating a single [Example]
type EvaluatorCallbackResponse = EvaluationResult

// evaluatorMetadata builds the shared action metadata for an evaluator.
func evaluatorMetadata(opts *EvaluatorOptions, configSchema map[string]any) map[string]any {
	return actionMetadata(api.ActionTypeEvaluator, map[string]any{
		"evaluator": map[string]any{
			"evaluatorIsBilled":    opts.IsBilled,
			"evaluatorDisplayName": opts.DisplayName,
			"evaluatorDefinition":  opts.Definition,
			"customOptions":        configSchema,
		},
	}, opts.Metadata)
}

// NewEvaluatorAction creates an unregistered [EvaluatorAction]: return it from
// a plugin's Init for the framework to register, or call
// [EvaluatorAction.Register] directly. Applications should define evaluators
// with [genkit.DefineEvaluatorAction].
// This method processes the input dataset one-by-one.
//
// Config is the evaluator's typed configuration; it is usually inferred from
// fn's signature. The framework deserializes the request's raw options into
// Config before calling fn: the exact Config type (or a pointer to it) and
// map[string]any (from the Dev UI and other JSON callers) are accepted, and
// mismatched types are rejected. The config's JSON schema is inferred from
// Config unless [EvaluatorOptions.ConfigSchema] overrides it.
func NewEvaluatorAction[Config any](
	name string,
	opts *EvaluatorOptions,
	fn EvaluatorActionFunc[Config],
) *EvaluatorAction {
	if name == "" {
		panic("ai.NewEvaluatorAction: name is required")
	}

	o := EvaluatorOptions{}
	if opts != nil {
		o = *opts
	}

	configSchema, inputSchema := actionConfigSchemas[Config](o.ConfigSchema, EvaluatorRequest{}, "options")

	rawFn := typedConfigFn(func(r *EvaluatorRequest) *any { return &r.Options },
		func(ctx context.Context, req *EvaluatorRequest, cfg Config) (*EvaluatorResponse, error) {
			// The callback's Options slot is type-erased like the request's,
			// so it takes the same guard: boxing a typed nil would make it
			// compare non-nil while dereferences still panic, and would split
			// this path's semantics from NewBatchEvaluatorAction's.
			cfgAny := req.Options

			var results []EvaluationResult
			for _, datapoint := range req.Dataset {
				if datapoint.TestCaseId == "" {
					datapoint.TestCaseId = uuid.New().String()
				}
				spanMetadata := &tracing.SpanMetadata{
					Name:    fmt.Sprintf("TestCase %s", datapoint.TestCaseId),
					Type:    "evaluator",
					Subtype: "evaluator",
				}
				_, err := tracing.RunInNewSpan(ctx, spanMetadata, datapoint,
					func(ctx context.Context, input *Example) (*EvaluatorCallbackResponse, error) {
						traceId := trace.SpanContextFromContext(ctx).TraceID().String()
						spanId := trace.SpanContextFromContext(ctx).SpanID().String()

						callbackRequest := EvaluatorCallbackRequest{
							Input:   *input,
							Options: cfgAny,
						}

						result, err := fn(ctx, &callbackRequest, cfg)
						if err != nil {
							failedScore := Score{
								Status: ScoreStatusFail.String(),
								Error:  fmt.Sprintf("Evaluation of test case %s failed: \n %s", input.TestCaseId, err.Error()),
							}
							failedResult := EvaluationResult{
								TestCaseId: input.TestCaseId,
								Evaluation: []Score{failedScore},
								TraceID:    traceId,
								SpanID:     spanId,
							}
							results = append(results, failedResult)
							// return error to mark span as failed
							return nil, err
						}

						result.TraceID = traceId
						result.SpanID = spanId

						results = append(results, *result)

						return result, nil
					})
				if err != nil {
					logger.FromContext(ctx).Debug("EvaluatorAction", "err", err)
					continue
				}
			}
			return &results, nil
		})

	return &EvaluatorAction{
		action: *core.NewActionOf(api.ActionTypeEvaluator, name, &core.ActionOptions{
			Metadata:    evaluatorMetadata(&o, configSchema),
			InputSchema: inputSchema,
		}, rawFn),
	}
}

// NewBatchEvaluatorAction creates an unregistered [EvaluatorAction]: return it
// from a plugin's Init for the framework to register, or call
// [EvaluatorAction.Register] directly. Applications should define batch
// evaluators with [genkit.DefineBatchEvaluatorAction].
// This method provides the full [EvaluatorRequest] to the callback function,
// giving more flexibility to the user for processing the data, such as batching or parallelization.
//
// Config is the evaluator's typed configuration; it is usually inferred from
// fn's signature. See [NewEvaluatorAction] for how the request's options
// are deserialized.
//
// [EvaluatorOptions.ConfigSchema] is enforced: it becomes the options slot of
// the action's input schema and every request is validated against it, so a
// schema narrower than what callers actually send now fails at the action
// boundary. Batch evaluators did not validate options before; leave
// ConfigSchema unset to accept anything.
func NewBatchEvaluatorAction[Config any](
	name string,
	opts *EvaluatorOptions,
	fn BatchEvaluatorActionFunc[Config],
) *EvaluatorAction {
	if name == "" {
		panic("ai.NewBatchEvaluatorAction: name is required")
	}

	o := EvaluatorOptions{}
	if opts != nil {
		o = *opts
	}

	configSchema, inputSchema := actionConfigSchemas[Config](o.ConfigSchema, EvaluatorRequest{}, "options")

	return &EvaluatorAction{
		action: *core.NewActionOf(api.ActionTypeEvaluator, name, &core.ActionOptions{
			Metadata:    evaluatorMetadata(&o, configSchema),
			InputSchema: inputSchema,
		}, typedConfigFn(func(r *EvaluatorRequest) *any { return &r.Options }, fn)),
	}
}

// NewEvaluator creates a new [Evaluator].
// This method processes the input dataset one-by-one.
//
// Deprecated: Use [NewEvaluatorAction], which passes the request's
// options to fn as a typed value instead of leaving them type-erased on the
// request.
func NewEvaluator(name string, opts *EvaluatorOptions, fn EvaluatorFunc) Evaluator {
	if name == "" {
		panic("ai.NewEvaluator: name is required")
	}
	return NewEvaluatorAction(name, opts, func(ctx context.Context, req *EvaluatorCallbackRequest, _ any) (*EvaluatorCallbackResponse, error) {
		return fn(ctx, req)
	})
}

// NewBatchEvaluator creates a new [Evaluator].
// This method provides the full [EvaluatorRequest] to the callback function,
// giving more flexibility to the user for processing the data, such as batching or parallelization.
//
// Deprecated: Use [NewBatchEvaluatorAction], which passes the request's
// options to fn as a typed value instead of leaving them type-erased on the
// request.
func NewBatchEvaluator(name string, opts *EvaluatorOptions, fn BatchEvaluatorFunc) Evaluator {
	if name == "" {
		panic("ai.NewBatchEvaluator: name is required")
	}
	return NewBatchEvaluatorAction(name, opts, func(ctx context.Context, req *EvaluatorRequest, _ any) (*EvaluatorResponse, error) {
		return fn(ctx, req)
	})
}

// LookupEvaluator looks up a registered [Evaluator] by name.
// It returns nil if the evaluator was not defined.
func LookupEvaluator(r api.Registry, name string) Evaluator {
	action := core.ResolveActionFor[*EvaluatorRequest, *EvaluatorResponse, struct{}](r, api.ActionTypeEvaluator, name)
	if action == nil {
		return nil
	}
	return &EvaluatorAction{*action}
}

// Evaluate runs the given [Evaluator].
func (e *EvaluatorAction) Evaluate(ctx context.Context, req *EvaluatorRequest) (*EvaluatorResponse, error) {
	if e == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Evaluator.Evaluate: evaluator called on a nil evaluator; check that all evaluators are defined")
	}

	return e.Run(ctx, req, nil)
}

// Evaluate calls the retrivers with provided options.
func Evaluate(ctx context.Context, r api.Registry, opts ...EvaluatorOption) (*EvaluatorResponse, error) {
	evalOpts := &evaluatorOptions{}
	for _, opt := range opts {
		opt.applyEvaluator(evalOpts)
	}

	if evalOpts.Evaluator == nil {
		return nil, fmt.Errorf("ai.Evaluate: evaluator must be set")
	}
	e, ok := evalOpts.Evaluator.(Evaluator)
	if !ok {
		e = LookupEvaluator(r, evalOpts.Evaluator.Name())
	}
	if e == nil {
		return nil, fmt.Errorf("ai.Evaluate: evaluator not found: %s", evalOpts.Evaluator.Name())
	}

	if evalRef, ok := evalOpts.Evaluator.(EvaluatorRef); ok && evalOpts.Config == nil {
		evalOpts.Config = evalRef.Config()
	}

	req := &EvaluatorRequest{
		Dataset:      evalOpts.Dataset,
		EvaluationId: evalOpts.ID,
		Options:      evalOpts.Config,
	}

	return e.Evaluate(ctx, req)
}
