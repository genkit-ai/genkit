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
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
)

// Observer reports the calls made under a context, typed. It is a
// [tracing.Instrumentation]: add it to a context with
// [tracing.WithInstrumentation], and it sees every call under that context,
// whoever makes it: a generate loop, a nested generate in a tool, a subagent,
// a judge, a fallback model a hook calls directly. That is what sets it apart
// from middleware, whose hooks wrap the calls of one Generate and can change
// them; an observer can only watch.
//
// Each field is optional, and a nil field costs nothing. New events are added
// as new fields.
//
// Hooks run synchronously on the goroutine that made the call, before the
// caller continues, and may run concurrently with themselves, so state they
// share must be guarded. They must not modify what they are given. A hook
// that calls a model must not pass its ctx on, or it observes its own call.
// Work an agent continues detached keeps reporting after the caller returns.
//
// Middleware that observes the calls of its Generate adds the observer in
// WrapGenerate's first iteration, whose context every later iteration runs
// under:
//
//	WrapGenerate: func(ctx context.Context, p *ai.GenerateParams, next ai.GenerateNext) (*ai.ModelResponse, error) {
//		if p.Iteration == 0 {
//			ctx = tracing.WithInstrumentation(ctx, aix.Observer{
//				ModelDone: func(_ context.Context, call *aix.ModelCall) {
//					if u := call.Response.Usage; u != nil {
//						spent.Add(int64(u.TotalTokens))
//					}
//				},
//			})
//		}
//		return next(ctx, p)
//	},
type Observer struct {
	// ModelDone is called after each model call ends, failed calls
	// included, with the context the model was called with. A response that
	// a WrapModel hook supplies without calling the model, such as a cache
	// hit, is not a model call and is not reported.
	ModelDone func(ctx context.Context, call *ModelCall)
}

// ModelCall describes one model call. New fields may be added.
type ModelCall struct {
	// Model is the name of the model action that ran, such as
	// "googleai/gemini-flash-latest".
	Model string
	// Response is the model's response, never nil. Its Request is the request
	// the model received. A call that failed without a response gets the
	// record a failed Generate returns: FinishReason failed or aborted, and
	// Error.
	Response *ai.ModelResponse
	// Err is the error the call failed with, nil on success. Response.Error
	// is its wire form.
	Err error
}

// StartSpan implements [tracing.Instrumentation]. A span that is not a model
// call gets no [tracing.Span], which costs nothing.
func (o Observer) StartSpan(ctx context.Context, info *tracing.SpanInfo) (context.Context, tracing.Span) {
	if o.ModelDone == nil || info.Type() != "action" || info.Subtype() != string(api.ActionTypeModel) {
		return ctx, nil
	}
	req, _ := info.Input().(*ai.ModelRequest)
	return ctx, &modelSpan{ctx: ctx, model: info.Name(), req: req, done: o.ModelDone}
}

// modelSpan reports one model call when it ends.
type modelSpan struct {
	ctx   context.Context
	model string
	req   *ai.ModelRequest
	done  func(context.Context, *ModelCall)
}

func (*modelSpan) TraceInfo() tracing.TraceInfo { return tracing.TraceInfo{} }

func (*modelSpan) SetMetadata(map[string]string) {}

func (s *modelSpan) End(res *tracing.SpanResult) {
	resp, _ := res.Output().(*ai.ModelResponse)
	if resp == nil {
		err := res.Err()
		if err == nil {
			resp = &ai.ModelResponse{Request: s.req}
		} else {
			resp, _ = base.FailedModelResponse(s.ctx, s.req, err).(*ai.ModelResponse)
		}
	}
	s.done(s.ctx, &ModelCall{Model: s.model, Response: resp, Err: res.Err()})
}
