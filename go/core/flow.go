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

package core

import (
	"context"
	"errors"
	"fmt"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
)

// A Flow is a user-defined Action. A Flow[In, Out, StreamOut, StreamIn] represents a function from In to Out.
// The StreamOut parameter is for flows that support streaming: providing their results incrementally. The StreamIn parameter is for bidi flows.
type Flow[In, Out, StreamOut, StreamIn any] struct {
	*Action[In, Out, StreamOut, StreamIn]
}

// StreamingFlowValue is either a streamed value or a final output of a flow.
type StreamingFlowValue[Out, StreamOut any] struct {
	Done   bool
	Output Out       // valid if Done is true
	Stream StreamOut // valid if Done is false
}

// flowContextKey is a context key that indicates whether the current context is a flow context.
var flowContextKey = base.NewContextKey[*flowContext]()

// flowContext is a context that contains flow-specific information.
type flowContext struct {
	flowName string
}

// NewFlow creates a Flow that runs fn without registering it. fn takes an input of type In and returns an output of type Out.
func NewFlow[In, Out any](name string, fn Func[In, Out]) *Flow[In, Out, struct{}, struct{}] {
	return &Flow[In, Out, struct{}, struct{}]{NewAction(name, api.ActionTypeFlow, nil, nil, func(ctx context.Context, input In) (Out, error) {
		fc := &flowContext{
			flowName: name,
		}
		ctx = flowContextKey.NewContext(ctx, fc)
		return fn(ctx, input)
	})}
}

// NewStreamingFlow creates a streaming Flow that runs fn without registering it.
func NewStreamingFlow[In, Out, StreamOut any](name string, fn StreamingFunc[In, Out, StreamOut]) *Flow[In, Out, StreamOut, struct{}] {
	return &Flow[In, Out, StreamOut, struct{}]{NewStreamingAction(name, api.ActionTypeFlow, nil, nil, func(ctx context.Context, input In, cb func(context.Context, StreamOut) error) (Out, error) {
		fc := &flowContext{
			flowName: name,
		}
		ctx = flowContextKey.NewContext(ctx, fc)
		if cb == nil {
			cb = func(context.Context, StreamOut) error { return nil }
		}
		return fn(ctx, input, cb)
	})}
}

// NewBidiFlow creates a bidirectional streaming Flow without registering it.
// Flow context is injected so that [Run] works inside the bidi function.
func NewBidiFlow[In, Out, StreamOut, StreamIn any](name string, fn BidiFunc[In, Out, StreamOut, StreamIn]) *Flow[In, Out, StreamOut, StreamIn] {
	wrapped := func(ctx context.Context, in In, inCh <-chan StreamIn, outCh chan<- StreamOut) (Out, error) {
		ctx = flowContextKey.NewContext(ctx, &flowContext{flowName: name})
		return fn(ctx, in, inCh, outCh)
	}
	return &Flow[In, Out, StreamOut, StreamIn]{NewBidiAction(name, api.ActionTypeFlow, nil, wrapped)}
}

// DefineFlow creates a Flow that runs fn, and registers it as an action. fn takes an input of type In and returns an output of type Out.
func DefineFlow[In, Out any](r api.Registry, name string, fn Func[In, Out]) *Flow[In, Out, struct{}, struct{}] {
	f := NewFlow(name, fn)
	f.Register(r)
	return f
}

// DefineStreamingFlow creates a streaming Flow that runs fn, and registers it as an action.
//
// fn takes an input of type In and returns an output of type Out, optionally
// streaming values of type StreamOut incrementally by invoking a callback.
//
// If the function supports streaming and the callback is non-nil, it should
// stream the results by invoking the callback periodically, ultimately returning
// with a final return value that includes all the streamed data.
// Otherwise, it should ignore the callback and just return a result.
func DefineStreamingFlow[In, Out, StreamOut any](r api.Registry, name string, fn StreamingFunc[In, Out, StreamOut]) *Flow[In, Out, StreamOut, struct{}] {
	f := NewStreamingFlow(name, fn)
	f.Register(r)
	return f
}

// DefineBidiFlow creates a bidirectional streaming Flow that runs fn, and registers it as an action.
// Flow context is injected so that [Run] works inside the bidi function.
func DefineBidiFlow[In, Out, StreamOut, StreamIn any](r api.Registry, name string, fn BidiFunc[In, Out, StreamOut, StreamIn]) *Flow[In, Out, StreamOut, StreamIn] {
	f := NewBidiFlow(name, fn)
	f.Register(r)
	return f
}

// Run runs the function f in the context of the current flow
// and returns what f returns.
// It returns an error if no flow is active.
//
// Each call to Run results in a new step in the flow.
// A step has its own span in the trace, and its result is cached so that if the flow
// is restarted, f will not be called a second time.
func Run[Out any](ctx context.Context, name string, fn func() (Out, error)) (Out, error) {
	fc := flowContextKey.FromContext(ctx)
	if fc == nil {
		var z Out
		return z, fmt.Errorf("flow.Run(%q): must be called from a flow", name)
	}
	spanMetadata := &tracing.SpanMetadata{
		Name:    name,
		Type:    "flowStep",
		Subtype: "flowStep",
	}
	return tracing.RunInNewSpan(ctx, spanMetadata, nil, func(ctx context.Context, _ any) (Out, error) {
		o, err := fn()
		if err != nil {
			return base.Zero[Out](), err
		}
		return o, nil
	})
}

// Run runs the flow in the context of another flow.
func (f *Flow[In, Out, StreamOut, StreamIn]) Run(ctx context.Context, input In) (Out, error) {
	return f.Action.Run(ctx, input, nil)
}

// Stream runs the flow in the context of another flow and streams the output.
// It returns a function whose argument function (the "yield function") will be repeatedly
// called with the results.
//
// If the yield function is passed a non-nil error, the flow has failed with that
// error; the yield function will not be called again.
//
// If the yield function's [StreamingFlowValue] argument has Done == true, the value's
// Output field contains the final output; the yield function will not be called
// again.
//
// Otherwise the Stream field of the passed [StreamingFlowValue] holds a streamed result.
func (f *Flow[In, Out, StreamOut, StreamIn]) Stream(ctx context.Context, input In) func(func(*StreamingFlowValue[Out, StreamOut], error) bool) {
	return func(yield func(*StreamingFlowValue[Out, StreamOut], error) bool) {
		cb := func(ctx context.Context, s StreamOut) error {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if !yield(&StreamingFlowValue[Out, StreamOut]{Stream: s}, nil) {
				return errStop
			}
			return nil
		}
		output, err := f.Action.Run(ctx, input, cb)
		if errors.Is(err, errStop) {
			// Consumer broke out of the loop; don't yield again.
			return
		}
		if err != nil {
			yield(nil, err)
		} else {
			yield(&StreamingFlowValue[Out, StreamOut]{Done: true, Output: output}, nil)
		}
	}
}

var errStop = errors.New("stop")

// FlowNameFromContext returns the flow name from context if we're in a flow, empty string otherwise.
func FlowNameFromContext(ctx context.Context) string {
	if fc := flowContextKey.FromContext(ctx); fc != nil {
		return fc.flowName
	}
	return ""
}
