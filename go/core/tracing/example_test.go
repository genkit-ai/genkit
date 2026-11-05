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

package tracing_test

import (
	"context"
	"errors"
	"fmt"

	"github.com/firebase/genkit/go/core/tracing"
)

// PrintingInstrumentation prints each span as it starts and ends.
type PrintingInstrumentation struct{}

func (PrintingInstrumentation) StartSpan(ctx context.Context, info *tracing.SpanInfo) (context.Context, tracing.Span) {
	fmt.Printf("-> %s (%s) at %s\n", info.Name(), info.Type(), info.Path())
	return ctx, printingSpan{name: info.Name()}
}

// printingSpan mints no ids, so it reports empty ones; Genkit then mints local
// ids for correlation.
type printingSpan struct{ name string }

func (printingSpan) TraceInfo() tracing.TraceInfo  { return tracing.TraceInfo{} }
func (printingSpan) SetMetadata(map[string]string) {}

func (s printingSpan) End(res *tracing.SpanResult) {
	if err := res.Err(); err != nil {
		fmt.Printf("!! %s: %v\n", s.name, err)
		return
	}
	fmt.Printf("<- %s: %v\n", s.name, res.Output())
}

func ExampleSetInstrumentation() {
	// Replace the OpenTelemetry default. To keep it, list it too:
	// tracing.SetInstrumentation(&tracing.OTelInstrumentation{}, PrintingInstrumentation{})
	tracing.SetInstrumentation(PrintingInstrumentation{})
	defer tracing.SetInstrumentation() // restore the default

	ctx := context.Background()
	_, _ = tracing.RunInNewSpan(ctx, &tracing.SpanMetadata{Name: "greet", Type: "flow"}, "world",
		func(ctx context.Context, name string) (string, error) {
			_, _ = tracing.RunInNewSpan(ctx, &tracing.SpanMetadata{Name: "lookup", Type: "flowStep"}, name,
				func(context.Context, string) (string, error) { return "", errors.New("not found") })
			return "hello " + name, nil
		})
	// Output:
	// -> greet (flow) at /{greet,t:flow}
	// -> lookup (flowStep) at /{greet,t:flow}/{lookup,t:flowStep}
	// !! lookup: not found
	// <- greet: hello world
}
