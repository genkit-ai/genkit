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

// A model-free telemetry testapp: a Go port of js/testapps/flow-sample1. It
// exercises the tracing paths (nested steps, streaming, errors, caught errors)
// without any model calls, so the traces it writes to .genkit/traces are a
// stable fixture for verifying the instrumentation refactor did not change the
// exported trace shape.
//
// Run it under the Dev UI and call the flows from the browser at
// http://localhost:4000, then inspect .genkit/traces:
//
//	genkit start -- go run .
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/server"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// basic: two sequential steps, output of the first feeds the second.
	genkit.DefineFlow(g, "basic", func(ctx context.Context, subject string) (string, error) {
		foo, err := genkit.Run(ctx, "call-llm", func() (string, error) {
			return "subject: " + subject, nil
		})
		if err != nil {
			return "", err
		}
		return genkit.Run(ctx, "call-llm1", func() (string, error) {
			return "foo: " + foo, nil
		})
	})

	// parent: calls another flow's function inline.
	genkit.DefineFlow(g, "parent", func(ctx context.Context, _ struct{}) (string, error) {
		foo, err := genkit.Run(ctx, "call-llm", func() (string, error) {
			return "subject: foo", nil
		})
		if err != nil {
			return "", err
		}
		return genkit.Run(ctx, "call-llm1", func() (string, error) {
			return "foo: " + foo, nil
		})
	})

	// multiSteps: several steps of varying output shapes.
	genkit.DefineFlow(g, "multiSteps", func(ctx context.Context, input string) (int, error) {
		out1, err := genkit.Run(ctx, "step1", func() (string, error) {
			return fmt.Sprintf("Hello, %s! step 1", input), nil
		})
		if err != nil {
			return 0, err
		}
		out2, err := genkit.Run(ctx, "step2", func() (string, error) {
			return out1 + " Faf ", nil
		})
		if err != nil {
			return 0, err
		}
		out3, err := genkit.Run(ctx, "step3-array", func() ([]string, error) {
			return []string{out2, out2}, nil
		})
		if err != nil {
			return 0, err
		}
		if _, err := genkit.Run(ctx, "step4-num", func() (string, error) {
			return strings.Join(out3, "-()-"), nil
		}); err != nil {
			return 0, err
		}
		return 42, nil
	})

	// throwy: fails partway, so a step succeeds and the flow span records the error.
	genkit.DefineFlow(g, "throwy", func(ctx context.Context, subject string) (string, error) {
		foo, err := genkit.Run(ctx, "call-llm", func() (string, error) {
			return "subject: " + subject, nil
		})
		if err != nil {
			return "", err
		}
		if subject != "" {
			return "", errors.New(subject)
		}
		return genkit.Run(ctx, "call-llm", func() (string, error) {
			return "foo: " + foo, nil
		})
	})

	// throwy2: the failure originates inside the step, so the step span is the
	// failure source and the error propagates through the flow span.
	genkit.DefineFlow(g, "throwy2", func(ctx context.Context, subject string) (string, error) {
		foo, err := genkit.Run(ctx, "call-llm", func() (string, error) {
			if subject != "" {
				return "", errors.New(subject)
			}
			return "subject: " + subject, nil
		})
		if err != nil {
			return "", err
		}
		return genkit.Run(ctx, "call-llm", func() (string, error) {
			return "foo: " + foo, nil
		})
	})

	// flowMultiStepCaughtError: a step fails but the flow recovers, so only the
	// failing step span carries the error while the flow span succeeds.
	genkit.DefineFlow(g, "flowMultiStepCaughtError", func(ctx context.Context, input string) (string, error) {
		i := 1
		result1, err := genkit.Run(ctx, "step1", func() (string, error) {
			r := fmt.Sprintf("%s %d,", input, i)
			i++
			return r, nil
		})
		if err != nil {
			return "", err
		}

		result2, err := genkit.Run(ctx, "step2", func() (string, error) {
			if result1 != "" {
				return "", errors.New("Got an error!")
			}
			r := fmt.Sprintf("%s %d,", result1, i)
			i++
			return r, nil
		})
		if err != nil {
			// Swallow the step error and keep going, matching the JS sample.
			result2 = ""
		}

		return genkit.Run(ctx, "step3", func() (string, error) {
			return fmt.Sprintf("%s %d", result2, i), nil
		})
	})

	// streamy: streams a running count, then returns a summary.
	genkit.DefineStreamingFlow(g, "streamy",
		func(ctx context.Context, count int, stream core.StreamCallback[int]) (string, error) {
			i := 0
			for ; i < count; i++ {
				time.Sleep(200 * time.Millisecond)
				if stream != nil {
					if err := stream(ctx, i); err != nil {
						return "", err
					}
				}
			}
			return fmt.Sprintf("done: %d, streamed: %d times", count, i), nil
		})

	// streamyThrowy: streams a few chunks, then fails.
	genkit.DefineStreamingFlow(g, "streamyThrowy",
		func(ctx context.Context, count int, stream core.StreamCallback[int]) (string, error) {
			i := 0
			for ; i < count; i++ {
				if i == 3 {
					return "", errors.New("whoops")
				}
				time.Sleep(200 * time.Millisecond)
				if stream != nil {
					if err := stream(ctx, i); err != nil {
						return "", err
					}
				}
			}
			return fmt.Sprintf("done: %d, streamed: %d times", count, i), nil
		})

	mux := http.NewServeMux()
	for _, flow := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+flow.Name(), genkit.Handler(flow))
	}
	fmt.Println("Starting server on http://127.0.0.1:8080 ...")
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
