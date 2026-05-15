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

// This program can be manually tested like so:
// Start the server listening on port 3100:
//
//	go run . &
//
// Tell it to run a flow:
//
//	curl -d '{"key":"/flow/parent/parent", "input":{"start": {"input":null}}}' http://localhost:3100/api/runAction
//
// In production mode (GENKIT_ENV missing or set to "prod"):
// Start the server listening on port 3400:
//
//	go run . &
//
// Tell it to run a flow:
//
// curl -d '{}' http://localhost:3400/parent

package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/server"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	basic := genkit.DefineFlow(g, "basic", func(ctx context.Context, subject string) (string, error) {
		foo, err := genkit.Run(ctx, "call-llm", func() (string, error) { return "subject: " + subject, nil })
		if err != nil {
			return "", err
		}
		return genkit.Run(ctx, "call-llm", func() (string, error) { return "foo: " + foo, nil })
	})

	genkit.DefineFlow(g, "parent", func(ctx context.Context, _ any) (string, error) {
		return basic.Run(ctx, "foo")
	})

	type complex struct {
		Key   string `json:"key"`
		Value int    `json:"value"`
	}

	genkit.DefineFlow(g, "complex", func(ctx context.Context, c complex) (string, error) {
		foo, err := core.Run(ctx, "call-llm", func() (string, error) { return c.Key + ": " + strconv.Itoa(c.Value), nil })
		if err != nil {
			return "", err
		}
		return foo, nil
	})

	genkit.DefineFlow(g, "throwy", func(ctx context.Context, err string) (string, error) {
		return "", errors.New(err)
	})

	type chunk struct {
		Count int `json:"count"`
	}

	genkit.DefineStreamingFlow(g, "streamy", func(ctx context.Context, count int, cb func(context.Context, chunk) error) (string, error) {
		i := 0
		if cb != nil {
			for ; i < count; i++ {
				if err := cb(ctx, chunk{i}); err != nil {
					return "", err
				}
			}
		}
		return fmt.Sprintf("done: %d, streamed: %d times", count, i), nil
	})

	genkit.DefineStreamingFlow(g, "streamyThrowy", func(ctx context.Context, count int, cb func(context.Context, chunk) error) (string, error) {
		i := 0
		if cb != nil {
			for ; i < count; i++ {
				if i == 3 {
					return "", errors.New("boom!")
				}
				if err := cb(ctx, chunk{i}); err != nil {
					return "", err
				}
			}
		}
		return fmt.Sprintf("done: %d, streamed: %d times", count, i), nil
	})

	// Long-running flow for testing early trace ID transmission and cancellation.
	// Each step takes ~5 seconds with nested sub-steps.
	//
	// Test with:
	//   curl -d '{"key":"/flow/longRunning/longRunning", "input":{"start": {"input":3}}}' \
	//        http://localhost:3100/api/runAction?stream=true
	//
	// To test cancellation, note the X-Genkit-Trace-Id header and call:
	//   curl -d '{"traceId":"<trace-id>"}' http://localhost:3100/api/cancelAction
	type stepResult struct {
		Step      int    `json:"step"`
		Timestamp string `json:"timestamp"`
		Elapsed   int64  `json:"elapsed_ms"`
	}

	type longRunningResult struct {
		TotalDuration  int64        `json:"total_duration_ms"`
		StepsCompleted int          `json:"steps_completed"`
		Timeline       []stepResult `json:"timeline"`
	}

	genkit.DefineStreamingFlow(g, "longRunning",
		func(ctx context.Context, steps int, cb func(context.Context, stepResult) error) (longRunningResult, error) {
			if steps <= 0 {
				steps = 3
			}
			startTime := time.Now()
			timeline := make([]stepResult, 0, steps)

			log.Printf("🚀 Starting long-running flow: %d steps × 5s = ~%ds", steps, steps*5)

			for i := 1; i <= steps; i++ {
				stepStart := time.Now()

				// Check for cancellation before each step
				select {
				case <-ctx.Done():
					log.Printf("❌ Cancelled at step %d/%d", i, steps)
					return longRunningResult{
						TotalDuration:  time.Since(startTime).Milliseconds(),
						StepsCompleted: i - 1,
						Timeline:       timeline,
					}, ctx.Err()
				default:
				}

				log.Printf("[%s] 🔄 Step %d/%d starting...", time.Now().Format(time.RFC3339), i, steps)

				// Nested sub-steps (like the TS version)
				_, err := core.Run(ctx, fmt.Sprintf("step-%d-fetch", i), func() (string, error) {
					select {
					case <-ctx.Done():
						return "", ctx.Err()
					case <-time.After(1500 * time.Millisecond):
					}
					log.Printf("  📡 Fetched data for step %d", i)
					return fmt.Sprintf("fetch-%d", i), nil
				})
				if err != nil {
					return longRunningResult{TotalDuration: time.Since(startTime).Milliseconds(), StepsCompleted: i - 1, Timeline: timeline}, err
				}

				_, err = core.Run(ctx, fmt.Sprintf("step-%d-process", i), func() (string, error) {
					select {
					case <-ctx.Done():
						return "", ctx.Err()
					case <-time.After(1500 * time.Millisecond):
					}
					log.Printf("  ⚙️  Processed data for step %d", i)
					return fmt.Sprintf("process-%d", i), nil
				})
				if err != nil {
					return longRunningResult{TotalDuration: time.Since(startTime).Milliseconds(), StepsCompleted: i - 1, Timeline: timeline}, err
				}

				_, err = core.Run(ctx, fmt.Sprintf("step-%d-save", i), func() (string, error) {
					select {
					case <-ctx.Done():
						return "", ctx.Err()
					case <-time.After(1500 * time.Millisecond):
					}
					log.Printf("  💾 Saved results for step %d", i)
					return fmt.Sprintf("save-%d", i), nil
				})
				if err != nil {
					return longRunningResult{TotalDuration: time.Since(startTime).Milliseconds(), StepsCompleted: i - 1, Timeline: timeline}, err
				}

				elapsed := time.Since(stepStart).Milliseconds()
				log.Printf("[%s] ✅ Step %d/%d completed (%dms)", time.Now().Format(time.RFC3339), i, steps, elapsed)

				result := stepResult{
					Step:      i,
					Timestamp: time.Now().Format(time.RFC3339),
					Elapsed:   elapsed,
				}
				timeline = append(timeline, result)

				// Stream progress if callback provided
				if cb != nil {
					if err := cb(ctx, result); err != nil {
						return longRunningResult{}, err
					}
				}
			}

			totalDuration := time.Since(startTime).Milliseconds()
			log.Printf("🎉 Long-running flow completed in %dms", totalDuration)

			return longRunningResult{
				TotalDuration:  totalDuration,
				StepsCompleted: steps,
				Timeline:       timeline,
			}, nil
		})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
