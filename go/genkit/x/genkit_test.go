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

package x

import (
	"context"
	"errors"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/firebase/genkit/go/genkit"
)

func TestDefineStreamingFlow(t *testing.T) {
	t.Run("streams values via channel", func(t *testing.T) {
		ctx := context.Background()
		g := genkit.Init(ctx)

		flow := DefineStreamingFlow(g, "test/counter", func(ctx context.Context, n int, stream chan<- int) (string, error) {
			for i := 0; i < n; i++ {
				select {
				case stream <- i:
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}
			return "done", nil
		})

		var streamedValues []int
		var finalOutput string

		for v, err := range flow.Stream(ctx, 3) {
			if err != nil {
				t.Fatalf("Stream error: %v", err)
			}
			if v.Done {
				finalOutput = v.Output
			} else {
				streamedValues = append(streamedValues, v.Stream)
			}
		}

		wantStreamed := []int{0, 1, 2}
		if !slices.Equal(streamedValues, wantStreamed) {
			t.Errorf("streamed values = %v, want %v", streamedValues, wantStreamed)
		}
		if finalOutput != "done" {
			t.Errorf("final output = %q, want %q", finalOutput, "done")
		}
	})

	t.Run("runs without streaming", func(t *testing.T) {
		ctx := context.Background()
		g := genkit.Init(ctx)

		flow := DefineStreamingFlow(g, "test/nostream", func(ctx context.Context, n int, stream chan<- int) (string, error) {
			for i := 0; i < n; i++ {
				stream <- i
			}
			return "complete", nil
		})

		output, err := flow.Run(ctx, 3)
		if err != nil {
			t.Fatalf("Run error: %v", err)
		}
		if output != "complete" {
			t.Errorf("output = %q, want %q", output, "complete")
		}
	})

	t.Run("handles errors", func(t *testing.T) {
		ctx := context.Background()
		g := genkit.Init(ctx)

		expectedErr := errors.New("flow failed")
		flow := DefineStreamingFlow(g, "test/failing", func(ctx context.Context, _ int, stream chan<- int) (string, error) {
			return "", expectedErr
		})

		var gotErr error
		for _, err := range flow.Stream(ctx, 1) {
			if err != nil {
				gotErr = err
			}
		}

		if gotErr == nil {
			t.Error("expected error, got nil")
		}
	})

	t.Run("handles context cancellation", func(t *testing.T) {
		ctx := context.Background()
		g := genkit.Init(ctx)

		flow := DefineStreamingFlow(g, "test/cancel", func(ctx context.Context, n int, stream chan<- int) (int, error) {
			for i := 0; i < n; i++ {
				select {
				case stream <- i:
				case <-ctx.Done():
					return 0, ctx.Err()
				}
			}
			return n, nil
		})

		cancelCtx, cancel := context.WithCancel(ctx)
		cancel()

		var gotErr error
		for _, err := range flow.Stream(cancelCtx, 100) {
			if err != nil {
				gotErr = err
			}
		}

		if gotErr == nil {
			t.Error("expected context cancellation error, got nil")
		}
	})

	t.Run("handles empty stream", func(t *testing.T) {
		ctx := context.Background()
		g := genkit.Init(ctx)

		flow := DefineStreamingFlow(g, "test/empty", func(ctx context.Context, _ struct{}, stream chan<- int) (string, error) {
			return "empty", nil
		})

		var streamedValues []int
		var finalOutput string

		for v, err := range flow.Stream(ctx, struct{}{}) {
			if err != nil {
				t.Fatalf("Stream error: %v", err)
			}
			if v.Done {
				finalOutput = v.Output
			} else {
				streamedValues = append(streamedValues, v.Stream)
			}
		}

		if len(streamedValues) != 0 {
			t.Errorf("streamed values = %v, want empty", streamedValues)
		}
		if finalOutput != "empty" {
			t.Errorf("final output = %q, want %q", finalOutput, "empty")
		}
	})

	t.Run("handles consumer breaking early", func(t *testing.T) {
		ctx := context.Background()
		g := genkit.Init(ctx)

		var produced atomic.Int32
		flow := DefineStreamingFlow(g, "test/earlybreak", func(ctx context.Context, n int, stream chan<- int) (string, error) {
			for i := 0; i < n; i++ {
				select {
				case stream <- i:
					produced.Add(1)
				case <-ctx.Done():
					return "cancelled", ctx.Err()
				}
			}
			return "done", nil
		})

		var received []int
		done := make(chan struct{})
		go func() {
			defer close(done)
			for v, err := range flow.Stream(ctx, 1000) {
				if err != nil {
					return
				}
				if !v.Done {
					received = append(received, v.Stream)
					if len(received) >= 3 {
						break // Stop early
					}
				}
			}
		}()

		// Should complete without deadlock
		select {
		case <-done:
			// Success - no deadlock
		case <-time.After(2 * time.Second):
			t.Fatal("timeout - likely deadlock when consumer breaks early")
		}

		if len(received) != 3 {
			t.Errorf("received %d values, want 3", len(received))
		}

		// Producer should have been signaled to stop (though may have produced a few more)
		// The important thing is no deadlock occurred
		t.Logf("producer created %d chunks before stopping", produced.Load())
	})
}
