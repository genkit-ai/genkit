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

// Package x provides experimental Genkit APIs.
//
// APIs in this package are under active development and may change in any
// minor version release. Use with caution in production environments.
//
// When these APIs stabilize, they will be moved to the genkit package
// and these exports will be deprecated.
package x

import (
	"context"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

// StreamingFunc is a streaming function that uses a channel instead of a callback.
//
// The function receives a send-only channel to which it should write stream chunks.
// The channel is managed by the framework and will be closed automatically after
// the function returns. The function should NOT close the channel itself.
//
// When writing to the channel, the function should respect context cancellation:
//
//	select {
//	case streamCh <- chunk:
//	case <-ctx.Done():
//	    return zero, ctx.Err()
//	}
type StreamingFunc[In, Out, Stream any] = func(ctx context.Context, input In, streamCh chan<- Stream) (Out, error)

// DefineStreamingFlow defines a streaming flow that uses a channel for streaming,
// registers it as a [core.Action] of type Flow, and returns a [core.Flow] runner.
//
// Unlike [genkit.DefineStreamingFlow] which uses a callback, this function accepts
// a [StreamingFunc] that writes stream chunks to a channel. This can be
// more ergonomic when integrating with other channel-based APIs or when the
// streaming logic is more naturally expressed with channels.
//
// The channel passed to the function is unbuffered and managed by the framework.
// The function should NOT close the channel - it will be closed automatically
// after the function returns.
//
// Example:
//
//	countdown := x.DefineStreamingFlow(g, "countdown",
//	    func(ctx context.Context, start int, streamCh chan<- int) (string, error) {
//	        for i := start; i > 0; i-- {
//	            select {
//	            case streamCh <- i:
//	            case <-ctx.Done():
//	                return "", ctx.Err()
//	            }
//	        }
//	        return "liftoff!", nil
//	    })
//
//	// Run with streaming
//	for val, err := range countdown.Stream(ctx, 5) {
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    if val.Done {
//	        fmt.Println(val.Output)  // "liftoff!"
//	    } else {
//	        fmt.Println(val.Stream)  // 5, 4, 3, 2, 1
//	    }
//	}
func DefineStreamingFlow[In, Out, Stream any](g *genkit.Genkit, name string, fn StreamingFunc[In, Out, Stream]) *core.Flow[In, Out, Stream] {
	// Wrap the channel-based function to work with the callback-based API
	wrappedFn := func(ctx context.Context, input In, sendChunk core.StreamCallback[Stream]) (Out, error) {
		if sendChunk == nil {
			// Create a channel that discards all values
			discardCh := make(chan Stream)
			go func() {
				for range discardCh {
				}
			}()
			defer close(discardCh)
			return fn(ctx, input, discardCh)
		}

		// Create a cancellable context for the user function.
		// We cancel this if the callback returns an error, signaling
		// the user's function to stop producing chunks.
		fnCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		streamCh := make(chan Stream)

		type result struct {
			output Out
			err    error
		}
		resultCh := make(chan result, 1)

		go func() {
			output, err := fn(fnCtx, input, streamCh)
			close(streamCh)
			resultCh <- result{output, err}
		}()

		// Forward chunks from the channel to the callback.
		// If callback returns an error, cancel context and drain remaining
		// chunks to prevent the goroutine from blocking.
		var callbackErr error
		for chunk := range streamCh {
			if callbackErr != nil {
				continue
			}
			if err := sendChunk(ctx, chunk); err != nil {
				callbackErr = err
				cancel()
			}
		}

		res := <-resultCh
		if callbackErr != nil {
			var zero Out
			return zero, callbackErr
		}
		return res.output, res.err
	}

	return genkit.DefineStreamingFlow(g, name, wrappedFn)
}
