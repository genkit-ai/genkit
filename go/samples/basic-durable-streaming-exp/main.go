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

// This sample demonstrates durable streaming: a stream a caller can rejoin,
// whether it is still running or already finished.
//
// countdown streams one chunk a second. An ordinary stream is gone the moment
// the connection drops, so a caller that loses the network loses the run. A
// durable stream survives that: the manager buffers every chunk under a stream
// ID, and a caller who reconnects with that ID gets the buffered chunks
// replayed before live updates resume. Reconnect after the run has finished and
// the whole stream comes back, final result included.
//
// The stream manager lives in core/x/streaming, whose API is still in preview,
// which is what the -exp in this sample's name marks.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, to call the flow from a browser and read a trace of every
// run at http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. Ask for a stream and keep the X-Genkit-Stream-Id it answers
// with, which is why this curl passes -i:
//
//	curl -N -i -X POST http://localhost:8080/countdown \
//	  -H "Content-Type: application/json" \
//	  -H "Accept: text/event-stream" \
//	  -d '{"data": {"count": 5}}'
//
// Then rejoin that stream from anywhere, as many times as you like, by sending
// the ID back. Do it mid-countdown to see the replay, or after "Liftoff!" to
// get the finished stream whole:
//
//	curl -N -X POST http://localhost:8080/countdown \
//	  -H "Content-Type: application/json" \
//	  -H "Accept: text/event-stream" \
//	  -H "X-Genkit-Stream-Id: <id-from-above>" \
//	  -d '{"data": {"count": 5}}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/firebase/genkit/go/core/x/streaming"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/server"
)

type (
	// CountdownRequest is what the flow takes. A struct rather than a bare int
	// lets the field carry a description and a default, which the Dev UI
	// pre-fills its form from. The default is not applied in transit, and a
	// field without omitempty is required.
	CountdownRequest struct {
		Count int `json:"count" jsonschema:"default=5" jsonschema_description:"How many seconds to count down from"`
	}

	// CountdownChunk is one tick of the countdown. Chunks are what the manager
	// buffers, so this is the shape a reconnecting caller gets replayed.
	CountdownChunk struct {
		Count     int    `json:"count"`
		Message   string `json:"message"`
		Timestamp string `json:"timestamp"`
	}
)

func main() {
	ctx := context.Background()

	g := genkit.Init(ctx)

	// A slow flow, so there is time to reconnect while it is still running.
	countdown := genkit.DefineStreamingFlow(g, "countdown",
		func(ctx context.Context, input CountdownRequest, sendChunk func(context.Context, CountdownChunk) error) (string, error) {
			for i := input.Count; i > 0; i-- {
				select {
				case <-ctx.Done():
					return "", ctx.Err()
				case <-time.After(1 * time.Second):
				}

				chunk := CountdownChunk{
					Count:     i,
					Message:   fmt.Sprintf("T-%d...", i),
					Timestamp: time.Now().Format(time.RFC3339),
				}

				if err := sendChunk(ctx, chunk); err != nil {
					return "", fmt.Errorf("could not send countdown chunk: %w", err)
				}
			}

			return "Liftoff!", nil
		})

	// The stream manager is what makes the route durable. This one holds
	// streams in memory and drops a finished one 10 minutes later, so a
	// reconnect works for that long and a restart loses them all. Serving the
	// flow without it leaves the flow itself unchanged and the stream ordinary.
	mux := http.NewServeMux()
	mux.HandleFunc("POST /countdown", genkit.Handler(countdown,
		genkit.WithStreamManager(streaming.NewInMemoryStreamManager(streaming.WithTTL(10*time.Minute))),
	))
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
