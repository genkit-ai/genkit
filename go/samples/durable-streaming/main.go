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

// This sample demonstrates durable streaming, which allows clients to reconnect
// to in-progress or completed streams using a stream ID.
//
// Start the server:
//
//	go run .
//
// Test streaming (get a stream ID back in X-Genkit-Stream-Id header):
//
//	curl -N -i -H "Accept: text/event-stream" \
//	     -d '{"data": 5}' \
//	     http://localhost:8080/countdown
//
// Subscribe to an existing stream using the stream ID from the previous response:
//
//	curl -N -H "Accept: text/event-stream" \
//	     -H "X-Genkit-Stream-Id: <stream-id-from-above>" \
//	     -d '{"data": 5}' \
//	     http://localhost:8080/countdown
//
// The subscription will replay any buffered chunks and then continue with live updates.
// If the stream has already completed, all chunks plus the final result are returned.
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

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	type CountdownChunk struct {
		Count     int    `json:"count"`
		Message   string `json:"message"`
		Timestamp string `json:"timestamp"`
	}

	// Define a streaming flow that counts down with delays.
	countdown := genkit.DefineStreamingFlow(g, "countdown",
		func(ctx context.Context, count int, sendChunk func(context.Context, CountdownChunk) error) (string, error) {
			if count <= 0 {
				count = 5
			}

			for i := count; i > 0; i-- {
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
					return "", err
				}
			}

			return "Liftoff!", nil
		})

	// Set up HTTP server with durable streaming enabled.
	// Completed streams are kept for 10 minutes before cleanup (while server is running).
	mux := http.NewServeMux()
	mux.HandleFunc("POST /countdown", genkit.Handler(countdown,
		genkit.WithStreamManager(streaming.NewInMemoryStreamManager(streaming.WithTTL(10*time.Minute))),
	))
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
