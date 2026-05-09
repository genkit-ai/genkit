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

// This sample demonstrates durable streaming with Firestore backend.
// Unlike in-memory streaming, Firestore-backed streams survive server restarts
// and can be accessed across multiple server instances.
//
// Prerequisites:
//   - Firebase/GCP project with Firestore enabled
//   - Run: gcloud auth application-default login
//   - Set: export FIREBASE_PROJECT_ID=your-project-id
//
// To run:
//
//	go run .
//
// In another terminal, start a streaming request:
//
//	curl -N -i -H "Accept: text/event-stream" \
//	  -d '{"data": 5}' \
//	  http://localhost:8088/countdown
//
// Note the X-Genkit-Stream-Id header. To reconnect to the same stream:
//
//	curl -N -H "Accept: text/event-stream" \
//	  -H "X-Genkit-Stream-Id: <stream-id>" \
//	  -d '{"data": 5}' \
//	  http://localhost:8088/countdown
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/firebase"
	firebasex "github.com/firebase/genkit/go/plugins/firebase/x"
	"github.com/firebase/genkit/go/plugins/server"
)

func main() {
	ctx := context.Background()

	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{}))

	type CountdownChunk struct {
		Count     int    `json:"count"`
		Message   string `json:"message"`
		Timestamp string `json:"timestamp"`
	}

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

	sm, err := firebasex.NewFirestoreStreamManager(ctx, g,
		firebasex.WithCollection("genkit-streams"),
		firebasex.WithTimeout(2*time.Minute),
		firebasex.WithTTL(10*time.Minute),
	)
	if err != nil {
		log.Fatalf("Failed to create Firestore stream manager: %v", err)
	}

	// Set up HTTP server with durable streaming enabled.
	// Completed streams are kept for 10 minutes before cleanup.
	mux := http.NewServeMux()
	mux.HandleFunc("POST /countdown", genkit.Handler(countdown, genkit.WithStreamManager(sm)))
	log.Fatal(server.Start(ctx, "127.0.0.1:8088", mux))
}
