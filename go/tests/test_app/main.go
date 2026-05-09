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

// This program doesn't do anything interesting.
// It is used by go/tests/api_test.go.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/server"
)

func main() {
	// used for streamed flows
	type chunk struct {
		Count int `json:"count"`
	}

	ctx := context.Background()
	g := genkit.Init(ctx)
	model := genkit.DefineModel(g, "customReflector", nil, echo)
	genkit.DefineFlow(g, "testFlow", func(ctx context.Context, in string) (string, error) {
		res, err := genkit.Generate(ctx, g, ai.WithModel(model), ai.WithPrompt(in))
		if err != nil {
			return "", err
		}
		_ = res
		return "TBD", nil
	})

	genkit.DefineStreamingFlow(g, "streamy", func(ctx context.Context, count int, cb func(context.Context, chunk) error) (string, error) {
		i := 0
		if cb != nil {
			for ; i < count; i++ {
				if err := cb(ctx, chunk{i}); err != nil {
					return "", err
				}
			}
		}
		return fmt.Sprintf("done %d, streamed: %d times", count, i), nil
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:3400", mux))
}

func echo(ctx context.Context, req *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	jsonBytes, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	return &ai.ModelResponse{
		FinishReason: "stop",
		Message: &ai.Message{
			Role:    "model",
			Content: []*ai.Part{ai.NewTextPart(string(jsonBytes))},
		},
	}, nil
}
