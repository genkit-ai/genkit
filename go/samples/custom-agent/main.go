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

// This sample demonstrates the AgentFlow API for multi-turn conversation
// with token-level streaming. It runs a CLI REPL where conversation history
// is managed automatically by the session.
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	chatFlow := genkit.DefineCustomAgent(g, "chat",
		func(ctx context.Context, resp aix.Responder[any], sess *aix.AgentSession[any]) (*aix.AgentFlowResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentFlowInput) error {
				for chunk, err := range genkit.GenerateStream(ctx, g,
					ai.WithModel(googlegenai.ModelRef("googleai/gemini-3-flash-preview", &genai.GenerateContentConfig{
						ThinkingConfig: &genai.ThinkingConfig{
							ThinkingBudget: genai.Ptr[int32](0),
						},
					})),
					ai.WithSystem("You are a helpful assistant. Keep responses concise."),
					ai.WithMessages(sess.Messages()...),
				) {
					if err != nil {
						return err
					}
					if chunk.Done {
						sess.AddMessages(chunk.Response.Message)
						break
					}
					resp.SendModelChunk(chunk.Chunk)
				}

				return nil
			}); err != nil {
				return nil, err
			}
			return sess.Result(), nil
		},
		aix.WithSessionStore(aix.NewInMemorySessionStore[any]()),
		aix.WithSnapshotOn[any](aix.SnapshotEventTurnEnd),
	)

	fmt.Println("Agent Flow Chat (type 'quit' to exit)")
	fmt.Println()

	conn, err := chatFlow.StreamBidi(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "quit" || input == "exit" {
			break
		}
		if input == "" {
			continue
		}

		if err := conn.SendText(input); err != nil {
			fmt.Fprintf(os.Stderr, "Send error: %v\n", err)
			break
		}

		fmt.Println()

		for chunk, err := range conn.Receive() {
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				break
			}
			if chunk.ModelChunk != nil {
				fmt.Print(chunk.ModelChunk.Text())
			}
			if chunk.SnapshotID != "" {
				fmt.Printf("\n[snapshot: %s]", chunk.SnapshotID)
			}
			if chunk.EndTurn {
				fmt.Println()
				fmt.Println()
				break
			}
		}
	}

	conn.Close()
	fmt.Println(conn.Output())
}
