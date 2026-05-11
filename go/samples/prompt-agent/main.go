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

// This sample demonstrates DefinePromptAgent, which creates a multi-turn
// conversational agent backed by a .prompt file. The conversation loop
// (render prompt, call model, stream chunks, update history) is handled
// automatically. Compare with custom-agent which wires the same loop
// manually.
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

type ChatPromptInput struct {
	Personality string `json:"personality"`
}

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	chatAgent := genkit.DefinePromptAgent[any](g, "chat",
		ChatPromptInput{Personality: "a sarcastic pirate"},
		aix.WithSessionStore(aix.NewInMemorySessionStore[any]()),
		aix.WithSnapshotCallback(func(ctx context.Context, sc *aix.SnapshotContext[any]) bool {
			return sc.Event == aix.SnapshotEventInvocationEnd || sc.TurnIndex%5 == 0
		}),
	)

	fmt.Println("Agent Chat (type 'quit' to exit)")
	fmt.Println()

	conn, err := chatAgent.StreamBidi(ctx)
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
			if chunk.TurnEnd != nil {
				if chunk.TurnEnd.SnapshotID != "" {
					fmt.Printf("\n[snapshot: %s]", chunk.TurnEnd.SnapshotID)
				}
				fmt.Println()
				fmt.Println()
				break
			}
		}
	}

	conn.Close()
}
