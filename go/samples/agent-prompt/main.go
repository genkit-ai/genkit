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

// This sample demonstrates DefineAgent with aix.FromPrompt, which
// creates a multi-turn conversational agent backed by a .prompt file.
// The conversation loop (render prompt, call model, stream chunks,
// update history) is handled automatically. Compare with agent-custom
// (DefineCustomAgent), which wires the same loop manually, and
// agent-inline (DefineAgent + aix.FromInline), which defines the
// prompt inline alongside the agent.
package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

type ChatPromptInput struct {
	Personality string `json:"personality"`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	genkit.DefineSchemaFor[ChatPromptInput](g)

	store, err := localstore.NewFileSessionStore[any]("./sessions")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	chatAgent := genkit.DefineAgent(g, "chat",
		aix.FromPrompt(ChatPromptInput{Personality: "a sarcastic pirate"}),
		aix.WithSessionStore(store),
		aix.WithSnapshotCallback(func(ctx context.Context, sc *aix.SnapshotContext[any]) bool {
			return sc.Event == aix.SnapshotEventInvocationEnd || sc.TurnIndex%5 == 0
		}),
	)

	fmt.Println("Agent Chat (type 'quit' to exit, Ctrl+C to abort)")
	fmt.Println()

	conn, err := chatAgent.StreamBidi(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	inputCh := readLines(ctx)

repl:
	for {
		fmt.Print("> ")
		var input string
		select {
		case <-ctx.Done():
			fmt.Println()
			break repl
		case line, ok := <-inputCh:
			if !ok {
				fmt.Println()
				break repl
			}
			input = strings.TrimSpace(line)
		}

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

	out, err := conn.Output()
	if err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if out != nil && out.SnapshotID != "" {
		fmt.Printf("[snapshot: %s]\n", out.SnapshotID)
	}
	fmt.Println("You left the conversation. Goodbye!")
}

// readLines reads lines from stdin on a background goroutine and yields
// them via the returned channel. The channel is closed on EOF, read error,
// or ctx cancellation. The goroutine cannot interrupt a blocked stdin
// read; on ctx cancellation it exits as soon as a line completes (or, in
// practice, when the process terminates).
func readLines(ctx context.Context) <-chan string {
	ch := make(chan string)
	go func() {
		defer close(ch)
		reader := bufio.NewReader(os.Stdin)
		for {
			line, err := reader.ReadString('\n')
			if line != "" {
				select {
				case ch <- line:
				case <-ctx.Done():
					return
				}
			}
			if err != nil {
				return
			}
		}
	}()
	return ch
}
