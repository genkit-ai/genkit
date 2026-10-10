// Copyright 2026 Google LLC
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

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/google/uuid"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	middlewarex "github.com/firebase/genkit/go/plugins/middleware/exp"
)

// runOrchestrator starts the researcher service, registers it as a remote
// agent, and chats with an orchestrator that delegates to it.
func runOrchestrator(ctx context.Context) error {
	researcherURL, stopResearcher, err := startResearcher(ctx)
	if err != nil {
		return err
	}
	defer stopResearcher()

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}), genkit.WithExperimental())

	// The researcher is now an agent of this app, under the same name. url is
	// its turn route; the snapshot routes the background tools use sit next
	// to it. The metadata declares what the service supports: it keeps the
	// session (a store) and can stop background work (a store that observes
	// aborts). The middleware gates background delegation on it.
	researcher := genkitx.DefineRemoteAgent(g, "researcher", researcherURL+"/agents/researcher",
		aix.WithDescription[any]("Researches one question and answers in a few sentences. Runs as its own service."),
		aix.WithAgentMetadata(&aix.AgentMetadata{
			StateManagement: aix.AgentStateManagementServer,
			Abortable:       true,
		}),
	)

	orchestrator := genkitx.DefineAgent(g, "orchestrator",
		aix.InlinePrompt{
			ai.WithModel(model),
			ai.WithSystem("You are a planning assistant. When a question has several parts, " +
				"give each part to the researcher as its own task, all of them in the background " +
				"in one turn, then wait for every task and combine the answers into one reply. " +
				"Answer a simple question yourself."),
			// The same middleware a local sub-agent uses. The reference is by
			// name, so nothing here says that the researcher is remote.
			ai.WithUse(&middlewarex.Agents{
				Agents: []aix.AgentRef{researcher.Ref()},
				Async:  true,
			}),
			// Launch, wait, and answer are three tool rounds at least.
			ai.WithMaxTurns(10),
		},
		aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
	)

	return chat(ctx, orchestrator)
}

// chat reads questions from stdin and streams the orchestrator's replies,
// with one line per tool call, until end of input, an empty line, or ctx ends.
// Every turn resumes the same session, so the orchestrator keeps the
// conversation.
func chat(ctx context.Context, agent *aix.Agent[any]) error {
	sessionID := uuid.NewString()
	lines := make(chan string)
	go func() {
		defer close(lines)
		in := bufio.NewScanner(os.Stdin)
		for in.Scan() {
			lines <- in.Text()
		}
	}()

	fmt.Println("Ask a question. An empty line or Ctrl-D quits.")
	for {
		fmt.Print("\n> ")
		var text string
		select {
		case <-ctx.Done():
			return nil
		case line, ok := <-lines:
			if !ok {
				return nil
			}
			text = strings.TrimSpace(line)
		}
		if text == "" {
			return nil
		}
		if err := turn(ctx, agent, sessionID, text); err != nil {
			return err
		}
	}
}

// turn runs one turn of the session and prints it as it streams.
func turn(ctx context.Context, agent *aix.Agent[any], sessionID, text string) error {
	conn, err := agent.Connect(ctx, aix.WithSessionID[any](sessionID))
	if err != nil {
		return err
	}
	if err := conn.SendText(text); err != nil {
		return err
	}
	conn.Close()
	for chunk, err := range conn.Receive() {
		if err != nil {
			return err
		}
		if mc := chunk.ModelChunk; mc != nil {
			fmt.Print(mc.Text())
			for _, p := range mc.Content {
				if p.IsToolRequest() && !p.ToolRequest.Partial {
					input, _ := json.Marshal(p.ToolRequest.Input)
					fmt.Printf("\n  [%s %s]\n", p.ToolRequest.Name, input)
				}
			}
		}
	}
	out, err := conn.Output()
	if err != nil {
		return err
	}
	if out.FinishReason == aix.AgentFinishReasonFailed {
		fmt.Printf("\n(the turn failed: %v)", out.Error)
	}
	fmt.Println()
	return nil
}
