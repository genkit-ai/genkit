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
	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	middlewarex "github.com/firebase/genkit/go/plugins/middleware/exp"
	"google.golang.org/genai"
)

// flashModel is the model shared by the orchestrator and its sub-agents:
// gemini-flash-latest with thinking disabled for snappy, low-cost turns.
var flashModel = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](0)},
})

// defineOrchestratorAgent demonstrates the experimental Agents middleware: an
// orchestrator that delegates to specialized sub-agents through per-agent tools.
//
// The middleware injects one delegation tool per sub-agent (delegate_to_<name>),
// lists the sub-agents and their descriptions in the system prompt, and runs the
// chosen sub-agent when the orchestrator model calls its tool. It mirrors the
// JS "orchestrator" sample.
//
// The two sub-agents (researcher, engineer) are client-managed (no session
// store): each delegation runs them one-shot and leaves no snapshots behind, so
// only the orchestrator appears in the CLI. Both use the Artifacts middleware so
// they can persist output as named session artifacts; with
// ArtifactStrategySession those artifacts are merged into the orchestrator's
// session, and Artifacts{Readonly: true} gives the orchestrator a read_artifact
// tool to review them before answering.
func defineOrchestratorAgent(g *genkit.Genkit) *aix.Agent[any] {
	researcher := genkitx.DefineAgent(g, "researcher",
		aix.InlinePrompt{
			ai.WithModel(flashModel),
			ai.WithSystem("You are a thorough research assistant. Answer the question " +
				"concisely, then call write_artifact to save your findings as a named " +
				"markdown artifact (for example \"findings.md\")."),
			ai.WithUse(&middlewarex.Artifacts{}),
		},
		aix.WithDescription[any]("Researches a topic and summarizes well-sourced findings."),
	)

	engineer := genkitx.DefineAgent(g, "engineer",
		aix.InlinePrompt{
			ai.WithModel(flashModel),
			ai.WithSystem("You are an expert programmer. Write clean, well-commented code, " +
				"then call write_artifact to save it as a named file artifact (for " +
				"example \"main.go\")."),
			ai.WithUse(&middlewarex.Artifacts{}),
		},
		aix.WithDescription[any]("Writes and explains code, producing file artifacts."),
	)

	return genkitx.DefineAgent(g, "orchestrator",
		aix.InlinePrompt{
			ai.WithModel(flashModel),
			ai.WithSystem("You are a project coordinator. Analyze the user's request and " +
				"delegate to the appropriate sub-agent using its delegation tool. If a " +
				"request needs both research and code, delegate to each in turn. After " +
				"the sub-agents respond you may call read_artifact to review their work, " +
				"then synthesize a final answer for the user."),
			ai.WithUse(
				// One delegation tool per sub-agent. Descriptions are
				// auto-discovered from each agent (set via WithDescription and
				// captured by Ref). historyLength forwards recent turns to the
				// client-managed sub-agents; artifactStrategy "session" merges
				// their artifacts into this session.
				&middlewarex.Agents{
					Agents:           []aix.AgentRef{researcher.Ref(), engineer.Ref()},
					MaxDelegations:   5,
					HistoryLength:    4,
					ArtifactStrategy: middlewarex.ArtifactStrategySession,
				},
				// Read-only artifact access: the orchestrator reviews sub-agent
				// artifacts but does not produce its own.
				&middlewarex.Artifacts{Readonly: true},
			),
		},
		aix.WithSessionStore(mustStore("orchestrator")),
		aix.WithDescription[any]("Coordinates research and coding sub-agents via the agents middleware"),
	)
}
