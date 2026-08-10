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
)

// defineInlineAgent demonstrates DefineAgent with aix.InlinePrompt. The
// prompt is declared right next to the agent definition; the registered
// prompt and the agent share a name. Each turn the framework renders the
// prompt, appends the conversation history, calls the model, and updates
// session state. This is the shortest path from "I want a chat agent" to
// a working one.
func defineInlineAgent(g *genkit.Genkit) *aix.Agent[any] {
	const name = "pirate"
	return genkitx.DefineAgent(g, name,
		aix.InlinePrompt{
			ai.WithModel(flashModel),
			ai.WithSystem("You are a sarcastic pirate. Keep every reply to a sentence or two, sharp and to the point."),
		},
		aix.WithSessionStore(mustStore(name)),
		aix.WithDescription[any]("Sarcastic pirate (inline-defined prompt)"),
	)
}
