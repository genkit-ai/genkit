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

package main

import (
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
)

// ChatPromptInput is the input schema referenced by ./prompts/chef.prompt.
// Registering it via DefineSchemaFor (in main) lets the .prompt file refer to
// it by name in its YAML frontmatter.
type ChatPromptInput struct {
	Personality string `json:"personality"`
}

// definePromptAgent demonstrates DefineAgent with aix.FromPrompt. The
// prompt is loaded from ./prompts/<agent-name>.prompt by genkit's prompt
// registry. Defining the prompt in a file lets you tune model, config,
// schema, and template independently of the Go code — useful when prompt
// authors are not the same people writing the agent wiring.
//
// FromPrompt's argument is the default input passed to the prompt's
// Render on every turn; the inline-prompt variant has no per-turn input
// of its own.
func definePromptAgent(g *genkit.Genkit) *aix.Agent[any] {
	const name = "chef"
	return genkit.DefineAgent(g, name,
		aix.FromPrompt(ChatPromptInput{Personality: "a Michelin-starred chef who loves explaining technique"}),
		aix.WithSessionStore(mustStore(name)),
		aix.WithDescription[any]("Michelin-starred chef (prompt loaded from ./prompts/chef.prompt)"),
	)
}
