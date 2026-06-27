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
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
)

// ChatPromptInput is the input schema referenced by ./prompts/chef.prompt.
// Registering it via DefineSchemaFor lets the .prompt file refer to it by
// name in its YAML frontmatter.
type ChatPromptInput struct {
	Personality string `json:"personality"`
}

// definePromptAgent demonstrates DefinePromptAgent. The prompt is loaded from
// ./prompts/<agent-name>.prompt by genkit's prompt registry. Defining the
// prompt in a file lets you tune model, config, schema, template, and default
// input independently of the Go code, which is useful when prompt authors are
// not the people writing the agent wiring.
//
// With no source option, DefinePromptAgent defaults to the prompt registered
// under the agent's own name and renders it with the prompt's own default
// input each turn (here, the personality set in chef.prompt's frontmatter).
// The prompt source is a typed option, so it sits in the same variadic as the
// other agent options. To supply an input from code, or to back several agents
// with one shared prompt, add aix.WithNamedPrompt(name, input).
func definePromptAgent(g *genkit.Genkit) *aix.Agent[any] {
	const name = "chef"
	// chef.prompt's frontmatter references ChatPromptInput by name, so the
	// schema must be registered before DefinePromptAgent renders the prompt
	// at definition time.
	genkit.DefineSchemaFor[ChatPromptInput](g)
	return genkitx.DefinePromptAgent(g, name,
		aix.WithSessionStore(mustStore(name)),
		aix.WithDescription[any]("Michelin-starred chef (prompt loaded from ./prompts/chef.prompt)"),
	)
}
