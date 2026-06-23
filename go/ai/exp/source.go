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

package exp

import "github.com/firebase/genkit/go/ai"

// AgentSource selects the prompt backing a prompt-based agent. Pass an
// AgentSource as the third argument to [DefineAgent]. There are three
// forms:
//
//   - [InlinePrompt] defines the prompt inline from a set of
//     [ai.PromptOption] values; the prompt is registered with the
//     registry under the agent's name.
//   - [SameNamedPrompt] references the prompt already registered under the
//     agent's own name (e.g. one defined via [ai.DefinePrompt] or loaded
//     from a .prompt file).
//   - [NamedPrompt] references any registered prompt by name and renders it
//     with an input supplied from code, so a single prompt can back many
//     agents with different inputs.
//
// For full control over the per-turn loop, define a custom agent via
// [DefineCustomAgent] instead.
type AgentSource interface {
	isAgentSource()
}

type inlineSource struct {
	opts []ai.PromptOption
}

func (inlineSource) isAgentSource() {}

// InlinePrompt defines the agent's prompt inline from the given options.
// The prompt is registered with the registry under the agent's name. To
// give the template a default render input, include [ai.WithInputType]
// among the options.
func InlinePrompt(opts ...ai.PromptOption) AgentSource {
	return inlineSource{opts: opts}
}

type existingSource struct {
	name  string // "" => resolve by the agent's own name
	input any
}

func (existingSource) isAgentSource() {}

// SameNamedPrompt references the prompt registered under the agent's own
// name (e.g. one defined via [ai.DefinePrompt] or loaded from a .prompt
// file). The prompt renders with its own default input each turn. It is
// shorthand for NamedPrompt(<agent name>, nil).
func SameNamedPrompt() AgentSource {
	return existingSource{}
}

// NamedPrompt references the prompt registered under name, rendered with
// input on every turn (pass nil for the prompt's own default input). name
// need not match the agent's name, so a single prompt can back many agents
// with different inputs.
//
// input is rendered through the prompt once at definition time as a smoke
// check, so an input that fails the prompt's schema panics there rather
// than on the first invocation.
func NamedPrompt(name string, input any) AgentSource {
	return existingSource{name: name, input: input}
}
