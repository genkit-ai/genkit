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
// AgentSource as the third argument to [DefineAgent]. There are two
// forms:
//
//   - [FromInline] defines the prompt inline from a set of
//     [ai.PromptOption] values; the prompt is registered with the
//     registry under the agent's name.
//   - [FromPrompt] references an existing prompt registered with the
//     registry under the same name as the agent (e.g. one defined via
//     [ai.DefinePrompt] or loaded from a .prompt file).
//
// The agent and its backing prompt always share a name; if you need
// the lookup name to differ from the agent name, define a custom agent
// via [DefineCustomAgent] instead.
type AgentSource interface {
	isAgentSource()
}

type inlineSource struct {
	opts []ai.PromptOption
}

func (inlineSource) isAgentSource() {}

// FromInline defines the agent's prompt inline from the given options.
// The prompt is registered with the registry under the agent's name.
func FromInline(opts ...ai.PromptOption) AgentSource {
	return inlineSource{opts: opts}
}

type promptSource struct {
	defaultInput any
}

func (promptSource) isAgentSource() {}

// FromPrompt references an existing prompt registered with the
// registry under the same name as the agent (e.g. one defined via
// [ai.DefinePrompt] or loaded from a .prompt file).
//
// defaultInput, if provided, is the input passed to the prompt's
// Render on every turn. Call FromPrompt() with no arguments when the
// prompt takes no input. Only the first argument is used; any
// additional arguments are ignored.
func FromPrompt(defaultInput ...any) AgentSource {
	var input any
	if len(defaultInput) > 0 {
		input = defaultInput[0]
	}
	return promptSource{defaultInput: input}
}
