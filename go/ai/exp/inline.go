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
//
// SPDX-License-Identifier: Apache-2.0

package exp

import "github.com/firebase/genkit/go/ai"

// InlinePrompt is an inline prompt definition for an agent: the list of
// [ai.PromptOption] values that configure the agent's prompt. Pass one to
// [DefineAgent], which registers the prompt under the agent's name:
//
//	agent := DefineAgent(r, "pirate",
//		InlinePrompt{
//			ai.WithModelName("googleai/gemini-flash-latest"),
//			ai.WithSystem("You are a sarcastic pirate."),
//		},
//		WithSessionStore(store),
//	)
//
// To give the template a default render input, include [ai.WithInputType] among
// the options. For an agent backed by a prompt already in the registry (e.g.
// one defined via [ai.DefinePrompt] or loaded from a .prompt file), use
// [DefinePromptAgent] instead, which takes no InlinePrompt.
type InlinePrompt []ai.PromptOption
