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

// AgentRef refers to an agent by name, optionally carrying a description. It is
// the agent analog of [ai.ModelRef] / [ai.ToolRef]: a small, JSON-serializable
// value that names an agent for resolution against a registry. Like those, it
// resolves by name (the path the Dev UI, HTTP serving, and ListAgents all use),
// so the referenced agent must be registered wherever the ref is consumed.
//
// Build one by name with a struct literal, or derive it from an agent value
// with [Agent.Ref], which fills in the name and description so callers need not
// restate either:
//
//	aix.AgentRef{Name: "researcher"}
//	coderAgent.Ref()
type AgentRef struct {
	// Name identifies the agent, resolved as /agent/<Name>. Required.
	Name string `json:"name"`
	// Description is a human-readable description used by consumers that list
	// agents (e.g. the agents middleware's system prompt). [Agent.Ref] fills it
	// from the agent's descriptor. Optional.
	Description string `json:"description,omitempty"`
}

// Ref returns an [AgentRef] for this agent, capturing its name and description
// so callers can reference it without restating either, and without a name
// string that can drift from the agent. Resolution remains by name, so the
// agent must be registered (as [DefineAgent] does) wherever the ref is used.
func (a *Agent[State]) Ref() AgentRef {
	return AgentRef{
		Name:        a.Name(),
		Description: a.Desc().Description,
	}
}
