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

package internal

import (
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
)

// SortToolDefinitions returns a new slice of tool definitions sorted by Name.
//
// Genkit's tool registry is a Go map, so iteration order is randomized —
// without sorting, two requests with identical tool sets produce different
// wire bytes. Stable order makes the provider payload bytewise identical
// across turns, which is a precondition for provider-side prompt caching
// (e.g. Anthropic's cache_control) to match.
//
// Nil entries are filtered out so callers can iterate the result without
// guarding against a nil-pointer dereference on field access.
func SortToolDefinitions(tools []*ai.ToolDefinition) []*ai.ToolDefinition {
	if len(tools) <= 1 {
		if len(tools) == 1 && tools[0] == nil {
			return tools[:0]
		}
		return tools
	}
	sorted := make([]*ai.ToolDefinition, 0, len(tools))
	for _, t := range tools {
		if t == nil {
			continue
		}
		sorted = append(sorted, t)
	}
	slices.SortStableFunc(sorted, func(a, b *ai.ToolDefinition) int {
		return strings.Compare(a.Name, b.Name)
	})
	return sorted
}
