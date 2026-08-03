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

package anthropic

import ant "github.com/firebase/genkit/go/plugins/internal/anthropic"

// ThinkingDisplay is the Anthropic thinking display mode for adaptive thinking.
type ThinkingDisplay = ant.ThinkingDisplay

const (
	ThinkingDisplaySummarized = ant.ThinkingDisplaySummarized
	ThinkingDisplayOmitted    = ant.ThinkingDisplayOmitted
)

// ThinkingConfig is the Genkit-shaped extended-thinking config matching the JS
// ThinkingConfigSchema (enabled / budgetTokens / adaptive / display).
//
// Prefer this shape in map-style model config for Dev UI / JS parity. Typed
// [anthropic.MessageNewParams] configs may still set Thinking via the SDK union.
type ThinkingConfig = ant.ThinkingConfig
