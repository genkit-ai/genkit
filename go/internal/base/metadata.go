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

package base

// Metadata keys that one package writes and another reads.
const (
	// PromptMessageKey is the message metadata key that tags the messages an
	// agent's prompt renders on every turn.
	PromptMessageKey = "_genkit_prompt"

	// PartPurposeKey is the part metadata key that records why the framework
	// added a part. PartPurposeOutput marks the output-format instructions
	// the generate loop injects.
	PartPurposeKey    = "purpose"
	PartPurposeOutput = "output"
)
