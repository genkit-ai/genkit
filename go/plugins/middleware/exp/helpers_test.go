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

import (
	"context"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

var ctx = context.Background()

// newTestGenkit returns a fresh Genkit instance for a test.
func newTestGenkit(t *testing.T) *genkit.Genkit {
	t.Helper()
	return genkit.Init(context.Background())
}

// findSystem returns the first system message, or nil.
func findSystem(msgs []*ai.Message) *ai.Message {
	for _, m := range msgs {
		if m.Role == ai.RoleSystem {
			return m
		}
	}
	return nil
}
