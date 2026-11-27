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

package modelgarden_test

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/internal/oailive"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

func TestLlamaLive(t *testing.T) {
	vertexEnv(t)
	g := livetest.Init(t, &modelgarden.Llama{})

	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model: ai.NewModelRef("vertexai/meta/llama-4-maverick-17b-128e-instruct-maas", nil),
		},
	})
}
