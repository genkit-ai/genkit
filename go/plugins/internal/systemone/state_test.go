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

package systemone

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

func TestStateJSONKeepsNumbersVerbatim(t *testing.T) {
	// The JSON goes out as written, so an ID past a float64's integer range
	// is not rounded.
	wire, err := BuildState(&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage(`{"id": 9007199254740993}`)}}, true)
	if err != nil {
		t.Fatal(err)
	}
	if got := base.JSONString(wire); got != `{"id":9007199254740993}` {
		t.Errorf("state on the wire = %s, want the number unchanged", got)
	}
}
