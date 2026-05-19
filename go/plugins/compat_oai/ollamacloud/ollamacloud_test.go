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

package ollamacloud

import (
	"context"
	"sort"
	"testing"
)

func TestInitCanBeCalledTwice(t *testing.T) {
	ctx := context.Background()
	plugin := &OllamaCloud{APIKey: "test-api-key"}

	firstActions := plugin.Init(ctx)
	if got, want := len(firstActions), len(supportedModels); got != want {
		t.Fatalf("first Init returned %d actions, want %d", got, want)
	}

	secondActions := plugin.Init(ctx)
	if got, want := len(secondActions), len(supportedModels); got != want {
		t.Fatalf("second Init returned %d actions, want %d", got, want)
	}

	for i := range firstActions {
		if firstActions[i] != secondActions[i] {
			t.Fatalf("action %d was not cached between Init calls", i)
		}
	}
}

func TestInitReturnsModelsInSortedOrder(t *testing.T) {
	ctx := context.Background()
	plugin := &OllamaCloud{APIKey: "test-api-key"}

	actions := plugin.Init(ctx)
	names := make([]string, 0, len(actions))
	for _, action := range actions {
		names = append(names, action.Name())
	}

	if !sort.StringsAreSorted(names) {
		t.Fatalf("Init returned unsorted action names: %v", names)
	}
}
