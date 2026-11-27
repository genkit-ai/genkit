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

import (
	"slices"
	"testing"
)

// TestPartSink_DropsPartsAttachedAfterClose pins the sink's lifetime: Close
// returns what was attached, in order, once, and a part attached after it,
// by a goroutine that outlived the call, is dropped rather than held.
func TestPartSink_DropsPartsAttachedAfterClose(t *testing.T) {
	var s PartSink
	s.Add("a")
	s.Add("b")
	if got := s.Close(); !slices.Equal(got, []any{"a", "b"}) {
		t.Fatalf("Close() = %v, want [a b]", got)
	}
	s.Add("late")
	if got := s.Close(); len(got) != 0 {
		t.Errorf("Close() after a late Add = %v, want nothing", got)
	}
	if len(s.parts) != 0 {
		t.Errorf("sink holds %v after close, want nothing", s.parts)
	}
}
