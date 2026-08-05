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

package ai

import (
	"testing"

	"github.com/firebase/genkit/go/core/status"
)

// Each domain sentinel must carry the status its call sites sent before they
// were classified. A drift here would change the HTTP code clients see and the
// retry/fallback decision, without any call site changing.
func TestDomainSentinelStatuses(t *testing.T) {
	for _, tt := range []struct {
		name string
		s    *status.Sentinel
		want status.Name
	}{
		{"ErrModelNotFound", ErrModelNotFound, status.NotFound},
		{"ErrToolNotFound", ErrToolNotFound, status.NotFound},
		{"ErrMaxTurnsExceeded", ErrMaxTurnsExceeded, status.Aborted},
		{"ErrToolFailed", ErrToolFailed, status.Internal},
		{"ErrUnsupportedByModel", ErrUnsupportedByModel, status.InvalidArgument},
		{"ErrInvalidPart", ErrInvalidPart, status.InvalidArgument},
		{"ErrUnresolvedToolRequest", ErrUnresolvedToolRequest, status.InvalidArgument},
	} {
		if got := tt.s.Status(); got != tt.want {
			t.Errorf("%s.Status() = %q, want %q", tt.name, got, tt.want)
		}
	}
}
