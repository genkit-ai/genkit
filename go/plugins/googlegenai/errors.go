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

package googlegenai

import (
	"errors"

	"github.com/firebase/genkit/go/core"
	"google.golang.org/genai"
)

// wrapAPIError wraps a [genai.APIError] in a [core.GenkitError] whose status
// matches the one the server reported so status-aware middleware (retry,
// fallback, ...) can reason about it. Non-APIError values pass through.
//
// The SDK's Status string is a canonical Google / gRPC status name which,
// by design, already matches the string value of every [core.StatusName]
// constant except INTERNAL (our constant spells it "INTERNAL_SERVER_ERROR").
// When Status is missing or unrecognised the HTTP Code is the fallback.
func wrapAPIError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		return err
	}
	return core.NewError(statusForAPIError(apiErr), "%s", err)
}

func statusForAPIError(e genai.APIError) core.StatusName {
	if e.Status == "INTERNAL" {
		return core.INTERNAL
	}
	s := core.StatusName(e.Status)
	if _, ok := core.StatusNameToCode[s]; ok {
		return s
	}
	return core.StatusFromHTTPCode(e.Code)
}
