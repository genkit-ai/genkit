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
	"strings"
	"time"

	"google.golang.org/genai"

	"github.com/firebase/genkit/go/core/status"
)

// wrapAPIError wraps a [genai.APIError] in a [status.Error] whose status
// matches the one the server reported so status-aware middleware (retry,
// fallback, ...) can reason about it. Non-APIError values pass through.
// When the service attached retry information (rate limits report the delay
// to wait before retrying), it is carried in the error's Details under
// "retryAfterMs" and is also readable through [RetryDelay].
//
// The SDK's Status string is a canonical Google / gRPC status name and so
// matches the string value of each [status.Name] constant directly.
// When Status is missing or unrecognised the HTTP Code is the fallback.
func wrapAPIError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		return err
	}
	serr := status.Errorf(status.Base(statusForAPIError(apiErr)), "%w", err)
	if d, ok := retryDelayFromAPIError(apiErr); ok {
		serr = serr.WithDetails(map[string]any{"retryAfterMs": d.Milliseconds()})
	}
	return serr
}

// RetryDelay reports the delay the service asked the caller to wait before
// retrying, taken from the google.rpc.RetryInfo detail that Gemini and
// Vertex AI attach to RESOURCE_EXHAUSTED (429) errors. It reads errors
// returned by this plugin's actions, including wrapped ones. The second
// result is false when err carries no retry information.
func RetryDelay(err error) (time.Duration, bool) {
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		return 0, false
	}
	return retryDelayFromAPIError(apiErr)
}

// retryDelayFromAPIError extracts the google.rpc.RetryInfo delay from the
// error's details.
func retryDelayFromAPIError(apiErr genai.APIError) (time.Duration, bool) {
	for _, detail := range apiErr.Details {
		typ, _ := detail["@type"].(string)
		if !strings.HasSuffix(typ, "google.rpc.RetryInfo") {
			continue
		}
		switch rd := detail["retryDelay"].(type) {
		case string:
			// Proto JSON renders a Duration as a decimal-seconds string,
			// e.g. "58s" or "0.5s".
			if d, perr := time.ParseDuration(rd); perr == nil {
				return d, true
			}
		case map[string]any:
			// Some transports render the Duration message fields instead.
			secs, okSecs := castToFloat64(rd["seconds"])
			nanos, okNanos := castToFloat64(rd["nanos"])
			if okSecs || okNanos {
				return time.Duration(secs*float64(time.Second) + nanos), true
			}
		}
	}
	return 0, false
}

func statusForAPIError(e genai.APIError) status.Name {
	if n := status.Name(e.Status); n.IsValid() {
		return n
	}
	return status.FromHTTPCode(e.Code)
}
