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

package compat_oai

import (
	"errors"

	"github.com/openai/openai-go"

	"github.com/firebase/genkit/go/core/status"
)

// WrapAPIError wraps an error the OpenAI SDK returned for an HTTP response in
// a [status.Error] carrying the status the server reported, so status-aware
// middleware (retry, fallback, ...) can tell a rate limit from a request the
// provider rejected. Without it every API failure is unclassified, which the
// retry middleware treats as retryable: a 401 would be reissued unchanged
// until the attempts ran out.
//
// It is exported for the provider packages built on this one, which reach the
// SDK directly for model discovery.
//
// Values that are not an SDK API error pass through untouched. The SDK returns
// transport failures unwrapped, and leaving those unclassified is the right
// answer: a dial timeout really is worth retrying.
func WrapAPIError(err error) error {
	if err == nil {
		return nil
	}
	// The SDK returns *openai.Error; a value-typed target would never match.
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return err
	}
	return status.Errorf(status.Base(status.FromHTTPCode(apiErr.StatusCode)), "%w", err)
}
