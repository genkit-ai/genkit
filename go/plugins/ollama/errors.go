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

package ollama

import (
	"context"
	"errors"
	"fmt"
	"net"

	"github.com/firebase/genkit/go/core/status"
)

// sendError classifies a request that never got a response from the Ollama
// server. An unreachable server is Unavailable and a client timeout is
// DeadlineExceeded, so Fallback can move on to the next model when the local
// server is down. When the caller's context ended, the context error is kept
// as is so the request reports Cancelled or DeadlineExceeded from the caller.
func sendError(ctx context.Context, serverAddress string, err error) error {
	if ctx.Err() != nil {
		return fmt.Errorf("ollama request to %s: %w", serverAddress, err)
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return status.Errorf(status.ErrDeadlineExceeded, "request to Ollama server at %s timed out: %w", serverAddress, err)
	}
	return status.Errorf(status.ErrUnavailable,
		"cannot reach the Ollama server at %s; start it with `ollama serve` or set ServerAddress to a reachable host: %w",
		serverAddress, err)
}

// responseError classifies a non-200 reply by its HTTP status: a model that
// isn't pulled is NotFound, a bad request is InvalidArgument, a server failure
// is Internal.
func responseError(code int, body []byte) error {
	return status.Errorf(status.Base(status.FromHTTPCode(code)), "ollama server returned status %d: %s", code, body)
}
