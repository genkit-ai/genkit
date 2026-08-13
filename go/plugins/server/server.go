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

package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

// shutdownTimeout bounds the graceful drain of in-flight requests after an
// interrupt, matching the reflection server's shutdown timeout.
const shutdownTimeout = 5 * time.Second

// Start starts a new HTTP server and manages its lifecycle.
// This is a convenience function since Go does not manage interrupt signals directly.
func Start(ctx context.Context, addr string, mux *http.ServeMux) error {
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	srv := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	errChan := make(chan error, 1)

	go func() {
		slog.Info("server listening", "addr", addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errChan <- fmt.Errorf("server error: %w", err)
		}
		cancel()
	}()

	select {
	case err := <-errChan:
		return err
	case <-ctx.Done():
		slog.Info("server shutting down", "addr", addr)
		// Stop intercepting signals so a second interrupt kills the process
		// immediately instead of being swallowed while requests drain.
		cancel()
		// ctx is already canceled here; Shutdown needs a fresh context or it
		// returns immediately without draining in-flight requests.
		shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), shutdownTimeout)
		defer cancelShutdown()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("failed to shutdown server: %w", err)
		}
	}
	return nil
}
