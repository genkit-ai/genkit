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

/*
Package logger provides context-scoped structured logging for Genkit.

This package wraps the standard library's [log/slog] package to provide
context-aware logging throughout Genkit operations. Logs are automatically
associated with the current action or flow context.

# Usage

Retrieve the logger from context within action or flow handlers:

	func myFlow(ctx context.Context, input string) (string, error) {
		log := logger.FromContext(ctx)

		log.Info("Processing input", "size", len(input))
		log.Debug("Input details", "value", input)

		result, err := process(input)
		if err != nil {
			log.Error("Processing failed", "error", err)
			return "", err
		}

		log.Info("Processing complete", "resultSize", len(result))
		return result, nil
	}

# Log Levels

Control the global log level to filter output:

	// Show debug logs (verbose)
	logger.SetLevel(slog.LevelDebug)

	// Show info and above (default)
	logger.SetLevel(slog.LevelInfo)

	// Show only warnings and errors
	logger.SetLevel(slog.LevelWarn)

	// Show only errors
	logger.SetLevel(slog.LevelError)

	// Get the current log level
	level := logger.GetLevel()

# Context Integration

The logger is automatically available in action and flow contexts. It
inherits from the context passed to [genkit.Init] and flows through
all nested operations.

For custom operations outside of actions/flows, attach a logger to context:

	log := slog.Default()
	ctx = logger.WithContext(ctx, log)

# slog Compatibility

The logger returned by [FromContext] is a standard [*slog.Logger] and
supports all slog methods:

	log := logger.FromContext(ctx)

	// Structured logging with attributes
	log.Info("User action",
		"userId", userID,
		"action", "login",
		"duration", elapsed,
	)

	// Grouped attributes
	log.Info("Request completed",
		slog.Group("request",
			"method", r.Method,
			"path", r.URL.Path,
		),
		slog.Group("response",
			"status", status,
			"bytes", written,
		),
	)

	// With pre-set attributes
	requestLog := log.With("requestId", requestID)
	requestLog.Info("Starting")
	// ... later ...
	requestLog.Info("Finished")

This package is primarily used by Genkit internals but is useful for
plugin developers who need consistent logging that integrates with
Genkit's observability features.
*/
package logger
