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

This package wraps the standard library's [log/slog] package. Genkit itself
logs through it, and application code inside flows and tools can use it to get
the same behavior: logs flow through the process default logger, carry any
attributes bound to the context's logger, and, during local development, are
streamed to the Genkit Dev UI attached to the trace span that emitted them.

# Usage

Log with the package-level functions, passing the context:

	func myFlow(ctx context.Context, input string) (string, error) {
		logger.Info(ctx, "processing input", "size", len(input))

		result, err := process(input)
		if err != nil {
			logger.Error(ctx, "processing failed", "error", err)
			return "", err
		}

		logger.Debug(ctx, "processing complete", "resultSize", len(result))
		return result, nil
	}

Passing the context is what ties a record to its surroundings: the context
carries the active trace span (for Dev UI and Cloud Logging correlation) and
optionally a logger with pre-bound attributes. Equivalent behavior is
available from any standard logger via the *Context methods, e.g.
slog.InfoContext(ctx, ...).

# Log levels

The default minimum level for the console is Info, and the level only ever
governs the console: during development the Dev UI receives every record at
debug level and above regardless, so the terminal stays quiet while the full
debug narrative lands in the trace viewer. To also see Genkit's per-request
detail (action runs, model calls, tool loops) in the terminal, run with:

	GENKIT_LOG_LEVEL=debug go run .

or set the level programmatically:

	logger.SetLevel(slog.LevelDebug)

For an interactive CLI app whose terminal should stay pristine, run with
GENKIT_LOG_LEVEL=warn to silence the startup info lines too; the Dev UI still
receives everything.

[SetLevel] installs Genkit's console handler as the process default.
Applications that configure their own [slog] handler should set that handler's
level instead: Genkit respects a custom default handler, so both SetLevel and
GENKIT_LOG_LEVEL warn and leave it alone rather than replacing it.

# The Dev UI

In the dev environment (GENKIT_ENV=dev, as set by `genkit start`), Genkit
tees every record at debug level and above to the Dev UI's telemetry server
in addition to the console, independent of the console level. Records logged
with a context are attached to the trace span active at that moment and
appear in the span's Logs panel in the trace viewer. Set
GENKIT_OTEL_ENABLE_LOGS=false to turn this off.

# Context integration

[FromContext] returns the context's logger (or the process default) bound to
that context, so records logged even through its plain methods (Info, Error,
...) still carry the span that was active when the logger was obtained. Store
a derived logger with [WithContext] to bind attributes to everything logged
downstream:

	ctx = logger.WithContext(ctx, logger.FromContext(ctx).With("requestId", id))

The package-level logging functions use the context's logger automatically,
so code below the WithContext call needs no extra plumbing for its records to
carry requestId.

# Additional destinations

[AddHandler] tees the default logger's records to another [slog.Handler]
without disturbing console output. Genkit uses it internally for the Dev UI
export; applications can use it to mirror logs to a file or a test recorder.
*/
package logger
