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

// Package tracingbridge is an internal bridge that lets first-party Genkit
// packages (genkit.Init and the reflection server) drive dev-only tracing
// wiring in core/tracing without it being public API. core/tracing installs
// the hooks from its init; because this package lives under go/internal, code
// outside the Genkit module cannot import it.
package tracingbridge

// SetDevTelemetryServer points the Developer UI's trace export at the
// telemetry server at url. An empty url is ignored, and repeated calls with
// the same url are no-ops; a new url retargets export without disturbing spans
// that are still open. Installed by core/tracing's init.
var SetDevTelemetryServer func(url string)
