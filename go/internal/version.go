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

package internal

import (
	"runtime/debug"
	"strings"
)

// modulePath is this module's path, as it appears in the go.mod of programs
// that depend on Genkit.
const modulePath = "github.com/firebase/genkit/go"

// Version is the version of the Genkit module the running program was built
// against, without the "v" prefix (e.g. "1.11.0"). It is read from the
// binary's embedded build info, so it tracks the version in the program's
// go.mod rather than a hardcoded value. It is "dev" when Genkit is built from
// source: from within this repository or through a directory replace
// directive.
var Version = versionFromBuildInfo(buildInfo())

const GENKIT_REFLECTION_API_SPEC_VERSION = 1

// buildInfo returns the running binary's build info, or nil if the binary
// was built without module support.
func buildInfo() *debug.BuildInfo {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return nil
	}
	return info
}

// versionFromBuildInfo extracts the Genkit module's version from build info.
// Genkit normally appears among the dependencies with the version its go.mod
// requires; when it is the main module instead (tests and samples in this
// repository), or when a replace directive points at a source checkout, there
// is no release version to report and the version is "dev".
func versionFromBuildInfo(info *debug.BuildInfo) string {
	if info == nil {
		return "dev"
	}
	for _, dep := range info.Deps {
		if dep.Path != modulePath {
			continue
		}
		m := dep
		if dep.Replace != nil {
			// A module replacement carries the replacement's version; a
			// directory replacement has none.
			m = dep.Replace
		}
		if v, ok := strings.CutPrefix(m.Version, "v"); ok {
			return v
		}
		return "dev"
	}
	return "dev"
}
