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

package base

import "context"

// FailedModelResponse builds the record of a model call that failed without a
// response: an *ai.ModelResponse for req (an *ai.ModelRequest) and err, as a
// failed Generate reports it. Package ai, which owns those types, sets it
// when it loads; it is type-erased for the same reason as [WithPromptState].
var FailedModelResponse func(ctx context.Context, req any, err error) any
