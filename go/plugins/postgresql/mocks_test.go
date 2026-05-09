// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package postgresql

import (
	"context"
	"errors"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
)

type mockEmbedderFail struct{}

func (m mockEmbedderFail) Name() string { return "mock" }
func (m mockEmbedderFail) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	return nil, errors.New("mock fail")
}
func (m mockEmbedderFail) Register(r api.Registry) {}
