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

package anthropic

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
)

var defaultClaudeOpts = ai.ModelOptions{
	Supports: &ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		ToolChoice:  true,
		SystemRole:  true,
		Media:       true,
		Constrained: ai.ConstrainedSupportAll,
	},
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// listModels returns a list of model names supported by the Anthropic client
func listModels(ctx context.Context, client *anthropic.Client) ([]string, error) {
	iter := client.Models.ListAutoPaging(ctx, anthropic.ModelListParams{})
	models := []string{}

	for iter.Next() {
		m := iter.Current()
		models = append(models, m.ID)
	}

	if err := iter.Err(); err != nil {
		return nil, err
	}

	return models, nil
}
