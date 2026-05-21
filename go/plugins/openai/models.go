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

package openai

import (
	"context"
	"strings"

	"github.com/firebase/genkit/go/ai"
	oa "github.com/openai/openai-go"
)

var defaultOpenAIOpts = ai.ModelOptions{
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

// listModels returns model names that can be used for generation.
func listModels(ctx context.Context, client *oa.Client) ([]string, error) {
	iter := client.Models.ListAutoPaging(ctx)
	models := []string{}
	for iter.Next() {
		name := iter.Current().ID
		if isGenerativeModel(name) {
			models = append(models, name)
		}
	}
	if err := iter.Err(); err != nil {
		return nil, err
	}
	return models, nil
}

func isGenerativeModel(name string) bool {
	nonGenerative := []string{
		"embedding",
		"moderation",
		"whisper",
		"tts",
		"dall-e",
		"image",
		"transcribe",
		"realtime",
	}
	for _, part := range nonGenerative {
		if strings.Contains(name, part) {
			return false
		}
	}
	return true
}
