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

package alloydb

import (
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestRetriever_Fail_WrongTypeOfOption(t *testing.T) {
	ds := DocStore{}
	res, err := ds.Retrieve(context.Background(), &ai.RetrieverRequest{Options: struct{}{}})
	if res != nil {
		t.Errorf("Retrieve() res = %v, want nil", res)
	}
	if err == nil {
		t.Fatal("Retrieve() expected error, got nil")
	}
	if want := "postgres.Retrieve options have type"; !strings.Contains(err.Error(), want) {
		t.Errorf("Retrieve() error = %q, want it to contain %q", err.Error(), want)
	}
}

func TestRetriever_Fail_EmbedReturnError(t *testing.T) {
	ds := DocStore{
		config: &Config{Embedder: mockEmbedderFail{}},
	}
	res, err := ds.Retrieve(context.Background(), &ai.RetrieverRequest{})
	if res != nil {
		t.Fatalf("Retrieve() res = %v, want nil", res)
	}
	if err == nil {
		t.Fatal("Retrieve() expected error, got nil")
	}
}
