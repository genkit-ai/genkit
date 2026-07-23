// Copyright 2024 Google LLC
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

package firebase

import (
	"context"
	"testing"

	firebasev4 "firebase.google.com/go/v4"
)

/*
  - Pre-requisites to run this test:
  - 1. Create a Firebase project and enable Firestore (https://console.firebase.google.com/).
  - 2. Authenticate locally: `gcloud auth application-default login`.
  - 3. Set the FIREBASE_PROJECT_ID environment variable to your project ID.
*/
func TestInit(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	firebaseApp, _ := firebasev4.NewApp(ctx, nil)

	tests := []struct {
		name      string
		projectId string
		app       *firebasev4.App
	}{
		{
			name:      "Successful initialization with project id",
			projectId: "test-app",
			app:       nil,
		},
		{
			name:      "Successful initialization with app",
			projectId: "",
			app:       firebaseApp,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &Firebase{
				ProjectId: tt.projectId,
				App:       tt.app,
			}
			f.Init(ctx)
		})
	}
}
