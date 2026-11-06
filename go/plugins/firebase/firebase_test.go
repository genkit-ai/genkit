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
	"net"
	"strings"
	"testing"

	firebasev4 "firebase.google.com/go/v4"
	"github.com/firebase/genkit/go/internal"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

/*
  - Pre-requisites to run this test:

Same as that in retriever_test.go
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

func TestDefaultAppOptionsSetUserAgent(t *testing.T) {
	// With the emulator, Firestore dials its own connection and ignores the
	// user agent option.
	t.Setenv("FIRESTORE_EMULATOR_HOST", "")

	// A local gRPC server that records request metadata and fails the call.
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	mdc := make(chan metadata.MD, 1)
	srv := grpc.NewServer(grpc.UnknownServiceHandler(func(_ any, stream grpc.ServerStream) error {
		md, _ := metadata.FromIncomingContext(stream.Context())
		select {
		case mdc <- md:
		default:
		}
		return status.Error(codes.PermissionDenied, "recorded by test")
	}))
	go srv.Serve(lis)
	t.Cleanup(srv.Stop)

	ctx := context.Background()
	opts := append(defaultAppOptions(),
		option.WithEndpoint(lis.Addr().String()),
		option.WithoutAuthentication(),
		option.WithGRPCDialOption(grpc.WithTransportCredentials(insecure.NewCredentials())),
	)
	app, err := firebasev4.NewApp(ctx, &firebasev4.Config{ProjectID: "test-project"}, opts...)
	if err != nil {
		t.Fatal(err)
	}
	client, err := app.Firestore(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { client.Close() })

	if _, err := client.Collection("c").Doc("d").Get(ctx); status.Code(err) != codes.PermissionDenied {
		t.Fatalf("Get() error = %v, want PermissionDenied from the test server", err)
	}
	md := <-mdc
	// gRPC appends its own token, e.g. "genkit-go/1.2.3 grpc-go/1.73.0".
	want := "genkit-go/" + internal.Version + " "
	if ua := md.Get("user-agent"); len(ua) != 1 || !strings.HasPrefix(ua[0], want) {
		t.Errorf("user-agent = %q, want prefix %q", ua, want)
	}
}
