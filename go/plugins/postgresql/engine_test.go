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
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

func TestApplyEngineOptionsConfig(t *testing.T) {

	testCases := []struct {
		name       string
		opts       []Option
		wantErr    bool
		wantIpType IpType
	}{
		{
			name: "valid config with connection pool",
			opts: []Option{
				WithPool(&pgxpool.Pool{}),
				WithDatabase("testdb"),
			},
			wantErr:    false,
			wantIpType: PUBLIC,
		},
		{
			name: "connection pool without database",
			opts: []Option{
				WithPool(&pgxpool.Pool{}),
			},
			wantErr:    false,
			wantIpType: PUBLIC,
		},
		{
			name: "valid config with instance details",
			opts: []Option{
				WithCloudSQLInstance("testproject", "testregion", "testinstance"),
				WithDatabase("testdb"),
			},
			wantErr:    false,
			wantIpType: PUBLIC,
		},
		{
			name: "missing database",
			opts: []Option{
				WithCloudSQLInstance("testproject", "testregion", "testinstance"),
			},
			wantErr:    true,
			wantIpType: PUBLIC,
		},
		{
			name: "missing all connection details",
			opts: []Option{
				WithDatabase("testdb"),
			},
			wantErr:    true,
			wantIpType: PUBLIC,
		},
		{
			name: "ip type private",
			opts: []Option{
				WithCloudSQLInstance("testproject", "testregion", "testinstance"),
				WithDatabase("testdb"),
				WithIPType(PRIVATE),
			},
			wantErr:    false,
			wantIpType: PRIVATE,
		},
		{
			name: "custom EmailRetriever",
			opts: []Option{
				WithCloudSQLInstance("testproject", "testregion", "testinstance"),
				WithDatabase("testdb"),
			},
			wantErr:    false,
			wantIpType: PUBLIC,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := applyEngineOptions(tc.opts)
			if tc.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if got, want := cfg.ipType, tc.wantIpType; got != want {
					t.Errorf("ipType = %v, want %v", got, want)
				}
			}
		})
	}
}

func TestGetUser(t *testing.T) {
	testCases := []struct {
		name        string
		cfg         engineConfig
		wantUser    string
		wantIAMAuth bool
		wantErr     bool
	}{
		{
			name: "user and password provided",
			cfg: engineConfig{
				user:     "testuser",
				password: "testpassword",
			},
			wantUser:    "testuser",
			wantIAMAuth: false,
			wantErr:     false,
		},
		{
			name: "iam account email provided",
			cfg: engineConfig{
				iamAccountEmail: "iam@example.com",
			},
			wantUser:    "iam@example.com",
			wantIAMAuth: true,
			wantErr:     false,
		},
		{
			// Cloud SQL names a service account's database user without the
			// domain suffix; the full email fails to log in.
			name: "service account email drops its domain suffix",
			cfg: engineConfig{
				iamAccountEmail: "runner@my-project.iam.gserviceaccount.com",
			},
			wantUser:    "runner@my-project.iam",
			wantIAMAuth: true,
			wantErr:     false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			user, iamAuth, err := getUser(ctx, tc.cfg)
			if tc.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if got, want := user, tc.wantUser; got != want {
					t.Errorf("user = %q, want %q", got, want)
				}
				if got, want := iamAuth, tc.wantIAMAuth; got != want {
					t.Errorf("iamAuth = %v, want %v", got, want)
				}
			}
		})
	}
}

func TestPoolConfig(t *testing.T) {
	// pgx reads a password from the environment; the config must not keep it.
	t.Setenv("PGPASSWORD", "from-the-environment")

	// Credentials reach the connection config verbatim, whatever they contain.
	cfg := engineConfig{user: "app user", password: `p a's\"s=word`, database: "my db"}
	config, err := poolConfig(cfg, false)
	if err != nil {
		t.Fatalf("poolConfig: %v", err)
	}
	if got := config.ConnConfig; got.User != cfg.user || got.Password != cfg.password || got.Database != cfg.database {
		t.Errorf("config has user %q, password %q, database %q; want %q, %q, %q",
			got.User, got.Password, got.Database, cfg.user, cfg.password, cfg.database)
	}
	if config.ConnConfig.TLSConfig != nil {
		t.Error("config enables TLS, but the Cloud SQL connector encrypts the connection itself")
	}

	// An IAM login sends no password: the connector authenticates it.
	iam, err := poolConfig(engineConfig{user: "runner@my-project.iam", password: "unused", database: "db"}, true)
	if err != nil {
		t.Fatalf("poolConfig: %v", err)
	}
	if iam.ConnConfig.Password != "" {
		t.Errorf("IAM config has password %q, want none", iam.ConnConfig.Password)
	}
}

// TestCloseLeavesCallerPool needs a PostgreSQL server: set
// GENKIT_TEST_POSTGRES_URL to a connection URL for a database the test may use.
func TestCloseLeavesCallerPool(t *testing.T) {
	url := os.Getenv("GENKIT_TEST_POSTGRES_URL")
	if url == "" {
		t.Skip("Skipping: GENKIT_TEST_POSTGRES_URL not set")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	engine, err := NewPostgresEngine(ctx, WithPool(pool))
	if err != nil {
		t.Fatalf("NewPostgresEngine: %v", err)
	}
	engine.Close()
	if err := pool.Ping(ctx); err != nil {
		t.Errorf("the engine closed the pool its caller gave it: %v", err)
	}
}
