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

package pgengine

import "testing"

func TestIAMUser(t *testing.T) {
	for email, want := range map[string]string{
		"runner@my-project.iam.gserviceaccount.com": "runner@my-project.iam",
		"someone@example.com":                       "someone@example.com",
	} {
		if got := IAMUser(email); got != want {
			t.Errorf("IAMUser(%q) = %q, want %q", email, got, want)
		}
	}
}

func TestPoolConfig(t *testing.T) {
	// pgx reads a password from the environment; the config must not keep it.
	t.Setenv("PGPASSWORD", "from-the-environment")

	// Credentials reach the connection config verbatim, whatever they contain.
	user, password, database := "app user", `p a's\"s=word`, "my db"
	config, err := PoolConfig(user, password, database, false)
	if err != nil {
		t.Fatalf("PoolConfig: %v", err)
	}
	if got := config.ConnConfig; got.User != user || got.Password != password || got.Database != database {
		t.Errorf("config has user %q, password %q, database %q; want %q, %q, %q",
			got.User, got.Password, got.Database, user, password, database)
	}
	if config.ConnConfig.TLSConfig != nil {
		t.Error("config enables TLS, but the connector encrypts the connection itself")
	}

	// An IAM login sends no password: the connector authenticates it.
	iam, err := PoolConfig("runner@my-project.iam", "unused", "db", true)
	if err != nil {
		t.Fatalf("PoolConfig: %v", err)
	}
	if iam.ConnConfig.Password != "" {
		t.Errorf("IAM config has password %q, want none", iam.ConnConfig.Password)
	}
}
