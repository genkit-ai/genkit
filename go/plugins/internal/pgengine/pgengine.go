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

// Package pgengine holds the connection setup the postgresql and alloydb
// plugins share, so their engines connect the same way.
package pgengine

import (
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

// IAMUser returns the database user name Cloud SQL and AlloyDB give an IAM
// principal: a service account's email without its ".gserviceaccount.com"
// suffix, and any other email unchanged.
func IAMUser(email string) string {
	return strings.TrimSuffix(email, ".gserviceaccount.com")
}

// PoolConfig returns the pool configuration for a connection that a Cloud SQL
// or AlloyDB connector dials. The credentials go into the config's fields
// rather than a connection string, so a value with spaces or quotes reaches the
// server intact. TLS is off at this layer because the connector encrypts the
// connection itself, and an IAM login sends no password because the connector
// authenticates it.
func PoolConfig(user, password, database string, iam bool) (*pgxpool.Config, error) {
	config, err := pgxpool.ParseConfig("sslmode=disable")
	if err != nil {
		return nil, fmt.Errorf("failed to parse connection config: %w", err)
	}
	config.ConnConfig.User = user
	config.ConnConfig.Database = database
	// Set the password even when it is empty: ParseConfig fills it from
	// PGPASSWORD or a .pgpass file.
	config.ConnConfig.Password = password
	if iam {
		config.ConnConfig.Password = ""
	}
	return config, nil
}
