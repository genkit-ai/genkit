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
	"errors"
	"fmt"
	"net"

	"cloud.google.com/go/alloydbconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/oauth2/v2"
	"google.golang.org/api/option"

	"github.com/firebase/genkit/go/plugins/internal/pgengine"
)

// IpType type of IP address, public or private
type IpType string

const (
	PUBLIC  IpType = "PUBLIC"
	PRIVATE IpType = "PRIVATE"
)

// PostgresEngine postgres engine
type PostgresEngine struct {
	Pool *pgxpool.Pool
}

// NewPostgresEngine creates a new Postgres Engine.
func NewPostgresEngine(ctx context.Context, opts ...Option) (*PostgresEngine, error) {
	pgEngine := new(PostgresEngine)
	cfg, err := applyEngineOptions(opts)
	if err != nil {
		return nil, err
	}
	if cfg.connPool == nil {
		user, usingIAMAuth, err := getUser(ctx, cfg)
		if err != nil {
			// If no user can be determined, return an error.
			return nil, fmt.Errorf("unable to retrieve a valid username. Err: %w", err)
		}
		if usingIAMAuth {
			cfg.user = user
		}
		cfg.connPool, err = createPool(ctx, cfg, usingIAMAuth)
		if err != nil {
			return nil, err
		}
	}

	if err := cfg.connPool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("failed to connect with database %v", err)
	}

	pgEngine.Pool = cfg.connPool
	return pgEngine, nil
}

func (pgEngine *PostgresEngine) GetClient() *pgxpool.Pool {
	return pgEngine.Pool
}

func applyEngineOptions(opts []Option) (engineConfig, error) {
	cfg := &engineConfig{
		ipType:     PUBLIC,
		userAgents: defaultUserAgent,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	if cfg.connPool == nil && (cfg.projectID == "" || cfg.region == "" || cfg.instance == "") {
		return engineConfig{}, errors.New("missing connection: provide a connection pool or db instance fields")
	}
	if cfg.database == "" {
		return engineConfig{}, errors.New("missing database field")
	}

	return *cfg, nil
}

// getUser retrieves the username, a flag indicating if IAM authentication will be used and an error.
func getUser(ctx context.Context, config engineConfig) (string, bool, error) {
	if config.user != "" && config.password != "" {
		// If both username and password are provided use provided username.
		return config.user, false, nil
	}
	if config.iamAccountEmail != "" {
		// If iamAccountEmail is provided use it as user.
		return pgengine.IAMUser(config.iamAccountEmail), true, nil
	}
	// If neither user and password nor iamAccountEmail are provided,
	// retrieve IAM email from the environment.
	serviceAccountEmail, err := getServiceAccountEmail(ctx)
	if err != nil {
		return "", false, fmt.Errorf("unable to retrieve service account email: %w", err)
	}
	return pgengine.IAMUser(serviceAccountEmail), true, nil

}

// getServiceAccountEmail retrieves the IAM principal email with users account.
func getServiceAccountEmail(ctx context.Context) (string, error) {
	scopes := []string{"https://www.googleapis.com/auth/userinfo.email"}
	// Get credentials using email scope
	credentials, err := google.FindDefaultCredentials(ctx, scopes...)
	if err != nil {
		return "", fmt.Errorf("unable to get default credentials: %w", err)
	}

	// Verify valid TokenSource.
	if credentials.TokenSource == nil {
		return "", fmt.Errorf("missing or invalid credentials")
	}

	oauth2Service, err := oauth2.NewService(ctx, option.WithTokenSource(credentials.TokenSource))
	if err != nil {
		return "", fmt.Errorf("failed to create new service: %w", err)
	}

	// Fetch IAM principal email.
	userInfo, err := oauth2Service.Userinfo.Get().Do()
	if err != nil {
		return "", fmt.Errorf("failed to get user info: %w", err)
	}
	return userInfo.Email, nil
}

// createPool creates a connection pool to the PostgreSQL database.
func createPool(ctx context.Context, cfg engineConfig, usingIAMAuth bool) (*pgxpool.Pool, error) {
	config, err := pgengine.PoolConfig(cfg.user, cfg.password, cfg.database, usingIAMAuth)
	if err != nil {
		return nil, err
	}
	dialeropts := []alloydbconn.Option{alloydbconn.WithUserAgent(cfg.userAgents)}
	if usingIAMAuth {
		dialeropts = append(dialeropts, alloydbconn.WithIAMAuthN())
	}
	d, err := alloydbconn.NewDialer(ctx, dialeropts...)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize connection: %w", err)
	}
	instanceURI := fmt.Sprintf("projects/%s/locations/%s/clusters/%s/instances/%s", cfg.projectID, cfg.region, cfg.cluster, cfg.instance)
	config.ConnConfig.DialFunc = func(ctx context.Context, _ string, _ string) (net.Conn, error) {
		if cfg.ipType == PRIVATE {
			return d.Dial(ctx, instanceURI, alloydbconn.WithPrivateIP())
		}
		return d.Dial(ctx, instanceURI, alloydbconn.WithPublicIP())
	}
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("unable to create connection pool: %w", err)
	}
	return pool, nil
}

// Close closes the pool connection.
func (pgEngine *PostgresEngine) Close() {
	if pgEngine.Pool != nil {
		pgEngine.Pool.Close()
	}
}

type Column struct {
	Name     string
	DataType string
	Nullable bool
}

type VectorstoreTableOptions struct {
	TableName          string
	VectorSize         int
	SchemaName         string
	ContentColumnName  string
	EmbeddingColumn    string
	MetadataJSONColumn string
	IDColumn           Column
	MetadataColumns    []Column
	OverwriteExisting  bool
	StoreMetadata      bool
}

// validateVectorstoreTableOptions initializes the options struct with the default values for
// the InitVectorstoreTable function.
func validateVectorstoreTableOptions(opts *VectorstoreTableOptions) error {
	if opts.TableName == "" {
		return fmt.Errorf("missing table name in options")
	}
	if opts.VectorSize == 0 {
		return fmt.Errorf("missing vector size in options")
	}

	if opts.SchemaName == "" {
		opts.SchemaName = "public"
	}

	if opts.ContentColumnName == "" {
		opts.ContentColumnName = "content"
	}

	if opts.EmbeddingColumn == "" {
		opts.EmbeddingColumn = "embedding"
	}

	if opts.MetadataJSONColumn == "" {
		opts.MetadataJSONColumn = "langchain_metadata"
	}

	if opts.IDColumn.Name == "" {
		opts.IDColumn.Name = "langchain_id"
	}

	if opts.IDColumn.DataType == "" {
		opts.IDColumn.DataType = "UUID"
	}

	return nil
}

// initVectorstoreTable creates a table for saving of vectors to be used with PostgresVectorStore.
func (pgEngine *PostgresEngine) InitVectorstoreTable(ctx context.Context, opts VectorstoreTableOptions) error {
	err := validateVectorstoreTableOptions(&opts)
	if err != nil {
		return fmt.Errorf("failed to validate vectorstore table options: %w", err)
	}

	// Ensure the vector extension exists
	_, err = pgEngine.Pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	if err != nil {
		return fmt.Errorf("failed to create extension: %w", err)
	}

	// Drop table if exists and overwrite flag is true
	if opts.OverwriteExisting {
		_, err = pgEngine.Pool.Exec(ctx, fmt.Sprintf(`DROP TABLE IF EXISTS "%s"."%s"`, opts.SchemaName, opts.TableName))
		if err != nil {
			return fmt.Errorf("failed to drop table: %w", err)
		}
	}

	// Build the SQL query that creates the table
	query := fmt.Sprintf(`CREATE TABLE "%s"."%s" (
		"%s" %s PRIMARY KEY,
		"%s" TEXT NOT NULL,
		"%s" vector(%d) NOT NULL`, opts.SchemaName, opts.TableName, opts.IDColumn.Name, opts.IDColumn.DataType, opts.ContentColumnName, opts.EmbeddingColumn, opts.VectorSize)

	// Add metadata columns  to the query string if provided
	for _, column := range opts.MetadataColumns {
		nullable := ""
		if !column.Nullable {
			nullable = "NOT NULL"
		}
		query += fmt.Sprintf(`, "%s" %s %s`, column.Name, column.DataType, nullable)
	}

	// Add JSON metadata column to the query string if storeMetadata is true
	if opts.StoreMetadata {
		query += fmt.Sprintf(`, "%s" JSON`, opts.MetadataJSONColumn)
	}
	// Close the query string
	query += ");"

	// Execute the query to create the table
	_, err = pgEngine.Pool.Exec(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to create table: %w", err)
	}

	return nil
}
