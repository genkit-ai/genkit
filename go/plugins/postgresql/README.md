# PostgreSQL plugin

PostgreSQL plugin provides indexer and retriever implementations that use PostgreSQL with the pgvector extension for vector similarity search.

> **Building with a coding agent? Install the Genkit Go skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-go
> ```
>
> It teaches your agent the current Genkit Go APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

## Configuration

To use this plugin, first create a `PostgresEngine` instance:

```go
// Create PostgresEngine instance
// with basic authentication
pEngine, err := NewPostgresEngine(ctx,
		WithUser('user'),
		WithPassword('password'),
		WithCloudSQLInstance('my-project', 'us-central1', 'my-instance'),
		WithDatabase('my-database')

// with email authentication
pEngine, err := NewPostgresEngine(ctx,
    WithCloudSQLInstance('my-project', 'us-central1', 'my-instance'),
    WithDatabase('my-database'),
    WithIAMAccountEmail('mail@company.com'))

// with custom pool
pool, err := pgxpool.New(ctx, "add_your_connection_string")
if err != nil {
  return err
}

pEngine, err := NewPostgresEngine(ctx,
    WithDatabase("db_test"),
    WithPool(pool))

// Create the vector store table

err = pEngine.InitVectorstoreTable(ctx, VectorstoreTableOptions{
    TableName:          "documents",
    VectorSize:         768,
    SchemaName:         "public",
    ContentColumnName:  "content",
    EmbeddingColumn:    "embedding",
    MetadataJSONColumn: "custom_metadata",
    IDColumn: Column{
      Name:     custom_id,
      Nullable: false,
    },
    MetadataColumns: []Column{
      {
        Name:     "source",
        DataType: "text",
        Nullable: true,
      },
      {
        Name:     "category",
        DataType: "text",
        Nullable: true,
      },
    },
    OverwriteExisting: true,
    StoreMetadata:     true,
})
```

Then, specify the plugin when you initialize Genkit:

```go
	postgres := &Postgres{
		engine: pEngine,
	}

	g, err := genkit.Init(ctx, genkit.WithPlugins(postgres))
  if err != nil {
    return nil
}

// To use the table you configured when you loaded the plugin:

  cfg := &Config{
    TableName:             'documents',
    SchemaName:            'public',
    ContentColumn:         "content",
    EmbeddingColumn:       "embedding",
    MetadataColumns:       []string{"source", "category"},
    IDColumn:              "custom_id",
    MetadataJSONColumn:    "custom_metadata",
    Embedder:              embedder,
    EmbedderOptions:       nil,
  }

// To index and retrieve from the configured table:
doc, retriever, err := DefineRetriever(ctx, g, postgres, cfg)
if err != nil {
  retrun err
}

docs := []*ai.Document{{
  Content: []*ai.Part{{
    Kind:        ai.PartText,
    ContentType: "text/plain",
    Text:        "The product features include...",
  }},
  Metadata: map[string]any{"source": "website", "category": "product-docs", "custom_id": "doc-123"},
}}

if err := doc.Index(ctx, docs); err != nil {
 return err
}

d2 := ai.DocumentFromText( "The product features include..." , nil)

resp, err := retriever.Retrieve(ctx, &ai.RetrieverRequest{
    Query: d2,
    k:5,
    filter: "source='website' AND category='product-docs'"
})

if err != nil {
    retrun err
}

```
See the [Retrieval-augmented generation](../rag.md) page for a general discussion on indexers and retrievers.
