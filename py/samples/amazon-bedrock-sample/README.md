# Amazon Bedrock

Run text generation, streaming, structured output, tool calling, reasoning,
and embeddings through Genkit with Amazon Bedrock's Converse, ConverseStream,
and InvokeModel APIs.

You need an AWS account with Amazon Bedrock model access granted for the models
the sample runs on startup:

- `us.amazon.nova-lite-v1:0`
- `us.meta.llama3-3-70b-instruct-v1:0`
- `us.deepseek.r1-v1:0`
- `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- `amazon.titan-embed-text-v2:0`

The other embedders need one grant each, and only fail when you run their flow:

- `amazon.titan-embed-image-v1` for `embed_image`
- `cohere.embed-english-v3` for `embed_batch` with `embedder` set to Cohere
- `amazon.nova-2-multimodal-embeddings-v1:0` is offered in `us-east-1` only, and
  fails everywhere else however you invoke it

The Anthropic model additionally needs the account's one-time use-case
agreement (Bedrock console, Model access, Anthropic use case details); the
`thinking` flow fails with `ResourceNotFoundException` until it is granted.

Credentials come from the standard AWS chain; environment variables, an
`AWS_PROFILE` (including SSO profiles after `aws sso login`), or instance
credentials. A region is required, from `AWS_REGION`, `AWS_DEFAULT_REGION`, or
the active profile:

```bash
export AWS_PROFILE=my-profile
export AWS_REGION=us-east-1
```

Run the quick smoke test:

```bash
uv sync
uv run src/main.py
```

To explore all flows in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Then open [http://localhost:4000](http://localhost:4000) and try:

- `haiku`
- `haiku_stream`
- `cat_profile`
- `weather_report`
- `weather_report_stream`
- `reasoning`
- `thinking`
- `thinking_stream`
- `embed_text`
- `embed_batch`
- `embed_similarity`
- `embed_image`

The `*_stream` flows go through ConverseStream. Chunks are deltas rather than
snapshots, so the Dev UI's streamed output is the concatenation of them; a tool
call is the exception, arriving whole in one chunk because its arguments come
over the wire as JSON fragments.

The plugin resolves any Bedrock model ID, inference profile, or ARN on demand,
so the Dev UI model runner also works with models beyond the four declared
ones.

Bedrock has no constrained-decoding mode, so structured output is carried by
prompt instructions: pass `output_instructions=True` alongside `output_format`
and `output_schema`, as `cat_profile` does. Without it the schema never reaches
the model and it answers in prose.

## Embedders

Embedding models go through InvokeModel, not Converse, and each family has its
own request shape. Four flows cover them:

- `embed_text` embeds one document with Titan Text v2 and reports the vector's
  1024 dimensions, its first few values, and its magnitude (~1.0, since Titan
  v2 normalizes).
- `embed_batch` embeds several documents in one `embed_many` call. The last
  default text repeats the first, so `repeated_text_pairs` showing a similarity
  of 1.0 proves the vectors came back aligned to their inputs. Titan has no
  batch API and is fanned out into concurrent calls; set `embedder` to
  `cohere.embed-english-v3` to send up to 96 texts in a single call instead, or
  to `amazon.nova-2-multimodal-embeddings-v1:0` (us-east-1 only) for 3072
  dimensions.
- `embed_similarity` ranks candidate texts against a query by cosine
  similarity, so the output shows the embeddings are meaningful and not merely
  well shaped. It stays on Titan because Cohere requests are pinned to
  `input_type: search_document` for now, which is the wrong side of that
  asymmetry for a query.
- `embed_image` puts a caption and an inline 1x1 PNG in one document and gets a
  single joint 1024-float vector from Titan Multimodal. It embeds the caption on
  its own too: the similarity below 1.0 is the evidence the image reached the
  model. Titan accepts PNG and JPEG only, one image per document, as a data URL.

Cohere embedding is text-only on Bedrock, so there is no Cohere image flow; use
Titan Multimodal for images.

Declared embedders also appear in the Dev UI's Embedders section, where they can
be invoked directly with a raw request — useful for a model no flow covers:

```json
{"input": [{"content": [{"text": "a red apple"}]}]}
```

The response is the raw vector, which is why the flows above summarize instead.
Declaring an embedder costs nothing at startup: no call is made until a flow or
the Dev UI runs one.
