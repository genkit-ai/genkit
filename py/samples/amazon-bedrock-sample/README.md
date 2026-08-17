# Amazon Bedrock

Run text generation, streaming, structured output, tool calling, reasoning,
prompt caching, image and document input, embeddings, image generation, and
reranking through Genkit with Amazon Bedrock's Converse, ConverseStream, and
InvokeModel APIs.

You need an AWS account with Amazon Bedrock model access granted for the models
the CLI smoke run below uses:

- `us.amazon.nova-lite-v1:0`
- `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- `amazon.titan-embed-text-v2:0`

The remaining models need one grant each, and only fail when you run their flow:

- `us.meta.llama3-3-70b-instruct-v1:0` for `cat_profile`
- `us.deepseek.r1-v1:0` for `reasoning`
- `amazon.titan-embed-image-v1` for `embed_image`
- `cohere.embed-english-v3` for `embed_batch` with `embedder` set to Cohere
- `amazon.nova-2-multimodal-embeddings-v1:0` is offered in `us-east-1` only, and
  fails everywhere else however you invoke it
- `stability.sd3-5-large-v1:0` for `generate_image`, offered in `us-west-2` only
- `cohere.rerank-v3-5:0` for `rerank`; `amazon.rerank-v1:0` works too, but is
  not offered in `us-east-1`

`describe_image`, `summarize_pdf` and `prompt_caching` need no grant beyond the
smoke-run list, since they run on models already on it: Nova Lite for
`describe_image`, and Claude for the other two.

The Anthropic model additionally needs the account's one-time use-case
agreement (Bedrock console, Model access, Anthropic use case details); the
`thinking`, `thinking_stream`, `summarize_pdf` and `prompt_caching` flows fail
with `ResourceNotFoundException` until it is granted.

Credentials come from the standard AWS chain; environment variables, an
`AWS_PROFILE` (including SSO profiles after `aws sso login`), or instance
credentials. A region is required, from `AWS_REGION`, `AWS_DEFAULT_REGION`, or
the active profile:

```bash
export AWS_PROFILE=my-profile
export AWS_REGION=us-west-2
```

`us-west-2` runs the most flows: it is the only region with active
text-to-image models, and it has both rerank models. The one flow it cannot run
is `embed_batch` switched to `amazon.nova-2-multimodal-embeddings-v1:0`, which
is offered in `us-east-1` only. In `us-east-1` the trade goes the other way:
`generate_image` fails with `ValidationException: The provided model identifier
is invalid`, which is Bedrock's way of saying the model is not offered in the
calling region, and `amazon.rerank-v1:0` is unavailable, though
`cohere.rerank-v3-5:0` works. A region set in `~/.aws/config` counts, so a
`us-east-1` profile default quietly wins when no `AWS_REGION` is exported.

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
- `describe_image`
- `summarize_pdf`
- `prompt_caching`
- `generate_image`
- `rerank`

The `*_stream` flows go through ConverseStream. Chunks are deltas rather than
snapshots, so the Dev UI's streamed output is the concatenation of them; a tool
call is the exception, arriving whole in one chunk because its arguments come
over the wire as JSON fragments.

The plugin resolves any Bedrock model ID, inference profile, or ARN on demand,
so the Dev UI model runner also works with models beyond the five declared
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

## Vision and documents

Media rides along on a Converse prompt as a data URL. `describe_image` attaches
a PNG and asks a question about it, and `summarize_pdf` does the same with a
PDF. Remote `http(s)` URLs are refused rather than fetched, so the bytes have to
be in the request; the MIME type is read off the data URL when the media part
carries none of its own.

The MIME type is what decides the rest. A document MIME type becomes a Converse
`document` block rather than an `image` block, so attaching a CSV or a
spreadsheet is this same code path: the plugin accepts pdf, csv, doc, docx, xls,
xlsx, html, txt and md. Bedrock parses the document server-side, and wants
accompanying text in the message, which the question supplies.

Support for either kind of attachment is per model, not a plugin feature, which
is why `summarize_pdf` asks Claude rather than the Nova Lite default that
`describe_image` uses.

Both assets are inline constants in the sample, so there are no asset files to
keep around: a 64x48 PNG of three horizontal bands, and a one-page PDF holding
three lines of text.

## Prompt caching

`prompt_caching` puts two unrelated questions to the same large system prompt.
The cache point goes after the static prefix it should cache, and that prefix
has to be byte-identical between calls to hit, which is why the support manual
is a module-level constant rather than something the flow rebuilds per call.

Only cache reads surface, because the plugin deliberately drops the write
counter, so the evidence is the second call reporting `cached_content_tokens`
that the first did not. Bedrock counts cached tokens separately from
`inputTokens` rather than inside it, so the flow reports the uncached remainder
under its own name and the manual only shows up in the total. A prefix below
the model's minimum, roughly 1,000 tokens for Claude Sonnet, is silently not
cached rather than rejected, which is why the sample's manual is padded out to
about 2,000 tokens. Re-running inside the
cache's few-minute lifetime can show a read on the first call too, so the flow
reports what the second call did and claims nothing about the first.

## Image generation

Image models go through InvokeModel, but they are ordinary Genkit model actions,
so `generate_image` gets its result back as a media part holding a data URL. The
flow returns a size summary rather than the base64 itself, which is unreadable
in the Dev UI, and the image is visible on the model action in the Dev UI trace.

The two image families take incompatible request bodies and the plugin routes on
the model ID, so the flow's `aspect_ratio` and `output_format` inputs are
Stability fields. The Amazon family, Nova Canvas and Titan Image, reads only
`imageGenerationConfig` and silently ignores them. Image models never stream.

## Reranking

Reranking is a method on the plugin instance rather than a registered Genkit
action, because Genkit Python has no reranker primitive to register against.
That is why the sample keeps a reference to the `Bedrock` it passed to `Genkit`;
`rerank` is the only flow that needs it.

Results come back in the service's own descending-relevance order and are
neither re-sorted nor truncated locally, so what the flow prints is what Bedrock
returned. Each ranked document carries the input document's content verbatim
with a fresh score, and the input document's own metadata is dropped. A `top_n`
of 0 or less means all of the documents.
