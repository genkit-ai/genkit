# Vertex AI Image Generation (Imagen)

Demonstrates image generation using Vertex AI Imagen models.

> **Note:** Imagen models are only available through Vertex AI, not the Gemini API.
> For Gemini API image generation, see the `google-genai-image` sample.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Image Generation | `draw_image_with_imagen` | Generate images from text prompts |
| Imagen Config | `GenerateImagesConfig` | Control output (negative prompt, seed, etc.) |
| Safety Filters | `safety_filter_level` | Content safety controls |
| Aspect Ratio | `aspect_ratio` | Control image dimensions |
| Watermarking | `add_watermark` | Add watermark to generated images |

## Quick Start

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
./run.sh
```

That's it! The script will:

1. ✓ Prompt for your project ID if not set
2. ✓ Check gcloud authentication (and help you authenticate if needed)
3. ✓ Enable Vertex AI API (with your permission)
4. ✓ Install dependencies
5. ✓ Start the demo and open your browser

## Manual Setup (if needed)

If you prefer manual setup or the automatic setup fails:

### 1. Install gcloud CLI

Download from: https://cloud.google.com/sdk/docs/install

### 2. Authentication

```bash
gcloud auth application-default login
```

### 3. Enable Vertex AI API

```bash
gcloud services enable aiplatform.googleapis.com --project=$GOOGLE_CLOUD_PROJECT
```

### 4. Run the Demo

```bash
./run.sh
```

Or manually:

```bash
genkit start -- uv run src/main.py
```

Then open the Dev UI at http://localhost:4000

## Testing the Demo

1. Open DevUI at http://localhost:4000
2. Run `draw_image_with_imagen` flow
3. Try different prompts (landscapes, objects, etc.)
4. Verify output displays correctly

## Available Options

Options are based on `genai.types.GenerateImagesConfig`:

| Option | Type | Description |
|--------|------|-------------|
| `output_gcs_uri` | string | Cloud Storage URI for generated images |
| `negative_prompt` | string | What to discourage in generated images |
| `number_of_images` | integer | Number of images to generate |
| `guidance_scale` | float | Adherence to text prompt (higher = more adherence) |
| `seed` | integer | Random seed (not available with `add_watermark=true`) |
| `safety_filter_level` | enum | `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`, `BLOCK_NONE` |
| `person_generation` | enum | `DONT_ALLOW`, `ALLOW_ADULT`, `ALLOW_ALL` |
| `language` | enum | `auto`, `en`, `ja`, `ko`, `hi` |
| `aspect_ratio` | string | Aspect ratio of generated images |
| `add_watermark` | bool | Add watermark to generated images |

## Development

The `run.sh` script uses `watchmedo` to monitor changes in:
- `src/` (Python logic)
- `../../packages` (Genkit core)
- `../../plugins` (Genkit plugins)
- File patterns: `*.py`, `*.prompt`, `*.json`

Changes will automatically trigger a restart of the sample.
