# Microsoft Foundry Hello Sample

This sample demonstrates how to use the Microsoft Foundry plugin with Genkit.
Microsoft Foundry (formerly Azure AI Foundry) provides access to 11,000+ AI models.

## Documentation

- **Microsoft Foundry Portal**: https://ai.azure.com/
- **Model Catalog**: https://ai.azure.com/catalog/models
- **SDK Overview**: https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/sdk-overview
- **Models Documentation**: https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/models

## Prerequisites

1. **Azure Subscription**: You need an active Azure subscription with a Microsoft Foundry resource.
   - Follow the [Microsoft Foundry quickstart](https://learn.microsoft.com/en-us/azure/ai-foundry/quickstarts/get-started-code) to set up your resource.

2. **Model Deployment**: Deploy a model (e.g., `gpt-4o`) in your Microsoft Foundry resource.

3. **Find Your Credentials**: Get the endpoint and API key from the Microsoft Foundry portal:

   1. Go to [Microsoft Foundry Portal](https://ai.azure.com/)
   2. Select your **Project**
   3. Navigate to **Models** → **Deployments**
   4. Click on your **Deployment** (e.g., `gpt-4o`)
   5. Open the **Details** pane

   You'll see information like:
   - **Target URI**: `https://your-resource.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21`
   - **Key**: (click to reveal)
   - **Name**: `gpt-4o` (this is your deployment name)

4. **Environment Variables**: Set the following environment variables:

   ```bash
   # Extract the base URL from Target URI (everything before /openai/...)
   export AZURE_OPENAI_ENDPOINT="https://your-resource.cognitiveservices.azure.com/"
   
   # Your API key from the Details pane
   export AZURE_OPENAI_API_KEY="your-api-key"
   
   # Optional: API version from the Target URI query parameter (plugin has a sensible default)
   export AZURE_OPENAI_API_VERSION="2024-10-21"
   
   # Deployment name from the Details pane
   export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
   ```

   **Note**: The endpoint should be just the base URL without any path. Extract it from the Target URI:
   - Target URI: `https://your-resource.cognitiveservices.azure.com/openai/deployments/gpt-4o/...`
   - Endpoint: `https://your-resource.cognitiveservices.azure.com/`

## Authentication Methods

### 1. API Key (Simple)

```python
from genkit import Genkit
from genkit.plugins.microsoft_foundry import MicrosoftFoundry, gpt4o

ai = Genkit(
    plugins=[
        MicrosoftFoundry(
            api_key="your-api-key",
            endpoint="https://your-resource.openai.azure.com/",
            api_version="2024-10-21",
        )
    ],
    model=gpt4o,
)
```

### 2. Azure AD / Managed Identity (Recommended for Production)

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from genkit import Genkit
from genkit.plugins.microsoft_foundry import MicrosoftFoundry, gpt4o

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

ai = Genkit(
    plugins=[
        MicrosoftFoundry(
            azure_ad_token_provider=token_provider,
            endpoint="https://your-resource.openai.azure.com/",
            api_version="2024-10-21",
        )
    ],
    model=gpt4o,
)
```

## Running the Sample

1. **Install Dependencies**:
   Make sure you have `uv` installed.

2. **Run the Sample**:
   ```bash
   ./run.sh
   ```

   This will start the Genkit Developer UI with hot reloading.

3. **Test the Flows**:
   Open the Genkit Dev UI in your browser and test the available flows.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Simple Generation | `say_hi` | Basic text generation |
| Streaming | `say_hi_stream` | Streaming responses token by token |
| Tool Usage | `weather_flow` | Function calling with tools |
| Configuration | `say_hi_with_config` | Custom temperature, max_tokens, etc. |
| Multimodal | `describe_image` | Processing image inputs |
| Structured Output | `generate_character` | RPG character generation as JSON |
| Reasoning | `reasoning_flow` | Chain-of-thought reasoning with o4-mini |

## Supported Models

Microsoft Foundry provides access to 11,000+ models from multiple providers. Key supported models include:

### OpenAI GPT Models
- **GPT-4 Series**: gpt-4o, gpt-4o-mini, gpt-4, gpt-4.5, gpt-4.1
- **GPT-3.5**: gpt-3.5-turbo
- **GPT-5 Series**: gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.1, gpt-5.2
- **O-Series**: o1, o1-mini, o3, o3-mini, o4-mini

### Other Providers
- **Anthropic Claude**: claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5
- **DeepSeek**: DeepSeek-V3.2, DeepSeek-R1-0528
- **xAI Grok**: grok-4, grok-3, grok-3-mini
- **Meta Llama**: Llama-4-Maverick-17B-128E-Instruct-FP8
- **Mistral**: Mistral-Large-3

### Embedding Models
- text-embedding-3-small, text-embedding-3-large
- embed-v-4-0 (Cohere)

## References

- [What is Microsoft Foundry?](https://learn.microsoft.com/en-us/azure/ai-foundry/what-is-foundry?view=foundry&preserve-view=true)
- [Microsoft Foundry Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/)
- [Model Catalog](https://ai.azure.com/catalog/models)
- [SDK Overview](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/sdk-overview)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat/create)

## Disclaimer

This is a community sample and is not officially supported or endorsed by Microsoft Corporation.

"Microsoft", "Azure", "Azure OpenAI", "Microsoft Foundry", and "Azure AI Foundry" are
trademarks of Microsoft Corporation. This sample is developed independently and is not
affiliated with, endorsed by, or sponsored by Microsoft.

The use of Microsoft's APIs is subject to Microsoft's terms of service.
