# Amazon Bedrock Hello Sample

> **Community Plugin** – This plugin is maintained by the community and is supported on a best-effort basis. It is not an official AWS product.

This sample demonstrates how to use Amazon Bedrock models with Genkit, including AWS X-Ray telemetry for distributed tracing.

## Prerequisites

1. **AWS Account** with Bedrock access enabled
2. **AWS Credentials** configured (one of the following):
   - **Bedrock API Key** (`AWS_BEARER_TOKEN_BEDROCK`) - Simplest option, like OpenAI
   - IAM credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
   - AWS credentials file (`~/.aws/credentials`)
   - IAM role (for EC2, Lambda, ECS, etc.)
3. **Model Access** enabled in Amazon Bedrock console for the models you want to use

## Setup

### Option A: Bedrock API Key (Simplest - Recommended for Development)

Amazon Bedrock now supports API keys similar to OpenAI/Anthropic. This is the simplest way to get started.

**Step 1: Generate an API Key**

1. Go to [Amazon Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Select your region (e.g., `us-east-1`)
3. Click **API keys** in the left sidebar (under "Bedrock configurations")
4. Click **Generate API key**
5. Choose expiration (30 days recommended for development)
6. Copy the generated key (shown only once!)

**Step 2: Set Environment Variables**

```bash
export AWS_REGION="us-east-1"
export AWS_BEARER_TOKEN_BEDROCK="your-api-key-here"
```

That's it! No IAM user or access keys needed.

**Limitations of API Keys:**
- For development/exploration only (not recommended for production)
- Cannot be used with Bedrock Agents or Data Automation
- Keys expire based on your configuration

See: [Getting Started with Bedrock API Keys](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-api-keys.html)

---

### Option B: IAM Credentials (Recommended for Production)

#### Step 1: Create AWS Credentials

If you don't have AWS credentials yet, follow these steps:

1. **Sign in to AWS Console**: Go to [AWS Console](https://console.aws.amazon.com/)

2. **Navigate to IAM**:
   - Search for "IAM" in the AWS Console search bar
   - Or go directly to [IAM Console](https://console.aws.amazon.com/iam/)

3. **Create an IAM User** (recommended for development):
   - Click **Users** in the left sidebar
   - Click **Create user**
   - Enter a username (e.g., `genkit-bedrock-dev`)
   - Click **Next**

4. **Attach Permissions**:
   - Select **Attach policies directly**
   - Search for and select `AmazonBedrockFullAccess`
   - Click **Next**, then **Create user**

5. **Create Access Keys**:
   - Click on the user you just created
   - Go to the **Security credentials** tab
   - Scroll to **Access keys** and click **Create access key**
   - Select **Local code** as the use case
   - Click **Create access key**
   - **Important**: Copy both the **Access key ID** and **Secret access key**
   - These are shown only once! Save them securely.

### Step 2: Enable Model Access

1. Go to the [Amazon Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Select your region (e.g., `us-east-1`) from the top-right dropdown
3. Click **Model access** in the left sidebar (under "Bedrock configurations")
4. Click **Modify model access**
5. Check the models you want to use:
   - **Anthropic**: Claude Sonnet, Claude Haiku (recommended for testing)
   - **Amazon**: Nova Pro, Nova Lite, Nova Micro
   - **Meta**: Llama 3.3, Llama 4
6. Click **Next**, review, and click **Submit**
7. Wait for status to change from "In progress" to "Access granted"

### Step 3: Configure Environment Variables

Set your credentials as environment variables:

```bash
# Required: AWS Region where you enabled model access
export AWS_REGION="us-east-1"

# Required: Your IAM user credentials (from Step 1)
export AWS_ACCESS_KEY_ID="AKIA..."           # Starts with AKIA
export AWS_SECRET_ACCESS_KEY="wJalrXUt..."   # Your secret key
```

**Tip**: Add these to your shell profile (`~/.bashrc`, `~/.zshrc`) to persist them:

```bash
# Add to ~/.zshrc or ~/.bashrc
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
```

### Option C: Using AWS CLI Profile

If you prefer using the AWS CLI:

```bash
# Install AWS CLI (if not already installed)
# macOS: brew install awscli
# Linux: pip install awscli

# Configure credentials
aws configure
# Enter your Access Key ID, Secret Access Key, and region when prompted

# Then just set the region for this sample
export AWS_REGION="us-east-1"
```

### Option D: IAM Role (for AWS Infrastructure)

If running on EC2, Lambda, ECS, or EKS:

```bash
# Only the region is needed - credentials come from the IAM role
export AWS_REGION="us-east-1"
```

Make sure the IAM role has the `AmazonBedrockFullAccess` policy attached.

## Running the Sample

```bash
./run.sh
```

This will start the Genkit Dev UI and the sample application with hot reloading.

## Features Demonstrated

| Feature | Flow Name | Description |
|---------|-----------|-------------|
| Simple Generation | `say_hi` | Basic text generation |
| Streaming | `say_hi_stream` | Streaming text generation |
| Tool Use | `weather_flow` | Function calling with tools |
| Multi-turn Chat | `chat_demo` | Multi-turn conversation |
| Structured Output | `generate_character` | Generate JSON-structured output |
| Multimodal | `describe_image` | Image description (Claude, Nova) |
| Embeddings | `embed_text` | Text embeddings (Titan, Cohere) |
| AWS X-Ray Telemetry | `add_aws_telemetry()` | Distributed tracing to X-Ray console |

## Supported Models

The sample uses Claude Sonnet 4.5 by default and auto-detects your authentication method.

### Authentication & Model IDs

| Auth Method | Model ID Format | Example |
|-------------|-----------------|---------|
| IAM Credentials | Direct model ID | `anthropic.claude-sonnet-4-5-...` |
| API Key (Bearer Token) | Inference profile | `us.anthropic.claude-sonnet-4-5-...` |

**Important**: When using API keys (`AWS_BEARER_TOKEN_BEDROCK`), you must use **inference profiles** with a regional prefix (`us.`, `eu.`, or `apac.`).

### Using Pre-defined Model References (IAM Credentials)

```python
from genkit.plugins.amazon_bedrock import claude_sonnet_4_5, nova_pro, llama_3_3_70b

# Pre-defined references use direct model IDs
ai = Genkit(model=claude_sonnet_4_5)
```

### Using Inference Profiles (API Keys)

```python
from genkit.plugins.amazon_bedrock import inference_profile

# inference_profile() auto-detects region from AWS_REGION
model = inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0')

# Or specify region explicitly
model = inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0', 'eu-west-1')
```

### Regional Prefixes

| Region | Prefix | Example Regions |
|--------|--------|-----------------|
| US | `us.` | us-east-1, us-west-2 |
| Europe | `eu.` | eu-west-1, eu-central-1 |
| Asia Pacific | `apac.` | ap-northeast-1, ap-southeast-1 |

### Available Models

- **Claude (Anthropic)**: claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5
- **Nova (Amazon)**: nova-pro, nova-lite, nova-micro
- **Llama (Meta)**: llama-3.3-70b, llama-4-maverick
- **Mistral**: mistral-large-3, pixtral-large
- **DeepSeek**: deepseek-r1, deepseek-v3
- **Cohere**: command-r-plus, command-r

See:
- [Amazon Bedrock Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Cross-Region Inference Profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html)

## Testing the Flows

Once the Dev UI is running, you can test each flow:

1. Open the Dev UI (automatically opens in browser)
2. Select a flow from the sidebar
3. Click **Run** to execute with default inputs
4. Modify inputs to test different scenarios

## Troubleshooting

### "NoCredentialsError" or "Unable to locate credentials"

Your AWS credentials are not configured. Make sure you have set:

```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
```

Or configured credentials via `aws configure`.

### "InvalidAccessKeyId" or "SignatureDoesNotMatch"

Your credentials are invalid or incorrectly copied. Double-check:
- The Access Key ID starts with `AKIA`
- No extra spaces in the values
- The secret key was copied completely

### "AccessDeniedException" Error

This can mean:
1. **Model access not enabled**: Go to Bedrock Console > Model access and enable the model
2. **IAM permissions missing**: Make sure your IAM user has `AmazonBedrockFullAccess` policy
3. **Wrong region**: The model may not be available in your region

### "Region not supported" Error

Not all models are available in all regions. Check [model availability by region](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html).

Popular regions with most models:
- `us-east-1` (N. Virginia)
- `us-west-2` (Oregon)
- `eu-west-1` (Ireland)

### Timeout Errors

For Amazon Nova models, the inference timeout can be up to 60 minutes. The plugin automatically configures appropriate timeouts.

### "ValidationException: Malformed input request"

This usually means the model ID is incorrect. Check the exact model ID in the Bedrock console under "Model access".

### "ValidationException: Invocation of model ID ... with on-demand throughput isn't supported"

This error occurs when using API keys (`AWS_BEARER_TOKEN_BEDROCK`) with a direct model ID instead of an inference profile.

**Solution**: Use the cross-region inference profile ID with a regional prefix:

```python
# Wrong (direct model ID):
model = 'anthropic.claude-sonnet-4-5-20250929-v1:0'

# Correct (inference profile with us. prefix):
model = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

# Or use the pre-defined references (already use inference profiles):
from genkit.plugins.amazon_bedrock import claude_sonnet_4_5
model = claude_sonnet_4_5
```

## Security Best Practices

1. **Never commit credentials**: Add AWS credentials to `.gitignore`, never commit them
2. **Use IAM roles in production**: On AWS infrastructure, use IAM roles instead of access keys
3. **Rotate keys regularly**: Periodically rotate your access keys in IAM console
4. **Least privilege**: Create IAM users with only the permissions needed

## Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
- [Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)
- [Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
- [IAM Getting Started](https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started.html)
- [AWS CLI Configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)
