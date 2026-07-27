# DashScope Plugin

This plugin provides a simple interface for using Alibaba Cloud Model Studio's
DashScope service to call Qwen models through its OpenAI-compatible endpoint.

## Prerequisites

- Go installed on your system
- A DashScope API key

## Base URL

By default the plugin points at the shared international endpoint,
`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, which works out of the box for
standard API keys. If you're on a mainland-China account, use
`https://dashscope.aliyuncs.com/compatible-mode/v1` instead via `DASHSCOPE_BASE_URL`:

```bash
export DASHSCOPE_BASE_URL=<your-base-url>
```

Alibaba also offers a **workspace-dedicated domain**
(`https://<workspace-id>.<region>.maas.aliyuncs.com/compatible-mode/v1`), which they
recommend for production use (higher throughput, lower latency, workspace-level
isolation) — get this from the Model Studio console's API-key popup or Workspace
Management page if you want it. It's an optional upgrade, not a requirement: the shared
default endpoint above works fine for standard usage too.

See https://help.aliyun.com/en/model-studio/base-url for the full reference.

## Running Tests

First, set your DashScope API key as an environment variable:

```bash
export DASHSCOPE_API_KEY=<your-api-key>
```

### Running All Tests
To run all tests in the directory:
```bash
go test -v .
```

### Running Tests from Specific Files
To run tests from a specific file:
```bash
# Run only generate_live_test.go tests
go test -run "^TestGenerator"

# Run only dashscope_live_test.go tests
go test -run "^TestPlugin"
```

### Running Individual Tests
To run a specific test case:
```bash
# Run only the streaming test from dashscope_live_test.go
go test -run "TestPlugin/streaming"

# Run only the Complete test from generate_live_test.go
go test -run "TestGenerator_Complete"

# Run only the Stream test from generate_live_test.go
go test -run "TestGenerator_Stream"
```

### Test Output Verbosity
Add the `-v` flag for verbose output:
```bash
go test -v -run "TestPlugin/streaming"
```

Note: All live tests require the DASHSCOPE_API_KEY environment variable to be set. Tests will be skipped if the API key is not provided.
