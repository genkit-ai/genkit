# Anthropic Plugin

This plugin provides a simple interface for using Anthropic's services.

## Prerequisites

- Go installed on your system
- An Anthropic API key

## Running Tests

First, set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY=<your-api-key>
```
By default, `baseURL` is set to "https://api.anthropic.com/v1". However, if you
want to use a custom value, you can set `ANTHROPIC_BASE_URL` environment variable:

```bash
export ANTHROPIC_BASE_URL=<your-custom-base-url>
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

# Run only anthropic_live_test.go tests
go test -run "^TestPlugin"
```

### Running Individual Tests
To run a specific test case:
```bash
# Run only the streaming test from anthropic_live_test.go
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

Note: All live tests require the ANTHROPIC_API_KEY environment variable to be set. Tests will be skipped if the API key is not provided.
