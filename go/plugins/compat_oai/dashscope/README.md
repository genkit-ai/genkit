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

## Tool Choice

Qwen models support tool calling, but forced tool-choice modes (`required`/`none`)
carry model- and thinking-mode-specific restrictions. This plugin does not
advertise `ToolChoice` support and always uses automatic tool selection.

## Live tests

Live tests are skipped unless `DASHSCOPE_API_KEY` is set:

```bash
go test -race ./plugins/compat_oai/dashscope -run '^TestPluginLive$' -v -count=1
```
