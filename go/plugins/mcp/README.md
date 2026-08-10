# Genkit MCP Plugin

Model Context Protocol (MCP) integration for Go Genkit

Connect to MCP servers and expose Genkit tools as MCP servers.

## GenkitMCPClient - Single Server Connection

Connect to and use tools/prompts from a single MCP server:

```go
package main

import (
  "context"
  "log"

  "github.com/firebase/genkit/go/genkit"
  "github.com/firebase/genkit/go/plugins/mcp"
)

func main() {
  ctx := context.Background()
  g := genkit.Init(ctx)

  // Connect to the MCP everything server
  client, err := mcp.NewGenkitMCPClient(mcp.MCPClientOptions{
    Name: "everything-server",
    Stdio: &mcp.StdioConfig{
      Command: "npx",
      Args:    []string{"-y", "@modelcontextprotocol/server-everything"},
    },
  })
  if err != nil {
    log.Fatal(err)
  }

  // Get specific prompts from the everything server
  simplePrompt, err := client.GetPrompt(ctx, g, "simple-prompt", nil)
  if err != nil {
    log.Fatal(err)
  }

  // Get all available tools
  tools, err := client.GetActiveTools(ctx, g)
  if err != nil {
    log.Fatal(err)
  }
}
```

## GenkitMCPManager - Multiple Server Management

Manage connections to multiple MCP servers:

```go
package main

import (
    "context"
    "log"

    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/mcp"
)

func main() {
    ctx := context.Background()
    g, _ := genkit.Init(ctx)

    // Create manager with multiple servers
    manager, err := mcp.NewMCPManager(mcp.MCPManagerOptions{
        Name: "my-app",
        MCPServers: map[string]mcp.MCPClientOptions{
            "everything": {
                Name: "everything-server",
                Stdio: &mcp.StdioConfig{
                    Command: "npx",
                    Args: []string{"-y", "@modelcontextprotocol/server-everything"},
                },
            },
            "filesystem": {
                Name: "fs-server",
                Stdio: &mcp.StdioConfig{
                    Command: "npx",
                    Args: []string{"@modelcontextprotocol/server-filesystem", "/tmp"},
                },
            },
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    // Connect to new server at runtime
    err = manager.ConnectServer(ctx, "weather", mcp.MCPClientOptions{
        Name: "weather-server",
        Stdio: &mcp.StdioConfig{
            Command: "python",
            Args: []string{"weather_server.py"},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    // Temporarily disable/enable servers
    manager.DisableServer("filesystem")
    manager.EnableServer("filesystem")

    // Disconnect server
    manager.DisconnectServer("weather")

    // Get tools from all active servers
    tools, err := manager.GetActiveTools(ctx, g)
    if err != nil {
        log.Fatal(err)
    }
}
```

## GenkitMCPServer - Expose Genkit Tools

Turn your Genkit app into an MCP server:

```go
package main

import (
  "context"
  "log"

  "github.com/firebase/genkit/go/genkit"
  "github.com/firebase/genkit/go/plugins/mcp"
)

func main() {
  ctx := context.Background()
  g := genkit.Init(ctx)

  // Create a host with multiple servers
  host, err := mcp.NewMCPHost(g, mcp.MCPHostOptions{
    Name: "my-app",
    MCPServers: []mcp.MCPServerConfig{
      {
        Name: "everything-server",
        Config: mcp.MCPClientOptions{
          Name: "everything-server",
          Stdio: &mcp.StdioConfig{
            Command: "npx",
            Args:    []string{"-y", "@modelcontextprotocol/server-everything"},
          },
        },
      },
      {
        Name: "fs-server",
        Config: mcp.MCPClientOptions{
          Name: "fs-server",
          Stdio: &mcp.StdioConfig{
            Command: "npx",
            Args:    []string{"@modelcontextprotocol/server-filesystem", "/tmp"},
          },
        },
      },
    },
  })
  if err != nil {
    log.Fatal(err)
  }

  // Connect to new server at runtime
  err = host.Connect(ctx, g, "weather", mcp.MCPClientOptions{
    Name: "weather-server",
    Stdio: &mcp.StdioConfig{
      Command: "python",
      Args:    []string{"weather_server.py"},
    },
  })
  if err != nil {
    log.Fatal(err)
  }

  // Reconnect server
  host.Reconnect(ctx, "fs-server")

  // Disconnect server
  host.Disconnect(ctx, "weather")

  // Get tools from all active servers
  tools, err := host.GetActiveTools(ctx, g)
  if err != nil {
    log.Fatal(err)
  }
}

```

## Testing Your Server

```bash
# Run your server
go run main.go

# Test with MCP Inspector
npx @modelcontextprotocol/inspector go run main.go
```

## Transport Options

### Stdio (Standard)

```go
Stdio: &mcp.StdioConfig{
    Command: "uvx",
    Args: []string{"mcp-server-time"},
    Env: []string{"DEBUG=1"},
}
```

### SSE (Web clients)

```go
SSE: &mcp.SSEConfig{
    BaseURL: "http://localhost:3000/sse",
}
```
