# Simple MCP Demo

This demo shows how you can use Genkit's MCP package to expose an MCP Server that can be used by external consumers like Claude Desktop, VS Code with MCP, and other useful tools and clients.

## Files
- `server.go` - MCP server with 2 tools: `text_encode` and `hash_generate`
- `run-server.sh` - Simple wrapper script for easy client setup
- `client.go` - Client that uses tools with Gemini AI

## Two Ways to Run

### Option A: Expose an MCP Server that works with Desktop Clients

#### Step 1: Test the server works
```bash
./run-server.sh
```
You should see: `Starting server with tools: [text_encode hash_generate]`

#### Step 2: Configure client.
Point your client to run that bash command to start the server. Make sure you restart the client to ensure the server is properly started.

#### Step 4: Test your assistant's connection to the MCP server with these quries:
- *"Encode 'Hello World' as base64"*
- *"Generate an MD5 hash of 'password123'"*
- *"Decode this base64: SGVsbG8gV29ybGQ="*

### Option B: Genkit MCP Client spawns the MCP Server

First, change `client()` to `main()` in `go/samples/mcp-server/client.go`. Then follow the steps below:
 
```bash
export GOOGLE_AI_API_KEY=your_key
go run client.go
```

The `GenkitMCPClient` instance automatically spawns the server and makes tools available to use with `client.ActiveTools(ctx)`. You can use these tools with any `Generate` method or `ExecutablePrompt`.

Alternatively, you can use `MCPHost` for managing multiple servers:

```go
host, err := mcp.NewHost(mcp.MCPHostOptions{
    Name: "my-app",
    MCPServers: []mcp.MCPServerConfig{
        {
            Name: "textUtils",
            Config: mcp.MCPClientOptions{
                Stdio: &mcp.StdioConfig{
                    Command: "go",
                    Args:    []string{"run", "server.go"},
                },
            },
        },
        {
            Name: "otherServer",
            Config: mcp.MCPClientOptions{
                StreamableHTTP: &mcp.StreamableHTTPConfig{
                    BaseURL: "http://localhost:8080/mcp",
                },
            },
        },
    },
})
if err != nil {
    log.Fatal(err)
}
tools, err := host.ActiveTools(ctx)
if err != nil {
    log.Fatal(err)
}
log.Printf("Found %d tools", len(tools))
```

This example demonstrates the following workflow:
1. Client connects to server via MCP protocol
2. AI gets prompt: "Fetch content from URL and summarize it"  
3. AI automatically calls `fetch_url` then `summarize`
4. Returns the summary 

This option is useful when you have an genkit agent that needs access to utilities developed in the same project. 
