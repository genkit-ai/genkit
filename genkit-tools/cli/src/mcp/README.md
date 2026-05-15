## 🚀 Genkit MCP Server: Model Context Protocol Integration

The **Genkit MCP (Model Context Protocol) Server** bridges your Genkit projects with external AI tools and development environments. It exposes Genkit flows and functionalities via the Model Context Protocol, allowing **LLM agents** and **IDEs** to discover, interact with, and monitor your application.

> **Note:** The Genkit MCP server is an experimental feature and may change.

---

### What It Does

The MCP Server enables external tools to:

* **Discover Flows:** List all available Genkit flows, including their input schemas.
* **Run Flows:** Execute Genkit flows by providing inputs and receiving outputs.
* **Access Traces:** Retrieve and analyze detailed execution traces for performance insights.
* **Look up Documentation:** Access Genkit documentation directly.
* **Manage Runtime:** Start, stop, and restart the Genkit runtime process.

---

### Available MCP Tools

The following tools allow MCP-aware environments to interact with the Genkit server:

| Tool Name | Description |
| :--- | :--- |
| **`get_usage_guide`** | Fetches the Genkit AI framework usage guide (specifiable by language). Intended for AI assistants. |
| **`list_genkit_docs`** | Lists all available Genkit documentation files for discovery. |
| **`search_genkit_docs`** | Searches Genkit documentation using keywords. |
| **`read_genkit_docs`** | Reads the content of specific Genkit documentation files. |
| **`list_flows`** | Discovers and lists all defined Genkit flows with their input schemas. |
| **`run_flow`** | Executes a specified Genkit flow, requiring `flowName` and a JSON `input` conforming to the flow's schema. |
| **`get_trace`** | Retrieves the detailed execution trace for a flow using a `traceId`. |
| **`start_runtime`** | Starts a Genkit runtime process (e.g., `npm run dev` or `go run main.go`). |
| **`kill_runtime`** | Kills the runtime process started by `start_runtime`. |
| **`restart_runtime`** | Restarts the runtime process started by `start_runtime`. |

### Available Prompts

| Prompt Name | Description |
| :--- | :--- |
| **`genkit:init`** | Helps initialize a new Genkit project with best practices and language-specific setup. |
