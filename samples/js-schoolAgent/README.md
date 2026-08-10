# School Agent Sample

A demonstration of a conversational, multi-agent assistant for a school system using GenKit and Google's Gemini Pro. This agent helps parents with attendance reporting and school information queries.

In this example we have a RoutingAgent which is the main, general-purpose agent.
This agent comes equipped with additional specialized agents, that it can hand-off to as needed.

These specialized agents are represented as prompts and embedded as tools to the original agent.

## Agent Tools & Capabilities

- **Agent Structure**:
  - `RoutingAgent`: Main entry point and router, handling general queries and delegating to specialized agents
  - `AttendanceAgent`: Specialized agent for absence/tardy reporting
  - `GradesAgent`: Manages grade-related inquiries and academic performance

Each specialized agent has its own set of tools that are only accessible to that specific agent:

- **AttendanceAgent**:
  - `reportAbsence`: Submit absence notifications
  - `reportTardy`: Report late arrivals
- **GradesAgent**:
  - `getRecentGrades`: Retrieve latest grade information

The main RoutingAgent cannot directly access these specialized tools - it can only access its own tools and delegate to the specialized agents. This means the specialized agent descriptions need to clearly communicate their capabilities, since the main agent relies on these descriptions for appropriate routing.

For example, when the RoutingAgent sees a grade-related query, it needs to know from the GradesAgent's description that it can handle grade lookups, even though it can't directly see the `getRecentGrades` tool.

This architectural pattern:

- Maintains clear separation of concerns
- Allows specialized agents to evolve independently
- Allows scaling up to a larger number of tools

NOTE: The agent description is how the generalized agent knows what tools the specialized agent has available. An agent description that is too general may cause the routing agent to mess up by not knowing that a certain functionality was actually available.

## Prerequisites

- Node.js and genkit CLI installed
- Google AI API key

## Getting Started

1. Install dependencies:

```bash
npm install
```

1. Set up your Google AI API key:

```bash
export GEMINI_API_KEY=your_api_key_here
```

1. Start the development server:

```bash
npm run genkit:dev
```

In your terminal, a commandline chat interface should show up:

```terminal
Telemetry API running on http://localhost:4033
Genkit Developer UI: http://localhost:4000

> school-agent@1.0.0 dev
> tsx --no-warnings --watch src/terminal.ts

bell> Hi there, my name is Bell and I'm here to help! 👋🎉 I'm your friendly AI assistant for parents of Sparkyville High School. I can answer your questions about the school, events, grades, and more. Just ask me! 😊

prompt> [insert your chats here]
```

You can feel free to tweak the sample. The project builds in watch mode, so any changes will be picked up immediately and should restart the conversation.

## Usage

The agent uses a multi-agent architecture:

- Routing Agent: Acts as the main entry point and router, handling general queries while delegating specialized requests to appropriate agents
- Attendance Agent: Specialized agent focused on absence and tardy reporting
- Grades Agent: Manages academic performance queries, grade reports, and transcript requests

Example queries:

- "Evelyn will be late today"
- "What are the upcoming holidays I should be aware of?"
- "Show me my child's current grades"

## Development

- `npm run dev` - Run in development mode with hot reloading
- `npm run build` - Build the project
- `npm start` - Run the built version

## Project Structure

- `src/`
  - `agents/`
    - `routingAgent.ts` - Main agent that uses other agents as tools
    - `attendanceAgent.ts` - Specialized attendance agent
    - `gradesAgent.ts` - Academic performance and grades agent
  - `tools.ts` - Tool definitions
  - `types.ts` - TypeScript types
  - `data.ts` - Sample data
