# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""One FastAPI process that serves every agent in this folder over HTTP.

Each agent is mounted with ``serve_agent``: one ``include_router`` gives you the
turn route plus ``/getSnapshot`` and ``/abort``. Plain flows go through
``serve_flow``. The ``prefix='/api'`` mount puts everything at ``/api/<name>``,
so a small team can swap this in as the Python agent backend without changing
frontend paths.

    genkit start -- uv run testapp/server.py   # Dev UI (:4000) + API (:8080)

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import uvicorn

# Importing an agent module registers its agent (and Dev-UI flow) on the shared
# ``ai``. Listing them here is also what makes them show up in the Dev UI.
from background_agent import background_agent
from banking_agent import banking_agent
from branching_agent import branching_agent
from client_state_agent import weather_agent_stateless
from coding_agent import coding_agent, list_workspace_files, read_workspace_file
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from file_store_agent import file_store_agent
from genkit_fastapi import serve_agent, serve_flow
from orchestrator_agent import orchestrator_agent
from research_agent import research_agent
from task_agent import task_agent
from trip_planner_agent import trip_planner_agent
from weather_agent import weather_agent
from workspace_agent import workspace_agent

app = FastAPI(title='Genkit Agents (Python)')

# The frontend is a different origin, so let it in and expose the streaming
# header the client reads to correlate chunks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=['X-Genkit-Stream-Id'],
)

# Each agent's route path comes from its own name, so /api/<agentName> lines up
# with what the frontend calls. serve_agent also wires up getSnapshot/abort.
for agent in (
    weather_agent,
    weather_agent_stateless,
    file_store_agent,
    research_agent,
    task_agent,
    banking_agent,
    workspace_agent,
    background_agent,
    branching_agent,
    orchestrator_agent,
    trip_planner_agent,
    coding_agent,
):
    app.include_router(serve_agent(agent), prefix='/api')

# The coding-agent file browser hits these two paths. Pin base_path so the
# URLs stay stable instead of following the flow name.
app.include_router(serve_flow(list_workspace_files, base_path='/workspace/files'), prefix='/api')
app.include_router(serve_flow(read_workspace_file, base_path='/workspace/file'), prefix='/api')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
