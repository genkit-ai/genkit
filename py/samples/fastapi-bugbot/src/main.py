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

"""Review a snippet — three generate() calls in parallel, structured JSON back."""

import asyncio
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from genkit_fastapi import serve_flow
from genkit_google_genai import GoogleAI
from pydantic import BaseModel, Field

from genkit import Genkit

_ = load_dotenv()

ai = Genkit(
    plugins=[GoogleAI()],
    model=GoogleAI.gemini_model('gemini-flash-latest'),
)


class Issue(BaseModel):
    line: int
    title: str
    severity: Literal['critical', 'warning', 'info']
    category: Literal['security', 'bug', 'style']
    explanation: str
    suggestion: str


class Analysis(BaseModel):
    issues: list[Issue] = Field(default_factory=list)


class CodeInput(BaseModel):
    code: str
    language: str = 'python'


@ai.flow()
async def review_code(input: CodeInput) -> Analysis:
    # output_schema is the Pydantic model the model has to fill in — and
    # what you return from the route. Three focused calls so one review
    # doesn't wait for the others.
    security, bugs, style = await asyncio.gather(
        ai.generate(
            prompt=f'Find security issues in this {input.language} snippet:\n{input.code}',
            output_schema=Analysis,
        ),
        ai.generate(
            prompt=f'Find bugs in this {input.language} snippet:\n{input.code}',
            output_schema=Analysis,
        ),
        ai.generate(
            prompt=f'Find style issues in this {input.language} snippet:\n{input.code}',
            output_schema=Analysis,
        ),
    )
    issues = []
    for result in (security, bugs, style):
        if result.output:
            issues.extend(result.output.issues)
    return Analysis(issues=issues)


app = FastAPI(title='BugBot')
app.include_router(serve_flow(review_code, base_path='/review'))


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
