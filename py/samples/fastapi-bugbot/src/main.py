# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

r"""BugBot: AI Code Reviewer.

    genkit start -- uv run src/main.py
    curl localhost:8080/review -d '{"code": "query = f\"SELECT * FROM users WHERE id={user_input}\""}'

If something looks wrong, check localhost:4000 to see what the model actually received.
"""

import asyncio
from pathlib import Path
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing_extensions import Never

from genkit import Flow, Genkit
from genkit.plugins.fastapi import genkit_fastapi_handler
from genkit.plugins.google_genai import GoogleAI

_ = load_dotenv()

# The Dev UI reflection server starts automatically in a background thread
# when GENKIT_ENV=dev is set — no lifespan wiring needed.
ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.0-flash',
    prompt_dir=Path(__file__).resolve().parent.parent / 'prompts',
)


Severity = Literal['critical', 'warning', 'info']
Category = Literal['security', 'bug', 'style']


class Issue(BaseModel):
    """A single issue found in the code."""

    line: int = Field(description='Line number where the issue occurs')
    title: str = Field(description='Brief title like "SQL Injection Risk"')
    severity: Severity
    category: Category
    explanation: str = Field(description='Why this is a problem')
    suggestion: str = Field(description='How to fix it')


class Analysis(BaseModel):
    """Analysis result containing found issues."""

    issues: list[Issue] = Field(default_factory=list)


class CodeInput(BaseModel):
    """Input for code analysis."""

    code: str
    language: str = 'python'


class DiffInput(BaseModel):
    """Input for diff analysis."""

    diff: str
    context: str = ''


security_prompt = ai.prompt('analyze_security', input_schema=CodeInput, output_schema=Analysis)
bugs_prompt = ai.prompt('analyze_bugs', input_schema=CodeInput, output_schema=Analysis)
style_prompt = ai.prompt('analyze_style', input_schema=CodeInput, output_schema=Analysis)
diff_prompt = ai.prompt('analyze_diff', input_schema=DiffInput, output_schema=Analysis)


@ai.flow()
async def analyze_security(input: CodeInput) -> Analysis:
    """Analyze code for security vulnerabilities."""
    response = await security_prompt(input=input)
    return response.output


@ai.flow()
async def analyze_bugs(input: CodeInput) -> Analysis:
    """Analyze code for potential bugs."""
    response = await bugs_prompt(input=input)
    return response.output


@ai.flow()
async def analyze_style(input: CodeInput) -> Analysis:
    """Analyze code for style issues."""
    response = await style_prompt(input=input)
    return response.output


@ai.flow()
async def review_code(input: CodeInput) -> Analysis:
    """Run all analyzers in parallel and combine results."""
    security, bugs, style = await asyncio.gather(
        analyze_security(input),
        analyze_bugs(input),
        analyze_style(input),
    )
    return Analysis(issues=security.issues + bugs.issues + style.issues)


@ai.flow()
async def review_diff(input: DiffInput) -> Analysis:
    """Review a code diff for issues."""
    response = await diff_prompt(input=input)
    return response.output


app = FastAPI(title='BugBot', description='AI-powered code review API')


@app.post('/review')
async def review(input: CodeInput) -> Analysis:
    """Review code for security, bugs, and style issues."""
    return await review_code(input)


@app.post('/review/security')
async def review_security_endpoint(input: CodeInput) -> Analysis:
    """Review code for security issues only."""
    return await analyze_security(input)


@app.post('/review/diff')
async def review_diff_endpoint(input: DiffInput) -> Analysis:
    """Review a code diff."""
    return await review_diff(input)


@app.post('/flow/review', response_model=None)
@genkit_fastapi_handler(ai)
def flow_review() -> Flow[CodeInput, Analysis, Never]:
    """Expose review_code flow directly via {"data": {"code": "...", "language": "..."}}."""
    return review_code


@app.post('/flow/security', response_model=None)
@genkit_fastapi_handler(ai)
def flow_security() -> Flow[CodeInput, Analysis, Never]:
    """Expose analyze_security flow directly."""
    return analyze_security


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)  # noqa: S104
