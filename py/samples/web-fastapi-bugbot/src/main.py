# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

r"""BugBot: AI Code Reviewer.

    genkit start -- uv run src/main.py
    curl localhost:8080/review -d '{"code": "query = f\"SELECT * FROM users WHERE id={user_input}\""}'

If something looks wrong, check localhost:4000 to see what the model actually received.
"""

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Lifespan
from typing_extensions import Never

from genkit import Genkit, Input, Output
from genkit.ai import FlowWrapper
from genkit.plugins.fastapi import genkit_fastapi_handler, genkit_lifespan
from genkit.plugins.google_genai import GoogleAI

_ = load_dotenv()

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.0-flash',
    prompt_dir=Path(__file__).parent.parent / 'prompts',
)

# Dev UI lifespan - registers with Genkit Dev UI when GENKIT_ENV=dev
lifespan: Lifespan[ASGIApp] = genkit_lifespan(ai)


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


security_prompt = ai.prompt('analyze_security', input=Input(schema=CodeInput), output=Output(schema=Analysis))
bugs_prompt = ai.prompt('analyze_bugs', input=Input(schema=CodeInput), output=Output(schema=Analysis))
style_prompt = ai.prompt('analyze_style', input=Input(schema=CodeInput), output=Output(schema=Analysis))
diff_prompt = ai.prompt('analyze_diff', input=Input(schema=DiffInput), output=Output(schema=Analysis))


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


app = FastAPI(title='BugBot', description='AI-powered code review API', lifespan=lifespan)


@app.post('/review')
async def review(code: str, language: str = 'python') -> Analysis:
    """Review code for security, bugs, and style issues."""
    return await review_code(CodeInput(code=code, language=language))


@app.post('/review/security')
async def review_security_endpoint(code: str, language: str = 'python') -> Analysis:
    """Review code for security issues only."""
    return await analyze_security(CodeInput(code=code, language=language))


@app.post('/review/diff')
async def review_diff_endpoint(diff: str, context: str = '') -> Analysis:
    """Review a code diff."""
    return await review_diff(DiffInput(diff=diff, context=context))


@app.post('/flow/review', response_model=None)
@genkit_fastapi_handler(ai)
def flow_review() -> FlowWrapper[..., Awaitable[Analysis], Analysis, Never]:
    """Expose review_code flow directly via {"data": {"code": "...", "language": "..."}}."""
    return review_code


@app.post('/flow/security', response_model=None)
@genkit_fastapi_handler(ai)
def flow_security() -> FlowWrapper[..., Awaitable[Analysis], Analysis, Never]:
    """Expose analyze_security flow directly."""
    return analyze_security


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)  # noqa: S104
