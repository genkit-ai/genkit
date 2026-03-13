#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""FastAPI backend for Model Performance Tool."""

import asyncio
import logging
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import tool logic
from model_performance_test import (
    discover_models,
    discover_models_for_sample,
    generate_config_variations,
    parse_config_schema,
    run_model_test,
)
from pydantic import BaseModel

app = FastAPI(title='Model Performance Tool')

# CORS for local dev
app.add_middleware(
    cast(Any, CORSMiddleware),
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# --- Data Models ---


class Scenario(BaseModel):
    """Scenario model for test cases."""

    id: str
    name: str
    description: str
    system_prompt: str
    user_prompt: str


class TestRequest(BaseModel):
    """Request model for running a single test."""

    model: str
    config: dict[str, Any]
    scenario_id: str
    user_prompt: str
    system_prompt: str


class TestResult(BaseModel):
    """Result model for a single test run."""

    success: bool
    response: str | None
    error: str | None
    timing: float


# --- Scenarios Discovery ---


async def discover_scenarios() -> list[Scenario]:
    """Discover test scenarios from samples directory."""
    samples_dir = Path(__file__).parent.parent.parent / 'samples'
    scenarios = []

    if not samples_dir.exists():
        return scenarios

    # Iterate through sample directories
    for item in sorted(samples_dir.iterdir()):
        if item.is_dir() and not item.name.startswith(('.', '_')):
            # Try to read metadata from pyproject.toml
            pyproject_path = item / 'pyproject.toml'

            # Skip samples without pyproject.toml - they are likely legacy or broken
            if not pyproject_path.exists():
                logging.debug(f'Skipping {item.name}: no pyproject.toml found')
                continue

            name = item.name
            description = 'Genkit sample project'

            try:
                import sys

                if sys.version_info >= (3, 11):
                    import tomllib
                else:
                    import tomli as tomllib  # conditional dep for 3.10

                def load_toml_data(path: Path) -> dict[str, Any]:
                    with open(path, 'rb') as f:
                        return tomllib.load(f)

                data = await asyncio.to_thread(load_toml_data, pyproject_path)
                project = data.get('project', {})
                name = project.get('name', name)
                description = project.get('description', description)
            except Exception:  # noqa: S110
                pass

            scenarios.append(
                Scenario(
                    id=item.name,
                    name=name,
                    description=description,
                    system_prompt='You are a pirate.',
                    user_prompt='what is 5 + 3?',
                )
            )

    return scenarios


# We'll fetch SCENARIOS dynamically in the endpoint

# --- Endpoints ---


@app.get('/', response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Root endpoint that serves the index.html file."""
    logging.info('Root path accessed')
    static_dir = Path(__file__).parent / 'static'
    index_path = static_dir / 'index.html'
    if not index_path.exists():
        logging.error(f'index.html not found at {index_path}')
        return HTMLResponse(content='index.html not found', status_code=404)
    return HTMLResponse(content=index_path.read_text())


@app.get('/health')
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {'status': 'ok', 'message': 'Server is running'}


@app.get('/api/scenarios', response_model=list[Scenario])
async def get_scenarios() -> list[Scenario]:
    """Get all available test scenarios."""
    return await discover_scenarios()


@app.get('/api/models')
async def get_models(sample: str | None = None) -> list[dict[str, Any]]:
    """Get models available for a specific sample.

    Args:
        sample: Sample name to discover models for (e.g., 'google-genai-hello')

    Returns:
        List of models with numbering, info, and parsed parameters
    """
    try:
        if sample:
            logging.info(f'Discovering models for sample: {sample}')
            models = await discover_models_for_sample(sample)
        else:
            logging.info('Discovering all models')

            models = await discover_models()

        logging.info(f'Found {len(models)} models: {list(models.keys())}')

        # Build numbered model list
        filtered_models = []
        for idx, (name, info) in enumerate(models.items(), start=1):
            # Validate model name
            if not name or not isinstance(name, str):
                logging.warning(f'Invalid model name at index {idx}: {name}')
                continue

            # Extract config schema for UI
            config_schema = info.get('customOptions', {}) if isinstance(info, dict) else {}
            params = parse_config_schema(config_schema)

            model_entry = {
                'number': idx,
                'name': name,
                'display_name': f'{idx}. {name}',
                'info': info if isinstance(info, dict) else {},
                'params': params,
            }

            logging.debug(f'Model entry: {model_entry["display_name"]}')
            filtered_models.append(model_entry)

        logging.info(f'Returning {len(filtered_models)} filtered models')
        return filtered_models
    except Exception as e:
        logging.error(f'Error discovering models: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post('/api/run', response_model=TestResult)
async def run_test(request: TestRequest) -> TestResult:
    """Execute a single model test."""
    try:
        script_path = Path(__file__).parent / 'run_single_model_test.py'

        # Run test in thread pool to not block async loop (subprocess call inside)
        # Run test in thread pool to not block async loop (subprocess call inside)
        result = await asyncio.to_thread(
            run_wrapper,
            request.model,
            request.config,
            request.user_prompt,
            request.system_prompt,
            script_path,
        )

        return TestResult(
            success=result.get('success', False),
            response=str(result.get('response')) if result.get('response') else None,
            error=result.get('error'),
            timing=result.get('timing', 0.0),
        )

    except Exception as e:
        logging.error(f'Error running test: {e}')
        raise HTTPException(status_code=500, detail=str(e)) from e


class ComprehensiveTestRequest(BaseModel):
    """Request for running comprehensive tests."""

    sample: str
    model: str
    user_prompt: str
    system_prompt: str


class ComprehensiveTestResult(BaseModel):
    """Result of a comprehensive test run."""

    total_tests: int
    passed: int
    failed: int
    results: list[dict[str, Any]]


@app.post('/api/run-comprehensive', response_model=ComprehensiveTestResult)
async def run_comprehensive_test(
    request: ComprehensiveTestRequest,
) -> ComprehensiveTestResult:
    """Run comprehensive test with all parameter variations.

    Tests all config parameter variations:
    1. First test with all defaults ({})
    2. Then vary one parameter at a time
    """
    try:
        # Get model info
        models = await discover_models_for_sample(request.sample)
        if request.model not in models:
            raise HTTPException(status_code=404, detail=f'Model {request.model} not found in sample {request.sample}')

        model_info = models[request.model]
        config_schema = model_info.get('customOptions', {})
        params = parse_config_schema(config_schema)

        # Generate variations
        variations = generate_config_variations(params)

        # Run all tests
        script_path = Path(__file__).parent / 'run_single_model_test.py'
        all_results = []

        for config in variations:
            result = await asyncio.to_thread(
                run_model_test, request.model, config, request.user_prompt, request.system_prompt, script_path
            )

            all_results.append({
                'config': config,
                'success': result.get('success', False),
                'response': result.get('response'),
                'error': result.get('error'),
                'timing': result.get('timing', 0.0),
            })

        # Calculate stats
        passed = sum(1 for r in all_results if r['success'])
        failed = len(all_results) - passed

        # Save summary result to file
        try:
            results_dir = Path(__file__).parent / 'results'
            results_dir.mkdir(exist_ok=True)

            import datetime
            import json

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_slug = request.model.replace('/', '_')
            summary_file = results_dir / f'summary_{model_slug}_{timestamp}.json'

            summary_data = {
                'sample': request.sample,
                'model': request.model,
                'user_prompt': request.user_prompt,
                'system_prompt': request.system_prompt,
                'timestamp': timestamp,
                'total_tests': len(all_results),
                'passed': passed,
                'failed': failed,
                'results': all_results,
            }

            def write_summary(path: Path, data: dict[str, Any]) -> None:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)

            await asyncio.to_thread(write_summary, summary_file, summary_data)

            logging.info(f'Saved comprehensive test summary to {summary_file}')
        except Exception as e:
            logging.error(f'Failed to save test summary: {e}')
            # Continue even if saving fails

        return ComprehensiveTestResult(total_tests=len(all_results), passed=passed, failed=failed, results=all_results)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f'Error running comprehensive test: {e}', exc_info=True)
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'Internal Server Error: {str(e)}') from e


# Wrapper to run run_model_test inside the thread
def run_wrapper(
    model: str,
    config: dict[str, Any],
    user_prompt: str,
    system_prompt: str,
    script_path: Path,
) -> dict[str, Any]:
    """Wrapper function to run the model test and return result."""
    # run_model_test in test_model_performance is synchronous (uses subprocess.run)
    # We need to call it directly. Wait, run_model_test in test_model_performance IS
    # synchronous wrapper around subprocess.
    # So we don't need asyncio.run inside the lambda unless run_model_test was async.
    # Checking test_model_performance.py... run_model_test is synchronous.

    return run_model_test(model, config, user_prompt, system_prompt, script_path)


# Mount static files (CSS, JS, etc.) under /static/
STATIC_DIR = Path(__file__).parent / 'static'
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)
