#!/usr/bin/env python3
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

"""Tool to test model performance with dynamic configuration variations.

Usage:
    python test_model_performance.py [--models model1,model2] [--output report.md]

Example:
    python test_model_performance.py --models googleai/gemini-2.0-flash
"""

import argparse
import builtins
import importlib.util
import json
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import Any, cast

# Mock input to prevent blocking
builtins.input = lambda prompt='': 'dummy_value'  # type: ignore


async def discover_models() -> dict[str, Any]:
    """Discover all available models from Genkit registry.

    Returns:
        Dict mapping model names to model info
    """
    # Import Genkit
    from genkit import Genkit

    plugins = []

    # Try initializing various plugins
    plugin_imports = [
        ('genkit.plugins.google_genai', 'GoogleAI'),
        ('genkit.plugins.vertex_ai', 'VertexAI'),
        ('genkit.plugins.deepseek', 'DeepSeek'),
        ('genkit.plugins.anthropic', 'Anthropic'),
        ('genkit.plugins.xai', 'XAI'),
        ('genkit.plugins.ollama', 'Ollama'),
        ('genkit.plugins.mistral', 'Mistral'),
        ('genkit.plugins.amazon_bedrock', 'AmazonBedrock'),
    ]

    for module_path, class_name in plugin_imports:
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            plugins.append(plugin_class())
        except (ImportError, AttributeError):
            # Silently skip if plugin not installed or wrong class name
            pass
        except Exception:  # noqa: S110
            pass

    ai = Genkit(plugins=plugins)

    registry = ai.registry

    # Get all model actions via list_actions (which queries plugins)
    from genkit.core.action import ActionKind

    try:
        actions = await registry.list_actions(allowed_kinds=[ActionKind.MODEL])
    except Exception as e:
        # Silently skip discovery errors (like Ollama connection failures)
        # to avoid noisy logs for users who don't have all local services running
        import logging

        logging.getLogger(__name__).debug(f'Error listing models during discovery: {e}')
        actions = []

    model_info = {}
    for meta in actions:
        # Extract model info using metadata
        info = (meta.metadata or {}).get('model', {})
        model_info[meta.name] = info

    return model_info


async def discover_models_for_sample(sample_name: str) -> dict[str, Any]:
    """Discover models used by a specific sample.

    Args:
        sample_name: Name of the sample directory (e.g., 'google-genai-hello')

    Returns:
        Dict mapping model names to model info for models used by this sample
    """
    import importlib.util
    import logging
    import sys
    from pathlib import Path

    logger = logging.getLogger(__name__)

    # Find the sample directory
    samples_dir = Path(__file__).parent.parent.parent / 'samples'
    sample_dir = samples_dir / sample_name

    if not sample_dir.exists():
        logger.warning(f'Sample directory not found: {sample_dir}')
        return {}

    # Look for main.py in common locations
    main_file = None
    search_paths = [
        sample_dir / 'main.py',
        sample_dir / 'src' / 'main.py',  # Some samples have src/ subdirectory
        sample_dir / 'server.py',
        sample_dir / 'app.py',
    ]

    for candidate_path in search_paths:
        if candidate_path.exists():
            main_file = candidate_path
            break

    if not main_file:
        logger.warning(f'No main file found for sample {sample_name}')
        # Fall back to all models
        return await discover_models()

    try:
        # Add necessary paths to sys.path for importing samples and shared utilities
        root_dir = samples_dir.parent
        src_dir = sample_dir / 'src'

        # Save original path for restoration
        original_sys_path = sys.path[:]

        logger.debug(f'Original sys.path: {original_sys_path}')
        logger.debug(f'Adding root_dir: {root_dir}')
        logger.debug(f'Adding sample_dir: {sample_dir}')
        logger.debug(f'Adding src_dir: {src_dir}')

        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
            logger.debug(f'Added {root_dir} to sys.path')

        # Also add samples directory itself just in case
        if str(samples_dir) not in sys.path:
            sys.path.insert(0, str(samples_dir))
            logger.debug(f'Added {samples_dir} to sys.path')

        if str(sample_dir) not in sys.path:
            sys.path.insert(0, str(sample_dir))
            logger.debug(f'Added {sample_dir} to sys.path')
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            logger.debug(f'Added {src_dir} to sys.path')

        logger.debug(f'Updated sys.path: {sys.path}')

        # Mock input() to prevent blocking on API key prompts
        import builtins
        import os

        original_input = builtins.input
        builtins.input = lambda prompt='': ''  # type: ignore # Return empty string for all inputs

        # Mock common API key environment variables if not set
        env_vars_to_mock = {
            'ANTHROPIC_API_KEY': 'mock-key',
            'OPENAI_API_KEY': 'mock-key',
            'GOOGLE_API_KEY': 'mock-key',
            'GEMINI_API_KEY': 'mock-key',
        }
        original_env_vars: dict[str, str | None] = {}
        for key, value in env_vars_to_mock.items():
            current_val = os.environ.get(key)
            if not current_val:  # Also mock if empty string
                original_env_vars[key] = current_val
                os.environ[key] = value
            else:
                original_env_vars[key] = current_val

        try:
            # Dynamically import the sample module
            module_name = f'sample_{sample_name.replace("-", "_")}'
            spec = importlib.util.spec_from_file_location(module_name, main_file)
            if spec is None or spec.loader is None:
                logger.warning(f'Could not load spec for {main_file}')
                return await discover_models()

            module = importlib.util.module_from_spec(spec)

            # Add to sys.modules before executing
            old_module = sys.modules.get(module_name)
            sys.modules[module_name] = module

            try:
                # Execute the module to initialize Genkit
                logger.info(f'Loading sample module from {main_file}')
                spec.loader.exec_module(module)

                # Get the Genkit instance if available
                if hasattr(module, 'ai'):
                    ai = module.ai
                    registry = ai.registry

                    # Get all model actions
                    from genkit.core.action import ActionKind

                    try:
                        actions = await registry.list_actions(allowed_kinds=[ActionKind.MODEL])
                    except Exception as e:
                        logger.warning(f'Error listing models for sample {sample_name}: {e}')
                        actions = []

                    model_info = {}
                    for meta in actions:
                        # Ensure metadata and model info exist
                        meta_data = meta.metadata or {}
                        info = meta_data.get('model', {})
                        model_info[meta.name] = info

                    if not model_info:
                        logger.warning(
                            f'Discovered 0 models for sample {sample_name}. '
                            'This often happens if API keys (e.g. GEMINI_API_KEY) are missing or invalid.'
                        )
                        # Fall back to global discovery
                        return await discover_models()

                    logger.info(
                        f'Discovered {len(model_info)} models for sample {sample_name}: {list(model_info.keys())}'
                    )
                    return model_info
                else:
                    logger.warning(f"Sample {sample_name} has no 'ai' attribute, falling back to global discovery")
                    return await discover_models()

            finally:
                # Restore old module if it existed
                if old_module is not None:
                    sys.modules[module_name] = old_module
                elif module_name in sys.modules:
                    del sys.modules[module_name]

        finally:
            # Restore original input function
            builtins.input = original_input

            # Restore original environment variables
            for key, original_value in original_env_vars.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

            # Restore original sys.path
            sys.path[:] = original_sys_path

    except ModuleNotFoundError as e:
        error_msg = str(e)
        logger.warning(f'Sample {sample_name} missing module: {e}')

        # Heuristic: If it's a Google sample, falling back to default/global models (Gemini) is usually safe/helpful
        is_google_sample = 'google' in sample_name or 'gemini' in sample_name or 'vertex' in sample_name

        if is_google_sample:
            logger.info(f'Google sample {sample_name} failed to load, falling back to default models')
            return await discover_models()

        # For non-Google samples (e.g. Bedrock), strict failure is better than showing Gemini
        if 'genkit.plugins.' in error_msg:
            return {}

        # For other import errors, return empty
        return {}

    except Exception as e:
        logger.error(f'Failed to load sample {sample_name}: {e}', exc_info=True)

        # Checking sample name again for fallback
        if 'google' in sample_name or 'gemini' in sample_name or 'vertex' in sample_name:
            logger.info(f'Google sample {sample_name} failed, falling back to default models')
            return await discover_models()

        return {}  # Return empty for other failures


def parse_config_schema(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse JSON schema to extract configuration parameters.

    Args:
        schema: JSON schema dict

    Returns:
        Dict mapping parameter names to their schema info
    """
    if not schema or 'properties' not in schema:
        return {}

    params = {}
    for param_name, param_schema in schema['properties'].items():
        # Handle 'anyOf' schemas (common in Gemini for nullable types)
        # e.g., "maxOutputTokens": { "anyOf": [{ "type": "integer" }, { "type": "null" }] }
        eff_schema = param_schema
        if 'type' not in eff_schema and 'anyOf' in eff_schema:
            for option in eff_schema['anyOf']:
                if option.get('type') and option.get('type') != 'null':
                    # Found the real type definition
                    eff_schema = option
                    break

        # If still no type, skip or look deeper (but simple type extraction is usually enough)
        # Some fields like 'stopSequences' might be array of strings, so we need to check items too.

        param_type = eff_schema.get('type')
        if not param_type and 'items' in eff_schema:
            # Array type might be inferred if items is present, or check if type is 'array'
            pass

        params[param_name] = {
            'type': param_type,
            'minimum': eff_schema.get('minimum'),
            'maximum': eff_schema.get('maximum'),
            'default': param_schema.get('default'),  # Default usually at top level
            'enum': eff_schema.get('enum'),
            'items': eff_schema.get('items'),
        }

    return params


def generate_config_variations(params: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate configuration variations for testing.

    Strategy:
    For each discovered parameter:
    - Numeric (temperature, top_k, top_p, etc.):
      - Test: min, max, midpoint (if distinct from default)
    - String (version, etc.):
      - Test: all enum values (if enum, excluding default)
    - Array (stop_sequences, etc.):
      - Test: empty array, sample values (excluding default)
    - Boolean:
      - Test: opposite of default (or both if no default)

    Args:
        params: Parameter schema info from parse_config_schema

    Returns:
        List of config dicts to test
    """
    variations = []

    # Test 1: Always start with all defaults (empty config)
    variations.append({})

    # Subsequent tests: Vary one parameter at a time
    for param_name, param_info in params.items():
        param_type = param_info['type']
        default_value = param_info.get('default')

        # Note/Optimization: We do NOT explicitly test the default value here
        # because the first test case ({}) already covers defaults for all parameters.
        # We only want to test deviations from the default.

        if param_type == 'number' or param_type == 'integer':
            # Numeric parameters: test min, midpoint, max individually
            minimum = param_info.get('minimum')
            maximum = param_info.get('maximum')

            if minimum is not None and minimum != default_value:
                variations.append({param_name: minimum})

            if maximum is not None and maximum != default_value:
                variations.append({param_name: maximum})

            if minimum is not None and maximum is not None and minimum < maximum:
                midpoint = (minimum + maximum) / 2
                if param_type == 'integer':
                    midpoint = int(midpoint)

                if midpoint != default_value:
                    variations.append({param_name: midpoint})

        elif param_type == 'boolean':
            # Boolean parameters: test true, then false
            # Deduplicate with default if present
            if default_value is not True:
                variations.append({param_name: True})
            if default_value is not False:
                variations.append({param_name: False})

        elif param_type == 'string':
            # String parameters: test each enum value individually
            enum_values = param_info.get('enum')
            if enum_values:
                for value in enum_values:
                    if value != default_value:
                        variations.append({param_name: value})

        elif param_type == 'array':
            # Array parameters: test empty array, then sample value
            if default_value != []:
                variations.append({param_name: []})

            # Add sample array value if items are strings
            items_schema = param_info.get('items', {})
            if items_schema.get('type') == 'string':
                sample = ['STOP']
                if sample != default_value:
                    variations.append({param_name: sample})

    return variations


def run_model_test(
    model_name: str,
    config: dict[str, Any],
    user_prompt: str,
    system_prompt: str | None,
    helper_script: Path,
    timeout: int = 60,
) -> dict[str, Any]:
    """Run a single model test via subprocess.

    Args:
        model_name: Model to test
        config: Configuration dict
        user_prompt: User prompt
        system_prompt: System prompt
        helper_script: Path to run_single_model_test.py
        timeout: Timeout in seconds

    Returns:
        Test result dict
    """
    cmd = [
        'uv',
        'run',
        'run_single_model_test.py',
        model_name,
        '--config',
        json.dumps(config),
        '--user-prompt',
        user_prompt,
    ]

    if system_prompt:
        cmd.extend(['--system-prompt', system_prompt])

    try:
        result_proc = subprocess.run(  # noqa: S603 - cmd constructed from trusted paths
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=helper_script.parent,  # Run from script directory (tools/sample-flows)
        )

        # Parse JSON output
        stdout = result_proc.stdout
        if '---JSON_RESULT_START---' in stdout and '---JSON_RESULT_END---' in stdout:
            json_str = stdout.split('---JSON_RESULT_START---')[1].split('---JSON_RESULT_END---')[0].strip()
            return json.loads(json_str)
        else:
            return json.loads(stdout)

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'response': None,
            'error': f'Test timed out after {timeout}s',
            'timing': timeout,
        }
    except Exception as e:
        return {
            'success': False,
            'response': None,
            'error': f'Subprocess failed: {e}',
            'timing': 0.0,
        }


def generate_report(
    results: list[dict[str, Any]],
    output_file: str,
) -> None:
    """Generate markdown report from test results.

    Args:
        results: List of test result dicts
        output_file: Output file path
    """
    total_tests = len(results)
    passed = sum(1 for r in results if r['result']['success'])
    failed = total_tests - passed
    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0

    lines = []
    lines.append('# Model Performance Test Report')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    lines.append(f'- **Total Tests**: {total_tests}')
    lines.append(f'- **Passed**: {passed}')
    lines.append(f'- **Failed**: {failed}')
    lines.append(f'- **Success Rate**: {success_rate:.1f}%')
    lines.append('')

    # Failed tests summary
    if failed > 0:
        lines.append('## Failed Tests')
        lines.append('')
        for test in results:
            if not test['result']['success']:
                config_str = json.dumps(test['config'], sort_keys=True)
                error = test['result']['error']
                # Truncate long errors
                if len(error) > 200:
                    error = error[:200] + '...'
                lines.append(f'- **{test["model"]}** (config: `{config_str}`)')
                lines.append(f'  - Error: {error}')
                lines.append('')

    # Detailed results
    lines.append('## Detailed Results')
    lines.append('')

    # Group by model
    models = {}
    for test in results:
        model_name = test['model']
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(test)

    for model_name, model_tests in models.items():
        lines.append(f'### {model_name}')
        lines.append('')

        for test in model_tests:
            config_str = json.dumps(test['config'], sort_keys=True) if test['config'] else '{}'
            status = '✅ SUCCESS' if test['result']['success'] else '❌ FAILED'
            timing = test['result']['timing']

            lines.append(f'#### Config: `{config_str}`')
            lines.append('')
            lines.append(f'- **Status**: {status}')
            lines.append(f'- **Timing**: {timing}s')

            if test['result']['success']:
                response = test['result']['response']
                # Truncate long responses
                if len(response) > 500:
                    response = response[:500] + '...'
                lines.append(f'- **Response**: {response}')
            else:
                error = test['result']['error']
                if len(error) > 500:
                    error = error[:500] + '...'
                lines.append(f'- **Error**: {error}')

            lines.append('')

    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))


def main() -> None:  # noqa: ASYNC240, ASYNC230 - test script, blocking I/O acceptable
    """Test model performance with dynamic configuration variations."""
    parser = argparse.ArgumentParser(description='Test model performance.')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated list of models to test')
    parser.add_argument('--output', type=str, default='model_performance_report.md', help='Output report file')
    parser.add_argument('--user-prompt', type=str, default='what is 5 +3?', help='User prompt for testing')
    parser.add_argument('--system-prompt', type=str, default='pirate', help='System prompt for testing')
    args = parser.parse_args()

    # Suppress verbose logging
    import logging

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('genkit').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)

    import asyncio

    all_models = asyncio.run(discover_models())

    # Filter models if specified
    if args.models:
        model_names = [m.strip() for m in args.models.split(',')]
        models_to_test = {k: v for k, v in all_models.items() if k in model_names}
    else:
        models_to_test = all_models

    if not models_to_test:
        return

    # Get helper script path
    script_dir = Path(__file__).parent
    helper_script = script_dir / 'run_single_model_test.py'

    # Run tests
    all_results = []
    for model_name, model_info in models_to_test.items():
        # Parse config schema
        config_schema = model_info.get('customOptions', {})
        params = parse_config_schema(config_schema)

        # Generate variations
        variations = generate_config_variations(params)

        # Run tests
        for _i, config in enumerate(variations, 1):
            # Flush stdout to ensure progress is visible
            sys.stdout.flush()

            result = run_model_test(
                model_name,
                config,
                args.user_prompt,
                args.system_prompt,
                helper_script,
            )

            all_results.append({
                'model': model_name,
                'config': config,
                'result': result,
            })

            '✅' if result['success'] else '❌'

    # Generate report
    generate_report(all_results, args.output)

    # Print summary
    total = len(all_results)
    if total == 0:
        return

    passed = sum(1 for r in all_results if cast(dict[str, Any], r['result']).get('success'))
    total - passed


if __name__ == '__main__':
    main()
