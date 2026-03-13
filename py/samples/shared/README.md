# Shared Sample Utilities

This module provides shared utilities, types, and flow logic used across
multiple Genkit samples. It helps maintain consistency and reduces code
duplication in sample applications.

## Contents

### Tools (`tools.py`)

Reusable tool definitions for samples:

| Tool | Description |
|------|-------------|
| `get_weather` | Simulated weather lookup tool |
| `convert_currency` | Currency conversion tool |
| `calculate` | Basic calculator tool |

### Types (`types.py`)

Pydantic models for structured input/output:

| Type | Description |
|------|-------------|
| `WeatherInput` | Input schema for weather tool |
| `CurrencyExchangeInput` | Input schema for currency tool |
| `CalculatorInput` | Input schema for calculator tool |
| `RpgCharacter` | Output schema for character generation |

### Flow Logic (`flows.py`)

Reusable flow implementations:

| Function | Description |
|----------|-------------|
| `say_hi_logic` | Basic greeting flow |
| `say_hi_stream_logic` | Streaming greeting flow |
| `say_hi_with_config_logic` | Greeting with custom config |
| `weather_logic` | Weather lookup with tool calling |
| `currency_exchange_logic` | Currency conversion flow |
| `calculation_logic` | Calculator flow |
| `generate_character_logic` | RPG character generation |

## Usage

Import shared utilities in your sample:

```python
from samples.shared import (
    get_weather,
    convert_currency,
    calculate,
    WeatherInput,
    RpgCharacter,
    say_hi_logic,
)

# Use in your flows
@ai.flow()
async def my_flow(prompt: str) -> str:
    return await say_hi_logic(ai, prompt)
```

## Purpose

This module exists to:

1. **Reduce Duplication**: Common patterns are defined once
2. **Ensure Consistency**: All samples use the same tool definitions
3. **Simplify Maintenance**: Updates propagate to all samples
4. **Demonstrate Best Practices**: Shows how to structure reusable code

## Note

This is an internal module for samples only. It is not published as a
separate package and should not be used in production applications.
