# genkit

Genkit is a framework designed to help you build AI-powered applications and features.
It provides open source libraries for Python, Node.js and Go, plus developer tools for testing
and debugging.

You can deploy and run Genkit libraries anywhere Python is supported. It's designed to work with
many AI model providers and vector databases. While we offer integrations for Firebase and Google Cloud,
you can use Genkit independently of any Google services.

## Setup Instructions

```bash
pip install genkit
pip install genkit-plugin-google-genai
```


```python
from pydantic import BaseModel, Field
from genkit import Genkit
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.0-flash',
)


class RpgCharacter(BaseModel):
    """An RPG game character."""

    name: str = Field(description='name of the character')
    back_story: str = Field(description='back story')
    abilities: list[str] = Field(description='list of abilities (3-4)')


@ai.flow()
async def generate_character(name: str) -> RpgCharacter:
    result = await ai.generate(
        prompt=f'generate an RPG character named {name}',
        output_schema=RpgCharacter,
    )
    return result.output


async def main() -> None:
    """Main function."""
    character = await generate_character('Goblorb')
    print(character.model_dump_json(indent=2))


if __name__ == '__main__':
    ai.run_main(main())
```

See https://python.api.genkit.dev for more details.
