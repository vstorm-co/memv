"""PydanticAI adapter for LLMClient protocol."""

from typing import TypeVar

from pydantic_ai import Agent

T = TypeVar("T")


class PydanticAIAdapter:
    """
    LLM client using PydanticAI.

    Supports multiple providers out of the box:
    - "openai:gpt-4o-mini"
    - "anthropic:claude-3-5-sonnet-latest"
    - "google-gla:gemini-2.5-flash"
    - "groq:llama-3.3-70b-versatile"

    See https://ai.pydantic.dev/models/ for full list.
    """

    def __init__(self, model: str = "openai:gpt-4o-mini"):
        self.model = model

    async def generate(self, prompt: str) -> str:
        """Generate unstructured text response."""
        agent: Agent[None, str] = Agent(self.model)
        result = await agent.run(prompt)
        return result.output

    async def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        """Generate structured response matching the Pydantic model."""
        agent: Agent[None, T] = Agent(self.model, output_type=response_model)
        result = await agent.run(prompt)
        return result.output
